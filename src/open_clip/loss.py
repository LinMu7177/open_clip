from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            args=None
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        # cache state
        self.prev_num_logits = 0
        self.labels = {}
        self.args = args

    def forward(self, image_features, text_features, obj_image_features, obj_text_features, obj_text_mask, logit_scale, property_pos_features, property_neg_features, counting_pos_features, counting_neg_features, spatial_pos_features, spatial_neg_features):
        device = image_features.device

        neg_loss = torch.zeros(()).to(device)
        property_loss = torch.zeros(()).to(device)
        counting_loss = torch.zeros(()).to(device)
        spatial_loss = torch.zeros(()).to(device)

        obj_contrastive_loss = torch.zeros(()).to(device)

        if self.args.vl_negs:
            if property_pos_features is not None:
                property_loss = self.get_group_loss(
                    image_features, property_pos_features, property_neg_features, logit_scale)

            if counting_pos_features is not None:
                counting_loss = self.get_group_loss(
                    image_features, counting_pos_features, counting_neg_features, logit_scale)

            if spatial_pos_features is not None:
                spatial_loss = self.get_group_loss(
                    image_features, spatial_pos_features, spatial_neg_features, logit_scale)

        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad,
                self.rank, self.world_size, self.args, None
            )
            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        contrastive_loss = (
                                   F.cross_entropy(logits_per_image, labels) +
                                   F.cross_entropy(logits_per_text, labels)
                           ) / 2

        total_loss = contrastive_loss
        if self.args.vl_negs:
            property_weight, counting_weight, spatial_weight = self.args.neg_w
            neg_loss = property_weight * property_loss + counting_weight * counting_loss + spatial_weight * spatial_loss
            total_loss = total_loss + neg_loss
        
        if self.args.use_obj_tokens:
            obj_contrastive_loss = self.get_obj_contrastive_loss(
                obj_image_features, obj_text_features, logit_scale, obj_text_mask
            )
            total_loss = total_loss + obj_contrastive_loss

        return total_loss, contrastive_loss, obj_contrastive_loss, neg_loss, property_loss, counting_loss, spatial_loss

    def get_group_loss(self, image_feats, pos_feats, neg_feats, logit_scale):
        """
        image_feats : (B, D)
        pos_feats   : (B, D)
        neg_feats   : (B, K, D)  或  (B, D)  或  None
        """
        B, D = image_feats.shape
        device = image_feats.device

        pos_feats = pos_feats.unsqueeze(1)

        if neg_feats is None:
            feat_cat = pos_feats  # (B, 1, D)
        else:
            if neg_feats.dim() == 2:
                neg_feats = neg_feats.unsqueeze(1)  # (B, 1, D)
            feat_cat = torch.cat([pos_feats, neg_feats], dim=1)  # (B, 1+K, D)

        logits = logit_scale * torch.matmul(
            feat_cat, image_feats.unsqueeze(2)  # (B, 1+K, 1)
        ).squeeze(-1)

        target = torch.zeros(B, dtype=torch.long, device=device)  # 正样本在 0 位
        return F.cross_entropy(logits, target)

    def get_loss_neg_only(self, image_features, text_features, logit_scale,pos_that_have_negs):
        num_imgs = image_features.shape[0]
        image_features = image_features[pos_that_have_negs]
        pos = text_features[:num_imgs][pos_that_have_negs].unsqueeze(1)
        neg = text_features[num_imgs:].view((len(pos_that_have_negs), -1, text_features.shape[-1]))
        pos_neg = torch.cat([pos,neg], dim=1)
        image_features = image_features.unsqueeze(2)
        logits = logit_scale * torch.matmul(pos_neg,image_features)[:,:,0]
        ground_truth = torch.zeros(len(pos_that_have_negs)).long()
        ground_truth = ground_truth.to(self.args.device, non_blocking=True)
        total_loss = F.cross_entropy(logits, ground_truth)#zero is the right "class". the positive are always on the 0 place
        return total_loss

    def get_loss_pos_only(self, image_features, text_features,poss_features, logit_scale):
        logits_per_image_text_pos = logit_scale * image_features @ poss_features.t()
        ground_truth = (torch.arange(len(logits_per_image_text_pos)).long()).to(self.args.device, non_blocking=True)
        logits_text_pos_to_text = logit_scale * text_features @ poss_features.t()
        if self.args.symmetric:
            logits_per_image_text_pos_op = logit_scale * poss_features @ image_features.t()
            logits_text_pos_to_text_op = logit_scale * poss_features @ text_features.t()
            total_loss = (F.cross_entropy(logits_per_image_text_pos, ground_truth)
                          + F.cross_entropy(logits_per_image_text_pos_op, ground_truth)
                         ) / 2
            total_loss += ( F.cross_entropy(logits_text_pos_to_text, ground_truth)
                    + F.cross_entropy(logits_text_pos_to_text_op, ground_truth))/2
            total_loss = total_loss/2
        else:
            total_loss = F.cross_entropy(logits_per_image_text_pos, ground_truth)
            total_loss+= F.cross_entropy(logits_text_pos_to_text, ground_truth)
            total_loss = total_loss / 2


        if self.args.kl_pos:
            kl_loss = nn.KLDivLoss(reduction="batchmean")
            logits_per_image_text = logit_scale * image_features @ text_features.t()
            two_pos_feat = torch.stack([torch.diagonal(logits_per_image_text,0),torch.diagonal(logits_per_image_text_pos,0)],dim=1)
            ground_truth = F.softmax(0.5 + torch.zeros_like(two_pos_feat),dim=1).to(self.args.device, non_blocking=True)
            log_probs = F.log_softmax(two_pos_feat, dim=1)
            loss_kl = 0.1*kl_loss(log_probs, ground_truth)
            total_loss += loss_kl
        if self.args.common_batch_pos:
            kl_loss = nn.KLDivLoss(reduction="batchmean")
            text_and_pos_feat = torch.cat([text_features,poss_features])
            logits_per_image_text_and_pos_feat = logit_scale * image_features @ text_and_pos_feat.t()
            log_probs = F.log_softmax(logits_per_image_text_and_pos_feat, dim=1)
            ground_truth = F.softmax((torch.cat([torch.eye(self.args.batch_size), torch.eye(self.args.batch_size)], dim=1) / 2),dim=1).to(self.args.device, non_blocking=True)
            loss_kl_common_batch_pos = 0.01*kl_loss(log_probs, ground_truth)
            total_loss += loss_kl_common_batch_pos


        return total_loss

    def get_obj_contrastive_loss(self, obj_image_features, obj_text_features, logit_scale, obj_mask):
        """
        计算对象级别的对比学习损失，并支持掩码过滤。

        Args:
            obj_image_features (torch.Tensor): 视觉对象特征，形状 [B, N_obj, D]
            obj_text_features (torch.Tensor): 文本对象特征，形状 [B, N_obj, D]
            logit_scale (torch.Tensor or float): 用于缩放 logits 的温度参数倒数
            obj_mask (torch.Tensor): 布尔型掩码，形状 [B, N_obj]，
                                    True 表示该位置的对象有效，False 表示无效。

        Returns:
            torch.Tensor: 计算得到的对象对比学习损失。
        """
        batch_size, num_obj_per_sample, _ = obj_image_features.shape

        # 1. 计算相似度矩阵
        # 形状: [B, N_obj, N_obj]
        # (B, N_obj, D) @ (B, D, N_obj) -> (B, N_obj, N_obj)
        logits_per_visual_text = logit_scale * (obj_image_features @ obj_text_features.permute(0, 2, 1))

        # 2. 构建目标标签 (每个样本内的对角线)
        # 形状: [B, N_obj]
        labels = torch.arange(num_obj_per_sample, device=logits_per_visual_text.device).unsqueeze(0).expand(batch_size, -1)

        # 3. 展平 logits 和 labels 以适应 F.cross_entropy
        # 形状: [B * N_obj, N_obj]
        flat_logits = logits_per_visual_text.view(-1, num_obj_per_sample)
        # 形状: [B * N_obj]
        flat_labels = labels.reshape(-1)

        # 4. 展平掩码
        # 形状: [B * N_obj]
        flat_obj_mask = obj_mask.view(-1)

        # 5. 应用掩码，只选择有效对象的 logits 和 labels
        # 这些 `valid_` 张量只包含 `obj_mask` 中为 True 的行/元素
        valid_logits = flat_logits[flat_obj_mask] # 形状: [有效对象总数, N_obj]
        valid_labels = flat_labels[flat_obj_mask] # 形状: [有效对象总数]

        # 6. 检查是否存在有效对象以避免计算空损失
        if valid_logits.numel() == 0:
            # 如果没有有效对象，则损失为 0，避免 NaN
            return torch.tensor(0.0, device=obj_image_features.device)

        # 7. 计算从视觉到文本的损失
        loss_visual_text = F.cross_entropy(valid_logits, valid_labels)

        # 8. 计算从文本到视觉的对称损失
        # 需要先转置原始 logits_per_visual_text，然后同样应用掩码
        # [B, N_obj, N_obj] -> [B, N_obj, N_obj]
        logits_per_text_visual = logits_per_visual_text.permute(0, 2, 1).contiguous()
        # 展平以便应用掩码
        flat_logits_T = logits_per_text_visual.view(-1, num_obj_per_sample)
        
        # 同样应用掩码过滤，确保只计算有效文本特征对应的损失
        # 这里的 valid_logits_T 和 valid_labels 长度应该相同
        valid_logits_T = flat_logits_T[flat_obj_mask]

        # 计算对称损失
        loss_text_visual = F.cross_entropy(valid_logits_T, valid_labels)

        # 9. 计算最终的平均损失
        object_contrastive_loss = (loss_visual_text + loss_text_visual) / 2

        return object_contrastive_loss



class CoCaLoss(ClipLoss):
    def __init__(
            self,
            caption_loss_weight,
            clip_loss_weight,
            pad_id=0,  # pad_token for open_clip custom tokenizer
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod
        )

        self.clip_loss_weight = clip_loss_weight
        self.caption_loss_weight = caption_loss_weight
        self.caption_loss = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, image_features, text_features, logits, labels, logit_scale, output_dict=False):
        if self.clip_loss_weight:
            clip_loss = super().forward(image_features, text_features, logit_scale)
            clip_loss = self.clip_loss_weight * clip_loss
        else:
            clip_loss = torch.tensor(0, device=logits.device)

        caption_loss = self.caption_loss(
            logits.permute(0, 2, 1),
            labels,
        )
        caption_loss = caption_loss * self.caption_loss_weight

        if output_dict:
            return {"contrastive_loss": clip_loss, "caption_loss": caption_loss}

        return clip_loss, caption_loss


class DistillClipLoss(ClipLoss):

    def dist_loss(self, teacher_logits, student_logits):
        return -(teacher_logits.softmax(dim=1) * student_logits.log_softmax(dim=1)).sum(dim=1).mean(dim=0)

    def forward(
            self,
            image_features,
            text_features,
            logit_scale,
            dist_image_features,
            dist_text_features,
            dist_logit_scale,
            output_dict=False,
    ):
        logits_per_image, logits_per_text = \
            self.get_logits(image_features, text_features, logit_scale)

        dist_logits_per_image, dist_logits_per_text = \
            self.get_logits(dist_image_features, dist_text_features, dist_logit_scale)

        labels = self.get_ground_truth(image_features.device, logits_per_image.shape[0])

        contrastive_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        distill_loss = (
            self.dist_loss(dist_logits_per_image, logits_per_image) +
            self.dist_loss(dist_logits_per_text, logits_per_text)
        ) / 2

        if output_dict:
            return {"contrastive_loss": contrastive_loss, "distill_loss": distill_loss}

        return contrastive_loss, distill_loss


def neighbour_exchange(from_rank, to_rank, tensor, group=None):
    tensor_recv = torch.zeros_like(tensor)
    send_op = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor,
        to_rank,
        group=group,
    )
    recv_op = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_recv,
        from_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])
    for req in reqs:
        req.wait()
    return tensor_recv


def neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    tensor_from_left = torch.zeros_like(tensor_to_right)
    tensor_from_right = torch.zeros_like(tensor_to_left)
    send_op_left = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_left,
        left_rank,
        group=group,
    )
    send_op_right = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_right,
        right_rank,
        group=group,
    )
    recv_op_left = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_left,
        left_rank,
        group=group,
    )
    recv_op_right = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_right,
        right_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op_right, send_op_left, recv_op_right, recv_op_left])
    for req in reqs:
        req.wait()
    return tensor_from_right, tensor_from_left


class NeighbourExchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, from_rank, to_rank, group, tensor):
        ctx.group = group
        ctx.from_rank = from_rank
        ctx.to_rank = to_rank
        return neighbour_exchange(from_rank, to_rank, tensor, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None, None) + (NeighbourExchange.apply(ctx.to_rank, ctx.from_rank, ctx.group, grad_output),)


def neighbour_exchange_with_grad(from_rank, to_rank, tensor, group=None):
    return NeighbourExchange.apply(from_rank, to_rank, group, tensor)


class NeighbourExchangeBidir(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left_rank, right_rank, group, tensor_to_left, tensor_to_right):
        ctx.group = group
        ctx.left_rank = left_rank
        ctx.right_rank = right_rank
        return neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=group)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None, None, None) + \
            NeighbourExchangeBidir.apply(ctx.right_rank, ctx.left_rank, ctx.group, *grad_outputs)


def neighbour_exchange_bidir_with_grad(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    return NeighbourExchangeBidir.apply(left_rank, right_rank, group, tensor_to_left, tensor_to_right)


class SigLipLoss(nn.Module):
    """ Sigmoid Loss for Language Image Pre-Training (SigLIP) - https://arxiv.org/abs/2303.15343

    @article{zhai2023sigmoid,
      title={Sigmoid loss for language image pre-training},
      author={Zhai, Xiaohua and Mustafa, Basil and Kolesnikov, Alexander and Beyer, Lucas},
      journal={arXiv preprint arXiv:2303.15343},
      year={2023}
    }
    """
    def __init__(
            self,
            cache_labels: bool = False,
            rank: int = 0,
            world_size: int = 1,
            dist_impl: Optional[str] = None,
    ):
        super().__init__()
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.dist_impl = dist_impl or 'bidir'  # default to bidir exchange for now, this will likely change
        assert self.dist_impl in ('bidir', 'shift', 'reduce', 'gather')

        # cache state FIXME cache not currently used, worthwhile?
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, dtype, num_logits, negative_only=False) -> torch.Tensor:
        labels = -torch.ones((num_logits, num_logits), device=device, dtype=dtype)
        if not negative_only:
            labels = 2 * torch.eye(num_logits, device=device, dtype=dtype) + labels
        return labels

    def get_logits(self, image_features, text_features, logit_scale, logit_bias=None):
        logits = logit_scale * image_features @ text_features.T
        if logit_bias is not None:
            logits += logit_bias
        return logits

    def _loss(self, image_features, text_features, logit_scale, logit_bias=None, negative_only=False):
        logits = self.get_logits(image_features, text_features, logit_scale, logit_bias)
        labels = self.get_ground_truth(
            image_features.device,
            image_features.dtype,
            image_features.shape[0],
            negative_only=negative_only,
        )
        loss = -F.logsigmoid(labels * logits).sum() / image_features.shape[0]
        return loss

    def forward(self, image_features, text_features, logit_scale, logit_bias, output_dict=False):
        loss = self._loss(image_features, text_features, logit_scale, logit_bias)

        if self.world_size > 1:
            if self.dist_impl == 'bidir':
                right_rank = (self.rank + 1) % self.world_size
                left_rank = (self.rank - 1 + self.world_size) % self.world_size
                text_features_to_right = text_features_to_left = text_features
                num_bidir, remainder = divmod(self.world_size - 1, 2)
                for i in range(num_bidir):
                    text_features_recv = neighbour_exchange_bidir_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_left,
                        text_features_to_right,
                    )
                    for f in text_features_recv:
                        loss += self._loss(
                            image_features,
                            f,
                            logit_scale,
                            logit_bias,
                            negative_only=True,
                        )
                    text_features_to_left, text_features_to_right = text_features_recv

                if remainder:
                    text_features_recv = neighbour_exchange_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_right
                    )
                    loss += self._loss(
                        image_features,
                        text_features_recv,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
            elif self.dist_impl == "shift":
                right_rank = (self.rank + 1) % self.world_size
                left_rank = (self.rank - 1 + self.world_size) % self.world_size
                text_features_to_right = text_features
                for i in range(self.world_size - 1):
                    text_features_from_left = neighbour_exchange_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_right,
                    )
                    loss += self._loss(
                        image_features,
                        text_features_from_left,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
                    text_features_to_right = text_features_from_left
            elif self.dist_impl == "reduce":
                for i in range(self.world_size):
                    text_from_other = torch.distributed.nn.all_reduce(
                        text_features * (self.rank == i),
                        torch.distributed.ReduceOp.SUM,
                    )
                    loss += float(i != self.rank) * self._loss(
                        image_features,
                        text_from_other,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
            elif self.dist_impl == "gather":
                all_text = torch.distributed.nn.all_gather(text_features)
                for i in range(self.world_size):
                    loss += float(i != self.rank) * self._loss(
                        image_features,
                        all_text[i],
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
            else:
                assert False

        return {"contrastive_loss": loss} if output_dict else loss
