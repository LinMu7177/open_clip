import json
import logging
import math
import os
import time
import pickle
import pycocotools.mask as mask_util

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel

try:
    import wandb
except ImportError:
    wandb = None

from open_clip import get_input_dtype, CLIP, CustomTextCLIP
from open_clip_train.distributed import is_master
from open_clip_train.zero_shot import zero_shot_eval
from open_clip_train.precision import get_autocast


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def postprocess_clip_output(model_out):
    return {
        "image_features": model_out[0],
        "text_features": model_out[1],
        "logit_scale": model_out[2]
    }


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model

def prepare_data_for_neg_loss(negs, args, device, texts):
    negs = negs.to(device=device, non_blocking=True)
    negs = negs.view(-1, negs.shape[-1])
    # clean non-negs that are zero. they r there because not every text has a negative
    pos_that_have_negs = [i for i, l in enumerate(list(negs[::args.num_negs])) if l.nonzero().any()]
    negs = [l for l in list(negs) if l.nonzero().any()]
    if len(negs) == 0:
        pos_that_have_negs=None
    else:
        texts = torch.cat((texts, torch.stack(negs)), dim=0)

    return texts,pos_that_have_negs

def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()


def train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    model.train()
    if args.distill:
        dist_model.eval()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        objects_sense = None
        visible_matrix = None
        property_pos = None
        property_neg = None
        counting_pos = None
        counting_neg = None
        spatial_pos = None
        spatial_neg = None

        images, texts = batch[0].to(device=device, non_blocking=True), batch[1].to(device=device, non_blocking=True)
        if args.objects_sense_format:
            objects_sense = batch[2].to(device=device, non_blocking=True)
            if args.use_visible_matrix:
                visible_matrix = batch[3].to(device=device, non_blocking=True)

        if args.use_obj_tokens:
            visible_matrix = batch[2].to(device=device, non_blocking=True)
            obj_texts, obj_texts_mask = batch[3].to(device=device, non_blocking=True), batch[4].to(device=device, non_blocking=True)

        if args.vl_negs:
            start_idx = 2 + (1 if args.objects_sense_format else 0) + (1 if args.use_visible_matrix else 0)
            
            property_pos = batch[start_idx].to(device=device, non_blocking=True)
            property_neg = batch[start_idx + 1].to(device=device, non_blocking=True)
            counting_pos = batch[start_idx + 2].to(device=device, non_blocking=True)
            counting_neg = batch[start_idx + 3].to(device=device, non_blocking=True)
            spatial_pos = batch[start_idx + 4].to(device=device, non_blocking=True)
            spatial_neg = batch[start_idx + 5].to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            with autocast():
                model_out = model(
                            image=images,
                            text=texts,
                            obj_texts=obj_texts if args.use_obj_tokens else None,
                            objects_sense=objects_sense,
                            visible_matrix=visible_matrix,
                            visible_matrix_layers=args.visible_matrix_layers if args.use_visible_matrix else None,
                            property_pos=property_pos,
                            property_neg=property_neg,
                            counting_pos=counting_pos,
                            counting_neg=counting_neg,
                            spatial_pos=spatial_pos,
                            spatial_neg=spatial_neg,
                        )
                logit_scale = model_out["logit_scale"]
                if args.distill:
                    with torch.no_grad():
                        dist_model_out = dist_model(images, texts)
                    model_out.update({f'dist_{k}': v for k, v in dist_model_out.items()})
                # losses = loss(**model_out, output_dict=True)
                losses = {}
                total_loss, contrastive_loss, obj_contrative_loss, neg_loss, property_loss, counting_loss, spatial_loss = loss(
                    model_out["image_features"],
                    model_out["text_features"],
                    model_out["obj_image_features"],
                    model_out["obj_text_features"],
                    obj_texts_mask,
                    model_out["logit_scale"],
                    model_out["property_pos_features"],
                    model_out["property_neg_features"],
                    model_out["counting_pos_features"],
                    model_out["counting_neg_features"],
                    model_out["spatial_pos_features"],
                    model_out["spatial_neg_features"],
                )

                losses["loss"] = total_loss
                losses["contrastive_loss"] = contrastive_loss
                losses["neg_loss"] = neg_loss
                losses["property_loss"] = property_loss
                losses["counting_loss"] = counting_loss
                losses["spatial_loss"] = spatial_loss
                losses["obj_contrative_loss"] = obj_contrative_loss

            backward(total_loss, scaler)
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    model_out = model(images, texts)

                    for f in ("logit_scale", "logit_bias"):
                        model_out.pop(f, None)

                    for key, val in model_out.items():
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]

                accum_images.append(images)
                accum_texts.append(texts)

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()
            for j in range(args.accum_freq):
                images = accum_images[j]
                texts = accum_texts[j]
                with autocast():
                    model_out = model(images, texts)

                    inputs_no_accum = {}
                    inputs_no_accum["logit_scale"] = logit_scale = model_out.pop("logit_scale")
                    if "logit_bias" in model_out:
                        inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")

                    inputs = {}
                    for key, val in accum_features.items():
                        accumulated = accum_features[key]
                        inputs[key] = torch.cat(accumulated[:j] + [model_out[key]] + accumulated[j + 1:])

                    losses = loss(**inputs, **inputs_no_accum, output_dict=True)
                    del inputs
                    del inputs_no_accum
                    losses["loss"] = total_loss

                backward(total_loss, scaler)

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_features = [], [], {}

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})"
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }
            log_data.update({name: val.val for name, val in losses_m.items()})

            log_data = {"train/" + name: val for name, val in log_data.items()}

            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, step)

            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                log_data['step'] = step  # for backwards compatibility
                wandb.log(log_data, step=step)

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for


def evaluate(model, data, epoch, args, tb_writer=None, tokenizer=None):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_total_loss = torch.zeros(()).to(device)
        cumulative_contrastive_loss = torch.zeros(()).to(device)
        cumulative_obj_contrastive_loss = torch.zeros(()).to(device)
        cumulative_gen_loss = torch.zeros(()).to(device)
        cumulative_neg_loss = torch.zeros(()).to(device)
        cumulative_property_loss = torch.zeros(()).to(device)
        cumulative_counting_loss = torch.zeros(()).to(device)
        cumulative_spatial_loss = torch.zeros(()).to(device)
        all_image_features, all_text_features = [], []
        with torch.inference_mode():
            for i, batch in enumerate(dataloader):
                objects_sense = None
                visible_matrix = None
                property_pos = None
                property_neg = None
                counting_pos = None
                counting_neg = None
                spatial_pos = None
                spatial_neg = None

                images, texts = batch[0].to(device=device, non_blocking=True), batch[1].to(device=device, non_blocking=True)
                if args.objects_sense_format:
                    objects_sense = batch[2].to(device=device, non_blocking=True)
                    if args.use_visible_matrix:
                        visible_matrix = batch[3].to(device=device, non_blocking=True)
                if args.vl_negs:
                    start_idx = 2 + (1 if args.objects_sense_format else 0) + (1 if args.use_visible_matrix else 0)
                    property_pos = batch[start_idx].to(device=device, non_blocking=True)
                    property_neg = batch[start_idx + 1].to(device=device, non_blocking=True)
                    counting_pos = batch[start_idx + 2].to(device=device, non_blocking=True)
                    counting_neg = batch[start_idx + 3].to(device=device, non_blocking=True)
                    spatial_pos = batch[start_idx + 4].to(device=device, non_blocking=True)
                    spatial_neg = batch[start_idx + 5].to(device=device, non_blocking=True)

                if args.use_obj_tokens:
                    visible_matrix = batch[2].to(device=device, non_blocking=True)
                    obj_texts, obj_texts_mask = batch[3].to(device=device, non_blocking=True), batch[4].to(device=device, non_blocking=True)

                with autocast():
                    model_out = model(
                            image=images,
                            text=texts,
                            obj_texts=obj_texts if args.use_obj_tokens else None,
                            objects_sense=objects_sense,
                            visible_matrix=visible_matrix,
                            visible_matrix_layers=args.visible_matrix_layers if args.use_visible_matrix else None,
                            property_pos=property_pos,
                            property_neg=property_neg,
                            counting_pos=counting_pos,
                            counting_neg=counting_neg,
                            spatial_pos=spatial_pos,
                            spatial_neg=spatial_neg,
                        )
                    image_features = model_out["image_features"]
                    text_features = model_out["text_features"]
                    logit_scale = model_out["logit_scale"]
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    all_image_features.append(image_features.cpu())
                    all_text_features.append(text_features.cpu())
                    logit_scale = logit_scale.mean()
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()

                    batch_size = images.shape[0]
                    labels = torch.arange(batch_size, device=device).long()
                    contrastive_loss = (
                                         F.cross_entropy(logits_per_image, labels) +
                                         F.cross_entropy(logits_per_text, labels)
                                 ) / 2
                    
                    gen_loss = maybe_compute_generative_loss(model_out)

                    neg_loss, property_loss, counting_loss, spatial_loss = maybe_compute_neg_loss(args, model_out)

                    from open_clip.loss import get_obj_contrastive_loss
                    obj_contrastive_loss = get_obj_contrastive_loss(
                        model_out["obj_image_features"],
                        model_out["obj_text_features"],
                        logit_scale,
                        obj_texts_mask,
                    )

                cumulative_contrastive_loss += contrastive_loss * batch_size
                cumulative_total_loss += contrastive_loss * batch_size
                if neg_loss is not None:
                    cumulative_neg_loss += neg_loss * batch_size
                    cumulative_property_loss += property_loss * batch_size
                    cumulative_counting_loss += counting_loss * batch_size
                    cumulative_spatial_loss += spatial_loss * batch_size
                    # add neg loss to total loss
                    cumulative_total_loss += neg_loss * batch_size

                if obj_contrastive_loss is not None:
                    cumulative_obj_contrastive_loss += obj_contrastive_loss * batch_size
                    cumulative_total_loss += obj_contrastive_loss * batch_size
        
                num_samples += batch_size 
                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Total Loss: {cumulative_total_loss / num_samples:.6f}, Contrastive Loss: {cumulative_contrastive_loss / num_samples:.6f}\t"
                        f"Object Contrastive Loss: {cumulative_obj_contrastive_loss / num_samples:.6f}\t"
                        f"Negative Loss: {cumulative_neg_loss / num_samples:.6f}, Property Loss: {cumulative_property_loss / num_samples:.6f}, Counting Loss: {cumulative_counting_loss / num_samples:.6f}, Spatial Loss: {cumulative_spatial_loss / num_samples:.6f}\t"
                        )
                    

                    if gen_loss is not None:
                        cumulative_gen_loss += gen_loss * batch_size
                        logging.info(
                            f"Generative Loss: {cumulative_gen_loss / num_samples:.6f}\t")
                

            # val_metrics = get_clip_metrics(
            #     image_features=torch.cat(all_image_features),
            #     text_features=torch.cat(all_text_features),
            #     logit_scale=logit_scale.cpu(),
            # )
            val_metrics = {}
            loss = cumulative_total_loss / num_samples
            metrics.update(
                {**val_metrics, "clip_val_total_loss": loss.item(), "epoch": epoch, "num_samples": num_samples}
            )

            if cumulative_neg_loss is not None:
                metrics.update(
                    {
                        "clip_val_contrastive_loss": (cumulative_contrastive_loss / num_samples).item(),
                        "clip_val_neg_loss": (cumulative_neg_loss / num_samples).item(),
                        "clip_val_property_loss": (cumulative_property_loss / num_samples).item(),
                        "clip_val_counting_loss": (cumulative_counting_loss / num_samples).item(),
                        "clip_val_spatial_loss": (cumulative_spatial_loss / num_samples).item(),
                    }
                )

            if cumulative_obj_contrastive_loss is not None:
                metrics.update(
                    {
                        "clip_val_obj_contrastive_loss": (cumulative_obj_contrastive_loss / num_samples).item(),
                    }
                )

            if gen_loss is not None:
                gen_loss = cumulative_gen_loss / num_samples
                metrics.update({"val_generative_loss": gen_loss.item()})

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    log_data = {"val/" + name: val for name, val in metrics.items()}

    if args.save_logs:
        if tb_writer is not None:
            for name, val in log_data.items():
                tb_writer.add_scalar(name, val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        if 'train' in data:
            dataloader = data['train'].dataloader
            num_batches_per_epoch = dataloader.num_batches // args.accum_freq
            step = num_batches_per_epoch * epoch
        else:
            step = None
        log_data['epoch'] = epoch
        wandb.log(log_data, step=step)

    return metrics


def get_clip_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics


def maybe_compute_generative_loss(model_out):
    if "logits" in model_out and "labels" in model_out:
        token_logits = model_out["logits"]
        token_labels = model_out["labels"]
        return F.cross_entropy(token_logits.permute(0, 2, 1), token_labels)

def maybe_compute_neg_loss(args, model_out):

    def get_group_loss(image_feats, pos_feats, neg_feats, logit_scale):
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
    
    if args.vl_negs:
        image_features = model_out["image_features"]
        logit_scale = model_out["logit_scale"]
        property_pos_features = model_out["property_pos_features"]
        property_neg_features = model_out["property_neg_features"]
        counting_pos_features = model_out["counting_pos_features"]
        counting_neg_features = model_out["counting_neg_features"]
        spatial_pos_features = model_out["spatial_pos_features"]
        spatial_neg_features = model_out["spatial_neg_features"]
        if property_pos_features is not None:
            property_loss = get_group_loss(
                image_features, property_pos_features, property_neg_features, logit_scale)

        if counting_pos_features is not None:
            counting_loss = get_group_loss(
                image_features, counting_pos_features, counting_neg_features, logit_scale)

        if spatial_pos_features is not None:
            spatial_loss = get_group_loss(
                image_features, spatial_pos_features, spatial_neg_features, logit_scale)

        

        property_weight, counting_weight, spatial_weight = args.neg_w
        neg_loss = property_weight * property_loss + counting_weight * counting_loss + spatial_weight * spatial_loss
        return neg_loss, property_loss, counting_loss, spatial_loss
    else:
        return None, None, None, None