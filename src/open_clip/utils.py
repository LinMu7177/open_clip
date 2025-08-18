from itertools import repeat
import collections.abc

import torch
from torch import nn as nn
from torchvision.ops.misc import FrozenBatchNorm2d


def freeze_batch_norm_2d(module, module_match={}, name=''):
    """
    Converts all `BatchNorm2d` and `SyncBatchNorm` layers of provided module into `FrozenBatchNorm2d`. If `module` is
    itself an instance of either `BatchNorm2d` or `SyncBatchNorm`, it is converted into `FrozenBatchNorm2d` and
    returned. Otherwise, the module is walked recursively and submodules are converted in place.

    Args:
        module (torch.nn.Module): Any PyTorch module.
        module_match (dict): Dictionary of full module names to freeze (all if empty)
        name (str): Full module name (prefix)

    Returns:
        torch.nn.Module: Resulting module

    Inspired by https://github.com/pytorch/pytorch/blob/a5895f85be0f10212791145bfedc0261d364f103/torch/nn/modules/batchnorm.py#L762
    """
    res = module
    is_match = True
    if module_match:
        is_match = name in module_match
    if is_match and isinstance(module, (nn.modules.batchnorm.BatchNorm2d, nn.modules.batchnorm.SyncBatchNorm)):
        res = FrozenBatchNorm2d(module.num_features)
        res.num_features = module.num_features
        res.affine = module.affine
        if module.affine:
            res.weight.data = module.weight.data.clone().detach()
            res.bias.data = module.bias.data.clone().detach()
        res.running_mean.data = module.running_mean.data
        res.running_var.data = module.running_var.data
        res.eps = module.eps
    else:
        for child_name, child in module.named_children():
            full_child_name = '.'.join([name, child_name]) if name else child_name
            new_child = freeze_batch_norm_2d(child, module_match, full_child_name)
            if new_child is not child:
                res.add_module(child_name, new_child)
    return res


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = lambda n, x: _ntuple(n)(x)

# Replaces all linear layers with linear_replacement
# TODO: add int8 support for other linear layers including attn and convnets
def replace_linear(model, linear_replacement, include_modules=['c_fc', 'c_proj'], copy_weights=True):
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_linear(module, linear_replacement, include_modules, copy_weights)

        if isinstance(module, torch.nn.Linear) and name in include_modules:
            old_module = model._modules[name]
            model._modules[name] = linear_replacement(
                module.in_features,
                module.out_features,
                module.bias is not None,
            )
            if copy_weights:
                model._modules[name].weight.data.copy_(old_module.weight.data)
                if model._modules[name].bias is not None:
                    model._modules[name].bias.data.copy_(old_module.bias)

    return model

def convert_int8_model_to_inference_mode(model):
    for m in model.modules():
        if hasattr(m, 'prepare_for_eval'):
            int8_original_dtype = m.weight.dtype
            m.prepare_for_eval()
            m.int8_original_dtype = int8_original_dtype

import numpy as np
from skimage.measure import label
from scipy.ndimage import binary_fill_holes

def fill_edges_to_objects(edges_mask):
    """
    将只包含边缘的掩码（1 表示物体边缘）转换为物体区域。
    1) 先将 edges_mask > 0 转为 bool 类型
    2) 用 binary_fill_holes 填充内部区域
    3) 用 label 对各连通区域打 label ID
    返回：labels，其值为 0 表示背景，1,2,... 表示不同物体。
    """
    # edges_mask 可能是 (H, W) 的 0/1 矩阵
    bool_mask = edges_mask > 0
    # 填充物体内部
    filled_mask = binary_fill_holes(bool_mask)
    # 连通组件标记
    labels = label(filled_mask)
    return labels

def patch_to_object_mapping(labels, patch_size):
    """
    给定按 label 分好的物体区域，以及指定的 patch 大小，
    统计每个 patch 内哪个 label(对象)面积占比最大，
    返回一个 2D 数组 patch_map，形状是 (n_patch_row, n_patch_col)，
    其中每个元素是该 patch 最占主导的对象 ID。
    """
    H, W = labels.shape
    n_patch_row = (H + patch_size - 1) // patch_size
    n_patch_col = (W + patch_size - 1) // patch_size
    
    patch_map = np.zeros((n_patch_row, n_patch_col), dtype=int)
    
    for pr in range(n_patch_row):
        for pc in range(n_patch_col):
            # 每个 patch 的图像坐标范围
            y_start = pr * patch_size
            y_end   = min((pr+1)*patch_size, H)
            x_start = pc * patch_size
            x_end   = min((pc+1)*patch_size, W)
            
            patch_region = labels[y_start:y_end, x_start:x_end]
            if patch_region.size == 0:
                continue
            
            # 计算出现最多的 label
            # (不计算 label=0，因为那是背景)
            vals, counts = np.unique(patch_region, return_counts=True)
            # 去掉背景
            bg_idx = np.where(vals == 0)
            vals = np.delete(vals, bg_idx)
            counts = np.delete(counts, bg_idx)
            
            if len(vals) > 0:                
                dominant_label = vals[np.argmax(counts)]
            else:
                dominant_label = 0  # 如果patch里没有任何对象
                
            patch_map[pr, pc] = dominant_label
    
    return patch_map

def generate_object_attention_mask(patch_map):
    patch_map_flat = patch_map.flatten()  # (n_patches,)
    # add CLS
    patch_map_flat = np.concatenate((np.array([0]), patch_map_flat, ))  # (n_patches + 1,)
    n_patches = patch_map_flat.shape[0]
    
    object_id_row = patch_map_flat[:, None]
    object_id_col = patch_map_flat[None, :]
    
    same_object = (object_id_row == object_id_col)
    
    return same_object.astype(np.float32)  # 1.0表示可以看见，0.0表示不能

def get_visible_matrix(edges_mask_batch):
    num_heads = 12 
    if edges_mask_batch is None:
        return None
    
    # 获取原张量所在的设备
    original_device = edges_mask_batch.device

    batch_size = edges_mask_batch.size(0)
    attn_masks = []

    for b in range(batch_size):
        edges_mask = edges_mask_batch[b, 0]  # (B, 1, H, W) -> (H, W)
        
        # Convert edges_mask tensor to NumPy
        edges_mask_numpy = edges_mask.cpu().detach().numpy()
        
        # 使用NumPy数组进行必要的运算
        labels = fill_edges_to_objects(edges_mask_numpy)
        patch_map = patch_to_object_mapping(labels, patch_size=32)
        attn_mask_numpy = generate_object_attention_mask(patch_map)

        # Convert back to tensor
        attn_mask_tensor = torch.tensor(attn_mask_numpy)

        # Move back to original device
        attn_mask_tensor = attn_mask_tensor.to(original_device)

        # 构造attention mask
        attn_mask_new = torch.where(
            attn_mask_tensor.bool(),
            torch.zeros_like(attn_mask_tensor, dtype=torch.float32),    # 相同object，mask 0（正常）
            torch.full_like(attn_mask_tensor, fill_value=-1e9, dtype=torch.float32)  # 不同object，mask为-∞
        )

        
        attn_masks.extend([attn_mask_new] * num_heads)  # 扩展到num_heads个

    # Return stacked tensor
    return torch.stack(attn_masks, dim=0)

def get_visible_matrix_v2(edges_mask, patch_size):
    # fill + 连通组件分析
    labels = fill_edges_to_objects(edges_mask)
    # 按照 patch_size，对 labels 做统计，得到 patch -> object
    patch_map = patch_to_object_mapping(labels, patch_size)

    visible_matrix = generate_object_attention_mask(patch_map)

    visible_matrix = torch.tensor(visible_matrix, dtype=torch.float32)

    # 构造attention mask
    visible_matrix = torch.where(
        visible_matrix.bool(),
        torch.zeros_like(visible_matrix, dtype=torch.float32),    # 相同object，mask 0（正常）
        torch.full_like(visible_matrix, fill_value=-1e9, dtype=torch.float32)  # 不同object，mask为-∞
    )
    return visible_matrix



### Object Token Attention Mask
def get_convert_patches(bbox, patch_size, n_patches_row, n_patches_col):
    """
    将 bbox 转换为 patch 坐标
    bbox: [x1, y1, x2, y2] (左上角和右下角坐标)
    patch_size: 每个 patch 的大小
    返回：patches: [[patch_x1, patch_y1, patch_x2, patch_y2], ...]
    """
    def point_to_patch(x, y, patch_size=32):
        patch_x = int(x // patch_size)
        patch_y = int(y // patch_size)
        return (patch_x, patch_y)


    x1, y1, x2, y2 = bbox
    top_left_patch = point_to_patch(x1, y1, patch_size)
    bottom_right_patch = point_to_patch(x2, y2, patch_size)

    # 解包输入的 patch 坐标
    patch_x_min, patch_y_min = top_left_patch
    patch_x_max, patch_y_max = bottom_right_patch

    patches = []
    # 遍历边框所跨的所有 patch 索引
    for px in range(patch_x_min, patch_x_max + 1):
        for py in range(patch_y_min, patch_y_max + 1):
            # 二维坐标压缩为一维
            patches.append(px * n_patches_col + py)

    return patches

def get_object_token_attention_mask(bboxes, image_original_size, image_resize_size, patch_size=32, obj_token_nums=10, background_token_nums=1, use_vm=False):
    """
    生成 object token 的 attention mask
    """
    if isinstance(image_resize_size, int):
        H, W = image_resize_size, image_resize_size
    else:
        H, W = image_resize_size
    W_old, H_old = image_original_size

    scale_w = W / W_old
    scale_h = H / H_old

    n_patches_row = (H + patch_size - 1) // patch_size
    n_patches_col = (W + patch_size - 1) // patch_size

    # 1. 初始化 vm
    vm = np.full((1 + obj_token_nums + 1 + n_patches_row * n_patches_col,
                        1 + obj_token_nums + 1 + n_patches_row * n_patches_col,), 
                       fill_value=-1e9, dtype=np.float32)
    np.fill_diagonal(vm, 0)
    
    # 2. 计算所有有 obj 的 patch，以及背景 patch
    all_obj_infos = []
    all_obj_patches, background_patches = [], []
    
    for idx, bbox in enumerate(bboxes):
        # bbox 边界检查与修正
        x1 = min(max(bbox[0], 0), W_old - 1)
        y1 = min(max(bbox[1], 0), H_old - 1)
        x2 = min(max(bbox[2], 0), W_old - 1)
        y2 = min(max(bbox[3], 0), H_old - 1)
        # 如果右下角坐标比左上角小，做调整
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        # 对于 bbox 中的每个元素 resize
        bbox_resized = [x1 * scale_w, y1 * scale_h, x2 * scale_w, y2 * scale_h]

        patches = get_convert_patches(bbox_resized, patch_size, n_patches_row, n_patches_col)
        all_obj_infos.append({
            'obj_id': idx,
            'patches': patches,
            'size': (bbox[2] - bbox[0]) *  (bbox[3] - bbox[1]),
        })
        all_obj_patches.extend(patches)
    background_patches = list(set(range(n_patches_row * n_patches_col)) - set(all_obj_patches))

    # 3. 填充 vm
    try:
        img_patch_start_idx = 1 + obj_token_nums + background_token_nums
        for idx, sub_vm in enumerate(vm):
            if idx == 0:
                # CLS token 可以看见所有 patch
                sub_vm[img_patch_start_idx:] = 0  
            elif idx <= obj_token_nums:
                # obj token 只能看到对应 obj 所在的 patch
                idx_ = idx - 1
                if idx_ >= len(all_obj_infos):
                    # 图像中的 obj 数量小于 obj_token_nums，也作为背景处理
                    for p_idx in background_patches:
                        sub_vm[img_patch_start_idx + p_idx] = 0
                else:
                    for p_idx in all_obj_infos[idx_]['patches']:
                        sub_vm[img_patch_start_idx + p_idx] = 0
            elif idx <= obj_token_nums + background_token_nums:
                # 背景 patch
                for p_idx in background_patches:
                    sub_vm[img_patch_start_idx + p_idx] = 0
            else:
                # 图像 patch
                idx_ = idx - (1 + obj_token_nums + background_token_nums)
                # CLS 可以看见
                sub_vm[0] = 0
                # 可以看见的 object token
                smallest_obj_id, smamllest_size = -1, float('inf')
                for obj_info in all_obj_infos:
                    if idx_ in obj_info['patches']:
                        # 该 patch 属于某个 obj
                        sub_vm[1 + obj_info['obj_id']] = 0
                        # 记录最小的 obj
                        if obj_info['size'] < smamllest_size:
                            smallest_obj_id, smamllest_size = obj_info['obj_id'], obj_info['size']
                
                # 如果不属于任何一个 obj，与 background token 交互
                if smallest_obj_id == -1:
                    for p_idx in range(len(all_obj_infos) + 1, 1 + obj_token_nums + background_token_nums):
                        sub_vm[1 + p_idx] = 0
                
                # 使用 image patch vm
                if use_vm:
                    # 如果使用 vm，则只保留最小的 obj 的 patch
                    if smallest_obj_id != -1:
                        for p_idx in all_obj_infos[smallest_obj_id]['patches']:
                            sub_vm[img_patch_start_idx + p_idx] = 0
                else:
                    # 如果不使用 vm，则将所有 patch 都置为 0
                    sub_vm[img_patch_start_idx:] = 0
    except Exception as e:
        print(f"Error while generating object token attention mask: {e}")
        print(f"bboxes: {bboxes}, image_original_size: {image_original_size}, image_resize_size: {image_resize_size}, patch_size: {patch_size}")
        raise e
    # 4. 转为 tensor
    attention_mask = torch.tensor(vm, dtype=torch.float32)
    return attention_mask

def get_obj_token_mask(bboxes, image_original_size, image_resize_size, patch_size=32, obj_token_nums=10, background_token_nums=1):
    """
    生成 object token 的 attn mask
    """
    if isinstance(image_resize_size, int):
        W, H = image_resize_size, image_resize_size
    else:
        W, H = image_resize_size
    W_old, H_old = image_original_size

    scale_w, scale_h = W / W_old, H / H_old
    n_patches_row, n_patches_col = (H + patch_size - 1) // patch_size, (W + patch_size - 1) // patch_size

    # 1. 初始化 attn_mask
    attn_mask = np.full((
            1 + obj_token_nums + background_token_nums + n_patches_row * n_patches_col, 
            1 + obj_token_nums + background_token_nums + n_patches_row * n_patches_col
        ), 
        fill_value=-1e9, dtype=np.float32
    )
    np.fill_diagonal(attn_mask, 0)

    # 2. 计算所有有 obj 的 patch，以及背景 patch
    all_obj_infos = []
    all_obj_patches, background_patches = [], []
    
    for idx, bbox in enumerate(bboxes):
        # bbox 边界检查与修正
        x1 = min(max(bbox[0], 0), W_old - 1)
        y1 = min(max(bbox[1], 0), H_old - 1)
        x2 = min(max(bbox[2], 0), W_old - 1)
        y2 = min(max(bbox[3], 0), H_old - 1)
        # 如果右下角坐标比左上角小，做调整
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        # 对于 bbox 中的每个元素 resize
        bbox_resized = [x1 * scale_w, y1 * scale_h, x2 * scale_w, y2 * scale_h]

        patches = get_convert_patches(bbox_resized, patch_size, n_patches_row, n_patches_col)
        all_obj_infos.append({
            'obj_id': idx,
            'patches': patches,
            'size': (bbox[2] - bbox[0]) *  (bbox[3] - bbox[1]),
        })
        all_obj_patches.extend(patches)
    background_patches = list(set(range(n_patches_row * n_patches_col)) - set(all_obj_patches))

    # 3. 填充 attn_mask
    img_patch_start_idx = 1 + obj_token_nums + background_token_nums
    for idx, sub_attn_mask in enumerate(attn_mask):
        if idx == 0:
            # CLS token 可以看见所有 patch
            sub_attn_mask[img_patch_start_idx:] = 0  
        elif idx <= obj_token_nums:
            # obj token 只能看到对应 obj 所在的 patch
            idx_ = idx - 1
            if idx_ >= len(all_obj_infos):
                # 图像中的 obj 数量小于 obj_token_nums，也作为背景处理
                for p_idx in background_patches:
                    sub_attn_mask[img_patch_start_idx + p_idx] = 0
            else:
                for p_idx in all_obj_infos[idx_]['patches']:
                    sub_attn_mask[img_patch_start_idx + p_idx] = 0
        elif idx <= obj_token_nums + background_token_nums:
            # 背景 patch
            for p_idx in background_patches:
                sub_attn_mask[img_patch_start_idx + p_idx] = 0
        else:
            # 图像 patch
            idx_ = idx - (1 + obj_token_nums + background_token_nums)
            # CLS 可以看见
            sub_attn_mask[0] = 0
            # 可以看见的 object token
            smallest_obj_id, smamllest_size = -1, float('inf')
            for obj_info in all_obj_infos:
                if idx_ in obj_info['patches']:
                    # 该 patch 属于某个 obj
                    sub_attn_mask[1 + obj_info['obj_id']] = 0
                    # 记录最小的 obj
                    if obj_info['size'] < smamllest_size:
                        smallest_obj_id, smamllest_size = obj_info['obj_id'], obj_info['size']
            
            # 如果不属于任何一个 obj，与 background token 交互
            if smallest_obj_id == -1:
                for p_idx in range(len(all_obj_infos) + 1, 1 + obj_token_nums + background_token_nums):
                    sub_attn_mask[1 + p_idx] = 0
            
            # 将所有 img patch 都置为 0
            sub_attn_mask[img_patch_start_idx:] = 0
    
    # 4. 转为 tensor
    attn_mask_tensor = torch.tensor(attn_mask, dtype=torch.float32)
    return attn_mask_tensor


def get_img_token_vm_mask(bboxes, image_original_size, image_resize_size, patch_size=32):
    """
    生成 img token visible matrix 的 attn_mask
    """
    if isinstance(image_resize_size, int):
        H, W = image_resize_size, image_resize_size
    else:
        H, W = image_resize_size
    W_old, H_old = image_original_size

    scale_w = W / W_old
    scale_h = H / H_old

    n_patches_row = (H + patch_size - 1) // patch_size
    n_patches_col = (W + patch_size - 1) // patch_size

    # 1. 初始化 attn_mask ([CLS] + img tokens)
    attn_mask = np.full((
        1 + n_patches_row * n_patches_col,
        1 + n_patches_row * n_patches_col
        ), 
        fill_value=-1e9, dtype=np.float32
    )
    np.fill_diagonal(attn_mask, 0)

    # 2. 计算所有有 obj 的 patch，以及背景 patch
    all_obj_infos = []
    all_obj_patches, background_patches = [], []

    for idx, bbox in enumerate(bboxes):
        # bbox 边界检查与修正
        x1 = min(max(bbox[0], 0), W_old - 1)
        y1 = min(max(bbox[1], 0), H_old - 1)
        x2 = min(max(bbox[2], 0), W_old - 1)
        y2 = min(max(bbox[3], 0), H_old - 1)
        # 如果右下角坐标比左上角小，做调整
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        # 对于 bbox 中的每个元素 resize
        bbox_resized = [x1 * scale_w, y1 * scale_h, x2 * scale_w, y2 * scale_h]

        patches = get_convert_patches(bbox_resized, patch_size, n_patches_row, n_patches_col)
        all_obj_infos.append({
            'obj_id': idx,
            'patches': patches,
            'size': (bbox[2] - bbox[0]) *  (bbox[3] - bbox[1]),
        })
        all_obj_patches.extend(patches)

    # 3. 填充 attn_mask
    for idx, sub_attn_mask in enumerate(attn_mask):
        if idx == 0:
            # CLS token 可以看见所有 patch
            sub_attn_mask[1:] = 0
        else:
            # 图像 patch
            idx_ = idx - 1
            # CLS 可以看见
            sub_attn_mask[0] = 0

            # 统计占有该 patch 的 obj 中，最小的一个
            smallest_obj_id, smamllest_size = -1, float('inf')
            for obj_info in all_obj_infos:
                if idx_ in obj_info['patches']:
                    # 该 patch 属于某个 obj
                    sub_attn_mask[1 + obj_info['obj_id']] = 0
                    # 记录最小的 obj
                    if obj_info['size'] < smamllest_size:
                        smallest_obj_id, smamllest_size = obj_info['obj_id'], obj_info['size']
            
            # 如果不属于任何一个 obj，跳过
            if smallest_obj_id == -1:
                continue

            # 找到最小的 obj 所包含的所有其他 patch，能够看到
            for p_idx in all_obj_infos[smallest_obj_id]['patches']:
                sub_attn_mask[1 + p_idx] = 0

    # 4. 转为 tensor
    attn_mask_tensor = torch.tensor(attn_mask, dtype=torch.float32)
    return attn_mask_tensor
