# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pycocotools.mask as mask_util
import torch
from mmengine.utils import slice_list


def split_combined_polys(polys, poly_lens, polys_per_mask):
    """Split the combined 1-D polys into masks.

    A mask is represented as a list of polys, and a poly is represented as
    a 1-D array. In dataset, all masks are concatenated into a single 1-D
    tensor. Here we need to split the tensor into original representations.

    Args:
        polys (list): a list (length = image num) of 1-D tensors
        poly_lens (list): a list (length = image num) of poly length
        polys_per_mask (list): a list (length = image num) of poly number
            of each mask

    Returns:
        list: a list (length = image num) of list (length = mask num) of \
            list (length = poly num) of numpy array.
    """
    mask_polys_list = []
    for img_id in range(len(polys)):
        polys_single = polys[img_id]
        polys_lens_single = poly_lens[img_id].tolist()
        polys_per_mask_single = polys_per_mask[img_id].tolist()

        split_polys = slice_list(polys_single, polys_lens_single)
        mask_polys = slice_list(split_polys, polys_per_mask_single)
        mask_polys_list.append(mask_polys)
    return mask_polys_list


# TODO: move this function to more proper place
def encode_mask_results(mask_results):
    """Encode bitmap mask to RLE code.

    Args:
        mask_results (list): bitmap mask results.

    Returns:
        list | tuple: RLE encoded mask.
    """
    encoded_mask_results = []
    for mask in mask_results:
        encoded_mask_results.append(
            mask_util.encode(
                np.array(mask[:, :, np.newaxis], order='F',
                         dtype='uint8'))[0])  # encoded with RLE
    return encoded_mask_results


def mask2bbox(masks):
    n, h, w = masks.shape
    device = masks.device

    # 生成坐标网格（确保为 Float32）
    x = torch.arange(w, device=device, dtype=torch.float32).view(1, 1, -1)  # 强制使用 float32
    y = torch.arange(h, device=device, dtype=torch.float32).view(1, -1, 1)  # 强制使用 float32

    x_expanded = x.expand(n, h, w)
    y_expanded = y.expand(n, h, w)

    x_masked = torch.where(masks > 0, x_expanded, float('inf'))
    y_masked = torch.where(masks > 0, y_expanded, float('inf'))

    # 计算 min/max（保留 float32）
    x_min = x_masked.flatten(1).min(dim=1).values
    y_min = y_masked.flatten(1).min(dim=1).values

    x_masked = torch.where(masks > 0, x_expanded, float('-inf'))
    y_masked = torch.where(masks > 0, y_expanded, float('-inf'))

    x_max = x_masked.flatten(1).max(dim=1).values
    y_max = y_masked.flatten(1).max(dim=1).values

    # 标记有效区域（bool -> float32）
    valid_mask = masks.flatten(1).any(dim=1).float()  # 转换为 float32

    # 统一数据类型：两个分支均为 float32
    zero = torch.tensor(0.0, device=device, dtype=torch.float32)  # 显式定义 float32
    x_max = x_max * valid_mask + (x_max + 1) * (1 - valid_mask)   # 替换 torch.where
    y_max = y_max * valid_mask + (y_max + 1) * (1 - valid_mask)

    bboxes = torch.stack([x_min, y_min, x_max, y_max], dim=1)
    print(bboxes)
    return bboxes