"""
Mask 数据完整性诊断工具

检查：
1. 原始 mask 文件的像素值分布
2. resize 后的 mask 像素值分布
3. remap 后的 mask 像素值分布
4. 数据增强后的 mask 像素值分布
"""
import sys
from pathlib import Path
import numpy as np
import cv2
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data.dataset import CableDefectDataset


def remap_mask_to_3class(mask):
    """将7类mask重新映射为3类"""
    mapping = {
        0: 0,  # background -> background
        1: 1,  # cable -> cable
        2: 2,  # tape -> tape
        3: 0,  # bulge -> background
        4: 0,  # loose -> background
        5: 0,  # burr -> background
        6: 0   # thin -> background
    }

    if torch.is_tensor(mask):
        remapped = torch.zeros_like(mask)
        for old_val, new_val in mapping.items():
            remapped[mask == old_val] = new_val
    else:
        remapped = mask.copy()
        for old_val, new_val in mapping.items():
            remapped[mask == old_val] = new_val

    return remapped


def check_original_masks(mask_dir, n=5):
    """检查原始 mask 文件"""
    print("=" * 70)
    print("【1】检查原始 mask 文件")
    print("=" * 70)

    import os
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])[:n]

    for mask_file in mask_files:
        mask_path = os.path.join(mask_dir, mask_file)
        mask_array = np.fromfile(mask_path, dtype=np.uint8)
        mask = cv2.imdecode(mask_array, cv2.IMREAD_UNCHANGED)

        if mask.ndim == 3:
            mask = mask[:, :, 0]

        unique = np.unique(mask)
        print(f"\n{mask_file}:")
        print(f"  shape: {mask.shape}, dtype: {mask.dtype}")
        print(f"  unique values: {unique} (共 {len(unique)} 个)")
        print(f"  min: {mask.min()}, max: {mask.max()}")

        # 检查是否有非类别ID的值
        valid_classes = [0, 1, 2, 3, 4, 5, 6]
        invalid = unique[~np.isin(unique, valid_classes)]
        if len(invalid) > 0:
            print(f"  [WARNING] 发现非法值: {invalid}")


def check_dataset_before_remap(n=5):
    """检查数据集加载后（remap 之前）"""
    print("\n" + "=" * 70)
    print("【2】检查数据集加载后（remap 之前）")
    print("=" * 70)

    dataset = CableDefectDataset(
        "dataset/processed/train/images",
        "dataset/processed/train/masks",
        augment=False,
        target_size=(512, 512)
    )

    for i in range(min(n, len(dataset))):
        img, mask = dataset[i]

        mask_np = mask.numpy() if torch.is_tensor(mask) else mask
        unique = np.unique(mask_np)

        print(f"\n样本 {i}:")
        print(f"  shape: {mask_np.shape}, dtype: {mask_np.dtype}")
        print(f"  unique values: {unique} (共 {len(unique)} 个)")
        print(f"  min: {mask_np.min()}, max: {mask_np.max()}")

        # 检查是否有非类别ID的值
        valid_classes = [0, 1, 2, 3, 4, 5, 6]
        invalid = unique[~np.isin(unique, valid_classes)]
        if len(invalid) > 0:
            print(f"  [WARNING] 发现非法值: {invalid}")


def check_dataset_after_remap(n=5):
    """检查 remap 之后"""
    print("\n" + "=" * 70)
    print("【3】检查 remap 之后（3类）")
    print("=" * 70)

    dataset = CableDefectDataset(
        "dataset/processed/train/images",
        "dataset/processed/train/masks",
        augment=False,
        target_size=(512, 512)
    )

    for i in range(min(n, len(dataset))):
        img, mask = dataset[i]

        # remap
        mask = remap_mask_to_3class(mask)

        mask_np = mask.numpy() if torch.is_tensor(mask) else mask
        unique = np.unique(mask_np)

        print(f"\n样本 {i}:")
        print(f"  unique values: {unique} (共 {len(unique)} 个)")

        # 检查是否只有 0, 1, 2
        valid_classes = [0, 1, 2]
        invalid = unique[~np.isin(unique, valid_classes)]
        if len(invalid) > 0:
            print(f"  [WARNING] 发现非法值: {invalid}")
        else:
            print(f"  [OK] 值域正确: [0, 1, 2]")


def check_dataset_with_augment(n=5):
    """检查数据增强之后"""
    print("\n" + "=" * 70)
    print("【4】检查数据增强之后")
    print("=" * 70)

    dataset = CableDefectDataset(
        "dataset/processed/train/images",
        "dataset/processed/train/masks",
        augment=True,  # 启用增强
        target_size=(512, 512)
    )

    for i in range(min(n, len(dataset))):
        img, mask = dataset[i]

        # remap
        mask = remap_mask_to_3class(mask)

        mask_np = mask.numpy() if torch.is_tensor(mask) else mask
        unique = np.unique(mask_np)

        print(f"\n样本 {i} (augment=True):")
        print(f"  unique values: {unique} (共 {len(unique)} 个)")

        # 检查是否只有 0, 1, 2
        valid_classes = [0, 1, 2]
        invalid = unique[~np.isin(unique, valid_classes)]
        if len(invalid) > 0:
            print(f"  [WARNING] 发现非法值: {invalid}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--mask-dir', type=str,
                       default='dataset/processed/train/masks',
                       help='mask 目录')
    parser.add_argument('--n', type=int, default=5,
                       help='检查样本数')
    args = parser.parse_args()

    # 检查原始文件
    check_original_masks(args.mask_dir, n=args.n)

    # 检查数据加载后
    check_dataset_before_remap(n=args.n)

    # 检查 remap 后
    check_dataset_after_remap(n=args.n)

    # 检查增强后
    check_dataset_with_augment(n=args.n)

    print("\n" + "=" * 70)
    print("诊断完成")
    print("=" * 70)
