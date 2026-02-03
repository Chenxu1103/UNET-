"""
统计训练集和验证集的类别分布
"""
import sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

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
        return remapped
    else:
        remapped = mask.copy()
        for old_val, new_val in mapping.items():
            remapped[mask == old_val] = new_val
        return remapped


def analyze_dataset(dataset_name, image_dir, mask_dir):
    """分析数据集"""
    print("=" * 70)
    print(f"数据集: {dataset_name}")
    print("=" * 70)

    dataset = CableDefectDataset(
        image_dir,
        mask_dir,
        augment=False,
        target_size=(512, 512)
    )

    print(f"总样本数: {len(dataset)}\n")

    # 统计每个样本的类别分布
    class_counts_per_sample = {
        'has_bg': 0,
        'has_cable': 0,
        'has_tape': 0,
        'only_bg': 0,
        'bg_cable': 0,
        'bg_tape': 0,
        'cable_tape': 0,
        'all_three': 0
    }

    pixel_counts = {
        0: 0,  # background
        1: 0,  # cable
        2: 0   # tape
    }

    total_pixels = 0

    for idx in tqdm(range(len(dataset)), desc=f"处理{dataset_name}"):
        img, mask = dataset[idx]
        mask = remap_mask_to_3class(mask)

        if torch.is_tensor(mask):
            mask_np = mask.numpy()
        else:
            mask_np = mask

        unique = np.unique(mask_np)

        # 统计类别组合
        has_bg = 0 in unique
        has_cable = 1 in unique
        has_tape = 2 in unique

        if has_bg:
            class_counts_per_sample['has_bg'] += 1
        if has_cable:
            class_counts_per_sample['has_cable'] += 1
        if has_tape:
            class_counts_per_sample['has_tape'] += 1

        if len(unique) == 1:
            if unique[0] == 0:
                class_counts_per_sample['only_bg'] += 1
        elif len(unique) == 2:
            if has_bg and has_cable:
                class_counts_per_sample['bg_cable'] += 1
            elif has_bg and has_tape:
                class_counts_per_sample['bg_tape'] += 1
            elif has_cable and has_tape:
                class_counts_per_sample['cable_tape'] += 1
        elif len(unique) == 3:
            class_counts_per_sample['all_three'] += 1

        # 统计像素数
        for cls in [0, 1, 2]:
            count = np.sum(mask_np == cls)
            pixel_counts[cls] += count
        total_pixels += mask_np.size

    # 打印统计结果
    print("\n【类别组合统计】")
    print(f"  包含背景的样本: {class_counts_per_sample['has_bg']} ({class_counts_per_sample['has_bg']/len(dataset)*100:.1f}%)")
    print(f"  包含电缆的样本: {class_counts_per_sample['has_cable']} ({class_counts_per_sample['has_cable']/len(dataset)*100:.1f}%)")
    print(f"  包含胶带的样本: {class_counts_per_sample['has_tape']} ({class_counts_per_sample['has_tape']/len(dataset)*100:.1f}%)")
    print(f"\n  只有背景: {class_counts_per_sample['only_bg']} ({class_counts_per_sample['only_bg']/len(dataset)*100:.1f}%)")
    print(f"  背景+电缆: {class_counts_per_sample['bg_cable']} ({class_counts_per_sample['bg_cable']/len(dataset)*100:.1f}%)")
    print(f"  背景+胶带: {class_counts_per_sample['bg_tape']} ({class_counts_per_sample['bg_tape']/len(dataset)*100:.1f}%)")
    print(f"  电缆+胶带: {class_counts_per_sample['cable_tape']} ({class_counts_per_sample['cable_tape']/len(dataset)*100:.1f}%)")
    print(f"  完整三类: {class_counts_per_sample['all_three']} ({class_counts_per_sample['all_three']/len(dataset)*100:.1f}%)")

    print("\n【像素级统计】")
    print(f"  背景像素: {pixel_counts[0]:,} ({pixel_counts[0]/total_pixels*100:.2f}%)")
    print(f"  电缆像素: {pixel_counts[1]:,} ({pixel_counts[1]/total_pixels*100:.2f}%)")
    print(f"  胶带像素: {pixel_counts[2]:,} ({pixel_counts[2]/total_pixels*100:.2f}%)")
    print(f"  总像素: {total_pixels:,}")

    return class_counts_per_sample, pixel_counts


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train-images', type=str,
                       default='dataset/processed/train/images',
                       help='训练图像目录')
    parser.add_argument('--train-masks', type=str,
                       default='dataset/processed/train/masks',
                       help='训练mask目录')
    parser.add_argument('--val-images', type=str,
                       default='dataset/processed/val/images',
                       help='验证图像目录')
    parser.add_argument('--val-masks', type=str,
                       default='dataset/processed/val/masks',
                       help='验证mask目录')
    args = parser.parse_args()

    # 分析训练集
    train_sample_stats, train_pixel_stats = analyze_dataset(
        "训练集",
        args.train_images,
        args.train_masks
    )

    # 分析验证集
    val_sample_stats, val_pixel_stats = analyze_dataset(
        "验证集",
        args.val_images,
        args.val_masks
    )

    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)
    if train_sample_stats['has_cable'] < len([x for x in [train_sample_stats]]) * 0.5:
        print("[WARNING] 电缆样本覆盖率过低，可能导致模型无法有效学习")
    if train_sample_stats['has_tape'] < len([x for x in [train_sample_stats]]) * 0.3:
        print("[WARNING] 胶带样本覆盖率极低，这是导致 IoU 低的主要原因")
    if train_pixel_stats[0] / sum(train_pixel_stats.values()) > 0.8:
        print("[WARNING] 背景像素占比过高，建议使用 focal loss 或调整类别权重")
