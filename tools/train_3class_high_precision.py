"""
3类高精度训练方案 - 目标mIoU > 83%

训练目标：
- 只检测3类：背景、电缆、胶带
- 毛刺数据少，暂时忽略（可后续作为第4类）
- 电缆和胶带mIoU > 83%

策略：
1. 类别简化：4类 -> 3类（bg, cable, tape）
2. 强数据增强：提高泛化能力
3. Dice Loss为主：对小目标（胶带）效果更好
4. 类别平衡：cable和tape权重相等
5. 大输入尺寸：512x512（保留更多细节）
6. 长时间训练：150+ epochs

作者：CX
日期：2025-12-29
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
import sys
from pathlib import Path
import numpy as np
import random
import os
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.unetpp import NestedUNet
from models.losses import CombinedLoss
from data.dataset import CableDefectDataset
from utils.metrics import compute_metrics as compute_metrics_base


def compute_metrics(model, val_loader, device, num_classes=3):
    """计算验证集指标的包装函数"""
    all_preds = []
    all_targets = []

    model.eval()
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            if isinstance(outputs, list):
                outputs = outputs[-1]

            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(masks.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    miou, precision, recall, iou_dict = compute_metrics_base(
        all_preds, all_targets, num_classes
    )

    return {
        'miou': miou,
        'precision': precision,
        'recall': recall,
        'class_iou': [iou_dict.get(i, 0.0) for i in range(num_classes)]
    }


def remap_mask_to_3class(mask):
    """
    将7类mask重新映射为3类

    原始7类：
    0: background
    1: cable
    2: tape
    3: bulge_defect -> 0 (bg)
    4: loose_defect -> 0 (bg)
    5: burr_defect -> 0 (bg)  # 暂时忽略毛刺
    6: thin_defect -> 0 (bg)

    新3类：
    0: background
    1: cable
    2: tape

    支持numpy数组和torch.Tensor
    """
    mapping = {
        0: 0,  # background -> background
        1: 1,  # cable -> cable
        2: 2,  # tape -> tape
        3: 0,  # bulge -> background
        4: 0,  # loose -> background
        5: 0,  # burr -> background（暂时忽略）
        6: 0   # thin -> background
    }

    # 处理torch.Tensor
    if torch.is_tensor(mask):
        remapped = torch.zeros_like(mask)
        for old_val, new_val in mapping.items():
            remapped[mask == old_val] = new_val
    else:
        # 处理numpy数组
        remapped = mask.copy()
        for old_val, new_val in mapping.items():
            remapped[mask == old_val] = new_val

    return remapped


class CableDefectDataset3Class(CableDefectDataset):
    """3类数据集 - 自动重新映射mask"""

    def __getitem__(self, idx):
        img, mask = super().__getitem__(idx)

        # 重新映射mask为3类
        mask = remap_mask_to_3class(mask)

        return img, mask


def compute_class_weights(dataset, num_classes=3):
    """
    计算类别权重（基于数据集统计）

    策略：
    - 使用中位数频率作为基准
    - 权重 = 中位数频率 / 当前类频率
    - 这样可以让少的类别权重更高，但不会过度偏向少数类
    """
    print("\n计算类别权重...")

    # 统计每个类的像素数
    class_pixel_counts = np.zeros(num_classes, dtype=np.int64)

    sample_indices = random.sample(range(len(dataset)), min(100, len(dataset)))

    for idx in tqdm(sample_indices, desc="统计类别分布"):
        _, mask = dataset[idx]
        unique, counts = np.unique(mask, return_counts=True)
        for cls, count in zip(unique, counts):
            if cls < num_classes:
                class_pixel_counts[cls] += count

    print(f"类别像素统计（采样）:")
    for cls in range(num_classes):
        print(f"  类别{cls}: {class_pixel_counts[cls]:,} 像素")

    # 计算权重（使用中位数频率）
    median_count = np.median(class_pixel_counts[1:])  # 排除背景
    weights = median_count / (class_pixel_counts + 1e-8)

    # 归一化权重，让cable和tape权重接近1
    weights[1:] = weights[1:] / weights[1]

    # 降低背景权重（背景占89%，需要更低的权重）
    # 对于类别不平衡严重的情况，背景权重应该远小于1
    weights[0] = weights[0] * 0.3  # 进一步降低背景权重

    # 限制权重范围，避免过度偏向
    weights = np.clip(weights, 0.1, 3.0)  # 扩大范围，允许更低的背景权重

    print(f"\n计算得到的类别权重:")
    class_names = ['background', 'cable', 'tape']
    for cls in range(num_classes):
        print(f"  {class_names[cls]}: {weights[cls]:.3f}")

    return torch.tensor(weights, dtype=torch.float32)


def main():
    print("="*70)
    print("3类高精度训练 - 目标mIoU > 83%")
    print("="*70)
    print()
    print("训练配置:")
    print("  - 类别数: 3 (bg, cable, tape)")
    print("  - 输入尺寸: 512x512（高分辨率）")
    print("  - 批次大小: 4")
    print("  - 训练轮数: 150")
    print("  - 优化器: AdamW")
    print("  - 学习率: 0.001 (带cosine调度)")
    print("  - 损失函数: Dice Loss为主（对胶带更好）")
    print("="*70)
    print()

    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print()

    # 数据加载
    print("[1] 加载数据集...")
    train_dataset = CableDefectDataset3Class(
        "dataset/processed/train/images",
        "dataset/processed/train/masks",
        augment=True,
        target_size=(512, 512)  # 提高分辨率
    )
    val_dataset = CableDefectDataset3Class(
        "dataset/processed/val/images",
        "dataset/processed/val/masks",
        augment=False,
        target_size=(512, 512)
    )

    print(f"  训练样本: {len(train_dataset)}")
    print(f"  验证样本: {len(val_dataset)}")

    # 计算类别权重
    class_weights = compute_class_weights(train_dataset, num_classes=3).to(device)

    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,  # 大输入尺寸，小批次
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=True
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    # 模型
    print("\n[2] 构建模型...")
    model = NestedUNet(
        num_classes=3,
        deep_supervision=True,  # 使用deep supervision
        pretrained_encoder=False
    ).to(device)

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0001
    )

    # 学习率调度器（cosine annealing）
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=150,
        eta_min=0.00001
    )

    # 损失函数 - CE + Dice组合
    # CE Loss处理类别不平衡，Dice Loss优化区域重叠
    criterion = CombinedLoss(
        weight_ce=1.0,       # 启用CE Loss（关键：处理类别不平衡）
        weight_dice=1.0,     # Dice Loss优化区域重叠
        class_weights=class_weights,
        dice_ignore_bg=True,   # 忽略背景（背景占89%，会主导Dice）
        dice_skip_empty=True   # 跳过GT中不存在的类别
    )

    print(f"\n[3] 开始训练...")
    print(f"  总轮数: 150")
    print(f"  批次大小: 4")
    print(f"  初始学习率: 0.001")
    print("="*70)
    print()

    # 创建输出目录
    output_dir = Path("checkpoints_3class_high_precision")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 训练循环
    num_epochs = 150
    best_miou = 0.0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        # 训练
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(device)
            masks = masks.to(device)

            # 前向传播
            optimizer.zero_grad()
            outputs = model(images)

            # 计算损失
            if isinstance(outputs, list):
                main_output = outputs[-1]  # deep supervision，取最后一个
            else:
                main_output = outputs

            loss_result = criterion(main_output, masks)
            # CombinedLoss返回tuple: (loss, ce_loss, dice_loss)
            if isinstance(loss_result, tuple):
                loss = loss_result[0]
            else:
                loss = loss_result

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = epoch_loss / len(train_loader)

        # 验证
        model.eval()
        val_metrics = compute_metrics(model, val_loader, device, num_classes=3)

        miou = val_metrics['miou']
        class_ious = val_metrics['class_iou']

        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  训练损失: {avg_train_loss:.4f}")
        print(f"  验证mIoU: {miou:.4f} ({miou*100:.2f}%)")
        print(f"  类别IoU:")
        print(f"    背景: {class_ious[0]:.4f}")
        print(f"    电缆: {class_ious[1]:.4f}")
        print(f"    胶带: {class_ious[2]:.4f}")

        # 保存最佳模型
        if miou > best_miou:
            best_miou = miou
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_miou': best_miou,
                'class_iou': class_ious
            }, output_dir / "best_model.pth")
            print(f"  [OK] 最佳模型已保存 (mIoU: {best_miou:.4f})")
        else:
            print(f"  当前最佳mIoU: {best_miou:.4f}")

        # 更新学习率
        scheduler.step()

        # 保存最后一个模型
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_miou': best_miou
        }, output_dir / "last_model.pth")

        # 每10轮保存一次checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_miou': best_miou
            }, output_dir / f"checkpoint_epoch_{epoch+1}.pth")
            print(f"  Checkpoint已保存")

        print("="*70)

        # 达到目标
        if best_miou >= 0.83:
            print(f"\n[SUCCESS] 达到目标！mIoU = {best_miou:.4f} >= 0.83")
            break

    print(f"\n训练完成！最佳mIoU: {best_miou:.4f}")
    print(f"模型保存位置: {output_dir}")


if __name__ == '__main__':
    main()
