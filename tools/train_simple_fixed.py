"""简化版训练脚本 - 快速测试修复效果

使用修复后的metrics和优化的参数
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

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.unetpp import NestedUNet
from models.losses import CombinedLoss
from data.dataset import CableDefectDataset
from utils.metrics import compute_metrics

def main():
    print("="*70)
    print("修复版训练 - 50 Epochs快速测试")
    print("="*70)

    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 数据加载
    print("\n[1] 加载数据集...")
    train_dataset = CableDefectDataset(
        "dataset/processed_v2/train/images",
        "dataset/processed_v2/train/masks",
        augment=True,
        target_size=(256, 256)
    )
    val_dataset = CableDefectDataset(
        "dataset/processed_v2/val/images",
        "dataset/processed_v2/val/masks",
        augment=False,
        target_size=(256, 256)
    )

    # 缺陷过采样
    weights = []
    defect_ids = [3, 4, 5]
    for i in range(len(train_dataset)):
        _, m = train_dataset[i]
        uniq = set(torch.unique(m).tolist())
        has_defect = len(uniq.intersection(defect_ids)) > 0
        weights.append(3.0 if has_defect else 1.0)

    n_defect = sum(1 for w in weights if w > 1.0)
    print(f"  缺陷样本: {n_defect}/{len(weights)} ({n_defect/len(weights)*100:.1f}%)")

    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        sampler=sampler,
        num_workers=0,
        drop_last=False
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    print(f"  训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}")

    # 模型
    print("\n[2] 构建模型...")
    model = NestedUNet(
        num_classes=6,
        deep_supervision=True,
        pretrained_encoder=False
    ).to(device)

    # 优化的类别权重
    class_weights = torch.tensor([
        0.5,   # background
        1.0,   # cable
        1.0,   # tape
        30.0,  # burr_defect - 大幅提升
        25.0,  # loose_defect
        30.0   # wrap_uneven
    ]).to(device)

    # 调整后的损失函数
    criterion = CombinedLoss(
        weight_ce=0.3,   # 降低CE权重
        weight_dice=3.0,  # 提高Dice权重（对小目标更好）
        class_weights=class_weights,
        dice_ignore_bg=False,  # 不忽略背景
        dice_skip_empty=False   # 不跳过空类
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=50, eta_min=1e-6
    )

    print(f"  类别权重: {class_weights.tolist()}")
    print(f"  损失权重: CE=0.3, Dice=3.0")

    # 训练
    print("\n[3] 开始训练...")
    print("-"*70)

    best_miou = 0.0
    best_epoch = 0

    for epoch in range(1, 51):
        # 训练
        model.train()
        train_loss = 0.0
        num_batches = 0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            if isinstance(outputs, list):
                # 深度监督：使用最后一个输出
                outputs = outputs[-1]

            loss, ce_loss, dice_loss = criterion(outputs, masks)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        avg_train_loss = train_loss / num_batches if num_batches > 0 else 0.0

        # 验证
        model.eval()
        all_preds, all_targets = [], []

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

        val_miou, precision, recall, iou_dict = compute_metrics(
            all_preds, all_targets, 6
        )

        # 更新学习率
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        # 打印进度（每10轮或第1轮）
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:2d}/50:")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val mIoU: {val_miou:.4f}")
            print(f"  LR: {current_lr:.6f}")
            print(f"  Class IoU:")
            for cls, iou_val in iou_dict.items():
                if cls < 6:  # 只打印有数据的类别
                    print(f"    类别 {cls}: {iou_val:.4f}")
            print()

        # 保存最佳模型
        if val_miou > best_miou:
            best_miou = val_miou
            best_epoch = epoch

            os.makedirs("checkpoints_v3", exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_miou": best_miou,
            }, "checkpoints_v3/best_model.pth")
            print(f"  *** 最佳模型更新 (Epoch {epoch}, mIoU={best_miou:.4f}) ***")
            print()

    # 保存训练历史
    print("\n" + "="*70)
    print("训练完成!")
    print("="*70)
    print(f"最佳 mIoU: {best_miou:.4f} (Epoch {best_epoch})")
    print(f"模型已保存: checkpoints_v3/best_model.pth")
    print("="*70)

if __name__ == "__main__":
    main()
