"""Binary defect segmentation with patch-based training

P1 + P2-A: Patch training + Binary classification
- 50% defect patches, 50% normal patches
- Binary task: defect vs non-defect
- Larger patch size for better defect visibility
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
from pathlib import Path
import numpy as np
import random
import os

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.unetpp import NestedUNet
from models.losses import CombinedLoss
from data.patch_dataset import PatchDefectDataset
from utils.metrics import compute_metrics


def main():
    print("="*70)
    print("二分类缺陷分割 + Patch训练")
    print("="*70)
    print("P1: Patch训练 (50%% 缺陷 + 50%% 正常)")
    print("P2-A: 二分类 (defect vs non-defect)")
    print("="*70)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 使用6类数据集，但转换为二分类
    print("\n[1] 加载数据集（Patch + 二分类）...")
    train_dataset = PatchDefectDataset(
        "dataset/processed_v2/train/images",
        "dataset/processed_v2/train/masks",
        patch_size=384,  # Fits min image size (448px)
        defect_ratio=0.5,  # 50% defect patches
        augment=True,
        target_size=(256, 256)  # Downsample for memory
    )

    val_dataset = PatchDefectDataset(
        "dataset/processed_v2/val/images",
        "dataset/processed_v2/val/masks",
        patch_size=384,
        defect_ratio=0.5,
        augment=False,
        target_size=(256, 256)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,  # Can use larger batch with patches
        shuffle=True,
        num_workers=0,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    print(f"  训练样本: {len(train_dataset)}")
    print(f"  验证样本: {len(val_dataset)}")
    print(f"  Patch size: 384 -> 256")
    print(f"  类别数: 2 (background, defect)")

    # 模型 - 二分类
    print("\n[2] 构建模型...")
    model = NestedUNet(
        num_classes=2,  # Binary: background, defect
        deep_supervision=False,  # Disable for simpler task
        pretrained_encoder=False
    ).to(device)

    # 损失函数 - 二分类优化
    # 对于极不平衡的二分类，提高缺陷权重
    class_weights = torch.tensor([
        1.0,    # background
        50.0    # defect - very high weight for rare defects
    ]).to(device)

    criterion = CombinedLoss(
        weight_ce=0.5,   # Balanced CE + Dice
        weight_dice=2.0,
        class_weights=class_weights,
        dice_ignore_bg=False,
        dice_skip_empty=False
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=5e-4,  # Higher LR for faster convergence on simple task
        weight_decay=1e-5
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-6
    )

    print(f"  类别权重: {class_weights.tolist()}")
    print(f"  损失函数: CE + Dice")
    print(f"  学习率: 5e-4 (with ReduceLROnPlateau)")

    # 训练
    print("\n[3] 开始训练...")
    print("-"*70)

    best_miou = 0.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, 101):
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
                outputs = outputs[-1]

            loss, ce_loss, dice_loss = criterion(outputs, masks)
            loss.backward()

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
            all_preds, all_targets, 2
        )

        # 只看缺陷类的 IoU
        defect_iou = iou_dict.get(1, 0.0)

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_miou)

        # 打印进度
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/100:")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val mIoU: {val_miou:.4f}")
            print(f"  Defect IoU: {defect_iou:.4f}")
            print(f"  LR: {current_lr:.6f}")
            print()

        # 保存最佳模型
        if val_miou > best_miou:
            best_miou = val_miou
            best_epoch = epoch
            patience_counter = 0

            os.makedirs("checkpoints_binary_patch", exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_miou": best_miou,
                "num_classes": 2,
            }, "checkpoints_binary_patch/best_model.pth")
            print(f"  *** 最佳模型更新 (Epoch {epoch}, mIoU={best_miou:.4f}, Defect IoU={defect_iou:.4f}) ***")
            print()
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= 20:
            print(f"Early stopping at epoch {epoch}")
            break

    print("\n" + "="*70)
    print("训练完成!")
    print("="*70)
    print(f"最佳 mIoU: {best_miou:.4f} (Epoch {best_epoch})")
    print(f"模型已保存: checkpoints_binary_patch/best_model.pth")
    print("="*70)


if __name__ == "__main__":
    main()
