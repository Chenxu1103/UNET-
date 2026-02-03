"""3类训练 - Cable + Tape + Burr

目标：
- 识别3个类别：cable, tape, burr_defect
- 忽略其他缺陷类型（bulge, loose, damage）
- 用于胶带缠绕均匀性检测
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
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.unetpp import NestedUNet
from models.losses import CombinedLoss
from data.dataset import CableDefectDataset
from utils.metrics import compute_metrics


class CableTapeBurrDataset(CableDefectDataset):
    """3类数据集 - 只保留cable, tape, burr"""

    def __init__(self, image_dir, mask_dir, augment=True, target_size=(256, 256)):
        super().__init__(image_dir, mask_dir, augment, target_size)

        # 类别映射：7类 -> 3类
        # 0(bg)->0, 1(cable)->1, 2(tape)->2, 3(burr)->3, 4(bulge)->0, 5(loose)->0, 6(damage)->0
        self.class_mapping = {
            0: 0,  # background
            1: 1,  # cable
            2: 2,  # tape
            3: 3,  # burr_defect
            4: 0,  # bulge_defect -> background (忽略)
            5: 0,  # loose_defect -> background (忽略)
            6: 0   # damage_defect -> background (忽略)
        }

    def __getitem__(self, idx):
        image, mask = super().__getitem__(idx)

        # 重新映射mask
        remapped = torch.zeros_like(mask)
        for old_id, new_id in self.class_mapping.items():
            remapped[mask == old_id] = new_id

        return image, remapped


def main():
    print("="*70)
    print("3类训练 - Cable + Tape + Burr")
    print("="*70)
    print("目标: 识别电缆、胶带、毛刺，用于缠绕均匀性检测")
    print("="*70)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 使用原始7类数据，转换为3类
    print("\n[1] 加载数据集（7类 -> 3类）...")
    train_dataset = CableTapeBurrDataset(
        "dataset/processed/train/images",
        "dataset/processed/train/masks",
        augment=True,
        target_size=(256, 256)
    )

    val_dataset = CableTapeBurrDataset(
        "dataset/processed/val/images",
        "dataset/processed/val/masks",
        augment=False,
        target_size=(256, 256)
    )

    # 缺陷过采样（毛刺缺陷）
    weights = []
    defect_ids = set([3])  # 只需要burr_defect

    for i in range(len(train_dataset)):
        _, m = train_dataset[i]
        uniq = set(torch.unique(m).tolist())
        has_defect = len(uniq.intersection(defect_ids)) > 0
        weights.append(3.0 if has_defect else 1.0)

    n_defect = sum(1 for w in weights if w > 1.0)
    print(f"  毛刺样本: {n_defect}/{len(weights)} ({n_defect/len(weights)*100:.1f}%)")

    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        sampler=sampler,
        num_workers=0,
        drop_last=False
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    print(f"  训练样本: {len(train_dataset)}")
    print(f"  验证样本: {len(val_dataset)}")
    print(f"  类别数: 4 (background, cable, tape, burr)")

    # 模型
    print("\n[2] 构建模型...")
    model = NestedUNet(
        num_classes=4,  # 0:bg, 1:cable, 2:tape, 3:burr
        deep_supervision=True,
        pretrained_encoder=False
    ).to(device)

    # 类别权重 - 平衡4个类别
    class_weights = torch.tensor([
        0.5,   # background
        1.0,   # cable
        1.0,   # tape
        30.0   # burr_defect - 小目标，高权重
    ]).to(device)

    criterion = CombinedLoss(
        weight_ce=0.3,
        weight_dice=3.0,
        class_weights=class_weights,
        dice_ignore_bg=False,
        dice_skip_empty=False
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=100, eta_min=1e-6
    )

    print(f"  类别权重: {class_weights.tolist()}")

    # 训练
    print("\n[3] 开始训练...")
    print("-"*70)

    best_miou = 0.0
    best_epoch = 0

    for epoch in range(1, 101):
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
            all_preds, all_targets, 4
        )

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/100:")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val mIoU: {val_miou:.4f}")
            print(f"  LR: {current_lr:.6f}")
            print(f"  Class IoU:")
            class_names = ['bg', 'cable', 'tape', 'burr']
            for cls, iou_val in iou_dict.items():
                if cls < 4:
                    print(f"    {class_names[cls]:8s}: {iou_val:.4f}")
            print()

        if val_miou > best_miou:
            best_miou = val_miou
            best_epoch = epoch

            os.makedirs("checkpoints_3class", exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_miou": best_miou,
                "num_classes": 4,
            }, "checkpoints_3class/best_model.pth")
            print(f"  *** 最佳模型更新 (Epoch {epoch}, mIoU={best_miou:.4f}) ***")
            print()

    print("\n" + "="*70)
    print("训练完成!")
    print("="*70)
    print(f"最佳 mIoU: {best_miou:.4f} (Epoch {best_epoch})")
    print(f"模型已保存: checkpoints_3class/best_model.pth")
    print("="*70)


if __name__ == "__main__":
    main()
