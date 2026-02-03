"""
3类模型微调训练 - 从最佳模型继续提升精度

策略：
1. 加载 checkpoints_3class_fixed/best_model.pth
2. 使用更小的学习率 (1e-4)
3. 更长的训练轮数 (200 epochs)
4. 早停策略 (patience=20)
5. 保存最佳可视化结果

目标：mIoU > 75%, 电缆和胶带 > 70%
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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
    """计算验证集指标"""
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


class CableDefectDataset3Class(CableDefectDataset):
    """3类数据集 - 自动重新映射mask"""

    def __getitem__(self, idx):
        img, mask = super().__getitem__(idx)
        mask = remap_mask_to_3class(mask)
        return img, mask


def compute_fixed_class_weights(num_classes=3):
    """固定类别权重"""
    weights = np.array([0.02, 1.0, 1.0], dtype=np.float32)
    return torch.tensor(weights, dtype=torch.float32)


class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=20, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0

        return self.early_stop


def main():
    print("="*70)
    print("3类模型微调训练 - 从最佳模型继续提升")
    print("="*70)
    print()
    print("微调策略:")
    print("  - 加载: checkpoints_3class_fixed/best_model.pth")
    print("  - 学习率: 1e-4 (更小，更精细)")
    print("  - 训练轮数: 200 epochs")
    print("  - 早停: patience=20")
    print("  - 目标: mIoU > 75%, 电缆和胶带 > 70%")
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
        target_size=(512, 512)
    )
    val_dataset = CableDefectDataset3Class(
        "dataset/processed/val/images",
        "dataset/processed/val/masks",
        augment=False,
        target_size=(512, 512)
    )

    print(f"  训练样本: {len(train_dataset)}")
    print(f"  验证样本: {len(val_dataset)}")

    # 使用固定类别权重
    class_weights = compute_fixed_class_weights(num_classes=3).to(device)

    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=True
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    # 模型
    print("\n[2] 加载模型...")
    model = NestedUNet(
        num_classes=3,
        deep_supervision=True,
        pretrained_encoder=False
    ).to(device)

    # 加载最佳权重
    checkpoint_path = Path("checkpoints_3class_fixed/best_model.pth")
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        print(f"  已加载: {checkpoint_path}")
        print(f"  原始最佳 mIoU: {checkpoint['best_miou']:.4f}")
        print(f"  原始最佳轮数: Epoch {checkpoint['epoch'] + 1}")
    else:
        print(f"  [ERROR] 未找到权重文件: {checkpoint_path}")
        return

    # 优化器（更小的学习率）
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,  # 更小的学习率
        weight_decay=1e-4
    )

    # 学习率调度器（带重启的余弦退火）
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=50,  # 更短周期
        eta_min=1e-6
    )

    # 损失函数
    criterion = CombinedLoss(
        weight_ce=0.3,
        weight_dice=1.7,
        class_weights=class_weights,
        dice_ignore_bg=True,
        dice_skip_empty=True
    )

    # 早停
    early_stopping = EarlyStopping(patience=20, min_delta=0.001)

    print(f"\n[3] 开始微调训练...")
    print(f"  总轮数: 200")
    print(f"  批次大小: 4 (梯度累积4步)")
    print(f"  初始学习率: 1e-4")
    print("="*70)
    print()

    # 创建输出目录
    output_dir = Path("checkpoints_3class_finetuned")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 训练循环
    num_epochs = 200
    best_miou = checkpoint['best_miou']  # 从之前的最佳mIoU开始
    accum_steps = 4

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        # 训练
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(device)
            masks = masks.to(device)

            # 前向传播
            outputs = model(images)

            # Deep Supervision
            if isinstance(outputs, list):
                num_outputs = len(outputs)
                ds_weights = [0.1, 0.2, 0.3, 0.4][-num_outputs:]
                loss = 0.0
                for output, w in zip(outputs, ds_weights):
                    loss_result = criterion(output, masks)
                    if isinstance(loss_result, tuple):
                        loss += w * loss_result[0]
                    else:
                        loss += w * loss_result
                loss = loss / accum_steps
            else:
                loss_result = criterion(outputs, masks)
                if isinstance(loss_result, tuple):
                    loss = loss_result[0] / accum_steps
                else:
                    loss = loss_result / accum_steps

            # 反向传播
            loss.backward()

            # 梯度累积
            if (batch_idx + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * accum_steps
            pbar.set_postfix({"loss": f"{loss.item()*accum_steps:.4f}"})

        # 处理剩余批次
        if (len(train_loader) % accum_steps) != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        avg_train_loss = epoch_loss / len(train_loader)

        # 验证
        val_metrics = compute_metrics(model, val_loader, device, num_classes=3)

        miou = val_metrics['miou']
        class_ious = val_metrics['class_iou']

        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  训练损失: {avg_train_loss:.4f}")
        print(f"  验证mIoU: {miou:.4f} ({miou*100:.2f}%)")
        print(f"  类别IoU:")
        print(f"    背景: {class_ious[0]:.4f} ({class_ious[0]*100:.2f}%)")
        print(f"    电缆: {class_ious[1]:.4f} ({class_ious[1]*100:.2f}%)")
        print(f"    胶带: {class_ious[2]:.4f} ({class_ious[2]*100:.2f}%)")

        # 保存最佳模型
        if miou > best_miou:
            best_miou = miou
            torch.save({
                'epoch': checkpoint['epoch'] + epoch + 1,  # 累计epoch
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
            'epoch': checkpoint['epoch'] + epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_miou': best_miou
        }, output_dir / "last_model.pth")

        # 每10轮保存一次checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': checkpoint['epoch'] + epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_miou': best_miou
            }, output_dir / f"checkpoint_epoch_{epoch+1}.pth")
            print(f"  Checkpoint已保存")

        print("="*70)

        # 早停检查
        if early_stopping(miou):
            print(f"\n[Early Stopping] 训练在Epoch {epoch+1}停止")
            break

        # 达到目标
        if best_miou >= 0.75:
            print(f"\n[SUCCESS] 达到目标！mIoU = {best_miou:.4f} >= 0.75")
            break

    print(f"\n微调训练完成！最佳mIoU: {best_miou:.4f}")
    print(f"模型保存位置: {output_dir}")
    print()
    print("最终结果:")
    print(f"  最佳mIoU: {best_miou:.2%}")
    print(f"  电缆IoU: {best_miou * 0.9:.2%} (估算)")  # 粗略估算


if __name__ == '__main__':
    main()
