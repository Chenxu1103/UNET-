"""
高级3类模型训练 - 使用Focal + Tversky + Dice损失

改进策略：
1. Focal Loss: 聚焦难样本，降低假阳性
2. Tversky Loss: 平衡precision和recall (alpha=0.3, beta=0.7 -> 更高precision)
3. 胶带中心裁剪: 随机围绕胶带区域裁剪
4. Hard Negative Mining: 从线上视频抽取困难背景样本
5. 强数据增强: Mosaic、颜色抖动、噪声、模糊等

目标：
- 电缆 IoU > 80%
- 胶带 IoU > 75%
- Background IoU > 95%
- mIoU > 83%
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
from models.losses import AdvancedCombinedLoss
from data.advanced_dataset import CableDefectDataset3Class, create_hard_negative_dataset
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

    miou, precision_dict, recall_dict, iou_dict = compute_metrics_base(
        all_preds, all_targets, num_classes
    )

    # 计算平均precision和recall（非背景类）
    precisions = [precision_dict.get(i, 0.0) for i in range(num_classes)]
    recalls = [recall_dict.get(i, 0.0) for i in range(num_classes)]
    avg_precision = np.mean(precisions[1:]) if len(precisions) > 1 else precisions[0]
    avg_recall = np.mean(recalls[1:]) if len(recalls) > 1 else recalls[0]

    return {
        'miou': miou,
        'precision': avg_precision,
        'recall': avg_recall,
        'precision_dict': precision_dict,
        'recall_dict': recall_dict,
        'class_iou': [iou_dict.get(i, 0.0) for i in range(num_classes)]
    }


def compute_class_weights(num_classes=3):
    """计算类别权重（针对背景主导的数据集）"""
    weights = np.array([0.02, 1.0, 1.0], dtype=np.float32)
    return torch.tensor(weights, dtype=torch.float32)


class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=30, min_delta=0.001):
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
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0

        return self.early_stop


def main():
    print("="*80)
    print("高级3类模型训练 - Focal + Tversky + Dice")
    print("="*80)
    print()
    print("改进策略:")
    print("  1. Focal Loss (gamma=2.0): 聚焦难样本，降低假阳性")
    print("  2. Tversky Loss (alpha=0.3, beta=0.7): 提升precision")
    print("  3. 胶带中心裁剪: 增强胶带细节学习")
    print("  4. Hard Negative Mining: 从线上视频抽取困难背景")
    print("  5. 强数据增强: Mosaic、颜色、噪声等")
    print()
    print("目标指标:")
    print("  - 电缆 IoU > 80%")
    print("  - 胶带 IoU > 75%")
    print("  - mIoU > 83%")
    print("="*80)
    print()

    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print()

    # ========================================
    # 1. 创建Hard Negative数据集（如果不存在）
    # ========================================
    hard_negative_dir = "dataset/hard_negatives"
    if not os.path.exists(hard_negative_dir):
        print("[1] 创建Hard Negative数据集...")
        print("  从data/raw视频中抽取困难背景样本...")
        create_hard_negative_dataset(
            raw_videos_dir="data/raw",
            output_dir=hard_negative_dir,
            num_frames=200,
            frame_size=(512, 512)
        )
        print()
    else:
        print(f"[1] Hard Negative数据集已存在: {hard_negative_dir}")
        print()

    # ========================================
    # 2. 加载数据集
    # ========================================
    print("[2] 加载数据集...")
    train_dataset = CableDefectDataset3Class(
        "dataset/processed/train/images",
        "dataset/processed/train/masks",
        augment=True,
        target_size=(512, 512),
        tape_crop_prob=0.3,  # 30%概率使用胶带中心裁剪
        hard_negative_dir=hard_negative_dir,
        hard_negative_prob=0.15,  # 15%概率采样困难背景
        use_albumentations=True
    )
    val_dataset = CableDefectDataset3Class(
        "dataset/processed/val/images",
        "dataset/processed/val/masks",
        augment=False,
        target_size=(512, 512),
        use_albumentations=True
    )

    print(f"  训练样本: {len(train_dataset)}")
    print(f"  验证样本: {len(val_dataset)}")
    print(f"  包含Hard Negative采样: {hard_negative_dir}")
    print()

    # 类别权重
    class_weights = compute_class_weights(num_classes=3).to(device)

    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,  # 降低batch size避免GPU OOM
        shuffle=True,
        num_workers=2,  # 使用多线程加载
        drop_last=True,
        pin_memory=True,
        prefetch_factor=2
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    # ========================================
    # 3. 模型
    # ========================================
    print("[3] 创建模型...")
    model = NestedUNet(
        num_classes=3,
        deep_supervision=True,
        pretrained_encoder=False
    ).to(device)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print()

    # ========================================
    # 4. 损失函数 - AdvancedCombinedLoss（87%+优化）
    # ========================================
    print("[4] 损失函数配置:")
    print("  - Focal Loss (weight=0.35, gamma=2.0): 聚焦难样本")
    print("  - Tversky Loss (weight=0.45, alpha=0.25, beta=0.75): 提升precision")
    print("  - Dice Loss (weight=0.20): 保证区域重叠")
    print()

    criterion = AdvancedCombinedLoss(
        weight_focal=0.35,
        weight_tversky=0.45,
        weight_dice=0.20,
        focal_gamma=2.0,
        tversky_alpha=0.25,  # 更低的alpha -> 更高的precision
        tversky_beta=0.75,   # 更高的beta -> 惩罚假阳性
        class_weights=class_weights,
        dice_ignore_bg=True
    )

    # ========================================
    # 5. 优化器和学习率调度
    # ========================================
    print("[5] 优化器配置:")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-4,  # 初始学习率
        weight_decay=1e-4
    )

    # OneCycleLR学习率调度
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=2e-4,
        epochs=150,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,  # 前10%用于warmup
        anneal_strategy='cos',
        div_factor=10,
        final_div_factor=100
    )

    print("  - AdamW优化器: lr=2e-4, weight_decay=1e-4")
    print("  - OneCycleLR调度器: max_lr=2e-4")
    print()

    # ========================================
    # 6. 训练
    # ========================================
    print("[6] 开始训练...")
    print("="*80)
    print()

    output_dir = Path("checkpoints_3class_advanced")
    output_dir.mkdir(parents=True, exist_ok=True)

    num_epochs = 150
    best_miou = 0.0
    accum_steps = 2  # 梯度累积
    early_stopping = EarlyStopping(patience=30, min_delta=0.001)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_focal = 0.0
        epoch_tversky = 0.0
        epoch_dice = 0.0

        optimizer.zero_grad()

        # 训练循环
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
                scheduler.step()
                optimizer.zero_grad()

            # 记录损失
            if isinstance(loss_result, tuple):
                focal_loss, tversky_loss, dice_loss = loss_result[1], loss_result[2], loss_result[3]
            else:
                focal_loss = tversky_loss = dice_loss = 0.0

            epoch_loss += loss.item() * accum_steps
            epoch_focal += focal_loss
            epoch_tversky += tversky_loss
            epoch_dice += dice_loss

            pbar.set_postfix({
                "loss": f"{loss.item()*accum_steps:.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}"
            })

        # 处理剩余批次
        if (len(train_loader) % accum_steps) != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # 平均损失
        num_batches = len(train_loader)
        avg_loss = epoch_loss / num_batches
        avg_focal = epoch_focal / num_batches
        avg_tversky = epoch_tversky / num_batches
        avg_dice = epoch_dice / num_batches

        # 验证
        val_metrics = compute_metrics(model, val_loader, device, num_classes=3)
        miou = val_metrics['miou']
        class_ious = val_metrics['class_iou']
        precision = val_metrics['precision']
        recall = val_metrics['recall']

        # 打印结果
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  训练损失: {avg_loss:.4f}")
        print(f"    - Focal: {avg_focal:.4f}, Tversky: {avg_tversky:.4f}, Dice: {avg_dice:.4f}")
        print(f"  验证 mIoU: {miou:.4f} ({miou*100:.2f}%)")
        print(f"  验证 Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"  验证 Recall: {recall:.4f} ({recall*100:.2f}%)")
        print(f"  类别 IoU:")
        print(f"    - 背景: {class_ious[0]:.4f} ({class_ious[0]*100:.2f}%)")
        print(f"    - 电缆: {class_ious[1]:.4f} ({class_ious[1]*100:.2f}%)")
        print(f"    - 胶带: {class_ious[2]:.4f} ({class_ious[2]*100:.2f}%)")

        # 保存最佳模型
        if miou > best_miou:
            best_miou = miou
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_miou': best_miou,
                'class_iou': class_ious,
                'precision': precision,
                'recall': recall
            }, output_dir / "best_model.pth")
            print(f"  [*] 最佳模型已保存 (mIoU: {best_miou:.4f})")
        else:
            print(f"  当前最佳mIoU: {best_miou:.4f}")

        # 保存checkpoint
        if (epoch + 1) % 25 == 0:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_miou': best_miou
            }, output_dir / f"checkpoint_epoch_{epoch+1}.pth")
            print(f"  Checkpoint已保存")

        # 早停检查
        if early_stopping(miou):
            print(f"\n[Early Stopping] 训练在Epoch {epoch+1}停止")
            break

        # 达到目标
        if best_miou >= 0.83:
            print(f"\n[SUCCESS] 达到目标！mIoU = {best_miou:.4f} >= 0.83")
            break

        print(f"{'='*80}\n")

    # 训练完成
    print(f"\n训练完成！最佳mIoU: {best_miou:.4f}")
    print(f"模型保存位置: {output_dir}/best_model.pth")
    print()
    print("最终结果:")
    print(f"  最佳 mIoU: {best_miou:.2%}")
    print(f"  电缆 IoU: {class_ious[1]:.2%}")
    print(f"  胶带 IoU: {class_ious[2]:.2%}")


if __name__ == '__main__':
    main()
