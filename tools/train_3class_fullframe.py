"""
3类模型全画面训练 - 支持任意尺寸的输入图像

特点：
1. 全画面训练 - 不使用固定ROI
2. 从头训练或从预训练模型微调
3. 自动类别权重平衡
4. 早停和学习率调度
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


def compute_class_weights_from_dataset(dataset, num_classes=3):
    """从数据集计算类别权重"""
    print("计算类别分布...")
    class_pixel_counts = np.zeros(num_classes, dtype=np.int64)

    # 采样计算（避免遍历整个数据集）
    sample_indices = random.sample(range(len(dataset)), min(100, len(dataset)))

    for idx in sample_indices:
        _, mask = dataset[idx]
        if torch.is_tensor(mask):
            mask_np = mask.cpu().numpy()
        else:
            mask_np = mask

        for cls in range(num_classes):
            class_pixel_counts[cls] += (mask_np == cls).sum()

    # 计算权重：median / count
    median_count = np.median(class_pixel_counts[class_pixel_counts > 0])
    weights = median_count / (class_pixel_counts + 1e-8)

    # 归一化，让cable和tape权重接近1
    weights[1:] = weights[1:] / weights[1]

    print(f"\n类别权重计算结果:")
    class_names = ['background', 'cable', 'tape']
    for cls in range(num_classes):
        print(f"  {class_names[cls]}: {weights[cls]:.3f}")

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
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0

        return self.early_stop


def main():
    print("="*70)
    print("3类模型全画面训练")
    print("="*70)
    print()
    print("训练配置:")
    print("  - 全画面输入: 保留原始画面尺寸")
    print("  - 输入尺寸: 512x512 (resize)")
    print("  - 类别数: 3 (background, cable, tape)")
    print("  - 自动类别权重平衡")
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

    # 数据路径
    data_root = Path("dataset/processed")
    train_img_dir = data_root / "train" / "images"
    train_mask_dir = data_root / "train" / "masks"
    val_img_dir = data_root / "val" / "images"
    val_mask_dir = data_root / "val" / "masks"

    # 检查数据是否存在
    if not train_img_dir.exists():
        print(f"\n[错误] 训练数据不存在: {train_img_dir}")
        print("请先运行数据预处理脚本生成训练数据")
        return

    print("\n[1] 加载数据集...")
    train_dataset = CableDefectDataset3Class(
        str(train_img_dir),
        str(train_mask_dir),
        augment=True,
        target_size=(512, 512)
    )
    val_dataset = CableDefectDataset3Class(
        str(val_img_dir),
        str(val_mask_dir),
        augment=False,
        target_size=(512, 512)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    print(f"  训练样本: {len(train_dataset)}")
    print(f"  验证样本: {len(val_dataset)}")
    print()

    # 计算类别权重
    class_weights = compute_class_weights_from_dataset(train_dataset, num_classes=3)
    class_weights = class_weights.to(device)

    # 模型
    print("[2] 构建模型...")
    model = NestedUNet(
        num_classes=3,
        deep_supervision=True,
        pretrained_encoder=False
    ).to(device)

    # 尝试加载预训练权重
    pretrained_path = "checkpoints_3class_finetuned/best_model.pth"
    if Path(pretrained_path).exists():
        print(f"  加载预训练模型: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location=device, weights_only=False)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print("  预训练权重已加载（微调模式）")
    else:
        print("  从头训练（无预训练权重）")

    # 损失函数
    criterion = CombinedLoss(
        num_classes=3,
        weight=class_weights,
        dice_weight=0.7,
        ce_weight=0.3
    )

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=100, eta_min=1e-6
    )

    # 早停
    early_stopping = EarlyStopping(patience=20, min_delta=0.001)

    # 创建输出目录
    output_dir = Path("checkpoints_3class_fullframe")
    output_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("[3] 开始训练...")
    print("="*70)

    num_epochs = 100
    best_miou = 0.0

    for epoch in range(num_epochs):
        # 训练
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            if isinstance(outputs, list):
                outputs = outputs[-1]

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        train_loss /= len(train_loader)

        # 验证
        val_metrics = compute_metrics(model, val_loader, device, num_classes=3)
        val_miou = val_metrics['miou']

        # 学习率调度
        scheduler.step()

        # 打印结果
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val mIoU: {val_miou:.4f}")
        print(f"  Class IoU: BG={val_metrics['class_iou'][0]:.4f} "
              f"Cable={val_metrics['class_iou'][1]:.4f} "
              f"Tape={val_metrics['class_iou'][2]:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # 保存最佳模型
        if val_miou > best_miou:
            best_miou = val_miou
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'miou': val_miou,
                'class_iou': val_metrics['class_iou'],
                'class_names': ['background', 'cable', 'tape']
            }, output_dir / 'best_model.pth')
            print(f"  *** Best model saved (mIoU={best_miou:.4f}) ***")

        # 保存周期性checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'miou': val_miou,
                'class_iou': val_metrics['class_iou'],
                'class_names': ['background', 'cable', 'tape']
            }, output_dir / f'checkpoint_epoch_{epoch+1}.pth')

        # 早停检查
        if early_stopping(val_miou):
            print(f"\n早停触发，停止训练 (Epoch {epoch+1})")
            break

        print("="*70)

    # 保存最后一个模型
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'miou': val_miou,
        'best_miou': best_miou,
        'class_iou': val_metrics['class_iou'],
        'class_names': ['background', 'cable', 'tape']
    }, output_dir / 'last_model.pth')

    print()
    print("="*70)
    print("训练完成!")
    print(f"最佳 mIoU: {best_miou:.4f}")
    print(f"模型保存在: {output_dir.absolute()}")
    print("="*70)


if __name__ == '__main__':
    main()
