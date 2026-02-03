"""模型训练脚本（优化版）

完整的训练流程：数据加载 → 模型构建 → 训练循环 → 验证评估 → 模型保存

优化项：
- 修复 float(total_loss) warning，使用 detach().item()
- drop_last=False 避免小数据集丢样本
- 缺陷样本过采样（WeightedRandomSampler）
- 验证阶段使用 outputs[-1] 而非 outputs[0]
- 添加 AMP 混合精度训练
- 添加梯度裁剪
- 添加 Cosine LR Scheduler
- 改进 checkpoint 保存/加载
- 添加随机种子设置
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
import argparse
from pathlib import Path
import sys
import os
import numpy as np
import random
from torch.cuda.amp import autocast, GradScaler

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.unetpp import NestedUNet
from models.losses import CombinedLoss
from data.dataset import CableDefectDataset
from utils.metrics import compute_metrics, print_metrics


class TrainArgs:
    """训练参数配置"""
    def __init__(self):
        self.train_img_dir = "dataset/processed/train/images"
        self.train_mask_dir = "dataset/processed/train/masks"
        self.val_img_dir = "dataset/processed/val/images"
        self.val_mask_dir = "dataset/processed/val/masks"
        self.num_classes = 7  # 包括背景
        self.num_epochs = 100
        self.batch_size = 4
        self.learning_rate = 1e-3
        self.weight_decay = 1e-4
        self.use_pretrained_encoder = False  # 使用自定义编码器
        self.deep_supervision = False  # 简化训练
        self.model_save_dir = "checkpoints"
        self.resume = None  # 从检查点恢复训练的路径
        self.start_epoch = 1  # 起始 epoch
        self.target_size = (256, 256)  # 统一图像尺寸
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 新增优化参数
        self.seed = 42
        self.use_amp = True  # 混合精度训练
        self.grad_clip = 1.0  # 梯度裁剪阈值（0表示不裁剪）
        self.use_weighted_sampler = True  # 是否使用缺陷样本过采样
        self.defect_boost = 2.0  # 含缺陷样本的采样权重倍数（改为2x避免过拟合）
        self.scheduler = "cosine"  # 学习率调度策略："cosine" or "none"
        self.min_lr = 1e-5  # 最小学习率


def set_seed(seed: int):
    """固定随机种子，提升复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    scaler: GradScaler = None,
    use_amp: bool = False,
    grad_clip: float = 0.0
) -> float:
    """单个epoch的训练

    Args:
        model: 分割模型
        train_loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备(CPU/GPU)
        epoch: 当前epoch编号
        scaler: AMP GradScaler
        use_amp: 是否使用混合精度训练
        grad_clip: 梯度裁剪阈值

    Returns:
        平均训练损失
    """
    model.train()
    train_loss = 0.0
    num_batches = 0

    for batch_idx, (images, masks) in enumerate(train_loader):
        images = images.to(device)  # shape (B,3,H,W)
        masks = masks.to(device)    # shape (B,H,W)

        optimizer.zero_grad(set_to_none=True)

        # 前向传播（支持 AMP）
        if use_amp and scaler is not None and device.type == "cuda":
            with autocast():
                outputs = model(images)
                # 计算损失
                if isinstance(outputs, list):
                    # 深度监督：对更"深"的输出给予更高权重（更贴近最终输出）
                    weights = torch.linspace(1.0, 2.0, steps=len(outputs), device=device)
                    weights = weights / weights.sum()
                    total_loss = 0.0
                    for w, out in zip(weights, outputs):
                        loss, _, _ = criterion(out, masks)
                        total_loss = total_loss + w * loss
                else:
                    total_loss, _, _ = criterion(outputs, masks)

            # 反向传播（AMP）
            scaler.scale(total_loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            # 计算损失
            if isinstance(outputs, list):
                weights = torch.linspace(1.0, 2.0, steps=len(outputs), device=device)
                weights = weights / weights.sum()
                total_loss = 0.0
                for w, out in zip(weights, outputs):
                    loss, _, _ = criterion(out, masks)
                    total_loss = total_loss + w * loss
            else:
                total_loss, _, _ = criterion(outputs, masks)

            # 反向传播
            total_loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        # 使用 detach().item() 而非 float() 避免 warning
        loss_item = total_loss.detach().item()
        train_loss += loss_item
        num_batches += 1

        # 定期打印进度
        if (batch_idx + 1) % max(1, len(train_loader) // 5) == 0:
            print(f"  [{batch_idx+1}/{len(train_loader)}] Loss: {loss_item:.4f}")

    avg_loss = train_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    num_classes: int
) -> tuple:
    """验证模型性能

    Args:
        model: 分割模型
        val_loader: 验证数据加载器
        device: 设备(CPU/GPU)
        num_classes: 类别数

    Returns:
        tuple: (mIoU, precision_dict, recall_dict, iou_dict)
    """
    model.eval()

    all_preds = []
    all_targets = []

    for images, masks in val_loader:
        images = images.to(device)
        masks = masks.to(device)

        # 前向传播（仅主输出）
        outputs = model(images)
        if isinstance(outputs, list):
            # 深度监督时通常最后一个输出更接近最终预测
            outputs = outputs[-1]

        # 获取预测mask
        preds = torch.argmax(outputs, dim=1)  # (B,H,W)

        # 收集结果
        all_preds.append(preds.cpu().numpy())
        all_targets.append(masks.cpu().numpy())

    # 合并所有批次
    all_preds = np.concatenate(all_preds, axis=0)  # (N,H,W)
    all_targets = np.concatenate(all_targets, axis=0)

    # 计算指标
    mIoU, precision, recall, iou_dict = compute_metrics(
        all_preds, all_targets, num_classes
    )

    return mIoU, precision, recall, iou_dict


def main(args: TrainArgs):
    """主训练函数

    Args:
        args: 训练参数
    """
    # 设置随机种子
    set_seed(args.seed)

    print("="*70)
    print("Cable Tape Defect Segmentation - Training (Optimized)")
    print("="*70)
    print(f"Configuration:")
    print(f"  Device: {args.device}")
    print(f"  AMP: {args.use_amp}")
    print(f"  Grad Clip: {args.grad_clip}")
    print(f"  Weighted Sampler: {args.use_weighted_sampler} (boost={args.defect_boost}x)")
    print(f"  Scheduler: {args.scheduler}")
    print(f"  Seed: {args.seed}")

    # 创建模型保存目录
    Path(args.model_save_dir).mkdir(parents=True, exist_ok=True)

    # 数据加载
    print("\n[1] Loading datasets...")
    train_dataset = CableDefectDataset(
        args.train_img_dir, args.train_mask_dir, augment=True, target_size=args.target_size
    )
    val_dataset = CableDefectDataset(
        args.val_img_dir, args.val_mask_dir, augment=False, target_size=args.target_size
    )

    # 缺陷样本过采样（小数据 + 类别不平衡时非常关键）
    if args.use_weighted_sampler:
        print("  [1.1] Computing sample weights for defect oversampling...")
        weights = []
        # 缺陷类别 ID（根据您的类别定义）
        defect_ids = set([3, 4, 5, 6])  # bulge_defect, loose_defect, burr_defect, thin_defect

        for i in range(len(train_dataset)):
            _, m = train_dataset[i]  # m: (H,W) tensor
            uniq = set(torch.unique(m).tolist())
            has_defect = len(uniq.intersection(defect_ids)) > 0
            weights.append(args.defect_boost if has_defect else 1.0)

        n_defect = sum(1 for w in weights if w > 1.0)
        print(f"    Defect samples: {n_defect}/{len(weights)} ({n_defect/len(weights)*100:.1f}%)")

        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=sampler,
            num_workers=0, drop_last=False  # 小数据集不建议丢弃尾batch
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=0, drop_last=False
        )

    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=0
    )

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")

    # 模型、损失、优化器
    print("\n[2] Building model...")
    model = NestedUNet(
        num_classes=args.num_classes,
        deep_supervision=args.deep_supervision,
        pretrained_encoder=args.use_pretrained_encoder
    ).to(args.device)

    print(f"  Model: NestedUNet")
    print(f"  Deep supervision: {args.deep_supervision}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 类别权重：使用等权重，避免过度强调缺陷类别导致误报
    # 顺序: [bg, cable, tape, bulge, loose, burr, thin]
    class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32).to(args.device)
    criterion = CombinedLoss(
        weight_ce=1.0,
        weight_dice=1.0,
        class_weights=class_weights,
        dice_ignore_bg=True,
        dice_skip_empty=True
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # 学习率调度（小数据更稳）
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.num_epochs, eta_min=args.min_lr
        )
        print(f"  Scheduler: CosineAnnealingLR (lr: {args.learning_rate} -> {args.min_lr})")
    else:
        scheduler = None
        print(f"  Scheduler: None (fixed LR)")

    # AMP GradScaler
    scaler = GradScaler(enabled=(args.use_amp and args.device.type == "cuda"))

    # 从检查点恢复
    best_mIoU = 0.0
    best_epoch = 0
    start_epoch = args.start_epoch

    if args.resume and os.path.exists(args.resume):
        print(f"\n[2.1] Loading checkpoint from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=args.device)
        # 兼容：旧版只存 state_dict，新版存 dict
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
            if "optimizer" in checkpoint and checkpoint["optimizer"] is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if scheduler is not None and "scheduler" in checkpoint and checkpoint["scheduler"] is not None:
                scheduler.load_state_dict(checkpoint["scheduler"])
            if "best_mIoU" in checkpoint:
                best_mIoU = float(checkpoint["best_mIoU"])
            if "epoch" in checkpoint:
                start_epoch = int(checkpoint["epoch"]) + 1
        else:
            model.load_state_dict(checkpoint)
        print(f"  Checkpoint loaded successfully, resuming from epoch {start_epoch}")
    elif args.resume:
        print(f"\n[WARNING] Checkpoint file not found: {args.resume}")
        print("  Starting training from scratch...")

    # 训练循环
    print(f"\n[3] Starting training from epoch {start_epoch}...")
    print("-"*70)

    for epoch in range(start_epoch, args.num_epochs + 1):
        # 训练
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, args.device, epoch,
            scaler=scaler, use_amp=args.use_amp, grad_clip=args.grad_clip
        )

        # 验证
        val_mIoU, precision, recall, iou_dict = validate(
            model, val_loader, args.device, args.num_classes
        )

        # 更新学习率
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            scheduler.step()

        # 打印进度
        print(f"Epoch {epoch}/{args.num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val mIoU: {val_mIoU:.4f}")
        print(f"  LR: {current_lr:.6f}")

        # 保存最佳模型
        if val_mIoU > best_mIoU:
            best_mIoU = val_mIoU
            best_epoch = epoch
            best_model_path = os.path.join(args.model_save_dir, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "best_mIoU": best_mIoU,
                "config": {
                    "num_classes": args.num_classes,
                    "target_size": args.target_size
                }
            }, best_model_path)
            print(f"  *** Best model updated (mIoU={best_mIoU:.4f}) ***")

        print()

    # 保存最后一轮模型
    print(f"\n[4] Training completed!")
    print(f"  Best mIoU: {best_mIoU:.4f} (Epoch {best_epoch})")

    last_model_path = os.path.join(args.model_save_dir, "last_model.pth")
    torch.save({
        "epoch": args.num_epochs,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "best_mIoU": best_mIoU,
        "config": {
            "num_classes": args.num_classes,
            "target_size": args.target_size
        }
    }, last_model_path)
    print(f"  Last model saved: {last_model_path}")
    print(f"  Best model saved: {os.path.join(args.model_save_dir, 'best_model.pth')}")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net++ for cable defect segmentation (Optimized)")
    parser.add_argument("--train_img_dir", type=str, default="dataset/processed/train/images")
    parser.add_argument("--train_mask_dir", type=str, default="dataset/processed/train/masks")
    parser.add_argument("--val_img_dir", type=str, default="dataset/processed/val/images")
    parser.add_argument("--val_mask_dir", type=str, default="dataset/processed/val/masks")
    parser.add_argument("--num_classes", type=int, default=7)
    parser.add_argument("--num_epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--model_save_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--start_epoch", type=int, default=1, help="Start epoch number (when resuming)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP (mixed precision)")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping threshold (0 to disable)")
    parser.add_argument("--no-weighted-sampler", action="store_true", help="Disable defect oversampling")
    parser.add_argument("--defect_boost", type=float, default=2.0, help="Defect sample weight boost factor")
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "none"], help="LR scheduler")

    parsed_args = parser.parse_args()

    args = TrainArgs()
    args.train_img_dir = parsed_args.train_img_dir
    args.train_mask_dir = parsed_args.train_mask_dir
    args.val_img_dir = parsed_args.val_img_dir
    args.val_mask_dir = parsed_args.val_mask_dir
    args.num_classes = parsed_args.num_classes
    args.num_epochs = parsed_args.num_epochs
    args.batch_size = parsed_args.batch_size
    args.learning_rate = parsed_args.learning_rate
    args.model_save_dir = parsed_args.model_save_dir
    args.resume = parsed_args.resume
    args.start_epoch = parsed_args.start_epoch
    args.seed = parsed_args.seed
    args.use_amp = not parsed_args.no_amp
    args.grad_clip = parsed_args.grad_clip
    args.use_weighted_sampler = not parsed_args.no_weighted_sampler
    args.defect_boost = parsed_args.defect_boost
    args.scheduler = parsed_args.scheduler

    main(args)
