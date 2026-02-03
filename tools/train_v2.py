"""优化版训练脚本 - 6类分割

改进项：
1. 去除鼓包缺陷，厚度不足改为包裹不均匀
2. 调整类别权重以应对不平衡
3. 降低学习率，启用深度监督
4. 增加训练轮次
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


class TrainArgsV2:
    """V2训练参数配置 - 针对6类分割优化"""
    def __init__(self):
        # 数据路径
        self.train_img_dir = "dataset/processed_v2/train/images"
        self.train_mask_dir = "dataset/processed_v2/train/masks"
        self.val_img_dir = "dataset/processed_v2/val/images"
        self.val_mask_dir = "dataset/processed_v2/val/masks"

        # 类别配置: background, cable, tape, burr_defect, loose_defect, wrap_uneven
        self.num_classes = 6

        # 训练参数
        self.num_epochs = 200  # 小数据需要更多轮次
        self.batch_size = 4    # 保持较小batch size
        self.learning_rate = 1e-4  # 降低学习率提高稳定性
        self.weight_decay = 1e-4

        # 模型参数
        self.use_pretrained_encoder = False
        self.deep_supervision = True  # 启用深度监督

        # 保存路径
        self.model_save_dir = "checkpoints_v2"
        self.resume = None
        self.start_epoch = 1
        self.target_size = (256, 256)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 优化参数
        self.seed = 42
        self.use_amp = True
        self.grad_clip = 1.0
        self.use_weighted_sampler = True
        self.defect_boost = 3.0  # 增加到3x（缺陷样本更少）
        self.scheduler = "cosine"
        self.min_lr = 1e-6

        # 类别权重 (根据数据分布调整)
        # background: 88.88% -> 0.5
        # cable: 5.87% -> 1.0
        # tape: 5.12% -> 1.0
        # burr_defect: 0.00% -> 15.0 (极少)
        # loose_defect: 0.08% -> 12.0
        # wrap_uneven: 0.05% -> 15.0 (最少)
        self.class_weights = torch.tensor([
            0.5,   # background - 降低权重
            1.0,   # cable
            1.0,   # tape
            15.0,  # burr_defect - 大幅提升
            12.0,  # loose_defect
            15.0   # wrap_uneven - 最稀缺
        ])


def set_seed(seed: int):
    """固定随机种子"""
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
    """单个epoch的训练"""
    model.train()
    train_loss = 0.0
    num_batches = 0

    for batch_idx, (images, masks) in enumerate(train_loader):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad(set_to_none=True)

        # 前向传播（支持 AMP）
        if use_amp and scaler is not None and device.type == "cuda":
            with autocast():
                outputs = model(images)
                if isinstance(outputs, list):
                    # 深度监督：更深的输出权重更高
                    weights = torch.linspace(1.0, 2.0, steps=len(outputs), device=device)
                    weights = weights / weights.sum()
                    total_loss = 0.0
                    for w, out in zip(weights, outputs):
                        loss, _, _ = criterion(out, masks)
                        total_loss = total_loss + w * loss
                else:
                    total_loss, _, _ = criterion(outputs, masks)

            scaler.scale(total_loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            if isinstance(outputs, list):
                weights = torch.linspace(1.0, 2.0, steps=len(outputs), device=device)
                weights = weights / weights.sum()
                total_loss = 0.0
                for w, out in zip(weights, outputs):
                    loss, _, _ = criterion(out, masks)
                    total_loss = total_loss + w * loss
            else:
                total_loss, _, _ = criterion(outputs, masks)

            total_loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        loss_item = total_loss.detach().item()
        train_loss += loss_item
        num_batches += 1

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
    """验证模型性能"""
    model.eval()
    all_preds = []
    all_targets = []

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

    mIoU, precision, recall, iou_dict = compute_metrics(
        all_preds, all_targets, num_classes
    )

    return mIoU, precision, recall, iou_dict


def main(args: TrainArgsV2):
    """主训练函数"""
    set_seed(args.seed)

    print("="*70)
    print("Cable Tape Defect Segmentation - Training V2 (6 Classes)")
    print("="*70)
    print(f"Classes: background, cable, tape, burr_defect, loose_defect, wrap_uneven")
    print(f"\nConfiguration:")
    print(f"  Device: {args.device}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  Deep Supervision: {args.deep_supervision}")
    print(f"  AMP: {args.use_amp}")
    print(f"  Grad Clip: {args.grad_clip}")
    print(f"  Weighted Sampler: {args.use_weighted_sampler} (boost={args.defect_boost}x)")
    print(f"  Scheduler: {args.scheduler}")

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

    # 缺陷样本过采样
    if args.use_weighted_sampler:
        print("  [1.1] Computing sample weights for defect oversampling...")
        weights = []
        defect_ids = set([3, 4, 5])  # burr_defect, loose_defect, wrap_uneven

        for i in range(len(train_dataset)):
            _, m = train_dataset[i]
            uniq = set(torch.unique(m).tolist())
            has_defect = len(uniq.intersection(defect_ids)) > 0
            weights.append(args.defect_boost if has_defect else 1.0)

        n_defect = sum(1 for w in weights if w > 1.0)
        print(f"    Defect samples: {n_defect}/{len(weights)} ({n_defect/len(weights)*100:.1f}%)")

        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=sampler,
            num_workers=0, drop_last=False
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=0, drop_last=False
        )

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

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

    # 类别权重
    class_weights = args.class_weights.to(args.device)
    print(f"\n  Class weights: {class_weights.tolist()}")

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

    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.num_epochs, eta_min=args.min_lr
        )
        print(f"  Scheduler: CosineAnnealingLR (lr: {args.learning_rate} -> {args.min_lr})")
    else:
        scheduler = None

    scaler = GradScaler(enabled=(args.use_amp and args.device.type == "cuda"))

    # 从检查点恢复
    best_mIoU = 0.0
    best_epoch = 0
    start_epoch = args.start_epoch

    if args.resume and os.path.exists(args.resume):
        print(f"\n[2.1] Loading checkpoint from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=args.device, weights_only=False)
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
        print(f"  Checkpoint loaded, resuming from epoch {start_epoch}")

    # 训练循环
    print(f"\n[3] Starting training from epoch {start_epoch}...")
    print("-"*70)

    for epoch in range(start_epoch, args.num_epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, args.device, epoch,
            scaler=scaler, use_amp=args.use_amp, grad_clip=args.grad_clip
        )

        val_mIoU, precision, recall, iou_dict = validate(
            model, val_loader, args.device, args.num_classes
        )

        current_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            scheduler.step()

        print(f"Epoch {epoch}/{args.num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val mIoU: {val_mIoU:.4f}")
        print(f"  Class IoU: {iou_dict}")
        print(f"  LR: {current_lr:.6f}")

        if val_mIoU > best_mIoU:
            best_mIoU = val_mIoU
            best_epoch = epoch
            best_model_path = os.path.join(args.model_save_dir, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler else None,
                "best_mIoU": best_mIoU,
                "num_classes": args.num_classes,
                "class_names": ["background", "cable", "tape", "burr_defect", "loose_defect", "wrap_uneven"]
            }, best_model_path)
            print(f"  *** Best model saved (mIoU={best_mIoU:.4f}) ***")

        print()

    # 保存最后一轮模型
    print(f"\n[4] Training completed!")
    print(f"  Best mIoU: {best_mIoU:.4f} (Epoch {best_epoch})")

    last_model_path = os.path.join(args.model_save_dir, "last_model.pth")
    torch.save({
        "epoch": args.num_epochs,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "best_mIoU": best_mIoU,
        "num_classes": args.num_classes,
        "class_names": ["background", "cable", "tape", "burr_defect", "loose_defect", "wrap_uneven"]
    }, last_model_path)
    print(f"  Last model saved: {last_model_path}")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train V2 - 6 classes optimized")
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--no_amp", action="store_true", help="Disable AMP")
    parser.add_argument("--defect_boost", type=float, default=3.0, help="Defect sample boost factor")

    parsed_args = parser.parse_args()

    args = TrainArgsV2()
    args.num_epochs = parsed_args.num_epochs
    args.batch_size = parsed_args.batch_size
    args.learning_rate = parsed_args.learning_rate
    args.resume = parsed_args.resume
    args.use_amp = not parsed_args.no_amp
    args.defect_boost = parsed_args.defect_boost

    main(args)
