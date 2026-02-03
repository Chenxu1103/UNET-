"""
电缆胶带缠绕缺陷检测 - 训练脚本

支持轻量化U-Net++模型（MobileNetV3等）
根据项目方案：绕包机器算法检测项目方案以及实施计划
"""
import os
import sys
import argparse
import yaml
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
import numpy as np

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.models.unetpp_lightweight import LightweightNestedUNet, create_lightweight_unet
from src.models.unetpp import NestedUNet
from src.models.losses import CombinedLoss
from src.data.dataset import CableDefectDataset
from src.utils.metrics import compute_metrics, print_metrics


class TrainConfig:
    """训练配置"""
    def __init__(self, config_path: str):
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)

        # 数据配置
        data = cfg['data']
        self.train_img_dir = data['train_img_dir']
        self.train_mask_dir = data['train_mask_dir']
        self.val_img_dir = data['val_img_dir']
        self.val_mask_dir = data['val_mask_dir']
        self.num_classes = data['num_classes']
        self.target_size = tuple(data['target_size'])
        self.batch_size = data['batch_size']
        self.num_workers = data.get('num_workers', 4)
        self.augmentation = data['augmentation']

        # 模型配置
        model = cfg['model']
        self.model_name = model['name']
        self.encoder = model['encoder']
        self.pretrained_encoder = model['pretrained_encoder']
        self.deep_supervision = model['deep_supervision']
        self.input_channels = model['input_channels']

        # 训练配置
        train = cfg['training']
        self.num_epochs = train['num_epochs']
        self.learning_rate = train['learning_rate']
        self.min_lr = train['min_lr']
        self.weight_decay = train['weight_decay']
        self.scheduler = train['scheduler']
        self.optimizer = train['optimizer']
        self.use_amp = train['use_amp']
        self.grad_clip = train['grad_clip']
        self.use_weighted_sampler = train['use_weighted_sampler']
        self.defect_boost = train['defect_boost']
        self.seed = train['seed']

        # 检查点配置
        ckpt = cfg['checkpoint']
        self.checkpoint_dir = ckpt['save_dir']
        self.save_every_n_epochs = ckpt.get('save_every_n_epochs', 10)
        self.save_best_model = ckpt.get('save_best_model', True)

        # 设备配置
        self.device = torch.device(cfg['device']['type'] if torch.cuda.is_available() else 'cpu')


def set_seed(seed: int):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def create_model(config: TrainConfig):
    """创建模型"""
    if config.model_name == "unetpp_lightweight":
        model = LightweightNestedUNet(
            num_classes=config.num_classes,
            encoder=config.encoder,
            pretrained_encoder=config.pretrained_encoder,
            deep_supervision=config.deep_supervision
        )
    else:
        model = NestedUNet(
            num_classes=config.num_classes,
            deep_supervision=config.deep_supervision,
            pretrained_encoder=False
        )

    print(f"Model: {config.model_name}")
    if config.model_name == "unetpp_lightweight":
        print(f"Encoder: {config.encoder}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model


def create_optimizer(model, config: TrainConfig):
    """创建优化器"""
    if config.optimizer == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    elif config.optimizer == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    else:  # sgd
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=config.weight_decay
        )

    return optimizer


def create_scheduler(optimizer, config: TrainConfig):
    """创建学习率调度器"""
    if config.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.num_epochs,
            eta_min=config.min_lr
        )
    elif config.scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=30,
            gamma=0.1
        )
    else:  # plateau
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=10
        )

    return scheduler


def create_weighted_sampler(dataset, defect_classes, defect_boost):
    """创建加权采样器"""
    weights = []
    for i in range(len(dataset)):
        _, mask = dataset[i]
        uniq = set(torch.unique(mask).tolist())
        has_defect = len(uniq.intersection(defect_classes)) > 0
        weights.append(defect_boost if has_defect else 1.0)

    n_defect = sum(1 for w in weights if w > 1.0)
    print(f"Defect samples: {n_defect}/{len(weights)} ({n_defect/len(weights)*100:.1f}%)")

    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return sampler


def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    scaler=None,
    use_amp=False,
    grad_clip=0.0,
    deep_supervision=False
):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (images, masks) in enumerate(train_loader):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad(set_to_none=True)

        # 前向传播
        if use_amp and scaler is not None and device.type == "cuda":
            with autocast():
                outputs = model(images)

                # 计算损失
                if isinstance(outputs, list):
                    if deep_supervision:
                        weights = torch.linspace(1.0, 2.0, steps=len(outputs), device=device)
                        weights = weights / weights.sum()
                        loss = 0.0
                        for w, out in zip(weights, outputs):
                            l, _, _ = criterion(out, masks)
                            loss = loss + w * l
                    else:
                        loss, _, _ = criterion(outputs[-1], masks)
                else:
                    loss, _, _ = criterion(outputs, masks)

            # 反向传播
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)

            if isinstance(outputs, list):
                if deep_supervision:
                    weights = torch.linspace(1.0, 2.0, steps=len(outputs), device=device)
                    weights = weights / weights.sum()
                    loss = 0.0
                    for w, out in zip(weights, outputs):
                        l, _, _ = criterion(out, masks)
                        loss = loss + w * l
                else:
                    loss, _, _ = criterion(outputs[-1], masks)
            else:
                loss, _, _ = criterion(outputs, masks)

            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        loss_item = loss.detach().item()
        total_loss += loss_item
        num_batches += 1

        if (batch_idx + 1) % max(1, len(train_loader) // 5) == 0:
            print(f"  [{batch_idx+1}/{len(train_loader)}] Loss: {loss_item:.4f}")

    return total_loss / num_batches if num_batches > 0 else 0.0


@torch.no_grad()
def validate(model, val_loader, device, num_classes, deep_supervision=False):
    """验证"""
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


def main():
    parser = argparse.ArgumentParser(description="训练电缆胶带缺陷检测模型")
    parser.add_argument("--config", type=str, default="configs/train_inspection.yaml",
                        help="配置文件路径")
    parser.add_argument("--resume", type=str, default=None,
                        help="恢复训练的checkpoint路径")

    args = parser.parse_args()

    # 加载配置
    config = TrainConfig(args.config)

    # 设置随机种子
    set_seed(config.seed)

    print("=" * 70)
    print("电缆胶带缠绕缺陷检测 - 模型训练")
    print("=" * 70)
    print(f"Device: {config.device}")
    print(f"Model: {config.model_name}")
    if config.model_name == "unetpp_lightweight":
        print(f"Encoder: {config.encoder}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"AMP: {config.use_amp}")
    print(f"Weighted Sampler: {config.use_weighted_sampler} (boost={config.defect_boost}x)")
    print("=" * 70)

    # 创建checkpoint目录
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # 加载数据
    print("\n[1] 加载数据集...")
    train_dataset = CableDefectDataset(
        config.train_img_dir,
        config.train_mask_dir,
        augment=config.augmentation['enabled'],
        target_size=config.target_size
    )
    val_dataset = CableDefectDataset(
        config.val_img_dir,
        config.val_mask_dir,
        augment=False,
        target_size=config.target_size
    )

    # 创建数据加载器
    if config.use_weighted_sampler:
        print("  [1.1] 计算样本权重...")
        defect_classes = [3, 4, 5, 6]  # 缺陷类别
        sampler = create_weighted_sampler(
            train_dataset, defect_classes, config.defect_boost
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=config.num_workers,
            drop_last=False
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            drop_last=False
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers
    )

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")

    # 创建模型
    print("\n[2] 创建模型...")
    model = create_model(config)
    model = model.to(config.device)

    # 创建损失、优化器、调度器
    criterion = CombinedLoss(weight_ce=1.0, weight_dice=1.0)
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)

    print(f"  Optimizer: {config.optimizer}")
    print(f"  Scheduler: {config.scheduler}")
    print(f"  Loss: CombinedLoss (CE + Dice)")

    # AMP
    scaler = GradScaler(enabled=(config.use_amp and config.device.type == "cuda"))

    # 恢复训练
    start_epoch = 1
    best_mIoU = 0.0

    if args.resume and os.path.exists(args.resume):
        print(f"\n[2.1] 加载checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=config.device)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
            if "optimizer" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if "scheduler" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler"])
            if "epoch" in checkpoint:
                start_epoch = checkpoint["epoch"] + 1
            if "best_mIoU" in checkpoint:
                best_mIoU = checkpoint["best_mIoU"]
        else:
            model.load_state_dict(checkpoint)
        print(f"  恢复从 epoch {start_epoch}")

    # 训练循环
    print(f"\n[3] 开始训练...")
    print("-" * 70)

    for epoch in range(start_epoch, config.num_epochs + 1):
        # 训练
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, config.device,
            scaler=scaler, use_amp=config.use_amp, grad_clip=config.grad_clip,
            deep_supervision=config.deep_supervision
        )

        # 验证
        val_mIoU, precision, recall, iou_dict = validate(
            model, val_loader, config.device, config.num_classes,
            deep_supervision=config.deep_supervision
        )

        # 更新学习率
        current_lr = optimizer.param_groups[0]['lr']
        if config.scheduler != "plateau":
            scheduler.step()
        else:
            scheduler.step(val_mIoU)

        # 打印
        print(f"Epoch {epoch}/{config.num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val mIoU: {val_mIoU:.4f}")
        print(f"  LR: {current_lr:.6f}")

        # 保存最佳模型
        if val_mIoU > best_mIoU:
            best_mIoU = val_mIoU
            best_path = os.path.join(config.checkpoint_dir, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
                "best_mIoU": best_mIoU,
                "config": {
                    "num_classes": config.num_classes,
                    "encoder": config.encoder if config.model_name == "unetpp_lightweight" else None,
                    "target_size": config.target_size
                }
            }, best_path)
            print(f"  *** 最佳模型已保存 (mIoU={best_mIoU:.4f}) ***")

        # 定期保存
        if epoch % config.save_every_n_epochs == 0:
            save_path = os.path.join(config.checkpoint_dir, f"model_epoch_{epoch}.pth")
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
                "best_mIoU": best_mIoU,
                "config": {
                    "num_classes": config.num_classes,
                    "encoder": config.encoder if config.model_name == "unetpp_lightweight" else None,
                    "target_size": config.target_size
                }
            }, save_path)
            print(f"  Checkpoint已保存: {save_path}")

        print()

    # 保存最后模型
    print(f"[4] 训练完成!")
    print(f"  Best mIoU: {best_mIoU:.4f} (Epoch {epoch})")

    last_path = os.path.join(config.checkpoint_dir, "last_model.pth")
    torch.save({
        "epoch": config.num_epochs,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
        "best_mIoU": best_mIoU,
        "config": {
            "num_classes": config.num_classes,
            "encoder": config.encoder if config.model_name == "unetpp_lightweight" else None,
            "target_size": config.target_size
        }
    }, last_path)
    print(f"  最后模型已保存: {last_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
