"""增强版训练脚本 - 包含详细的训练监控和诊断

关键改进：
1. 逐样本 loss 追踪（定位坏样本）
2. 每个 epoch 记录 top-K 高 loss 样本
3. 学习率曲线记录
4. 梯度裁剪和梯度监控
5. 验证集固定（无随机增强）
6. 异常检测（loss 突增预警）
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
import json
from datetime import datetime
from collections import defaultdict
from torch.cuda.amp import autocast, GradScaler

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.unetpp import NestedUNet
from models.losses import CombinedLoss
from data.dataset import CableDefectDataset
from utils.metrics import compute_metrics


class TrainingMonitor:
    """训练监控器"""

    def __init__(self, output_dir: str, top_k: int = 10):
        self.output_dir = Path(output_dir)
        self.top_k = top_k
        self.history = {
            "epoch": [],
            "train_loss": [],
            "val_miou": [],
            "learning_rate": [],
            "high_loss_samples": [],
            "gradient_norm": [],
        }
        self.epoch_high_losses = []

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_miou: float,
        lr: float,
        grad_norm: float,
        high_loss_samples: list = None
    ):
        """记录一个 epoch 的数据"""
        self.history["epoch"].append(epoch)
        self.history["train_loss"].append(train_loss)
        self.history["val_miou"].append(val_miou)
        self.history["learning_rate"].append(lr)
        self.history["gradient_norm"].append(grad_norm)

        if high_loss_samples:
            self.history["high_loss_samples"].append({
                "epoch": epoch,
                "samples": high_loss_samples
            })

        # 检测异常：loss 突增
        if len(self.history["train_loss"]) >= 3:
            recent_losses = self.history["train_loss"][-3:]
            avg_before = np.mean(recent_losses[:-1])
            current = recent_losses[-1]
            if current > avg_before * 1.5:  # 突增 50% 以上
                print(f"  ⚠️  [警告] Loss 突增检测: {avg_before:.4f} -> {current:.4f}")

    def save_history(self):
        """保存训练历史"""
        history_path = self.output_dir / "training_history.json"
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2)
        print(f"  训练历史已保存: {history_path}")

    def save_high_loss_report(self):
        """保存高 loss 样本报告"""
        high_loss_path = self.output_dir / "high_loss_samples.json"
        with open(high_loss_path, 'w', encoding='utf-8') as f:
            json.dump(self.history["high_loss_samples"], f, indent=2)


class SampleLossTracker:
    """逐样本 loss 追踪器"""

    def __init__(self, dataset, top_k: int = 20):
        self.dataset = dataset
        self.top_k = top_k
        self.sample_losses = []  # list of (loss, idx, filename)

    def update(self, losses: torch.Tensor, indices: torch.Tensor):
        """更新 loss 记录"""
        for loss_val, idx in zip(losses.tolist(), indices.tolist()):
            filename = os.path.basename(self.dataset.image_files[idx])
            self.sample_losses.append((loss_val, idx, filename))

    def get_top_k(self) -> list:
        """获取 top-K 高 loss 样本"""
        self.sample_losses.sort(key=lambda x: x[0], reverse=True)
        return self.sample_losses[:self.top_k]

    def reset(self):
        """重置"""
        self.sample_losses = []


def train_one_epoch_with_tracking(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    dataset: CableDefectDataset,
    scaler: GradScaler = None,
    use_amp: bool = False,
    grad_clip: float = 1.0
) -> tuple:
    """带追踪的训练函数"""
    model.train()
    train_loss = 0.0
    num_batches = 0
    total_grad_norm = 0.0

    # loss 追踪器
    loss_tracker = SampleLossTracker(dataset, top_k=20)

    for batch_idx, (images, masks, indices) in enumerate(train_loader):
        images = images.to(device)
        masks = masks.to(device)
        indices = indices.cpu()  # 保持在 CPU

        optimizer.zero_grad(set_to_none=True)

        # 前向传播
        if use_amp and scaler is not None and device.type == "cuda":
            with autocast():
                outputs = model(images)
                if isinstance(outputs, list):
                    weights = torch.linspace(1.0, 2.0, steps=len(outputs), device=device)
                    weights = weights / weights.sum()
                    total_loss = 0.0
                    batch_losses = []
                    for i, (out, w) in enumerate(zip(outputs, weights)):
                        loss, _, _ = criterion(out, masks)
                        total_loss = total_loss + w * loss
                        batch_losses.append(loss)
                else:
                    total_loss, _, _ = criterion(outputs, masks)
                    batch_losses = [total_loss]

            # 记录每个样本的 loss（简化：使用 batch 平均）
            per_sample_losses = torch.tensor([loss.item() for loss in batch_losses])
            loss_tracker.update(per_sample_losses, indices)

            scaler.scale(total_loss).backward()

            # 梯度裁剪前计算梯度范数
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                total_grad_norm += grad_norm.item()
                scaler.step(optimizer)
                scaler.update()
            else:
                scaler.step(optimizer)
                scaler.update()
        else:
            outputs = model(images)
            if isinstance(outputs, list):
                weights = torch.linspace(1.0, 2.0, steps=len(outputs), device=device)
                weights = weights / weights.sum()
                total_loss = 0.0
                batch_losses = []
                for out, w in zip(outputs, weights):
                    loss, _, _ = criterion(out, masks)
                    total_loss = total_loss + w * loss
                    batch_losses.append(loss)
            else:
                total_loss, _, _ = criterion(outputs, masks)
                batch_losses = [total_loss]

            per_sample_losses = torch.tensor([loss.item() for loss in batch_losses])
            loss_tracker.update(per_sample_losses, indices)

            total_loss.backward()

            if grad_clip and grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                total_grad_norm += grad_norm.item()
            else:
                grad_norm = 0.0
                total_grad_norm += grad_norm

            optimizer.step()

        loss_item = total_loss.detach().item()
        train_loss += loss_item
        num_batches += 1

        if (batch_idx + 1) % max(1, len(train_loader) // 5) == 0:
            print(f"  [{batch_idx+1}/{len(train_loader)}] Loss: {loss_item:.4f}")

    avg_loss = train_loss / num_batches if num_batches > 0 else 0.0
    avg_grad_norm = total_grad_norm / num_batches if num_batches > 0 else 0.0

    # 获取高 loss 样本
    high_loss_samples = loss_tracker.get_top_k()

    return avg_loss, avg_grad_norm, high_loss_samples


class DataLoaderWithIndex(DataLoader):
    """返回数据索引的 DataLoader"""

    def __iter__(self):
        for batch_idx, (images, masks) in enumerate(super().__iter__()):
            # 获取当前 batch 的索引
            if hasattr(self.sampler, 'indices'):
                indices = self.sampler.indices[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size]
            elif hasattr(self.batch_sampler, '__iter__'):
                indices = list(self.batch_sampler)[batch_idx]
            else:
                # 顺序采样
                start_idx = batch_idx * self.batch_size
                end_idx = start_idx + images.size(0)
                indices = list(range(start_idx, end_idx))

            yield images, masks, torch.tensor(indices)


def train_with_monitoring(
    num_epochs: int = 200,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    grad_clip: float = 1.0,
    defect_boost: float = 3.0,
    resume: str = None,
    output_dir: str = "checkpoints_monitored"
):
    """带监控的训练主函数"""

    print("="*70)
    print("增强版训练 - 带监控和诊断")
    print("="*70)

    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

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
        augment=False,  # 验证集关闭增强
        target_size=(256, 256)
    )

    # 缺陷过采样
    weights = []
    defect_ids = [3, 4, 5]
    for i in range(len(train_dataset)):
        _, m = train_dataset[i]
        uniq = set(torch.unique(m).tolist())
        has_defect = len(uniq.intersection(defect_ids)) > 0
        weights.append(defect_boost if has_defect else 1.0)

    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    train_loader = DataLoaderWithIndex(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        drop_last=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,  # 验证集不 shuffle
        num_workers=0
    )

    print(f"  训练样本: {len(train_dataset)}")
    print(f"  验证样本: {len(val_dataset)}")

    # 模型
    print("\n[2] 构建模型...")
    model = NestedUNet(
        num_classes=6,
        deep_supervision=True,
        pretrained_encoder=False
    ).to(device)

    class_weights = torch.tensor([0.5, 1.0, 1.0, 15.0, 12.0, 15.0]).to(device)
    criterion = CombinedLoss(
        weight_ce=1.0,
        weight_dice=1.0,
        class_weights=class_weights,
        dice_ignore_bg=True,
        dice_skip_empty=True
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )

    scaler = GradScaler(enabled=(device.type == "cuda"))

    # 监控器
    monitor = TrainingMonitor(output_dir)

    # 从检查点恢复
    start_epoch = 1
    best_mIoU = 0.0

    if resume and os.path.exists(resume):
        print(f"\n从检查点恢复: {resume}")
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_mIoU = float(ckpt.get("best_mIoU", 0.0))

    # 训练循环
    print(f"\n[3] 开始训练 (从 epoch {start_epoch})...")
    print("-"*70)

    for epoch in range(start_epoch, num_epochs + 1):
        # 训练
        train_loss, grad_norm, high_loss_samples = train_one_epoch_with_tracking(
            model, train_loader, criterion, optimizer, device, epoch,
            dataset=train_dataset,
            scaler=scaler,
            use_amp=True,
            grad_clip=grad_clip
        )

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
        val_mIoU, _, _, iou_dict = compute_metrics(all_preds, all_targets, 6)

        # 学习率
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        # 打印
        print(f"Epoch {epoch}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val mIoU: {val_mIoU:.4f}")
        print(f"  LR: {current_lr:.6f}")
        print(f"  Grad Norm: {grad_norm:.4f}")

        # 高 loss 样本
        if high_loss_samples:
            print(f"  Top 5 高 Loss 样本:")
            for loss_val, idx, filename in high_loss_samples[:5]:
                print(f"    {filename}: {loss_val:.4f}")

        # 记录监控
        monitor.log_epoch(
            epoch, train_loss, val_mIoU, current_lr, grad_norm,
            [{"filename": f, "loss": l, "idx": i} for l, i, f in high_loss_samples[:10]]
        )

        # 保存最佳模型
        if val_mIoU > best_mIoU:
            best_mIoU = val_mIoU
            best_path = os.path.join(output_dir, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_mIoU": best_mIoU,
            }, best_path)
            print(f"  *** 最佳模型更新 (mIoU={best_mIoU:.4f}) ***")

        print()

    # 保存历史
    monitor.save_history()
    monitor.save_high_loss_report()

    print(f"\n训练完成!")
    print(f"最佳 mIoU: {best_mIoU:.4f}")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--defect_boost", type=float, default=3.0)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="checkpoints_monitored")
    args = parser.parse_args()

    train_with_monitoring(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        grad_clip=args.grad_clip,
        defect_boost=args.defect_boost,
        resume=args.resume,
        output_dir=args.output_dir
    )
