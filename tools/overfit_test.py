"""10张过拟合测试 - 快速验证数据格式和模型结构

目标：在10张图片上训练到训练集 mIoU > 0.98

如果不能达到：
→ 数据/标注有问题（而不是模型/超参问题）

如果能轻松达到：
→ 数据格式没问题，不稳定来自：数据量小/增强过强/验证集小
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
from pathlib import Path
import numpy as np
import random

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.unetpp import NestedUNet
from models.losses import CombinedLoss
from data.dataset import CableDefectDataset
from utils.metrics import compute_metrics


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def overfit_test(
    num_samples=10,
    num_epochs=200,
    learning_rate=1e-3,
    device="cuda"
):
    """10张过拟合测试"""

    print("="*70)
    print(f"过拟合测试 - {num_samples} 张样本")
    print("="*70)

    set_seed(42)

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 加载数据集
    print("\n[1] 加载数据集...")
    full_dataset = CableDefectDataset(
        "dataset/processed_v2/train/images",
        "dataset/processed_v2/train/masks",
        augment=False,  # 关闭所有增强
        target_size=(256, 256)
    )

    # 随机选择 num_samples 张
    indices = random.sample(range(len(full_dataset)), min(num_samples, len(full_dataset)))
    print(f"  选择样本: {indices}")

    # 创建子数据集
    from torch.utils.data import Subset
    test_dataset = Subset(full_dataset, indices)

    test_loader = DataLoader(
        test_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0
    )

    print(f"  测试样本数: {len(test_dataset)}")

    # 模型
    print("\n[2] 构建模型...")
    model = NestedUNet(
        num_classes=6,
        deep_supervision=True,
        pretrained_encoder=False
    ).to(device)

    # 使用简单的类别权重
    class_weights = torch.tensor([1.0, 1.0, 1.0, 2.0, 2.0, 2.0]).to(device)
    criterion = CombinedLoss(
        weight_ce=1.0,
        weight_dice=1.0,
        class_weights=class_weights,
        dice_ignore_bg=True,
        dice_skip_empty=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 训练
    print("\n[3] 开始过拟合训练...")
    print("-"*70)

    best_loss = float('inf')
    best_miou = 0.0

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0

        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            if isinstance(outputs, list):
                outputs = outputs[-1]

            loss, _, _ = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(test_loader)

        # 验证（在同一个测试集上）
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for images, masks in test_loader:
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
        miou, _, _, iou_dict = compute_metrics(all_preds, all_targets, 6)

        if miou > best_miou:
            best_miou = miou
        if avg_loss < best_loss:
            best_loss = avg_loss

        # 每10轮打印一次
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{num_epochs}: Loss={avg_loss:.4f}, mIoU={miou:.4f}, "
                  f"Best: mIoU={best_miou:.4f}")

        # 提前退出条件
        if miou >= 0.98:
            print(f"\n✓ 达到目标 mIoU >= 0.98 (Epoch {epoch})")
            break

        if avg_loss < 0.01:
            print(f"\n✓ Loss 已收敛至 {avg_loss:.4f} (Epoch {epoch})")
            break

    # 结果判断
    print("\n" + "="*70)
    print("测试结果")
    print("="*70)
    print(f"最终 Loss: {avg_loss:.4f}")
    print(f"最终 mIoU: {miou:.4f}")
    print(f"最佳 mIoU: {best_miou:.4f}")
    print(f"各类别 IoU:")
    for cls, iou_val in iou_dict.items():
        print(f"  类别 {cls}: {iou_val:.4f}")

    print("\n" + "-"*70)

    if best_miou >= 0.95:
        print("✓ PASS: 数据格式和模型结构基本正常")
        print("  不稳定原因可能是：")
        print("  - 验证集太小导致 mIoU 波动")
        print("  - 数据增强过强")
        print("  - 学习率调度问题")
        print("  - 少量异常样本")
    elif best_miou >= 0.80:
        print("⚠ PARTIAL: 可以过拟合但不够完美")
        print("  可能原因：")
        print("  - 标注质量问题（部分样本标注不准）")
        print("  - 类别不平衡严重")
        print("  - 模型容量不足")
    else:
        print("✗ FAIL: 无法过拟合，数据/标注有严重问题")
        print("  请检查：")
        print("  - Labelme JSON 格式是否正确")
        print("  - 类别映射是否匹配")
        print("  - Mask 生成是否正确")
        print("  - 是否有坏样本/异常标注")
        print("\n建议运行数据审计脚本: python tools/audit_dataset.py")

    print("="*70)

    return best_miou


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=10, help="测试样本数")
    parser.add_argument("--num_epochs", type=int, default=200, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="学习率")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    args = parser.parse_args()

    overfit_test(
        num_samples=args.num_samples,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        device=args.device
    )
