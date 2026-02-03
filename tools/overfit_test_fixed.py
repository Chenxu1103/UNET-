"""修复版过拟合测试 - 针对极度不平衡数据

关键修复：
1. 手动选择包含缺陷的样本
2. 使用更强的类别权重
3. 增加 Dice Loss 权重
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


def overfit_test_fixed(
    num_samples=10,
    num_epochs=200,
    learning_rate=1e-3,
    device="cuda"
):
    """修复版过拟合测试 - 强制选择含缺陷样本"""

    print("="*70)
    print(f"修复版过拟合测试 - 选择含缺陷样本")
    print("="*70)

    set_seed(42)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 加载数据集
    print("\n[1] 加载数据集...")
    full_dataset = CableDefectDataset(
        "dataset/processed_v2/train/images",
        "dataset/processed_v2/train/masks",
        augment=False,
        target_size=(256, 256)
    )

    # 找出所有包含缺陷的样本
    defect_indices = []
    for i in range(len(full_dataset)):
        _, mask = full_dataset[i]
        uniq = torch.unique(mask).tolist()
        if any(u in [3, 4, 5] for u in uniq):  # burr, loose, wrap_uneven
            defect_indices.append(i)

    print(f"  找到缺陷样本: {len(defect_indices)} 个")

    if len(defect_indices) < num_samples:
        print(f"  [警告] 缺陷样本不足 {num_samples} 个，补齐正常样本")
        normal_indices = [i for i in range(len(full_dataset)) if i not in defect_indices]
        selected_indices = defect_indices + normal_indices[:num_samples-len(defect_indices)]
    else:
        selected_indices = defect_indices[:num_samples]

    print(f"  选择样本: {selected_indices}")

    # 创建数据集
    from torch.utils.data import Subset
    test_dataset = Subset(full_dataset, selected_indices)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=True, num_workers=0)

    # 模型
    print("\n[2] 构建模型...")
    model = NestedUNet(
        num_classes=6,
        deep_supervision=True,
        pretrained_encoder=False
    ).to(device)

    # 使用更强的类别权重
    # 背景: 0.5, 正常: 1.0, 缺陷: 20.0
    class_weights = torch.tensor([0.5, 1.0, 1.0, 20.0, 20.0, 20.0]).to(device)

    # 增加 Dice Loss 权重，减少 CE 权重
    criterion = CombinedLoss(
        weight_ce=0.5,  # 降低 CE
        weight_dice=2.0,  # 增加 Dice
        class_weights=class_weights,
        dice_ignore_bg=True,
        dice_skip_empty=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  类别权重: {class_weights.tolist()}")
    print(f"  损失权重: CE=0.5, Dice=2.0")

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

            loss, ce_loss, dice_loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(test_loader)

        # 验证
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

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{num_epochs}: Loss={avg_loss:.4f}, mIoU={miou:.4f}, "
                  f"Best: mIoU={best_miou:.4f}")

        if miou >= 0.98:
            print(f"\n✓ 达到目标 mIoU >= 0.98 (Epoch {epoch})")
            break

        if avg_loss < 0.01:
            print(f"\n✓ Loss 已收敛至 {avg_loss:.4f} (Epoch {epoch})")
            break

    # 结果
    print("\n" + "="*70)
    print("测试结果")
    print("="*70)
    print(f"最终 Loss: {avg_loss:.4f}")
    print(f"最终 mIoU: {miou:.4f}")
    print(f"最佳 mIoU: {best_miou:.4f}")
    print(f"\n各类别 IoU:")
    for cls, iou_val in iou_dict.items():
        print(f"  类别 {cls}: {iou_val:.4f}")

    print("\n" + "-"*70)

    if best_miou >= 0.95:
        print("✓ PASS: 数据格式正常，使用强权重可以过拟合")
        print("\n建议:")
        print("  1. 在完整训练中使用强类别权重 [0.5, 1.0, 1.0, 20.0, 20.0, 20.0]")
        print("  2. 增加 Dice Loss 权重至 2.0")
        print("  3. 确保训练时缺陷样本被充分采样")
    elif best_miou >= 0.70:
        print("⚠ PARTIAL: 可以学习但不够好")
        print("\n建议:")
        print("  1. 进一步增加缺陷类别权重")
        print("  2. 检查是否有标注错误")
        print("  3. 考虑使用 Focal Loss")
    else:
        print("✗ FAIL: 仍然无法过拟合")
        print("\n可能的问题:")
        print("  1. 标注格式错误")
        print("  2. Mask 值不正确")
        print("  3. 模型结构问题")

    print("="*70)

    return best_miou


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    overfit_test_fixed(
        num_samples=args.num_samples,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        device=args.device
    )
