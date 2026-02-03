"""
评估训练结果
"""
import torch
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.unetpp import NestedUNet
from data.dataset import CableDefectDataset
from torch.utils.data import DataLoader
from utils.metrics import compute_metrics
from tqdm import tqdm


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
    """3类数据集"""

    def __getitem__(self, idx):
        img, mask = super().__getitem__(idx)
        mask = remap_mask_to_3class(mask)
        return img, mask


def evaluate_model(checkpoint_path, num_classes=3):
    """评估模型性能"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载数据集
    print("\n加载验证数据集...")
    val_dataset = CableDefectDataset3Class(
        "dataset/processed/val/images",
        "dataset/processed/val/masks",
        augment=False,
        target_size=(512, 512)
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f"验证样本数: {len(val_dataset)}")

    # 加载模型
    print(f"\n加载模型: {checkpoint_path}")
    model = NestedUNet(
        num_classes=num_classes,
        deep_supervision=True,
        pretrained_encoder=False
    ).to(device)

    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
        if 'epoch' in checkpoint:
            print(f"训练轮数: {checkpoint['epoch'] + 1}")
        if 'best_miou' in checkpoint:
            print(f"最佳 mIoU: {checkpoint['best_miou']:.4f}")
    else:
        model.load_state_dict(checkpoint)

    # 评估
    print("\n开始评估...")
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="推理"):
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

    # 计算指标
    miou, precision, recall, iou_dict = compute_metrics(
        all_preds, all_targets, num_classes
    )

    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    print(f"mIoU: {miou:.4f} ({miou*100:.2f}%)")
    if isinstance(precision, dict):
        print(f"Precision (平均): {np.mean(list(precision.values())):.4f}")
    else:
        print(f"Precision: {precision:.4f}")
    if isinstance(recall, dict):
        print(f"Recall (平均): {np.mean(list(recall.values())):.4f}")
    else:
        print(f"Recall: {recall:.4f}")
    print(f"\n各类别 IoU:")
    class_names = ['背景', '电缆', '胶带']
    for cls in range(num_classes):
        iou = iou_dict.get(cls, 0.0)
        print(f"  {class_names[cls]}: {iou:.4f} ({iou*100:.2f}%)")
    print("=" * 60)

    return miou, iou_dict


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                       default='checkpoints_3class_high_precision/best_model.pth',
                       help='模型权重路径')
    parser.add_argument('--num-classes', type=int, default=3,
                       help='类别数')
    args = parser.parse_args()

    evaluate_model(args.checkpoint, args.num_classes)
