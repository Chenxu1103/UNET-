"""
模型预测结果可视化工具

功能：
1. 加载训练好的模型
2. 对验证集进行预测
3. 对比显示：原图、真实标注、预测结果、叠加对比
4. 保存结果图
"""
import torch
import sys
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.unetpp import NestedUNet
from data.dataset import CableDefectDataset
from torch.utils.data import DataLoader


def remap_mask_to_3class(mask):
    """将7类mask重新映射为3类"""
    mapping = {
        0: 0, 1: 1, 2: 2, 3: 0, 4: 0, 5: 0, 6: 0
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


def mask_to_color(mask, class_names=['背景', '电缆', '胶带']):
    """
    将mask转换为彩色可视化图像

    Args:
        mask: numpy array (H, W), 值为0,1,2
        class_names: 类别名称列表

    Returns:
        彩色RGB图像
    """
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    # 定义颜色 (BGR格式)
    colors = {
        0: [0, 0, 0],       # 背景: 黑色
        1: [255, 0, 0],     # 电缆: 蓝色
        2: [0, 255, 0]      # 胶带: 绿色
    }

    for cls_id, color in colors.items():
        color_mask[mask == cls_id] = color

    return color_mask


def overlay_mask(image, mask, alpha=0.5):
    """
    将mask叠加到原图上

    Args:
        image: 原图 RGB (H, W, 3), 值域0-255
        mask: mask彩色图像 (H, W, 3), BGR格式
        alpha: 叠加透明度

    Returns:
        叠加后的图像
    """
    # 将原图从RGB转BGR
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 叠加
    overlay = cv2.addWeighted(image_bgr, 1-alpha, mask, alpha, 0)

    # 转回RGB
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    return overlay


def visualize_sample(model, image, mask_gt, device, save_path=None):
    """
    可视化单个样本的预测结果

    显示4个子图：
    1. 原图
    2. 真实标注 (GT)
    3. 预测结果
    4. 预测叠加图
    """
    model.eval()

    # 预测
    with torch.no_grad():
        image_tensor = image.unsqueeze(0).to(device)
        output = model(image_tensor)
        if isinstance(output, list):
            output = output[-1]
        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    # 转换为numpy
    if torch.is_tensor(image):
        image_np = image.cpu().numpy().transpose(1, 2, 0)
        image_np = (image_np * 255).astype(np.uint8)
    else:
        image_np = image

    if torch.is_tensor(mask_gt):
        mask_gt_np = mask_gt.cpu().numpy()
    else:
        mask_gt_np = mask_gt

    # 生成彩色mask
    mask_gt_color = mask_to_color(mask_gt_np)
    mask_pred_color = mask_to_color(pred)

    # 生成叠加图
    overlay_gt = overlay_mask(image_np, mask_gt_color, alpha=0.4)
    overlay_pred = overlay_mask(image_np, mask_pred_color, alpha=0.4)

    # 创建图形
    fig = plt.figure(figsize=(16, 4))
    gs = gridspec.GridSpec(1, 4, figure=fig)

    # 1. 原图
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image_np)
    ax1.set_title("原图", fontsize=14, fontproperties='SimHei')
    ax1.axis('off')

    # 2. 真实标注
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(cv2.cvtColor(mask_gt_color, cv2.COLOR_BGR2RGB))
    ax2.set_title("真实标注 (GT)", fontsize=14, fontproperties='SimHei')
    ax2.axis('off')

    # 添加图例（和mask颜色一致）
    legend_elements = [
        Patch(facecolor='black', edgecolor='white', label='背景'),
        Patch(facecolor='blue', edgecolor='white', label='电缆'),  # 蓝色=电缆
        Patch(facecolor='lime', edgecolor='white', label='胶带')  # 绿色=胶带
    ]
    ax2.legend(handles=legend_elements, loc='upper right',
               fontsize=10, prop={'family': 'SimHei'})

    # 3. 预测结果
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(cv2.cvtColor(mask_pred_color, cv2.COLOR_BGR2RGB))
    ax3.set_title("预测结果", fontsize=14, fontproperties='SimHei')
    ax3.axis('off')

    # 4. 预测叠加图
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(overlay_pred)
    ax4.set_title("预测叠加", fontsize=14, fontproperties='SimHei')
    ax4.axis('off')

    plt.tight_layout()

    # 保存
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  保存: {save_path}")

    plt.close()


def compute_iou(pred, gt, num_classes=3):
    """计算IoU"""
    ious = []
    for cls in range(num_classes):
        intersection = np.sum((pred == cls) & (gt == cls))
        union = np.sum((pred == cls) | (gt == cls))
        if union == 0:
            iou = 0.0
        else:
            iou = intersection / union
        ious.append(iou)
    return ious


def main():
    print("="*70)
    print("模型预测结果可视化")
    print("="*70)
    print()

    # 配置
    checkpoint_path = "checkpoints_3class_finetuned/best_model.pth"
    output_dir = Path("visualization_results_verified")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print()

    # 加载数据
    print("[1] 加载数据集...")
    val_dataset = CableDefectDataset3Class(
        "dataset/processed/val/images",
        "dataset/processed/val/masks",
        augment=False,
        target_size=(512, 512)
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f"  验证样本: {len(val_dataset)}")
    print()

    # 加载模型
    print("[2] 加载模型...")
    model = NestedUNet(
        num_classes=3,
        deep_supervision=True,
        pretrained_encoder=False
    ).to(device)

    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        print(f"  模型: {checkpoint_path}")
        print(f"  最佳 mIoU: {checkpoint['best_miou']:.4f}")
        print(f"  Epoch: {checkpoint['epoch'] + 1}")
    else:
        print(f"  [ERROR] 未找到模型: {checkpoint_path}")
        return
    print()

    # 可视化所有样本
    print("[3] 生成可视化结果...")
    print()

    all_ious = {'background': [], 'cable': [], 'tape': []}
    num_samples = len(val_dataset)

    for idx in range(num_samples):
        image, mask_gt = val_dataset[idx]

        # 生成可视化
        save_path = output_dir / f"result_{idx:03d}.png"
        visualize_sample(model, image, mask_gt, device, save_path=str(save_path))

        # 计算IoU
        model.eval()
        with torch.no_grad():
            image_tensor = image.unsqueeze(0).to(device)
            output = model(image_tensor)
            if isinstance(output, list):
                output = output[-1]
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

        mask_gt_np = mask_gt.cpu().numpy() if torch.is_tensor(mask_gt) else mask_gt
        ious = compute_iou(pred, mask_gt_np, num_classes=3)

        all_ious['background'].append(ious[0])
        all_ious['cable'].append(ious[1])
        all_ious['tape'].append(ious[2])

        # 打印进度
        if (idx + 1) % 5 == 0 or idx == num_samples - 1:
            print(f"  已处理: {idx+1}/{num_samples}")
            print(f"    背景 IoU: {ious[0]:.2%}")
            print(f"    电缆 IoU: {ious[1]:.2%}")
            print(f"    胶带 IoU: {ious[2]:.2%}")
            print()

    # 统计
    print("="*70)
    print("统计结果")
    print("="*70)
    print(f"背景平均 IoU: {np.mean(all_ious['background']):.4f} ({np.mean(all_ious['background'])*100:.2f}%)")
    print(f"电缆平均 IoU: {np.mean(all_ious['cable']):.4f} ({np.mean(all_ious['cable'])*100:.2f}%)")
    print(f"胶带平均 IoU: {np.mean(all_ious['tape']):.4f} ({np.mean(all_ious['tape'])*100:.2f}%)")
    print()
    print(f"可视化结果已保存到: {output_dir.absolute()}")
    print("="*70)


if __name__ == '__main__':
    main()
