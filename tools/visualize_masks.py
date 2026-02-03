"""将灰度 mask 可视化为彩色图像"""
import cv2
import numpy as np
import os
import glob

# 类别颜色映射 (BGR格式)
CLASS_COLORS = {
    0: [0, 0, 0],        # background - 黑色
    1: [255, 0, 0],      # cable - 蓝色 (BGR)
    2: [0, 255, 0],      # tape - 绿色
    3: [0, 0, 255],      # bulge_defect - 红色
    4: [0, 255, 255],    # loose_defect - 黄色
    5: [255, 0, 255],    # burr_defect - 紫色
    6: [255, 255, 0],    # thin_defect - 青色
}

CLASS_NAMES = {
    0: "background",
    1: "cable",
    2: "tape",
    3: "bulge_defect",
    4: "loose_defect",
    5: "burr_defect",
    6: "thin_defect",
}

def mask_to_color(mask):
    """将灰度 mask 转换为彩色图像"""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in CLASS_COLORS.items():
        color_mask[mask == class_id] = color

    return color_mask

def visualize_masks(mask_dir, output_dir, overlay_dir=None):
    """批量可视化 mask

    Args:
        mask_dir: mask 目录
        output_dir: 彩色 mask 保存目录
        overlay_dir: 可选，叠加图保存目录
    """
    os.makedirs(output_dir, exist_ok=True)
    if overlay_dir:
        os.makedirs(overlay_dir, exist_ok=True)

    mask_files = glob.glob(os.path.join(mask_dir, "*.png"))

    print(f"Found {len(mask_files)} mask files")

    for mask_path in mask_files:
        # 读取 mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # 转换为彩色
        color_mask = mask_to_color(mask)

        # 保存彩色 mask
        filename = os.path.basename(mask_path)
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, color_mask)

        # 创建叠加图（如果有对应图像）
        if overlay_dir:
            # 查找对应图像
            base_name = os.path.splitext(filename)[0]
            img_path = os.path.join(mask_path.replace("masks", "images"), base_name + ".jpg")

            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                img = cv2.resize(img, (mask.shape[1], mask.shape[0]))

                # 半透明叠加
                overlay = cv2.addWeighted(img, 0.6, color_mask, 0.4, 0)
                overlay_path = os.path.join(overlay_dir, filename)
                cv2.imwrite(overlay_path, overlay)

        print(f"Processed: {filename}")

    print(f"\nColor masks saved to: {output_dir}")
    if overlay_dir:
        print(f"Overlay images saved to: {overlay_dir}")

if __name__ == "__main__":
    # 可视化验证集 masks
    visualize_masks(
        mask_dir="dataset/processed/val/masks",
        output_dir="results/val_masks_color",
        overlay_dir="results/val_masks_overlay"
    )

    print("\n" + "="*50)
    print("Class Legend:")
    for class_id, (name, color) in enumerate(zip(CLASS_NAMES.values(), CLASS_COLORS.values())):
        print(f"  {class_id}: {name} - BGR{tuple(color)}")
