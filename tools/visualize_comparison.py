"""生成原图、灰度mask、彩色mask的对比图"""
import cv2
import numpy as np
import os
import glob

# 类别颜色映射 (BGR格式)
CLASS_COLORS = {
    0: [0, 0, 0],        # background - 黑色
    1: [255, 0, 0],      # cable - 蓝色
    2: [0, 255, 0],      # tape - 绿色
    3: [0, 0, 255],      # bulge_defect - 红色
    4: [0, 255, 255],    # loose_defect - 黄色
    5: [255, 0, 255],    # burr_defect - 紫色
    6: [255, 255, 0],    # thin_defect - 青色
}

def mask_to_color(mask):
    """将灰度 mask 转换为彩色图像"""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        color_mask[mask == class_id] = color
    return color_mask

def create_comparison(image_dir, mask_dir, output_dir):
    """创建对比图

    Args:
        image_dir: 原始图像目录
        mask_dir: mask 目录
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    mask_files = glob.glob(os.path.join(mask_dir, "*.png"))
    print(f"Found {len(mask_files)} mask files")

    for mask_path in mask_files:
        filename = os.path.basename(mask_path)
        base_name = os.path.splitext(filename)[0]

        # 查找对应图像
        img_path = os.path.join(image_dir, base_name + ".jpg")
        if not os.path.exists(img_path):
            print(f"Image not found: {base_name}")
            continue

        # 读取图像和 mask
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # 调整大小
        h, w = 256, 256
        img = cv2.resize(img, (w, h))
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # 生成彩色 mask
        color_mask = mask_to_color(mask)

        # 灰度 mask 转 RGB 显示
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # 创建对比图：原图 | 灰度mask | 彩色mask | 叠加图
        # 添加标题
        padding = 30
        row1 = np.ones((h, w, 3), dtype=np.uint8) * 255

        # 第1行：原图
        row2 = img

        # 第2行：灰度 mask
        row3 = mask_rgb

        # 第3行：彩色 mask
        row4 = color_mask

        # 第4行：叠加图
        overlay = cv2.addWeighted(img, 0.6, color_mask, 0.4, 0)
        row5 = overlay

        # 拼接
        comparison = np.vstack([row1, row2, row3, row4, row5])

        # 添加文字标签
        cv2.putText(comparison, "Original Image", (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        cv2.putText(comparison, "Gray Mask", (10, h + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        cv2.putText(comparison, "Color Mask", (10, 2*h + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        cv2.putText(comparison, "Overlay", (10, 3*h + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

        # 保存
        output_path = os.path.join(output_dir, base_name + "_comparison.png")
        cv2.imwrite(output_path, comparison)

        print(f"Created: {base_name}_comparison.png")

    print(f"\nComparison images saved to: {output_dir}")

if __name__ == "__main__":
    create_comparison(
        image_dir="dataset/processed/val/images",
        mask_dir="dataset/processed/val/masks",
        output_dir="results/val_comparison"
    )

    print("\n" + "="*50)
    print("Mask 说明:")
    print("  灰度 mask: 每个像素值 0-6 代表不同类别")
    print("  彩色 mask: 将类别 ID 映射为不同颜色")
    print("  叠加图: 原图与彩色 mask 的半透明叠加")
