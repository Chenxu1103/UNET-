"""Labelme 标注转换为 mask 图像，包含数据增强和数据集划分

将 Labelme JSON 标注文件转换为像素级 mask 图像，支持类别映射、
数据增强和训练/验证/测试集的随机划分。
"""
import os
import json
import random
import glob
from pathlib import Path
import numpy as np
import cv2
from typing import Dict, Tuple, List


# 定义类别名到类别ID的映射（新版本：去除鼓包，厚度不足改为包裹不均匀）
CLASS_MAP = {
    "background": 0,
    "cable": 1,           # 电缆
    "tape": 2,            # 胶带
    "burr_defect": 3,     # 毛刺缺陷
    "loose_defect": 4,    # 松脱缺陷
    "wrap_uneven": 5,     # 包裹不均匀缺陷（原thin_defect改名）
}

# 旧类别名到新类别名的映射（用于兼容旧标注）
CLASS_NAME_MAPPING = {
    "thin_defect": "wrap_uneven",      # 厚度不足 -> 包裹不均匀
    "bulge_defect": None,              # 鼓包缺陷 -> 移除（映射为背景或忽略）
    "damage_defect": None,             # 破损缺陷 -> 移除
}

# 反向映射：ID到类别名
CLASS_NAMES = {v: k for k, v in CLASS_MAP.items()}


def json_to_mask(json_file: str, save_mask: bool = False) -> np.ndarray:
    """将单个Labelme JSON标注转换为mask图（numpy数组），可选保存为图像文件
    
    Args:
        json_file: Labelme JSON 文件路径
        save_mask: 是否保存 mask 为 PNG 文件
        
    Returns:
        mask 数组，shape (H,W)，像素值为类别ID
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 根据原图尺寸创建空白mask
    img_h = data.get("imageHeight")
    img_w = data.get("imageWidth")
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    
    # 按照每个标注的shape绘制多边形
    for shape in data["shapes"]:
        label = shape["label"]

        # 处理旧类别名映射
        if label in CLASS_NAME_MAPPING:
            new_label = CLASS_NAME_MAPPING[label]
            if new_label is None:
                # 该类别已被移除，跳过不绘制
                continue
            label = new_label

        if label not in CLASS_MAP:
            continue  # 未知类别跳过

        cls_id = CLASS_MAP[label]
        points = np.array(shape["points"], dtype=np.int32)  # 多边形点

        # 填充多边形区域像素为类别ID
        cv2.fillPoly(mask, [points], color=cls_id)
    
    # 注意：如果存在缺陷区域与胶带重叠，应确保JSON标注中缺陷形状后绘制，
    # 上述fillPoly按顺序覆盖，因此后出现的shape会覆盖前面的区域。
    
    if save_mask:
        mask_path = os.path.splitext(json_file)[0] + "_mask.png"
        cv2.imwrite(mask_path, mask)
        print(f"Mask saved: {mask_path}")
    
    return mask


def prepare_dataset(
    labelme_dir: str,
    images_dir: str = None,
    output_dir: str = "dataset/processed",
    val_ratio: float = 0.1,
    test_ratio: float = 0.1
) -> Dict[str, List[Tuple[str, str]]]:
    """遍历labelme标注目录，生成mask标签并划分训练/验证/测试集

    Args:
        labelme_dir: Labelme 标注目录路径，包含 JSON 文件
        images_dir: 图像目录路径（如果为None，则在labelme_dir中查找）
        output_dir: 输出目录，用于保存划分后的数据集
        val_ratio: 验证集比例（默认10%）
        test_ratio: 测试集比例（默认10%）

    Returns:
        字典，包含 'train', 'val', 'test' 三个key，值为 (image_path, mask_path) 元组列表
    """
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 如果未指定图像目录，使用标注目录的兄弟images目录
    if images_dir is None:
        parent = os.path.dirname(labelme_dir)
        images_dir = os.path.join(parent, "images")

    # 收集所有图像和对应JSON路径
    json_files = glob.glob(os.path.join(labelme_dir, "*.json"))
    data_list = []

    for json_file in json_files:
        # 尝试匹配对应的图像文件
        basename = os.path.splitext(os.path.basename(json_file))[0]

        # 先在images_dir中查找
        found = False
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.PNG', '.JPEG']:
            img_path = os.path.join(images_dir, basename + ext)
            if os.path.exists(img_path):
                data_list.append((img_path, json_file))
                found = True
                break

        # 如果没找到，尝试在labelme_dir中查找
        if not found:
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.PNG', '.JPEG']:
                img_path = os.path.join(labelme_dir, basename + ext)
                if os.path.exists(img_path):
                    data_list.append((img_path, json_file))
                    break
    
    print(f"Found {len(data_list)} image-annotation pairs")
    
    # 划分数据集
    random.seed(42)  # 设置随机种子保证可重复性
    random.shuffle(data_list)
    
    total = len(data_list)
    val_count = int(total * val_ratio)
    test_count = int(total * test_ratio)
    train_count = total - val_count - test_count
    
    train_list = data_list[:train_count]
    val_list = data_list[train_count:train_count + val_count]
    test_list = data_list[train_count + val_count:]
    
    print(f"Train: {len(train_list)}, Val: {len(val_list)}, Test: {len(test_list)}")
    
    # 保存mask并记录路径
    splits = {"train": train_list, "val": val_list, "test": test_list}
    result = {}
    
    for split, file_list in splits.items():
        # 创建分割目录
        split_dir = os.path.join(output_dir, split)
        img_dir = os.path.join(split_dir, "images")
        mask_dir = os.path.join(split_dir, "masks")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        
        split_paths = []
        for img_path, json_path in file_list:
            # 转换JSON为mask
            mask = json_to_mask(json_path, save_mask=False)

            # 保存图像和mask
            fname = os.path.basename(img_path)
            output_img_path = os.path.join(img_dir, fname)

            # 复制原始图像
            img = cv2.imread(img_path)
            if img is None:
                print(f"  [Warning] Failed to read image: {img_path}, skipping...")
                continue
            cv2.imwrite(output_img_path, img)

            # 保存mask（PNG格式）
            mask_fname = os.path.splitext(fname)[0] + ".png"
            output_mask_path = os.path.join(mask_dir, mask_fname)
            cv2.imwrite(output_mask_path, mask)

            split_paths.append((output_img_path, output_mask_path))
        
        result[split] = split_paths
        print(f"Saved {split} set: {len(split_paths)} samples")
    
    return result


def apply_augmentation(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """对图像和mask应用基本数据增强
    
    Args:
        image: RGB 图像 (H,W,3)
        mask: mask 标签 (H,W)
        
    Returns:
        增强后的 (image, mask) 元组
    """
    # 随机水平翻转
    if random.random() < 0.5:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)
    
    # 随机垂直翻转
    if random.random() < 0.5:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)
    
    # 随机旋转（90度倍数）
    if random.random() < 0.3:
        k = random.randint(1, 3)  # 旋转次数（90度倍数）
        image = np.rot90(image, k=k)
        mask = np.rot90(mask, k=k)
    
    # 随机亮度调整
    if random.random() < 0.5:
        factor = 0.7 + random.random() * 0.6  # 0.7~1.3之间
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 2] *= factor
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    return image, mask


if __name__ == "__main__":
    # 示例：使用本脚本处理Labelme数据
    labelme_dir = "path/to/labelme/annotations"  # Labelme标注目录
    output_dir = "dataset"  # 输出数据集目录
    
    result = prepare_dataset(labelme_dir, output_dir, val_ratio=0.1, test_ratio=0.1)
    print("Dataset preparation completed!")
    print(f"Output directory: {output_dir}")
    print(f"Class mapping: {CLASS_MAP}")
