"""更新数据集 - 处理分离的图像和标注目录

将 dataset/raw/images 和 dataset/raw/annotations 中的数据
转换为 dataset/processed 格式，并重新划分 train/val/test
"""
import os
import sys
import json
import glob
import random
import shutil
from pathlib import Path
import numpy as np
import cv2

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# 类别映射
CLASS_MAP = {
    "background": 0,
    "cable": 1,
    "tape": 2,
    "bulge_defect": 3,
    "loose_defect": 4,
    "burr_defect": 5,
    "thin_defect": 6
}


def json_to_mask(json_file: str) -> np.ndarray:
    """将 Labelme JSON 转换为 mask"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    img_h = data.get("imageHeight")
    img_w = data.get("imageWidth")
    mask = np.zeros((img_h, img_w), dtype=np.uint8)

    for shape in data["shapes"]:
        label = shape["label"]
        if label not in CLASS_MAP:
            continue

        cls_id = CLASS_MAP[label]
        points = np.array(shape["points"], dtype=np.int32)
        cv2.fillPoly(mask, [points], color=cls_id)

    return mask


def find_matching_image(json_file: str, images_dir: str) -> str:
    """为 JSON 文件查找匹配的图像"""
    basename = os.path.splitext(os.path.basename(json_file))[0]

    # 尝试各种扩展名
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.PNG']:
        img_path = os.path.join(images_dir, basename + ext)
        if os.path.exists(img_path):
            return img_path

    return None


def update_dataset(
    images_dir: str,
    annotations_dir: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42
):
    """更新数据集

    Args:
        images_dir: 原始图像目录
        annotations_dir: Labelme JSON 标注目录
        output_dir: 输出目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
    """
    random.seed(seed)

    print("="*70)
    print("更新数据集")
    print("="*70)

    # 创建输出目录
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'masks'), exist_ok=True)

    # 收集所有有效的标注-图像对
    print(f"\n[1] 扫描标注文件...")
    json_files = glob.glob(os.path.join(annotations_dir, "*.json"))
    print(f"  找到 {len(json_files)} 个标注文件")

    valid_pairs = []
    for json_file in json_files:
        img_path = find_matching_image(json_file, images_dir)
        if img_path:
            valid_pairs.append((img_path, json_file))

    print(f"  有效标注-图像对: {len(valid_pairs)}")

    if len(valid_pairs) == 0:
        print("  [错误] 没有找到有效的标注-图像对！")
        return

    # 打乱数据
    random.shuffle(valid_pairs)

    # 划分数据集
    print(f"\n[2] 划分数据集...")
    n_total = len(valid_pairs)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_pairs = valid_pairs[:n_train]
    val_pairs = valid_pairs[n_train:n_train + n_val]
    test_pairs = valid_pairs[n_train + n_val:]

    print(f"  训练集: {len(train_pairs)} 张")
    print(f"  验证集: {len(val_pairs)} 张")
    print(f"  测试集: {len(test_pairs)} 张")

    # 处理并复制文件
    def process_split(pairs, split_name):
        print(f"\n[3.{split_name}] 处理 {split_name} 集...")
        for idx, (img_path, json_path) in enumerate(pairs):
            # 读取并生成 mask
            mask = json_to_mask(json_path)

            # 文件名
            basename = os.path.splitext(os.path.basename(img_path))[0]

            # 复制图像
            dst_img = os.path.join(output_dir, split_name, 'images', basename + '.jpg')
            shutil.copy2(img_path, dst_img)

            # 保存 mask
            dst_mask = os.path.join(output_dir, split_name, 'masks', basename + '.png')
            cv2.imwrite(dst_mask, mask)

            if (idx + 1) % max(1, len(pairs) // 5) == 0 or idx == 0:
                print(f"  进度: {idx+1}/{len(pairs)}")

    # 处理各个数据集
    process_split(train_pairs, 'train')
    process_split(val_pairs, 'val')
    process_split(test_pairs, 'test')

    print(f"\n[4] 数据集更新完成！")
    print(f"  输出目录: {output_dir}")
    print("="*70)

    # 统计类别分布
    print("\n[5] 类别分布统计:")
    class_counts = {name: 0 for name in CLASS_MAP.keys()}

    for json_path, _ in [(j, i) for i, j in train_pairs + val_pairs + test_pairs]:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for shape in data["shapes"]:
            label = shape["label"]
            if label in class_counts:
                class_counts[label] += 1

    for cls_name, count in class_counts.items():
        print(f"  {cls_name}: {count}")


if __name__ == "__main__":
    update_dataset(
        images_dir="dataset/raw/images",
        annotations_dir="dataset/raw/annotations",
        output_dir="dataset/processed",
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        seed=42
    )
