"""
重新映射 mask 值：将类别 3 映射到类别 2
"""
import cv2
import numpy as np
from pathlib import Path

def remap_masks(dataset_dir):
    dataset_dir = Path(dataset_dir)

    for split in ['train', 'val', 'test']:
        mask_dir = dataset_dir / split / 'masks'
        if not mask_dir.exists():
            continue

        mask_files = list(mask_dir.glob('*.png'))
        print(f"\n{split}: 处理 {len(mask_files)} 个 mask 文件...")

        for mask_file in mask_files:
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)

            # 重新映射：3 -> 2
            mask[mask == 3] = 2

            # 保存
            cv2.imwrite(str(mask_file), mask)

        print(f"  完成！")

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python remap_masks.py <dataset_dir>")
        sys.exit(1)

    dataset_dir = sys.argv[1]
    print(f"重新映射数据集: {dataset_dir}")
    remap_masks(dataset_dir)
    print("\n所有 mask 已重新映射完成！")
