"""重新处理数据集：应用新的类别映射

去除鼓包缺陷，将厚度不足改为包裹不均匀
"""
import os
import sys
from pathlib import Path

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data.prepare_dataset import prepare_dataset, CLASS_MAP

def main():
    print("="*70)
    print("数据集重新处理")
    print("="*70)
    print(f"\n新的类别映射:")
    for name, idx in CLASS_MAP.items():
        print(f"  {idx}: {name}")

    # 路径配置
    base_dir = Path(__file__).parent.parent
    labelme_dir = base_dir / "dataset" / "raw" / "annotations"
    images_dir = base_dir / "dataset" / "raw" / "images"
    output_dir = base_dir / "dataset" / "processed_v2"

    print(f"\n输入配置:")
    print(f"  Labelme标注目录: {labelme_dir}")
    print(f"  图像目录: {images_dir}")
    print(f"  输出目录: {output_dir}")

    # 检查输入目录
    if not labelme_dir.exists():
        print(f"\n[错误] 标注目录不存在: {labelme_dir}")
        return

    if not images_dir.exists():
        print(f"\n[错误] 图像目录不存在: {images_dir}")
        return

    # 重新处理数据集
    print(f"\n开始处理...")
    print("-"*70)

    result = prepare_dataset(
        labelme_dir=str(labelme_dir),
        images_dir=str(images_dir),
        output_dir=str(output_dir),
        val_ratio=0.15,  # 验证集15%
        test_ratio=0.15  # 测试集15%
    )

    print(f"\n处理完成!")
    print(f"数据集统计:")
    for split, paths in result.items():
        print(f"  {split}: {len(paths)} 样本")

    print(f"\n输出目录: {output_dir}")
    print("="*70)

if __name__ == "__main__":
    main()
