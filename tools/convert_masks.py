"""转换现有mask到新的类别映射

旧映射 -> 新映射:
0 (background) -> 0
1 (cable) -> 1
2 (tape) -> 2
3 (burr_defect) -> 3
4 (bulge_defect) -> 移除(转为背景)
5 (loose_defect) -> 4
6 (thin_defect) -> 5 (wrap_uneven)
"""
import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image

# 旧类别ID -> 新类别ID的映射
CLASS_ID_MAPPING = {
    0: 0,  # background -> background
    1: 1,  # cable -> cable
    2: 2,  # tape -> tape
    3: 3,  # burr_defect -> burr_defect
    4: 0,  # bulge_defect -> background (移除)
    5: 4,  # loose_defect -> loose_defect
    6: 5,  # thin_defect -> wrap_uneven
}

def convert_mask(mask_array: np.ndarray) -> np.ndarray:
    """转换mask的类别ID"""
    converted = np.zeros_like(mask_array)
    for old_id, new_id in CLASS_ID_MAPPING.items():
        converted[mask_array == old_id] = new_id
    return converted

def main():
    print("="*70)
    print("Mask类别转换脚本")
    print("="*70)
    print("\n类别ID映射:")
    for old_id, new_id in CLASS_ID_MAPPING.items():
        if old_id == new_id:
            print(f"  {old_id} -> {new_id} (保持不变)")
        elif new_id == 0:
            print(f"  {old_id} -> {new_id} (移除该类别)")
        else:
            print(f"  {old_id} -> {new_id} (重新映射)")

    # 路径配置
    base_dir = Path(__file__).parent.parent
    input_dir = base_dir / "dataset" / "processed"
    output_dir = base_dir / "dataset" / "processed_v2"

    print(f"\n输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")

    # 处理每个split
    for split in ["train", "val", "test"]:
        print(f"\n处理 {split} 集...")

        split_input_dir = input_dir / split
        split_output_dir = output_dir / split
        output_img_dir = split_output_dir / "images"
        output_mask_dir = split_output_dir / "masks"

        output_img_dir.mkdir(parents=True, exist_ok=True)
        output_mask_dir.mkdir(parents=True, exist_ok=True)

        input_img_dir = split_input_dir / "images"
        input_mask_dir = split_input_dir / "masks"

        if not input_mask_dir.exists():
            print(f"  跳过: 目录不存在")
            continue

        # 获取所有mask文件
        mask_files = list(input_mask_dir.glob("*.png"))
        print(f"  发现 {len(mask_files)} 个mask文件")

        converted_count = 0
        for mask_file in mask_files:
            try:
                # 读取并转换mask
                mask = Image.open(mask_file)
                mask_array = np.array(mask)
                if mask_array.ndim == 3:
                    mask_array = mask_array[:, :, 0]

                # 转换类别ID
                converted_mask = convert_mask(mask_array)

                # 保存转换后的mask
                output_mask_path = output_mask_dir / mask_file.name
                Image.fromarray(converted_mask.astype(np.uint8)).save(output_mask_path)

                # 复制图像文件
                img_file = input_img_dir / (mask_file.stem + ".jpg")
                if not img_file.exists():
                    # 尝试其他扩展名
                    for ext in [".png", ".jpeg", ".JPG", ".PNG", ".JPEG"]:
                        img_file = input_img_dir / (mask_file.stem + ext)
                        if img_file.exists():
                            break

                if img_file.exists():
                    output_img_path = output_img_dir / img_file.name
                    import shutil
                    shutil.copy2(img_file, output_img_path)
                    converted_count += 1
                else:
                    print(f"    [警告] 图像不存在: {img_file.name}")

            except Exception as e:
                print(f"    [错误] 处理失败 {mask_file.name}: {e}")

        print(f"  成功转换: {converted_count} 个样本")

    print(f"\n转换完成!")
    print(f"新数据集位置: {output_dir}")
    print("="*70)

if __name__ == "__main__":
    main()
