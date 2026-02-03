"""可视化与日志工具模块

提供分割结果可视化、图像上色和异常截图保存等功能
"""
import cv2
import os
import datetime
import numpy as np
from typing import Dict, Tuple, Optional


# 为每个类别定义可视化颜色 (B,G,R)
COLOR_MAP = {
    0: (0, 0, 0),        # 背景 - 黑
    1: (255, 0, 0),      # 电缆 - 蓝
    2: (0, 255, 0),      # 胶带 - 绿
    3: (0, 0, 255),      # 鼓包缺陷 - 红
    4: (0, 165, 255),    # 松脱缺陷 - 橙
    5: (255, 255, 0),    # 毛刺缺陷 - 青
    6: (255, 0, 255)     # 厚度不足缺陷 - 洋红
}

# 类别名称映射
CLASS_NAMES = {
    0: "background",
    1: "cable",
    2: "tape",
    3: "bulge_defect",
    4: "loose_defect",
    5: "burr_defect",
    6: "thin_defect"
}


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color_map: Optional[Dict[int, Tuple[int, int, int]]] = None,
    alpha: float = 0.45
) -> np.ndarray:
    """将mask以颜色覆盖到原图像上，返回叠加图

    Args:
        image: BGR 图像，shape (H,W,3)
        mask: mask 标签图，shape (H,W)，像素值为类别ID
        color_map: 类别ID到BGR颜色的映射字典
        alpha: 透明度(0~1)，越小mask越透明

    Returns:
        叠加后的 BGR 图像，shape (H,W,3)
    """
    if color_map is None:
        color_map = COLOR_MAP

    # Ensure mask is 2D
    if mask is None:
        raise ValueError("mask must be provided")
    if mask.ndim == 3 and mask.shape[2] == 1:
        mask = mask[:, :, 0]
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D (H,W); got shape={mask.shape}")

    # Ensure image has 3 channels (BGR). If grayscale, convert to BGR
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 1:
        image = cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)

    overlay = image.copy().astype(np.float32)
    
    for cls_id, color in color_map.items():
        if cls_id == 0:
            continue  # 背景不覆盖
        
        # 创建类别mask
        class_mask = (mask == cls_id)
        if class_mask.any():
            # BGR颜色
            color_bgr = np.array(color, dtype=np.float32)
            # 加权混合
            overlay[class_mask] = (
                overlay[class_mask] * (1 - alpha) + 
                color_bgr * alpha
            )
    
    return overlay.astype(np.uint8)


def colorize_mask(
    mask: np.ndarray,
    color_map: Optional[Dict[int, Tuple[int, int, int]]] = None
) -> np.ndarray:
    """将mask转换为彩色图像

    Args:
        mask: mask 标签图，shape (H,W)，像素值为类别ID
        color_map: 类别ID到BGR颜色的映射字典

    Returns:
        彩色图像，shape (H,W,3)，BGR格式
    """
    if color_map is None:
        color_map = COLOR_MAP

    # Normalize mask to 2D
    if mask is None:
        raise ValueError("mask must be provided")
    if mask.ndim == 3 and mask.shape[2] == 1:
        mask = mask[:, :, 0]
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D (H,W); got shape={mask.shape}")

    H, W = mask.shape
    colored = np.zeros((H, W, 3), dtype=np.uint8)

    for cls_id, color in color_map.items():
        class_mask = (mask == cls_id)
        if class_mask.any():
            colored[class_mask] = color

    return colored


def draw_bboxes(
    image: np.ndarray,
    bboxes: list,
    labels: Optional[list] = None,
    color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2
) -> np.ndarray:
    """在图像上绘制边界框

    Args:
        image: 输入图像，shape (H,W,3)
        bboxes: 边界框列表，每个元素为 (x_min, y_min, x_max, y_max)
        labels: 标签列表，与bboxes一一对应
        color: 边界框颜色 (B,G,R)
        thickness: 线条宽度

    Returns:
        绘制后的图像
    """
    result = image.copy()
    
    for i, bbox in enumerate(bboxes):
        x_min, y_min, x_max, y_max = bbox
        
        # 绘制矩形
        cv2.rectangle(result, (x_min, y_min), (x_max, y_max), color, thickness)
        
        # 绘制标签
        if labels and i < len(labels):
            label = labels[i]
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            text_thickness = 2
            
            # 获取文字尺寸
            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, text_thickness)
            
            # 绘制背景矩形
            cv2.rectangle(
                result,
                (x_min, y_min - text_h - 10),
                (x_min + text_w + 5, y_min),
                color,
                -1
            )
            
            # 绘制文字
            cv2.putText(
                result,
                label,
                (x_min + 2, y_min - 5),
                font,
                font_scale,
                (255, 255, 255),
                text_thickness
            )
    
    return result


def save_anomaly_snapshot(
    image: np.ndarray,
    output_dir: str = "log/snapshots"
) -> str:
    """保存异常截图到日志目录

    Args:
        image: BGR 图像，shape (H,W,3)
        output_dir: 输出目录

    Returns:
        保存的文件路径
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename = f"anomaly_{timestamp}.jpg"
    filepath = os.path.join(output_dir, filename)
    
    # Handle grayscale images
    if image.ndim == 2:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 3:
        # Assume input is BGR (consistent with OpenCV)
        image_bgr = image
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    cv2.imwrite(filepath, image_bgr)
    
    return filepath


def create_comparison_image(
    original: np.ndarray,
    predicted: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    color_map: Optional[Dict[int, Tuple[int, int, int]]] = None
) -> np.ndarray:
    """创建对比图像(原图 + 预测 + 真值)

    Args:
        original: 原始图像，shape (H,W,3)，BGR格式
        predicted: 预测mask，shape (H,W)，像素值为类别ID
        ground_truth: 真值mask，shape (H,W)，可选
        color_map: 类别颜色映射

    Returns:
        水平拼接的对比图像，BGR格式
    """
    # Ensure predicted/gt masks are converted to colored BGR images
    predicted_colored = colorize_mask(predicted, color_map)

    images = []

    # Normalize original to BGR if grayscale
    if original.ndim == 2:
        orig_bgr = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    else:
        orig_bgr = original
    images.append(orig_bgr)

    images.append(predicted_colored)

    if ground_truth is not None:
        gt_colored = colorize_mask(ground_truth, color_map)
        images.append(gt_colored)

    # Ensure all images have same height
    heights = [img.shape[0] for img in images]
    if len(set(heights)) != 1:
        # Resize to the height of the original
        h = orig_bgr.shape[0]
        resized = []
        for img in images:
            if img.shape[0] != h:
                scale = h / img.shape[0]
                new_w = int(img.shape[1] * scale)
                img = cv2.resize(img, (new_w, h), interpolation=cv2.INTER_NEAREST)
            resized.append(img)
        images = resized

    # Horizontal concat
    comparison = np.hstack(images)
    return comparison
