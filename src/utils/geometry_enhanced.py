"""
增强几何测量模块 - 电缆胶带缺陷检测

根据项目方案实现的几何测量算法：
- Dc（电缆直径）：在没有胶带覆盖的位置测电缆横向宽度
- Dt（胶带外径）：在胶带覆盖区域内测胶带横向宽度
- ΔD（厚度增量）：Dt - Dc，用于判定鼓包、厚度不足等缺陷

参考：绕包机器算法检测项目方案以及实施计划（修订）
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2


@dataclass
class DiameterMetrics:
    """单帧直径测量结果"""
    # 像素值
    dc_px: float  # 电缆直径 (px)
    dt_px: float  # 胶带外径 (px)
    delta_d_px: float  # 厚度增量 Dt - Dc (px)

    # 毫米值（需要标定 mm_per_px）
    dc_mm: float
    dt_mm: float
    delta_d_mm: float

    # 统计量
    valid_rows: int  # 有效行数
    cable_coverage: float  # 电缆覆盖率 (0-1)
    tape_coverage: float  # 胶带覆盖率 (0-1)


@dataclass
class ThicknessProfile:
    """厚度分布轮廓"""
    y_coords: np.ndarray  # Y坐标 (行号)
    delta_d_mm: np.ndarray  # 每行的厚度增量 (mm)
    valid_mask: np.ndarray  # 有效行掩码


def _compute_width_per_row(
    mask: np.ndarray,
    smooth: bool = True,
    kernel_size: int = 21
) -> np.ndarray:
    """
    计算二值mask每一行的宽度（像素）

    Args:
        mask: (H, W) 二值mask
        smooth: 是否高斯平滑
        kernel_size: 平滑核大小（奇数）

    Returns:
        widths: (H,) 每行宽度，像素
    """
    H, W = mask.shape
    widths = np.zeros(H, dtype=np.float32)

    for y in range(H):
        xs = np.where(mask[y] > 0)[0]
        if xs.size > 0:
            widths[y] = float(xs.max() - xs.min() + 1)

    # 高斯平滑去噪
    if smooth and kernel_size > 1:
        k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        widths = cv2.GaussianBlur(
            widths.reshape(-1, 1),
            (1, k),
            sigmaX=0
        ).reshape(-1)

    return widths


def _largest_connected_component(
    binary_mask: np.ndarray,
    min_area: int = 100
) -> np.ndarray:
    """
    保留最大连通分量（去噪）

    Args:
        binary_mask: (H, W) 二值mask
        min_area: 最小面积阈值

    Returns:
        cleaned_mask: 清理后的二值mask
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )

    if num_labels <= 1:
        return binary_mask

    # 跳过背景(0)，找最大连通域
    areas = stats[1:, cv2.CC_STAT_AREA]
    valid_idx = np.where(areas >= min_area)[0]

    if len(valid_idx) == 0:
        return np.zeros_like(binary_mask)

    largest_idx = 1 + int(valid_idx[np.argmax(areas[valid_idx])])
    return (labels == largest_idx).astype(np.uint8)


def compute_diameter_metrics(
    pred_mask: np.ndarray,
    cable_cls: int = 1,
    tape_cls: int = 2,
    mm_per_px: float = 0.05,
    min_valid_rows: int = 20
) -> DiameterMetrics:
    """
    计算电缆和胶带的直径指标

    Args:
        pred_mask: (H, W) 分割预测mask，类别ID
        cable_cls: 电缆类别ID
        tape_cls: 胶带类别ID
        mm_per_px: 毫米/像素比例
        min_valid_rows: 最小有效行数

    Returns:
        DiameterMetrics: 直径测量结果
    """
    H, W = pred_mask.shape

    # 提取电缆和胶带mask
    cable_mask = (pred_mask == cable_cls).astype(np.uint8)
    tape_mask = (pred_mask == tape_cls).astype(np.uint8)

    # 保留最大连通域（去噪）
    cable_mask = _largest_connected_component(cable_mask, min_area=50)
    tape_mask = _largest_connected_component(tape_mask, min_area=50)

    # 计算每行宽度
    cable_widths = _compute_width_per_row(cable_mask, smooth=True, kernel_size=31)
    tape_widths = _compute_width_per_row(tape_mask, smooth=True, kernel_size=31)

    # 有效行：同时存在电缆和胶带
    valid = (cable_widths > 0) & (tape_widths > 0)
    valid_rows = valid.sum()

    # 覆盖率
    cable_coverage = cable_mask.sum() / (H * W)
    tape_coverage = tape_mask.sum() / (H * W)

    if valid_rows < min_valid_rows:
        # 数据不足，返回零值
        return DiameterMetrics(
            dc_px=0.0, dt_px=0.0, delta_d_px=0.0,
            dc_mm=0.0, dt_mm=0.0, delta_d_mm=0.0,
            valid_rows=valid_rows,
            cable_coverage=cable_coverage,
            tape_coverage=tape_coverage
        )

    # 中位数抗噪
    dc_px = float(np.median(cable_widths[valid]))
    dt_px = float(np.median(tape_widths[valid]))
    delta_d_px = dt_px - dc_px

    # 转换为毫米
    dc_mm = dc_px * mm_per_px
    dt_mm = dt_px * mm_per_px
    delta_d_mm = dt_mm - dc_mm

    return DiameterMetrics(
        dc_px=dc_px,
        dt_px=dt_px,
        delta_d_px=delta_d_px,
        dc_mm=dc_mm,
        dt_mm=dt_mm,
        delta_d_mm=delta_d_mm,
        valid_rows=valid_rows,
        cable_coverage=cable_coverage,
        tape_coverage=tape_coverage
    )


def compute_thickness_profile(
    pred_mask: np.ndarray,
    cable_cls: int = 1,
    tape_cls: int = 2,
    mm_per_px: float = 0.05
) -> ThicknessProfile:
    """
    计算沿Y轴的厚度分布轮廓（用于检测鼓包和不均匀）

    Args:
        pred_mask: (H, W) 分割预测mask
        cable_cls: 电缆类别ID
        tape_cls: 胶带类别ID
        mm_per_px: 毫米/像素比例

    Returns:
        ThicknessProfile: 厚度轮廓
    """
    H = pred_mask.shape[0]

    cable_mask = (pred_mask == cable_cls).astype(np.uint8)
    tape_mask = (pred_mask == tape_cls).astype(np.uint8)

    cable_widths = _compute_width_per_row(cable_mask, smooth=True, kernel_size=31)
    tape_widths = _compute_width_per_row(tape_mask, smooth=True, kernel_size=31)

    # 计算每行厚度增量
    delta_d_px = tape_widths - cable_widths
    delta_d_mm = delta_d_px * mm_per_px

    # 有效行掩码
    valid = (cable_widths > 0) & (tape_widths > 0)

    return ThicknessProfile(
        y_coords=np.arange(H),
        delta_d_mm=delta_d_mm,
        valid_mask=valid
    )


@dataclass
class DefectAnalysis:
    """缺陷分析结果"""
    # 胶带缺陷
    tape_hole_ratio: float  # 孔洞率（孔洞面积/胶带面积）
    tape_num_holes: int  # 孔洞数量
    tape_coverage: float  # 胶带覆盖率

    # 连通域分析
    cable_num_components: int  # 电缆连通域数量
    tape_num_components: int  # 胶带连通域数量
    tape_largest_area_ratio: float  # 最大连通域占比

    # 缺陷类别统计
    defect_areas: Dict[int, int]  # 各缺陷类别面积（像素）
    total_defect_area: int  # 总缺陷面积


def analyze_defects(
    pred_mask: np.ndarray,
    cable_cls: int = 1,
    tape_cls: int = 2,
    defect_classes: List[int] = (3, 4, 5, 6),
    hole_min_size: int = 10
) -> DefectAnalysis:
    """
    分析缺陷特征（覆盖率、连通域、孔洞率）

    Args:
        pred_mask: (H, W) 分割预测mask
        cable_cls: 电缆类别ID
        tape_cls: 胶带类别ID
        defect_classes: 缺陷类别ID列表
        hole_min_size: 孔洞最小尺寸

    Returns:
        DefectAnalysis: 缺陷分析结果
    """
    H, W = pred_mask.shape
    total_pixels = H * W

    # 胶带覆盖率
    tape_mask = (pred_mask == tape_cls).astype(np.uint8)
    tape_coverage = tape_mask.sum() / total_pixels

    # 胶带孔洞分析
    # 孔洞定义：在胶带区域内，被背景或其他类别包围的孤点/小区域
    tape_num_components, tape_labels, tape_stats, _ = cv2.connectedComponentsWithStats(
        tape_mask, connectivity=8
    )

    # 孔洞 = 胶带区域内的背景点（简单方法）
    # 更精确：形态学闭运算后的差值
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    tape_closed = cv2.morphologyEx(tape_mask, cv2.MORPH_CLOSE, kernel)
    holes = tape_closed.astype(np.int16) - tape_mask.astype(np.int16)
    holes = (holes > 0).astype(np.uint8)

    # 统计孔洞
    num_holes, hole_labels, hole_stats, _ = cv2.connectedComponentsWithStats(
        holes, connectivity=8
    )
    # 过滤小孔洞
    hole_areas = hole_stats[1:, cv2.CC_STAT_AREA] if num_holes > 1 else []
    valid_holes = [a for a in hole_areas if a >= hole_min_size]
    tape_num_holes = len(valid_holes)
    tape_hole_area = sum(valid_holes)
    tape_hole_ratio = tape_hole_area / max(tape_mask.sum(), 1)

    # 连通域分析
    cable_mask = (pred_mask == cable_cls).astype(np.uint8)
    cable_num_components, _, _, _ = cv2.connectedComponentsWithStats(
        cable_mask, connectivity=8
    )
    cable_num_components = max(0, cable_num_components - 1)  # 减去背景

    tape_num_components = max(0, tape_num_components - 1)

    if tape_num_components > 0:
        tape_areas = tape_stats[1:, cv2.CC_STAT_AREA]
        tape_largest_area = tape_areas.max()
        tape_largest_area_ratio = tape_largest_area / tape_mask.sum()
    else:
        tape_largest_area_ratio = 0.0

    # 缺陷类别统计
    defect_areas = {}
    total_defect_area = 0
    for cls_id in defect_classes:
        area = (pred_mask == cls_id).sum()
        defect_areas[cls_id] = int(area)
        total_defect_area += area

    return DefectAnalysis(
        tape_hole_ratio=tape_hole_ratio,
        tape_num_holes=tape_num_holes,
        tape_coverage=tape_coverage,
        cable_num_components=cable_num_components,
        tape_num_components=tape_num_components,
        tape_largest_area_ratio=tape_largest_area_ratio,
        defect_areas=defect_areas,
        total_defect_area=total_defect_area
    )
