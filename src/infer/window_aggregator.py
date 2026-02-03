"""
3秒窗口聚合模块 - 电缆胶带缺陷检测

按照项目方案要求：
- 每3秒评估一次缠绕是否均匀
- 窗口内多帧统计聚合（6-12帧，约2-4fps推理）
- 输出最终判定（OK/NG + 原因）

参考：绕包机器算法检测项目方案以及实施计划（修订）
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np

from src.utils.geometry_enhanced import (
    DiameterMetrics,
    ThicknessProfile,
    DefectAnalysis
)


@dataclass
class FrameResult:
    """单帧推理结果"""
    timestamp_ns: int
    frame_id: int

    # 几何测量
    diameter: DiameterMetrics
    thickness_profile: Optional[ThicknessProfile] = None

    # 缺陷分析
    defect_analysis: Optional[DefectAnalysis] = None

    # 原始数据
    delta_d_mm: float = 0.0
    wrap_diameter_mm: float = 0.0


@dataclass
class WindowStatistics:
    """窗口统计结果"""
    window_id: str
    start_time_ns: int
    end_time_ns: int
    num_frames: int

    # ΔD 统计（厚度增量）
    delta_d_mean: float  # 平均厚度增量
    delta_d_std: float  # 厚度标准差（不均匀度）
    delta_d_max: float  # 最大厚度（鼓包检测）
    delta_d_min: float  # 最小厚度（厚度不足检测）
    delta_d_p95: float  # 95分位数
    delta_d_range: float  # 极差（max - min）

    # 直径统计
    dc_mean: float  # 平均电缆直径
    dt_mean: float  # 平均胶带外径

    # 覆盖率统计
    cable_coverage_mean: float
    tape_coverage_mean: float
    tape_hole_ratio_max: float  # 最大孔洞率

    # 缺陷统计
    total_defect_area: int  # 总缺陷面积
    frames_with_defects: int  # 有缺陷的帧数
    tape_components_avg: float  # 胶带连通域平均数
    tape_components_max: int  # 胶带连通域最大数
    defect_areas_by_class: Dict[int, int] = field(default_factory=dict)


class WindowAggregator:
    """
    3秒窗口聚合器

    用法：
        aggregator = WindowAggregator(window_duration_sec=3.0, min_frames=6)

        for frame in video_stream:
            aggregator.add_frame(frame_result)
            if aggregator.is_ready():
                stats = aggregator.get_statistics()
                decision = make_decision(stats)
                aggregator.reset()
    """

    def __init__(
        self,
        window_duration_sec: float = 3.0,
        min_frames: int = 6,
        max_frames: int = 12
    ):
        """
        Args:
            window_duration_sec: 窗口时长（秒）
            min_frames: 最小帧数（不满足则不评估）
            max_frames: 最大帧数（超过后强制评估）
        """
        self.window_duration_ns = int(window_duration_sec * 1e9)
        self.min_frames = min_frames
        self.max_frames = max_frames

        self.frames: List[FrameResult] = []
        self.window_count = 0

    def reset(self):
        """重置窗口"""
        self.frames = []
        self.window_count += 1

    def add_frame(self, frame_result: FrameResult):
        """
        添加一帧结果

        Args:
            frame_result: 单帧推理结果
        """
        self.frames.append(frame_result)

    def is_ready(self) -> bool:
        """
        检查窗口是否准备好评估

        Returns:
            True if ready to evaluate
        """
        if len(self.frames) < self.min_frames:
            return False

        # 检查时间窗口
        if len(self.frames) >= self.max_frames:
            return True

        time_span = self.frames[-1].timestamp_ns - self.frames[0].timestamp_ns
        return time_span >= self.window_duration_ns

    def get_statistics(self) -> WindowStatistics:
        """
        计算窗口统计量

        Returns:
            WindowStatistics: 窗口统计结果
        """
        if len(self.frames) == 0:
            raise ValueError("No frames in window")

        # 提取所有帧的指标
        delta_d_list = []
        dc_list = []
        dt_list = []
        cable_cov_list = []
        tape_cov_list = []
        tape_hole_list = []
        tape_comp_list = []

        total_defect_area = 0
        defect_areas_by_class = {}
        frames_with_defects = 0

        for frame in self.frames:
            d = frame.diameter
            delta_d_list.append(d.delta_d_mm)
            dc_list.append(d.dc_mm)
            dt_list.append(d.dt_mm)
            cable_cov_list.append(d.cable_coverage)
            tape_cov_list.append(d.tape_coverage)

            if frame.defect_analysis:
                tape_hole_list.append(frame.defect_analysis.tape_hole_ratio)
                tape_comp_list.append(frame.defect_analysis.tape_num_components)

                total_defect_area += frame.defect_analysis.total_defect_area

                for cls_id, area in frame.defect_analysis.defect_areas.items():
                    defect_areas_by_class[cls_id] = \
                        defect_areas_by_class.get(cls_id, 0) + area

                if frame.defect_analysis.total_defect_area > 0:
                    frames_with_defects += 1
            else:
                tape_hole_list.append(0.0)
                tape_comp_list.append(0)

        # 转为numpy数组
        delta_d_arr = np.array(delta_d_list)

        # 统计量
        delta_d_mean = float(np.mean(delta_d_arr))
        delta_d_std = float(np.std(delta_d_arr))
        delta_d_max = float(np.max(delta_d_arr))
        delta_d_min = float(np.min(delta_d_arr))
        delta_d_p95 = float(np.percentile(delta_d_arr, 95))
        delta_d_range = delta_d_max - delta_d_min

        dc_mean = float(np.mean(dc_list))
        dt_mean = float(np.mean(dt_list))

        cable_coverage_mean = float(np.mean(cable_cov_list))
        tape_coverage_mean = float(np.mean(tape_cov_list))
        tape_hole_ratio_max = float(np.max(tape_hole_list))

        tape_components_avg = float(np.mean(tape_comp_list))
        tape_components_max = int(np.max(tape_comp_list))

        # 窗口ID
        start_time = self.frames[0].timestamp_ns
        end_time = self.frames[-1].timestamp_ns
        window_id = f"win_{self.window_count:06d}_{start_time}"

        return WindowStatistics(
            window_id=window_id,
            start_time_ns=start_time,
            end_time_ns=end_time,
            num_frames=len(self.frames),
            delta_d_mean=delta_d_mean,
            delta_d_std=delta_d_std,
            delta_d_max=delta_d_max,
            delta_d_min=delta_d_min,
            delta_d_p95=delta_d_p95,
            delta_d_range=delta_d_range,
            dc_mean=dc_mean,
            dt_mean=dt_mean,
            cable_coverage_mean=cable_coverage_mean,
            tape_coverage_mean=tape_coverage_mean,
            tape_hole_ratio_max=tape_hole_ratio_max,
            total_defect_area=total_defect_area,
            frames_with_defects=frames_with_defects,
            defect_areas_by_class=defect_areas_by_class,
            tape_components_avg=tape_components_avg,
            tape_components_max=tape_components_max
        )


@dataclass
class ThresholdConfig:
    """判定阈值配置"""
    # 厚度阈值（mm）
    target_delta_d: float = 20.0  # 目标厚度增量
    delta_d_tolerance: float = 5.0  # 容差范围
    delta_d_min_tolerance: float = 3.0  # 最小厚度容差

    # 鼓包检测
    bulge_delta_max: float = 28.0  # ΔD最大值阈值（20 + 8）
    bulge_delta_p95: float = 26.0  # ΔD P95阈值

    # 不均匀检测
    uneven_std_threshold: float = 3.0  # 标准差阈值
    uneven_range_threshold: float = 10.0  # 极差阈值

    # 脱落检测
    tape_coverage_min: float = 0.3  # 最小胶带覆盖率
    tape_hole_ratio_max: float = 0.15  # 最大孔洞率
    tape_components_max: int = 5  # 最大连通域数（超过说明断裂）

    # 缺陷检测
    defect_area_threshold: int = 500  # 缺陷面积阈值（像素）
    defect_frame_ratio: float = 0.5  # 有缺陷帧的比例阈值


@dataclass
class DecisionResult:
    """判定结果"""
    window_id: str
    result: str  # "OK" or "NG"
    reasons: List[str]  # NG原因列表
    severity: str  # "P1" or "P2"

    # 关键指标（用于日志）
    metrics: Dict[str, Any]

    timestamp: str


def make_decision(
    stats: WindowStatistics,
    config: ThresholdConfig
) -> DecisionResult:
    """
    基于窗口统计量做出判定

    Args:
        stats: 窗口统计结果
        config: 判定阈值配置

    Returns:
        DecisionResult: 判定结果
    """
    reasons = []
    severity = "P2"  # 默认严重程度

    # 1. 厚度不足检测
    if stats.delta_d_min < (config.target_delta_d - config.delta_d_min_tolerance):
        reasons.append(
            f"thickness_insufficient: ΔD_min={stats.delta_d_min:.2f}mm < "
            f"{config.target_delta_d - config.delta_d_min_tolerance:.2f}mm"
        )
        severity = "P1"

    if stats.delta_d_mean < (config.target_delta_d - config.delta_d_tolerance):
        reasons.append(
            f"thickness_low_average: ΔD_mean={stats.delta_d_mean:.2f}mm < "
            f"{config.target_delta_d - config.delta_d_tolerance:.2f}mm"
        )
        severity = "P1"

    # 2. 鼓包检测（局部厚度异常）
    if stats.delta_d_max > config.bulge_delta_max:
        reasons.append(
            f"bulge_detected: ΔD_max={stats.delta_d_max:.2f}mm > "
            f"{config.bulge_delta_max:.2f}mm"
        )
        severity = "P1"

    if stats.delta_d_p95 > config.bulge_delta_p95:
        reasons.append(
            f"bulge_p95_exceeded: ΔD_p95={stats.delta_d_p95:.2f}mm > "
            f"{config.bulge_delta_p95:.2f}mm"
        )
        severity = "P2"

    # 3. 不均匀检测
    if stats.delta_d_std > config.uneven_std_threshold:
        reasons.append(
            f"wrap_uneven_std: ΔD_std={stats.delta_d_std:.2f}mm > "
            f"{config.uneven_std_threshold:.2f}mm"
        )
        severity = "P2"

    if stats.delta_d_range > config.uneven_range_threshold:
        reasons.append(
            f"wrap_uneven_range: ΔD_range={stats.delta_d_range:.2f}mm > "
            f"{config.uneven_range_threshold:.2f}mm"
        )
        severity = "P2"

    # 4. 脱落检测
    if stats.tape_coverage_mean < config.tape_coverage_min:
        reasons.append(
            f"tape_low_coverage: coverage={stats.tape_coverage_mean:.2%} < "
            f"{config.tape_coverage_min:.2%}"
        )
        severity = "P1"

    if stats.tape_hole_ratio_max > config.tape_hole_ratio_max:
        reasons.append(
            f"tape_excessive_holes: hole_ratio={stats.tape_hole_ratio_max:.2%} > "
            f"{config.tape_hole_ratio_max:.2%}"
        )
        severity = "P1"

    if stats.tape_components_max > config.tape_components_max:
        reasons.append(
            f"tape_fragmented: components={stats.tape_components_max} > "
            f"{config.tape_components_max}"
        )
        severity = "P1"

    # 5. 缺陷检测（电缆损伤等）
    if stats.total_defect_area > config.defect_area_threshold:
        defect_ratio = stats.frames_with_defects / max(stats.num_frames, 1)
        if defect_ratio > config.defect_frame_ratio:
            reasons.append(
                f"cable_defect_detected: total_area={stats.total_defect_area}px, "
                f"frames={stats.frames_with_defects}/{stats.num_frames}"
            )
            severity = "P1"

    # 判定结果
    result = "NG" if reasons else "OK"

    # 生成时间戳
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 关键指标
    metrics = {
        "delta_d_mean": round(stats.delta_d_mean, 2),
        "delta_d_std": round(stats.delta_d_std, 2),
        "delta_d_min": round(stats.delta_d_min, 2),
        "delta_d_max": round(stats.delta_d_max, 2),
        "delta_d_range": round(stats.delta_d_range, 2),
        "dc_mean": round(stats.dc_mean, 2),
        "dt_mean": round(stats.dt_mean, 2),
        "tape_coverage": round(stats.tape_coverage_mean, 3),
        "tape_hole_ratio_max": round(stats.tape_hole_ratio_max, 3),
        "defect_area": stats.total_defect_area,
        "num_frames": stats.num_frames
    }

    return DecisionResult(
        window_id=stats.window_id,
        result=result,
        reasons=reasons,
        severity=severity,
        metrics=metrics,
        timestamp=timestamp
    )
