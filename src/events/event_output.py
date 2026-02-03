"""
事件输出模块 - 电缆胶带缺陷检测

按照项目方案要求输出事件：
- 图片：保存 raw.jpg + overlay.jpg
- 文本：JSON Lines 格式（每行一个窗口事件）

参考：绕包机器算法检测项目方案以及实施计划（修订）
"""
from __future__ import annotations
import os
import json
import cv2
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List
from pathlib import Path

from src.infer.window_aggregator import DecisionResult, WindowStatistics


@dataclass
class EventConfig:
    """事件输出配置"""
    output_dir: str = "./events"
    save_raw_image: bool = True
    save_overlay_image: bool = True
    save_jsonl: bool = True
    jsonl_filename: str = "inspection_events.jsonl"

    # 图像保存配置
    image_format: str = ".jpg"
    jpeg_quality: int = 95

    # 子目录
    raw_subdir: str = "raw"
    overlay_subdir: str = "overlay"
    ok_subdir: str = "ok"
    ng_subdir: str = "ng"


class InspectionEventLogger:
    """
    检测事件日志记录器

    用法：
        logger = InspectionEventLogger(config)

        # 记录一个窗口事件
        logger.log_event(
            decision_result=decision,
            window_stats=stats,
            frame_bgr=frame_image,
            overlay_bgr=overlay_image
        )
    """

    def __init__(self, config: EventConfig):
        """
        Args:
            config: 事件配置
        """
        self.config = config
        self.output_dir = Path(config.output_dir)

        # 创建目录结构
        self.raw_dir = self.output_dir / config.raw_subdir
        self.overlay_dir = self.output_dir / config.overlay_subdir
        self.ok_dir = self.output_dir / config.ok_subdir
        self.ng_dir = self.output_dir / config.ng_subdir

        for dir_path in [self.raw_dir, self.overlay_dir, self.ok_dir, self.ng_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # JSONL 文件路径
        self.jsonl_path = self.output_dir / config.jsonl_filename

    def _generate_filename(self, decision: DecisionResult) -> str:
        """生成文件名"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_{decision.window_id}"

    def _save_images(
        self,
        decision: DecisionResult,
        frame_bgr: Optional[np.ndarray],
        overlay_bgr: Optional[np.ndarray]
    ) -> Dict[str, str]:
        """
        保存图像文件

        Returns:
            图像文件路径字典
        """
        filename = self._generate_filename(decision)
        image_paths = {}

        # 根据结果选择子目录
        subdir = self.ok_dir if decision.result == "OK" else self.ng_dir

        # 保存原始图像
        if self.config.save_raw_image and frame_bgr is not None:
            raw_path = subdir / self.config.raw_subdir / f"{filename}{self.config.image_format}"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(raw_path),
                frame_bgr,
                [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality]
            )
            image_paths["raw_image"] = str(raw_path)

        # 保存叠加图像
        if self.config.save_overlay_image and overlay_bgr is not None:
            overlay_path = subdir / self.config.overlay_subdir / f"{filename}{self.config.image_format}"
            overlay_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(overlay_path),
                overlay_bgr,
                [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality]
            )
            image_paths["overlay_image"] = str(overlay_path)

        return image_paths

    def log_event(
        self,
        decision: DecisionResult,
        window_stats: WindowStatistics,
        frame_bgr: Optional[np.ndarray] = None,
        overlay_bgr: Optional[np.ndarray] = None,
        camera_id: str = "cam0"
    ) -> Dict[str, Any]:
        """
        记录一个检测事件

        Args:
            decision: 判定结果
            window_stats: 窗口统计
            frame_bgr: 原始帧（BGR）
            overlay_bgr: 叠加可视化帧（BGR）
            camera_id: 相机ID

        Returns:
            事件记录字典
        """
        # 保存图像
        image_paths = self._save_images(decision, frame_bgr, overlay_bgr)

        # 构建事件记录
        event_record = {
            # 基本信息
            "window_id": decision.window_id,
            "timestamp": decision.timestamp,
            "camera_id": camera_id,

            # 判定结果
            "result": decision.result,
            "severity": decision.severity,
            "reasons": decision.reasons,

            # 关键指标
            "metrics": decision.metrics,

            # 窗口信息
            "window_info": {
                "start_time_ns": window_stats.start_time_ns,
                "end_time_ns": window_stats.end_time_ns,
                "num_frames": window_stats.num_frames,
                "duration_sec": (window_stats.end_time_ns - window_stats.start_time_ns) / 1e9
            },

            # 图像路径
            "images": image_paths
        }

        # 写入JSONL
        if self.config.save_jsonl:
            self._append_jsonl(event_record)

        return event_record

    def _append_jsonl(self, event_record: Dict[str, Any]):
        """追加到JSONL文件"""
        with open(self.jsonl_path, 'a', encoding='utf-8') as f:
            json.dump(event_record, f, ensure_ascii=False)
            f.write('\n')

    def get_summary(self) -> Dict[str, Any]:
        """
        获取事件摘要统计

        Returns:
            统计摘要
        """
        if not self.jsonl_path.exists():
            return {
                "total_events": 0,
                "ok_count": 0,
                "ng_count": 0,
                "p1_count": 0,
                "p2_count": 0
            }

        total = 0
        ok_count = 0
        ng_count = 0
        p1_count = 0
        p2_count = 0

        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    event = json.loads(line)
                    total += 1
                    if event.get("result") == "OK":
                        ok_count += 1
                    else:
                        ng_count += 1
                        if event.get("severity") == "P1":
                            p1_count += 1
                        else:
                            p2_count += 1

        return {
            "total_events": total,
            "ok_count": ok_count,
            "ng_count": ng_count,
            "p1_count": p1_count,
            "p2_count": p2_count,
            "ng_rate": round(ng_count / max(total, 1) * 100, 2)
        }

    def print_summary(self):
        """打印摘要统计"""
        summary = self.get_summary()
        print("\n" + "=" * 60)
        print("Inspection Event Summary")
        print("=" * 60)
        print(f"Total Events: {summary['total_events']}")
        print(f"OK Count: {summary['ok_count']}")
        print(f"NG Count: {summary['ng_count']}")
        print(f"  - P1 (Critical): {summary['p1_count']}")
        print(f"  - P2 (Warning): {summary['p2_count']}")
        print(f"NG Rate: {summary['ng_rate']}%")
        print("=" * 60)


# 类别名称映射（用于日志）
CLASS_NAMES = {
    0: "background",
    1: "cable",
    2: "tape",
    3: "bulge_defect",
    4: "loose_defect",
    5: "damage_defect",
    6: "thin_defect"
}


def format_reasons_readable(reasons: List[str]) -> str:
    """
    将原因列表转换为可读文本

    Args:
        reasons: 原因列表

    Returns:
        格式化的原因文本
    """
    if not reasons:
        return "Normal"

    # 提取关键信息
    formatted = []
    for reason in reasons:
        # 简化显示
        if "thickness_insufficient" in reason:
            formatted.append(f"厚度不足")
        elif "thickness_low_average" in reason:
            formatted.append(f"平均厚度偏低")
        elif "bulge_detected" in reason:
            formatted.append(f"鼓包异常")
        elif "bulge_p95_exceeded" in reason:
            formatted.append(f"局部厚度偏高")
        elif "wrap_uneven" in reason:
            formatted.append(f"缠绕不均匀")
        elif "tape_low_coverage" in reason:
            formatted.append(f"胶带覆盖率低")
        elif "tape_excessive_holes" in reason:
            formatted.append(f"胶带孔洞过多")
        elif "tape_fragmented" in reason:
            formatted.append(f"胶带断裂/脱落")
        elif "cable_defect_detected" in reason:
            formatted.append(f"电缆损伤")
        else:
            formatted.append(reason)

    return "; ".join(formatted)


# 导入numpy类型检查
import numpy as np
