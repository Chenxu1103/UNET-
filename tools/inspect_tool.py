"""
电缆胶带缠绕缺陷检测脚本

完整的检测流程：
1. 图像采集 / 读取
2. 模型推理
3. 几何测量（Dc, Dt, ΔD）
4. 缺陷分析
5. 3秒窗口聚合
6. 判定决策
7. 事件输出（JSONL + 图像）

根据项目方案：绕包机器算法检测项目方案以及实施计划
"""
from __future__ import annotations
import os
import sys
import argparse
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

import cv2
import numpy as np
import torch
import yaml

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.models.unetpp_lightweight import LightweightNestedUNet
from src.models.unetpp import NestedUNet
from src.utils.geometry_enhanced import (
    compute_diameter_metrics,
    compute_thickness_profile,
    analyze_defects,
    DiameterMetrics,
    DefectAnalysis
)
from src.infer.window_aggregator import (
    WindowAggregator,
    FrameResult,
    WindowStatistics,
    ThresholdConfig,
    make_decision,
    DecisionResult
)
from src.events.event_output import (
    InspectionEventLogger,
    EventConfig,
    CLASS_NAMES
)
from src.infer.visualize import overlay as overlay_mask


class InspectionConfig:
    """检测配置"""
    def __init__(self, config_path: str):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.cfg = yaml.safe_load(f)

        self.device = torch.device(
            self.cfg['device']['type'] if torch.cuda.is_available() else 'cpu'
        )

        # 类别配置
        self.num_classes = self.cfg['model']['num_classes']
        self.class_names = self.cfg.get('class_names', CLASS_NAMES)

        # ROI配置
        self.roi = None
        if self.cfg['camera'].get('roi', {}).get('enabled', False):
            roi = self.cfg['camera']['roi']
            self.roi = (roi['x'], roi['y'], roi['w'], roi['h'])

        # 模型配置
        self.input_size = tuple(self.cfg['model']['input_size'])
        self.model_path = self.cfg['model']['weights']

        # 标定配置
        self.mm_per_px = self.cfg['scale'].get('mm_per_px', 0.05)
        self.cable_diameter_mm = self.cfg['scale']['cable_diameter_mm']

        # 类别ID
        self.cable_cls = 1
        self.tape_cls = 2
        self.defect_classes = [3, 4, 5, 6]

        # 阈值配置
        thr = self.cfg['thresholds']
        self.thresholds = ThresholdConfig(
            target_delta_d=thr['target_delta_d'],
            delta_d_tolerance=thr['delta_d_tolerance'],
            delta_d_min_tolerance=thr.get('delta_d_min_tolerance', 3.0),
            bulge_delta_max=thr['bulge_delta_max'],
            bulge_delta_p95=thr['bulge_delta_p95'],
            uneven_std_threshold=thr['uneven_std_threshold'],
            uneven_range_threshold=thr['uneven_range_threshold'],
            tape_coverage_min=thr['tape_coverage_min'],
            tape_hole_ratio_max=thr['tape_hole_ratio_max'],
            tape_components_max=thr['tape_components_max'],
            defect_area_threshold=thr['defect_area_threshold'],
            defect_frame_ratio=thr['defect_frame_ratio']
        )

        # 窗口配置
        win = self.cfg.get('window', {})
        self.window_duration = win.get('duration_sec', 3.0)
        self.min_frames = win.get('min_frames', 6)
        self.max_frames = win.get('max_frames', 12)


class InspectionSystem:
    """
    电缆胶带缠绕缺陷检测系统

    用法：
        system = InspectionSystem(config_path)

        # 处理视频流
        system.process_video(video_path, output_dir)

        # 处理图像目录
        system.process_images(image_dir, output_dir)
    """

    def __init__(self, config_path: str):
        """
        Args:
            config_path: 配置文件路径
        """
        self.config = InspectionConfig(config_path)

        # 加载模型
        self.model = self._load_model()

        # 创建窗口聚合器
        self.aggregator = WindowAggregator(
            window_duration_sec=self.config.window_duration,
            min_frames=self.config.min_frames,
            max_frames=self.config.max_frames
        )

        # 创建事件日志记录器
        event_cfg = EventConfig(
            output_dir=self.config.cfg['event']['out_dir'],
            save_jsonl=self.config.cfg['event'].get('save_jsonl', True),
            save_overlay_image=self.config.cfg['event'].get('save_overlay', True)
        )
        self.logger = InspectionEventLogger(event_cfg)

        self.frame_count = 0
        self.window_count = 0

    def _load_model(self) -> torch.nn.Module:
        """加载训练好的模型"""
        model_path = self.config.model_path

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        # 加载checkpoint
        checkpoint = torch.load(model_path, map_location=self.config.device)

        # 确定模型类型
        if 'config' in checkpoint and checkpoint['config'].get('encoder'):
            # 轻量化模型
            model = LightweightNestedUNet(
                num_classes=self.config.num_classes,
                encoder=checkpoint['config'].get('encoder', 'mobilenet_v3_small'),
                pretrained_encoder=False,
                deep_supervision=False
            )
        else:
            # 标准U-Net++
            model = NestedUNet(
                num_classes=self.config.num_classes,
                deep_supervision=False,
                pretrained_encoder=False
            )

        # 加载权重
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)

        model = model.to(self.config.device)
        model.eval()

        print(f"Model loaded: {model_path}")
        print(f"Device: {self.config.device}")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

        return model

    def _preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        """图像预处理"""
        # ROI裁剪
        if self.config.roi is not None:
            x, y, w, h = self.config.roi
            image_bgr = image_bgr[y:y+h, x:x+w]

        # Resize到模型输入尺寸
        image = cv2.resize(image_bgr, self.config.input_size)

        # 归一化
        image = image.astype(np.float32) / 255.0
        # BGR -> RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # HWC -> CHW
        image = np.transpose(image, (2, 0, 1))
        # 添加batch维度
        image = np.expand_dims(image, axis=0)

        return image

    def _infer(self, image_bgr: np.ndarray) -> np.ndarray:
        """模型推理"""
        # 预处理
        input_tensor = self._preprocess(image_bgr)
        input_tensor = torch.from_numpy(input_tensor).to(self.config.device)

        # 推理
        with torch.no_grad():
            output = self.model(input_tensor)

            # 处理多输出（深度监督）
            if isinstance(output, list):
                output = output[-1]  # 使用最后一个输出

            # Argmax得到类别
            pred = torch.argmax(output, dim=1)[0].cpu().numpy()

        return pred

    def process_frame(
        self,
        image_bgr: np.ndarray,
        timestamp_ns: int
    ) -> Optional[DecisionResult]:
        """
        处理单帧图像

        Args:
            image_bgr: 输入图像（BGR）
            timestamp_ns: 时间戳（纳秒）

        Returns:
            DecisionResult 如果窗口完成，否则None
        """
        self.frame_count += 1

        # 模型推理
        pred_mask = self._infer(image_bgr)

        # Resize回原始尺寸（用于几何测量）
        if self.config.roi is not None:
            # 如果有ROI，mask已经是ROI尺寸
            h, w = image_bgr.shape[:2]
        else:
            h, w = image_bgr.shape[:2]

        pred_mask_resized = cv2.resize(
            pred_mask.astype(np.uint8),
            (w, h),
            interpolation=cv2.INTER_NEAREST
        )

        # 几何测量
        diameter_metrics = compute_diameter_metrics(
            pred_mask_resized,
            cable_cls=self.config.cable_cls,
            tape_cls=self.config.tape_cls,
            mm_per_px=self.config.mm_per_px
        )

        # 缺陷分析
        defect_analysis = analyze_defects(
            pred_mask_resized,
            cable_cls=self.config.cable_cls,
            tape_cls=self.config.tape_cls,
            defect_classes=self.config.defect_classes
        )

        # 创建帧结果
        frame_result = FrameResult(
            timestamp_ns=timestamp_ns,
            frame_id=self.frame_count,
            diameter=diameter_metrics,
            defect_analysis=defect_analysis,
            delta_d_mm=diameter_metrics.delta_d_mm,
            wrap_diameter_mm=diameter_metrics.dt_mm
        )

        # 添加到窗口
        self.aggregator.add_frame(frame_result)

        # 检查窗口是否完成
        if self.aggregator.is_ready():
            # 获取窗口统计
            window_stats = self.aggregator.get_statistics()

            # 做出判定
            decision = make_decision(window_stats, self.config.thresholds)

            # 生成叠加图
            overlay_bgr = None
            if self.config.cfg['event'].get('save_overlay', True):
                # 使用可视化配置的颜色
                vis_cfg = self.config.cfg.get('visualization', {})
                colors = vis_cfg.get('colors', {})
                alpha = vis_cfg.get('overlay_alpha', 0.45)

                # 简单叠加（使用现有visualize模块）
                overlay_bgr = overlay_mask(image_bgr, pred_mask_resized, alpha=alpha)

            # 记录事件
            self.logger.log_event(
                decision_result=decision,
                window_stats=window_stats,
                frame_bgr=image_bgr,
                overlay_bgr=overlay_bgr
            )

            # 打印结果
            self._print_result(decision, window_stats)

            # 重置窗口
            self.aggregator.reset()
            self.window_count += 1

            return decision

        return None

    def _print_result(self, decision: DecisionResult, stats: WindowStatistics):
        """打印判定结果"""
        print(f"\n{'='*60}")
        print(f"Window {self.window_count}: {decision.result} [{decision.severity}]")
        print(f"{'='*60}")
        print(f"Time: {decision.timestamp}")
        print(f"Frames: {stats.num_frames}")

        print(f"\nMetrics:")
        for key, value in decision.metrics.items():
            print(f"  {key}: {value}")

        if decision.result == "NG":
            print(f"\nReasons:")
            for reason in decision.reasons:
                print(f"  - {reason}")

        print(f"{'='*60}\n")

    def process_video(
        self,
        video_path: str,
        camera_id: str = "video_input"
    ):
        """
        处理视频文件

        Args:
            video_path: 视频路径
            camera_id: 相机ID
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return

        print(f"Processing video: {video_path}")
        print(f"Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
        print(f"Total frames: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")

        frame_interval = int(cap.get(cv2.CAP_PROP_FPS)) // 4  # 约4fps采样

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 降采样处理
                if self.frame_count % frame_interval == 0:
                    timestamp_ns = time.time_ns()
                    self.process_frame(frame, timestamp_ns)

                # 显示进度
                if self.frame_count % 100 == 0:
                    print(f"Processed {self.frame_count} frames, {self.window_count} windows")

        finally:
            cap.release()

        # 打印摘要
        self.logger.print_summary()

    def process_images(
        self,
        image_dir: str,
        extensions: List[str] = ['.jpg', '.png', '.jpeg']
    ):
        """
        处理图像目录

        Args:
            image_dir: 图像目录
            extensions: 支持的扩展名
        """
        image_files = []
        for ext in extensions:
            image_files.extend(Path(image_dir).glob(f"*{ext}"))

        image_files = sorted(image_files)

        print(f"Found {len(image_files)} images in {image_dir}")

        for img_path in image_files:
            image_bgr = cv2.imread(str(img_path))
            if image_bgr is None:
                print(f"Warning: Cannot read {img_path}")
                continue

            timestamp_ns = int(time.time() * 1e9)
            self.process_frame(image_bgr, timestamp_ns)

        # 处理剩余窗口
        if len(self.aggregator.frames) >= self.config.min_frames:
            window_stats = self.aggregator.get_statistics()
            decision = make_decision(window_stats, self.config.thresholds)
            self.logger.log_event(decision, window_stats)
            self._print_result(decision, window_stats)

        # 打印摘要
        self.logger.print_summary()


def main():
    parser = argparse.ArgumentParser(description="电缆胶带缠绕缺陷检测系统")
    parser.add_argument("--config", type=str, default="configs/inspection_config.yaml",
                        help="配置文件路径")
    parser.add_argument("--input", type=str, required=True,
                        help="输入：视频文件或图像目录")
    parser.add_argument("--type", type=str, choices=["video", "images"], default="video",
                        help="输入类型")
    parser.add_argument("--camera-id", type=str, default="cam0",
                        help="相机ID")

    args = parser.parse_args()

    # 创建检测系统
    system = InspectionSystem(args.config)

    # 处理输入
    if args.type == "video":
        system.process_video(args.input, args.camera_id)
    else:
        system.process_images(args.input)


if __name__ == "__main__":
    main()
