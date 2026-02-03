"""
电缆包裹质量检测 - 生产级实时推理脚本

主要改进：
1. 集成窗口聚合器实现多帧统计判断（减少误报漏报）
2. 精确的生产速度控制（模拟实际产线速度）
3. 增强的几何测量和缺陷分析
4. 优化的事件检测逻辑
5. 完整的事件记录和可视化

使用方法：
    python infer_video_production.py --video <video_path> --production-mode
"""

import os
import sys
import cv2
import argparse
import time
import datetime
import numpy as np
import torch
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

# 添加项目路径
project_root = Path(__file__).parent
sys_path_add = str(project_root / 'src')
if sys_path_add not in os.sys.path:
    os.sys.path.insert(0, sys_path_add)

from src.models.unetpp import NestedUNet
from src.utils.geometry_enhanced import (
    compute_diameter_metrics,
    analyze_defects,
    DiameterMetrics,
    DefectAnalysis
)
from src.infer.window_aggregator import (
    WindowAggregator,
    FrameResult,
    ThresholdConfig,
    make_decision
)


# ============================================================================
# 配置
# ============================================================================

CLASS_NAMES = {
    0: 'background',
    1: 'cable',
    2: 'tape',
    3: 'bulge_defect',   # 鼓包
    4: 'loose_defect',   # 脱落
    5: 'damage_defect',  # 破损/毛刺
    6: 'thin_defect',    # 厚度不足
}

CLASS_COLORS = {
    0: (0, 0, 0),
    1: (255, 0, 0),      # 蓝色：电缆
    2: (0, 255, 0),      # 绿色：胶带
    3: (0, 0, 255),      # 红色：鼓包缺陷
    4: (255, 255, 0),    # 青色：松脱缺陷
    5: (255, 0, 255),    # 洋红：破损缺陷
    6: (128, 0, 128),    # 紫色：厚度不足缺陷
}


# ============================================================================
# 生产级推理器
# ============================================================================

@dataclass
class ProductionConfig:
    """生产环境配置"""
    # 生产速度控制
    production_fps: float = 10.0  # 生产环境检测帧率
    enable_realtime_control: bool = True  # 启用实时速度控制

    # 窗口聚合配置
    window_duration_sec: float = 3.0  # 窗口时长（秒）
    min_frames_per_window: int = 6   # 最小帧数
    max_frames_per_window: int = 12  # 最大帧数

    # 标定参数
    mm_per_px: float = 0.05  # 毫米/像素比例
    cable_diameter_mm_known: float = 30.0  # 已知电缆直径（用于自动标定）

    # 检测阈值
    target_delta_d_mm: float = 20.0  # 目标厚度增量
    delta_d_tolerance_mm: float = 5.0  # 厚度容差
    bulge_delta_max_mm: float = 28.0  # 鼓包最大厚度阈值
    uneven_std_threshold_mm: float = 3.0  # 不均匀标准差阈值

    # 缺陷检测
    defect_classes: List[int] = field(default_factory=lambda: [3, 4, 5, 6])
    defect_area_threshold_px: int = 100  # 缺陷最小面积（像素）

    # 输出配置
    save_snapshots: bool = True
    save_overlays: bool = True
    show_preview: bool = False


class ProductionInferenceEngine:
    """生产级推理引擎"""

    def __init__(
        self,
        model_path: str,
        config: ProductionConfig,
        device: str = 'cuda'
    ):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # 加载模型
        self.model = NestedUNet(
            num_classes=7,
            input_channels=3,
            deep_supervision=False
        ).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'])
        elif 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()

        # 初始化窗口聚合器
        self.aggregator = WindowAggregator(
            window_duration_sec=config.window_duration_sec,
            min_frames=config.min_frames_per_window,
            max_frames=config.max_frames_per_window
        )

        # 判定配置
        self.threshold_config = ThresholdConfig(
            target_delta_d=config.target_delta_d_mm,
            delta_d_tolerance=config.delta_d_tolerance_mm,
            bulge_delta_max=config.bulge_delta_max_mm,
            uneven_std_threshold=config.uneven_std_threshold_mm,
            defect_area_threshold=config.defect_area_threshold_px
        )

        # 状态统计
        self.total_frames = 0
        self.processed_frames = 0
        self.window_count = 0
        self.ng_count = 0

    def preprocess(self, frame_bgr: np.ndarray) -> torch.Tensor:
        """预处理图像"""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (512, 512))
        frame_normalized = frame_resized.astype(np.float32) / 255.0
        frame_tensor = torch.from_numpy(frame_normalized).permute(2, 0, 1).unsqueeze(0)
        return frame_tensor.to(self.device)

    @torch.no_grad()
    def predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        """预测单帧"""
        input_tensor = self.preprocess(frame_bgr)
        output = self.model(input_tensor)

        if isinstance(output, list):
            output = output[0]

        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        h, w = frame_bgr.shape[:2]
        pred_mask = cv2.resize(pred_mask.astype(np.uint8), (w, h),
                               interpolation=cv2.INTER_NEAREST)
        return pred_mask

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        timestamp_ns: int,
        frame_id: int
    ) -> Optional[FrameResult]:
        """
        处理单帧并返回结果

        Returns:
            FrameResult 或 None（如果无法计算有效指标）
        """
        # 推理
        pred_mask = self.predict(frame_bgr)

        # 计算直径指标
        diameter_metrics = compute_diameter_metrics(
            pred_mask,
            cable_cls=1,
            tape_cls=2,
            mm_per_px=self.config.mm_per_px
        )

        # 如果有效行数太少，返回None
        if diameter_metrics.valid_rows < 20:
            return None

        # 缺陷分析
        defect_analysis = analyze_defects(
            pred_mask,
            cable_cls=1,
            tape_cls=2,
            defect_classes=self.config.defect_classes
        )

        return FrameResult(
            timestamp_ns=timestamp_ns,
            frame_id=frame_id,
            diameter=diameter_metrics,
            thickness_profile=None,
            defect_analysis=defect_analysis,
            delta_d_mm=diameter_metrics.delta_d_mm,
            wrap_diameter_mm=diameter_metrics.dt_mm
        )

    def overlay_mask(self, frame_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """将mask叠加到原始图像"""
        overlay = frame_bgr.copy()
        for class_id, color in CLASS_COLORS.items():
            if class_id == 0:
                continue
            overlay[mask == class_id] = color
        return cv2.addWeighted(frame_bgr, 1 - alpha, overlay, alpha, 0)

    def add_info_to_display(
        self,
        display: np.ndarray,
        metrics: Dict[str, Any],
        decision_result: Optional[Any] = None
    ):
        """添加调试信息到显示图像"""
        y_offset = 30
        line_height = 25

        # 基本信息白色
        info_lines = [
            f"Frames: {self.total_frames} (Processed: {self.processed_frames})",
            f"Windows: {self.window_count} (NG: {self.ng_count})",
            "",
        ]

        for line in info_lines:
            cv2.putText(display, line, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += line_height

        # 几何指标 - 蓝色
        if 'delta_d_mean' in metrics:
            geo_lines = [
                f"ΔD: {metrics.get('delta_d_mean', 0):.2f}mm "
                f"(min:{metrics.get('delta_d_min', 0):.2f}, "
                f"max:{metrics.get('delta_d_max', 0):.2f})",
                f"Std: {metrics.get('delta_d_std', 0):.2f}mm "
                f"Range: {metrics.get('delta_d_range', 0):.2f}mm",
            ]
            for line in geo_lines:
                cv2.putText(display, line, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 150, 0), 2)
                y_offset += line_height

        # 覆盖率 - 绿色
        if 'tape_coverage' in metrics:
            cv2.putText(display,
                       f"Tape Cov: {metrics['tape_coverage']:.1%}",
                       (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += line_height

        # 判定结果
        if decision_result:
            if decision_result.result == "NG":
                color = (0, 0, 255)  # 红色
                status_text = f"NG: {decision_result.reasons[0][:40]}"
            else:
                color = (0, 255, 0)  # 绿色
                status_text = "OK"

            cv2.putText(display, status_text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


# ============================================================================
# 主处理流程
# ============================================================================

def process_video_production(
    video_path: str,
    model_path: str,
    output_dir: str,
    config: ProductionConfig,
    device: str = 'cuda'
):
    """生产级视频处理流程"""

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'snapshots'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'overlays'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'windows'), exist_ok=True)

    # 初始化引擎
    engine = ProductionInferenceEngine(model_path, config, device)

    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"=" * 60)
    print(f"生产级实时检测模式")
    print(f"=" * 60)
    print(f"视频信息: {width}x{height} @ {video_fps:.2f}fps, 共 {total_frames} 帧")
    print(f"生产速度: {config.production_fps:.1f} 帧/秒")
    print(f"窗口聚合: {config.window_duration_sec}秒窗口, "
          f"{config.min_frames_per_window}-{config.max_frames_per_window}帧")
    print(f"检测阈值: 目标ΔD={config.target_delta_d_mm}mm, "
          f"容差±{config.delta_d_tolerance_mm}mm")
    print(f"=" * 60)

    # 计算采样间隔（匹配生产速度）
    sample_interval = max(1, int(round(video_fps / config.production_fps)))
    print(f"采样间隔: 每 {sample_interval} 帧采样一次")

    # 日志文件
    log_path = os.path.join(output_dir, 'events_log.csv')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("timestamp,window_id,result,severity,reasons,metrics\n")

    # 时间控制变量
    frame_time_accumulator = 0.0
    target_frame_time = 1.0 / config.production_fps  # 目标帧处理时间
    last_wall_time = time.time()

    # 缓存帧用于保存
    frame_buffer: Dict[int, np.ndarray] = {}
    mask_buffer: Dict[int, np.ndarray] = {}

    # 处理循环
    frame_idx = 0

    try:
        while True:
            loop_start = time.time()

            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            engine.total_frames += 1

            # 采样
            if frame_idx % sample_interval != 0:
                continue

            engine.processed_frames += 1

            # 生成时间戳（纳秒）
            timestamp_ns = int(time.time() * 1e9)

            # 处理帧
            frame_result = engine.process_frame(frame, timestamp_ns, frame_idx)

            if frame_result is None:
                continue

            # 缓存帧和mask（用于事件保存）
            pred_mask = engine.predict(frame)
            frame_buffer[frame_idx] = frame.copy()
            mask_buffer[frame_idx] = pred_mask.copy()

            # 添加到聚合器
            engine.aggregator.add_frame(frame_result)

            # 检查窗口是否准备好评估
            decision_result = None
            window_metrics = {}

            if engine.aggregator.is_ready():
                engine.window_count += 1

                # 获取窗口统计
                window_stats = engine.aggregator.get_statistics()

                # 做出判定
                decision_result = make_decision(window_stats, engine.threshold_config)
                window_metrics = decision_result.metrics

                # 记录事件
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")

                # 保存日志
                with open(log_path, 'a', encoding='utf-8') as f:
                    reasons_str = '; '.join(decision_result.reasons)
                    metrics_str = json.dumps(window_metrics, ensure_ascii=False)
                    f.write(f"{ts},{decision_result.window_id},{decision_result.result},"
                           f"{decision_result.severity},{reasons_str},{metrics_str}\n")

                # 统计NG
                if decision_result.result == "NG":
                    engine.ng_count += 1
                    print(f"  [窗口 {engine.window_count}] NG - "
                          f"{'; '.join(decision_result.reasons[:2])}")

                    # 保存NG窗口的帧
                    if config.save_snapshots:
                        window_dir = os.path.join(output_dir, 'windows',
                                                 f"{decision_result.window_id}")
                        os.makedirs(window_dir, exist_ok=True)

                        for frame_res in engine.aggregator.frames:
                            fid = frame_res.frame_id
                            if fid in frame_buffer and fid in mask_buffer:
                                # 保存原图和mask叠加图
                                snap_path = os.path.join(window_dir, f"frame_{fid}.jpg")
                                overlay_path = os.path.join(window_dir,
                                                           f"frame_{fid}_overlay.jpg")
                                cv2.imwrite(snap_path, frame_buffer[fid])

                                overlay = engine.overlay_mask(frame_buffer[fid],
                                                             mask_buffer[fid], 0.5)
                                cv2.imwrite(overlay_path, overlay)

                        # 保存窗口JSON
                        json_path = os.path.join(window_dir, "window_info.json")
                        with open(json_path, 'w', encoding='utf-8') as f:
                            json.dump({
                                'window_id': decision_result.window_id,
                                'timestamp': ts,
                                'result': decision_result.result,
                                'severity': decision_result.severity,
                                'reasons': decision_result.reasons,
                                'metrics': window_metrics,
                                'num_frames': window_stats.num_frames
                            }, f, ensure_ascii=False, indent=2)

                # 重置窗口
                engine.aggregator.reset()

                # 清理旧缓存
                oldest_fid = frame_idx - sample_interval * config.max_frames_per_window
                frame_buffer = {fid: buf for fid, buf in frame_buffer.items()
                               if fid > oldest_fid}
                mask_buffer = {fid: buf for fid, buf in mask_buffer.items()
                              if fid > oldest_fid}

            # 显示
            if config.show_preview:
                overlay = engine.overlay_mask(frame, pred_mask, 0.5)
                display = cv2.resize(overlay, (1024, 768))

                engine.add_info_to_display(display, window_metrics, decision_result)

                cv2.imshow('Production Inspection', display)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
                elif key == 32:
                    cv2.waitKey(0)

            # 实时速度控制
            if config.enable_realtime_control:
                # 计算实际经过时间
                current_wall_time = time.time()
                elapsed = current_wall_time - last_wall_time

                # 如果处理太快，等待以匹配生产速度
                if elapsed < target_frame_time:
                    sleep_time = target_frame_time - elapsed
                    time.sleep(sleep_time)

                last_wall_time = time.time()

    except KeyboardInterrupt:
        print("\n处理被中断")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    # 输出统计
    print(f"\n{'='*60}")
    print(f"处理完成!")
    print(f"  总帧数: {engine.total_frames}")
    print(f"  处理帧数: {engine.processed_frames}")
    print(f"  评估窗口: {engine.window_count}")
    print(f"  NG窗口: {engine.ng_count}")
    if engine.window_count > 0:
        print(f"  NG率: {engine.ng_count/engine.window_count:.1%}")
    print(f"  结果保存在: {output_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description='电缆包裹质量检测 - 生产级实时推理'
    )

    # 必需参数
    parser.add_argument('--video', type=str, required=True,
                       help='输入视频路径')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pth',
                       help='模型路径')
    parser.add_argument('--output', type=str, default='log/production_output',
                       help='输出目录')

    # 生产模式
    parser.add_argument('--production-mode', action='store_true', default=True,
                       help='启用生产模式（默认启用）')
    parser.add_argument('--production-fps', type=float, default=10.0,
                       help='生产环境检测速度（帧/秒）')
    parser.add_argument('--no-realtime-control', action='store_true',
                       help='禁用实时速度控制')

    # 标定参数
    parser.add_argument('--mm-per-px', type=float, default=0.05,
                       help='毫米/像素比例')
    parser.add_argument('--cable-diameter-mm', type=float, default=30.0,
                       help='已知电缆直径（毫米）')

    # 检测阈值
    parser.add_argument('--target-delta-d', type=float, default=20.0,
                       help='目标厚度增量（毫米）')
    parser.add_argument('--delta-d-tolerance', type=float, default=5.0,
                       help='厚度容差（毫米）')
    parser.add_argument('--bulge-max', type=float, default=28.0,
                       help='鼓包最大厚度（毫米）')

    # 窗口配置
    parser.add_argument('--window-duration', type=float, default=3.0,
                       help='窗口时长（秒）')
    parser.add_argument('--min-frames', type=int, default=6,
                       help='窗口最小帧数')
    parser.add_argument('--max-frames', type=int, default=12,
                       help='窗口最大帧数')

    # 其他
    parser.add_argument('--device', type=str, default='cuda',
                       help='计算设备')
    parser.add_argument('--show-preview', action='store_true',
                       help='显示实时预览')

    args = parser.parse_args()

    # 创建配置
    config = ProductionConfig(
        production_fps=args.production_fps,
        enable_realtime_control=not args.no_realtime_control,
        window_duration_sec=args.window_duration,
        min_frames_per_window=args.min_frames,
        max_frames_per_window=args.max_frames,
        mm_per_px=args.mm_per_px,
        cable_diameter_mm_known=args.cable_diameter_mm,
        target_delta_d_mm=args.target_delta_d,
        delta_d_tolerance_mm=args.delta_d_tolerance,
        bulge_delta_max_mm=args.bulge_max,
        show_preview=args.show_preview
    )

    # 处理视频
    process_video_production(
        video_path=args.video,
        model_path=args.model,
        output_dir=args.output,
        config=config,
        device=args.device
    )


if __name__ == '__main__':
    main()
