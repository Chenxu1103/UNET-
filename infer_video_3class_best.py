"""
3类最佳模型推理脚本 - 实时检测

使用模型：checkpoints_3class_finetuned/best_model.pth
类别：背景(0)、电缆(1)、胶带(2)
mIoU: 70.96%

支持功能：
- 视频文件推理
- 实时直径测量
- 厚度均匀度分析
- 可视化输出
- 事件记录
"""
import os
import cv2
import argparse
import time
import datetime
import numpy as np
import torch
import json
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional

# 添加目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

from src.models.unetpp import NestedUNet


# 类别配置 (3类)
CLASS_NAMES = {
    0: '背景',
    1: '电缆',
    2: '胶带',
}

# 类别颜色 (BGR格式)
CLASS_COLORS = {
    0: (0, 0, 0),         # 黑色：背景
    1: (255, 0, 0),       # 蓝色：电缆
    2: (0, 255, 0),       # 绿色：胶带
}


def softmax_np(x):
    """Numpy softmax"""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def thresholded_argmax(probs, t_cable=0.45, t_tape=0.50, bg_margin=0.15):
    """
    互斥阈值化argmax分割

    Args:
        probs: (H, W, C) 概率图，C=3 (bg, cable, tape)
        t_cable: 电缆置信度阈值
        t_tape: 胶带置信度阈值
        bg_margin: 与背景的最小margin

    Returns:
        mask_cable, mask_tape: 互斥的二值mask
    """
    bg = probs[..., 0]
    cable = probs[..., 1]
    tape = probs[..., 2]

    # argmax得到互斥的winner
    winner = np.argmax(probs[..., :3], axis=-1)

    # 电缆：winner==1 且 满足置信度 且 与背景有明显margin
    mask_cable = (winner == 1) & (cable >= t_cable) & ((cable - bg) >= bg_margin)

    # 胶带：winner==2 且 满足置信度 且 与背景有明显margin
    mask_tape = (winner == 2) & (tape >= t_tape) & ((tape - bg) >= bg_margin)

    return mask_cable.astype(np.uint8), mask_tape.astype(np.uint8)


def keep_largest_cc(mask, min_area=3000):
    """只保留最大连通域"""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels <= 1:
        return mask

    # 找到最大连通域（排除背景label 0）
    areas = stats[1:, cv2.CC_STAT_AREA]
    if len(areas) == 0 or np.max(areas) < min_area:
        return np.zeros_like(mask)

    largest_label = np.argmax(areas) + 1
    return (labels == largest_label).astype(np.uint8)


def measure_diameters_simple(mask_cable, mask_tape):
    """
    测量电缆和胶带直径

    Returns:
        dc_px: 电缆直径（像素）
        dt_px: 胶带外径（像素）
        delta_d_px: 厚度增量（像素）
    """
    # 计算每行宽度
    def calc_width(mask):
        H, W = mask.shape
        widths = []
        for y in range(H):
            xs = np.where(mask[y] > 0)[0]
            if xs.size > 1:
                widths.append(xs.max() - xs.min() + 1)
        return np.median(widths) if widths else 0

    dc_px = calc_width(mask_cable)
    dt_px = calc_width(mask_tape)
    delta_d_px = dt_px - dc_px if dc_px > 0 else 0

    return dc_px, dt_px, delta_d_px


@dataclass
class DetectionEvent:
    """检测事件"""
    timestamp: str
    frame_id: int
    dc_px: float
    dt_px: float
    delta_d_px: float
    cable_coverage: float
    tape_coverage: float


class VideoInference3Class:
    """3类模型视频推理器"""

    def __init__(self,
                 model_path: str,
                 device: str = 'cuda',
                 conf_thresh_cable: float = 0.45,
                 conf_thresh_tape: float = 0.50):
        """
        Args:
            model_path: 模型权重路径
            device: cuda或cpu
            conf_thresh_cable: 电缆置信度阈值
            conf_thresh_tape: 胶带置信度阈值
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        # 加载模型
        print(f"加载模型: {model_path}")
        self.model = NestedUNet(num_classes=3, deep_supervision=True).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()

        self.conf_thresh_cable = conf_thresh_cable
        self.conf_thresh_tape = conf_thresh_tape

        print(f"模型加载成功 (Epoch {checkpoint['epoch']+1}, mIoU {checkpoint['best_miou']:.2%})")

    def infer_frame(self, frame: np.ndarray):
        """
        推理单帧

        Args:
            frame: BGR图像

        Returns:
            mask_cable: 电缆mask
            mask_tape: 胶带mask
            metrics: 测量指标
        """
        # 预处理
        roi = frame[:, 220:220+360, :]  # 裁剪ROI
        resized = cv2.resize(roi, (512, 512))

        # 归一化
        img_tensor = resized.astype(np.float32) / 255.0
        img_tensor = np.transpose(img_tensor, (2, 0, 1))  # (3, H, W)
        img_tensor = torch.from_numpy(img_tensor).unsqueeze(0).to(self.device)

        # 推理
        with torch.no_grad():
            outputs = self.model(img_tensor)
            if isinstance(outputs, list):
                outputs = outputs[-1]

        # 后处理
        probs = softmax_np(outputs[0].cpu().numpy().transpose(1, 2, 0))  # (H, W, C)
        mask_cable, mask_tape = thresholded_argmax(
            probs,
            t_cable=self.conf_thresh_cable,
            t_tape=self.conf_thresh_tape
        )

        # 连通域过滤
        mask_cable = keep_largest_cc(mask_cable, min_area=3000)
        mask_tape = keep_largest_cc(mask_tape, min_area=2000)

        # 几何测量
        dc_px, dt_px, delta_d_px = measure_diameters_simple(mask_cable, mask_tape)

        # 计算覆盖率
        cable_coverage = mask_cable.sum() / mask_cable.size
        tape_coverage = mask_tape.sum() / mask_tape.size

        metrics = {
            'dc_px': dc_px,
            'dt_px': dt_px,
            'delta_d_px': delta_d_px,
            'cable_coverage': cable_coverage,
            'tape_coverage': tape_coverage
        }

        # 返回原始尺寸的mask（512x512，后面会根据ROI调整）
        return mask_cable, mask_tape, metrics

    def create_overlay(self, frame, mask_cable, mask_tape, metrics):
        """创建可视化叠加图 - 保持原始尺寸"""
        overlay = frame.copy()

        # 获取ROI区域
        roi_width = 360
        roi_start_x = 220
        roi = overlay[:, roi_start_x:roi_start_x+roi_width, :]

        # 调整mask到ROI尺寸
        mask_cable_resized = cv2.resize(mask_cable, (roi_width, roi.shape[0]))
        mask_tape_resized = cv2.resize(mask_tape, (roi_width, roi.shape[0]))

        # 电缆 - 蓝色 (半透明)
        cable_mask = mask_cable_resized > 0
        roi[cable_mask] = roi[cable_mask] * 0.5 + np.array([0, 0, 255]) * 0.5

        # 胶带 - 绿色 (半透明)
        tape_mask = mask_tape_resized > 0
        roi[tape_mask] = roi[tape_mask] * 0.5 + np.array([0, 255, 0]) * 0.5

        # 添加文字信息
        y_offset = 30
        texts = [
            f"Dc: {metrics['dc_px']:.1f} px",
            f"Dt: {metrics['dt_px']:.1f} px",
            f"Delta D: {metrics['delta_d_px']:.1f} px",
            f"Cable: {metrics['cable_coverage']*100:.1f}%",
            f"Tape: {metrics['tape_coverage']*100:.1f}%"
        ]

        for text in texts:
            cv2.putText(overlay, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += 25

        # 保持原始尺寸
        return overlay


def main():
    parser = argparse.ArgumentParser(description='3类模型实时检测')
    parser.add_argument('--video', type=str, required=True, help='输入视频路径')
    parser.add_argument('--model', type=str, default='checkpoints_3class_finetuned/best_model.pth', help='模型路径')
    parser.add_argument('--output', type=str, default=None, help='输出目录')
    parser.add_argument('--device', type=str, default='cuda', help='设备 (cuda/cpu)')
    parser.add_argument('--frame-stride', type=int, default=5, help='帧采样间隔')
    parser.add_argument('--show-preview', action='store_true', help='显示预览窗口')
    parser.add_argument('--conf-cable', type=float, default=0.45, help='电缆置信度阈值')
    parser.add_argument('--conf-tape', type=float, default=0.50, help='胶带置信度阈值')
    args = parser.parse_args()

    # 生成输出目录
    if args.output is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"log/detection_3class_best_{timestamp}"

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("3类最佳模型实时检测")
    print("="*70)
    print(f"输入视频: {args.video}")
    print(f"输出目录: {output_dir}")
    print(f"模型: {args.model}")
    print("="*70)
    print()

    # 初始化推理器
    inferencer = VideoInference3Class(
        model_path=args.model,
        device=args.device,
        conf_thresh_cable=args.conf_cable,
        conf_thresh_tape=args.conf_tape
    )

    # 打开视频
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"错误：无法打开视频 {args.video}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频FPS: {fps}, 总帧数: {total_frames}")
    print()

    # 视频写入 - 保持原始尺寸和帧率
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(str(output_dir / "result.mp4"), fourcc, fps, (frame_width, frame_height))

    # 统计信息
    events = []
    frame_count = 0
    processing_count = 0

    all_dc = []
    all_dt = []
    all_delta_d = []

    print("开始推理...")
    print("="*70)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # 跳帧
            if frame_count % args.frame_stride != 0:
                continue

            processing_count += 1

            # 推理
            mask_cable, mask_tape, metrics = inferencer.infer_frame(frame)

            # 记录统计
            if metrics['dc_px'] > 0:
                all_dc.append(metrics['dc_px'])
                all_dt.append(metrics['dt_px'])
                all_delta_d.append(metrics['delta_d_px'])

                # 创建事件
                event = DetectionEvent(
                    timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    frame_id=frame_count,
                    dc_px=metrics['dc_px'],
                    dt_px=metrics['dt_px'],
                    delta_d_px=metrics['delta_d_px'],
                    cable_coverage=metrics['cable_coverage'],
                    tape_coverage=metrics['tape_coverage']
                )
                events.append(asdict(event))

            # 创建叠加图
            overlay = inferencer.create_overlay(frame, mask_cable, mask_tape, metrics)
            output_video.write(overlay)

            # 显示预览（调整显示大小但不改变输出视频尺寸）
            if args.show_preview:
                # 调整预览窗口显示大小（适应屏幕）
                display_height = 720
                h, w = overlay.shape[:2]
                scale = display_height / h
                display_width = int(w * scale)
                display = cv2.resize(overlay, (display_width, display_height))
                cv2.imshow('Detection Preview (Original Size Output)', display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # 打印进度
            if processing_count % 30 == 0:
                print(f"Frame {frame_count}/{total_frames}: Dc={metrics['dc_px']:.1f}, Dt={metrics['dt_px']:.1f}, DeltaD={metrics['delta_d_px']:.1f}")

    except KeyboardInterrupt:
        print("\n检测中断")

    finally:
        cap.release()
        output_video.release()
        cv2.destroyAllWindows()

    # 保存结果
    print()
    print("="*70)
    print("推理完成！")
    print("="*70)

    if len(all_dc) > 0:
        stats = {
            "total_frames_processed": len(all_dc),
            "dc_px_mean": float(np.mean(all_dc)),
            "dc_px_std": float(np.std(all_dc)),
            "dt_px_mean": float(np.mean(all_dt)),
            "dt_px_std": float(np.std(all_dt)),
            "delta_d_px_mean": float(np.mean(all_delta_d)),
            "delta_d_px_std": float(np.std(all_delta_d)),
            "delta_d_px_min": float(np.min(all_delta_d)),
            "delta_d_px_max": float(np.max(all_delta_d))
        }

        print("\n统计结果:")
        print(f"  处理帧数: {stats['total_frames_processed']}")
        print(f"  电缆直径: {stats['dc_px_mean']:.1f} ± {stats['dc_px_std']:.1f} px")
        print(f"  胶带外径: {stats['dt_px_mean']:.1f} ± {stats['dt_px_std']:.1f} px")
        print(f"  厚度增量: {stats['delta_d_px_mean']:.1f} ± {stats['delta_d_px_std']:.1f} px")
        print(f"  厚度范围: {stats['delta_d_px_min']:.1f} - {stats['delta_d_px_max']:.1f} px")

        # 保存统计结果
        with open(output_dir / "statistics.json", 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        # 保存事件记录
        with open(output_dir / "events.jsonl", 'w', encoding='utf-8') as f:
            for event in events:
                f.write(json.dumps(event, ensure_ascii=False) + '\n')

        print(f"\n事件记录: {len(events)} 条")

    print(f"\n输出文件:")
    print(f"  视频结果: {output_dir / 'result.mp4'}")
    print(f"  统计数据: {output_dir / 'statistics.json'}")
    print(f"  事件记录: {output_dir / 'events.jsonl'}")
    print("="*70)


if __name__ == '__main__':
    import sys
    main()
