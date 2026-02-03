"""
简化版视频检测 - 最小化过滤
只保留基本的置信度阈值，移除所有过度过滤
"""
import os
import cv2
import argparse
import datetime
import numpy as np
import torch
import json
from pathlib import Path

project_root = Path(__file__).parent
import sys
sys.path.insert(0, str(project_root / 'src'))

from src.models.unetpp import NestedUNet


# 类别配置（和标注一致）
CLASS_NAMES = {0: '背景', 1: '电缆', 2: '胶带'}
CLASS_COLORS = {
    0: (0, 0, 0),         # 黑色：背景
    1: (255, 0, 0),       # 蓝色：电缆
    2: (0, 255, 0),       # 绿色：胶带
}


def softmax_np(x):
    """Numpy softmax"""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def simple_threshold(probs, conf_threshold=0.3):
    """
    简单阈值处理 - 只过滤置信度极低的区域

    Args:
        probs: (H, W, C) 概率图
        conf_threshold: 置信度阈值（较低，不过度过滤）

    Returns:
        mask_cable, mask_tape
    """
    bg = probs[..., 0]
    cable = probs[..., 1]
    tape = probs[..., 2]

    # argmax得到winner
    winner = np.argmax(probs[..., :3], axis=-1)

    # 只要求置信度大于阈值
    mask_cable = (winner == 1) & (cable >= conf_threshold)
    mask_tape = (winner == 2) & (tape >= conf_threshold)

    return mask_cable.astype(np.uint8), mask_tape.astype(np.uint8)


def measure_diameters_simple(mask_cable, mask_tape):
    """测量电缆和胶带直径"""
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


class SimpleInference:
    """简化推理器 - 最小化过滤"""

    def __init__(self,
                 model_path: str,
                 device: str = 'cuda',
                 conf_threshold: float = 0.3):
        """
        Args:
            model_path: 模型权重路径
            device: cuda或cpu
            conf_threshold: 置信度阈值（0.3较低，减少漏检）
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.conf_threshold = conf_threshold

        print(f"使用设备: {self.device}")

        print(f"加载模型: {model_path}")
        self.model = NestedUNet(num_classes=3, deep_supervision=True).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()

        print(f"模型加载成功 (Epoch {checkpoint['epoch']+1}, mIoU {checkpoint['best_miou']:.2%})")
        print("="*70)
        print("简化模式（最小化过滤）:")
        print(f"  置信度阈值: {conf_threshold}")
        print(f"  过滤: 无（保留所有预测结果）")
        print("="*70)

    def infer_frame(self, frame: np.ndarray):
        """推理单帧"""
        h, w = frame.shape[:2]

        # BGR -> RGB（和dataset.py一致）
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(frame_rgb, (512, 512))

        # 归一化到[0,1]（和dataset.py一致）
        img_tensor = resized.astype(np.float32) / 255.0
        img_tensor = np.transpose(img_tensor, (2, 0, 1))  # HWC -> CHW
        img_tensor = torch.from_numpy(img_tensor).unsqueeze(0).to(self.device)

        # 推理
        with torch.no_grad():
            outputs = self.model(img_tensor)
            if isinstance(outputs, list):
                outputs = outputs[-1]

        # 计算概率
        probs = softmax_np(outputs[0].cpu().numpy().transpose(1, 2, 0))  # (H, W, C)

        # 简单阈值处理（不过度过滤）
        mask_cable_small, mask_tape_small = simple_threshold(probs, self.conf_threshold)

        # 调整回原始尺寸
        mask_cable_full = cv2.resize(mask_cable_small, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_tape_full = cv2.resize(mask_tape_small, (w, h), interpolation=cv2.INTER_NEAREST)

        # 几何测量
        dc_px, dt_px, delta_d_px = measure_diameters_simple(mask_cable_small, mask_tape_small)

        # 计算覆盖率
        cable_coverage = mask_cable_small.sum() / mask_cable_small.size
        tape_coverage = mask_tape_small.sum() / mask_tape_small.size

        metrics = {
            'dc_px': dc_px,
            'dt_px': dt_px,
            'delta_d_px': delta_d_px,
            'cable_coverage': cable_coverage,
            'tape_coverage': tape_coverage
        }

        # 生成用于调试的pred_mask
        pred_mask = np.zeros_like(mask_cable_small, dtype=np.uint8)
        pred_mask[mask_cable_small > 0] = 1
        pred_mask[mask_tape_small > 0] = 2

        return mask_cable_full, mask_tape_full, metrics, probs, pred_mask

    def create_overlay(self, frame, mask_cable, mask_tape, metrics, outputs, pred_mask):
        """创建可视化叠加图"""
        overlay = frame.copy()

        # 电缆 - 蓝色 BGR(255, 0, 0)
        cable_mask = mask_cable > 0
        overlay[cable_mask] = overlay[cable_mask] * 0.4 + np.array([255, 0, 0]) * 0.6

        # 胶带 - 绿色 BGR(0, 255, 0)
        tape_mask = mask_tape > 0
        overlay[tape_mask] = overlay[tape_mask] * 0.4 + np.array([0, 255, 0]) * 0.6

        # 添加文字信息
        y_offset = 30
        texts = [
            f"Cable(Blue): {metrics['cable_coverage']*100:.1f}%",
            f"Tape(Green): {metrics['tape_coverage']*100:.1f}%",
        ]

        for text in texts:
            cv2.putText(overlay, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += 25

        return overlay


def main():
    parser = argparse.ArgumentParser(description='简化版视频检测')
    parser.add_argument('--video', type=str, required=True, help='输入视频路径')
    parser.add_argument('--model', type=str, default='checkpoints_3class_finetuned/best_model.pth', help='模型路径')
    parser.add_argument('--output', type=str, default=None, help='输出目录')
    parser.add_argument('--device', type=str, default='cuda', help='设备 (cuda/cpu)')
    parser.add_argument('--frame-stride', type=int, default=1, help='帧采样间隔')
    parser.add_argument('--show-preview', action='store_true', help='显示预览窗口')
    parser.add_argument('--conf-threshold', type=float, default=0.3, help='置信度阈值（默认0.3）')
    args = parser.parse_args()

    # 生成输出目录
    if args.output is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"log/detection_simple_{timestamp}"

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("简化版视频检测")
    print("="*70)
    print(f"输入视频: {args.video}")
    print(f"输出目录: {output_dir}")
    print(f"模型: {args.model}")
    print("="*70)
    print()

    # 初始化推理器
    inferencer = SimpleInference(
        model_path=args.model,
        device=args.device,
        conf_threshold=args.conf_threshold
    )

    # 打开视频
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"错误：无法打开视频 {args.video}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"视频信息: {frame_width}x{frame_height} @ {fps:.2f}fps, 总帧数: {total_frames}")
    print()

    # 视频写入
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(str(output_dir / "result.mp4"), fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    processing_count = 0

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
            mask_cable, mask_tape, metrics, outputs, pred_mask = inferencer.infer_frame(frame)

            # 打印进度
            if processing_count % 30 == 0:
                print(f"Frame {frame_count}/{total_frames}: Cable={metrics['cable_coverage']*100:.1f}%, Tape={metrics['tape_coverage']*100:.1f}%")

            # 创建叠加图
            overlay = inferencer.create_overlay(frame, mask_cable, mask_tape, metrics, outputs, pred_mask)
            output_video.write(overlay)

            # 显示预览
            if args.show_preview:
                display_height = 720
                h, w = overlay.shape[:2]
                scale = display_height / h
                display_width = int(w * scale)
                display = cv2.resize(overlay, (display_width, display_height))
                cv2.imshow('Simple Detection', display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("\n检测中断")

    finally:
        cap.release()
        output_video.release()
        cv2.destroyAllWindows()

    print()
    print("="*70)
    print("推理完成！")
    print(f"输出文件: {output_dir / 'result.mp4'}")
    print("="*70)


if __name__ == '__main__':
    main()
