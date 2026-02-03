"""
修复版视频检测 - 解决背景误检和特征混合问题
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


def strict_threshold_with_bg_check(probs, t_cable=0.6, t_tape=0.65, bg_margin=0.4):
    """
    严格阈值 + 强制背景检查 - 解决背景误检问题

    Args:
        probs: (H, W, 3) 概率图
        t_cable: 电缆阈值（更高）
        t_tape: 胶带阈值（更高）
        bg_margin: 背景margin（更大，确保背景真正是低概率）

    Returns:
        mask_cable, mask_tape
    """
    bg = probs[..., 0]
    cable = probs[..., 1]
    tape = probs[..., 2]

    # 第一步：argmax得到winner
    winner = np.argmax(probs[..., :3], axis=-1)

    # 第二步：严格阈值 + 背景margin检查
    # 只有同时满足：winner是该类别 AND 置信度高 AND 背景概率足够低
    mask_cable = (
        (winner == 1) &
        (cable >= t_cable) &
        (bg <= bg_margin)
    )

    mask_tape = (
        (winner == 2) &
        (tape >= t_tape) &
        (bg <= bg_margin)
    )

    # 第三步：确保互斥（电缆和胶带区域不重叠）
    # 如果有重叠，保留置信度更高的
    overlap = mask_cable & mask_tape
    if overlap.any():
        cable_on_overlap = cable[overlap]
        tape_on_overlap = tape[overlap]
        cable_wins = cable_on_overlap >= tape_on_overlap

        # 电缆胜出的区域，去掉胶带
        mask_tape[overlap] = np.where(cable_wins, 0, mask_tape[overlap])
        # 胶带胜出的区域，去掉电缆
        mask_cable[overlap] = np.where(~cable_wins, 0, mask_cable[overlap])

    return mask_cable.astype(np.uint8), mask_tape.astype(np.uint8)


def filter_by_size_and_shape(mask, min_area=2000, max_area=100000, min_circularity=0.0):
    """
    过滤掉太小或太大或形状不合理的区域
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels <= 1:
        return mask

    filtered_mask = np.zeros_like(mask)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        # 过滤太小或太大的区域
        if area < min_area or area > max_area:
            continue

        filtered_mask[labels == i] = 1

    return filtered_mask


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


class FixedVideoInference:
    """修复版推理器 - 解决背景误检和特征混合"""

    def __init__(self,
                 model_path: str,
                 device: str = 'cuda',
                 conf_cable: float = 0.6,
                 conf_tape: float = 0.65,
                 bg_margin: float = 0.4,
                 min_area_cable: int = 3000,
                 min_area_tape: int = 1500):
        """
        Args:
            conf_cable: 电缆置信度阈值（更高，0.6）
            conf_tape: 胶带置信度阈值（更高，0.65）
            bg_margin: 背景margin（更大，0.4）
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.conf_cable = conf_cable
        self.conf_tape = conf_tape
        self.bg_margin = bg_margin
        self.min_area_cable = min_area_cable
        self.min_area_tape = min_area_tape

        print(f"使用设备: {self.device}")

        print(f"加载模型: {model_path}")
        self.model = NestedUNet(num_classes=3, deep_supervision=True).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()

        print(f"模型加载成功 (Epoch {checkpoint['epoch']+1}, mIoU {checkpoint['best_miou']:.2%})")
        print("="*70)
        print("修复版检测（解决背景误检和特征混合）:")
        print(f"  电缆阈值: {conf_cable}")
        print(f"  胶带阈值: {conf_tape}")
        print(f"  背景margin: {bg_margin}")
        print(f"  电缆最小面积: {min_area_cable}")
        print(f"  胶带最小面积: {min_area_tape}")
        print("="*70)

    def infer_frame(self, frame: np.ndarray):
        """推理单帧"""
        h, w = frame.shape[:2]

        # BGR -> RGB（保持原始亮度）
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(frame_rgb, (512, 512))

        # 归一化到[0,1]
        img_tensor = resized.astype(np.float32) / 255.0
        img_tensor = np.transpose(img_tensor, (2, 0, 1))
        img_tensor = torch.from_numpy(img_tensor).unsqueeze(0).to(self.device)

        # 推理
        with torch.no_grad():
            outputs = self.model(img_tensor)
            if isinstance(outputs, list):
                outputs = outputs[-1]

        # 计算概率
        probs = softmax_np(outputs[0].cpu().numpy().transpose(1, 2, 0))

        # 严格阈值 + 背景检查 + 互斥处理
        mask_cable_small, mask_tape_small = strict_threshold_with_bg_check(
            probs,
            t_cable=self.conf_cable,
            t_tape=self.conf_tape,
            bg_margin=self.bg_margin
        )

        # 大小过滤
        mask_cable_small = filter_by_size_and_shape(
            mask_cable_small,
            min_area=self.min_area_cable,
            max_area=100000
        )
        mask_tape_small = filter_by_size_and_shape(
            mask_tape_small,
            min_area=self.min_area_tape,
            max_area=80000
        )

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

        # 调试信息
        pred_mask = np.zeros_like(mask_cable_small, dtype=np.uint8)
        pred_mask[mask_cable_small > 0] = 1
        pred_mask[mask_tape_small > 0] = 2

        return mask_cable_full, mask_tape_full, metrics, probs, pred_mask

    def create_overlay(self, frame, mask_cable, mask_tape, metrics, probs, pred_mask):
        """创建可视化叠加图 - 保持原始亮度"""
        overlay = frame.copy()

        # 电缆 - 蓝色 BGR(255, 0, 0)
        cable_mask = mask_cable > 0
        overlay[cable_mask] = overlay[cable_mask] * 0.6 + np.array([255, 0, 0]) * 0.4

        # 胶带 - 绿色 BGR(0, 255, 0)
        tape_mask = mask_tape > 0
        overlay[tape_mask] = overlay[tape_mask] * 0.6 + np.array([0, 255, 0]) * 0.4

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
    parser = argparse.ArgumentParser(description='修复版视频检测')
    parser.add_argument('--video', type=str, required=True, help='输入视频路径')
    parser.add_argument('--model', type=str, default='checkpoints_3class_finetuned/best_model.pth', help='模型路径')
    parser.add_argument('--output', type=str, default=None, help='输出目录')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--frame-stride', type=int, default=1, help='帧采样间隔')
    parser.add_argument('--show-preview', action='store_true', help='显示预览窗口')
    parser.add_argument('--conf-cable', type=float, default=0.6, help='电缆阈值')
    parser.add_argument('--conf-tape', type=float, default=0.65, help='胶带阈值')
    parser.add_argument('--bg-margin', type=float, default=0.4, help='背景margin')
    args = parser.parse_args()

    if args.output is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"log/detection_fixed_{timestamp}"

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("修复版视频检测")
    print("="*70)
    print(f"输入视频: {args.video}")
    print(f"输出目录: {output_dir}")
    print(f"模型: {args.model}")
    print("="*70)
    print()

    inferencer = FixedVideoInference(
        model_path=args.model,
        device=args.device,
        conf_cable=args.conf_cable,
        conf_tape=args.conf_tape,
        bg_margin=args.bg_margin
    )

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

            if frame_count % args.frame_stride != 0:
                continue

            processing_count += 1

            mask_cable, mask_tape, metrics, probs, pred_mask = inferencer.infer_frame(frame)

            if processing_count % 30 == 0:
                print(f"Frame {frame_count}/{total_frames}: Cable={metrics['cable_coverage']*100:.1f}%, Tape={metrics['tape_coverage']*100:.1f}%")

            overlay = inferencer.create_overlay(frame, mask_cable, mask_tape, metrics, probs, pred_mask)
            output_video.write(overlay)

            if args.show_preview:
                display_height = 720
                h, w = overlay.shape[:2]
                scale = display_height / h
                display_width = int(w * scale)
                display = cv2.resize(overlay, (display_width, display_height))
                cv2.imshow('Fixed Detection', display)
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
