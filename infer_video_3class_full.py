"""
3类最佳模型推理脚本 - 全画面检测（无固定ROI）
使用模型：checkpoints_3class_finetuned/best_model.pth
类别：背景(0)、电缆(1)、胶带(2)
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

# 添加目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

from src.models.unetpp import NestedUNet


# 类别配置
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


def thresholded_argmax(probs, t_cable=0.45, t_tape=0.50, bg_margin=0.15):
    """
    互斥阈值化argmax分割 - 防止背景误检

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


def keep_largest_cc(mask, min_area=500):
    """只保留最大连通域，去除噪声"""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels <= 1:
        return mask

    # 找到最大连通域（排除背景label 0）
    areas = stats[1:, cv2.CC_STAT_AREA]
    if len(areas) == 0 or np.max(areas) < min_area:
        return np.zeros_like(mask)

    largest_label = np.argmax(areas) + 1
    return (labels == largest_label).astype(np.uint8)


def select_primary_component(mask, min_area=1000, min_aspect=1.6):
    """Select a cable-like component: tall, near center, and large enough."""
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    h, w = mask.shape
    best_score = -1.0
    best_label = None

    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        aspect = float(bh) / max(1.0, float(bw))
        if aspect < min_aspect:
            continue

        cx = float(centroids[i][0])
        center_dist = abs(cx - (w * 0.5)) / max(1.0, float(w))
        score = area * aspect * (1.0 - center_dist)
        if score > best_score:
            best_score = score
            best_label = i

    if best_label is None:
        return np.zeros_like(mask)
    return (labels == best_label).astype(np.uint8)




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


@dataclass
class DetectionEvent:
    timestamp: str
    frame_id: int
    dc_px: float
    dt_px: float
    delta_d_px: float
    cable_coverage: float
    tape_coverage: float


class VideoInference3Class:
    """3类模型视频推理器 - 全画面版本"""

    def __init__(self,
                 model_path: str,
                 device: str = 'cuda',
                 conf_thresh_cable: float = 0.45,
                 conf_thresh_tape: float = 0.50,
                 bg_margin: float = 0.15,
                 use_cc_filter: bool = True,
                 cc_min_area_cable: int = 1000,
                 cc_min_area_tape: int = 500,
                 cable_min_aspect: float = 1.6,
                 tape_dilate_px: int = 15):
        """
        Args:
            model_path: 模型权重路径
            device: cuda或cpu
            conf_thresh_cable: 电缆置信度阈值
            conf_thresh_tape: 胶带置信度阈值
            use_cc_filter: 是否使用连通域过滤
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.conf_thresh_cable = conf_thresh_cable
        self.conf_thresh_tape = conf_thresh_tape
        self.bg_margin = bg_margin
        self.use_cc_filter = use_cc_filter
        self.cc_min_area_cable = cc_min_area_cable
        self.cc_min_area_tape = cc_min_area_tape
        self.cable_min_aspect = cable_min_aspect
        self.tape_dilate_px = tape_dilate_px

        print(f"使用设备: {self.device}")

        print(f"加载模型: {model_path}")
        self.model = NestedUNet(num_classes=3, deep_supervision=True).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()

        print(f"模型加载成功 (Epoch {checkpoint['epoch']+1}, mIoU {checkpoint['best_miou']:.2%})")
        print(f"后处理方式: 阈值化argmax (电缆阈值={conf_thresh_cable}, 胶带阈值={conf_thresh_tape})")
        if use_cc_filter:
            print("连通域过滤: 启用")

    def infer_frame(self, frame: np.ndarray):
        """
        推理单帧 - 全画面处理（不使用固定ROI）
        """
        h, w = frame.shape[:2]

        # 直接resize全画面到512x512
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

        # 计算概率并使用阈值化argmax（防止背景误检）
        probs = softmax_np(outputs[0].cpu().numpy().transpose(1, 2, 0))  # (H, W, C)
        mask_cable_small, mask_tape_small = thresholded_argmax(
            probs,
            t_cable=self.conf_thresh_cable,
            t_tape=self.conf_thresh_tape,
            bg_margin=self.bg_margin
        )

        # 连通域过滤（去除噪声）
        if self.use_cc_filter:
            mask_cable_small = select_primary_component(
                mask_cable_small,
                min_area=self.cc_min_area_cable,
                min_aspect=self.cable_min_aspect
            )
            if mask_cable_small.sum() > 0 and self.tape_dilate_px > 0:
                k = 2 * int(self.tape_dilate_px) + 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                tape_roi = cv2.dilate(mask_cable_small, kernel, iterations=1)
                mask_tape_small = (mask_tape_small & tape_roi).astype(np.uint8)
            mask_tape_small = keep_largest_cc(mask_tape_small, min_area=self.cc_min_area_tape)

        # 调整回原始尺寸
        mask_cable_full = cv2.resize(mask_cable_small, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_tape_full = cv2.resize(mask_tape_small, (w, h), interpolation=cv2.INTER_NEAREST)

        # 几何测量（使用512尺寸的mask）
        dc_px, dt_px, delta_d_px = measure_diameters_simple(mask_cable_small, mask_tape_small)

        # 计算覆盖率（使用512尺寸的mask）
        cable_coverage = mask_cable_small.sum() / mask_cable_small.size
        tape_coverage = mask_tape_small.sum() / mask_tape_small.size

        metrics = {
            'dc_px': dc_px,
            'dt_px': dt_px,
            'delta_d_px': delta_d_px,
            'cable_coverage': cable_coverage,
            'tape_coverage': tape_coverage
        }

        # 生成用于调试的pred_mask（仅显示）
        pred_mask = np.zeros_like(mask_cable_small, dtype=np.uint8)
        pred_mask[mask_cable_small > 0] = 1
        pred_mask[mask_tape_small > 0] = 2

        return mask_cable_full, mask_tape_full, metrics, probs, pred_mask

    def create_overlay(self, frame, mask_cable, mask_tape, metrics, outputs, pred_mask):
        """创建可视化叠加图 - 保持原始尺寸"""
        overlay = frame.copy()

        # 电缆 - 蓝色 BGR(255, 0, 0)
        cable_mask = mask_cable > 0
        overlay[cable_mask] = overlay[cable_mask] * 0.4 + np.array([255, 0, 0]) * 0.6

        # 胶带 - 绿色 BGR(0, 255, 0)
        tape_mask = mask_tape > 0
        overlay[tape_mask] = overlay[tape_mask] * 0.4 + np.array([0, 255, 0]) * 0.6

        # 计算类别分布（用于调试）
        unique, counts = np.unique(pred_mask, return_counts=True)
        class_dist = {int(u): int(c) for u, c in zip(unique, counts)}
        total = pred_mask.size

        # 添加文字信息
        y_offset = 30
        texts = [
            f"Cable(Blue): {metrics['dc_px']:.1f}px ({metrics['cable_coverage']*100:.1f}%)",
            f"Tape(Green): {metrics['dt_px']:.1f}px ({metrics['tape_coverage']*100:.1f}%)",
            f"Delta: {metrics['delta_d_px']:.1f}px",
            f"ClassDist: BG={class_dist.get(0,0)*100//total}% C1={class_dist.get(1,0)*100//total}% C2={class_dist.get(2,0)*100//total}%",
        ]

        for text in texts:
            cv2.putText(overlay, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += 25

        return overlay


def main():
    parser = argparse.ArgumentParser(description='3类模型全画面检测')
    parser.add_argument('--video', type=str, required=True, help='输入视频路径')
    parser.add_argument('--model', type=str, default='checkpoints_3class_finetuned/best_model.pth', help='模型路径')
    parser.add_argument('--output', type=str, default=None, help='输出目录')
    parser.add_argument('--device', type=str, default='cuda', help='设备 (cuda/cpu)')
    parser.add_argument('--frame-stride', type=int, default=1, help='帧采样间隔')
    parser.add_argument('--show-preview', action='store_true', help='显示预览窗口')
    parser.add_argument('--debug', action='store_true', help='显示调试信息')
    parser.add_argument('--conf-cable', type=float, default=0.45, help='电缆置信度阈值（防止背景误检）')
    parser.add_argument('--conf-tape', type=float, default=0.50, help='胶带置信度阈值（防止背景误检）')
    parser.add_argument('--no-cc-filter', action='store_true', help='禁用连通域过滤')
    parser.add_argument('--bg-margin', type=float, default=0.15, help='background margin threshold')
    parser.add_argument('--cc-min-area-cable', type=int, default=1000, help='min area for cable CC')
    parser.add_argument('--cc-min-area-tape', type=int, default=500, help='min area for tape CC')
    parser.add_argument('--cable-min-aspect', type=float, default=1.6, help='min H/W for cable CC')
    parser.add_argument('--tape-dilate-px', type=int, default=15, help='dilate cable mask to keep nearby tape')
    args = parser.parse_args()

    # 生成输出目录
    if args.output is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"log/detection_3class_full_{timestamp}"

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("3类模型检测 - 全画面模式（已优化背景误检）")
    print("="*70)
    print(f"输入视频: {args.video}")
    print(f"输出目录: {output_dir}")
    print(f"模型: {args.model}")
    print(f"检测模式: 全画面处理")
    print(f"置信度阈值: 电缆={args.conf_cable}, 胶带={args.conf_tape}")
    print(f"连通域过滤: {'禁用' if args.no_cc_filter else '启用'}")
    print("="*70)
    print()

    # 初始化推理器
    inferencer = VideoInference3Class(
        model_path=args.model,
        device=args.device,
        conf_thresh_cable=args.conf_cable,
        conf_thresh_tape=args.conf_tape,
        bg_margin=args.bg_margin,
        use_cc_filter=not args.no_cc_filter,
        cc_min_area_cable=args.cc_min_area_cable,
        cc_min_area_tape=args.cc_min_area_tape,
        cable_min_aspect=args.cable_min_aspect,
        tape_dilate_px=args.tape_dilate_px
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

    # 视频写入 - 保持原始尺寸和帧率
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
            mask_cable, mask_tape, metrics, outputs, pred_mask = inferencer.infer_frame(frame)

            # 调试信息
            if args.debug and processing_count % 30 == 0:
                unique, counts = np.unique(pred_mask, return_counts=True)
                class_dist = {int(u): int(c) for u, c in zip(unique, counts)}
                print(f"[Frame {frame_count}] Cable={metrics['cable_coverage']*100:.1f}%, Tape={metrics['tape_coverage']*100:.1f}%")
                print(f"  ClassDist: {class_dist}")
                print(f"  Diameters: Dc={metrics['dc_px']:.1f}px, Dt={metrics['dt_px']:.1f}px")

            # 记录统计
            if metrics['dc_px'] > 0:
                all_dc.append(metrics['dc_px'])
                all_dt.append(metrics['dt_px'])
                all_delta_d.append(metrics['delta_d_px'])

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
            overlay = inferencer.create_overlay(frame, mask_cable, mask_tape, metrics, outputs, pred_mask)
            output_video.write(overlay)

            # 显示预览
            if args.show_preview:
                display_height = 720
                h, w = overlay.shape[:2]
                scale = display_height / h
                display_width = int(w * scale)
                display = cv2.resize(overlay, (display_width, display_height))
                cv2.imshow('Detection (Full Frame)', display)
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

        with open(output_dir / "statistics.json", 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

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
    main()
