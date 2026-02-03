"""
基于空间过滤的视频检测 - 解决严重域偏移问题
"""
import cv2
import numpy as np
import torch
import sys
from pathlib import Path
import argparse
from datetime import datetime

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

from src.models.unetpp import NestedUNet


def softmax_np(x):
    """Numpy softmax"""
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)


def spatial_filter(mask, min_width=50, max_width=300, min_height_ratio=0.3):
    """
    空间过滤 - 电缆/胶带通常是垂直条状

    Args:
        mask: 二值mask
        min_width: 最小宽度
        max_width: 最大宽度
        min_height_ratio: 最小高度占画面比例
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels <= 1:
        return mask

    h, w = mask.shape
    filtered = np.zeros_like(mask)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]

        # 必须满足：足够大、不太宽、高度足够
        if (area > 1000 and
            min_width <= width <= max_width and
            height >= h * min_height_ratio):
            filtered[labels == i] = 1

    return filtered


def create_vertical_focus_region(h, w):
    """
    创建垂直聚焦区域 - 电缆/胶带通常在画面中央垂直条带
    """
    # 创建中央垂直条带mask
    focus_mask = np.zeros((h, w), dtype=np.uint8)

    # 中央50%区域
    x_start = int(w * 0.25)
    x_end = int(w * 0.75)
    focus_mask[:, x_start:x_end] = 1

    return focus_mask


def relative_threshold(probs, cable_bg_ratio=2.0, tape_bg_ratio=2.5):
    """
    基于相对概率的阈值 - 解决域偏移

    不使用绝对概率，而是使用"相对于背景的优势"
    """
    bg = probs[..., 0]
    cable = probs[..., 1]
    tape = probs[..., 2]

    # 电缆：电缆概率 > 背景概率 * ratio
    mask_cable = (cable > bg * cable_bg_ratio).astype(np.uint8)

    # 胶带：胶带概率 > 背景概率 * ratio
    mask_tape = (tape > bg * tape_bg_ratio).astype(np.uint8)

    # 确保互斥
    overlap = mask_cable & mask_tape
    if overlap.any():
        # 重叠区域取概率更高的
        cable_prob = cable[overlap]
        tape_prob = tape[overlap]
        cable_wins = cable_prob >= tape_prob

        mask_cable[overlap] = cable_wins.astype(np.uint8)
        mask_tape[overlap] = (~cable_wins).astype(np.uint8)

    return mask_cable, mask_tape


class SpatialFilterInference:
    """基于空间过滤的推理器"""

    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        print(f"使用设备: {self.device}")
        print(f"加载模型: {model_path}")

        self.model = NestedUNet(num_classes=3, deep_supervision=True).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()

        print(f"模型加载成功 (Epoch {checkpoint['epoch']+1}, mIoU {checkpoint['best_miou']:.2%})")
        print("="*70)
        print("空间过滤模式（解决严重域偏移）:")
        print("  基于相对概率而非绝对概率")
        print("  垂直条带形状约束")
        print("  中央区域聚焦")
        print("="*70)

    def infer_frame(self, frame: np.ndarray):
        """推理单帧"""
        h, w = frame.shape[:2]

        # 预处理
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(frame_rgb, (512, 512))

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

        # 方法1：基于相对概率的阈值
        mask_cable_small, mask_tape_small = relative_threshold(probs)

        # 方法2：空间过滤
        mask_cable_small = spatial_filter(mask_cable_small, min_width=30, max_width=200)
        mask_tape_small = spatial_filter(mask_tape_small, min_width=20, max_width=150)

        # 方法3：聚焦中央区域
        focus_mask = create_vertical_focus_region(512, 512)

        # 只保留中央区域的检测
        mask_cable_small = mask_cable_small & focus_mask
        mask_tape_small = mask_tape_small & focus_mask

        # 调整回原始尺寸
        mask_cable_full = cv2.resize(mask_cable_small, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_tape_full = cv2.resize(mask_tape_small, (w, h), interpolation=cv2.INTER_NEAREST)

        # 统计
        cable_coverage = mask_cable_small.sum() / mask_cable_small.size
        tape_coverage = mask_tape_small.sum() / mask_tape_small.size

        metrics = {
            'cable_coverage': cable_coverage,
            'tape_coverage': tape_coverage
        }

        return mask_cable_full, mask_tape_full, metrics, probs

    def create_overlay(self, frame, mask_cable, mask_tape, metrics, probs):
        """创建可视化叠加图"""
        overlay = frame.copy()

        # 电缆 - 蓝色
        cable_mask = mask_cable > 0
        overlay[cable_mask] = overlay[cable_mask] * 0.6 + np.array([255, 0, 0]) * 0.4

        # 胶带 - 绿色
        tape_mask = mask_tape > 0
        overlay[tape_mask] = overlay[tape_mask] * 0.6 + np.array([0, 255, 0]) * 0.4

        # 绘制中央聚焦区域
        h, w = overlay.shape[:2]
        x_start = int(w * 0.25)
        x_end = int(w * 0.75)
        cv2.rectangle(overlay, (x_start, 0), (x_end, h), (0, 255, 255), 1)

        # 添加文字
        y_offset = 30
        texts = [
            f"Cable: {metrics['cable_coverage']*100:.1f}%",
            f"Tape: {metrics['tape_coverage']*100:.1f}%",
        ]

        for text in texts:
            cv2.putText(overlay, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += 25

        return overlay


def main():
    parser = argparse.ArgumentParser(description='空间过滤视频检测')
    parser.add_argument('--video', type=str, required=True, help='输入视频路径')
    parser.add_argument('--model', type=str, default='checkpoints_3class_finetuned/best_model.pth')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--frame-stride', type=int, default=1)
    parser.add_argument('--show-preview', action='store_true')
    args = parser.parse_args()

    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"log/detection_spatial_{timestamp}"

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("基于空间过滤的视频检测")
    print("="*70)
    print(f"输入视频: {args.video}")
    print(f"输出目录: {output_dir}")
    print("="*70)
    print()

    inferencer = SpatialFilterInference(
        model_path=args.model,
        device=args.device
    )

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"错误：无法打开视频 {args.video}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"视频信息: {frame_width}x{frame_height} @ {fps:.2f}fps")
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

            mask_cable, mask_tape, metrics, probs = inferencer.infer_frame(frame)

            if processing_count % 30 == 0:
                print(f"Frame {frame_count}/{total_frames}: Cable={metrics['cable_coverage']*100:.1f}%, Tape={metrics['tape_coverage']*100:.1f}%")

            overlay = inferencer.create_overlay(frame, mask_cable, mask_tape, metrics, probs)
            output_video.write(overlay)

            if args.show_preview:
                display = cv2.resize(overlay, (1024, 576))
                cv2.imshow('Spatial Filter Detection', display)
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
