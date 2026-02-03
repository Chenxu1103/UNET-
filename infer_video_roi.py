"""
基于ROI的视频检测 - 先定位电缆/胶带区域，再进行分割
"""
import cv2
import numpy as np
import torch
import sys
from pathlib import Path
import argparse

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

from src.models.unetpp import NestedUNet


def softmax_np(x):
    """Numpy softmax"""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True)
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def detect_roi_by_projection(frame_rgb):
    """
    基于垂直投影检测ROI
    电缆/胶带通常呈垂直分布，左右两侧是背景
    """
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

    # 边缘检测
    edges = cv2.Canny(gray, 50, 150)

    # 垂直投影（统计每列的边缘像素数）
    vertical_proj = np.sum(edges, axis=0)

    # 平滑投影
    kernel = np.ones(30) / 30
    vertical_proj_smooth = np.convolve(vertical_proj, kernel, mode='same')

    # 找到投影值较大的区域（可能是电缆/胶带）
    threshold = np.max(vertical_proj_smooth) * 0.3
    significant_cols = np.where(vertical_proj_smooth > threshold)[0]

    if len(significant_cols) > 0:
        x_min = int(significant_cols[0] * (frame_rgb.shape[1] / 512))
        x_max = int(significant_cols[-1] * (frame_rgb.shape[1] / 512))

        # 扩展边界
        margin = int((x_max - x_min) * 0.1)
        x_min = max(0, x_min - margin)
        x_max = min(frame_rgb.shape[1], x_max + margin)

        return x_min, x_max
    else:
        # 如果检测失败，使用中心区域
        w = frame_rgb.shape[1]
        return int(w * 0.25), int(w * 0.75)


def adaptive_thresholding(probs):
    """
    自适应阈值 - 根据整体置信度分布调整阈值

    如果整个画面的置信度都很高（域偏移），使用更高的阈值
    """
    cable_probs = probs[..., 1]
    tape_probs = probs[..., 2]

    # 计算统计信息
    cable_mean = cable_probs.mean()
    cable_max = cable_probs.max()
    tape_mean = tape_probs.mean()
    tape_max = tape_probs.max()

    print(f"    自适应阈值分析:")
    print(f"      电缆: mean={cable_mean:.3f}, max={cable_max:.3f}")
    print(f"      胶带: mean={tape_mean:.3f}, max={tape_max:.3f}")

    # 如果平均置信度很高（>0.3），说明域偏移严重
    if cable_mean > 0.3:
        t_cable = min(0.85, cable_mean + 0.4)  # 动态提高阈值
        print(f"      -> 域偏移检测：电缆阈值提高到 {t_cable:.3f}")
    else:
        t_cable = 0.5

    if tape_mean > 0.15:
        t_tape = min(0.85, tape_mean + 0.5)
        print(f"      -> 域偏移检测：胶带阈值提高到 {t_tape:.3f}")
    else:
        t_tape = 0.55

    # 背景margin也要根据背景概率调整
    bg_mean = probs[..., 0].mean()
    bg_margin = max(0.2, 1.0 - bg_mean)  # 背景越弱，margin越严格
    print(f"      背景margin: {bg_margin:.3f}")

    return t_cable, t_tape, bg_margin


def ultra_strict_threshold(probs, t_cable, t_tape, bg_margin):
    """
    超严格阈值 - 极力压制背景误检
    """
    bg = probs[..., 0]
    cable = probs[..., 1]
    tape = probs[..., 2]

    winner = np.argmax(probs[..., :3], axis=-1)

    # 超严格条件
    mask_cable = (
        (winner == 1) &
        (cable >= t_cable) &
        (cable > bg * 2) &  # 电缆置信度必须是背景的2倍以上
        (cable >= bg + bg_margin)
    )

    mask_tape = (
        (winner == 2) &
        (tape >= t_tape) &
        (tape > bg * 2) &  # 胶带置信度必须是背景的2倍以上
        (tape >= bg + bg_margin)
    )

    return mask_cable.astype(np.uint8), mask_tape.astype(np.uint8)


def refine_mask_by_geometry(mask, expected_aspect_ratio=3.0):
    """
    基于几何特征优化mask
    电缆/胶带通常高宽比很大（垂直条状）
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels <= 1:
        return mask

    filtered_mask = np.zeros_like(mask)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        # 过滤太小的区域
        if area < 2000:
            continue

        # 检查高宽比（电缆/胶带应该是垂直条状）
        if w > 0:
            aspect_ratio = h / w
            # 期望高宽比大于2，或者宽度很小
            if aspect_ratio < 2.0 and w > 100:
                continue

        # 检查位置（应该在画面中央区域，不太可能在边缘）
        cx = int(centroids[i][0])
        cy = int(centroids[i][1])
        img_w, img_h = mask.shape[1], mask.shape[0]
        if cx < img_w * 0.1 or cx > img_w * 0.9:
            # 靠近左右边缘的可能是背景
            if area < 10000:  # 除非区域很大
                continue

        filtered_mask[labels == i] = 1

    return filtered_mask


class ROIVideoInference:
    """基于ROI的视频推理器"""

    def __init__(self, model_path: str, device: str = 'cuda', use_roi: bool = True):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.use_roi = use_roi

        print(f"使用设备: {self.device}")

        print(f"加载模型: {model_path}")
        self.model = NestedUNet(num_classes=3, deep_supervision=True).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()

        print(f"模型加载成功 (Epoch {checkpoint['epoch']+1}, mIoU {checkpoint['best_miou']:.2%})")
        print("="*70)
        print("ROI检测模式（先定位区域，再分割）:")
        print(f"  ROI检测: {'启用' if use_roi else '禁用'}")
        print(f"  自适应阈值: 启用")
        print(f"  几何过滤: 启用")
        print("="*70)

    def infer_frame(self, frame: np.ndarray):
        """推理单帧"""
        h, w = frame.shape[:2]

        # BGR -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 检测ROI
        if self.use_roi:
            x_min, x_max = detect_roi_by_projection(frame_rgb)
            roi_crop = frame_rgb[:, x_min:x_max]
        else:
            roi_crop = frame_rgb
            x_min, x_max = 0, w

        # Resize到512x512
        resized = cv2.resize(roi_crop, (512, 512))

        # 归一化
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

        # 自适应阈值
        t_cable, t_tape, bg_margin = adaptive_thresholding(probs)

        # 超严格阈值
        mask_cable_small, mask_tape_small = ultra_strict_threshold(
            probs, t_cable, t_tape, bg_margin
        )

        # 几何优化
        mask_cable_small = refine_mask_by_geometry(mask_cable_small)
        mask_tape_small = refine_mask_by_geometry(mask_tape_small)

        # 调整回ROI尺寸
        roi_h, roi_w = roi_crop.shape[:2]
        mask_cable_roi = cv2.resize(mask_cable_small, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
        mask_tape_roi = cv2.resize(mask_tape_small, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)

        # 放回全图
        mask_cable_full = np.zeros((h, w), dtype=np.uint8)
        mask_tape_full = np.zeros((h, w), dtype=np.uint8)

        mask_cable_full[:, x_min:x_max] = mask_cable_roi
        mask_tape_full[:, x_min:x_max] = mask_tape_roi

        # 统计
        cable_coverage = mask_cable_small.sum() / mask_cable_small.size
        tape_coverage = mask_tape_small.sum() / mask_tape_small.size

        metrics = {
            'cable_coverage': cable_coverage,
            'tape_coverage': tape_coverage,
            'roi': (x_min, x_max)
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

        # 绘制ROI边界
        if 'roi' in metrics:
            x_min, x_max = metrics['roi']
            cv2.rectangle(overlay, (x_min, 0), (x_max, overlay.shape[0]), (0, 255, 255), 2)

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
    parser = argparse.ArgumentParser(description='基于ROI的视频检测')
    parser.add_argument('--video', type=str, required=True, help='输入视频路径')
    parser.add_argument('--model', type=str, default='checkpoints_3class_finetuned/best_model.pth', help='模型路径')
    parser.add_argument('--output', type=str, default=None, help='输出目录')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--frame-stride', type=int, default=1, help='帧采样间隔')
    parser.add_argument('--show-preview', action='store_true', help='显示预览窗口')
    parser.add_argument('--no-roi', action='store_true', help='禁用ROI检测')
    args = parser.parse_args()

    if args.output is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"log/detection_roi_{timestamp}"

    from pathlib import Path
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("基于ROI的视频检测")
    print("="*70)
    print(f"输入视频: {args.video}")
    print(f"输出目录: {output_dir}")
    print("="*70)
    print()

    inferencer = ROIVideoInference(
        model_path=args.model,
        device=args.device,
        use_roi=not args.no_roi
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

            mask_cable, mask_tape, metrics, probs = inferencer.infer_frame(frame)

            if processing_count % 30 == 0:
                print(f"Frame {frame_count}/{total_frames}: Cable={metrics['cable_coverage']*100:.1f}%, Tape={metrics['tape_coverage']*100:.1f}%")

            overlay = inferencer.create_overlay(frame, mask_cable, mask_tape, metrics, probs)
            output_video.write(overlay)

            if args.show_preview:
                display_height = 720
                h, w = overlay.shape[:2]
                scale = display_height / h
                display_width = int(w * scale)
                display = cv2.resize(overlay, (display_width, display_height))
                cv2.imshow('ROI Detection', display)
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
