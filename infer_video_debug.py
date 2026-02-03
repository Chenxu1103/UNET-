"""
调试版视频检测 - 去除所有过滤，只保留核心功能
用于诊断模型本身是否正常工作
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


def softmax_np(x):
    """Numpy softmax"""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


class DebugVideoInference:
    """调试推理器 - 和训练时可视化完全一致"""

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
        print("后处理方式: 直接argmax（和训练可视化脚本一致）")
        print("过滤: 无（全部禁用）")
        print("="*70)

    def infer_frame(self, frame: np.ndarray):
        """
        推理单帧 - 完全按照训练时的方式
        """
        h, w = frame.shape[:2]

        # BGR -> RGB（和dataset.py第80行一致）
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(frame_rgb, (512, 512))

        # 归一化到[0,1]（和dataset.py第95行一致）
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

        # 直接使用argmax（和训练可视化第122行一致）
        pred_mask = np.argmax(probs, axis=-1)

        # 转换为二值mask
        mask_cable_small = (pred_mask == 1).astype(np.uint8)
        mask_tape_small = (pred_mask == 2).astype(np.uint8)

        # 调整回原始尺寸
        mask_cable_full = cv2.resize(mask_cable_small, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_tape_full = cv2.resize(mask_tape_small, (w, h), interpolation=cv2.INTER_NEAREST)

        # 计算统计信息
        cable_pixels = mask_cable_small.sum()
        tape_pixels = mask_tape_small.sum()
        total_pixels = pred_mask.size

        metrics = {
            'cable_coverage': cable_pixels / total_pixels,
            'tape_coverage': tape_pixels / total_pixels,
            'bg_coverage': (total_pixels - cable_pixels - tape_pixels) / total_pixels,
            'cable_prob_mean': probs[..., 1].mean(),
            'cable_prob_max': probs[..., 1].max(),
            'tape_prob_mean': probs[..., 2].mean(),
            'tape_prob_max': probs[..., 2].max(),
            'bg_prob_mean': probs[..., 0].mean(),
        }

        return mask_cable_full, mask_tape_full, metrics, probs, pred_mask

    def create_overlay(self, frame, mask_cable, mask_tape, metrics, pred_mask):
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
            f"Bg: {metrics['bg_coverage']*100:.1f}% (prob={metrics['bg_prob_mean']:.3f})",
            f"Cable: {metrics['cable_coverage']*100:.1f}% (mean={metrics['cable_prob_mean']:.3f}, max={metrics['cable_prob_max']:.3f})",
            f"Tape: {metrics['tape_coverage']*100:.1f}% (mean={metrics['tape_prob_mean']:.3f}, max={metrics['tape_prob_max']:.3f})",
        ]

        for text in texts:
            cv2.putText(overlay, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += 25

        return overlay


def main():
    parser = argparse.ArgumentParser(description='调试版视频检测')
    parser.add_argument('--video', type=str, required=True, help='输入视频路径')
    parser.add_argument('--model', type=str, default='checkpoints_3class_finetuned/best_model.pth', help='模型路径')
    parser.add_argument('--output', type=str, default=None, help='输出目录')
    parser.add_argument('--device', type=str, default='cuda', help='设备 (cuda/cpu)')
    parser.add_argument('--frame-stride', type=int, default=1, help='帧采样间隔')
    parser.add_argument('--show-preview', action='store_true', help='显示预览窗口')
    parser.add_argument('--save-frames', action='store_true', help='保存前几帧的原始预测图')
    args = parser.parse_args()

    # 生成输出目录
    if args.output is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"log/detection_debug_{timestamp}"

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("调试版视频检测")
    print("="*70)
    print(f"输入视频: {args.video}")
    print(f"输出目录: {output_dir}")
    print(f"模型: {args.model}")
    print("="*70)
    print()

    # 初始化推理器
    inferencer = DebugVideoInference(
        model_path=args.model,
        device=args.device
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

            if frame_count % args.frame_stride != 0:
                continue

            processing_count += 1

            # 推理
            mask_cable, mask_tape, metrics, probs, pred_mask = inferencer.infer_frame(frame)

            # 调试：前10帧打印详细信息
            if processing_count <= 10 or processing_count % 30 == 0:
                print(f"\nFrame {frame_count}/{total_frames}:")
                print(f"  背景: {metrics['bg_coverage']*100:.1f}% (平均概率={metrics['bg_prob_mean']:.3f})")
                print(f"  电缆: {metrics['cable_coverage']*100:.1f}% (平均概率={metrics['cable_prob_mean']:.3f}, 最大={metrics['cable_prob_max']:.3f})")
                print(f"  胶带: {metrics['tape_coverage']*100:.1f}% (平均概率={metrics['tape_prob_mean']:.3f}, 最大={metrics['tape_prob_max']:.3f})")

                # 保存前3帧的原始预测图用于调试
                if args.save_frames and processing_count <= 3:
                    # 保存概率图可视化
                    prob_viz = np.zeros((512, 512, 3), dtype=np.uint8)
                    prob_viz[:, :, 0] = (probs[..., 1] * 255).astype(np.uint8)  # R=电缆
                    prob_viz[:, :, 1] = (probs[..., 2] * 255).astype(np.uint8)  # G=胶带
                    prob_viz[:, :, 2] = (probs[..., 0] * 255).astype(np.uint8)  # B=背景
                    cv2.imwrite(str(output_dir / f"frame_{processing_count}_prob.png"), prob_viz)

                    # 保存argmax结果
                    pred_viz = np.zeros((512, 512, 3), dtype=np.uint8)
                    pred_viz[pred_mask == 1] = [255, 0, 0]  # 电缆=蓝色
                    pred_viz[pred_mask == 2] = [0, 255, 0]  # 胶带=绿色
                    cv2.imwrite(str(output_dir / f"frame_{processing_count}_pred.png"), cv2.cvtColor(pred_viz, cv2.COLOR_RGB2BGR))

                    print(f"  已保存调试图: frame_{processing_count}_*.png")

            # 创建叠加图
            overlay = inferencer.create_overlay(frame, mask_cable, mask_tape, metrics, pred_mask)
            output_video.write(overlay)

            # 显示预览
            if args.show_preview:
                display_height = 720
                h, w = overlay.shape[:2]
                scale = display_height / h
                display_width = int(w * scale)
                display = cv2.resize(overlay, (display_width, display_height))
                cv2.imshow('Debug Detection', display)
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
    print(f"输出文件:")
    print(f"  视频结果: {output_dir / 'result.mp4'}")
    if args.save_frames:
        print(f"  调试图像: {output_dir / 'frame_*.png'}")
    print("="*70)


if __name__ == '__main__':
    main()
