"""
简化版推理脚本 - 紧急备用版本

如果 infer_video.py 的 ROI/letterbox 版本出现严重问题，
可以使用这个简化版本，只做最基础的全图推理。
"""
import os
import cv2
import argparse
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
import sys
from pathlib import Path

# 添加目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root))

from src.models.unetpp import NestedUNet

# 类别颜色
CLASS_COLORS = {
    0: (0, 0, 0),
    1: (255, 0, 0),
    2: (0, 255, 0),
    3: (0, 0, 255),
    4: (255, 255, 0),
    5: (255, 0, 255),
    6: (0, 165, 255),
}


class SimpleInference:
    """简化推理器 - 全图推理，无复杂处理"""

    def __init__(self, model_path: str, num_classes: int = 7, device: str = 'cpu'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        self.input_size = 256

        print(f"加载模型: {model_path}")
        self.model = NestedUNet(
            num_classes=num_classes,
            input_channels=3,
            deep_supervision=False
        ).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()
        print("模型加载完成")

    @torch.no_grad()
    def predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        """简单预测：全图 resize，无 ROI"""
        # 预处理
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (self.input_size, self.input_size))
        frame_normalized = frame_resized.astype(np.float32) / 255.0
        frame_tensor = torch.from_numpy(frame_normalized).permute(2, 0, 1).unsqueeze(0).to(self.device)

        # 推理
        output = self.model(frame_tensor)
        if isinstance(output, list):
            output = output[0]

        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

        # resize 回原始尺寸
        h, w = frame_bgr.shape[:2]
        pred_mask = cv2.resize(pred_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

        # 轻微后处理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        for cid in [1, 2]:
            m = (pred_mask == cid).astype(np.uint8)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)
            pred_mask[m > 0] = cid

        return pred_mask

    def overlay_mask(self, frame_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """简单叠加"""
        h, w = frame_bgr.shape[:2]
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)

        for class_id, color in CLASS_COLORS.items():
            if class_id >= self.num_classes:
                continue
            class_mask = (mask == class_id).astype(np.uint8)
            color_mask[class_mask > 0] = color

        # 只混合 mask 区域
        result = frame_bgr.copy()
        region = (mask > 0)
        if np.any(region):
            blended = ((1 - alpha) * frame_bgr.astype(np.float32) + alpha * color_mask.astype(np.float32)).astype(np.uint8)
            result[region] = blended[region]

        return result


def main():
    parser = argparse.ArgumentParser(description='简化版电缆检测（备用）')
    parser.add_argument('--video', type=str, required=True, help='视频路径')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pth', help='模型路径')
    parser.add_argument('--output', type=str, default='log/simple_backup', help='输出目录')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--show-preview', action='store_true', help='显示预览')

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, 'overlays'), exist_ok=True)

    # 初始化
    inferencer = SimpleInference(args.model, device=args.device)

    # 打开视频
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"无法打开视频: {args.video}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频: {total_frames}帧 @ {fps:.2f}fps")
    print("开始处理...")

    frame_idx = 0
    save_interval = 30  # 每30帧保存一次

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # 预测
        mask = inferencer.predict(frame)

        # 统计
        counts = {cid: int((mask == cid).sum()) for cid in range(7)}
        if frame_idx % 10 == 0:
            print(f"[frame {frame_idx}] {counts}")

        # 定期保存
        if frame_idx % save_interval == 0:
            overlay = inferencer.overlay_mask(frame, mask, alpha=0.6)
            overlay_path = os.path.join(args.output, 'overlays', f'frame_{frame_idx}.jpg')
            cv2.imwrite(overlay_path, overlay)

        # 预览
        if args.show_preview and frame_idx % 3 == 0:
            overlay = inferencer.overlay_mask(frame, mask, alpha=0.6)
            display = cv2.resize(overlay, (1024, 768))
            cv2.imshow('Simple Inference', display)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n完成！结果保存在: {args.output}")


if __name__ == '__main__':
    main()
