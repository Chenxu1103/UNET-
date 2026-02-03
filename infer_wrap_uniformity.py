"""胶带缠绕均匀性检测

基于3类模型（cable + tape + burr）的缠绕均匀性检测

检测逻辑：
1. 使用3类模型分割cable和tape区域
2. 计算cable和tape的直径
3. 计算tape/cable直径比例
4. 检测比例异常（过薄/过厚）
5. 统计缠绕均匀性
"""
import os
import cv2
import argparse
import numpy as np
import torch
import sys
from pathlib import Path
from tqdm import tqdm
from collections import deque

sys.path.insert(0, str(Path(__file__).parent / 'src'))
from models.unetpp import NestedUNet

# 导入直径测量
import importlib.util
spec = importlib.util.spec_from_file_location("diameter", str(Path(__file__).parent / "utils" / "diameter.py"))
diameter_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diameter_module)
measure_cable_tape_diameter_px = diameter_module.measure_cable_tape_diameter_px


class WrapUniformityDetector:
    """缠绕均匀性检测器"""

    def __init__(
        self,
        model_path,
        device='cuda',
        ratio_min=1.05,
        ratio_max=1.5,
        window_size=30,  # 滑动窗口大小（帧）
        std_threshold=0.15  # 标准差阈值
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.ratio_min = ratio_min
        self.ratio_max = ratio_max
        self.window_size = window_size
        self.std_threshold = std_threshold

        # 加载模型
        print(f"加载3类模型: {model_path}")
        self.model = NestedUNet(
            num_classes=4,  # bg, cable, tape, burr
            deep_supervision=False
        ).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()
        print(f"  训练mIoU: {checkpoint['best_miou']:.4f}")

        # 比例历史记录
        self.ratio_history = deque(maxlen=window_size)

    def predict_frame(self, frame):
        """预测单帧，返回分割mask"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]

        # Resize到256x256
        frame_small = cv2.resize(frame_rgb, (256, 256))
        tensor = torch.from_numpy(frame_small).float() / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)
            if isinstance(output, list):
                output = output[-1]
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

        # Resize回原图
        pred_large = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)

        return pred_large

    def detect_wrap_uniformity(self, mask):
        """检测缠绕均匀性

        Returns:
            (ratio, is_thin, is_thick, is_uniform, stats)
        """
        # 测量直径
        measurement = measure_cable_tape_diameter_px(mask, cable_id=1, tape_id=2)

        if measurement is None:
            return None, False, False, False, None

        cable_d, tape_d, delta = measurement
        ratio = tape_d / max(1e-6, cable_d)

        # 判断厚度异常
        is_thin = ratio < self.ratio_min
        is_thick = ratio > self.ratio_max

        # 添加到历史记录
        self.ratio_history.append(ratio)

        # 计算均匀性（基于滑动窗口）
        is_uniform = False
        if len(self.ratio_history) >= self.window_size:
            ratios = list(self.ratio_history)
            std = np.std(ratios)
            mean = np.mean(ratios)

            # 均匀性判断：
            # 1. 标准差小于阈值
            # 2. 所有值都在正常范围内
            in_range = all(self.ratio_min <= r <= self.ratio_max for r in ratios)
            is_uniform = (std < self.std_threshold) and in_range

            stats = {
                'mean': mean,
                'std': std,
                'min': np.min(ratios),
                'max': np.max(ratios)
            }
        else:
            stats = None

        return ratio, is_thin, is_thick, is_uniform, stats

    def visualize(self, frame, mask, ratio, is_thin, is_thick, is_uniform, stats, frame_idx, total_frames):
        """可视化结果"""
        overlay = frame.copy()

        # 分割可视化
        # Cable: 蓝色
        overlay[mask == 1] = [255, 0, 0]
        # Tape: 绿色
        overlay[mask == 2] = [0, 255, 0]
        # Burr: 红色
        overlay[mask == 3] = [0, 0, 255]

        # 混合
        result = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)

        # 添加信息
        y_offset = 30

        # 标题
        cv2.putText(result, f"Frame: {frame_idx}/{total_frames}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 35

        # 比例信息
        if ratio is not None:
            color = (0, 255, 0)  # 绿色（正常）
            if is_thin or is_thick:
                color = (0, 0, 255)  # 红色（异常）

            cv2.putText(result, f"Ratio: {ratio:.3f}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            y_offset += 35

            # 状态
            if is_thin:
                cv2.putText(result, "STATUS: THIN!", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            elif is_thick:
                cv2.putText(result, "STATUS: THICK!", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            elif is_uniform:
                cv2.putText(result, "STATUS: UNIFORM", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(result, "STATUS: CHECKING...", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            y_offset += 35

            # 统计信息
            if stats:
                cv2.putText(result, f"Mean: {stats['mean']:.3f}", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                y_offset += 25
                cv2.putText(result, f"Std:  {stats['std']:.3f}", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        return result


def main():
    parser = argparse.ArgumentParser(description='胶带缠绕均匀性检测')
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--model', type=str, default='checkpoints_3class/best_model.pth')
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--ratio-min', type=float, default=1.05, help='最小比例（胶带过薄）')
    parser.add_argument('--ratio-max', type=float, default=1.5, help='最大比例（胶带过厚）')
    parser.add_argument('--window-size', type=int, default=30, help='滑动窗口大小（帧）')
    parser.add_argument('--std-threshold', type=float, default=0.15, help='标准差阈值')
    parser.add_argument('--show-preview', action='store_true')
    args = parser.parse_args()

    print("="*70)
    print("胶带缠绕均匀性检测")
    print("="*70)
    print(f"比例范围: {args.ratio_min:.2f} - {args.ratio_max:.2f}")
    print(f"均匀性阈值: std < {args.std_threshold}")
    print(f"滑动窗口: {args.window_size} 帧")
    print("="*70)

    # 初始化检测器
    detector = WrapUniformityDetector(
        model_path=args.model,
        device=args.device,
        ratio_min=args.ratio_min,
        ratio_max=args.ratio_max,
        window_size=args.window_size,
        std_threshold=args.std_threshold
    )

    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'overlays').mkdir(exist_ok=True)

    # 打开视频
    cap = cv2.VideoCapture(args.video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\n视频: {width}x{height} @ {fps}fps, 共{total_frames}帧")
    print("\n开始处理...")
    print("-"*70)

    # 输出视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_dir / 'result.mp4'), fourcc, fps, (width, height))

    # 日志
    log_file = output_dir / 'wrap_uniformity.csv'
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("frame_idx,ratio,cable_px,tape_px,delta_px,is_thin,is_thick,is_uniform\n")

    frame_count = 0
    thin_count = 0
    thick_count = 0
    uniform_count = 0

    pbar = tqdm(total=total_frames, desc="Processing")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 预测
        mask = detector.predict_frame(frame)

        # 检测均匀性
        ratio, is_thin, is_thick, is_uniform, stats = detector.detect_wrap_uniformity(mask)

        # 统计
        if is_thin:
            thin_count += 1
        if is_thick:
            thick_count += 1
        if is_uniform:
            uniform_count += 1

        # 可视化
        result = detector.visualize(
            frame, mask, ratio, is_thin, is_thick, is_uniform, stats,
            frame_count, total_frames
        )

        # 写入视频
        writer.write(result)

        # 记录日志
        if ratio is not None:
            m = measure_cable_tape_diameter_px(mask, 1, 2)
            if m:
                cable_d, tape_d, delta = m
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"{frame_count},{ratio:.3f},{cable_d:.1f},{tape_d:.1f},{delta:.1f},{is_thin},{is_thick},{is_uniform}\n")

        # 保存异常帧
        if is_thin or is_thick:
            cv2.imwrite(str(output_dir / 'overlays' / f'frame_{frame_count:06d}.jpg'), result)

        frame_count += 1
        pbar.update(1)

        # 预览
        if args.show_preview and frame_count % 30 == 0:
            cv2.imshow('Wrap Uniformity Detection', result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    pbar.close()
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    # 最终统计
    print("\n" + "="*70)
    print("检测完成!")
    print("="*70)
    print(f"  处理帧数: {frame_count}")
    print(f"  过薄帧数: {thin_count} ({thin_count/frame_count*100:.2f}%)")
    print(f"  过厚帧数: {thick_count} ({thick_count/frame_count*100:.2f}%)")
    print(f"  均匀帧数: {uniform_count}")
    print(f"  异常帧数: {thin_count + thick_count}")
    print(f"\n  输出: {output_dir}")
    print(f"  日志: {log_file}")
    print("="*70)


if __name__ == '__main__':
    main()
