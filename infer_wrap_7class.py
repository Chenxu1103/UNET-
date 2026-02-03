"""胶带缠绕均匀性检测 - 使用7类模型

基于已有的7类模型（73% mIoU）检测缠绕均匀性
只使用：cable + tape + burr，忽略其他缺陷
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

import importlib.util
spec = importlib.util.spec_from_file_location("diameter", str(Path(__file__).parent / "utils" / "diameter.py"))
diameter_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diameter_module)
measure_cable_tape_diameter_px = diameter_module.measure_cable_tape_diameter_px


def main():
    parser = argparse.ArgumentParser(description='胶带缠绕均匀性检测（7类模型）')
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pth')
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--ratio-min', type=float, default=1.05)
    parser.add_argument('--ratio-max', type=float, default=1.5)
    parser.add_argument('--window-size', type=int, default=30)
    parser.add_argument('--std-threshold', type=float, default=0.15)
    parser.add_argument('--show-preview', action='store_true')
    args = parser.parse_args()

    print("="*70)
    print("胶带缠绕均匀性检测（使用7类模型）")
    print("="*70)
    print(f"模型: {args.model}")
    print(f"比例范围: {args.ratio_min:.2f} - {args.ratio_max:.2f}")
    print("="*70)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    # 加载7类模型
    print(f"\n加载模型...")
    model = NestedUNet(
        num_classes=7,
        deep_supervision=False,
        input_channels=3
    ).to(device)

    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print("  模型加载完成")

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
        f.write("frame_idx,ratio,cable_px,tape_px,delta_px,status\n")

    # 比例历史
    ratio_history = deque(maxlen=args.window_size)

    frame_count = 0
    thin_count = thick_count = uniform_count = 0

    pbar = tqdm(total=total_frames, desc="Processing")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 预测
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_small = cv2.resize(frame_rgb, (256, 256))
        tensor = torch.from_numpy(frame_small).float() / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor)
            if isinstance(output, list):
                output = output[-1]
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

        pred_large = cv2.resize(pred, (width, height), interpolation=cv2.INTER_NEAREST)

        # 测量直径
        measurement = measure_cable_tape_diameter_px(pred_large, cable_id=1, tape_id=2)

        ratio = None
        status = "OK"
        status_color = (0, 255, 0)

        if measurement is not None:
            cable_d, tape_d, delta = measurement
            ratio = tape_d / max(1e-6, cable_d)

            # 判断状态
            is_thin = ratio < args.ratio_min
            is_thick = ratio > args.ratio_max

            if is_thin:
                status = "THIN"
                status_color = (0, 255, 255)
                thin_count += 1
            elif is_thick:
                status = "THICK"
                status_color = (255, 255, 0)
                thick_count += 1
            else:
                ratio_history.append(ratio)
                if len(ratio_history) >= args.window_size:
                    std = np.std(list(ratio_history))
                    if std < args.std_threshold:
                        status = "UNIFORM"
                        uniform_count += 1

            # 记录日志
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"{frame_count},{ratio:.3f},{cable_d:.1f},{tape_d:.1f},{delta:.1f},{status}\n")

        # 可视化
        overlay = frame.copy()
        # Cable: 蓝色, Tape: 绿色, Burr: 红色
        overlay[pred_large == 1] = [255, 0, 0]
        overlay[pred_large == 2] = [0, 255, 0]
        overlay[pred_large == 3] = [0, 0, 255]

        result = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)

        # 添加信息
        cv2.putText(result, f"Frame: {frame_count}/{total_frames}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if ratio is not None:
            cv2.putText(result, f"Ratio: {ratio:.3f}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
            cv2.putText(result, f"Status: {status}", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

        writer.write(result)

        # 保存异常帧
        if status in ["THIN", "THICK"]:
            cv2.imwrite(str(output_dir / 'overlays' / f'frame_{frame_count:06d}_{status}.jpg'), result)

        frame_count += 1
        pbar.update(1)

        if args.show_preview and frame_count % 30 == 0:
            cv2.imshow('Wrap Detection', result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    pbar.close()
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print("\n" + "="*70)
    print("检测完成!")
    print("="*70)
    print(f"  处理帧数: {frame_count}")
    print(f"  过薄帧数: {thin_count}")
    print(f"  过厚帧数: {thick_count}")
    print(f"  均匀帧数: {uniform_count}")
    print(f"  输出: {output_dir}")
    print("="*70)


if __name__ == '__main__':
    main()
