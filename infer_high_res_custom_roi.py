"""
高分辨率视频专用检测脚本
- 自定义ROI（更大、向右调整）
- 独立于主检测脚本
- 专门针对2448x2048视频优化
"""
import os
import cv2
import argparse
import time
import numpy as np
import torch
from pathlib import Path
import sys

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

from src.models.unetpp import NestedUNet


# 高分辨率视频专用ROI配置（归一化到800x448后）
# 原ROI: X[218, 421] Y[0, 448] - 宽度203px
# 新ROI: 放大并向右移动
CUSTOM_ROI = {
    'x1': 250,   # 向右移动 (原218 -> 250)
    'y1': 0,     # 保持顶部
    'x2': 550,   # 放大宽度 (原421 -> 550, 宽度从203增加到300)
    'y2': 448    # 保持底部
}

CLASS_NAMES = {0: '背景', 1: '电缆', 2: '胶带', 3: '毛刺'}
CLASS_COLORS = {
    0: (0, 0, 0),
    1: (0, 255, 0),      # 绿色：电缆
    2: (255, 0, 0),      # 蓝色：胶带
    3: (255, 0, 255),    # 紫色：毛刺
}


def preprocess_image(frame, target_size=(512, 512)):
    """预处理图像"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, target_size, interpolation=cv2.INTER_LINEAR)
    img_tensor = resized.astype(np.float32) / 255.0
    img_tensor = np.transpose(img_tensor, (2, 0, 1))
    return torch.from_numpy(img_tensor).float()


def detect_burrs_on_cable(frame_gray, mask_cable, config):
    """毛刺检测"""
    if mask_cable.max() == 0:
        return np.zeros_like(mask_cable)

    contours, _ = cv2.findContours(mask_cable, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return np.zeros_like(mask_cable)

    # 创建检测带
    cable_dilated = cv2.dilate(mask_cable,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8)),
                                iterations=1)
    detection_band = cable_dilated & (~mask_cable)

    # Canny边缘检测
    blurred = cv2.GaussianBlur(frame_gray, (5, 5), 1.0)
    edges = cv2.Canny(blurred, 50, 150)

    burr_candidates = edges & detection_band

    # 形态学处理
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    burr_candidates = cv2.morphologyEx(burr_candidates, cv2.MORPH_CLOSE, kernel_close)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    burr_candidates = cv2.morphologyEx(burr_candidates, cv2.MORPH_OPEN, kernel_open)

    # 连通域过滤
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(burr_candidates, connectivity=8)

    burr_mask = np.zeros_like(mask_cable)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        aspect_ratio = max(width, height) / (min(width, height) + 1e-6)

        if (config['min_area'] <= area <= config['max_area'] and
            aspect_ratio < 5.0 and
            width > 3 and height > 3):
            burr_mask[labels == i] = 1

    return burr_mask


def visualize_detection(frame, mask_cable, mask_tape, mask_burr, roi):
    """可视化检测结果"""
    h, w = frame.shape[:2]
    result = frame.copy()

    x1, y1, x2, y2 = roi['x1'], roi['y1'], roi['x2'], roi['y2']

    # ROI外遮罩
    roi_mask_overlay = np.zeros((h, w), dtype=np.uint8)
    roi_mask_overlay[y1:y2, x1:x2] = 255
    roi_mask_overlay_inv = cv2.bitwise_not(roi_mask_overlay)
    overlay = result.copy()
    overlay[roi_mask_overlay_inv > 0] = [0, 0, 0]
    result = cv2.addWeighted(result, 0.7, overlay, 0.3, 0)

    # 叠加检测结果
    cable_overlay = np.zeros_like(result)
    cable_overlay[mask_cable > 0] = CLASS_COLORS[1]
    tape_overlay = np.zeros_like(result)
    tape_overlay[mask_tape > 0] = CLASS_COLORS[2]
    burr_overlay = np.zeros_like(result)
    burr_overlay[mask_burr > 0] = CLASS_COLORS[3]

    result = cv2.addWeighted(result, 0.6, cable_overlay, 0.4, 0)
    result = cv2.addWeighted(result, 0.6, tape_overlay, 0.4, 0)
    result = cv2.addWeighted(result, 0.5, burr_overlay, 0.5, 0)

    # ROI框
    cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.putText(result, 'ROI', (x1 + 5, y1 + 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # 轮廓
    cable_contours, _ = cv2.findContours(mask_cable, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, cable_contours, -1, (0, 255, 0), 2)

    tape_contours, _ = cv2.findContours(mask_tape, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, tape_contours, -1, (255, 0, 0), 2)

    burr_contours, _ = cv2.findContours(mask_burr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, burr_contours, -1, (255, 0, 255), 3)

    return result


def main():
    parser = argparse.ArgumentParser(description='高分辨率视频专用检测（自定义ROI）')
    parser.add_argument('--video', type=str, required=True, help='输入视频路径')
    parser.add_argument('--model', type=str, default='checkpoints_3class_advanced/best_model.pth')
    parser.add_argument('--output', type=str, required=True, help='输出目录')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 毛刺检测配置
    burr_config = {
        'band_out': 15,
        'laplacian_threshold': 25,
        'min_area': 30,
        'max_area': 800,
        'morph_kernel': 3
    }

    print("=" * 70)
    print("高分辨率视频专用检测系统")
    print("=" * 70)
    print(f"自定义ROI: X[{CUSTOM_ROI['x1']}, {CUSTOM_ROI['x2']}] Y[{CUSTOM_ROI['y1']}, {CUSTOM_ROI['y2']}]")
    print(f"ROI宽度: {CUSTOM_ROI['x2'] - CUSTOM_ROI['x1']}px (原203px -> 300px)")
    print(f"ROI位置: 向右移动 {CUSTOM_ROI['x1'] - 218}px")
    print("=" * 70)

    # 加载模型
    print(f"\n加载模型: {args.model}")
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = NestedUNet(num_classes=3, deep_supervision=True, pretrained_encoder=False).to(device)
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    print(f"模型已加载到 {device}")

    # 打开视频
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"错误: 无法打开视频 {args.video}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\n原始视频: {width_orig}x{height_orig} @ {fps:.2f}fps, 总帧数: {total_frames}")

    # 处理流程：旋转 -> 归一化到800x448
    target_width = 800
    target_height = 448

    print(f"处理流程: 旋转90度 -> 归一化到 {target_width}x{target_height}")
    print("-" * 70)

    # 视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(
        str(output_dir / 'result.mp4'),
        fourcc, fps, (target_width, target_height)
    )

    frame_count = 0
    processed_count = 0
    burr_detected_frames = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 旋转90度（逆时针）
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # 归一化到800x448
        frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

        processed_count += 1

        # 模型推理
        img_tensor = preprocess_image(frame).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            if isinstance(outputs, list):
                outputs = outputs[-1]

        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        pred_512 = np.argmax(probs, axis=0).astype(np.uint8)

        # 提取mask
        mask_cable_512 = (pred_512 == 1).astype(np.uint8)
        mask_tape_512 = (pred_512 == 2).astype(np.uint8)

        # 调整到目标尺寸
        mask_cable_full = cv2.resize(mask_cable_512, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
        mask_tape_full = cv2.resize(mask_tape_512, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

        # 限制检测结果只在ROI内
        x1, y1, x2, y2 = CUSTOM_ROI['x1'], CUSTOM_ROI['y1'], CUSTOM_ROI['x2'], CUSTOM_ROI['y2']
        mask_cable = np.zeros_like(mask_cable_full)
        mask_tape = np.zeros_like(mask_tape_full)
        mask_cable[y1:y2, x1:x2] = mask_cable_full[y1:y2, x1:x2]
        mask_tape[y1:y2, x1:x2] = mask_tape_full[y1:y2, x1:x2]

        # 毛刺检测
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask_burr = detect_burrs_on_cable(frame_gray, mask_cable, burr_config)

        # 统计
        burr_pixels = np.sum(mask_burr)
        if burr_pixels > 0:
            burr_detected_frames += 1

        # 可视化
        result = visualize_detection(frame, mask_cable, mask_tape, mask_burr, CUSTOM_ROI)

        # 显示信息
        elapsed = time.time() - start_time
        fps_current = processed_count / elapsed if elapsed > 0 else 0
        info_text = f"Frame: {frame_count}/{total_frames} | FPS: {fps_current:.1f}"

        cable_pixels = np.sum(mask_cable)
        tape_pixels = np.sum(mask_tape)
        roi_area = (x2 - x1) * (y2 - y1)

        cable_ratio = cable_pixels / roi_area * 100 if roi_area > 0 else 0
        tape_ratio = tape_pixels / roi_area * 100 if roi_area > 0 else 0
        burr_ratio = burr_pixels / roi_area * 100 if roi_area > 0 else 0

        status = "[BURR!]" if burr_pixels > 0 else "[OK]"
        defect_text = f"{status} Cable:{cable_ratio:.1f}% Tape:{tape_ratio:.1f}% Burr:{burr_ratio:.1f}%"

        cv2.putText(result, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(result, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        cv2.putText(result, defect_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(result, defect_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

        video_writer.write(result)

        # 缩小显示窗口
        display_height, display_width = result.shape[:2]
        result_display = cv2.resize(result, (display_width // 2, display_height // 2))
        cv2.imshow('High-Res Custom ROI Detection', result_display)

        if processed_count % 60 == 0:
            print(f"[{processed_count:4d}] Frame {frame_count}/{total_frames} {status} | "
                  f"Cable:{cable_ratio:5.1f}% Tape:{tape_ratio:5.1f}% Burr:{burr_ratio:5.1f}%")

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    print("-" * 70)
    print(f"完成! 处理了 {processed_count} 帧, 用时 {elapsed:.1f}s, 平均 FPS: {processed_count/elapsed:.2f}")
    print(f"检测到毛刺的帧数: {burr_detected_frames}/{processed_count} ({burr_detected_frames/processed_count*100:.1f}%)")
    print(f"结果已保存到: {output_dir}")


if __name__ == "__main__":
    main()
