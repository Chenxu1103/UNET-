"""
增强版毛刺检测脚本
- 图像预处理增强（直方图均衡化、对比度增强）
- 改进的毛刺检测算法
- 专门针对垂直视角优化
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


# 垂直视角视频ROI配置（归一化到800x448后）
VERTICAL_ROI = {
    'x1': 200,
    'y1': 0,
    'x2': 600,
    'y2': 448
}

CLASS_COLORS = {
    0: (0, 0, 0),
    1: (0, 255, 0),      # 绿色：电缆
    2: (255, 0, 0),      # 蓝色：胶带
    3: (255, 0, 255),    # 紫色：毛刺
}


def enhance_image(frame):
    """
    图像增强处理（方案C）
    - 直方图均衡化
    - 对比度增强
    - 去噪
    """
    # 转换到LAB色彩空间
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # 对L通道进行CLAHE（对比度限制自适应直方图均衡化）
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)

    # 合并通道
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    # 去噪
    enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)

    # 锐化
    kernel_sharpen = np.array([[-1,-1,-1],
                               [-1, 9,-1],
                               [-1,-1,-1]])
    enhanced = cv2.filter2D(enhanced, -1, kernel_sharpen)

    return enhanced


def detect_burrs_enhanced(frame_gray, mask_cable, config):
    """
    增强版毛刺检测
    基于标注数据特征分析的改进算法
    """
    if mask_cable.max() == 0:
        return np.zeros_like(mask_cable)

    contours, _ = cv2.findContours(mask_cable, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return np.zeros_like(mask_cable)

    # 创建更宽的检测带（毛刺可能突出较远）
    cable_dilated = cv2.dilate(mask_cable,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)),
                                iterations=1)
    detection_band = cable_dilated & (~mask_cable)

    # 多尺度边缘检测
    # 1. Canny边缘
    blurred = cv2.GaussianBlur(frame_gray, (5, 5), 1.0)
    edges_canny = cv2.Canny(blurred, 30, 100)

    # 2. Sobel边缘
    sobelx = cv2.Sobel(frame_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(frame_gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobelx**2 + sobely**2)
    sobel_mag = np.uint8(sobel_mag / sobel_mag.max() * 255)
    _, edges_sobel = cv2.threshold(sobel_mag, 50, 255, cv2.THRESH_BINARY)

    # 3. Laplacian边缘
    laplacian = cv2.Laplacian(frame_gray, cv2.CV_64F)
    laplacian_abs = np.abs(laplacian).astype(np.uint8)
    _, edges_laplacian = cv2.threshold(laplacian_abs, 15, 255, cv2.THRESH_BINARY)

    # 融合多种边缘检测结果
    edges_combined = cv2.bitwise_or(edges_canny, edges_sobel)
    edges_combined = cv2.bitwise_or(edges_combined, edges_laplacian)

    # 限制在检测带内
    burr_candidates = edges_combined & detection_band

    # 形态学处理
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    burr_candidates = cv2.morphologyEx(burr_candidates, cv2.MORPH_CLOSE, kernel_close)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    burr_candidates = cv2.morphologyEx(burr_candidates, cv2.MORPH_OPEN, kernel_open)

    # 连通域过滤（基于标注数据的特征）
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(burr_candidates, connectivity=8)

    burr_mask = np.zeros_like(mask_cable)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]

        # 根据标注数据，毛刺特征：
        # - 面积：50-300像素（中等大小）
        # - 长宽比：不能太细长
        # - 最小尺寸：至少5x5像素
        aspect_ratio = max(width, height) / (min(width, height) + 1e-6)

        if (config['min_area'] <= area <= config['max_area'] and
            aspect_ratio < 6.0 and
            width >= 5 and height >= 5):
            burr_mask[labels == i] = 1

    return burr_mask


def preprocess_image(frame, target_size=(512, 512)):
    """预处理图像"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, target_size, interpolation=cv2.INTER_LINEAR)
    img_tensor = resized.astype(np.float32) / 255.0
    img_tensor = np.transpose(img_tensor, (2, 0, 1))
    return torch.from_numpy(img_tensor).float()


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
    parser = argparse.ArgumentParser(description='增强版毛刺检测（图像增强+改进算法）')
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--model', type=str, default='checkpoints_3class_advanced/best_model.pth')
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--enhance', action='store_true', help='启用图像增强')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 毛刺检测配置（基于标注数据特征）
    burr_config = {
        'min_area': 50,      # 最小面积（根据标注数据）
        'max_area': 500,     # 最大面积
        'morph_kernel': 5
    }

    print("=" * 70)
    print("增强版毛刺检测系统")
    print("=" * 70)
    print(f"图像增强: {'启用' if args.enhance else '禁用'}")
    print(f"ROI: X[{VERTICAL_ROI['x1']}, {VERTICAL_ROI['x2']}] Y[{VERTICAL_ROI['y1']}, {VERTICAL_ROI['y2']}]")
    print(f"毛刺检测: 多尺度边缘融合 + 形态学过滤")
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

    # 处理流程：旋转 -> 归一化
    target_width = 800
    target_height = 448

    print(f"处理流程: 旋转90度 -> 归一化到 {target_width}x{target_height}")
    if args.enhance:
        print("图像增强: CLAHE + 去噪 + 锐化")
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

        # 旋转90度
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # 归一化
        frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

        # 图像增强（可选）
        if args.enhance:
            frame = enhance_image(frame)

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
        x1, y1, x2, y2 = VERTICAL_ROI['x1'], VERTICAL_ROI['y1'], VERTICAL_ROI['x2'], VERTICAL_ROI['y2']
        mask_cable = np.zeros_like(mask_cable_full)
        mask_tape = np.zeros_like(mask_tape_full)
        mask_cable[y1:y2, x1:x2] = mask_cable_full[y1:y2, x1:x2]
        mask_tape[y1:y2, x1:x2] = mask_tape_full[y1:y2, x1:x2]

        # 增强版毛刺检测
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask_burr = detect_burrs_enhanced(frame_gray, mask_cable, burr_config)

        # 统计
        burr_pixels = np.sum(mask_burr)
        if burr_pixels > 0:
            burr_detected_frames += 1

        # 可视化
        result = visualize_detection(frame, mask_cable, mask_tape, mask_burr, VERTICAL_ROI)

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
        cv2.imshow('Enhanced Burr Detection', result_display)

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
