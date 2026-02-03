"""
两阶段毛刺检测系统
阶段1: 使用高性能模型检测电缆和胶带
阶段2: 在电缆区域上使用规则法检测毛刺
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


CLASS_NAMES = {0: '背景', 1: '电缆', 2: '胶带', 3: '毛刺'}
CLASS_COLORS = {
    0: (0, 0, 0),
    1: (0, 255, 0),      # 绿色：电缆 (BGR)
    2: (255, 0, 0),      # 蓝色：胶带 (BGR)
    3: (255, 0, 255),    # 紫色：毛刺 (BGR)
}

FIXED_ROI_512 = {
    'x1': 140,
    'y1': 0,      # 从顶部开始
    'x2': 270,
    'y2': 512     # 到底部结束
}


def map_roi_to_original(original_size, target_size=(512, 512)):
    orig_w, orig_h = original_size
    target_w, target_h = target_size
    scale_x = orig_w / target_w
    scale_y = orig_h / target_h
    roi = FIXED_ROI_512
    x1 = int(roi['x1'] * scale_x)
    y1 = int(roi['y1'] * scale_y)
    x2 = int(roi['x2'] * scale_x)
    y2 = int(roi['y2'] * scale_y)
    return (x1, y1, x2, y2)


def detect_burrs_on_cable(frame_gray, mask_cable, config=None):
    """
    在电缆区域上检测毛刺

    改进方法：
    1. 只在电缆边界的窄带区域检测
    2. 使用Canny边缘检测 + 形态学约束
    3. 检测突出于正常边界的异常区域
    4. 严格过滤以减少误检
    """
    if config is None:
        config = {
            'band_out': 15,           # 外扩像素
            'laplacian_threshold': 25,  # Laplacian阈值
            'min_area': 30,            # 最小面积
            'max_area': 800,           # 最大面积
            'morph_kernel': 3,         # 形态学核大小
        }

    if mask_cable.max() == 0:
        return np.zeros_like(mask_cable)

    # 1. 获取电缆边界
    contours, _ = cv2.findContours(mask_cable, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return np.zeros_like(mask_cable)

    # 2. 创建窄的检测带（只在电缆边界外侧）
    cable_dilated = cv2.dilate(mask_cable,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8)),
                                iterations=1)
    detection_band = cable_dilated & (~mask_cable)  # 只检测外扩区域

    # 3. 使用Canny边缘检测（更鲁棒）
    # 先进行高斯模糊减少噪声
    blurred = cv2.GaussianBlur(frame_gray, (5, 5), 1.0)
    edges = cv2.Canny(blurred, 50, 150)

    # 4. 限制在检测带内
    burr_candidates = edges & detection_band

    # 5. 形态学处理 - 连接断裂的边缘
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    burr_candidates = cv2.morphologyEx(burr_candidates, cv2.MORPH_CLOSE, kernel_close)

    # 6. 去除小噪声
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    burr_candidates = cv2.morphologyEx(burr_candidates, cv2.MORPH_OPEN, kernel_open)

    # 7. 连通域过滤 - 更严格的条件
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(burr_candidates, connectivity=8)

    burr_mask = np.zeros_like(mask_cable)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]

        # 过滤条件：
        # - 面积在合理范围内
        # - 不能太细长（避免检测到正常边缘）
        # - 长宽比要合理
        aspect_ratio = max(width, height) / (min(width, height) + 1e-6)

        if (config['min_area'] <= area <= config['max_area'] and
            aspect_ratio < 5.0 and  # 不能太细长
            width > 3 and height > 3):  # 最小尺寸
            burr_mask[labels == i] = 1

    return burr_mask


def preprocess_image(frame, target_size=(512, 512)):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, target_size, interpolation=cv2.INTER_LINEAR)
    img_tensor = resized.astype(np.float32) / 255.0
    img_tensor = np.transpose(img_tensor, (2, 0, 1))
    return torch.from_numpy(img_tensor).float()


def visualize_two_stage(frame, mask_cable, mask_tape, mask_burr, roi_orig, draw_roi_box=True):
    h, w = frame.shape[:2]
    result = frame.copy()

    # ROI外遮罩
    roi_mask_overlay = np.zeros((h, w), dtype=np.uint8)
    x1, y1, x2, y2 = roi_orig
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
    result = cv2.addWeighted(result, 0.5, burr_overlay, 0.5, 0)  # 毛刺更明显

    # ROI框（可选）
    if draw_roi_box:
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(result, 'ROI', (x1 + 5, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # 轮廓
    cable_contours, _ = cv2.findContours(mask_cable, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, cable_contours, -1, (0, 255, 0), 2)  # 绿色

    tape_contours, _ = cv2.findContours(mask_tape, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, tape_contours, -1, (255, 0, 0), 2)  # 蓝色

    burr_contours, _ = cv2.findContours(mask_burr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, burr_contours, -1, (255, 0, 255), 3)  # 紫色，轮廓更粗

    return result


def main():
    parser = argparse.ArgumentParser(description='两阶段毛刺检测')
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--model', type=str, default='checkpoints_3class_advanced/best_model.pth')
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--frame-stride', type=int, default=1)
    parser.add_argument('--print-interval', type=int, default=60)
    parser.add_argument('--burr-sensitivity', type=str, default='medium',
                       choices=['low', 'medium', 'high'])
    parser.add_argument('--rotate', action='store_true', help='旋转视频90度（逆时针）')
    parser.add_argument('--normalize-resolution', action='store_true',
                       help='将视频归一化到标准分辨率（800x448），适用于高分辨率视频')
    parser.add_argument('--target-width', type=int, default=800, help='归一化目标宽度')
    parser.add_argument('--target-height', type=int, default=448, help='归一化目标高度')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 毛刺检测配置
    burr_configs = {
        'low': {'band_out': 10, 'laplacian_threshold': 35, 'min_area': 50, 'max_area': 800, 'morph_kernel': 3},
        'medium': {'band_out': 15, 'laplacian_threshold': 25, 'min_area': 30, 'max_area': 800, 'morph_kernel': 3},
        'high': {'band_out': 20, 'laplacian_threshold': 20, 'min_area': 20, 'max_area': 1000, 'morph_kernel': 5},
    }
    burr_config = burr_configs[args.burr_sensitivity]

    print("=" * 70)
    print("两阶段毛刺检测系统")
    print("=" * 70)
    print(f"阶段1: 电缆/胶带分割 (模型: {args.model})")
    print(f"阶段2: 毛刺检测 (灵敏度: {args.burr_sensitivity})")
    print(f"  - Laplacian阈值: {burr_config['laplacian_threshold']}")
    print(f"  - 检测带宽度: {burr_config['band_out']}px")
    print(f"  - 面积范围: {burr_config['min_area']}-{burr_config['max_area']}px")
    print("=" * 70)

    # 加载模型
    print(f"\n加载模型: {args.model}")
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = NestedUNet(num_classes=3, deep_supervision=True, pretrained_encoder=False).to(device)
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    print(f"模型已加载到 {device}")

    roi_512 = FIXED_ROI_512
    print(f"\n固定ROI（512x512）: X[{roi_512['x1']}, {roi_512['x2']}] Y[{roi_512['y1']}, {roi_512['y2']}]")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"错误: 无法打开视频 {args.video}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"原始视频: {width_orig}x{height_orig}")

    # 根据--rotate参数决定是否旋转
    if args.rotate:
        width_after_rotate = height_orig
        height_after_rotate = width_orig
        print(f"旋转后: {width_after_rotate}x{height_after_rotate}")
    else:
        width_after_rotate = width_orig
        height_after_rotate = height_orig

    # 根据--normalize-resolution参数决定是否归一化分辨率
    if args.normalize_resolution:
        width = args.target_width
        height = args.target_height
        print(f"归一化到: {width}x{height}")
    else:
        width = width_after_rotate
        height = height_after_rotate

    roi_orig = map_roi_to_original((width, height), (512, 512))
    print(f"固定ROI（{width}x{height}）: X[{roi_orig[0]}, {roi_orig[2]}] Y[{roi_orig[1]}, {roi_orig[3]}]")
    print(f"处理分辨率: {width}x{height} @ {fps:.2f}fps, 总帧数: {total_frames}")
    print("-" * 70)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(
        str(output_dir / 'result.mp4'),
        fourcc, fps, (width, height)
    )

    frame_count = 0
    processed_count = 0
    burr_detected_frames = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 根据参数决定是否旋转视频源
        if args.rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # 根据参数决定是否归一化分辨率
        if args.normalize_resolution:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

        frame_count += 1
        if frame_count % args.frame_stride != 0:
            continue

        processed_count += 1

        # 获取ROI区域
        x1, y1, x2, y2 = roi_orig

        # 阶段1: 模型推理（只对整个画面进行，后续会限制到ROI内）
        img_tensor = preprocess_image(frame).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            if isinstance(outputs, list):
                outputs = outputs[-1]

        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        pred_512 = np.argmax(probs, axis=0).astype(np.uint8)

        # 提取电缆和胶带mask
        mask_cable_512 = (pred_512 == 1).astype(np.uint8)
        mask_tape_512 = (pred_512 == 2).astype(np.uint8)

        # 调整到原始尺寸
        mask_cable_full = cv2.resize(mask_cable_512, (width, height), interpolation=cv2.INTER_NEAREST)
        mask_tape_full = cv2.resize(mask_tape_512, (width, height), interpolation=cv2.INTER_NEAREST)

        # 限制检测结果只在ROI内
        mask_cable = np.zeros_like(mask_cable_full)
        mask_tape = np.zeros_like(mask_tape_full)
        mask_cable[y1:y2, x1:x2] = mask_cable_full[y1:y2, x1:x2]
        mask_tape[y1:y2, x1:x2] = mask_tape_full[y1:y2, x1:x2]

        # 阶段2: 毛刺检测（只在ROI内的电缆上检测）
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask_burr = detect_burrs_on_cable(frame_gray, mask_cable, burr_config)

        # 统计
        burr_pixels = np.sum(mask_burr)
        if burr_pixels > 0:
            burr_detected_frames += 1

        # 可视化
        result = visualize_two_stage(frame, mask_cable, mask_tape, mask_burr, roi_orig)

        # 显示信息
        elapsed = time.time() - start_time
        fps_current = processed_count / elapsed if elapsed > 0 else 0
        info_text = f"Frame: {frame_count}/{total_frames} | FPS: {fps_current:.1f}"

        cable_pixels = np.sum(mask_cable)
        tape_pixels = np.sum(mask_tape)
        x1, y1, x2, y2 = roi_orig
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
        cv2.imshow('Two-Stage Burr Detection', result_display)

        if processed_count % args.print_interval == 0:
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
