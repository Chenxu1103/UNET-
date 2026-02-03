"""
电缆缠绕检测 - 优化版（解决胶带误检问题）

改进：
  1. 提高胶带检测阈值，减少误检
  2. 强制互斥：胶带不能标注在电缆区域
  3. 增加空间约束：胶带应该在电缆外围
  4. 更严格的连通域过滤
"""

import os
import cv2
import argparse
import numpy as np
import torch
import torch.nn as nn
import sys
from pathlib import Path

# 添加目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

from src.models.simple_unet import SimpleUNet

# 导入直径测量函数
import importlib.util
spec = importlib.util.spec_from_file_location("diameter", str(project_root / "utils" / "diameter.py"))
diameter_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diameter_module)
measure_cable_tape_diameter_px = diameter_module.measure_cable_tape_diameter_px


# 类别配置
CLASS_NAMES = {
    0: '背景',
    1: '电缆',
    2: '胶带',
    3: '鼓包缺陷',
    4: '松脱缺陷',
    5: '破损缺陷',  # 毛刺
    6: '厚度不足缺陷',
}

# 类别颜色 (BGR格式)
CLASS_COLORS = {
    0: (0, 0, 0),         # 黑色：背景
    1: (255, 0, 0),       # 蓝色：电缆
    2: (0, 255, 0),       # 绿色：胶带
    3: (0, 0, 255),       # 红色：鼓包缺陷
    4: (255, 255, 0),     # 青色：松脱缺陷
    5: (255, 0, 255),     # 洋红：破损缺陷（毛刺）
    6: (0, 165, 255),     # 橙色：厚度不足
}


class SimpleInferenceOptimized:
    """优化版推理器 - 减少胶带误检"""

    def __init__(self, model_path: str, num_classes: int = 7, device: str = 'cpu'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes

        print(f"设备: {self.device}")
        print(f"加载模型: {model_path}")

        # 使用SimpleUNet而不是NestedUNet
        self.model = SimpleUNet(num_classes=num_classes, num_channels=3).to(self.device)

        # 加载权重
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'])
        elif 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()
        print("模型加载完成")
        print("\n优化参数:")
        print("  - 电缆阈值: 0.30 (保持高召回率)")
        print("  - 胶带阈值: 0.55 (提高精确率)")
        print("  - 强制互斥: 胶带不能覆盖电缆")
        print("  - 空间过滤: 只保留合理位置的胶带")
        print()

    def spatial_filter_tape(self, tape_mask, cable_mask):
        """
        空间过滤：胶带应该在电缆的外围，不应该完全在电缆内部

        原理：真正的胶带应该在电缆的两侧形成"包裹"效果
        """
        h, w = tape_mask.shape

        # 如果没有电缆或没有胶带，直接返回
        if np.sum(cable_mask) == 0 or np.sum(tape_mask) == 0:
            return tape_mask

        # 找到电缆的中心X坐标
        cable_pixels = np.where(cable_mask > 0)
        if len(cable_pixels[0]) == 0:
            return tape_mask

        x_min = cable_pixels[1].min()
        x_max = cable_pixels[1].max()
        cable_center_x = (x_min + x_max) // 2
        cable_width = x_max - x_min

        # 定义合理的胶带区域：电缆中心两侧各一定范围内
        # 胶带应该在电缆的外围，但不会偏离太远
        valid_region = np.zeros_like(tape_mask)

        # 胶带的有效范围：电缆中心左右两侧
        # 左侧胶带区域：从电缆左边界向左扩展
        left_region_start = max(0, x_min - cable_width // 2)
        left_region_end = x_min + cable_width // 3

        # 右侧胶带区域：从电缆右边界向右扩展
        right_region_start = max(x_min + 2 * cable_width // 3, x_max - cable_width // 3)
        right_region_end = min(w, x_max + cable_width // 2)

        valid_region[:, left_region_start:left_region_end] = 1
        valid_region[:, right_region_start:right_region_end] = 1

        # 只保留在合理区域的胶带
        filtered_tape = tape_mask & valid_region

        # 统计过滤前后的面积
        original_area = np.sum(tape_mask)
        filtered_area = np.sum(filtered_tape)

        if original_area > 0 and filtered_area < original_area * 0.5:
            # 如果过滤后剩余太少，可能过滤过度，返回原mask
            return tape_mask

        return filtered_tape

    @torch.no_grad()
    def predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        """预测单帧 - 优化版"""
        h, w = frame_bgr.shape[:2]

        # 预处理
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_small = cv2.resize(frame_rgb, (256, 256))
        tensor = torch.from_numpy(frame_small).float() / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)

        # 推理
        output = self.model(tensor)

        # 获取概率 - 使用软分割
        probs = torch.softmax(output, dim=1)[0]  # [7, 256, 256]

        # 调整回原始尺寸
        probs_large = probs.cpu().numpy()
        probs_resized = np.zeros((7, h, w), dtype=np.float32)
        for c in range(7):
            probs_resized[c] = cv2.resize(probs_large[c], (w, h), interpolation=cv2.INTER_LINEAR)

        # 使用优化的概率阈值
        # 电缆(1): 保持较低阈值，提高召回率
        cable_prob = probs_resized[1]
        cable_mask = (cable_prob >= 0.30).astype(np.uint8)

        # 胶带(2): 提高阈值，减少误检（从0.35提高到0.55）
        tape_prob = probs_resized[2]
        tape_mask_raw = (tape_prob >= 0.55).astype(np.uint8)

        # 毛刺(5): 高阈值，减少误检
        burr_prob = probs_resized[5]
        burr_mask = (burr_prob >= 0.70).astype(np.uint8)

        # 形态学操作
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # 电缆：先闭运算填充空洞
        if np.sum(cable_mask) > 0:
            cable_mask = cv2.morphologyEx(cable_mask, cv2.MORPH_CLOSE, kernel_medium, iterations=2)

        # 胶带：闭运算填充空洞，但不膨胀（避免与电缆重叠）
        if np.sum(tape_mask_raw) > 0:
            tape_mask_raw = cv2.morphologyEx(tape_mask_raw, cv2.MORPH_CLOSE, kernel_medium, iterations=1)

        # ===== 关键改进：强制互斥 =====
        # 1. 胶带不能覆盖在电缆上
        tape_mask_excl = tape_mask_raw & (~cable_mask)

        # 2. 空间过滤：只保留在合理位置的胶带
        tape_mask_filtered = self.spatial_filter_tape(tape_mask_excl, cable_mask)

        # 3. 连通域过滤：只保留大面积的连通域
        if np.sum(tape_mask_filtered) > 0:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(tape_mask_filtered, connectivity=8)
            filtered_tape = np.zeros_like(tape_mask_filtered)
            min_tape_area = 500  # 最小面积，提高阈值过滤噪点

            for label in range(1, num_labels):
                area = stats[label, cv2.CC_STAT_AREA]
                width = stats[label, cv2.CC_STAT_WIDTH]

                # 只保留大面积且宽度合理的连通域
                if area >= min_tape_area and width >= 20:
                    filtered_tape[labels == label] = 1

            tape_mask = filtered_tape
        else:
            tape_mask = tape_mask_filtered

        # 毛刺：严格过滤
        if np.sum(burr_mask) > 0:
            # 开运算去除噪点
            burr_mask = cv2.morphologyEx(burr_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)

            # 连通域过滤
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(burr_mask, connectivity=8)
            filtered_burr = np.zeros_like(burr_mask)
            min_burr_area = 100  # 最小面积

            for label in range(1, num_labels):
                area = stats[label, cv2.CC_STAT_AREA]
                if area >= min_burr_area:
                    filtered_burr[labels == label] = 1
            burr_mask = filtered_burr

        # 按优先级合并：burr > tape > cable > background
        result = np.zeros((h, w), dtype=np.uint8)
        result[cable_mask > 0] = 1
        result[tape_mask > 0] = 2
        result[burr_mask > 0] = 5

        return result

    def overlay_mask(self, frame_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """将mask叠加到原始图像"""
        h, w = frame_bgr.shape[:2]
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)

        # 用户只关心：cable(1), tape(2), burr(5)
        display_classes = {1: CLASS_COLORS[1], 2: CLASS_COLORS[2], 5: CLASS_COLORS[5]}

        for class_id, color in display_classes.items():
            class_mask = (mask == class_id).astype(np.uint8)
            color_mask[class_mask > 0] = color

        result = frame_bgr.copy()
        region = (mask > 0)
        if np.any(region):
            blended = ((1 - alpha) * frame_bgr.astype(np.float32) +
                      alpha * color_mask.astype(np.float32)).astype(np.uint8)
            result[region] = blended[region]

        # 添加轮廓
        for class_id, color in display_classes.items():
            class_mask = (mask == class_id).astype(np.uint8) * 255
            contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(result, contours, -1, color, 2)

        return result


def process_video(
    model_path: str,
    video_path: str,
    output_dir: str,
    num_classes: int = 7,
    ratio_min: float = 1.05,
    ratio_max: float = 1.5,
    min_area_px: int = 50,
    device: str = 'cpu',
    show_preview: bool = True
):
    """处理视频"""

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'snapshots'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'overlays'), exist_ok=True)

    # 初始化推理器
    inferencer = SimpleInferenceOptimized(
        model_path=model_path,
        num_classes=num_classes,
        device=device
    )

    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"视频: {width}x{height} @ {fps}fps, 共{total_frames}帧")
    print(f"厚度阈值: {ratio_min:.2f} - {ratio_max:.2f}")
    print("\n开始处理...")

    # 日志文件
    log_path = os.path.join(output_dir, 'events.csv')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("frame_idx,event_type,ratio,cable_px,tape_px,delta_px\n")

    # 创建视频写入器 - 保存完整的检测结果视频
    output_video_path = os.path.join(output_dir, 'detection_result.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # 统计
    frame_idx = 0
    burr_count = 0
    thin_count = 0
    thick_count = 0

    print(f"正在处理并保存视频到: {output_video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # 预测
        mask = inferencer.predict(frame)

        # 检测毛刺（使用更严格的面积阈值）
        has_burr = False
        burr_mask = (mask == 5).astype(np.uint8)
        burr_area = np.sum(burr_mask)

        # 增加毛刺最小面积阈值（从50增加到200）
        min_burr_area = max(min_area_px, 200)

        # 进一步过滤：检查连通域的最大面积
        if burr_area >= min_burr_area:
            # 检查是否有足够大的单个连通域
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(burr_mask, connectivity=8)
            max_area = 0
            for label in range(1, num_labels):
                area = stats[label, cv2.CC_STAT_AREA]
                max_area = max(max_area, area)

            # 只有最大连通域超过阈值才算真正的毛刺
            if max_area >= 150:  # 单个毛刺区域至少150像素
                has_burr = True
                burr_count += 1
            else:
                # 最大连通域太小，可能是噪点，忽略
                pass

        # 检测直径/厚度异常
        ratio, is_thin, is_thick = None, False, False
        m = measure_cable_tape_diameter_px(mask, cable_id=1, tape_id=2)

        if m is not None:
            cable_d_px, tape_d_px, delta_px = m
            ratio = tape_d_px / max(1e-6, cable_d_px)

            if ratio < ratio_min:
                is_thin = True
                thin_count += 1
            elif ratio > ratio_max:
                is_thick = True
                thick_count += 1

        # 生成叠加图
        overlay = inferencer.overlay_mask(frame, mask, alpha=0.6)

        # 添加标注
        y_offset = 30

        # 帧信息
        cv2.putText(overlay, f"Frame: {frame_idx}/{total_frames}",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30

        # 优化模式标识
        cv2.putText(overlay, "Mode: Optimized (High Precision)",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y_offset += 25

        # 毛刺检测
        if has_burr:
            cv2.putText(overlay, f"BURR! area={burr_area}",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)
            y_offset += 35

        # 厚度检测
        if ratio is not None:
            cv2.putText(overlay, f"Cable: {cable_d_px:.0f}px",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            y_offset += 25
            cv2.putText(overlay, f"Tape: {tape_d_px:.0f}px",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25
            cv2.putText(overlay, f"Ratio: {ratio:.3f}",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += 25

            if is_thin:
                cv2.putText(overlay, f"THIN! ratio={ratio:.3f}",
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                y_offset += 35
            elif is_thick:
                cv2.putText(overlay, f"THICK! ratio={ratio:.3f}",
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
                y_offset += 35

        # 写入视频（每一帧都写入）
        video_writer.write(overlay)

        # 显示进度
        if frame_idx % 100 == 0:
            print(f"处理进度: {frame_idx}/{total_frames} 帧 ({frame_idx/total_frames*100:.1f}%)")

        # 实时预览窗口
        if show_preview:
            # 调整显示尺寸以便查看
            display_height = 720
            scale = display_height / height
            display_width = int(width * scale)
            display = cv2.resize(overlay, (display_width, display_height))

            cv2.imshow('Cable Wrap Detection - Optimized', display)

            # 按键处理
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC退出
                print("\n用户按ESC退出处理")
                break
            elif key == 32:  # 空格暂停
                print("\n已暂停，按任意键继续...")
                cv2.waitKey(0)

        # 保存事件截图
        if has_burr or is_thin or is_thick:
            # 保存原始帧
            snap_path = os.path.join(output_dir, 'snapshots', f"frame_{frame_idx:06d}.jpg")
            over_path = os.path.join(output_dir, 'overlays', f"frame_{frame_idx:06d}.jpg")

            # 使用cv2保存，如果失败则用PIL
            snap_success = cv2.imwrite(snap_path, frame)
            over_success = cv2.imwrite(over_path, overlay)

            # 如果cv2失败，尝试PIL
            if not snap_success or not over_success:
                try:
                    from PIL import Image
                    if not snap_success:
                        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        Image.fromarray(img_rgb).save(snap_path)
                        snap_success = True
                    if not over_success:
                        img_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                        Image.fromarray(img_rgb).save(over_path)
                        over_success = True
                except Exception as e:
                    print(f"[帧 {frame_idx}] 警告: 图片保存失败 - {e}")

            # 记录日志
            with open(log_path, 'a', encoding='utf-8') as f:
                if has_burr:
                    f.write(f"{frame_idx},burr_defect,NA,NA,NA,{burr_area}\n")
                if is_thin and ratio is not None:
                    m = measure_cable_tape_diameter_px(mask, cable_id=1, tape_id=2)
                    if m:
                        f.write(f"{frame_idx},wrap_thin,{ratio:.3f},{m[0]:.1f},{m[1]:.1f},{m[2]:.1f}\n")
                if is_thick and ratio is not None:
                    m = measure_cable_tape_diameter_px(mask, cable_id=1, tape_id=2)
                    if m:
                        f.write(f"{frame_idx},wrap_thick,{ratio:.3f},{m[0]:.1f},{m[1]:.1f},{m[2]:.1f}\n")

            ratio_str = f"{ratio:.3f}" if ratio is not None else "N/A"
            print(f"[帧 {frame_idx}] 毛刺={has_burr}, 厚度不足={is_thin}, 厚度过大={is_thick}, ratio={ratio_str}")

    # 释放资源
    video_writer.release()
    cap.release()
    if show_preview:
        cv2.destroyAllWindows()

    print("\n" + "="*70)
    print("处理完成!")
    print("="*70)
    print(f"  毛刺缺陷: {burr_count}")
    print(f"  厚度不足: {thin_count}")
    print(f"  厚度过大: {thick_count}")
    print(f"  总异常帧: {burr_count + thin_count + thick_count}")
    print(f"  检测视频: {output_video_path}")
    print(f"  输出目录: {output_dir}")
    print("\n优化效果:")
    print("  - 减少了胶带误检（覆盖在电缆上的假胶带）")
    print("  - 强制互斥：胶带不会标注在电缆区域")
    print("  - 空间约束：只保留合理位置的胶带")


def main():
    parser = argparse.ArgumentParser(description='电缆缠绕检测 - 优化版')
    parser.add_argument('--video', required=True, help='视频路径')
    parser.add_argument('--model', default=r'checkpoints\best_model.pth', help='模型路径')
    parser.add_argument('--output', required=True, help='输出目录')
    parser.add_argument('--num-classes', type=int, default=7, help='类别数')
    parser.add_argument('--device', default='cuda', help='设备 (cpu/cuda)')
    parser.add_argument('--ratio-min', type=float, default=1.05, help='tape/cable最小比例')
    parser.add_argument('--ratio-max', type=float, default=1.5, help='tape/cable最大比例')
    parser.add_argument('--min-area-px', type=int, default=50, help='毛刺最小像素面积')
    parser.add_argument('--show-preview', action='store_true', default=True, help='显示实时预览窗口')
    parser.add_argument('--no-preview', action='store_true', help='不显示实时预览窗口')

    args = parser.parse_args()
    show_preview = args.show_preview and not args.no_preview

    print("="*70)
    print("电缆缠绕均匀性检测 - 优化版")
    print("="*70)
    print(f"模型: {args.model}")
    print(f"视频: {args.video}")
    print(f"输出: {args.output}")
    print(f"检测类别: 电缆(1), 胶带(2), 毛刺(5)")
    print(f"厚度范围: {args.ratio_min:.2f} - {args.ratio_max:.2f}")
    if show_preview:
        print(f"实时预览: 开启")
    print("="*70)
    print()

    process_video(
        model_path=args.model,
        video_path=args.video,
        output_dir=args.output,
        num_classes=args.num_classes,
        ratio_min=args.ratio_min,
        ratio_max=args.ratio_max,
        min_area_px=args.min_area_px,
        device=args.device,
        show_preview=show_preview
    )


if __name__ == '__main__':
    main()
