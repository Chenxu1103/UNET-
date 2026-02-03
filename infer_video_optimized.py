"""
电缆包裹质量检测 - 优化版视频推理脚本

优化内容：
1. 多帧验证 - 减少误报
2. 置信度评分 - 提高检测可靠性
3. 智能事件过滤 - 去除噪点
4. 缺陷持续性追踪 - 追踪缺陷变化
5. 更精确的阈值 - 针对业务优化

支持类别:
  0: background
  1: cable (电缆)
  2: tape (胶带)
  3: bulge_defect (鼓包/局部厚度异常)
  4: loose_defect (脱落/翘边/明显断裂)
  5: damage_defect (白色电缆破损/毛边/划伤)
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
from collections import defaultdict, deque
from typing import List, Dict, Tuple

# 添加目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root))

from src.models.unetpp import NestedUNet
import importlib.util
spec = importlib.util.spec_from_file_location("diameter", str(project_root / "utils" / "diameter.py"))
diameter_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diameter_module)
measure_cable_tape_diameter_px = diameter_module.measure_cable_tape_diameter_px


# 类别配置
CLASS_NAMES = {
    0: 'background',
    1: 'cable',
    2: 'tape',
    3: 'bulge_defect',
    4: 'loose_defect',
    5: 'damage_defect',
}

CLASS_COLORS = {
    0: (0, 0, 0),
    1: (255, 0, 0),
    2: (0, 255, 0),
    3: (0, 0, 255),
    4: (255, 255, 0),
    5: (255, 0, 255),
}


class DefectTracker:
    """缺陷追踪器 - 追踪缺陷的持续性和变化"""

    def __init__(self, confirm_frames: int = 3, iou_threshold: float = 0.3):
        """
        Args:
            confirm_frames: 需要连续几帧确认缺陷（减少误报）
            iou_threshold: 缺陷重叠IOU阈值，用于判断是否为同一缺陷
        """
        self.confirm_frames = confirm_frames
        self.iou_threshold = iou_threshold

        # 活跃缺陷追踪 {defect_id: {'frames': deque, 'bbox': (x0,y0,x1,y1), 'count': int}}
        self.active_defects = {}

        # 缺陷ID计数器
        self.defect_id_counter = 0

        # 已确认缺陷列表
        self.confirmed_defects = []

    def calculate_iou(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """计算两个边界框的IOU"""
        x0_1, y0_1, x1_1, y1_1 = bbox1
        x0_2, y0_2, x1_2, y1_2 = bbox2

        # 计算交集
        x0_inter = max(x0_1, x0_2)
        y0_inter = max(y0_1, y0_2)
        x1_inter = min(x1_1, x1_2)
        y1_inter = min(y1_1, y1_2)

        if x1_inter <= x0_inter or y1_inter <= y0_inter:
            return 0.0

        inter_area = (x1_inter - x0_inter) * (y1_inter - y0_inter)

        # 计算并集
        area1 = (x1_1 - x0_1) * (y1_1 - y0_1)
        area2 = (x1_2 - x0_2) * (y1_2 - y0_2)
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def update(self, detections: List[Dict], frame_idx: int) -> List[Dict]:
        """
        更新缺陷追踪

        Args:
            detections: 当前帧的检测结果 [{'class_id': int, 'bbox': (x0,y0,x1,y1), 'area': int}]
            frame_idx: 当前帧索引

        Returns:
            确认的缺陷列表
        """
        confirmed = []

        # 标记当前帧的活跃缺陷
        current_frame_defects = set()

        for det in detections:
            class_id = det['class_id']
            bbox = det['bbox']
            area = det['area']

            # 尝试匹配到已有缺陷
            matched = False
            for defect_id, defect_info in list(self.active_defects.items()):
                if defect_info['class_id'] == class_id:
                    iou = self.calculate_iou(bbox, defect_info['bbox'])
                    if iou >= self.iou_threshold:
                        # 匹配成功，更新缺陷
                        defect_info['frames'].append(frame_idx)
                        defect_info['bbox'] = bbox  # 更新边界框
                        defect_info['area'] = area
                        defect_info['count'] += 1
                        current_frame_defects.add(defect_id)

                        # 检查是否达到确认条件
                        if defect_info['count'] >= self.confirm_frames:
                            # 计算置信度（基于持续性和面积）
                            confidence = min(0.95, 0.5 + (defect_info['count'] / self.confirm_frames) * 0.3)

                            confirmed.append({
                                'defect_id': defect_id,
                                'type': defect_info['type'],
                                'class_id': class_id,
                                'bbox': bbox,
                                'area': area,
                                'confidence': confidence,
                                'start_frame': defect_info['frames'][0],
                                'duration': defect_info['count']
                            })

                        matched = True
                        break

            if not matched:
                # 新缺陷
                defect_id = f"{class_id}_{frame_idx}_{self.defect_id_counter}"
                self.defect_id_counter += 1

                class_name = self._get_class_name(class_id)
                self.active_defects[defect_id] = {
                    'class_id': class_id,
                    'type': class_name,
                    'frames': deque([frame_idx], maxlen=self.confirm_frames + 5),
                    'bbox': bbox,
                    'area': area,
                    'count': 1
                }

        # 清理长时间未出现的缺陷
        to_remove = []
        for defect_id, defect_info in self.active_defects.items():
            if defect_id not in current_frame_defects:
                # 检查是否在最近几帧出现过
                if frame_idx - defect_info['frames'][-1] > 10:  # 10帧未出现
                    to_remove.append(defect_id)

        for defect_id in to_remove:
            del self.active_defects[defect_id]

        return confirmed

    def _get_class_name(self, class_id: int) -> str:
        """获取类别名称"""
        if class_id == 3:
            return 'bulge_defect'
        elif class_id == 4:
            return 'loose_defect'
        elif class_id == 5 or class_id == 6:
            return 'damage_defect'
        else:
            return f'unknown_defect_{class_id}'


class VideoInferenceOptimized:
    """优化版视频推理类"""

    def __init__(
        self,
        model_path: str,
        num_classes: int = 7,
        input_size: int = 256,
        device: str = 'cpu',
        # 优化参数
        min_defect_area: int = 100,  # 增大最小面积阈值
        confirm_frames: int = 3,      # 需要3帧确认
        edge_margin: int = 20,        # 边缘忽略区域
        min_cable_area: int = 1000,   # 最小电缆面积
        cable_coverage_threshold: float = 0.3  # 电缆覆盖率阈值
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        self.input_size = input_size

        # 优化参数
        self.min_defect_area = min_defect_area
        self.edge_margin = edge_margin
        self.min_cable_area = min_cable_area
        self.cable_coverage_threshold = cable_coverage_threshold

        print(f"设备: {device}")
        print(f"加载模型: {model_path}")
        print(f"优化参数:")
        print(f"  - 最小缺陷面积: {min_defect_area} px^2")
        print(f"  - 确认帧数: {confirm_frames}")
        print(f"  - 边缘忽略: {edge_margin} px")

        # 加载模型
        self.model = NestedUNet(
            num_classes=num_classes,
            input_channels=3,
            deep_supervision=False
        ).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'])
        elif 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()
        print("模型加载完成")

        # 缺陷追踪器
        self.defect_tracker = DefectTracker(confirm_frames=confirm_frames)

    def preprocess(self, frame_bgr: np.ndarray) -> torch.Tensor:
        """预处理图像"""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (self.input_size, self.input_size))
        frame_normalized = frame_resized.astype(np.float32) / 255.0
        frame_tensor = torch.from_numpy(frame_normalized).permute(2, 0, 1).unsqueeze(0)
        return frame_tensor.to(self.device)

    @torch.no_grad()
    def predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        """预测单帧图像"""
        input_tensor = self.preprocess(frame_bgr)
        output = self.model(input_tensor)

        if isinstance(output, list):
            output = output[0]

        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

        h, w = frame_bgr.shape[:2]
        pred_mask = cv2.resize(pred_mask.astype(np.uint8), (w, h),
                               interpolation=cv2.INTER_NEAREST)

        return pred_mask

    def overlay_mask(self, frame_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """将mask叠加到原始图像"""
        overlay = frame_bgr.copy()

        for class_id, color in CLASS_COLORS.items():
            if class_id == 0:
                continue
            overlay[mask == class_id] = color

        result = cv2.addWeighted(frame_bgr, 1 - alpha, overlay, alpha, 0)
        return result

    def validate_detection(self, mask: np.ndarray) -> Tuple[bool, List[Dict]]:
        """
        验证检测结果的有效性

        Returns:
            (is_valid, defects): 帧是否有效，缺陷列表
        """
        h, w = mask.shape

        # 1. 检查电缆覆盖率（避免误检空帧）
        cable_area = np.sum(mask == 1)
        total_area = h * w
        cable_coverage = cable_area / total_area

        if cable_area < self.min_cable_area:
            return False, []

        if cable_coverage < self.cable_coverage_threshold:
            return False, []

        # 2. 提取缺陷候选
        defects = []

        # 缺陷类别（模型输出）
        defect_classes = [3, 4, 5, 6]  # bulge, loose, burr, thin

        for class_id in defect_classes:
            defect_mask = (mask == class_id)
            area = np.sum(defect_mask)

            # 面积过滤
            if area < self.min_defect_area:
                continue

            # 获取边界框
            ys, xs = np.where(defect_mask)
            y0, y1 = int(ys.min()), int(ys.max())
            x0, x1 = int(xs.min()), int(xs.max())

            # 边缘过滤（避免边缘噪声）
            if (x0 < self.edge_margin or x1 > w - self.edge_margin or
                y0 < self.edge_margin or y1 > h - self.edge_margin):
                # 如果大部分在边缘，跳过
                center_x, center_y = (x0 + x1) // 2, (y0 + y1) // 2
                edge_pixels = 0
                total_pixels = (x1 - x0) * (y1 - y0)

                # 简化的边缘检查
                if x0 < self.edge_margin:
                    edge_pixels += (self.edge_margin - x0) * (y1 - y0)
                if x1 > w - self.edge_margin:
                    edge_pixels += (x1 - (w - self.edge_margin)) * (y1 - y0)
                if y0 < self.edge_margin:
                    edge_pixels += (self.edge_margin - y0) * (x1 - x0)
                if y1 > h - self.edge_margin:
                    edge_pixels += (y1 - (h - self.edge_margin)) * (x1 - x0)

                if edge_pixels / total_pixels > 0.5:
                    continue

            defects.append({
                'class_id': class_id,
                'bbox': (x0, y0, x1, y1),
                'area': area
            })

        return True, defects


def process_video_optimized(
    model_path: str,
    video_path: str,
    output_dir: str,
    num_classes: int = 7,
    input_size: int = 256,
    turn_hz: float = 3.0,
    eval_per_turn: int = 1,
    px_per_mm: float = 0.0,
    delta_mm: float = 20.0,
    tol_mm: float = 5.0,
    ratio_min: float = 1.05,
    ratio_max: float = 1.5,
    min_area_px: int = 100,
    device: str = 'cpu',
    save_overlay: bool = True,
    show_preview: bool = False,
    delay_ms: int = 0,
    simulate_production: bool = False,
    production_fps: float = 10.0,
    # 优化参数
    confirm_frames: int = 3,
    edge_margin: int = 20,
):
    """处理视频文件 - 优化版"""

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'snapshots'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'overlays'), exist_ok=True)

    # 初始化推理器
    inferencer = VideoInferenceOptimized(
        model_path=model_path,
        num_classes=num_classes,
        input_size=input_size,
        device=device,
        min_defect_area=min_area_px,
        confirm_frames=confirm_frames,
        edge_margin=edge_margin
    )

    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"视频信息: {width}x{height} @ {fps:.2f}fps, 共 {total_frames} 帧")

    # 生产速度模拟配置
    if simulate_production:
        print(f"\n生产速度模拟模式:")
        print(f"  生产检测速度: {production_fps:.1f} 帧/秒")
        print(f"  每帧处理时间: {1000/production_fps:.1f} 毫秒")
        delay_ms = int(1000 / production_fps)
    elif delay_ms > 0:
        print(f"\n手动延迟模式: {delay_ms} 毫秒/帧")
    else:
        print(f"\n快速处理模式（无延迟）")

    # 计算采样间隔
    stride = max(1, int(round(fps / (turn_hz * eval_per_turn))))
    print(f"采样间隔: 每 {stride} 帧处理一次")

    # 日志文件
    log_path = os.path.join(output_dir, 'events.csv')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("timestamp,frame_idx,event_type,confidence,detail\n")

    # 处理视频
    frame_idx = 0
    processed_count = 0
    event_count = 0

    print("\n开始处理视频...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # 跳帧采样
        if frame_idx % stride != 0:
            continue

        processed_count += 1

        # 预测
        mask = inferencer.predict(frame)

        # 验证检测
        is_valid, defects = inferencer.validate_detection(mask)

        if not is_valid:
            continue

        # 更新缺陷追踪
        confirmed_defects = inferencer.defect_tracker.update(defects, frame_idx)

        # 生成叠加图
        overlay = inferencer.overlay_mask(frame, mask, alpha=0.5)

        # 检测直径/厚度异常
        m = measure_cable_tape_diameter_px(mask, cable_id=1, tape_id=2)
        ratio_info = ""
        if m is not None:
            cable_d_px, tape_d_px, delta_px = m
            ratio = tape_d_px / max(1e-6, cable_d_px)
            ratio_info = f"ratio={ratio:.3f},cable={cable_d_px:.0f},tape={tape_d_px:.0f}"

            # 报告厚度异常（使用更严格的阈值）
            if ratio < ratio_min:
                confirmed_defects.append({
                    'defect_id': f"thin_{frame_idx}",
                    'type': 'thin_wrap',
                    'class_id': None,
                    'bbox': None,
                    'area': None,
                    'confidence': 0.9,
                    'start_frame': frame_idx,
                    'duration': 1,
                    'detail': f"{ratio_info}"
                })
            elif ratio > ratio_max:
                confirmed_defects.append({
                    'defect_id': f"thick_{frame_idx}",
                    'type': 'thick_wrap',
                    'class_id': None,
                    'bbox': None,
                    'area': None,
                    'confidence': 0.9,
                    'start_frame': frame_idx,
                    'duration': 1,
                    'detail': f"{ratio_info}"
                })

        # 保存确认的事件
        if confirmed_defects:
            event_count += len(confirmed_defects)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")

            # 保存截图
            snap_path = os.path.join(output_dir, 'snapshots', f"{ts}_f{frame_idx}.jpg")
            over_path = os.path.join(output_dir, 'overlays', f"{ts}_f{frame_idx}.jpg")
            cv2.imwrite(snap_path, frame)
            cv2.imwrite(over_path, overlay)

            # 记录日志
            with open(log_path, 'a', encoding='utf-8') as f:
                for defect in confirmed_defects:
                    detail = defect.get('detail', f"bbox={defect['bbox']},area={defect['area']}")
                    f.write(f"{ts},{frame_idx},{defect['type']},{defect['confidence']:.2f},{detail}\n")

            event_types = [d['type'] for d in confirmed_defects]
            print(f"  [帧 {frame_idx}] 确认事件: {', '.join(event_types)} (共{len(confirmed_defects)}个)")

            # 在叠加图上标注确认的缺陷
            for defect in confirmed_defects:
                if defect['bbox']:
                    x0, y0, x1, y1 = defect['bbox']
                    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 255), 3)
                    label = f"{defect['type']} {defect['confidence']:.2f}"
                    cv2.putText(overlay, label, (x0, max(0, y0 - 10)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 实时预览
        if show_preview:
            display = overlay.copy()

            # 添加实时信息
            cv2.putText(display, f"Frame: {frame_idx}/{total_frames}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(display, f"Events: {event_count}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            if m is not None:
                cv2.putText(display, ratio_info, (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            display_resized = cv2.resize(display, (1024, 768))
            cv2.imshow('Optimized Cable Inspection', display_resized)

            if delay_ms > 0:
                key = cv2.waitKey(max(1, delay_ms)) & 0xFF
            else:
                key = cv2.waitKey(1) & 0xFF

            if key == 27:
                break
            elif key == 32:
                cv2.waitKey(0)
        elif delay_ms > 0:
            import time
            time.sleep(delay_ms / 1000.0)

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n处理完成!")
    print(f"  总帧数: {frame_idx}")
    print(f"  处理帧数: {processed_count}")
    print(f"  检测事件: {event_count}")
    print(f"  结果保存在: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='电缆包裹质量检测 - 优化版')

    # 模型参数
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pth')
    parser.add_argument('--num-classes', type=int, default=7)
    parser.add_argument('--input-size', type=int, default=256)
    parser.add_argument('--device', type=str, default='cpu')

    # 视频参数
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--output', type=str, default='log')

    # 采样参数
    parser.add_argument('--turn-hz', type=float, default=3.0)
    parser.add_argument('--eval-per-turn', type=int, default=1)

    # 检测参数
    parser.add_argument('--px-per-mm', type=float, default=0.0)
    parser.add_argument('--delta-mm', type=float, default=20.0)
    parser.add_argument('--tol-mm', type=float, default=5.0)
    parser.add_argument('--ratio-min', type=float, default=1.05)
    parser.add_argument('--ratio-max', type=float, default=1.5)
    parser.add_argument('--min-area-px', type=int, default=100, help='最小缺陷面积（像素²）')

    # 优化参数
    parser.add_argument('--confirm-frames', type=int, default=3, help='需要连续几帧确认缺陷')
    parser.add_argument('--edge-margin', type=int, default=20, help='边缘忽略区域（像素）')

    # 显示参数
    parser.add_argument('--save-overlay', action='store_true', default=True)
    parser.add_argument('--show-preview', action='store_true')

    # 生产速度模拟参数
    parser.add_argument('--delay-ms', type=int, default=0)
    parser.add_argument('--simulate-production', action='store_true')
    parser.add_argument('--production-fps', type=float, default=10.0)

    args = parser.parse_args()

    process_video_optimized(
        model_path=args.model,
        video_path=args.video,
        output_dir=args.output,
        num_classes=args.num_classes,
        input_size=args.input_size,
        turn_hz=args.turn_hz,
        eval_per_turn=args.eval_per_turn,
        px_per_mm=args.px_per_mm,
        delta_mm=args.delta_mm,
        tol_mm=args.tol_mm,
        ratio_min=args.ratio_min,
        ratio_max=args.ratio_max,
        min_area_px=args.min_area_px,
        device=args.device,
        save_overlay=args.save_overlay,
        show_preview=args.show_preview,
        delay_ms=args.delay_ms,
        simulate_production=args.simulate_production,
        production_fps=args.production_fps,
        confirm_frames=args.confirm_frames,
        edge_margin=args.edge_margin
    )


if __name__ == '__main__':
    main()
