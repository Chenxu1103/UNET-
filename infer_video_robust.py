"""
3类模型鲁棒推理 - 实现互斥分割 + 空间约束 + 时间门控

关键改进：
1. RGB + letterbox预处理（与训练对齐）
2. 互斥分割（解决蓝绿混叠）
3. cable强约束（最大连通域）
4. tape环带约束（只能在cable外侧）
5. ROI截断（清除背景假阳性）
6. 事件门控（连续帧确认 + 冷却）
"""
import os
import cv2
import argparse
import time
import datetime
import numpy as np
import torch
import json
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import deque
from typing import List, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from models.unetpp import NestedUNet


# 类别配置
CLASS_NAMES = {0: '背景', 1: '电缆', 2: '胶带'}
CLASS_COLORS = {
    0: (0, 0, 0),         # 黑色：背景
    1: (255, 0, 0),       # 蓝色：电缆
    2: (0, 255, 0),       # 绿色：胶带
}


def letterbox_rgb(frame_bgr, new_size=512):
    """Letterbox预处理：等比缩放+padding，保证形状不变"""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = frame_rgb.shape[:2]
    scale = new_size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(frame_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)

    top = (new_size - nh) // 2
    left = (new_size - nw) // 2
    canvas = np.zeros((new_size, new_size, 3), dtype=resized.dtype)
    canvas[top:top+nh, left:left+nw] = resized
    meta = (scale, top, left, nh, nw, h, w)
    return canvas, meta


def unletterbox_mask(mask_512, meta):
    """将预测mask恢复到原始尺寸"""
    scale, top, left, nh, nw, h, w = meta
    crop = mask_512[top:top+nh, left:left+nw]
    out = cv2.resize(crop.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return out


def softmax_np(x):
    """Numpy softmax"""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def exclusive_threshold(probs, t_cable=0.55, t_tape=0.60, bg_margin=0.20, ct_margin=0.10):
    """
    互斥阈值argmax：解决蓝绿混叠

    Args:
        probs: HxWx3, (bg,cable,tape)
        t_cable: 电缆置信度阈值
        t_tape: 胶带置信度阈值
        bg_margin: 与背景的最小margin
        ct_margin: cable/tape间的margin
    """
    pbg = probs[..., 0]
    pc = probs[..., 1]
    pt = probs[..., 2]

    # 候选：置信度够 + 比背景高出margin
    cand_c = (pc >= t_cable) & (pc >= pbg + bg_margin)
    cand_t = (pt >= t_tape) & (pt >= pbg + bg_margin)

    # 互斥：谁更强给谁
    cable = cand_c & (pc >= pt + ct_margin)
    tape = cand_t & (pt >= pc + ct_margin)

    # 强制互斥（仍有少量重叠时）
    overlap = cable & tape
    if np.any(overlap):
        cable[overlap] = pc[overlap] >= pt[overlap]
        tape[overlap] = ~cable[overlap]

    return cable.astype(np.uint8), tape.astype(np.uint8)


def keep_best_cable_cc(mask: np.ndarray,
                       min_area: int = 2000,
                       min_h_ratio: float = 0.35,
                       min_aspect: float = 3.0,
                       max_w_ratio: float = 0.20,
                       debug=False) -> np.ndarray:
    """
    保留"最像电缆"的连通域（形状约束，而非最大面积）

    电缆特征：细长、纵向、高度占比大、宽度不能太大
    """
    H, W = mask.shape
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return np.zeros_like(mask)

    best_i = -1
    best_score = -1e9
    candidates = []

    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        if h < int(min_h_ratio * H):
            continue
        if w > int(max_w_ratio * W):
            continue
        aspect = h / (w + 1e-6)
        if aspect < min_aspect:
            continue

        # 分数：更高、更细长、面积适中的优先
        score = (h / H) * 3.0 + min(aspect, 12.0) * 0.5 + (area / (H * W)) * 0.5
        candidates.append({
            'id': i,
            'score': score,
            'area': area,
            'h_ratio': h / H,
            'w_ratio': w / W,
            'aspect': aspect
        })

        if score > best_score:
            best_score = score
            best_i = i

    if best_i == -1:
        if debug:
            print(f"  [Cable Shape] 没有找到符合条件的连通域")
        return np.zeros_like(mask)

    if debug and len(candidates) > 0:
        print(f"  [Cable Shape] 找到 {len(candidates)} 个候选，选择最佳:")
        for c in sorted(candidates, key=lambda x: x['score'], reverse=True)[:3]:
            print(f"    ID={c['id']}: score={c['score']:.2f}, area={c['area']}, "
                  f"h_ratio={c['h_ratio']:.2f}, w_ratio={c['w_ratio']:.2f}, aspect={c['aspect']:.1f}")

    return (labels == best_i).astype(np.uint8)


def keep_largest_cc(bin_mask, min_area=8000):
    """只保留最大连通域（已弃用，保留接口兼容性）"""
    # 改为调用形状约束版本
    return keep_best_cable_cc(bin_mask, min_area=min_area)


def restrict_tape_to_cable_ring(mask_tape, mask_cable, band_out=26, band_in=2, min_area=500):
    """
    胶带环带约束：只能在cable外侧环带且必须贴近cable

    Args:
        band_out: 环带外边界（距离cable的最大像素距离）
        band_in: 环带内边界（距离cable的最小像素距离）
    """
    if mask_cable.sum() == 0:
        return np.zeros_like(mask_tape)

    # 距离变换：对"电缆外部"算距离到电缆
    inv_cable = (mask_cable == 0).astype(np.uint8)
    dist = cv2.distanceTransform(inv_cable, cv2.DIST_L2, 3)

    # 定义环带区域
    ring = (dist >= band_in) & (dist <= band_out)
    tape = (mask_tape.astype(bool) & ring & (~mask_cable.astype(bool))).astype(np.uint8)

    # 去噪：移除小连通域
    num, labels, stats, _ = cv2.connectedComponentsWithStats(tape, 8)
    out = np.zeros_like(tape)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 1

    # 轻微闭运算补边
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel3, iterations=1)
    return out


def apply_roi_limit(mask, cable_mask, pad=80):
    """ROI截断：只在cable bbox附近保留预测（彻底清除背景假阳性）"""
    if cable_mask.sum() == 0:
        return np.zeros_like(mask)
    ys, xs = np.where(cable_mask > 0)
    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()
    h, w = mask.shape[:2]
    y1 = max(0, y1 - pad)
    y2 = min(h-1, y2 + pad)
    x1 = max(0, x1 - pad)
    x2 = min(w-1, x2 + pad)

    out = np.zeros_like(mask)
    out[y1:y2+1, x1:x2+1] = mask[y1:y2+1, x1:x2+1]
    return out


class EventGate:
    """事件门控：连续帧确认 + 冷却"""
    def __init__(self, hold_frames=8, cooldown_sec=3.0):
        self.hold_frames = hold_frames
        self.cooldown_sec = cooldown_sec
        self.hist = deque(maxlen=hold_frames)
        self.last_fire = 0.0

    def update(self, is_abnormal: bool):
        self.hist.append(1 if is_abnormal else 0)

    def should_fire(self):
        if len(self.hist) < self.hold_frames:
            return False
        if sum(self.hist) < self.hold_frames:
            return False
        now = time.time()
        if now - self.last_fire < self.cooldown_sec:
            return False
        self.last_fire = now
        return True


@dataclass
class DetectionEvent:
    timestamp: str
    frame_id: int
    dc_px: float
    dt_px: float
    delta_d_px: float
    cable_coverage: float
    tape_coverage: float


class VideoInferenceRobust:
    """鲁棒视频推理器"""

    def __init__(self, model_path: str, num_classes: int = 3, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        self.input_size = 512  # 必须对齐训练尺寸

        print(f"使用设备: {self.device}")
        print(f"类别数: {num_classes} (0:bg, 1:cable, 2:tape)")

        # 构建模型
        self.model = NestedUNet(
            num_classes=num_classes,
            input_channels=3,
            deep_supervision=False
        ).to(self.device)

        # 加载权重
        print(f"加载模型: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get('model', checkpoint.get('model_state_dict', checkpoint))

        # 严格加载：发现不一致直接报错
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

        miou = checkpoint.get('best_miou', checkpoint.get('miou', 0.0))
        print(f"模型加载成功 (mIoU {miou:.2%})")
        print("后处理方式: 互斥分割 + 形状约束 + 环带约束 + ROI截断")

    def infer_frame(self, frame: np.ndarray, debug=False):
        """
        推理单帧 - 鲁棒版本

        Returns:
            mask_cable, mask_tape, metrics, debug_info
        """
        h, w = frame.shape[:2]

        # 1. Letterbox预处理（RGB + 等比缩放 + padding）
        canvas, meta = letterbox_rgb(frame, new_size=512)

        # 2. 归一化到[0,1]
        img_tensor = canvas.astype(np.float32) / 255.0
        img_tensor = np.transpose(img_tensor, (2, 0, 1))  # HWC -> CHW
        img_tensor = torch.from_numpy(img_tensor).unsqueeze(0).to(self.device)

        # 3. 推理
        with torch.no_grad():
            outputs = self.model(img_tensor)
            if isinstance(outputs, list):
                outputs = outputs[-1]

        # 4. 输出通道断言（确保模型输出类别数正确）
        assert outputs.shape[1] == self.num_classes, \
            f"Model output channels={outputs.shape[1]} != num_classes={self.num_classes}"

        # 5. Softmax + 互斥分割（使用稳健阈值）
        probs = softmax_np(outputs[0].cpu().numpy().transpose(1, 2, 0))
        mask_cable_512, mask_tape_512 = exclusive_threshold(
            probs,
            t_cable=0.50,      # 提高到稳健阈值
            t_tape=0.42,       # 提高到稳健阈值
            bg_margin=0.15,    # 降低margin，提高召回
            ct_margin=0.10
        )

        # 6. Unletterbox恢复到原始尺寸
        mask_cable = unletterbox_mask(mask_cable_512, meta)
        mask_tape = unletterbox_mask(mask_tape_512, meta)

        # 7. Cable后处理：闭运算 + 形状约束（选最像电缆的连通域）
        kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_cable = cv2.morphologyEx(mask_cable, cv2.MORPH_CLOSE, kernel5, iterations=1)
        mask_cable = keep_best_cable_cc(
            mask_cable,
            min_area=2000,
            min_h_ratio=0.35,
            min_aspect=3.0,
            max_w_ratio=0.20,
            debug=debug
        )

        # 8. Tape后处理：环带约束（收紧环带范围）
        mask_tape = restrict_tape_to_cable_ring(
            mask_tape,
            mask_cable,
            band_out=20,      # 收紧环带到20px
            band_in=2,
            min_area=500
        )

        # 9. ROI截断（清除背景假阳性）
        mask_cable = apply_roi_limit(mask_cable, mask_cable, pad=80)
        mask_tape = apply_roi_limit(mask_tape, mask_cable, pad=80)

        # 10. 几何测量
        dc_px, dt_px, delta_d_px = self._measure_diameters(mask_cable, mask_tape)

        # 11. 计算覆盖率
        cable_coverage = mask_cable.sum() / mask_cable.size if mask_cable.size > 0 else 0
        tape_coverage = mask_tape.sum() / mask_tape.size if mask_tape.size > 0 else 0

        metrics = {
            'dc_px': dc_px,
            'dt_px': dt_px,
            'delta_d_px': delta_d_px,
            'cable_coverage': cable_coverage,
            'tape_coverage': tape_coverage
        }

        debug_info = {
            'probs': probs,
            'pred_mask': np.argmax(probs, axis=-1)
        }

        return mask_cable, mask_tape, metrics, debug_info

    def _measure_diameters(self, mask_cable, mask_tape):
        """测量电缆和胶带直径"""
        def calc_width(mask):
            H, W = mask.shape
            widths = []
            for y in range(H):
                xs = np.where(mask[y] > 0)[0]
                if xs.size > 1:
                    widths.append(xs.max() - xs.min() + 1)
            return np.median(widths) if widths else 0

        dc_px = calc_width(mask_cable)
        dt_px = calc_width(mask_tape)
        delta_d_px = dt_px - dc_px if dc_px > 0 else 0

        return dc_px, dt_px, delta_d_px

    def create_overlay(self, frame, mask_cable, mask_tape, metrics):
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
            f"Cable(Blue): {metrics['dc_px']:.1f}px ({metrics['cable_coverage']*100:.2f}%)",
            f"Tape(Green): {metrics['dt_px']:.1f}px ({metrics['tape_coverage']*100:.2f}%)",
            f"Delta: {metrics['delta_d_px']:.1f}px",
            f"Mode: Robust (Exclusive+Ring+ROI)",
        ]

        for text in texts:
            cv2.putText(overlay, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += 25

        return overlay


def main():
    parser = argparse.ArgumentParser(description='3类模型鲁棒推理')
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--model', type=str, default='checkpoints_3class_finetuned/best_model.pth')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--frame-stride', type=int, default=1)
    parser.add_argument('--show-preview', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    # 生成输出目录
    if args.output is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"log/detection_robust_{timestamp}"

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("3类模型鲁棒推理 - 互斥分割 + 空间约束")
    print("="*70)
    print(f"输入视频: {args.video}")
    print(f"输出目录: {output_dir}")
    print(f"模型: {args.model}")
    print(f"检测模式: 鲁棒模式（互斥+环带+ROI）")
    print("="*70)
    print()

    # 初始化推理器
    inferencer = VideoInferenceRobust(
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

    # 统计信息
    events = []
    frame_count = 0
    processing_count = 0

    all_dc = []
    all_dt = []
    all_delta_d = []

    print("开始推理...")
    print("="*70)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # 跳帧
            if frame_count % args.frame_stride != 0:
                continue

            processing_count += 1

            # 推理（每10帧输出一次详细调试信息）
            debug_mode = args.debug and (processing_count % 10 == 0)
            mask_cable, mask_tape, metrics, debug_info = inferencer.infer_frame(frame, debug=debug_mode)

            # 调试信息
            if args.debug and processing_count % 30 == 0:
                unique, counts = np.unique(debug_info['pred_mask'], return_counts=True)
                class_dist = {int(u): int(c) for u, c in zip(unique, counts)}
                print(f"[Frame {frame_count}] Cable={metrics['cable_coverage']*100:.1f}%, "
                      f"Tape={metrics['tape_coverage']*100:.1f}%")
                print(f"  ClassDist: {class_dist}")
                print(f"  MaxProbs: BG={debug_info['probs'][...,0].max():.3f}, "
                      f"Cable={debug_info['probs'][...,1].max():.3f}, "
                      f"Tape={debug_info['probs'][...,2].max():.3f}")
                print(f"  Diameters: Dc={metrics['dc_px']:.1f}px, Dt={metrics['dt_px']:.1f}px")

            # 记录统计
            if metrics['dc_px'] > 0:
                all_dc.append(metrics['dc_px'])
                all_dt.append(metrics['dt_px'])
                all_delta_d.append(metrics['delta_d_px'])

                event = DetectionEvent(
                    timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    frame_id=frame_count,
                    dc_px=metrics['dc_px'],
                    dt_px=metrics['dt_px'],
                    delta_d_px=metrics['delta_d_px'],
                    cable_coverage=metrics['cable_coverage'],
                    tape_coverage=metrics['tape_coverage']
                )
                events.append(asdict(event))

            # 创建叠加图
            overlay = inferencer.create_overlay(frame, mask_cable, mask_tape, metrics)
            output_video.write(overlay)

            # 显示预览
            if args.show_preview:
                display_height = 720
                h, w = overlay.shape[:2]
                scale = display_height / h
                display_width = int(w * scale)
                display = cv2.resize(overlay, (display_width, display_height))
                cv2.imshow('Robust Detection', display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # 打印进度
            if processing_count % 30 == 0:
                print(f"Frame {frame_count}/{total_frames}: Dc={metrics['dc_px']:.1f}, "
                      f"Dt={metrics['dt_px']:.1f}, DeltaD={metrics['delta_d_px']:.1f}")

    except KeyboardInterrupt:
        print("\n检测中断")

    finally:
        cap.release()
        output_video.release()
        cv2.destroyAllWindows()

    # 保存结果
    print()
    print("="*70)
    print("推理完成！")
    print("="*70)

    if len(all_dc) > 0:
        stats = {
            "total_frames_processed": len(all_dc),
            "dc_px_mean": float(np.mean(all_dc)),
            "dc_px_std": float(np.std(all_dc)),
            "dt_px_mean": float(np.mean(all_dt)),
            "dt_px_std": float(np.std(all_dt)),
            "delta_d_px_mean": float(np.mean(all_delta_d)),
            "delta_d_px_std": float(np.std(all_delta_d)),
            "delta_d_px_min": float(np.min(all_delta_d)),
            "delta_d_px_max": float(np.max(all_delta_d))
        }

        print("\n统计结果:")
        print(f"  处理帧数: {stats['total_frames_processed']}")
        print(f"  电缆直径: {stats['dc_px_mean']:.1f} ± {stats['dc_px_std']:.1f} px")
        print(f"  胶带外径: {stats['dt_px_mean']:.1f} ± {stats['dt_px_std']:.1f} px")
        print(f"  厚度增量: {stats['delta_d_px_mean']:.1f} ± {stats['delta_d_px_std']:.1f} px")
        print(f"  厚度范围: {stats['delta_d_px_min']:.1f} - {stats['delta_d_px_max']:.1f} px")

        with open(output_dir / "statistics.json", 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        with open(output_dir / "events.jsonl", 'w', encoding='utf-8') as f:
            for event in events:
                f.write(json.dumps(event, ensure_ascii=False) + '\n')

        print(f"\n事件记录: {len(events)} 条")

    print(f"\n输出文件:")
    print(f"  视频结果: {output_dir / 'result.mp4'}")
    print(f"  统计数据: {output_dir / 'statistics.json'}")
    print(f"  事件记录: {output_dir / 'events.jsonl'}")
    print("="*70)


if __name__ == '__main__':
    main()
