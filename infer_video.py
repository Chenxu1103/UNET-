"""
电缆包裹质量检测 - 视频推理脚本

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
from dataclasses import dataclass

# 添加目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root))

from src.models.unetpp import NestedUNet
# from src.infer.postprocess import compute_metrics
# from src.infer.decision import decide
# 直接导入 diameter 函数（utils 不是包）
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
    3: '鼓包缺陷',   # 鼓包
    4: '松脱缺陷',   # 脱落
    5: '破损缺陷',  # 破损/毛刺
    6: '厚度不足缺陷',    # 厚度不足（如存在）
}

# 类别颜色 (BGR格式)
CLASS_COLORS = {
    0: (0, 0, 0),         # 黑色：背景
    1: (255, 0, 0),       # 蓝色：电缆
    2: (0, 255, 0),       # 绿色：胶带
    3: (0, 0, 255),       # 红色：鼓包缺陷
    4: (255, 255, 0),     # 青色：松脱缺陷
    5: (255, 0, 255),     # 洋红：破损缺陷
    6: (0, 165, 255),     # 橙色：第6类（thin_defect）
}


@dataclass
class QualityResult:
    is_bad: bool
    lap_var: float
    gray_std: float
    mad: float
    reason: str


class FrameQualityGate:
    """
    帧质量闸门：用于过滤掉帧/强模糊/异常帧，避免误报（如 loose 乱飞）。
    仅做轻量计算（灰度、Laplacian方差、与上一帧差分）。
    """
    def __init__(
        self,
        enable: bool = True,
        blur_th: float = 80.0,     # Laplacian variance 小于该值通常较模糊（需现场微调）
        flat_th: float = 8.0,      # 灰度标准差过小：可能掉帧/异常（画面近乎一片灰）
        motion_th: float = 10.0,   # 与上一帧平均绝对差过大：说明运动/抖动明显
        glitch_flat_th: float = 3.0  # 更严格的"异常帧"标准差阈值
    ):
        self.enable = enable
        self.blur_th = float(blur_th)
        self.flat_th = float(flat_th)
        self.motion_th = float(motion_th)
        self.glitch_flat_th = float(glitch_flat_th)

    def check(self, frame_bgr: np.ndarray, prev_gray: np.ndarray | None) -> tuple[QualityResult, np.ndarray]:
        if not self.enable:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            return QualityResult(False, 0.0, float(gray.std()), 0.0, "disabled"), gray

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray_std = float(gray.std())
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        if prev_gray is None:
            mad = 0.0
        else:
            mad = float(cv2.absdiff(gray, prev_gray).mean())

        # 1) 异常/掉帧：画面几乎一片灰（std极低），不做识别
        if gray_std < self.glitch_flat_th:
            return QualityResult(True, lap_var, gray_std, mad, "revealed_glitch_frame(std<glitch_flat_th)"), gray

        # 2) 强模糊 + 明显运动：最容易产生"松脱/鼓包"伪检
        if (lap_var < self.blur_th) and (mad > self.motion_th):
            return QualityResult(True, lap_var, gray_std, mad, "motion_blur(lap<th & mad>th)"), gray

        # 3) 画面过平：也很容易误检（尤其低照/解码异常）
        if gray_std < self.flat_th:
            return QualityResult(True, lap_var, gray_std, mad, "too_flat(std<flat_th)"), gray

        return QualityResult(False, lap_var, gray_std, mad, "ok"), gray


class VideoInference:
    """视频推理类"""

    def __init__(
        self,
        model_path: str,
        num_classes: int = 7,
        input_size: int = 256,
        device: str = 'cpu'
    ):
        """初始化推理器"""
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        self.input_size = input_size

        print(f"设备: {self.device}")
        print(f"加载模型: {model_path}")

        # 加载模型
        self.model = NestedUNet(
            num_classes=num_classes,
            input_channels=3,
            deep_supervision=False
        ).to(self.device)

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

    def preprocess(self, frame_bgr: np.ndarray) -> torch.Tensor:
        """预处理图像"""
        # BGR -> RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # 调整大小
        frame_resized = cv2.resize(frame_rgb, (self.input_size, self.input_size))

        # 归一化
        frame_normalized = frame_resized.astype(np.float32) / 255.0

        # HWC -> CHW -> NCHW
        frame_tensor = torch.from_numpy(frame_normalized).permute(2, 0, 1).unsqueeze(0)

        return frame_tensor.to(self.device)

    @torch.no_grad()
    def predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        """预测单帧图像：干净的后处理，避免类别互相侵蚀"""
        # 预处理
        input_tensor = self.preprocess(frame_bgr)

        # 推理
        output = self.model(input_tensor)

        # 获取预测mask
        if isinstance(output, list):
            output = output[0]

        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

        # 调整回原始尺寸
        h, w = frame_bgr.shape[:2]
        pred_mask = cv2.resize(pred_mask.astype(np.uint8), (w, h),
                               interpolation=cv2.INTER_NEAREST)

        # 1. 移除未训练的class 4 (loose) -> 背景类
        pred_mask[pred_mask == 4] = 0

        # 2. 保存原始预测（已移除loose）
        raw_mask = pred_mask.copy()

        # 3. 对cable和tape做轻微闭运算，改善连通性
        kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        # 从原始预测中提取cable和tape
        mask_cable = (raw_mask == 1).astype(np.uint8)
        mask_tape = (raw_mask == 2).astype(np.uint8)
        mask_defect = ((raw_mask == 3) | (raw_mask == 5) | (raw_mask == 6)).astype(np.uint8)

        # 对cable和tape做闭运算
        mask_cable = cv2.morphologyEx(mask_cable, cv2.MORPH_CLOSE, kernel3, iterations=1)
        mask_tape = cv2.morphologyEx(mask_tape, cv2.MORPH_CLOSE, kernel3, iterations=1)

        # 4. 按优先级合并，避免重叠：defect > tape > cable > background
        pred_mask = np.zeros_like(raw_mask)
        pred_mask[mask_cable > 0] = 1      # 先放cable
        pred_mask[mask_tape > 0] = 2       # tape覆盖cable
        pred_mask[mask_defect > 0] = raw_mask[mask_defect > 0]  # defect覆盖tape

        return pred_mask

    def overlay_mask(self, frame_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """将mask叠加到原始图像：只在mask区域混合，避免整帧变暗"""
        h, w = frame_bgr.shape[:2]

        # 创建纯色mask图层
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)

        for class_id, color in CLASS_COLORS.items():
            if class_id == 0:
                continue  # 跳过背景
            if class_id >= self.num_classes:
                continue

            # 创建类别mask
            class_mask = (mask == class_id).astype(np.uint8)
            color_mask[class_mask > 0] = color

        # 只在 mask>0 区域混合（修复：避免背景被整体压暗）
        result = frame_bgr.copy()
        region = (mask > 0)
        if np.any(region):
            blended = ((1 - alpha) * frame_bgr.astype(np.float32) + alpha * color_mask.astype(np.float32)).astype(np.uint8)
            result[region] = blended[region]

        # 添加细轮廓让边界清晰但不粗
        for class_id, color in CLASS_COLORS.items():
            if class_id == 0:
                continue
            if class_id >= self.num_classes:
                continue

            # 使用原始mask（未膨胀）获取轮廓
            class_mask = (mask == class_id).astype(np.uint8) * 255
            contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 绘制细轮廓（2px，不过粗）
            cv2.drawContours(result, contours, -1, color, 2)

        return result


def process_video(
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
    min_area_px: int = 50,
    device: str = 'cpu',
    save_overlay: bool = True,
    show_preview: bool = False,
    delay_ms: int = 0,
    simulate_production: bool = False,
    production_fps: float = 10.0,
    enable_window_aggregation: bool = False,
    window_duration_sec: float = 3.0,
    min_frames_per_window: int = 6
):
    """处理视频文件"""

    # 确保 time 模块在局部作用域可用
    import time

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'snapshots'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'overlays'), exist_ok=True)

    # 初始化推理器
    inferencer = VideoInference(
        model_path=model_path,
        num_classes=num_classes,
        input_size=input_size,
        device=device
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

    # 窗口聚合模式配置
    if enable_window_aggregation:
        print(f"\n窗口聚合模式:")
        print(f"  窗口时长: {window_duration_sec}秒")
        print(f"  最小帧数: {min_frames_per_window}")

    # 计算采样间隔
    stride = max(1, int(round(fps / (turn_hz * eval_per_turn))))
    print(f"采样间隔: 每 {stride} 帧处理一次")

    # 日志文件
    log_path = os.path.join(output_dir, 'events.csv')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("timestamp,frame_idx,event_type,detail,confidence\n")

    # 处理视频
    frame_idx = 0
    processed_count = 0
    event_count = 0

    # 事件冷却机制（根据生产速度动态调整）
    cooldown_frames = max(15, int(production_fps * 1.5))  # 约1.5秒冷却
    last_event_time = {}

    # 窗口聚合状态
    if enable_window_aggregation:
        window_delta_d_list = []  # 收集窗口内的厚度增量
        window_start_time = time.time()
        window_frames = 0

    print("\n开始处理视频...")

    # 帧质量闸门：宽松阈值，只过滤最严重的坏帧
    quality_gate = FrameQualityGate(
        enable=True,
        blur_th=70.0,         # 模糊阈值（宽松，只过滤严重模糊）
        flat_th=7.0,          # 画面过平阈值（宽松）
        motion_th=10.0,       # 运动阈值（宽松）
        glitch_flat_th=1.5    # 掉帧检测阈值（只过滤极端掉帧）
    )
    prev_gray = None
    skipped_bad = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # 跳帧采样
        if frame_idx % stride != 0:
            continue

        processed_count += 1

        # === 先做帧质量检测（在跑模型之前）===
        q, gray = quality_gate.check(frame, prev_gray)
        prev_gray = gray
        if q.is_bad:
            skipped_bad += 1
            # 可选：记录到日志（方便统计掉帧/模糊比例）
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(f"{ts},{frame_idx},SKIP_BAD_FRAME,{q.reason}|lap={q.lap_var:.1f}|std={q.gray_std:.1f}|mad={q.mad:.1f},1.0\n")
            # 不做识别、不触发事件
            continue

        # 预测
        mask = inferencer.predict(frame)

        # 详细像素统计：帮助诊断鼓包为什么没检测到
        counts = {cid: int((mask == cid).sum()) for cid in range(7)}
        if counts[3] > 0 or counts[4] > 0 or counts[5] > 0 or counts[6] > 0:
            # 只有检测到缺陷时才输出详细统计
            defect_info = []
            if counts[3] > 0:
                defect_info.append(f"鼓包={counts[3]}")
            if counts[4] > 0:
                # loose 未训练，仅记录但不触发事件
                defect_info.append(f"松脱(未训练)={counts[4]}")
            if counts[5] > 0:
                defect_info.append(f"破损={counts[5]}")
            if counts[6] > 0:
                defect_info.append(f"厚度不足={counts[6]}")
            print(f"[frame {frame_idx}] 检测到缺陷: {', '.join(defect_info)}, 总缺陷={sum([counts[3], counts[4], counts[5], counts[6]])}")

        # 生成叠加图
        overlay = inferencer.overlay_mask(frame, mask, alpha=0.6)  # 增加透明度让mask更明显

        # 检测事件
        events = []

        # 1. 检测缺陷类别 (移除loose因为未训练标注)
        # 注意：class 4 (loose_defect) 未被训练，不应触发事件
        defect_classes = [3, 5, 6]  # bulge, burr, thin (移除 loose)
        for class_id in defect_classes:
            if np.any(mask == class_id):
                # 计算面积
                area = np.sum(mask == class_id)

                # 降低阈值：从原来的50降低到10，让小鼓包也能被检测
                effective_threshold = min(min_area_px, 10)

                if area >= effective_threshold:
                    # 检查冷却时间（仅在窗口聚合模式下使用较短的冷却时间）
                    if enable_window_aggregation:
                        # 窗口聚合模式：使用较短的冷却时间
                        current_cooldown = cooldown_frames // 2
                    else:
                        current_cooldown = cooldown_frames

                    current_time = frame_idx
                    if class_id in last_event_time:
                        if current_time - last_event_time[class_id] < current_cooldown:
                            continue

                    # 获取边界框
                    ys, xs = np.where(mask == class_id)
                    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

                    # 映射到用户需要的类别
                    if class_id == 3:
                        event_name = '鼓包缺陷'
                    elif class_id == 4:
                        event_name = '松脱缺陷'
                    elif class_id == 5:
                        event_name = '破损缺陷'
                    elif class_id == 6:
                        event_name = '厚度不足缺陷'
                    else:
                        event_name = f'类别{class_id}'

                    events.append({
                        'type': event_name,
                        'detail': f'bbox=({x0},{y0},{x1},{y1}),area={area}',
                        'class_id': class_id,
                        'bbox': (x0, y0, x1, y1)
                    })

                    last_event_time[class_id] = current_time

                    # 在叠加图上标注
                    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 255), 2)
                    cv2.putText(overlay, event_name, (x0, max(0, y0 - 5)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 2. 检测直径/厚度异常
        m = measure_cable_tape_diameter_px(mask, cable_id=1, tape_id=2)
        is_thickness_event = False

        if m is not None:
            cable_d_px, tape_d_px, delta_px = m

            if px_per_mm > 0:
                # 使用像素/毫米转换率
                delta_mm_val = delta_px / px_per_mm
                cable_mm = cable_d_px / px_per_mm
                tape_mm = tape_d_px / px_per_mm

                # 窗口聚合模式：收集数据
                if enable_window_aggregation:
                    window_delta_d_list.append(delta_mm_val)
                    window_frames += 1

                    # 检查窗口是否完成
                    window_elapsed = time.time() - window_start_time
                    if window_elapsed >= window_duration_sec or window_frames >= min_frames_per_window:
                        # 窗口统计判断
                        if len(window_delta_d_list) >= min_frames_per_window:
                            delta_mean = np.mean(window_delta_d_list)
                            delta_std = np.std(window_delta_d_list)
                            delta_min = np.min(window_delta_d_list)
                            delta_max = np.max(window_delta_d_list)

                            # 判定逻辑
                            if delta_min < (delta_mm - tol_mm):
                                events.append({
                                    'type': '包裹厚度不足_窗口检测',
                                    'detail': f'window_min={delta_min:.2f}mm,mean={delta_mean:.2f}mm,std={delta_std:.2f}mm,n={len(window_delta_d_list)}',
                                    'class_id': None
                                })
                                is_thickness_event = True
                            elif delta_max > (delta_mm + tol_mm * 1.5):
                                events.append({
                                    'type': '包裹厚度过大_窗口检测',
                                    'detail': f'window_max={delta_max:.2f}mm,mean={delta_mean:.2f}mm,std={delta_std:.2f}mm,n={len(window_delta_d_list)}',
                                    'class_id': None
                                })
                                is_thickness_event = True
                            elif delta_std > tol_mm * 0.8:
                                events.append({
                                    'type': '包裹厚度不均_窗口检测',
                                    'detail': f'window_std={delta_std:.2f}mm,range={delta_max-delta_min:.2f}mm,n={len(window_delta_d_list)}',
                                    'class_id': None
                                })
                                is_thickness_event = True

                        # 重置窗口
                        window_delta_d_list = []
                        window_frames = 0
                        window_start_time = time.time()

                else:
                    # 单帧判断模式（原有逻辑）
                    if delta_mm_val < (delta_mm - tol_mm):
                        events.append({
                            'type': '包裹厚度不足',
                            'detail': f'delta_mm={delta_mm_val:.2f},cable_mm={cable_mm:.2f},tape_mm={tape_mm:.2f}',
                            'class_id': None
                        })
                        is_thickness_event = True
            else:
                # 使用比例检查
                ratio = tape_d_px / max(1e-6, cable_d_px)
                if ratio < ratio_min:
                    events.append({
                        'type': '包裹厚度不足_比例检测',
                        'detail': f'ratio={ratio:.3f},cable_px={cable_d_px:.1f},tape_px={tape_d_px:.1f}',
                        'class_id': None
                    })
                    is_thickness_event = True
                elif ratio > ratio_max:
                    events.append({
                        'type': '包裹厚度过大_比例检测',
                        'detail': f'ratio={ratio:.3f},cable_px={cable_d_px:.1f},tape_px={tape_d_px:.1f}',
                        'class_id': None
                    })
                    is_thickness_event = True

        # 保存事件和截图
        if events:
            event_count += 1
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")

            # 保存截图（始终保存）
            # 使用绝对路径避免中文路径问题
            abs_output_dir = os.path.abspath(output_dir)
            snap_dir = os.path.join(abs_output_dir, 'snapshots')
            over_dir = os.path.join(abs_output_dir, 'overlays')

            snap_path = os.path.join(snap_dir, f"{ts}_f{frame_idx}.jpg")
            over_path = os.path.join(over_dir, f"{ts}_f{frame_idx}.jpg")

            # 确保目录存在
            try:
                os.makedirs(snap_dir, exist_ok=True)
                os.makedirs(over_dir, exist_ok=True)

                # 检查frame和overlay数据
                if frame is None or frame.size == 0:
                    print(f"  [警告] frame数据为空")
                if overlay is None or overlay.size == 0:
                    print(f"  [警告] overlay数据为空")

                # 保存原始帧和叠加图
                snap_success = cv2.imwrite(snap_path, frame)
                over_success = cv2.imwrite(over_path, overlay)

                # 如果cv2.imwrite失败，尝试使用PIL保存
                if not snap_success:
                    try:
                        from PIL import Image
                        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        img.save(snap_path)
                        snap_success = True
                        print(f"  [提示] 使用PIL成功保存原始帧")
                    except:
                        pass

                if not over_success:
                    try:
                        from PIL import Image
                        img = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
                        img.save(over_path)
                        over_success = True
                        print(f"  [提示] 使用PIL成功保存叠加图")
                    except:
                        pass

                # 检查保存结果
                if not snap_success:
                    print(f"  [警告] 无法保存原始帧截图: {snap_path}")
                    print(f"          frame shape: {frame.shape if frame is not None else 'None'}, dtype: {frame.dtype if frame is not None else 'None'}")
                if not over_success:
                    print(f"  [警告] 无法保存叠加图截图: {over_path}")
                    print(f"          overlay shape: {overlay.shape if overlay is not None else 'None'}, dtype: {overlay.dtype if overlay is not None else 'None'}")

            except Exception as e:
                print(f"  [错误] 保存截图时异常: {e}")
                snap_success = False
                over_success = False

            # 记录日志
            with open(log_path, 'a', encoding='utf-8') as f:
                for evt in events:
                    f.write(f"{ts},{frame_idx},{evt['type']},{evt['detail']},1.0\n")

            event_types = [e['type'] for e in events]
            save_status = "[OK]" if (snap_success and over_success) else "[FAIL]"
            print(f"  [帧 {frame_idx}] 检测到事件: {', '.join(event_types)} {save_status}")


        # 实时预览（每一帧都显示）
        if show_preview:
            # 在图像上添加实时信息
            display = overlay.copy()

            # 添加帧信息
            cv2.putText(display, f"Frame: {frame_idx}/{total_frames}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(display, f"Device: {device}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # 添加检测统计
            if m is not None:
                cable_d_px, tape_d_px, delta_px = m
                ratio = tape_d_px / max(1e-6, cable_d_px)
                cv2.putText(display, f"Cable: {cable_d_px:.0f}px", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                cv2.putText(display, f"Tape: {tape_d_px:.0f}px", (10, 115),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(display, f"Ratio: {ratio:.3f}", (10, 140),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # 添加事件计数
            if events:
                cv2.putText(display, f"EVENTS: {len(events)}", (10, 180),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # 调整大小以便显示
            display_resized = cv2.resize(display, (1024, 768))

            cv2.imshow('Cable Inspection - Real-time Detection', display_resized)

            # 生产速度模拟延迟
            if delay_ms > 0:
                # cv2.waitKey 会包含延迟时间
                key = cv2.waitKey(max(1, delay_ms)) & 0xFF
            else:
                key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                break
            elif key == 32:  # 空格暂停
                cv2.waitKey(0)

        # 如果没有预览窗口，仍然需要延迟
        elif delay_ms > 0:
            import time
            time.sleep(delay_ms / 1000.0)

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n处理完成!")
    print(f"  总帧数: {frame_idx}")
    print(f"  处理帧数: {processed_count}")
    print(f"  检测事件: {event_count}")
    print(f"  跳过坏帧: {skipped_bad}")
    print(f"  结果保存在: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='电缆包裹质量检测 - 视频推理')

    # 模型参数
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pth',
                       help='模型路径')
    parser.add_argument('--num-classes', type=int, default=7,
                       help='类别数')
    parser.add_argument('--input-size', type=int, default=256,
                       help='输入图像尺寸')
    parser.add_argument('--device', type=str, default='cpu',
                       help='计算设备 (cpu/cuda)')

    # 视频参数
    parser.add_argument('--video', type=str, required=True,
                       help='输入视频路径')
    parser.add_argument('--output', type=str, default='log',
                       help='输出目录')

    # 采样参数
    parser.add_argument('--turn-hz', type=float, default=3.0,
                       help='转速（转/秒）')
    parser.add_argument('--eval-per-turn', type=int, default=1,
                       help='每圈评估次数')

    # 检测参数
    parser.add_argument('--px-per-mm', type=float, default=0.0,
                       help='像素/毫米转换率 (0=仅使用比例检查)')
    parser.add_argument('--delta-mm', type=float, default=20.0,
                       help='期望的胶带-电缆直径差 (mm)')
    parser.add_argument('--tol-mm', type=float, default=5.0,
                       help='容差范围 (mm)')
    parser.add_argument('--ratio-min', type=float, default=1.05,
                       help='tape/cable 最小比例')
    parser.add_argument('--ratio-max', type=float, default=1.5,
                       help='tape/cable 最大比例')
    parser.add_argument('--min-area-px', type=int, default=50,
                       help='缺陷最小像素面积')

    # 显示参数
    parser.add_argument('--save-overlay', action='store_true', default=True,
                       help='保存叠加可视化图')
    parser.add_argument('--show-preview', action='store_true',
                       help='显示实时预览')

    # 生产速度模拟参数
    parser.add_argument('--delay-ms', type=int, default=0,
                       help='每帧处理延迟（毫秒），0=无延迟')
    parser.add_argument('--simulate-production', action='store_true',
                       help='启用生产速度模拟模式')
    parser.add_argument('--production-fps', type=float, default=10.0,
                       help='生产环境检测速度（帧/秒），默认10fps')

    # 窗口聚合参数
    parser.add_argument('--enable-window-aggregation', action='store_true',
                       help='启用窗口聚合模式（多帧统计判断，减少误报）')
    parser.add_argument('--window-duration', type=float, default=3.0,
                       help='窗口聚合时长（秒）')
    parser.add_argument('--min-frames-window', type=int, default=6,
                       help='窗口最小帧数')

    args = parser.parse_args()

    process_video(
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
        enable_window_aggregation=args.enable_window_aggregation,
        window_duration_sec=args.window_duration,
        min_frames_per_window=args.min_frames_window
    )


if __name__ == '__main__':
    main()
