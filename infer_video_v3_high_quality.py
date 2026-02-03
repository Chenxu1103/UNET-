"""
使用高质量6类NestedUNet模型进行推理

这个模型产生了高质量分割结果（checkpoints_v3/best_model.pth）
- 6类NestedUNet with deep supervision
- 使用简单的3x3闭运算后处理
- 训练良好，分割精度高
"""

import os
import cv2
import argparse
import numpy as np
import torch
import sys
from pathlib import Path

# 添加目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

from src.models.unetpp import NestedUNet

# 导入直径测量函数
import importlib.util
spec = importlib.util.spec_from_file_location("diameter", str(project_root / "utils" / "diameter.py"))
diameter_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diameter_module)
measure_cable_tape_diameter_px = diameter_module.measure_cable_tape_diameter_px


# 类别颜色 (BGR格式)
# 注意：6类模型使用类别 [0, 1, 2, 4, 5, 6]，不是标准的[0,1,2,3,4,5]
# 模型输出通道映射: 0→0, 1→1, 2→2, 3→4, 4→5, 5→6
CLASS_COLORS = {
    0: (0, 0, 0),         # 黑色：背景
    1: (255, 0, 0),       # 蓝色：电缆
    2: (0, 255, 0),       # 绿色：胶带
    4: (0, 165, 255),     # 橙色：loose缺陷
    5: (0, 0, 255),       # 红色：毛刺缺陷
    6: (255, 0, 255),     # 洋红：厚度不足
}

# 模型输出通道到实际类别的映射
# 6类模型输出6个通道，对应类别[0,1,2,4,5,6]
CHANNEL_TO_CLASS = {0: 0, 1: 1, 2: 2, 3: 4, 4: 5, 5: 6}


class NestedUNetInference:
    """使用NestedUNet的推理器"""

    def __init__(self, model_path: str, num_classes: int = 6, device: str = 'cpu'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        self.input_size = 256

        print(f"设备: {self.device}")
        print(f"加载模型: {model_path}")
        print(f"类别数: {num_classes}")

        # 使用NestedUNet模型
        self.model = NestedUNet(
            num_classes=num_classes,
            input_channels=3,
            deep_supervision=False  # 推理时不需要deep supervision
        ).to(self.device)

        # 加载权重
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if 'model' in checkpoint:
            # 过滤掉deep supervision的键
            state_dict = checkpoint['model']
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith('ds')}
            self.model.load_state_dict(state_dict, strict=False)
            print("  已过滤deep supervision层")
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith('ds')}
            self.model.load_state_dict(state_dict, strict=False)
        else:
            state_dict = {k: v for k, v in checkpoint.items() if not k.startswith('ds')}
            self.model.load_state_dict(state_dict, strict=False)

        self.model.eval()
        print("模型加载完成")

    @torch.no_grad()
    def predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        """预测单帧 - 使用概率阈值分割（可同时检测cable和tape）"""
        h, w = frame_bgr.shape[:2]

        # 预处理
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_small = cv2.resize(frame_rgb, (self.input_size, self.input_size))
        tensor = torch.from_numpy(frame_small).float() / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)

        # 推理
        output = self.model(tensor)
        if isinstance(output, list):
            output = output[0]

        # 使用概率阈值分割（而不是argmax）
        # 这样可以同时检测cable和tape，而不是互相竞争
        probs = torch.softmax(output, dim=1)[0]  # [num_classes, H, W]

        # 提取各类别概率图（256x256）
        # 注意：6类模型输出通道映射 [0,1,2,3,4,5] → 类别 [0,1,2,4,5,6]
        prob_cable = probs[1].cpu().numpy()   # 通道1 → 类别1: cable
        prob_tape = probs[2].cpu().numpy()    # 通道2 → 类别2: tape
        prob_loose = probs[3].cpu().numpy()   # 通道3 → 类别4: loose_defect
        prob_burr = probs[4].cpu().numpy()    # 通道4 → 类别5: burr_defect
        prob_thin = probs[5].cpu().numpy()    # 通道5 → 类别6: thin_defect

        # 调整回原始尺寸
        prob_cable = cv2.resize(prob_cable, (w, h), interpolation=cv2.INTER_LINEAR)
        prob_tape = cv2.resize(prob_tape, (w, h), interpolation=cv2.INTER_LINEAR)
        prob_loose = cv2.resize(prob_loose, (w, h), interpolation=cv2.INTER_LINEAR)
        prob_burr = cv2.resize(prob_burr, (w, h), interpolation=cv2.INTER_LINEAR)
        prob_thin = cv2.resize(prob_thin, (w, h), interpolation=cv2.INTER_LINEAR)

        # 使用概率阈值提取mask（允许cable和tape共存）
        # 提高阈值以提高精准度，减少误报
        cable_thresh = 0.60  # 提高到60%，只检测高置信度区域
        tape_thresh = 0.60   # 提高到60%，只检测高置信度区域
        defect_thresh = 0.70 # 提高到70%，减少缺陷误报

        # 基础mask
        mask_cable_base = (prob_cable >= cable_thresh)
        mask_tape_base = (prob_tape >= tape_thresh)

        # 关键改进：使用互斥策略避免cable和tape混在一起
        # 只有当cable概率明显高于tape时才判定为cable
        mask_cable = (mask_cable_base & (prob_cable > prob_tape * 1.2)).astype(np.uint8)
        # 只有当tape概率明显高于cable时才判定为tape
        mask_tape = (mask_tape_base & (prob_tape > prob_cable * 1.2)).astype(np.uint8)

        # 缺陷检测（使用正确的类别ID: 4, 5, 6）
        mask_loose = (prob_loose >= defect_thresh).astype(np.uint8)  # 类别4
        mask_burr = (prob_burr >= defect_thresh).astype(np.uint8)   # 类别5
        mask_thin = (prob_thin >= defect_thresh).astype(np.uint8)   # 类别6

        # 形态学后处理：3x3闭运算
        kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # Cable和tape: 闭运算填充空洞
        if np.sum(mask_cable) > 0:
            mask_cable = cv2.morphologyEx(mask_cable, cv2.MORPH_CLOSE, kernel3, iterations=1)
        if np.sum(mask_tape) > 0:
            mask_tape = cv2.morphologyEx(mask_tape, cv2.MORPH_CLOSE, kernel3, iterations=1)

        # Defect: 开运算去除噪点 + 闭运算填充
        if np.sum(mask_loose) > 0:
            mask_loose = cv2.morphologyEx(mask_loose, cv2.MORPH_OPEN, kernel3, iterations=1)
            mask_loose = cv2.morphologyEx(mask_loose, cv2.MORPH_CLOSE, kernel5, iterations=1)
        if np.sum(mask_burr) > 0:
            mask_burr = cv2.morphologyEx(mask_burr, cv2.MORPH_OPEN, kernel3, iterations=1)
            mask_burr = cv2.morphologyEx(mask_burr, cv2.MORPH_CLOSE, kernel5, iterations=1)
        if np.sum(mask_thin) > 0:
            mask_thin = cv2.morphologyEx(mask_thin, cv2.MORPH_OPEN, kernel3, iterations=1)
            mask_thin = cv2.morphologyEx(mask_thin, cv2.MORPH_CLOSE, kernel5, iterations=1)

        # 按优先级合并：defect > tape > cable > background
        # 使用正确的类别ID: 1, 2, 4, 5, 6
        result = np.zeros((h, w), dtype=np.uint8)
        result[mask_cable > 0] = 1           # 先放cable
        result[mask_tape > 0] = 2            # tape覆盖cable
        result[mask_loose > 0] = 4           # loose覆盖tape
        result[mask_burr > 0] = 5            # burr覆盖tape
        result[mask_thin > 0] = 6            # thin覆盖tape

        return result

    def overlay_mask(self, frame_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """将mask叠加到原始图像 - 只在mask区域混合"""
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

        # 只在 mask>0 区域混合（避免背景被整体压暗）
        result = frame_bgr.copy()
        region = (mask > 0)
        if np.any(region):
            blended = ((1 - alpha) * frame_bgr.astype(np.float32) +
                      alpha * color_mask.astype(np.float32)).astype(np.uint8)
            result[region] = blended[region]

        # 添加细轮廓让边界清晰
        for class_id, color in CLASS_COLORS.items():
            if class_id == 0:
                continue
            if class_id >= self.num_classes:
                continue

            # 使用原始mask获取轮廓
            class_mask = (mask == class_id).astype(np.uint8) * 255
            contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 绘制细轮廓（2px）
            cv2.drawContours(result, contours, -1, color, 2)

        return result


def process_video(
    model_path: str,
    video_path: str,
    output_dir: str,
    num_classes: int = 6,
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
    inferencer = NestedUNetInference(
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

    # 创建视频写入器
    output_video_path = os.path.join(output_dir, 'detection_result.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # 统计
    frame_idx = 0
    defect_count = 0
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

        # 检测缺陷类别（4=loose, 5=burr, 6=thin）
        has_defect = False
        defect_classes = [4, 5, 6]
        for cls in defect_classes:
            if np.any(mask == cls):
                area = np.sum(mask == cls)
                if area >= min_area_px:
                    has_defect = True
                    defect_count += 1
                    break

        # 检测直径/厚度异常
        ratio, is_thin, is_thick = None, False, False
        m = measure_cable_tape_diameter_px(mask, cable_id=1, tape_id=2)

        if m is not None:
            cable_d_px, tape_d_px, delta_px = m
            ratio = tape_d_px / max(1e-6, cable_d_px)

            # 检查测量质量：电缆和胶带直径必须合理
            # 电缆直径应该在50-150像素之间，胶带应该在30-200像素之间
            valid_measurement = (50 < cable_d_px < 150) and (30 < tape_d_px < 200)

            if valid_measurement:
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
        cv2.putText(overlay, f"Frame: {frame_idx}/{total_frames}",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30

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

        # 写入视频
        video_writer.write(overlay)

        # 显示进度
        if frame_idx % 100 == 0:
            print(f"处理进度: {frame_idx}/{total_frames} 帧 ({frame_idx/total_frames*100:.1f}%)")

        # 实时预览窗口
        if show_preview:
            display_height = 720
            scale = display_height / height
            display_width = int(width * scale)
            display = cv2.resize(overlay, (display_width, display_height))

            cv2.imshow('High Quality Detection (NestedUNet v3)', display)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                print("\n用户按ESC退出")
                break
            elif key == 32:  # 空格暂停
                print("\n已暂停，按任意键继续...")
                cv2.waitKey(0)

        # 保存异常帧
        if has_defect or is_thin or is_thick:
            snap_path = os.path.join(output_dir, 'snapshots', f"frame_{frame_idx:06d}.jpg")
            over_path = os.path.join(output_dir, 'overlays', f"frame_{frame_idx:06d}.jpg")

            cv2.imwrite(snap_path, frame)
            cv2.imwrite(over_path, overlay)

            # 记录日志
            with open(log_path, 'a', encoding='utf-8') as f:
                if is_thin and ratio is not None:
                    f.write(f"{frame_idx},wrap_thin,{ratio:.3f},{m[0]:.1f},{m[1]:.1f},{m[2]:.1f}\n")
                if is_thick and ratio is not None:
                    f.write(f"{frame_idx},wrap_thick,{ratio:.3f},{m[0]:.1f},{m[1]:.1f},{m[2]:.1f}\n")

            ratio_str = f"{ratio:.3f}" if ratio is not None else "N/A"
            print(f"[帧 {frame_idx}] 缺陷={has_defect}, 厚度不足={is_thin}, 厚度过大={is_thick}, ratio={ratio_str}")

    # 释放资源
    video_writer.release()
    cap.release()
    if show_preview:
        cv2.destroyAllWindows()

    print("\n" + "="*70)
    print("处理完成!")
    print("="*70)
    print(f"  缺陷检测: {defect_count}")
    print(f"  厚度不足: {thin_count}")
    print(f"  厚度过大: {thick_count}")
    print(f"  总异常帧: {defect_count + thin_count + thick_count}")
    print(f"  检测视频: {output_video_path}")
    print(f"  输出目录: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='高质量检测 - 使用NestedUNet v3模型')
    parser.add_argument('--video', required=True, help='视频路径')
    parser.add_argument('--model', default='checkpoints_v3/best_model.pth', help='模型路径')
    parser.add_argument('--output', required=True, help='输出目录')
    parser.add_argument('--num-classes', type=int, default=6, help='类别数')
    parser.add_argument('--device', default='cuda', help='设备')
    parser.add_argument('--ratio-min', type=float, default=1.15, help='最小比例（厚度不足阈值，提高以减少误报）')
    parser.add_argument('--ratio-max', type=float, default=1.35, help='最大比例（厚度过大阈值，降低以减少误报）')
    parser.add_argument('--min-area-px', type=int, default=100, help='最小面积（提高以减少误报）')
    parser.add_argument('--show-preview', action='store_true', default=True, help='显示预览')
    parser.add_argument('--no-preview', action='store_true', help='不显示预览')

    args = parser.parse_args()
    show_preview = args.show_preview and not args.no_preview

    print("="*70)
    print("高质量电缆缠绕检测（NestedUNet v3 - 6类模型）")
    print("="*70)
    print(f"模型: {args.model}")
    print(f"视频: {args.video}")
    print(f"类别数: {args.num_classes}")
    print(f"检测类别: 电缆(1), 胶带(2), 松动缺陷(4), 毛刺缺陷(5), 厚度不足(6)")
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
