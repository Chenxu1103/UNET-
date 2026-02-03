"""
视频切帧工具 - 用于标注数据准备
从视频中提取帧，用于标注毛刺特征
"""
import cv2
import argparse
from pathlib import Path
import json


def extract_frames(video_path, output_dir, frame_interval=30, max_frames=None,
                   start_frame=0, end_frame=None, quality=95):
    """
    从视频中提取帧

    Args:
        video_path: 视频路径
        output_dir: 输出目录
        frame_interval: 帧间隔（每N帧提取一帧）
        max_frames: 最大提取帧数
        start_frame: 起始帧
        end_frame: 结束帧
        quality: JPEG质量（0-100）
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建子目录
    frames_dir = output_dir / 'frames'
    frames_dir.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"错误: 无法打开视频 {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if end_frame is None:
        end_frame = total_frames

    print(f"视频信息:")
    print(f"  分辨率: {width}x{height}")
    print(f"  帧率: {fps:.2f} fps")
    print(f"  总帧数: {total_frames}")
    print(f"  提取范围: {start_frame} - {end_frame}")
    print(f"  帧间隔: {frame_interval}")
    print("-" * 60)

    # 跳转到起始帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    extracted_count = 0
    frame_info_list = []

    for frame_idx in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break

        # 按间隔提取
        if (frame_idx - start_frame) % frame_interval != 0:
            continue

        # 检查最大帧数限制
        if max_frames is not None and extracted_count >= max_frames:
            break

        # 保存帧
        frame_filename = f"frame_{frame_idx:06d}.jpg"
        frame_path = frames_dir / frame_filename
        cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])

        # 记录帧信息
        timestamp = frame_idx / fps
        frame_info = {
            'frame_id': frame_idx,
            'filename': frame_filename,
            'timestamp': f"{timestamp:.3f}s",
            'width': width,
            'height': height,
            'annotated': False,
            'has_burr': None,  # 待标注
            'burr_regions': []  # 毛刺区域列表
        }
        frame_info_list.append(frame_info)

        extracted_count += 1

        if extracted_count % 10 == 0:
            print(f"已提取 {extracted_count} 帧 (当前帧: {frame_idx}/{end_frame})")

    cap.release()

    # 保存帧信息到JSON
    metadata = {
        'video_path': str(video_path),
        'video_info': {
            'width': width,
            'height': height,
            'fps': fps,
            'total_frames': total_frames
        },
        'extraction_params': {
            'start_frame': start_frame,
            'end_frame': end_frame,
            'frame_interval': frame_interval,
            'extracted_count': extracted_count
        },
        'frames': frame_info_list
    }

    metadata_path = output_dir / 'frames_metadata.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("-" * 60)
    print(f"完成! 共提取 {extracted_count} 帧")
    print(f"帧保存在: {frames_dir}")
    print(f"元数据保存在: {metadata_path}")

    # 创建标注说明文件
    annotation_guide = output_dir / 'ANNOTATION_GUIDE.txt'
    with open(annotation_guide, 'w', encoding='utf-8') as f:
        f.write("毛刺标注指南\n")
        f.write("=" * 60 + "\n\n")
        f.write("1. 标注目标：\n")
        f.write("   - 毛刺（burr）：电缆表面的突起、毛边、不规则边缘\n\n")
        f.write("2. 标注方法：\n")
        f.write("   - 使用labelme或其他标注工具\n")
        f.write("   - 对每个毛刺区域绘制多边形或矩形框\n")
        f.write("   - 标签名称：burr\n\n")
        f.write("3. 标注标准：\n")
        f.write("   - 明显的毛刺：必须标注\n")
        f.write("   - 轻微的毛刺：根据实际需求决定\n")
        f.write("   - 模糊不清的：可以跳过\n\n")
        f.write("4. 文件组织：\n")
        f.write("   - 原始帧：frames/\n")
        f.write("   - 标注文件：annotations/ (labelme JSON格式)\n\n")
        f.write("5. 推荐工具：\n")
        f.write("   - labelme: pip install labelme\n")
        f.write("   - 使用命令: labelme frames/\n\n")

    print(f"标注指南保存在: {annotation_guide}")

    return extracted_count


def main():
    parser = argparse.ArgumentParser(description='视频切帧工具')
    parser.add_argument('--video', type=str, required=True, help='视频路径')
    parser.add_argument('--output', type=str, required=True, help='输出目录')
    parser.add_argument('--interval', type=int, default=30,
                        help='帧间隔（每N帧提取一帧，默认30）')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='最大提取帧数（默认无限制）')
    parser.add_argument('--start', type=int, default=0,
                        help='起始帧（默认0）')
    parser.add_argument('--end', type=int, default=None,
                        help='结束帧（默认到视频末尾）')
    parser.add_argument('--quality', type=int, default=95,
                        help='JPEG质量（0-100，默认95）')

    args = parser.parse_args()

    extract_frames(
        video_path=args.video,
        output_dir=args.output,
        frame_interval=args.interval,
        max_frames=args.max_frames,
        start_frame=args.start,
        end_frame=args.end,
        quality=args.quality
    )


if __name__ == '__main__':
    main()
