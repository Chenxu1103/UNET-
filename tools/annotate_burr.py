"""
简易毛刺标注工具
使用鼠标绘制矩形框标注毛刺区域
"""
import cv2
import json
import argparse
from pathlib import Path
import numpy as np


class BurrAnnotationTool:
    def __init__(self, frames_dir, metadata_path, output_dir):
        self.frames_dir = Path(frames_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 加载元数据
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

        self.frames = self.metadata['frames']
        self.current_idx = 0
        self.current_frame = None
        self.display_frame = None

        # 标注状态
        self.drawing = False
        self.start_point = None
        self.current_boxes = []  # 当前帧的所有标注框
        self.temp_box = None  # 正在绘制的框

        # 加载已有标注
        self.load_annotations()

        print("=" * 60)
        print("毛刺标注工具")
        print("=" * 60)
        print("操作说明:")
        print("  鼠标左键: 拖动绘制矩形框")
        print("  空格键: 保存当前标注并进入下一帧")
        print("  'u': 撤销最后一个标注框")
        print("  'c': 清除当前帧所有标注")
        print("  's': 保存当前标注")
        print("  'n': 跳过当前帧（无毛刺）")
        print("  'q': 退出")
        print("  左/右方向键: 前一帧/后一帧")
        print("=" * 60)

    def load_annotations(self):
        """加载已有标注"""
        annotations_file = self.output_dir / 'burr_annotations.json'
        if annotations_file.exists():
            with open(annotations_file, 'r', encoding='utf-8') as f:
                self.annotations = json.load(f)
            print(f"已加载 {len(self.annotations)} 个已标注帧")
        else:
            self.annotations = {}

    def save_annotations(self):
        """保存标注"""
        annotations_file = self.output_dir / 'burr_annotations.json'
        with open(annotations_file, 'w', encoding='utf-8') as f:
            json.dump(self.annotations, f, indent=2, ensure_ascii=False)
        print(f"标注已保存到: {annotations_file}")

    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.temp_box = (self.start_point[0], self.start_point[1], x, y)
                self.update_display()

        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                x1, y1 = self.start_point
                x2, y2 = x, y

                # 确保坐标顺序正确
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)

                # 过滤太小的框
                if (x2 - x1) > 10 and (y2 - y1) > 10:
                    self.current_boxes.append([x1, y1, x2, y2])
                    print(f"添加标注框: [{x1}, {y1}, {x2}, {y2}]")

                self.temp_box = None
                self.update_display()

    def load_frame(self, idx):
        """加载指定帧"""
        if idx < 0 or idx >= len(self.frames):
            return False

        self.current_idx = idx
        frame_info = self.frames[idx]
        frame_path = self.frames_dir / frame_info['filename']

        self.current_frame = cv2.imread(str(frame_path))
        if self.current_frame is None:
            print(f"错误: 无法加载帧 {frame_path}")
            return False

        # 加载该帧的已有标注
        frame_id = frame_info['frame_id']
        if str(frame_id) in self.annotations:
            self.current_boxes = self.annotations[str(frame_id)]['burr_regions'].copy()
        else:
            self.current_boxes = []

        self.update_display()
        return True

    def update_display(self):
        """更新显示"""
        self.display_frame = self.current_frame.copy()

        # 绘制已有标注框（绿色）
        for box in self.current_boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(self.display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(self.display_frame, 'burr', (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 绘制正在绘制的框（黄色）
        if self.temp_box is not None:
            x1, y1, x2, y2 = self.temp_box
            cv2.rectangle(self.display_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # 显示信息
        frame_info = self.frames[self.current_idx]
        info_text = f"Frame {self.current_idx + 1}/{len(self.frames)} | " \
                    f"ID: {frame_info['frame_id']} | " \
                    f"Burrs: {len(self.current_boxes)}"
        cv2.putText(self.display_frame, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(self.display_frame, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)

        cv2.imshow('Burr Annotation Tool', self.display_frame)

    def save_current_annotation(self):
        """保存当前帧标注"""
        frame_info = self.frames[self.current_idx]
        frame_id = frame_info['frame_id']

        self.annotations[str(frame_id)] = {
            'frame_id': frame_id,
            'filename': frame_info['filename'],
            'has_burr': len(self.current_boxes) > 0,
            'burr_count': len(self.current_boxes),
            'burr_regions': self.current_boxes.copy()
        }

        print(f"已保存帧 {frame_id} 的标注 ({len(self.current_boxes)} 个毛刺)")

    def run(self):
        """运行标注工具"""
        cv2.namedWindow('Burr Annotation Tool')
        cv2.setMouseCallback('Burr Annotation Tool', self.mouse_callback)

        # 加载第一帧
        self.load_frame(0)

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):  # 退出
                self.save_annotations()
                break

            elif key == ord(' '):  # 空格：保存并下一帧
                self.save_current_annotation()
                if self.current_idx < len(self.frames) - 1:
                    self.load_frame(self.current_idx + 1)
                else:
                    print("已到达最后一帧")

            elif key == ord('n'):  # 跳过（无毛刺）
                self.current_boxes = []
                self.save_current_annotation()
                if self.current_idx < len(self.frames) - 1:
                    self.load_frame(self.current_idx + 1)

            elif key == ord('s'):  # 保存
                self.save_current_annotation()
                self.save_annotations()

            elif key == ord('u'):  # 撤销
                if self.current_boxes:
                    removed = self.current_boxes.pop()
                    print(f"撤销标注框: {removed}")
                    self.update_display()

            elif key == ord('c'):  # 清除
                self.current_boxes = []
                print("已清除当前帧所有标注")
                self.update_display()

            elif key == 81 or key == 2:  # 左方向键
                if self.current_idx > 0:
                    self.load_frame(self.current_idx - 1)

            elif key == 83 or key == 3:  # 右方向键
                if self.current_idx < len(self.frames) - 1:
                    self.load_frame(self.current_idx + 1)

        cv2.destroyAllWindows()

        # 生成标注统计
        self.print_statistics()

    def print_statistics(self):
        """打印标注统计"""
        print("\n" + "=" * 60)
        print("标注统计")
        print("=" * 60)
        total_frames = len(self.frames)
        annotated_frames = len(self.annotations)
        frames_with_burr = sum(1 for ann in self.annotations.values() if ann['has_burr'])
        total_burrs = sum(ann['burr_count'] for ann in self.annotations.values())

        print(f"总帧数: {total_frames}")
        print(f"已标注帧数: {annotated_frames} ({annotated_frames/total_frames*100:.1f}%)")
        print(f"有毛刺的帧: {frames_with_burr}")
        print(f"无毛刺的帧: {annotated_frames - frames_with_burr}")
        print(f"毛刺总数: {total_burrs}")
        if frames_with_burr > 0:
            print(f"平均每帧毛刺数: {total_burrs/frames_with_burr:.2f}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='毛刺标注工具')
    parser.add_argument('--frames-dir', type=str, required=True,
                        help='帧图像目录')
    parser.add_argument('--metadata', type=str, required=True,
                        help='元数据JSON文件')
    parser.add_argument('--output', type=str, required=True,
                        help='标注输出目录')

    args = parser.parse_args()

    tool = BurrAnnotationTool(
        frames_dir=args.frames_dir,
        metadata_path=args.metadata,
        output_dir=args.output
    )

    tool.run()


if __name__ == '__main__':
    main()
