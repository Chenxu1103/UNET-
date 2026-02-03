"""异常事件日志记录模块

记录分割检测过程中发现的异常事件，支持CSV格式和JSON格式的日志
"""
import os
import csv
import json
import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Any


class AnomalyLogger:
    """异常事件日志记录器，将事件以CSV格式追加保存。"""
    
    def __init__(self, log_path: str = "log/events.log"):
        """初始化日志记录器
        
        Args:
            log_path: 日志文件路径
        """
        self.log_path = log_path
        
        # 创建日志目录
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        
        # 创建日志文件并写入表头
        if not os.path.exists(log_path):
            with open(log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'defect_type', 'bbox_xmin', 'bbox_ymin', 
                    'bbox_xmax', 'bbox_ymax', 'area_pixels'
                ])
    
    def log_event(
        self,
        timestamp: str,
        defect_type: str,
        bbox: Tuple[int, int, int, int],
        area_pixels: int = None
    ) -> None:
        """记录一条异常事件
        
        Args:
            timestamp: 事件时间戳
            defect_type: 缺陷类型
            bbox: 边界框 (x_min, y_min, x_max, y_max)
            area_pixels: 缺陷区域像素数，可选
        """
        x_min, y_min, x_max, y_max = bbox
        
        with open(self.log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                defect_type,
                x_min,
                y_min,
                x_max,
                y_max,
                area_pixels if area_pixels is not None else -1
            ])
    
    def read_log(self) -> List[Dict[str, Any]]:
        """读取日志文件并返回事件列表
        
        Returns:
            事件列表，每个事件是一个字典
        """
        events = []
        
        if not os.path.exists(self.log_path):
            return events
        
        with open(self.log_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row:
                    events.append({
                        'timestamp': row['timestamp'],
                        'defect_type': row['defect_type'],
                        'bbox': (
                            int(row['bbox_xmin']),
                            int(row['bbox_ymin']),
                            int(row['bbox_xmax']),
                            int(row['bbox_ymax'])
                        ),
                        'area_pixels': int(row['area_pixels']) if row['area_pixels'] != '-1' else None
                    })
        
        return events


class JSONLogger:
    """JSON格式的事件日志记录器，便于与其他系统集成"""
    
    def __init__(self, log_dir: str = "log/events"):
        """初始化JSON日志记录器
        
        Args:
            log_dir: 日志文件所在目录
        """
        self.log_dir = log_dir
        Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    def save_event(
        self,
        camera_id: str,
        timestamp_ns: int,
        findings: List[Dict[str, Any]],
        metrics: Dict[str, Any] = None,
        image_paths: Dict[str, str] = None
    ) -> str:
        """保存事件为JSON文件
        
        Args:
            camera_id: 相机ID
            timestamp_ns: 纳秒级时间戳
            findings: 发现的缺陷列表，每个元素包含 {'code', 'severity', 'detail'}
            metrics: 度量指标字典，可选
            image_paths: 图像路径字典，可选
            
        Returns:
            保存的JSON文件路径
        """
        # 创建事件字典
        event = {
            'camera_id': camera_id,
            'timestamp_ns': timestamp_ns,
            'timestamp': datetime.datetime.now().isoformat(),
            'findings': findings,
            'metrics': metrics or {},
            'images': image_paths or {}
        }
        
        # 生成文件名
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"{timestamp_str}_event.json"
        filepath = os.path.join(self.log_dir, filename)
        
        # 保存为JSON
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(event, f, indent=2, ensure_ascii=False)
        
        return filepath


class StatisticsLogger:
    """统计日志，记录周期性的统计数据"""
    
    def __init__(self, log_path: str = "log/statistics.csv"):
        """初始化统计日志记录器
        
        Args:
            log_path: 统计日志文件路径
        """
        self.log_path = log_path
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        
        # 创建日志文件并写入表头
        if not os.path.exists(log_path):
            with open(log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'total_frames', 'frames_with_defects',
                    'detection_rate', 'avg_processing_time_ms'
                ])
    
    def log_statistics(
        self,
        total_frames: int,
        frames_with_defects: int,
        avg_processing_time_ms: float
    ) -> None:
        """记录统计数据
        
        Args:
            total_frames: 处理的总帧数
            frames_with_defects: 含缺陷的帧数
            avg_processing_time_ms: 平均处理时间（毫秒）
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        detection_rate = frames_with_defects / total_frames if total_frames > 0 else 0.0
        
        with open(self.log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                total_frames,
                frames_with_defects,
                f"{detection_rate:.4f}",
                f"{avg_processing_time_ms:.2f}"
            ])


if __name__ == "__main__":
    # 示例用法
    logger = AnomalyLogger("log/events.log")
    
    # 记录事件
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.log_event(timestamp, "bulge_defect", (100, 200, 300, 400), area_pixels=5000)
    logger.log_event(timestamp, "loose_defect", (150, 250, 350, 450), area_pixels=3000)
    
    # 读取日志
    events = logger.read_log()
    print(f"Total events: {len(events)}")
    for event in events:
        print(event)
