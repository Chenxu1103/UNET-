"""推理脚本

使用训练好的模型对新图像进行分割推理。
"""
import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path
import argparse
import sys
from typing import Tuple, List

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.unetplusplus import NestedUNet
from utils.visualizer import Visualizer


class SegmentationInference:
    """分割推理类"""
    
    # 类别名称和颜色
    CLASS_NAMES = {
        0: 'background',
        1: 'cable',
        2: 'tape',
        3: 'bulge_defect',
        4: 'loose_defect',
        5: 'burr_defect',
        6: 'thin_defect'
    }
    
    CLASS_COLORS = {
        0: (0, 0, 0),           # 黑色：背景
        1: (255, 0, 0),         # 红色：电缆
        2: (0, 255, 0),         # 绿色：胶带
        3: (0, 0, 255),         # 蓝色：鼓包缺陷
        4: (255, 255, 0),       # 青色：松脱缺陷
        5: (255, 0, 255),       # 洋红：毛刺缺陷
        6: (0, 255, 255)        # 黄色：厚度不足缺陷
    }
    
    def __init__(
        self,
        model_path: str,
        num_classes: int = 7,
        device: str = 'cuda'
    ):
        """初始化推理器
        
        Args:
            model_path: 模型检查点路径
            num_classes: 类别数
            device: 计算设备
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        
        # 加载模型
        print(f"加载模型: {model_path}")
        self.model = NestedUNet(
            num_classes=num_classes,
            input_channels=3,
            deep_supervision=False
        ).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        print("模型加载完成")
        
        self.visualizer = Visualizer(class_colors=self.CLASS_COLORS)
    
    def preprocess(self, image_path: str) -> Tuple[torch.Tensor, np.ndarray]:
        """预处理图像
        
        Args:
            image_path: 图像路径
            
        Returns:
            (tensor, original_image): 张量和原始图像
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"图像不存在: {image_path}")
        
        # BGR → RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 归一化
        image_normalized = image_rgb.astype(np.float32) / 255.0
        
        # HWC → CHW
        image_tensor = np.transpose(image_normalized, (2, 0, 1))
        image_tensor = torch.from_numpy(image_tensor).float().unsqueeze(0)
        
        return image_tensor.to(self.device), image_rgb
    
    @torch.no_grad()
    def infer(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """推理单张图像
        
        Args:
            image_path: 图像路径
            
        Returns:
            (pred_mask, original_image): 预测 mask 和原始图像
        """
        # 预处理
        image_tensor, image_rgb = self.preprocess(image_path)
        
        # 前向传播
        output = self.model(image_tensor)
        if isinstance(output, list):
            output = output[0]
        
        # 获取预测
        pred_mask = torch.argmax(output, dim=1)  # (1, H, W)
        pred_mask = pred_mask.squeeze(0).cpu().numpy()  # (H, W)
        
        return pred_mask, image_rgb
    
    def postprocess(
        self,
        pred_mask: np.ndarray,
        original_image: np.ndarray,
        overlay_alpha: float = 0.5
    ) -> np.ndarray:
        """后处理：生成可视化结果
        
        Args:
            pred_mask: 预测 mask
            original_image: 原始图像 (RGB)
            overlay_alpha: 叠加透明度
            
        Returns:
            可视化结果
        """
        # 将 mask 转换为彩色
        mask_rgb = self.mask_to_rgb(pred_mask)
        
        # 叠加在原始图像上
        overlay = cv2.addWeighted(
            original_image,
            1 - overlay_alpha,
            mask_rgb,
            overlay_alpha,
            0
        )
        
        return overlay
    
    def mask_to_rgb(self, mask: np.ndarray) -> np.ndarray:
        """将 mask 转换为 RGB 彩色图像
        
        Args:
            mask: 单通道 mask (H, W)
            
        Returns:
            RGB 图像 (H, W, 3)
        """
        h, w = mask.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        
        for class_id, color in self.CLASS_COLORS.items():
            if class_id < self.num_classes:
                rgb[mask == class_id] = color
        
        return rgb
    
    def infer_batch(
        self,
        image_dir: str,
        output_dir: str = None,
        save_mask: bool = True,
        save_overlay: bool = True
    ) -> List[dict]:
        """批量推理
        
        Args:
            image_dir: 图像目录
            output_dir: 输出目录
            save_mask: 是否保存 mask
            save_overlay: 是否保存叠加图
            
        Returns:
            推理结果列表
        """
        image_dir = Path(image_dir)
        results = []
        
        # 创建输出目录
        if output_dir:
            output_dir = Path(output_dir)
            if save_mask:
                (output_dir / 'masks').mkdir(parents=True, exist_ok=True)
            if save_overlay:
                (output_dir / 'overlays').mkdir(parents=True, exist_ok=True)
        
        # 获取所有图像
        image_files = sorted([
            f for f in image_dir.glob('*')
            if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']
        ])
        
        print(f"找到 {len(image_files)} 张图像")
        
        for idx, image_path in enumerate(image_files, 1):
            try:
                # 推理
                pred_mask, original_image = self.infer(str(image_path))
                
                # 后处理
                overlay = self.postprocess(pred_mask, original_image)
                
                # 统计结果
                unique_classes = np.unique(pred_mask)
                class_info = {
                    class_id: np.sum(pred_mask == class_id)
                    for class_id in unique_classes
                }
                
                result = {
                    'image_name': image_path.name,
                    'pred_mask': pred_mask,
                    'overlay': overlay,
                    'class_distribution': class_info
                }
                results.append(result)
                
                # 保存结果
                if output_dir:
                    if save_mask:
                        mask_path = output_dir / 'masks' / f"{image_path.stem}_mask.png"
                        cv2.imwrite(str(mask_path), pred_mask)
                    
                    if save_overlay:
                        overlay_path = output_dir / 'overlays' / f"{image_path.stem}_overlay.jpg"
                        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(str(overlay_path), overlay_bgr)
                
                print(f"  [{idx}/{len(image_files)}] 已处理 {image_path.name}")
            
            except Exception as e:
                print(f"  [{idx}/{len(image_files)}] 错误: {image_path.name} - {e}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='模型推理脚本')
    
    # 模型参数
    parser.add_argument('--model-path', type=str, required=True,
                       help='模型检查点路径')
    parser.add_argument('--num-classes', type=int, default=7,
                       help='类别数')
    
    # 推理参数
    parser.add_argument('--image-dir', type=str,
                       help='输入图像目录')
    parser.add_argument('--image-path', type=str,
                       help='单张图像路径')
    parser.add_argument('--device', type=str, default='cuda',
                       help='计算设备')
    
    # 输出参数
    parser.add_argument('--output-dir', type=str, default='results/inference',
                       help='输出目录')
    parser.add_argument('--save-mask', action='store_true', default=True,
                       help='是否保存 mask')
    parser.add_argument('--save-overlay', action='store_true', default=True,
                       help='是否保存叠加图')
    
    args = parser.parse_args()
    
    # 初始化推理器
    inference = SegmentationInference(
        model_path=args.model_path,
        num_classes=args.num_classes,
        device=args.device
    )
    
    # 单张图像或批量推理
    if args.image_path:
        print(f"\n推理图像: {args.image_path}")
        pred_mask, original_image = inference.infer(args.image_path)
        overlay = inference.postprocess(pred_mask, original_image)
        
        # 显示结果
        print(f"预测类别分布: {np.unique(pred_mask)}")
        
        # 保存结果
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        mask_path = output_dir / f"{Path(args.image_path).stem}_mask.png"
        overlay_path = output_dir / f"{Path(args.image_path).stem}_overlay.jpg"
        
        cv2.imwrite(str(mask_path), pred_mask)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(overlay_path), overlay_bgr)
        
        print(f"结果已保存到 {output_dir}")
    
    elif args.image_dir:
        print(f"\n批量推理目录: {args.image_dir}")
        results = inference.infer_batch(
            image_dir=args.image_dir,
            output_dir=args.output_dir,
            save_mask=args.save_mask,
            save_overlay=args.save_overlay
        )
        print(f"\n推理完成！处理了 {len(results)} 张图像")
        print(f"结果已保存到 {args.output_dir}")
    
    else:
        print("错误: 请提供 --image-dir 或 --image-path")


if __name__ == '__main__':
    main()
