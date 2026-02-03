"""数据增强策略模块

提供多种数据增强方法，包括几何变换和色彩增强。
"""
import numpy as np
import cv2
from typing import Tuple
import random


class AugmentationPipeline:
    """数据增强管道
    
    支持多种增强操作，包括：
    - 几何变换：翻转、旋转、仿射变换
    - 色彩变换：亮度、对比度、饱和度
    - 空间变换：随机裁剪、缩放
    """
    
    def __init__(
        self,
        flip_h: bool = True,
        flip_v: bool = False,
        rotate: bool = True,
        rotate_range: Tuple[int, int] = (-15, 15),
        brightness: bool = True,
        brightness_range: Tuple[float, float] = (0.8, 1.2),
        contrast: bool = True,
        contrast_range: Tuple[float, float] = (0.8, 1.2),
        scale: bool = False,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        elastic_deform: bool = False,
        apply_probability: float = 0.5
    ):
        """初始化增强管道
        
        Args:
            flip_h: 是否启用水平翻转
            flip_v: 是否启用垂直翻转
            rotate: 是否启用旋转
            rotate_range: 旋转角度范围（度数）
            brightness: 是否启用亮度增强
            brightness_range: 亮度系数范围
            contrast: 是否启用对比度增强
            contrast_range: 对比度系数范围
            scale: 是否启用缩放
            scale_range: 缩放因子范围
            elastic_deform: 是否启用弹性变形
            apply_probability: 应用增强的概率（0-1）
        """
        self.flip_h = flip_h
        self.flip_v = flip_v
        self.rotate = rotate
        self.rotate_range = rotate_range
        self.brightness = brightness
        self.brightness_range = brightness_range
        self.contrast = contrast
        self.contrast_range = contrast_range
        self.scale = scale
        self.scale_range = scale_range
        self.elastic_deform = elastic_deform
        self.apply_probability = apply_probability
    
    def __call__(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """应用增强
        
        Args:
            image: RGB 图像 (H, W, 3)，值域 [0, 255]
            mask: 分割 mask (H, W)，像素值为类别 ID
            
        Returns:
            增强后的 (image, mask) 对
        """
        # 概率性应用增强
        if random.random() > self.apply_probability:
            return image, mask
        
        # 几何变换（对图像和 mask 同步应用）
        if self.flip_h and random.random() > 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
        
        if self.flip_v and random.random() > 0.5:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)
        
        if self.rotate and random.random() > 0.5:
            angle = random.randint(self.rotate_range[0], self.rotate_range[1])
            image, mask = self._rotate(image, mask, angle)
        
        if self.scale and random.random() > 0.5:
            scale_factor = random.uniform(self.scale_range[0], self.scale_range[1])
            image, mask = self._scale(image, mask, scale_factor)
        
        # 色彩变换（仅对图像应用）
        if self.brightness and random.random() > 0.5:
            factor = random.uniform(self.brightness_range[0], self.brightness_range[1])
            image = self._adjust_brightness(image, factor)
        
        if self.contrast and random.random() > 0.5:
            factor = random.uniform(self.contrast_range[0], self.contrast_range[1])
            image = self._adjust_contrast(image, factor)
        
        return image, mask
    
    @staticmethod
    def _rotate(
        image: np.ndarray,
        mask: np.ndarray,
        angle: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """旋转图像和 mask
        
        Args:
            image: 图像 (H, W, 3)
            mask: mask (H, W)
            angle: 旋转角度（度数）
            
        Returns:
            旋转后的 (image, mask) 对
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # 获取旋转矩阵
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # 旋转图像（使用线性插值）
        image = cv2.warpAffine(
            image, matrix, (w, h),
            borderMode=cv2.BORDER_REFLECT,
            flags=cv2.INTER_LINEAR
        )
        
        # 旋转 mask（使用最近邻插值以保持类别）
        mask = cv2.warpAffine(
            mask, matrix, (w, h),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
            flags=cv2.INTER_NEAREST
        )
        
        return image, mask
    
    @staticmethod
    def _scale(
        image: np.ndarray,
        mask: np.ndarray,
        scale_factor: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """缩放图像和 mask
        
        Args:
            image: 图像 (H, W, 3)
            mask: mask (H, W)
            scale_factor: 缩放因子
            
        Returns:
            缩放后的 (image, mask) 对
        """
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        
        # 缩放
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        # 如果缩放后尺寸不同，进行 padding 或裁剪
        if scale_factor > 1:
            # 裁剪到原始大小
            start_h = (new_h - h) // 2
            start_w = (new_w - w) // 2
            image = image[start_h:start_h+h, start_w:start_w+w]
            mask = mask[start_h:start_h+h, start_w:start_w+w]
        else:
            # Padding 到原始大小
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2
            image = cv2.copyMakeBorder(
                image, pad_h, h - new_h - pad_h, pad_w, w - new_w - pad_w,
                cv2.BORDER_REFLECT
            )
            mask = cv2.copyMakeBorder(
                mask, pad_h, h - new_h - pad_h, pad_w, w - new_w - pad_w,
                cv2.BORDER_CONSTANT, value=0
            )
        
        return image, mask
    
    @staticmethod
    def _adjust_brightness(
        image: np.ndarray,
        factor: float
    ) -> np.ndarray:
        """调整亮度
        
        Args:
            image: 图像 (H, W, 3)，值域 [0, 255]
            factor: 亮度系数（1.0 表示无变化）
            
        Returns:
            调整后的图像
        """
        image = image.astype(np.float32)
        image = image * factor
        image = np.clip(image, 0, 255)
        return image.astype(np.uint8)
    
    @staticmethod
    def _adjust_contrast(
        image: np.ndarray,
        factor: float
    ) -> np.ndarray:
        """调整对比度
        
        Args:
            image: 图像 (H, W, 3)，值域 [0, 255]
            factor: 对比度系数（1.0 表示无变化）
            
        Returns:
            调整后的图像
        """
        image = image.astype(np.float32)
        mean = image.mean()
        image = (image - mean) * factor + mean
        image = np.clip(image, 0, 255)
        return image.astype(np.uint8)
    
    @staticmethod
    def _adjust_saturation(
        image: np.ndarray,
        factor: float
    ) -> np.ndarray:
        """调整饱和度
        
        Args:
            image: RGB 图像 (H, W, 3)，值域 [0, 255]
            factor: 饱和度系数（1.0 表示无变化）
            
        Returns:
            调整后的图像
        """
        # 转换为 HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # 调整 S 通道
        hsv[:, :, 1] = hsv[:, :, 1] * factor
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        
        # 转换回 RGB
        hsv = hsv.astype(np.uint8)
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        return image


class StrongAugmentationPipeline(AugmentationPipeline):
    """强增强管道
    
    应用更激进的增强参数，适合数据较少的情况。
    """
    
    def __init__(self):
        """初始化强增强管道"""
        super().__init__(
            flip_h=True,
            flip_v=True,
            rotate=True,
            rotate_range=(-25, 25),
            brightness=True,
            brightness_range=(0.7, 1.3),
            contrast=True,
            contrast_range=(0.7, 1.3),
            scale=True,
            scale_range=(0.8, 1.2),
            apply_probability=0.7
        )


class WeakAugmentationPipeline(AugmentationPipeline):
    """弱增强管道
    
    应用较为保守的增强参数，适合大数据集。
    """
    
    def __init__(self):
        """初始化弱增强管道"""
        super().__init__(
            flip_h=True,
            flip_v=False,
            rotate=True,
            rotate_range=(-10, 10),
            brightness=True,
            brightness_range=(0.9, 1.1),
            contrast=False,
            scale=False,
            apply_probability=0.3
        )
