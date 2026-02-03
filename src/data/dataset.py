from __future__ import annotations
from typing import List, Tuple
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class CableDefectDataset(Dataset):
    """电缆胶带缺陷语义分割 PyTorch Dataset

    从 image_dir 和 mask_dir 中读取图像和对应的mask标签。
    目录结构应为:
      images/: *.jpg|png 图像文件
      masks/:  *.png mask文件（单通道，像素值为类别ID）

    Args:
        image_dir: 图像文件夹路径
        mask_dir: 与图像对应的mask文件夹路径
        augment: 是否在getitem时执行数据增强（默认False）
        target_size: 统一调整图像尺寸 (height, width)，None表示保持原尺寸（默认None）
    """
    def __init__(self, image_dir: str, mask_dir: str, augment: bool = False, target_size: Tuple[int, int] = None) -> None:
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.augment = augment
        self.target_size = target_size  # (height, width)
        
        # 列出所有图像文件
        self.image_files = sorted(
            [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        )
        
        # 验证所有图像对应的mask都存在
        for img_file in self.image_files:
            mask_file = os.path.splitext(img_file)[0] + '.png'
            mask_path = os.path.join(mask_dir, mask_file)
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask not found: {mask_path}")

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            (image_tensor, mask_tensor): 图像和mask的张量对
                - image_tensor: shape (3, H, W)，float32，值域[0,1]
                - mask_tensor: shape (H, W)，int64，值为类别ID
        """
        # 读取图像和mask
        img_file = self.image_files[idx]
        mask_file = os.path.splitext(img_file)[0] + '.png'
        
        img_path = os.path.join(self.image_dir, img_file)
        mask_path = os.path.join(self.mask_dir, mask_file)
        
        # 读取图像（BGR）- 支持中文路径
        img_array = np.fromfile(img_path, dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        # 读取mask（灰度，像素值为类别）- 支持中文路径
        mask_array = np.fromfile(mask_path, dtype=np.uint8)
        mask = cv2.imdecode(mask_array, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        
        # 如果mask是多通道，取第一通道
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        
        # BGR -> RGB 通道转换
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 统一图像尺寸
        if self.target_size is not None:
            h, w = self.target_size
            # 图像使用双线性插值
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
            # mask 使用最近邻插值以保持类别值
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # 可选的数据增强
        if self.augment:
            image, mask = self._apply_augmentation(image, mask)

        # 转换为张量格式 (C,H,W)，并归一化到[0,1]
        image = image.astype(np.float32) / 255.0
        # HWC -> CHW
        image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(image).float()
        
        mask = mask.astype(np.int64)  # mask作为LongTensor类型
        mask = torch.from_numpy(mask).long()
        
        return image, mask
    
    def _apply_augmentation(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """应用数据增强
        
        Args:
            image: RGB 图像 (H,W,3)
            mask: mask 标签 (H,W)
            
        Returns:
            增强后的 (image, mask) 元组
        """
        # 随机水平翻转
        if np.random.random() < 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
        
        # 随机垂直翻转
        if np.random.random() < 0.5:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)
        
        # 随机亮度调整
        if np.random.random() < 0.5:
            factor = 0.7 + np.random.random() * 0.6  # 0.7~1.3之间
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 2] *= factor
            hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
            image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        return image, mask
