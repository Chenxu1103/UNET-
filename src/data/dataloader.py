"""数据加载器工厂类

提供统一的接口创建训练、验证和测试数据加载器，
支持自定义参数和多进程加速。
"""
from typing import Optional
import torch
from torch.utils.data import DataLoader
from .dataset import CableDefectDataset


class DataLoaderFactory:
    """数据加载器工厂类
    
    提供创建不同用途数据加载器的统一接口。
    """
    
    @staticmethod
    def create_train_loader(
        images_dir: str,
        masks_dir: str,
        batch_size: int = 8,
        num_workers: int = 4,
        shuffle: bool = True,
        augment: bool = True,
        pin_memory: bool = True
    ) -> DataLoader:
        """创建训练数据加载器
        
        Args:
            images_dir: 训练图像目录
            masks_dir: 训练 mask 目录
            batch_size: 批次大小（默认8）
            num_workers: 数据加载工作进程数（默认4）
            shuffle: 是否打乱数据（默认True）
            augment: 是否启用数据增强（默认True）
            pin_memory: 是否固定内存以加速 GPU 传输（默认True）
            
        Returns:
            DataLoader: PyTorch 数据加载器
            
        Example:
            >>> train_loader = DataLoaderFactory.create_train_loader(
            ...     images_dir="dataset/processed/train/images",
            ...     masks_dir="dataset/processed/train/masks",
            ...     batch_size=16
            ... )
        """
        dataset = CableDefectDataset(
            image_dir=images_dir,
            mask_dir=masks_dir,
            augment=augment
        )
        
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True  # 丢弃最后一个不完整的批次
        )
    
    @staticmethod
    def create_val_loader(
        images_dir: str,
        masks_dir: str,
        batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = True
    ) -> DataLoader:
        """创建验证数据加载器
        
        Args:
            images_dir: 验证图像目录
            masks_dir: 验证 mask 目录
            batch_size: 批次大小（默认8）
            num_workers: 数据加载工作进程数（默认4）
            pin_memory: 是否固定内存（默认True）
            
        Returns:
            DataLoader: PyTorch 数据加载器（不打乱、不增强）
            
        Example:
            >>> val_loader = DataLoaderFactory.create_val_loader(
            ...     images_dir="dataset/processed/val/images",
            ...     masks_dir="dataset/processed/val/masks"
            ... )
        """
        dataset = CableDefectDataset(
            image_dir=images_dir,
            mask_dir=masks_dir,
            augment=False  # 验证不增强
        )
        
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,  # 验证不打乱
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )
    
    @staticmethod
    def create_test_loader(
        images_dir: str,
        masks_dir: Optional[str] = None,
        batch_size: int = 1,
        num_workers: int = 2,
        pin_memory: bool = True
    ) -> DataLoader:
        """创建测试数据加载器
        
        Args:
            images_dir: 测试图像目录
            masks_dir: 测试 mask 目录（可选，用于评估）
            batch_size: 批次大小（默认1，便于单张推理）
            num_workers: 数据加载工作进程数（默认2）
            pin_memory: 是否固定内存（默认True）
            
        Returns:
            DataLoader: PyTorch 数据加载器
            
        Example:
            >>> test_loader = DataLoaderFactory.create_test_loader(
            ...     images_dir="dataset/processed/test/images",
            ...     batch_size=1
            ... )
        """
        dataset = CableDefectDataset(
            image_dir=images_dir,
            mask_dir=masks_dir or images_dir,  # 如果无 mask，使用图像目录
            augment=False
        )
        
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )
    
    @staticmethod
    def create_inference_loader(
        images_dir: str,
        batch_size: int = 1,
        num_workers: int = 2
    ) -> DataLoader:
        """创建推理数据加载器（仅图像，无 mask）
        
        Args:
            images_dir: 推理图像目录
            batch_size: 批次大小（默认1）
            num_workers: 数据加载工作进程数（默认2）
            
        Returns:
            DataLoader: 仅包含图像的数据加载器
            
        Note:
            此加载器需要修改 CableDefectDataset 以支持无 mask 加载
        """
        dataset = CableDefectDataset(
            image_dir=images_dir,
            mask_dir=images_dir,
            augment=False
        )
        
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )


class DataLoaderConfig:
    """数据加载器配置类
    
    集中管理所有数据加载相关的配置参数。
    """
    
    def __init__(
        self,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        test_batch_size: int = 1,
        num_workers: int = 4,
        pin_memory: bool = True,
        augment_train: bool = True,
        shuffle_train: bool = True,
        drop_last_train: bool = True
    ):
        """初始化配置
        
        Args:
            train_batch_size: 训练批次大小
            val_batch_size: 验证批次大小
            test_batch_size: 测试批次大小
            num_workers: 数据加载工作进程数
            pin_memory: 是否固定内存
            augment_train: 是否增强训练数据
            shuffle_train: 是否打乱训练数据
            drop_last_train: 是否丢弃最后一个不完整批次
        """
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.augment_train = augment_train
        self.shuffle_train = shuffle_train
        self.drop_last_train = drop_last_train
    
    @classmethod
    def from_dict(cls, config_dict: dict):
        """从字典创建配置"""
        return cls(**config_dict)
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'train_batch_size': self.train_batch_size,
            'val_batch_size': self.val_batch_size,
            'test_batch_size': self.test_batch_size,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'augment_train': self.augment_train,
            'shuffle_train': self.shuffle_train,
            'drop_last_train': self.drop_last_train
        }
