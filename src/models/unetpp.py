"""U-Net++ (Nested U-Net) 模型实现

支持预训练编码器、深度监督等特性，用于电缆胶带缺陷分割
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional, List, Union


class ConvBlock(nn.Module):
    """基础卷积块：两层卷积 + BN + ReLU，用于U-Net++各级特征提取"""
    def __init__(self, in_channels: int, out_channels: int):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class NestedUNet(nn.Module):
    """U-Net++ 嵌套U型网络实现
    
    支持ResNet50预训练编码器，带深度监督输出。
    
    Args:
        num_classes: 输出分类数（包括背景）
        input_channels: 输入图像通道数（彩色图像为3）
        deep_supervision: 是否启用深度监督（多阶段输出）
        pretrained_encoder: 是否使用ResNet50预训练编码器
    """
    def __init__(
        self,
        num_classes: int,
        input_channels: int = 3,
        deep_supervision: bool = True,
        pretrained_encoder: bool = False
    ) -> None:
        super(NestedUNet, self).__init__()
        self.deep_supervision = deep_supervision
        nb_filter = [32, 64, 128, 256, 512]  # 各层卷积通道数设置

        # 编码器部分
        if pretrained_encoder:
            # 使用ResNet50预训练模型的前几层作为编码器
            backbone = models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V2
            )
            # 利用ResNet50的特征提取层
            self.conv0_0 = nn.Sequential(
                backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
            )  # 输出通道64
            self.conv1_0 = backbone.layer1  # 输出通道256
            self.conv2_0 = backbone.layer2  # 输出通道512
            self.conv3_0 = backbone.layer3  # 输出通道1024
            self.conv4_0 = backbone.layer4  # 输出通道2048
            up_channels = [64, 256, 512, 1024, 2048]
        else:
            # 使用自定义卷积块构建编码器各层
            self.conv0_0 = ConvBlock(input_channels, nb_filter[0])
            self.conv1_0 = ConvBlock(nb_filter[0], nb_filter[1])
            self.conv2_0 = ConvBlock(nb_filter[1], nb_filter[2])
            self.conv3_0 = ConvBlock(nb_filter[2], nb_filter[3])
            self.conv4_0 = ConvBlock(nb_filter[3], nb_filter[4])
            up_channels = nb_filter

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # 解码器部分 (Nested decoder blocks)
        self.conv3_1 = ConvBlock(up_channels[3] + up_channels[4], up_channels[3])
        self.conv2_2 = ConvBlock(up_channels[2] + up_channels[3], up_channels[2])
        self.conv1_3 = ConvBlock(up_channels[1] + up_channels[2], up_channels[1])
        self.conv0_4 = ConvBlock(up_channels[0] + up_channels[1], up_channels[0])

        # 最终输出卷积层
        self.final = nn.Conv2d(up_channels[0], num_classes, kernel_size=1)

        if self.deep_supervision:
            # 中间深度监督输出卷积层，将中间层特征映射为分割图（通道=num_classes）
            self.ds3_1 = nn.Conv2d(up_channels[3], num_classes, kernel_size=1)
            self.ds2_2 = nn.Conv2d(up_channels[2], num_classes, kernel_size=1)
            self.ds1_3 = nn.Conv2d(up_channels[1], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        """前向传播
        
        Args:
            x: 输入张量 (B,C,H,W)
            
        Returns:
            训练模式下若启用深度监督，返回 [out, out1, out2, out3]
            推理模式下只返回主输出 out
        """
        # 编码器：逐层下采样提取特征
        x0_0 = self.conv0_0(x)  # 第0层输出 (H, W)
        x1_0 = self.conv1_0(self.pool(x0_0))  # 第1层输出 (H/2, W/2)
        x2_0 = self.conv2_0(self.pool(x1_0))  # 第2层输出 (H/4, W/4)
        x3_0 = self.conv3_0(self.pool(x2_0))  # 第3层输出 (H/8, W/8)
        x4_0 = self.conv4_0(self.pool(x3_0))  # 第4层输出 (H/16, W/16)

        # 解码器：逐层上采样拼接
        x3_1 = self.conv3_1(
            torch.cat([x3_0, self.up(x4_0)], dim=1)
        )  # 上采样x4_0并与x3_0拼接
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], dim=1))

        # 最终输出
        out = self.final(x0_4)  # 主输出 (深度最高，完整解码)

        if self.deep_supervision and self.training:
            # 计算各深度辅助输出，并上采样到原始尺寸，用于训练中的深度监督
            out3 = F.interpolate(
                self.ds3_1(x3_1), size=x.shape[2:], mode='bilinear', align_corners=True
            )
            out2 = F.interpolate(
                self.ds2_2(x2_2), size=x.shape[2:], mode='bilinear', align_corners=True
            )
            out1 = F.interpolate(
                self.ds1_3(x1_3), size=x.shape[2:], mode='bilinear', align_corners=True
            )
            # 如果在训练模式，返回所有输出用于计算深度监督损失；推理时仅返回最终输出
            return [out, out1, out2, out3]

        return out  # 默认返回主输出
