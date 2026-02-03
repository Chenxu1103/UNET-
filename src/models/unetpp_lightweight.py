"""
轻量化 U-Net++ 模型实现

支持 MobileNetV3-Small / ShuffleNetV2 等轻量化编码器
适用于边缘设备部署（RV1126 3TOPS）

参考：绕包机器算法检测项目方案以及实施计划
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional, List, Union


class ConvBlock(nn.Module):
    """基础卷积块：两层卷积 + BN + ReLU"""
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


class LightweightNestedUNet(nn.Module):
    """
    轻量化 U-Net++ 模型

    支持多种轻量化编码器：
    - mobilenet_v3_small: MobileNetV3-Small (推荐用于RV1126)
    - mobilenet_v3_large: MobileNetV3-Large
    - shufflenet_v2_x1_0: ShuffleNetV2 1.0x
    - resnet18: ResNet18

    Args:
        num_classes: 输出类别数
        encoder: 编码器类型 ('mobilenet_v3_small', 'shufflenet_v2_x1_0', 'resnet18')
        pretrained_encoder: 是否使用预训练权重
        deep_supervision: 是否启用深度监督
        decoder_channels: 解码器通道数列表 [16, 24, 40, 80]
    """

    # 编码器输出通道数配置
    ENCODER_CHANNELS = {
        'mobilenet_v3_small': [16, 24, 40, 48, 576],   # MobileNetV3-Small
        'mobilenet_v3_large': [16, 24, 40, 112, 960],  # MobileNetV3-Large
        'shufflenet_v2_x1_0': [24, 116, 232, 464, 1024], # ShuffleNetV2
        'resnet18': [64, 64, 128, 256, 512],           # ResNet18
        'resnet34': [64, 64, 128, 256, 512],           # ResNet34
        'custom': [32, 64, 128, 256, 512],             # 自定义编码器
    }

    def __init__(
        self,
        num_classes: int,
        encoder: str = 'mobilenet_v3_small',
        pretrained_encoder: bool = False,
        deep_supervision: bool = False,
        decoder_channels: Optional[List[int]] = None
    ) -> None:
        super(LightweightNestedUNet, self).__init__()

        if encoder not in self.ENCODER_CHANNELS:
            raise ValueError(
                f"Unsupported encoder: {encoder}. "
                f"Choose from {list(self.ENCODER_CHANNELS.keys())}"
            )

        self.num_classes = num_classes
        self.encoder_name = encoder
        self.deep_supervision = deep_supervision

        # 解码器通道数（默认使用轻量化配置）
        if decoder_channels is None:
            # 根据编码器自动选择合适的解码器通道数
            if 'mobilenet_v3_small' in encoder:
                decoder_channels = [16, 24, 40, 80]
            elif 'mobilenet_v3_large' in encoder:
                decoder_channels = [24, 40, 80, 160]
            elif 'shufflenet' in encoder:
                decoder_channels = [32, 64, 128, 256]
            else:
                decoder_channels = [64, 128, 256, 512]

        # 构建编码器
        self.encoder = self._build_encoder(encoder, pretrained_encoder)

        # 获取编码器各层输出通道数
        enc_channels = self.ENCODER_CHANNELS[encoder]

        # 上采样层
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # 解码器 - U-Net++ 嵌套结构
        # x3_1: 连接 encoder[3] 和 up(encoder[4])
        self.conv3_1 = ConvBlock(decoder_channels[3] + enc_channels[4], decoder_channels[3])
        # x2_2: 连接 encoder[2] 和 up(x3_1)
        self.conv2_2 = ConvBlock(decoder_channels[2] + enc_channels[3], decoder_channels[2])
        # x1_3: 连接 encoder[1] 和 up(x2_2)
        self.conv1_3 = ConvBlock(decoder_channels[1] + enc_channels[2], decoder_channels[1])
        # x0_4: 连接 encoder[0] 和 up(x1_3)
        self.conv0_4 = ConvBlock(decoder_channels[0] + enc_channels[1], decoder_channels[0])

        # 最终输出
        self.final = nn.Conv2d(decoder_channels[0], num_classes, kernel_size=1)

        # 深度监督输出
        if deep_supervision:
            self.ds3_1 = nn.Conv2d(decoder_channels[3], num_classes, kernel_size=1)
            self.ds2_2 = nn.Conv2d(decoder_channels[2], num_classes, kernel_size=1)
            self.ds1_3 = nn.Conv2d(decoder_channels[1], num_classes, kernel_size=1)

    def _build_encoder(self, encoder: str, pretrained: bool) -> nn.Module:
        """构建编码器"""
        if encoder == 'mobilenet_v3_small':
            backbone = models.mobilenet_v3_small(
                weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
            )
            # MobileNetV3 层提取
            layers = [
                backbone.features[:2],   # 初始卷积 + InvertedResidual (输出16通道)
                backbone.features[2:4],  # 输出24通道
                backbone.features[4:7],  # 输出40通道
                backbone.features[7:9],  # 输出48通道
                backbone.features[9:]    # 输出576通道
            ]
            return nn.Sequential(*[nn.Sequential(*layer) if isinstance(layer, list) else layer
                                   for layer in layers])

        elif encoder == 'mobilenet_v3_large':
            backbone = models.mobilenet_v3_large(
                weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
            )
            layers = [
                backbone.features[:2],   # 输出16通道
                backbone.features[2:4],  # 输出24通道
                backbone.features[4:7],  # 输出40通道
                backbone.features[7:9],  # 输出112通道
                backbone.features[9:]    # 输出960通道
            ]
            return nn.Sequential(*[nn.Sequential(*layer) if isinstance(layer, list) else layer
                                   for layer in layers])

        elif encoder == 'shufflenet_v2_x1_0':
            backbone = models.shufflenet_v2_x1_0(
                weights=models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1 if pretrained else None
            )
            # ShuffleNetV2 最大池化后的特征层
            return nn.ModuleList([
                backbone.conv1,           # 初始卷积
                backbone.maxpool,         # 池化
                backbone.stage2,          # stage2
                backbone.stage3,          # stage3
                backbone.stage4,          # stage4
            ])

        elif encoder in ['resnet18', 'resnet34']:
            resnet = models.resnet18() if encoder == 'resnet18' else models.resnet34()
            if pretrained:
                weights = models.ResNet18_Weights.IMAGENET1K_V1 if encoder == 'resnet18' else \
                          models.ResNet34_Weights.IMAGENET1K_V1
                resnet.load_state_dict(weights.state_dict())

            return nn.ModuleList([
                nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool),
                resnet.layer1,
                resnet.layer2,
                resnet.layer3,
                resnet.layer4,
            ])

        else:  # custom encoder
            return nn.ModuleList([
                ConvBlock(3, 32),
                ConvBlock(32, 64),
                ConvBlock(64, 128),
                ConvBlock(128, 256),
                ConvBlock(256, 512),
            ])

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        前向传播

        Args:
            x: 输入 (B, 3, H, W)

        Returns:
            训练时启用深度监督返回 [out, out1, out2, out3]
            推理时返回主输出 out
        """
        # 编码器特征提取
        if 'mobilenet' in self.encoder_name:
            # MobileNetV3 特征提取
            x0_0 = self.encoder[0](x)
            x1_0 = self.encoder[1](x0_0)
            x2_0 = self.encoder[2](x1_0)
            x3_0 = self.encoder[3](x2_0)
            x4_0 = self.encoder[4](x3_0)

        elif 'shufflenet' in self.encoder_name:
            # ShuffleNetV2 特征提取
            x0 = self.encoder[0](x)
            x0_0 = self.encoder[1](x0)
            x1_0 = self.encoder[2](x0_0)
            x2_0 = self.encoder[3](x1_0)
            x3_0 = self.encoder[4](x2_0)
            x4_0 = F.max_pool2d(x3_0, kernel_size=2)

        else:  # ResNet or custom
            x0_0 = self.encoder[0](x)
            x1_0 = self.encoder[1](x0_0)
            x2_0 = self.encoder[2](x1_0)
            x3_0 = self.encoder[3](x2_0)
            x4_0 = self.encoder[4](x3_0)

        # 解码器 - U-Net++ 嵌套上采样
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], dim=1))

        # 主输出
        out = self.final(x0_4)

        # 深度监督
        if self.deep_supervision and self.training:
            out3 = F.interpolate(
                self.ds3_1(x3_1), size=x.shape[2:], mode='bilinear', align_corners=True
            )
            out2 = F.interpolate(
                self.ds2_2(x2_2), size=x.shape[2:], mode='bilinear', align_corners=True
            )
            out1 = F.interpolate(
                self.ds1_3(x1_3), size=x.shape[2:], mode='bilinear', align_corners=True
            )
            return [out, out1, out2, out3]

        return out

    def get_model_size(self) -> int:
        """计算模型参数量"""
        return sum(p.numel() for p in self.parameters())


def create_lightweight_unet(
    num_classes: int = 7,
    encoder: str = 'mobilenet_v3_small',
    pretrained: bool = False,
    deep_supervision: bool = False
) -> LightweightNestedUNet:
    """
    创建轻量化 U-Net++ 模型工厂函数

    Args:
        num_classes: 类别数（默认7：background + cable + tape + 4种缺陷）
        encoder: 编码器类型
        pretrained: 是否使用预训练权重
        deep_supervision: 是否启用深度监督

    Returns:
        LightweightNestedUNet 模型
    """
    model = LightweightNestedUNet(
        num_classes=num_classes,
        encoder=encoder,
        pretrained_encoder=pretrained,
        deep_supervision=deep_supervision
    )

    # 打印模型信息
    n_params = model.get_model_size()
    print(f"Model: Lightweight U-Net++ with {encoder} encoder")
    print(f"Parameters: {n_params:,} ({n_params / 1e6:.2f}M)")

    return model


if __name__ == "__main__":
    # 测试模型
    model = create_lightweight_unet(
        num_classes=7,
        encoder='mobilenet_v3_small',
        pretrained=False,
        deep_supervision=True
    )

    # 测试前向传播
    x = torch.randn(1, 3, 256, 256)
    model.train()
    outputs = model(x)
    if isinstance(outputs, list):
        print(f"Outputs: {[o.shape for o in outputs]}")
    else:
        print(f"Output: {outputs.shape}")

    # 测试推理模式
    model.eval()
    with torch.no_grad():
        output = model(x)
        print(f"Inference output: {output.shape}")
