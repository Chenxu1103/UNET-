"""Simple U-Net for cable wrapping defect detection

Compatible with checkpoints/best_model.pth (enc1.0 architecture)

The checkpoint has this exact structure:
- enc1: Conv -> ReLU -> Conv -> ReLU (keys: enc1.0, enc1.2)
- enc2: Conv -> ReLU -> Conv -> ReLU (keys: enc2.0, enc2.2) - MaxPool applied before
- enc3: Conv -> ReLU -> Conv -> ReLU (keys: enc3.0, enc3.2) - MaxPool applied before
- enc4: Conv -> ReLU -> Conv -> ReLU (keys: enc4.0, enc4.2) - MaxPool applied before
- up3, up2, up1: ConvTranspose2d
- dec3: Conv -> ReLU -> Conv -> ReLU (keys: dec3.0, dec3.2)
- dec2: Conv -> ReLU -> Conv -> ReLU (keys: dec2.0, dec2.2)
- dec1: Conv -> ReLU -> Conv -> ReLU (keys: dec1.0, dec1.2)
- final: Conv2d
"""
import torch
import torch.nn as nn


class SimpleUNet(nn.Module):
    """Simple U-Net matching the exact checkpoint structure"""

    def __init__(self, num_classes=7, num_channels=3):
        super(SimpleUNet, self).__init__()

        self.num_classes = num_classes
        self.num_channels = num_channels

        # Encoder - each enc is a ModuleList with [Conv, ReLU, Conv, ReLU]
        # Using ModuleList so we can index with 0, 1, 2, 3
        self.enc1 = nn.ModuleList([
            nn.Conv2d(num_channels, 64, kernel_size=3, padding=1),  # enc1.0
            nn.ReLU(inplace=True),                                    # enc1.1 (no params)
            nn.Conv2d(64, 64, kernel_size=3, padding=1),             # enc1.2
            nn.ReLU(inplace=True)                                     # enc1.3 (no params)
        ])

        self.enc2 = nn.ModuleList([
            nn.Conv2d(64, 128, kernel_size=3, padding=1),            # enc2.0
            nn.ReLU(inplace=True),                                    # enc2.1 (no params)
            nn.Conv2d(128, 128, kernel_size=3, padding=1),           # enc2.2
            nn.ReLU(inplace=True)                                     # enc2.3 (no params)
        ])

        self.enc3 = nn.ModuleList([
            nn.Conv2d(128, 256, kernel_size=3, padding=1),           # enc3.0
            nn.ReLU(inplace=True),                                    # enc3.1 (no params)
            nn.Conv2d(256, 256, kernel_size=3, padding=1),           # enc3.2
            nn.ReLU(inplace=True)                                     # enc3.3 (no params)
        ])

        self.enc4 = nn.ModuleList([
            nn.Conv2d(256, 512, kernel_size=3, padding=1),           # enc4.0
            nn.ReLU(inplace=True),                                    # enc4.1 (no params)
            nn.Conv2d(512, 512, kernel_size=3, padding=1),           # enc4.2
            nn.ReLU(inplace=True)                                     # enc4.3 (no params)
        ])

        # MaxPool layers (not in checkpoint keys, applied separately)
        self.pool = nn.MaxPool2d(2, 2)

        # Decoder upsampling
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)  # up3
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # up2
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)   # up1

        # Decoder
        self.dec3 = nn.ModuleList([
            nn.Conv2d(256 + 256, 256, kernel_size=3, padding=1),    # dec3.0
            nn.ReLU(inplace=True),                                    # dec3.1 (no params)
            nn.Conv2d(256, 256, kernel_size=3, padding=1),           # dec3.2
            nn.ReLU(inplace=True)                                     # dec3.3 (no params)
        ])

        self.dec2 = nn.ModuleList([
            nn.Conv2d(128 + 128, 128, kernel_size=3, padding=1),    # dec2.0
            nn.ReLU(inplace=True),                                    # dec2.1 (no params)
            nn.Conv2d(128, 128, kernel_size=3, padding=1),           # dec2.2
            nn.ReLU(inplace=True)                                     # dec2.3 (no params)
        ])

        self.dec1 = nn.ModuleList([
            nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1),       # dec1.0
            nn.ReLU(inplace=True),                                    # dec1.1 (no params)
            nn.Conv2d(64, 64, kernel_size=3, padding=1),            # dec1.2
            nn.ReLU(inplace=True)                                     # dec1.3 (no params)
        ])

        # Final classification
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)      # final

    def forward(self, x):
        # Encoder
        # enc1: Conv -> ReLU -> Conv -> ReLU
        enc1 = self.enc1[3](self.enc1[2](self.enc1[1](self.enc1[0](x))))

        # enc2: Pool -> Conv -> ReLU -> Conv -> ReLU
        enc2 = self.pool(enc1)
        enc2 = self.enc2[3](self.enc2[2](self.enc2[1](self.enc2[0](enc2))))

        # enc3: Pool -> Conv -> ReLU -> Conv -> ReLU
        enc3 = self.pool(enc2)
        enc3 = self.enc3[3](self.enc3[2](self.enc3[1](self.enc3[0](enc3))))

        # enc4: Pool -> Conv -> ReLU -> Conv -> ReLU
        enc4 = self.pool(enc3)
        enc4 = self.enc4[3](self.enc4[2](self.enc4[1](self.enc4[0](enc4))))

        # Decoder with skip connections
        # up3 -> cat -> dec3
        up3 = self.up3(enc4)
        dec3 = torch.cat([up3, enc3], dim=1)
        dec3 = self.dec3[3](self.dec3[2](self.dec3[1](self.dec3[0](dec3))))

        # up2 -> cat -> dec2
        up2 = self.up2(dec3)
        dec2 = torch.cat([up2, enc2], dim=1)
        dec2 = self.dec2[3](self.dec2[2](self.dec2[1](self.dec2[0](dec2))))

        # up1 -> cat -> dec1
        up1 = self.up1(dec2)
        dec1 = torch.cat([up1, enc1], dim=1)
        dec1 = self.dec1[3](self.dec1[2](self.dec1[1](self.dec1[0](dec1))))

        # Final
        out = self.final(dec1)

        return out
