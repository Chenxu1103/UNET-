"""简化训练脚本 - 用于快速测试"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sys
import os
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# 简单数据集类
class SimpleDataset(Dataset):
    def __init__(self, images_dir, masks_dir):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)

        # 查找对应的 mask
        mask_name = os.path.splitext(img_name)[0] + '.png'
        mask_path = os.path.join(self.masks_dir, mask_name)

        # 读取图像
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)

        # 读取 mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        mask = torch.from_numpy(mask).long()

        return image, mask


# 简单 U-Net 模型
class SimpleUNet(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()

        # 编码器
        self.enc1 = self.conv_block(3, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        # 解码器
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self.conv_block(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self.conv_block(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self.conv_block(128, 64)

        # 输出
        self.final = nn.Conv2d(64, num_classes, 1)

        # 池化
        self.pool = nn.MaxPool2d(2)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 编码
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # 解码
        d3 = self.up3(e4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.final(d1)


def iou_score(pred, target, num_classes):
    """计算 mIoU"""
    ious = []
    pred = pred.argmax(dim=1)
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        if union > 0:
            ious.append(intersection / union)
    return np.mean(ious) if ious else 0.0


def main():
    # 参数
    train_img_dir = "dataset/processed/train/images"
    train_mask_dir = "dataset/processed/train/masks"
    val_img_dir = "dataset/processed/val/images"
    val_mask_dir = "dataset/processed/val/masks"
    num_classes = 7
    num_epochs = 100
    batch_size = 2
    lr = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # 创建数据集
    train_dataset = SimpleDataset(train_img_dir, train_mask_dir)
    val_dataset = SimpleDataset(val_img_dir, val_mask_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # 创建模型
    model = SimpleUNet(num_classes=num_classes).to(device)

    # 尝试加载已有模型
    checkpoint_path = "checkpoints/best_model.pth"
    if os.path.exists(checkpoint_path):
        print(f"Loading existing model from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Model loaded successfully!")
    else:
        print("No existing model found, starting from scratch")

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 训练循环
    best_miou = 0.0

    for epoch in range(num_epochs):
        # 训练
        model.train()
        train_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            # 调整输出尺寸以匹配 mask
            if outputs.shape[2:] != masks.shape[1:]:
                outputs = nn.functional.interpolate(outputs, size=masks.shape[1:], mode='bilinear', align_corners=False)

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # 验证
        model.eval()
        val_loss = 0.0
        all_ious = []
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)

                if outputs.shape[2:] != masks.shape[1:]:
                    outputs = nn.functional.interpolate(outputs, size=masks.shape[1:], mode='bilinear', align_corners=False)

                loss = criterion(outputs, masks)
                val_loss += loss.item()

                # 计算 IoU
                iou = iou_score(outputs, masks, num_classes)
                all_ious.append(iou)

        avg_val_loss = val_loss / len(val_loader)
        miou = np.mean(all_ious)

        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, mIoU={miou:.4f}")

        # 保存最佳模型
        if miou > best_miou:
            best_miou = miou
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/best_model.pth")
            print(f"  -> Saved best model (mIoU={best_miou:.4f})")

    print(f"\nTraining completed! Best mIoU: {best_miou:.4f}")


if __name__ == "__main__":
    main()
