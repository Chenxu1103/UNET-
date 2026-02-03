"""Sliding window inference for binary patch model

Test the trained model on full validation images.
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
import sys
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.unetpp import NestedUNet


class SlidingWindowInference:
    """Sliding window inference for patch-based model"""

    def __init__(
        self,
        model,
        patch_size=384,
        stride=192,
        target_size=256,
        num_classes=2
    ):
        self.model = model
        self.patch_size = patch_size
        self.stride = stride  # 50% overlap
        self.target_size = target_size
        self.num_classes = num_classes

    def predict(self, image):
        """Predict on full image using sliding window"""
        h, w = image.shape[:2]

        # Calculate number of patches
        n_h = (h - self.patch_size) // self.stride + 1
        n_w = (w - self.patch_size) // self.stride + 1

        # Handle edge cases
        if (h - self.patch_size) % self.stride != 0:
            n_h += 1
        if (w - self.patch_size) % self.stride != 0:
            n_w += 1

        # Output arrays
        output = np.zeros((h, w, self.num_classes), dtype=np.float32)
        count = np.zeros((h, w, 1), dtype=np.float32)

        self.model.eval()
        with torch.no_grad():
            for i in range(n_h):
                for j in range(n_w):
                    # Calculate patch position
                    y = i * self.stride
                    x = j * self.stride

                    # Adjust for edge patches
                    y_end = min(y + self.patch_size, h)
                    x_end = min(x + self.patch_size, w)

                    # Adjust start position if near edge
                    y = max(0, y_end - self.patch_size)
                    x = max(0, x_end - self.patch_size)

                    # Extract patch
                    patch = image[y:y_end, x:x_end]

                    # Skip if patch is wrong size
                    if patch.shape[0] != self.patch_size or patch.shape[1] != self.patch_size:
                        # Pad if needed
                        pad_h = self.patch_size - patch.shape[0]
                        pad_w = self.patch_size - patch.shape[1]
                        patch = np.pad(patch, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')

                    # Resize to target size
                    patch_resized = cv2.resize(patch, (self.target_size, self.target_size),
                                              interpolation=cv2.INTER_LINEAR)

                    # Convert to tensor
                    patch_tensor = torch.from_numpy(patch_resized).float() / 255.0
                    patch_tensor = patch_tensor.permute(2, 0, 1).unsqueeze(0)

                    # Predict
                    if torch.cuda.is_available():
                        patch_tensor = patch_tensor.cuda()

                    pred = self.model(patch_tensor)
                    if isinstance(pred, list):
                        pred = pred[-1]

                    pred = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()

                    # Resize back to patch size
                    pred = cv2.resize(pred, (self.patch_size, self.patch_size),
                                     interpolation=cv2.INTER_LINEAR)

                    # Crop to actual size
                    pred = pred[:y_end-y, :x_end-x, :]

                    # Accumulate predictions
                    output[y:y_end, x:x_end, :] += pred
                    count[y:y_end, x:x_end, :] += 1

        # Average overlapping regions
        output = output / (count + 1e-8)

        # Get final prediction
        pred_mask = np.argmax(output, axis=-1)

        return pred_mask, output


def visualize_prediction(image, pred_mask, gt_mask, save_path):
    """Visualize prediction vs ground truth"""
    h, w = image.shape[:2]

    # Resize masks to image size
    pred_resized = cv2.resize(pred_mask.astype(np.uint8), (w, h),
                             interpolation=cv2.INTER_NEAREST)
    gt_resized = cv2.resize(gt_mask.astype(np.uint8), (w, h),
                           interpolation=cv2.INTER_NEAREST)

    # Create overlay
    overlay = image.copy()

    # Ground truth (green)
    overlay[gt_resized == 1] = [0, 255, 0]

    # Prediction (red)
    overlay[pred_resized == 1] = [0, 0, 255]

    # Both (yellow)
    overlay[(gt_resized == 1) & (pred_resized == 1)] = [0, 255, 255]

    # Create side-by-side visualization
    vis_image = image.copy()

    # Draw GT contour (green)
    gt_contour = gt_resized.copy()
    contours, _ = cv2.findContours(gt_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis_image, contours, -1, (0, 255, 0), 2)

    # Draw prediction contour (red)
    pred_contour = pred_resized.copy()
    contours, _ = cv2.findContours(pred_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis_image, contours, -1, (0, 0, 255), 2)

    # Add legend
    cv2.rectangle(vis_image, (10, 10), (200, 90), (255, 255, 255), -1)
    cv2.putText(vis_image, "Green: Ground Truth", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(vis_image, "Red: Prediction", (15, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(vis_image, "Yellow: Both", (15, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    cv2.imwrite(str(save_path), vis_image)


def main():
    print("="*70)
    print("滑窗推理 - 二分类缺陷检测")
    print("="*70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print("\n[1] 加载模型...")
    model = NestedUNet(
        num_classes=2,
        deep_supervision=False,
        pretrained_encoder=False
    ).to(device)

    checkpoint = torch.load("checkpoints_binary_patch/best_model.pth", map_location=device)
    model.load_state_dict(checkpoint["model"])
    print(f"  Loaded from epoch {checkpoint['epoch']}")
    print(f"  Best mIoU: {checkpoint['best_miou']:.4f}")

    # Initialize sliding window inference
    inference = SlidingWindowInference(
        model,
        patch_size=384,
        stride=192,  # 50% overlap
        target_size=256,
        num_classes=2
    )

    # Get validation images
    val_img_dir = Path("dataset/processed_v2/val/images")
    val_mask_dir = Path("dataset/processed_v2/val/masks")

    val_images = sorted(list(val_img_dir.glob("*.jpg")))
    print(f"\n[2] 验证集: {len(val_images)} 张图像")

    # Create output directory
    output_dir = Path("results_binary_patch")
    output_dir.mkdir(exist_ok=True)

    # Metrics
    from utils.metrics import compute_metrics
    all_miou = []
    all_precision = []
    all_recall = []
    all_defect_iou = []

    print("\n[3] 开始推理...")
    print("-"*70)

    for img_path in tqdm(val_images, desc="Processing"):
        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load ground truth
        mask_path = val_mask_dir / (img_path.stem + ".png")
        gt_mask = Image.open(mask_path)
        gt_array = np.array(gt_mask)
        if gt_array.ndim == 3:
            gt_array = gt_array[:, :, 0]

        # Convert to binary
        gt_binary = np.zeros_like(gt_array)
        gt_binary[(gt_array == 3) | (gt_array == 4) | (gt_array == 5)] = 1

        # Predict
        pred_mask, _ = inference.predict(image)

        # Resize prediction to original size
        pred_resized = cv2.resize(pred_mask.astype(np.uint8),
                                 (image.shape[1], image.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)

        # Compute metrics for this image
        miou, precision_dict, recall_dict, iou_dict = compute_metrics(
            pred_resized[np.newaxis, ...],
            gt_binary[np.newaxis, ...],
            2
        )

        all_miou.append(miou)
        all_precision.append(precision_dict.get(1, 0.0))
        all_recall.append(recall_dict.get(1, 0.0))
        all_defect_iou.append(iou_dict.get(1, 0.0))

        # Visualize
        vis_path = output_dir / f"{img_path.stem}_vis.jpg"
        visualize_prediction(image, pred_resized, gt_binary, vis_path)

    # Compute average metrics
    print("\n[4] 计算指标...")
    avg_miou = np.mean(all_miou)
    avg_precision = np.mean(all_precision)
    avg_recall = np.mean(all_recall)
    avg_defect_iou = np.mean(all_defect_iou)

    print("\n" + "="*70)
    print("推理完成!")
    print("="*70)
    print(f"平均 mIoU: {avg_miou:.4f}")
    print(f"平均 Precision: {avg_precision:.4f}")
    print(f"平均 Recall: {avg_recall:.4f}")
    print(f"平均 Defect IoU: {avg_defect_iou:.4f}")
    print(f"\n可视化结果保存到: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
