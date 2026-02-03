"""优化的滑窗推理 - 概率阈值扫描 + 后处理

A1-A4: 阈值扫描 + Hysteresis + 连通域过滤 + 窗口门控
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
import sys
from tqdm import tqdm
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.unetpp import NestedUNet
from utils.metrics import compute_metrics


class OptimizedSlidingWindowInference:
    """优化的滑窗推理 - 带后处理"""

    def __init__(
        self,
        model,
        patch_size=384,
        stride=192,
        target_size=256,
        num_classes=2,
        gate_thr=0.70  # A4: 窗口门控阈值
    ):
        self.model = model
        self.patch_size = patch_size
        self.stride = stride
        self.target_size = target_size
        self.num_classes = num_classes
        self.gate_thr = gate_thr

    def predict(self, image, use_gating=True):
        """Predict with probability output"""
        h, w = image.shape[:2]

        # Calculate number of patches
        n_h = (h - self.patch_size) // self.stride + 1
        n_w = (w - self.patch_size) // self.stride + 1

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
                    y = i * self.stride
                    x = j * self.stride

                    y_end = min(y + self.patch_size, h)
                    x_end = min(x + self.patch_size, w)

                    y = max(0, y_end - self.patch_size)
                    x = max(0, x_end - self.patch_size)

                    patch = image[y:y_end, x:x_end]

                    if patch.shape[0] != self.patch_size or patch.shape[1] != self.patch_size:
                        pad_h = self.patch_size - patch.shape[0]
                        pad_w = self.patch_size - patch.shape[1]
                        patch = np.pad(patch, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')

                    patch_resized = cv2.resize(patch, (self.target_size, self.target_size),
                                              interpolation=cv2.INTER_LINEAR)

                    patch_tensor = torch.from_numpy(patch_resized).float() / 255.0
                    patch_tensor = patch_tensor.permute(2, 0, 1).unsqueeze(0)

                    if torch.cuda.is_available():
                        patch_tensor = patch_tensor.cuda()

                    logits = self.model(patch_tensor)
                    if isinstance(logits, list):
                        logits = logits[-1]

                    # A4: 窗口门控 - 计算窗口缺陷置信度
                    if use_gating:
                        probs = torch.softmax(logits, dim=1)  # [1, 2, H, W]
                        defect_prob = probs[0, 1, :, :].cpu().numpy()  # 缺陷类概率
                        gate_score = np.max(defect_prob)  # 窗口最大置信度

                        # 如果窗口置信度太低，跳过融合
                        if gate_score < self.gate_thr:
                            continue

                        pred = probs.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    else:
                        pred = torch.softmax(logits, dim=1).squeeze(0).permute(1, 2, 0).cpu().numpy()

                    pred = cv2.resize(pred, (self.patch_size, self.patch_size),
                                     interpolation=cv2.INTER_LINEAR)
                    pred = pred[:y_end-y, :x_end-x, :]

                    output[y:y_end, x:x_end, :] += pred
                    count[y:y_end, x:x_end, :] += 1

        output = output / (count + 1e-8)

        return output


def apply_hysteresis(prob_map, thr_high=0.90, thr_low=0.70):
    """A2: 滞后阈值/种子生长

    高阈值找种子，低阈值只允许在种子附近生长
    """
    # 高阈值找种子
    seeds = (prob_map >= thr_high).astype(np.uint8)

    # 低阈值区域
    low_region = (prob_map >= thr_low).astype(np.uint8)

    # 形态学膨胀种子区域
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    seeds_dilated = cv2.dilate(seeds, kernel, iterations=3)

    # 只保留在种子附近的低阈值区域
    result = np.logical_or(seeds.astype(bool),
                           np.logical_and(low_region.astype(bool),
                                         seeds_dilated.astype(bool)))

    return result.astype(np.uint8)


def apply_morphological_and_filtering(pred_mask, prob_map, min_area=50, mean_prob_thr=0.85):
    """A3: 连通域过滤 + 形态学去噪

    - 形态学开闭运算
    - 删除小面积连通域
    - 过滤平均概率低的区域
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # 开运算：去噪点
    cleaned = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # 闭运算：补洞
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 连通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        cleaned, connectivity=8
    )

    # 过滤
    filtered = np.zeros_like(cleaned)

    for label in range(1, num_labels):  # 跳过背景
        area = stats[label, cv2.CC_STAT_AREA]

        # 面积过滤
        if area < min_area:
            continue

        # 平均概率过滤
        mask = (labels == label)
        mean_prob = np.mean(prob_map[mask])

        if mean_prob >= mean_prob_thr:
            filtered[mask] = 1

    return filtered


def scan_thresholds(val_images, val_masks, inference, device,
                    thr_range=(0.50, 0.99, 0.01)):
    """A1: 阈值扫描 - 找最优阈值

    扫描不同阈值，找到满足Recall>=90%时最大化Precision的阈值
    """
    print("\n[A1] 概率阈值扫描...")
    print("-"*70)

    results = []

    for thr in np.arange(thr_range[0], thr_range[1], thr_range[2]):
        thr = round(thr, 2)
        all_miou, all_precision, all_recall = [], [], []

        for img_path, mask_path in zip(val_images, val_masks):
            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            gt_mask = Image.open(mask_path)
            gt_array = np.array(gt_mask)
            if gt_array.ndim == 3:
                gt_array = gt_array[:, :, 0]

            gt_binary = np.zeros_like(gt_array)
            gt_binary[(gt_array == 3) | (gt_array == 4) | (gt_array == 5)] = 1

            # 推理
            prob_map = inference.predict(image, use_gating=False)
            defect_prob = prob_map[:, :, 1]  # 缺陷类概率

            # 阈值
            pred_mask = (defect_prob >= thr).astype(np.uint8)

            # Resize
            pred_resized = cv2.resize(pred_mask, (image.shape[1], image.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)

            # 指标
            miou, prec_dict, rec_dict, _ = compute_metrics(
                pred_resized[np.newaxis, ...],
                gt_binary[np.newaxis, ...],
                2
            )

            all_miou.append(miou)
            all_precision.append(prec_dict.get(1, 0.0))
            all_recall.append(rec_dict.get(1, 0.0))

        avg_miou = np.mean(all_miou)
        avg_precision = np.mean(all_precision)
        avg_recall = np.mean(all_recall)

        f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall + 1e-8)

        results.append({
            'thr': thr,
            'miou': avg_miou,
            'precision': avg_precision,
            'recall': avg_recall,
            'f1': f1
        })

        if thr % 0.05 < 0.01 or thr >= 0.90:
            print(f"  阈值 {thr:.2f}: mIoU={avg_miou:.4f}, P={avg_precision:.4f}, R={avg_recall:.4f}, F1={f1:.4f}")

    # 找最优阈值
    print("\n[最优阈值分析]")

    # 策略1: 最大化F1
    best_f1 = max(results, key=lambda x: x['f1'])
    print(f"  最大化F1: 阈值={best_f1['thr']:.2f}, F1={best_f1['f1']:.4f}, "
          f"mIoU={best_f1['miou']:.4f}, P={best_f1['precision']:.4f}, R={best_f1['recall']:.4f}")

    # 策略2: Recall>=90%时最大化Precision
    valid_results = [r for r in results if r['recall'] >= 0.90]
    if valid_results:
        best_prec = max(valid_results, key=lambda x: x['precision'])
        print(f"  Recall>=90%最大化Precision: 阈值={best_prec['thr']:.2f}, "
              f"P={best_prec['precision']:.4f}, R={best_prec['recall']:.4f}, "
              f"mIoU={best_prec['miou']:.4f}")
    else:
        best_prec = best_f1

    # 策略3: 最大化mIoU
    best_miou = max(results, key=lambda x: x['miou'])
    print(f"  最大化mIoU: 阈值={best_miou['thr']:.2f}, mIoU={best_miou['miou']:.4f}, "
          f"P={best_miou['precision']:.4f}, R={best_miou['recall']:.4f}")

    print()

    return best_miou['thr'], best_f1['thr'], best_prec['thr']


def visualize_prediction(image, pred_mask, gt_mask, save_path, title=""):
    """可视化预测结果"""
    h, w = image.shape[:2]

    pred_resized = cv2.resize(pred_mask.astype(np.uint8), (w, h),
                             interpolation=cv2.INTER_NEAREST)
    gt_resized = cv2.resize(gt_mask.astype(np.uint8), (w, h),
                           interpolation=cv2.INTER_NEAREST)

    overlay = image.copy()
    overlay[gt_resized == 1] = [0, 255, 0]  # GT green
    overlay[pred_resized == 1] = [0, 0, 255]  # Pred red
    overlay[(gt_resized == 1) & (pred_resized == 1)] = [0, 255, 255]  # Both yellow

    vis_image = image.copy()

    # GT contour (green)
    gt_contour = gt_resized.copy()
    contours, _ = cv2.findContours(gt_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis_image, contours, -1, (0, 255, 0), 2)

    # Prediction contour (red)
    pred_contour = pred_resized.copy()
    contours, _ = cv2.findContours(pred_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis_image, contours, -1, (0, 0, 255), 2)

    # Legend
    cv2.rectangle(vis_image, (10, 10), (250, 110), (255, 255, 255), -1)
    cv2.putText(vis_image, title, (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(vis_image, "Green: Ground Truth", (15, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
    cv2.putText(vis_image, "Red: Prediction", (15, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(vis_image, "Yellow: Both", (15, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    cv2.imwrite(str(save_path), vis_image)


def main():
    print("="*70)
    print("优化推理 - A1-A4 后处理")
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

    checkpoint = torch.load("checkpoints_binary_patch/best_model.pth", map_location=device,
                           weights_only=False)
    model.load_state_dict(checkpoint["model"])
    print(f"  Loaded from epoch {checkpoint['epoch']}")
    print(f"  Best mIoU: {checkpoint['best_miou']:.4f}")

    # Validation data
    val_img_dir = Path("dataset/processed_v2/val/images")
    val_mask_dir = Path("dataset/processed_v2/val/masks")

    val_images = sorted(list(val_img_dir.glob("*.jpg")))
    val_masks = [val_mask_dir / (img.stem + ".png") for img in val_images]

    print(f"\n[2] 验证集: {len(val_images)} 张图像")

    # Initialize inference
    inference = OptimizedSlidingWindowInference(
        model,
        patch_size=384,
        stride=192,
        target_size=256,
        num_classes=2,
        gate_thr=0.70  # A4
    )

    # A1: 阈值扫描（在小样本上快速扫描）
    sample_indices = np.random.choice(len(val_images), size=min(10, len(val_images)), replace=False)
    sample_images = [val_images[i] for i in sample_indices]
    sample_masks = [val_masks[i] for i in sample_indices]

    thr_miou, thr_f1, thr_prec90 = scan_thresholds(
        sample_images, sample_masks, inference, device,
        thr_range=(0.50, 0.99, 0.02)
    )

    print("\n" + "="*70)
    print("选择最佳阈值进行完整推理...")
    print("="*70)

    # 用户选择策略（默认：Recall>=90%最大化Precision）
    best_thr = thr_prec90
    print(f"使用阈值: {best_thr:.2f} (Recall>=90%最大化Precision)\n")

    # 完整推理
    output_dir = Path("results_binary_optimized")
    output_dir.mkdir(exist_ok=True)

    all_miou, all_precision, all_recall, all_defect_iou = [], [], [], []

    print("[3] 完整推理（A1-A4全开）...")
    print("-"*70)

    for img_path, mask_path in tqdm(zip(val_images, val_masks), total=len(val_images)):
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        gt_mask = Image.open(mask_path)
        gt_array = np.array(gt_mask)
        if gt_array.ndim == 3:
            gt_array = gt_array[:, :, 0]

        gt_binary = np.zeros_like(gt_array)
        gt_binary[(gt_array == 3) | (gt_array == 4) | (gt_array == 5)] = 1

        # 推理（带门控）
        prob_map = inference.predict(image, use_gating=True)
        defect_prob = prob_map[:, :, 1]

        # A1: 阈值
        pred_thr = (defect_prob >= best_thr).astype(np.uint8)

        # A2: Hysteresis
        pred_hyst = apply_hysteresis(defect_prob, thr_high=0.90, thr_low=0.70)

        # A3: 连通域过滤
        pred_final = apply_morphological_and_filtering(
            pred_hyst, defect_prob,
            min_area=50,
            mean_prob_thr=0.85
        )

        # Resize
        pred_resized = cv2.resize(pred_final, (image.shape[1], image.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)

        # 指标
        miou, prec_dict, rec_dict, _ = compute_metrics(
            pred_resized[np.newaxis, ...],
            gt_binary[np.newaxis, ...],
            2
        )

        all_miou.append(miou)
        all_precision.append(prec_dict.get(1, 0.0))
        all_recall.append(rec_dict.get(1, 0.0))

        # 可视化
        vis_path = output_dir / f"{img_path.stem}_optimized.jpg"
        visualize_prediction(image, pred_resized, gt_binary, vis_path,
                            title=f"Thr={best_thr:.2f} + Hysteresis + Filter")

    # 最终指标
    avg_miou = np.mean(all_miou)
    avg_precision = np.mean(all_precision)
    avg_recall = np.mean(all_recall)

    print("\n" + "="*70)
    print("优化推理完成!")
    print("="*70)
    print(f"平均 mIoU: {avg_miou:.4f}")
    print(f"平均 Precision: {avg_precision:.4f}")
    print(f"平均 Recall: {avg_recall:.4f}")
    print(f"\n对比之前:")
    print(f"  mIoU: 0.0767 -> {avg_miou:.4f} ({(avg_miou/0.0767-1)*100:+.1f}%)")
    print(f"  Precision: 0.0835 -> {avg_precision:.4f} ({(avg_precision/0.0835-1)*100:+.1f}%)")
    print(f"  Recall: 0.9412 -> {avg_recall:.4f}")
    print(f"\n可视化结果: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
