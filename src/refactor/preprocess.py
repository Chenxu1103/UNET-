"""
Preprocessing module for refactored cable wrapping detection system.
Handles grayscale enhancement, ROI cropping, and frame preprocessing.
"""

import cv2
import numpy as np
from typing import Tuple
from .config import PreprocessConfig, ROIConfig


def is_grayscale_frame(frame: np.ndarray, threshold: float = 10.0) -> bool:
    """
    Detect if frame is grayscale (RGB channels have minimal difference).

    Args:
        frame: Input frame (H, W, 3)
        threshold: Maximum mean difference between channels to consider grayscale

    Returns:
        True if frame is grayscale, False otherwise
    """
    if len(frame.shape) != 3 or frame.shape[2] != 3:
        return True  # Already grayscale

    b, g, r = cv2.split(frame)
    diff_bg = np.abs(b.astype(float) - g.astype(float)).mean()
    diff_gr = np.abs(g.astype(float) - r.astype(float)).mean()
    diff_rb = np.abs(r.astype(float) - b.astype(float)).mean()
    max_diff = max(diff_bg, diff_gr, diff_rb)

    return max_diff < threshold


def enhance_grayscale_frame(frame: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """
    Enhance grayscale frame with CLAHE, gamma correction, and denoising.

    Args:
        frame: Input frame (H, W, 3) or (H, W)
        cfg: Preprocessing configuration

    Returns:
        Enhanced frame (H, W, 3)
    """
    # Convert to grayscale if needed
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame.copy()

    # CLAHE enhancement
    clahe = cv2.createCLAHE(
        clipLimit=cfg.clahe_clip_limit,
        tileGridSize=(cfg.clahe_tile_size, cfg.clahe_tile_size)
    )
    enhanced = clahe.apply(gray)

    # Gamma correction
    if cfg.gamma != 1.0:
        inv_gamma = 1.0 / cfg.gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
        enhanced = cv2.LUT(enhanced, table)

    # Denoising
    if cfg.denoise_method == 'bilateral':
        enhanced = cv2.bilateralFilter(enhanced, cfg.denoise_strength, 75, 75)
    elif cfg.denoise_method == 'fastNlMeans':
        enhanced = cv2.fastNlMeansDenoising(enhanced, None, cfg.denoise_strength, 7, 21)

    # Convert back to 3-channel
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    return enhanced_bgr


def preprocess_frame(frame: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """
    Unified preprocessing entry point.

    Args:
        frame: Input frame (H, W, 3)
        cfg: Preprocessing configuration

    Returns:
        Preprocessed frame (H, W, 3)
    """
    if cfg.enable_grayscale_enhance and is_grayscale_frame(frame):
        return enhance_grayscale_frame(frame, cfg)
    else:
        return frame.copy()


def crop_roi(frame: np.ndarray, roi: ROIConfig) -> np.ndarray:
    """
    Crop ROI region from frame.

    Args:
        frame: Input frame (H, W, C)
        roi: ROI configuration

    Returns:
        Cropped ROI frame
    """
    h, w = frame.shape[:2]

    # Ensure ROI is within frame bounds
    x1 = max(0, roi.x)
    y1 = max(0, roi.y)
    x2 = min(w, roi.x + roi.w)
    y2 = min(h, roi.y + roi.h)

    return frame[y1:y2, x1:x2].copy()


def paste_roi_mask(full_mask: np.ndarray, roi_mask: np.ndarray, roi: ROIConfig) -> np.ndarray:
    """
    Paste ROI mask back to full frame mask.

    Args:
        full_mask: Full frame mask (H, W), will be modified in-place
        roi_mask: ROI mask to paste
        roi: ROI configuration

    Returns:
        Full mask with ROI pasted
    """
    h, w = full_mask.shape[:2]
    roi_h, roi_w = roi_mask.shape[:2]

    # Ensure ROI is within frame bounds
    x1 = max(0, roi.x)
    y1 = max(0, roi.y)
    x2 = min(w, roi.x + roi.w)
    y2 = min(h, roi.y + roi.h)

    # Calculate actual paste region
    paste_h = min(roi_h, y2 - y1)
    paste_w = min(roi_w, x2 - x1)

    # Paste
    full_mask[y1:y1+paste_h, x1:x1+paste_w] = roi_mask[:paste_h, :paste_w]

    return full_mask


def resize_for_model(frame: np.ndarray, target_size: int = 512) -> Tuple[np.ndarray, float]:
    """
    Resize frame for model input while maintaining aspect ratio.

    Args:
        frame: Input frame (H, W, C)
        target_size: Target size (will be used for both dimensions)

    Returns:
        Tuple of (resized_frame, scale_factor)
    """
    h, w = frame.shape[:2]
    scale = target_size / max(h, w)

    new_h = int(h * scale)
    new_w = int(w * scale)

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pad to square
    if new_h != target_size or new_w != target_size:
        padded = np.zeros((target_size, target_size, frame.shape[2]), dtype=frame.dtype)
        padded[:new_h, :new_w] = resized
        resized = padded

    return resized, scale


def resize_mask_back(mask: np.ndarray, original_shape: Tuple[int, int], scale: float) -> np.ndarray:
    """
    Resize mask back to original shape.

    Args:
        mask: Model output mask (target_size, target_size)
        original_shape: Original (H, W)
        scale: Scale factor from resize_for_model

    Returns:
        Resized mask (H, W)
    """
    h, w = original_shape
    scaled_h = int(h * scale)
    scaled_w = int(w * scale)

    # Crop padding
    mask_cropped = mask[:scaled_h, :scaled_w]

    # Resize back
    mask_resized = cv2.resize(mask_cropped, (w, h), interpolation=cv2.INTER_NEAREST)

    return mask_resized
