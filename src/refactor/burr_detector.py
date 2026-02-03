"""
Burr detection module for refactored cable wrapping detection system.
Rule-based burr detection using edge analysis.
"""

import cv2
import numpy as np
from .config import BurrConfig


def get_burr_mask_rulebased(frame_gray: np.ndarray, mask_cable: np.ndarray, cfg: BurrConfig) -> np.ndarray:
    """
    Rule-based burr detection using edge analysis.

    Args:
        frame_gray: Grayscale frame (H, W)
        mask_cable: Binary cable mask (H, W)
        cfg: Burr detection configuration

    Returns:
        Binary burr mask (H, W)
    """
    if mask_cable.max() == 0:
        return np.zeros_like(mask_cable)

    # Ensure grayscale
    if len(frame_gray.shape) == 3:
        frame_gray = cv2.cvtColor(frame_gray, cv2.COLOR_BGR2GRAY)

    # Get cable boundary
    contours, _ = cv2.findContours(mask_cable.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return np.zeros_like(mask_cable)

    # Create band region (outer expansion)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.band_out * 2 + 1, cfg.band_out * 2 + 1))
    dilated = cv2.dilate(mask_cable, kernel, iterations=1)

    # Band = dilated - original
    band = cv2.subtract(dilated, mask_cable)

    # Extract high-frequency response using Laplacian
    laplacian = cv2.Laplacian(frame_gray, cv2.CV_64F)
    laplacian_abs = np.abs(laplacian).astype(np.uint8)

    # Apply band mask
    laplacian_band = cv2.bitwise_and(laplacian_abs, laplacian_abs, mask=band)

    # Threshold
    _, burr_mask = cv2.threshold(laplacian_band, cfg.laplacian_threshold, 255, cv2.THRESH_BINARY)

    # Filter by connected component size
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        burr_mask.astype(np.uint8), connectivity=8
    )

    filtered_mask = np.zeros_like(burr_mask)

    for label in range(1, num_labels):  # Skip background
        area = stats[label, cv2.CC_STAT_AREA]

        if cfg.min_area <= area <= cfg.max_area:
            filtered_mask[labels == label] = 255

    return filtered_mask


def get_burr_mask_dog(frame_gray: np.ndarray, mask_cable: np.ndarray, cfg: BurrConfig) -> np.ndarray:
    """
    Alternative burr detection using Difference of Gaussians (DoG).

    Args:
        frame_gray: Grayscale frame (H, W)
        mask_cable: Binary cable mask (H, W)
        cfg: Burr detection configuration

    Returns:
        Binary burr mask (H, W)
    """
    if mask_cable.max() == 0:
        return np.zeros_like(mask_cable)

    # Ensure grayscale
    if len(frame_gray.shape) == 3:
        frame_gray = cv2.cvtColor(frame_gray, cv2.COLOR_BGR2GRAY)

    # Create band region
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.band_out * 2 + 1, cfg.band_out * 2 + 1))
    dilated = cv2.dilate(mask_cable, kernel, iterations=1)
    band = cv2.subtract(dilated, mask_cable)

    # DoG for edge detection
    blur1 = cv2.GaussianBlur(frame_gray, (3, 3), 1.0)
    blur2 = cv2.GaussianBlur(frame_gray, (7, 7), 2.0)
    dog = cv2.subtract(blur1, blur2)
    dog_abs = np.abs(dog)

    # Apply band mask
    dog_band = cv2.bitwise_and(dog_abs, dog_abs, mask=band)

    # Threshold
    _, burr_mask = cv2.threshold(dog_band, cfg.laplacian_threshold, 255, cv2.THRESH_BINARY)

    # Filter by size
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        burr_mask.astype(np.uint8), connectivity=8
    )

    filtered_mask = np.zeros_like(burr_mask)

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]

        if cfg.min_area <= area <= cfg.max_area:
            filtered_mask[labels == label] = 255

    return filtered_mask


def has_burr(burr_mask: np.ndarray, min_total_area: int = 50) -> bool:
    """
    Check if burr mask contains significant burr regions.

    Args:
        burr_mask: Binary burr mask (H, W)
        min_total_area: Minimum total area to consider as burr

    Returns:
        True if burr detected, False otherwise
    """
    total_area = np.sum(burr_mask > 0)
    return total_area >= min_total_area
