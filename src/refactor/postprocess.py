"""
Postprocessing module for refactored cable wrapping detection system.
Handles shape constraints and filtering for segmentation results.
"""

import cv2
import numpy as np
from typing import Tuple
from .config import PostprocessConfig


def filter_cable_by_shape(mask_cable: np.ndarray, cfg: PostprocessConfig, roi_width: int) -> np.ndarray:
    """
    Filter cable mask by shape constraints.

    Args:
        mask_cable: Binary cable mask (H, W)
        cfg: Postprocessing configuration
        roi_width: ROI width for center offset calculation

    Returns:
        Filtered cable mask
    """
    if mask_cable.max() == 0:
        return mask_cable

    # Connected component analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_cable.astype(np.uint8), connectivity=8
    )

    if num_labels <= 1:  # Only background
        return np.zeros_like(mask_cable)

    # Calculate ROI center x
    roi_center_x = roi_width / 2.0

    best_score = -1
    best_label = -1

    for label in range(1, num_labels):  # Skip background (0)
        area = stats[label, cv2.CC_STAT_AREA]
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]

        # Filter by area
        if area < cfg.cable_min_area:
            continue

        # Filter by aspect ratio
        aspect = max(w, h) / (min(w, h) + 1e-6)
        if aspect < cfg.cable_min_aspect:
            continue

        # Filter by center offset
        cx = centroids[label][0]
        center_offset = abs(cx - roi_center_x) / roi_width
        if center_offset > cfg.cable_max_center_offset:
            continue

        # Calculate score (higher is better)
        score = area * aspect * (1.0 - center_offset)

        if score > best_score:
            best_score = score
            best_label = label

    # Create filtered mask
    if best_label > 0:
        filtered_mask = (labels == best_label).astype(np.uint8) * 255
    else:
        filtered_mask = np.zeros_like(mask_cable)

    return filtered_mask


def constrain_tape_to_ring(mask_tape: np.ndarray, mask_cable: np.ndarray, cfg: PostprocessConfig) -> np.ndarray:
    """
    Constrain tape mask to ring region around cable.

    Args:
        mask_tape: Binary tape mask (H, W)
        mask_cable: Binary cable mask (H, W)
        cfg: Postprocessing configuration

    Returns:
        Constrained tape mask
    """
    if mask_cable.max() == 0 or mask_tape.max() == 0:
        return np.zeros_like(mask_tape)

    # Create ring region
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.tape_ring_dilate, cfg.tape_ring_dilate))
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.tape_ring_erode, cfg.tape_ring_erode))

    dilated = cv2.dilate(mask_cable, kernel_dilate, iterations=1)
    eroded = cv2.erode(mask_cable, kernel_erode, iterations=1)

    ring = cv2.subtract(dilated, eroded)

    # Constrain tape to ring
    constrained = cv2.bitwise_and(mask_tape, ring)

    # Keep only largest connected component
    if constrained.max() > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            constrained.astype(np.uint8), connectivity=8
        )

        if num_labels > 1:
            # Find largest component (excluding background)
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest_label = np.argmax(areas) + 1
            constrained = (labels == largest_label).astype(np.uint8) * 255

    return constrained


def postprocess_masks(mask_cable: np.ndarray, mask_tape: np.ndarray,
                      cfg: PostprocessConfig, roi_width: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Unified postprocessing entry point.

    Args:
        mask_cable: Binary cable mask (H, W)
        mask_tape: Binary tape mask (H, W)
        cfg: Postprocessing configuration
        roi_width: ROI width

    Returns:
        Tuple of (filtered_cable_mask, constrained_tape_mask)
    """
    # Filter cable by shape
    filtered_cable = filter_cable_by_shape(mask_cable, cfg, roi_width)

    # Constrain tape to ring
    constrained_tape = constrain_tape_to_ring(mask_tape, filtered_cable, cfg)

    return filtered_cable, constrained_tape


def apply_morphology_cleanup(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Apply morphological operations to clean up mask.

    Args:
        mask: Binary mask (H, W)
        kernel_size: Kernel size for morphological operations

    Returns:
        Cleaned mask
    """
    if mask.max() == 0:
        return mask

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Opening to remove small noise
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Closing to fill small holes
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    return closed
