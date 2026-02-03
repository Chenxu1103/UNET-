from __future__ import annotations
from typing import Dict
import numpy as np
import cv2


def _width_per_row(mask: np.ndarray) -> np.ndarray:
    """
    mask: (H,W) binary
    return widths: (H,) width in pixels for each row; 0 if absent
    """
    H, W = mask.shape
    widths = np.zeros((H,), dtype=np.float32)
    for y in range(H):
        xs = np.where(mask[y] > 0)[0]
        if xs.size > 0:
            widths[y] = float(xs.max() - xs.min() + 1)
    return widths


def smooth_1d(x: np.ndarray, k: int = 21) -> np.ndarray:
    if k <= 1:
        return x
    k = int(k) if int(k) % 2 == 1 else int(k) + 1
    return cv2.GaussianBlur(x.reshape(-1, 1), (1, k), 0).reshape(-1)


def diameter_profile_from_masks(
    pred: np.ndarray,
    cable_cls: int,
    wrap_cls: int,
) -> Dict[str, np.ndarray]:
    """
    pred: (H,W) int labels at model input resolution
    """
    cable = (pred == cable_cls).astype(np.uint8)
    wrap = (pred == wrap_cls).astype(np.uint8)

    # keep largest connected component to reduce noise
    def largest_cc(bin_mask: np.ndarray) -> np.ndarray:
        n, lab, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
        if n <= 1:
            return bin_mask
        areas = stats[1:, cv2.CC_STAT_AREA]
        i = 1 + int(np.argmax(areas))
        return (lab == i).astype(np.uint8)

    cable = largest_cc(cable)
    wrap = largest_cc(wrap)

    w_cable = _width_per_row(cable)
    w_wrap = _width_per_row(wrap)

    w_cable_s = smooth_1d(w_cable, 31)
    w_wrap_s = smooth_1d(w_wrap, 31)

    # valid rows where both exist
    valid = (w_cable_s > 0) & (w_wrap_s > 0)

    return {
        "w_cable_px": w_cable_s,
        "w_wrap_px": w_wrap_s,
        "valid": valid.astype(np.uint8),
    }
