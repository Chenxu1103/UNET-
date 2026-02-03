from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import cv2


def colorize_mask(mask: np.ndarray, palette: Dict[int, Tuple[int, int, int]]) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for k, bgr in palette.items():
        out[mask == k] = bgr
    return out


def overlay(bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    palette = {
        0: (0, 0, 0),
        1: (255, 255, 255),   # cable
        2: (0, 255, 255),     # wrap
        3: (0, 0, 255),       # defect
    }
    cm = colorize_mask(mask, palette)
    cm = cv2.resize(cm, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    return cv2.addWeighted(bgr, 1 - alpha, cm, alpha, 0)
