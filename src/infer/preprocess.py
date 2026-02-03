from __future__ import annotations
from typing import Tuple
import cv2
import numpy as np


def resize_and_normalize(bgr: np.ndarray, input_hw: Tuple[int, int]) -> np.ndarray:
    h, w = input_hw
    img = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    # ImageNet normalize
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    chw = np.transpose(img, (2, 0, 1))[None, ...].astype(np.float32)
    return chw
