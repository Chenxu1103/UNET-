"""
Refactored cable wrapping detection system.
"""

from .config import (
    ROIConfig,
    PreprocessConfig,
    PostprocessConfig,
    EventConfig,
    BurrConfig,
    RefactorConfig
)
from .preprocess import (
    is_grayscale_frame,
    enhance_grayscale_frame,
    preprocess_frame,
    crop_roi,
    paste_roi_mask
)
from .postprocess import (
    filter_cable_by_shape,
    constrain_tape_to_ring,
    postprocess_masks
)
from .burr_detector import get_burr_mask_rulebased
from .event_detector import FrameMetrics, EventDetector

__all__ = [
    'ROIConfig',
    'PreprocessConfig',
    'PostprocessConfig',
    'EventConfig',
    'BurrConfig',
    'RefactorConfig',
    'is_grayscale_frame',
    'enhance_grayscale_frame',
    'preprocess_frame',
    'crop_roi',
    'paste_roi_mask',
    'filter_cable_by_shape',
    'constrain_tape_to_ring',
    'postprocess_masks',
    'get_burr_mask_rulebased',
    'FrameMetrics',
    'EventDetector',
]
