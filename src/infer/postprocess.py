from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np

from src.utils.geometry import diameter_profile_from_masks


@dataclass
class Metrics:
    mm_per_px: float
    cable_diam_mm_med: float
    wrap_diam_mm_med: float
    delta_mm_max: float
    delta_mm_min: float
    bulge_mm: float
    cv_wrap: float
    defect_area_px: int


def compute_metrics(
    pred: np.ndarray,
    cable_cls: int,
    wrap_cls: int,
    defect_cls: Optional[int],
    mm_per_px: Optional[float],
    cable_diameter_mm_known: float,
) -> Metrics:
    prof = diameter_profile_from_masks(pred, cable_cls=cable_cls, wrap_cls=wrap_cls)
    w_cable = prof["w_cable_px"]
    w_wrap = prof["w_wrap_px"]
    valid = prof["valid"].astype(bool)

    if valid.sum() < 20:
        # not enough signal; fall back
        mmpp = mm_per_px if (mm_per_px is not None) else 0.1
        defect_area = int((pred == defect_cls).sum()) if defect_cls is not None else 0
        return Metrics(mmpp, 0, 0, 0, 0, 0, 0, defect_area)

    cable_px_med = float(np.median(w_cable[valid]))
    wrap_px_med = float(np.median(w_wrap[valid]))

    if mm_per_px is None:
        # derive from known cable diameter
        mmpp = float(cable_diameter_mm_known / max(cable_px_med, 1e-6))
    else:
        mmpp = float(mm_per_px)

    cable_mm = w_cable * mmpp
    wrap_mm = w_wrap * mmpp
    delta = wrap_mm - cable_mm

    delta_valid = delta[valid]
    wrap_valid = wrap_mm[valid]

    delta_max = float(np.max(delta_valid))
    delta_min = float(np.min(delta_valid))

    bulge = float(np.max(wrap_valid) - np.median(wrap_valid))
    cv = float(np.std(wrap_valid) / max(np.mean(wrap_valid), 1e-6))

    defect_area = int((pred == defect_cls).sum()) if defect_cls is not None else 0

    return Metrics(
        mm_per_px=mmpp,
        cable_diam_mm_med=float(cable_px_med * mmpp),
        wrap_diam_mm_med=float(wrap_px_med * mmpp),
        delta_mm_max=delta_max,
        delta_mm_min=delta_min,
        bulge_mm=bulge,
        cv_wrap=cv,
        defect_area_px=defect_area,
    )
