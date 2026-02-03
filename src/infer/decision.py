from __future__ import annotations
from dataclasses import dataclass
from typing import List
from src.infer.postprocess import Metrics


@dataclass
class Finding:
    code: str
    severity: str   # P1/P2
    detail: str


def decide(metrics: Metrics, thr) -> List[Finding]:
    out: List[Finding] = []

    if metrics.delta_mm_max > thr.wrap_delta_max_mm:
        out.append(Finding("wrap_too_large", "P1", f"delta_max={metrics.delta_mm_max:.2f}mm"))
    if metrics.delta_mm_min < thr.wrap_delta_min_mm:
        out.append(Finding("wrap_too_small", "P1", f"delta_min={metrics.delta_mm_min:.2f}mm"))

    if metrics.bulge_mm > thr.bulge_mm:
        out.append(Finding("wrap_bulge", "P2", f"bulge={metrics.bulge_mm:.2f}mm"))

    if metrics.cv_wrap > thr.cv_wrap:
        out.append(Finding("wrap_uneven", "P2", f"cv={metrics.cv_wrap:.3f}"))

    if metrics.defect_area_px > thr.defect_area_px:
        out.append(Finding("cable_damage_or_defect", "P1", f"defect_area={metrics.defect_area_px}px"))

    return out
