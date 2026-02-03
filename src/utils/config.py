from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class ROI:
    enabled: bool
    x: int
    y: int
    w: int
    h: int


@dataclass
class CameraCfg:
    type: str
    cti_path: str
    serial: str
    width: int
    height: int
    pixel_format: str
    exposure_us: int
    gain_db: float
    roi: ROI
    fps_limit: float


@dataclass
class ModelCfg:
    input_size: Tuple[int, int]
    num_classes: int
    encoder: str
    weights: str


@dataclass
class ScaleCfg:
    mm_per_px: Optional[float]
    cable_diameter_mm: float


@dataclass
class ThresholdCfg:
    wrap_delta_max_mm: float
    wrap_delta_min_mm: float
    bulge_mm: float
    cv_wrap: float
    defect_area_px: int


@dataclass
class MqttCfg:
    enabled: bool
    host: str
    port: int
    topic: str


@dataclass
class EventCfg:
    out_dir: str
    save_overlay: bool
    mqtt: MqttCfg


@dataclass
class AppCfg:
    camera: CameraCfg
    model: ModelCfg
    scale: ScaleCfg
    thresholds: ThresholdCfg
    event: EventCfg
    device_use_gpu: bool
    device_fp16: bool


def parse_cfg(d: Dict[str, Any]) -> AppCfg:
    roi = ROI(**d["camera"]["roi"])
    cam = CameraCfg(roi=roi, **{k: v for k, v in d["camera"].items() if k != "roi"})
    model = ModelCfg(
        input_size=tuple(d["model"]["input_size"]),
        num_classes=int(d["model"]["num_classes"]),
        encoder=str(d["model"]["encoder"]),
        weights=str(d["model"]["weights"]),
    )
    scale = ScaleCfg(**d["scale"])
    thr = ThresholdCfg(**d["thresholds"])
    mqtt = MqttCfg(**d["event"]["mqtt"])
    ev = EventCfg(out_dir=d["event"]["out_dir"], save_overlay=bool(d["event"]["save_overlay"]), mqtt=mqtt)

    return AppCfg(
        camera=cam,
        model=model,
        scale=scale,
        thresholds=thr,
        event=ev,
        device_use_gpu=bool(d["device"]["use_gpu"]),
        device_fp16=bool(d["device"]["fp16"]),
    )
