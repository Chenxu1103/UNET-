"""
Configuration module for refactored cable wrapping detection system.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import yaml
import json


@dataclass
class ROIConfig:
    """ROI configuration."""
    mode: str = 'fixed'  # 'fixed' or 'calibrate'
    x: int = 0
    y: int = 0
    w: int = 640
    h: int = 480

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ROIConfig':
        return cls(**data)

    @classmethod
    def from_json(cls, json_path: str) -> 'ROIConfig':
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'mode': self.mode,
            'x': self.x,
            'y': self.y,
            'w': self.w,
            'h': self.h
        }

    def to_json(self, json_path: str):
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class PreprocessConfig:
    """Preprocessing configuration."""
    enable_grayscale_enhance: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_size: int = 8
    gamma: float = 0.8
    denoise_method: str = 'bilateral'  # 'bilateral' or 'fastNlMeans'
    denoise_strength: int = 5

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PreprocessConfig':
        return cls(**data)


@dataclass
class PostprocessConfig:
    """Postprocessing configuration."""
    # Cable constraints
    cable_min_area: int = 1000
    cable_min_aspect: float = 1.6
    cable_max_center_offset: float = 0.3  # relative to ROI width

    # Tape constraints
    tape_min_area: int = 500
    tape_ring_dilate: int = 15
    tape_ring_erode: int = 5

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PostprocessConfig':
        return cls(**data)


@dataclass
class EventConfig:
    """Event detection configuration."""
    # Uniformity thresholds
    ratio_min: float = 1.1
    ratio_max: float = 1.4
    uniformity_window: int = 30  # frames
    uniformity_std_threshold: float = 0.05

    # Event triggering
    thin_wrap_frames: int = 5
    thick_wrap_frames: int = 5
    burr_frames: int = 3
    cooldown_frames: int = 30

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EventConfig':
        return cls(**data)


@dataclass
class BurrConfig:
    """Burr detection configuration."""
    band_out: int = 10  # outer expansion pixels
    laplacian_threshold: int = 30
    min_area: int = 20
    max_area: int = 500

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BurrConfig':
        return cls(**data)


@dataclass
class RefactorConfig:
    """Complete refactored system configuration."""
    roi: ROIConfig = field(default_factory=ROIConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)
    event: EventConfig = field(default_factory=EventConfig)
    burr: BurrConfig = field(default_factory=BurrConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'RefactorConfig':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        return cls(
            roi=ROIConfig.from_dict(data.get('roi', {})),
            preprocess=PreprocessConfig.from_dict(data.get('preprocess', {})),
            postprocess=PostprocessConfig.from_dict(data.get('postprocess', {})),
            event=EventConfig.from_dict(data.get('event', {})),
            burr=BurrConfig.from_dict(data.get('burr', {}))
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RefactorConfig':
        """Load configuration from dictionary."""
        return cls(
            roi=ROIConfig.from_dict(data.get('roi', {})),
            preprocess=PreprocessConfig.from_dict(data.get('preprocess', {})),
            postprocess=PostprocessConfig.from_dict(data.get('postprocess', {})),
            event=EventConfig.from_dict(data.get('event', {})),
            burr=BurrConfig.from_dict(data.get('burr', {}))
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'roi': self.roi.to_dict(),
            'preprocess': self.preprocess.__dict__,
            'postprocess': self.postprocess.__dict__,
            'event': self.event.__dict__,
            'burr': self.burr.__dict__
        }

    def to_yaml(self, yaml_path: str):
        """Save configuration to YAML file."""
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)
