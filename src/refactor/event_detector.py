"""
Event detection module for refactored cable wrapping detection system.
Handles uniformity calculation and event triggering with cooldown mechanism.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict
import numpy as np


@dataclass
class FrameMetrics:
    """Metrics for a single frame."""
    frame_id: int
    dc_px: float  # Cable diameter in pixels
    dt_px: float  # Tape diameter in pixels
    delta_d_px: float  # Difference (dt - dc)
    ratio: Optional[float]  # Tape/cable ratio
    has_burr: bool
    cable_coverage: float  # Cable mask coverage ratio
    tape_coverage: float  # Tape mask coverage ratio


class EventDetector:
    """
    Event detector with cooldown mechanism.

    Detects:
    - thin_wrap: ratio < ratio_min for consecutive frames
    - thick_wrap: ratio > ratio_max for consecutive frames
    - burr: has_burr for consecutive frames
    """

    def __init__(self, cfg):
        """
        Initialize event detector.

        Args:
            cfg: EventConfig instance
        """
        self.cfg = cfg
        self.history: List[FrameMetrics] = []
        self.last_event_frame: Dict[str, int] = {}  # event_type -> frame_id

        # Consecutive frame counters
        self.thin_wrap_count = 0
        self.thick_wrap_count = 0
        self.burr_count = 0

    def add_frame(self, metrics: FrameMetrics) -> List[str]:
        """
        Add frame metrics and detect events.

        Args:
            metrics: Frame metrics

        Returns:
            List of triggered event types
        """
        self.history.append(metrics)

        # Trim history to window size
        if len(self.history) > self.cfg.uniformity_window:
            self.history.pop(0)

        triggered_events = []

        # Check thin wrap
        if metrics.ratio is not None and metrics.ratio < self.cfg.ratio_min:
            self.thin_wrap_count += 1
            if self.thin_wrap_count >= self.cfg.thin_wrap_frames:
                if self._can_trigger('thin_wrap', metrics.frame_id):
                    triggered_events.append('thin_wrap')
                    self.last_event_frame['thin_wrap'] = metrics.frame_id
                    self.thin_wrap_count = 0  # Reset counter
        else:
            self.thin_wrap_count = 0

        # Check thick wrap
        if metrics.ratio is not None and metrics.ratio > self.cfg.ratio_max:
            self.thick_wrap_count += 1
            if self.thick_wrap_count >= self.cfg.thick_wrap_frames:
                if self._can_trigger('thick_wrap', metrics.frame_id):
                    triggered_events.append('thick_wrap')
                    self.last_event_frame['thick_wrap'] = metrics.frame_id
                    self.thick_wrap_count = 0
        else:
            self.thick_wrap_count = 0

        # Check burr
        if metrics.has_burr:
            self.burr_count += 1
            if self.burr_count >= self.cfg.burr_frames:
                if self._can_trigger('burr', metrics.frame_id):
                    triggered_events.append('burr')
                    self.last_event_frame['burr'] = metrics.frame_id
                    self.burr_count = 0
        else:
            self.burr_count = 0

        return triggered_events

    def _can_trigger(self, event_type: str, current_frame: int) -> bool:
        """
        Check if event can be triggered (cooldown check).

        Args:
            event_type: Event type
            current_frame: Current frame ID

        Returns:
            True if event can be triggered, False otherwise
        """
        if event_type not in self.last_event_frame:
            return True

        last_frame = self.last_event_frame[event_type]
        frames_since_last = current_frame - last_frame

        return frames_since_last >= self.cfg.cooldown_frames

    def compute_uniformity(self) -> Optional[float]:
        """
        Compute uniformity (rolling standard deviation of ratio).

        Returns:
            Uniformity value (std of ratio), or None if insufficient data
        """
        if len(self.history) < 2:
            return None

        ratios = [m.ratio for m in self.history if m.ratio is not None]

        if len(ratios) < 2:
            return None

        return float(np.std(ratios))

    def get_recent_metrics(self, n: int = 10) -> List[FrameMetrics]:
        """
        Get recent N frame metrics.

        Args:
            n: Number of recent frames

        Returns:
            List of recent frame metrics
        """
        return self.history[-n:]

    def get_average_ratio(self, n: int = 10) -> Optional[float]:
        """
        Get average ratio over recent N frames.

        Args:
            n: Number of recent frames

        Returns:
            Average ratio, or None if insufficient data
        """
        recent = self.get_recent_metrics(n)
        ratios = [m.ratio for m in recent if m.ratio is not None]

        if len(ratios) == 0:
            return None

        return float(np.mean(ratios))

    def reset(self):
        """Reset detector state."""
        self.history.clear()
        self.last_event_frame.clear()
        self.thin_wrap_count = 0
        self.thick_wrap_count = 0
        self.burr_count = 0
