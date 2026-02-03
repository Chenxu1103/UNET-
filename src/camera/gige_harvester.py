from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import cv2
from harvesters.core import Harvester


@dataclass
class Frame:
    image_bgr: np.ndarray
    timestamp_ns: int


class GigECameraHarvester:
    """
    GenICam/GigE Vision camera acquisition via Harvester.
    Requires a valid GenTL producer (.cti).
    """

    def __init__(
        self,
        cti_path: str,
        serial: str = "",
        pixel_format: str = "BayerRG8",
        exposure_us: int = 200,
        gain_db: float = 0.0,
        roi: Optional[Tuple[int, int, int, int]] = None,  # x,y,w,h
    ) -> None:
        self.cti_path = cti_path
        self.serial = serial
        self.pixel_format = pixel_format
        self.exposure_us = exposure_us
        self.gain_db = gain_db
        self.roi = roi

        self.h = Harvester()
        self.ia = None

    def open(self) -> None:
        self.h.add_file(self.cti_path)
        self.h.update()

        if len(self.h.device_info_list) == 0:
            raise RuntimeError(f"No camera found. Check CTI: {self.cti_path}")

        index = 0
        if self.serial:
            for i, info in enumerate(self.h.device_info_list):
                if self.serial in str(info):
                    index = i
                    break

        self.ia = self.h.create_image_acquirer(list_index=index)

        # Configure camera features (best-effort; feature names vary by vendor)
        node_map = self.ia.remote_device.node_map

        def _set_if_exists(name: str, value):
            if hasattr(node_map, name):
                try:
                    getattr(node_map, name).value = value
                except Exception:
                    pass

        _set_if_exists("ExposureTime", float(self.exposure_us))         # often in us
        _set_if_exists("Gain", float(self.gain_db))
        _set_if_exists("PixelFormat", self.pixel_format)

        if self.roi:
            x, y, w, h = self.roi
            _set_if_exists("OffsetX", int(x))
            _set_if_exists("OffsetY", int(y))
            _set_if_exists("Width", int(w))
            _set_if_exists("Height", int(h))

    def start(self) -> None:
        if self.ia is None:
            raise RuntimeError("Camera not opened")
        self.ia.start_acquisition()

    def stop(self) -> None:
        if self.ia:
            try:
                self.ia.stop_acquisition()
            except Exception:
                pass

    def close(self) -> None:
        self.stop()
        if self.ia:
            try:
                self.ia.destroy()
            except Exception:
                pass
        try:
            self.h.reset()
        except Exception:
            pass

    @staticmethod
    def _to_bgr(buffer: np.ndarray, pixel_format: str, width: int, height: int) -> np.ndarray:
        # buffer is typically 1D uint8 array; reshape to HxW
        img = buffer.reshape(height, width)
        if "BayerRG" in pixel_format:
            bgr = cv2.cvtColor(img, cv2.COLOR_BAYER_RG2BGR)
        elif "BayerBG" in pixel_format:
            bgr = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2BGR)
        elif "Mono" in pixel_format:
            bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            # fallback: treat as gray
            bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return bgr

    def read(self, timeout_ms: int = 1000) -> Frame:
        if self.ia is None:
            raise RuntimeError("Camera not opened")

        with self.ia.fetch_buffer(timeout=timeout_ms) as buffer:
            payload = buffer.payload
            comp = payload.components[0]
            w, h = comp.width, comp.height
            data = comp.data  # numpy view
            img_bgr = self._to_bgr(data, self.pixel_format, w, h)

            # timestamp (best-effort)
            ts = getattr(buffer, "timestamp_ns", 0)
            return Frame(image_bgr=img_bgr, timestamp_ns=int(ts))
