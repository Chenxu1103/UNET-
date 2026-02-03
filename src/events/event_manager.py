from __future__ import annotations
from dataclasses import asdict
from typing import Any, Dict, List, Optional
import os
import json
import time
import cv2

try:
    import paho.mqtt.client as mqtt
except Exception:
    mqtt = None


class EventManager:
    def __init__(self, out_dir: str, mqtt_cfg: Dict[str, Any]) -> None:
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        self.mqtt_enabled = bool(mqtt_cfg.get("enabled", False)) and mqtt is not None
        self.mqtt_topic = mqtt_cfg.get("topic", "cable/inspection/event")
        self.client = None
        if self.mqtt_enabled:
            self.client = mqtt.Client()
            self.client.connect(mqtt_cfg["host"], int(mqtt_cfg["port"]), keepalive=30)

    def emit(
        self,
        camera_id: str,
        frame_bgr,
        overlay_bgr,
        findings: List[dict],
        metrics: dict,
        timestamp_ns: int,
    ) -> Dict[str, Any]:
        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        base = f"{ts}_{timestamp_ns}"

        img_path = os.path.join(self.out_dir, f"{base}.jpg")
        ovl_path = os.path.join(self.out_dir, f"{base}_overlay.jpg")
        json_path = os.path.join(self.out_dir, f"{base}.json")

        cv2.imwrite(img_path, frame_bgr)
        if overlay_bgr is not None:
            cv2.imwrite(ovl_path, overlay_bgr)

        payload = {
            "camera_id": camera_id,
            "timestamp_ns": int(timestamp_ns),
            "findings": findings,
            "metrics": metrics,
            "image": os.path.abspath(img_path),
            "overlay": os.path.abspath(ovl_path) if overlay_bgr is not None else "",
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        if self.mqtt_enabled and self.client is not None:
            self.client.publish(self.mqtt_topic, json.dumps(payload, ensure_ascii=False))

        return payload
