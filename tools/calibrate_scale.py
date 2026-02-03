from __future__ import annotations
import os
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class CalibrateArgs:
    image_path: str
    known_size_mm: float
    known_size_px: Optional[float] = None
    interactive: bool = True


def manual_calibrate(image_path: str, known_size_mm: float) -> float:
    """
    Load image, let user mark two points with known distance, compute mm_per_px.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    display = img.copy()
    points = []

    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(display, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("calibrate", display)
            if len(points) == 2:
                p1, p2 = points
                dist_px = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                mmpp = float(known_size_mm / max(dist_px, 1e-6))
                print(f"Distance: {dist_px:.2f} px = {known_size_mm} mm => {mmpp:.6f} mm/px")
                cv2.destroyAllWindows()

    cv2.imshow("calibrate", display)
    cv2.setMouseCallback("calibrate", click)
    print("Click two points on the image with known distance")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(points) == 2:
        p1, p2 = points
        dist_px = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        mmpp = float(known_size_mm / max(dist_px, 1e-6))
        return mmpp
    else:
        raise RuntimeError("Need exactly 2 points")


if __name__ == "__main__":
    # Example usage
    args = CalibrateArgs(
        image_path="calibration_image.jpg",
        known_size_mm=100.0,
    )
    mmpp = manual_calibrate(args.image_path, args.known_size_mm)
    print(f"Calibration result: mm_per_px = {mmpp:.6f}")
    print(f"Update configs/default.yaml: scale.mm_per_px = {mmpp}")
