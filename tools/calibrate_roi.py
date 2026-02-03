"""
ROI Calibration Tool

Interactive tool to calibrate ROI for cable wrapping detection.
"""

import sys
import argparse
from pathlib import Path
import cv2
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.refactor.config import ROIConfig


class ROICalibrator:
    """Interactive ROI calibration tool."""

    def __init__(self, frame):
        """
        Initialize calibrator.

        Args:
            frame: Frame to calibrate on
        """
        self.frame = frame.copy()
        self.display_frame = frame.copy()
        self.roi = None
        self.drawing = False
        self.start_point = None
        self.end_point = None

    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for ROI selection."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.end_point = (x, y)
                self.update_display()

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_point = (x, y)
            self.update_display()

    def update_display(self):
        """Update display with current ROI."""
        self.display_frame = self.frame.copy()

        if self.start_point and self.end_point:
            cv2.rectangle(self.display_frame, self.start_point, self.end_point, (0, 255, 0), 2)

            # Draw center line
            x1, y1 = self.start_point
            x2, y2 = self.end_point
            center_x = (x1 + x2) // 2
            cv2.line(self.display_frame, (center_x, y1), (center_x, y2), (255, 0, 0), 1)

            # Draw dimensions
            w = abs(x2 - x1)
            h = abs(y2 - y1)
            cv2.putText(self.display_frame, f"{w}x{h}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow('ROI Calibration', self.display_frame)

    def get_roi_config(self) -> ROIConfig:
        """
        Get ROI configuration from selection.

        Returns:
            ROIConfig instance
        """
        if not self.start_point or not self.end_point:
            raise ValueError("No ROI selected")

        x1, y1 = self.start_point
        x2, y2 = self.end_point

        x = min(x1, x2)
        y = min(y1, y2)
        w = abs(x2 - x1)
        h = abs(y2 - y1)

        return ROIConfig(mode='fixed', x=x, y=y, w=w, h=h)

    def calibrate(self) -> ROIConfig:
        """
        Run calibration process.

        Returns:
            ROIConfig instance
        """
        cv2.namedWindow('ROI Calibration')
        cv2.setMouseCallback('ROI Calibration', self.mouse_callback)

        print("="*80)
        print("ROI Calibration Tool")
        print("="*80)
        print("Instructions:")
        print("1. Click and drag to select ROI")
        print("2. Press 'r' to reset")
        print("3. Press 's' to save and exit")
        print("4. Press 'q' to quit without saving")
        print("="*80)

        self.update_display()

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord('r'):
                # Reset
                self.start_point = None
                self.end_point = None
                self.display_frame = self.frame.copy()
                cv2.imshow('ROI Calibration', self.display_frame)
                print("ROI reset")

            elif key == ord('s'):
                # Save
                if self.start_point and self.end_point:
                    roi_config = self.get_roi_config()
                    print(f"ROI saved: x={roi_config.x}, y={roi_config.y}, w={roi_config.w}, h={roi_config.h}")
                    cv2.destroyAllWindows()
                    return roi_config
                else:
                    print("No ROI selected. Please select ROI first.")

            elif key == ord('q'):
                # Quit
                print("Calibration cancelled")
                cv2.destroyAllWindows()
                return None


def main():
    parser = argparse.ArgumentParser(description='ROI Calibration Tool')
    parser.add_argument('--video', type=str, required=True, help='Input video path')
    parser.add_argument('--output', type=str, default='roi.json', help='Output ROI JSON path')
    parser.add_argument('--frame', type=int, default=0, help='Frame number to use for calibration')

    args = parser.parse_args()

    # Open video
    cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        print(f"Error: Failed to open video: {args.video}")
        return

    # Seek to frame
    if args.frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)

    # Read frame
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Error: Failed to read frame {args.frame}")
        return

    # Run calibration
    calibrator = ROICalibrator(frame)
    roi_config = calibrator.calibrate()

    if roi_config:
        # Save to JSON
        roi_config.to_json(args.output)
        print(f"ROI configuration saved to: {args.output}")
    else:
        print("Calibration cancelled, no file saved")


if __name__ == '__main__':
    main()
