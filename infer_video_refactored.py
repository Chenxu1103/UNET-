"""
Refactored video inference script for cable wrapping detection system.

This script integrates all refactored modules to provide:
- Stable ROI-based detection
- Grayscale enhancement for B&W cameras
- Shape-constrained segmentation
- Burr detection
- Event detection with uniformity calculation
- Comprehensive logging and visualization
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional, List
import csv

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.models.unetpp import NestedUNet
from src.refactor import (
    RefactorConfig, ROIConfig,
    preprocess_frame, crop_roi, paste_roi_mask,
    postprocess_masks,
    get_burr_mask_rulebased,
    FrameMetrics, EventDetector
)


def setup_logging(output_dir: Path, debug: bool = False):
    """Setup logging configuration."""
    log_level = logging.DEBUG if debug else logging.INFO
    log_file = output_dir / 'inference.log'

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)


def load_model_with_auto_classes(model_path: str, device: torch.device):
    """
    Load model and automatically infer num_classes from checkpoint.

    Args:
        model_path: Path to model checkpoint
        device: Device to load model on

    Returns:
        Tuple of (model, num_classes)
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Infer num_classes from checkpoint
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
        final_weight_key = 'final.weight'

        if final_weight_key in state_dict:
            num_classes = state_dict[final_weight_key].shape[0]
        else:
            raise ValueError(f"Cannot infer num_classes: '{final_weight_key}' not found in checkpoint")
    else:
        raise ValueError("Checkpoint does not contain 'model' key")

    # Create and load model
    model = NestedUNet(num_classes=num_classes, deep_supervision=True).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    return model, num_classes


def model_inference(model, frame: np.ndarray, device: torch.device, input_size: int = 512) -> np.ndarray:
    """
    Run model inference on frame.

    Args:
        model: Model instance
        frame: Input frame (H, W, 3)
        device: Device
        input_size: Model input size

    Returns:
        Segmentation mask (H, W) with class indices
    """
    h, w = frame.shape[:2]

    # Resize to model input size
    resized = cv2.resize(frame, (input_size, input_size), interpolation=cv2.INTER_LINEAR)

    # Normalize
    img_tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(img_tensor)

        if isinstance(output, (list, tuple)):
            output = output[0]  # Use first output for deep supervision

        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    # Resize back to original size
    pred_resized = cv2.resize(pred.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

    return pred_resized


def extract_class_masks(pred_mask: np.ndarray, num_classes: int) -> dict:
    """
    Extract individual class masks from prediction.

    Args:
        pred_mask: Prediction mask (H, W) with class indices
        num_classes: Number of classes

    Returns:
        Dictionary of class_id -> binary mask
    """
    masks = {}

    for class_id in range(num_classes):
        masks[class_id] = (pred_mask == class_id).astype(np.uint8) * 255

    return masks


def measure_diameter(mask: np.ndarray) -> float:
    """
    Measure diameter of mask using minimum enclosing circle.

    Args:
        mask: Binary mask (H, W)

    Returns:
        Diameter in pixels, or 0 if no contours found
    """
    if mask.max() == 0:
        return 0.0

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return 0.0

    # Use largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Minimum enclosing circle
    (x, y), radius = cv2.minEnclosingCircle(largest_contour)

    return radius * 2.0


def create_overlay(frame: np.ndarray, mask_cable: np.ndarray, mask_tape: np.ndarray,
                   mask_burr: np.ndarray, metrics: FrameMetrics, events: List[str]) -> np.ndarray:
    """
    Create visualization overlay.

    Args:
        frame: Original frame
        mask_cable: Cable mask
        mask_tape: Tape mask
        mask_burr: Burr mask
        metrics: Frame metrics
        events: Triggered events

    Returns:
        Overlay frame
    """
    overlay = frame.copy()

    # Draw masks with transparency
    if mask_cable.max() > 0:
        overlay[mask_cable > 0] = overlay[mask_cable > 0] * 0.6 + np.array([0, 255, 0]) * 0.4

    if mask_tape.max() > 0:
        overlay[mask_tape > 0] = overlay[mask_tape > 0] * 0.6 + np.array([255, 0, 0]) * 0.4

    if mask_burr.max() > 0:
        overlay[mask_burr > 0] = overlay[mask_burr > 0] * 0.6 + np.array([0, 0, 255]) * 0.4

    # Draw text info
    y_offset = 30
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2

    # Frame ID
    cv2.putText(overlay, f"Frame: {metrics.frame_id}", (10, y_offset),
                font, font_scale, (255, 255, 255), thickness)
    y_offset += 30

    # Diameters
    cv2.putText(overlay, f"Cable D: {metrics.dc_px:.1f}px", (10, y_offset),
                font, font_scale, (0, 255, 0), thickness)
    y_offset += 30

    cv2.putText(overlay, f"Tape D: {metrics.dt_px:.1f}px", (10, y_offset),
                font, font_scale, (255, 0, 0), thickness)
    y_offset += 30

    # Ratio
    if metrics.ratio is not None:
        cv2.putText(overlay, f"Ratio: {metrics.ratio:.2f}", (10, y_offset),
                    font, font_scale, (255, 255, 0), thickness)
        y_offset += 30

    # Burr
    if metrics.has_burr:
        cv2.putText(overlay, "BURR DETECTED", (10, y_offset),
                    font, font_scale, (0, 0, 255), thickness)
        y_offset += 30

    # Events
    if events:
        for event in events:
            cv2.putText(overlay, f"EVENT: {event.upper()}", (10, y_offset),
                        font, font_scale, (0, 255, 255), thickness)
            y_offset += 30

    return overlay


def main():
    parser = argparse.ArgumentParser(description='Refactored cable wrapping detection inference')
    parser.add_argument('--video', type=str, required=True, help='Input video path')
    parser.add_argument('--model', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--config', type=str, default=None, help='Config YAML path')
    parser.add_argument('--roi', type=str, default=None, help='ROI JSON path')
    parser.add_argument('--input-size', type=int, default=512, help='Model input size')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device')
    parser.add_argument('--output', type=str, default='log/refactored_output', help='Output directory')
    parser.add_argument('--show-preview', action='store_true', help='Show preview window')
    parser.add_argument('--debug', action='store_true', help='Debug mode')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup subdirectories
    snapshots_dir = output_dir / 'snapshots'
    overlays_dir = output_dir / 'overlays'
    snapshots_dir.mkdir(exist_ok=True)
    overlays_dir.mkdir(exist_ok=True)

    # Setup logging
    logger = setup_logging(output_dir, args.debug)
    logger.info("="*80)
    logger.info("Refactored Cable Wrapping Detection System")
    logger.info("="*80)

    # Load configuration
    if args.config:
        logger.info(f"Loading config from: {args.config}")
        config = RefactorConfig.from_yaml(args.config)
    else:
        logger.info("Using default configuration")
        config = RefactorConfig()

    # Load or create ROI
    if args.roi:
        logger.info(f"Loading ROI from: {args.roi}")
        roi = ROIConfig.from_json(args.roi)
    else:
        logger.warning("No ROI specified, using default ROI")
        roi = config.roi

    logger.info(f"ROI: x={roi.x}, y={roi.y}, w={roi.w}, h={roi.h}")

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    logger.info(f"Using device: {device}")

    # Load model
    logger.info(f"Loading model from: {args.model}")
    model, num_classes = load_model_with_auto_classes(args.model, device)
    logger.info(f"Model loaded with {num_classes} classes")

    # Open video
    logger.info(f"Opening video: {args.video}")
    cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        logger.error(f"Failed to open video: {args.video}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logger.info(f"Video: {frame_width}x{frame_height}, {fps} FPS, {total_frames} frames")

    # Create video writer
    output_video_path = output_dir / 'detection_result.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (frame_width, frame_height))

    # Initialize event detector
    event_detector = EventDetector(config.event)

    # Open CSV file for events
    events_csv_path = output_dir / 'events.csv'
    csv_file = open(events_csv_path, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        'frame_id', 'timestamp', 'event_type', 'dc_px', 'dt_px', 'delta_d_px',
        'ratio', 'uniformity', 'snapshot_path', 'overlay_path'
    ])

    # Processing loop
    frame_id = 0
    pbar = tqdm(total=total_frames, desc="Processing")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Preprocess
            preprocessed = preprocess_frame(frame, config.preprocess)

            # Crop ROI
            roi_frame = crop_roi(preprocessed, roi)

            # Model inference
            pred_mask = model_inference(model, roi_frame, device, args.input_size)

            # Extract class masks (assuming class 1=cable, class 2=tape)
            class_masks = extract_class_masks(pred_mask, num_classes)

            # Get cable and tape masks
            mask_cable_roi = class_masks.get(1, np.zeros_like(pred_mask))
            mask_tape_roi = class_masks.get(2, np.zeros_like(pred_mask))

            # Postprocess
            mask_cable_roi, mask_tape_roi = postprocess_masks(
                mask_cable_roi, mask_tape_roi, config.postprocess, roi.w
            )

            # Paste back to full frame
            mask_cable_full = np.zeros((frame_height, frame_width), dtype=np.uint8)
            mask_tape_full = np.zeros((frame_height, frame_width), dtype=np.uint8)

            mask_cable_full = paste_roi_mask(mask_cable_full, mask_cable_roi, roi)
            mask_tape_full = paste_roi_mask(mask_tape_full, mask_tape_roi, roi)

            # Measure diameters
            dc_px = measure_diameter(mask_cable_full)
            dt_px = measure_diameter(mask_tape_full)
            delta_d_px = dt_px - dc_px
            ratio = dt_px / dc_px if dc_px > 0 else None

            # Burr detection
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mask_burr_full = get_burr_mask_rulebased(frame_gray, mask_cable_full, config.burr)
            has_burr = mask_burr_full.max() > 0

            # Calculate coverage
            cable_coverage = np.sum(mask_cable_full > 0) / (frame_width * frame_height)
            tape_coverage = np.sum(mask_tape_full > 0) / (frame_width * frame_height)

            # Create metrics
            metrics = FrameMetrics(
                frame_id=frame_id,
                dc_px=dc_px,
                dt_px=dt_px,
                delta_d_px=delta_d_px,
                ratio=ratio,
                has_burr=has_burr,
                cable_coverage=cable_coverage,
                tape_coverage=tape_coverage
            )

            # Event detection
            events = event_detector.add_frame(metrics)
            uniformity = event_detector.compute_uniformity()

            # Create overlay
            overlay = create_overlay(frame, mask_cable_full, mask_tape_full, mask_burr_full, metrics, events)

            # Write to video
            out.write(overlay)

            # Save event snapshots
            if events:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

                for event_type in events:
                    snapshot_path = snapshots_dir / f"frame_{frame_id:06d}_{event_type}.jpg"
                    overlay_path = overlays_dir / f"frame_{frame_id:06d}_{event_type}.jpg"

                    cv2.imwrite(str(snapshot_path), frame)
                    cv2.imwrite(str(overlay_path), overlay)

                    # Write to CSV
                    csv_writer.writerow([
                        frame_id,
                        timestamp,
                        event_type,
                        f"{dc_px:.2f}",
                        f"{dt_px:.2f}",
                        f"{delta_d_px:.2f}",
                        f"{ratio:.3f}" if ratio is not None else "",
                        f"{uniformity:.4f}" if uniformity is not None else "",
                        str(snapshot_path.relative_to(output_dir)),
                        str(overlay_path.relative_to(output_dir))
                    ])

                    logger.info(f"Frame {frame_id}: Event '{event_type}' detected")

            # Show preview
            if args.show_preview:
                cv2.imshow('Detection', overlay)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_id += 1
            pbar.update(1)

    finally:
        pbar.close()
        cap.release()
        out.release()
        csv_file.close()

        if args.show_preview:
            cv2.destroyAllWindows()

    logger.info("="*80)
    logger.info("Processing complete")
    logger.info(f"Total frames processed: {frame_id}")
    logger.info(f"Output video: {output_video_path}")
    logger.info(f"Events CSV: {events_csv_path}")
    logger.info(f"Snapshots: {snapshots_dir}")
    logger.info(f"Overlays: {overlays_dir}")
    logger.info("="*80)


if __name__ == '__main__':
    main()
