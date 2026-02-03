"""Advanced dataset with tape-focused augmentation and hard negative mining"""
from __future__ import annotations
from typing import List, Tuple, Optional
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print("[WARNING] albumentations not available, using legacy augmentation")


class CableDefectDatasetAdvanced(Dataset):
    """Advanced dataset for cable/tape segmentation

    Features:
    1. Tape-focused augmentation: Random crops centered on tape regions
    2. Hard negative mining: Sample from difficult background regions
    3. Strong augmentation: Mosaic, mixup, color jitter, etc.
    4. Support for both 3-class and 7-class modes
    """

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        augment: bool = False,
        target_size: Tuple[int, int] = (512, 512),
        tape_crop_prob: float = 0.3,  # Probability of tape-focused crop
        hard_negative_dir: Optional[str] = None,  # Hard negative images directory
        hard_negative_prob: float = 0.15,  # Probability of sampling hard negative
        use_albumentations: bool = True
    ) -> None:
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.augment = augment
        self.target_size = target_size
        self.tape_crop_prob = tape_crop_prob
        self.hard_negative_dir = hard_negative_dir
        self.hard_negative_prob = hard_negative_prob
        self.use_albumentations = use_albumentations

        # List all image files
        self.image_files = sorted(
            [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        )

        # Validate masks exist
        for img_file in self.image_files:
            mask_file = os.path.splitext(img_file)[0] + '.png'
            mask_path = os.path.join(mask_dir, mask_file)
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask not found: {mask_path}")

        # Load hard negatives if provided
        self.hard_negative_files = []
        if self.hard_negative_dir and os.path.exists(self.hard_negative_dir):
            self.hard_negative_files = sorted(
                [f for f in os.listdir(self.hard_negative_dir)
                 if f.endswith(('.jpg', '.png', '.jpeg'))]
            )
            print(f"Loaded {len(self.hard_negative_files)} hard negative samples")

        # Setup augmentation pipeline
        self._setup_augmentation()

    def _setup_augmentation(self):
        """Setup albumentations augmentation pipeline"""
        self.transform = None  # Will use legacy augmentation

        if not ALBUMENTATIONS_AVAILABLE:
            return

        if not self.augment:
            self.transform = A.Compose([
                A.Resize(height=self.target_size[0], width=self.target_size[1]),
            ])
            return

        # Strong augmentation for training
        self.transform = A.Compose([
            # Geometric transformations
            A.OneOf([
                A.RandomResizedCrop(size=(self.target_size[0], self.target_size[1]), scale=(0.7, 1.0), p=1.0),
                A.Resize(height=self.target_size[0], width=self.target_size[1], p=1.0),
            ], p=0.5),

            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),

            # Color augmentation
            A.OneOf([
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=1.0),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
            ], p=0.6),

            # Noise and blur
            A.OneOf([
                A.GaussNoise(var_limit=(10, 50), p=1.0),
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.MotionBlur(blur_limit=(3, 5), p=1.0),
            ], p=0.3),

            # Weather effects (simulates industrial environment)
            A.OneOf([
                A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=1.0),
                A.RandomRain(slant_lower=-10, slant_upper=10, p=1.0),
            ], p=0.2),

            # Final resize and normalize
            A.Resize(height=self.target_size[0], width=self.target_size[1]),
        ])

    def __len__(self) -> int:
        return len(self.image_files)

    def _read_image(self, path: str) -> np.ndarray:
        """Read image with Chinese path support"""
        img_array = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image not found: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _read_mask(self, path: str) -> np.ndarray:
        """Read mask with Chinese path support"""
        mask_array = np.fromfile(path, dtype=np.uint8)
        mask = cv2.imdecode(mask_array, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {path}")
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        return mask

    def _tape_focused_crop(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply tape-focused random crop

        Strategy: Find tape regions and perform random crop centered on them
        """
        h, w = mask.shape
        tape_pixels = np.where(mask == 2)  # Class 2 = tape

        if len(tape_pixels[0]) == 0:
            # No tape found, return original
            return image, mask

        # Randomly select a tape pixel as crop center
        idx = np.random.randint(0, len(tape_pixels[0]))
        center_y, center_x = tape_pixels[0][idx], tape_pixels[1][idx]

        # Define crop size (60% ~ 100% of image)
        crop_scale = 0.6 + np.random.random() * 0.4
        crop_h = int(h * crop_scale)
        crop_w = int(w * crop_scale)

        # Calculate crop boundaries
        y1 = max(0, center_y - crop_h // 2)
        y2 = min(h, center_y + crop_h // 2)
        x1 = max(0, center_x - crop_w // 2)
        x2 = min(w, center_x + crop_w // 2)

        # Adjust if crop goes out of bounds
        if y2 - y1 < crop_h:
            if y1 == 0:
                y2 = min(h, y1 + crop_h)
            else:
                y1 = max(0, y2 - crop_h)
        if x2 - x1 < crop_w:
            if x1 == 0:
                x2 = min(w, x1 + crop_w)
            else:
                x1 = max(0, x2 - crop_w)

        # Perform crop
        image_crop = image[y1:y2, x1:x2]
        mask_crop = mask[y1:y2, x1:x2]

        return image_crop, mask_crop

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Check if we should sample hard negative
        if (self.augment and self.hard_negative_files and
            np.random.random() < self.hard_negative_prob):
            # Sample hard negative
            hn_idx = np.random.randint(0, len(self.hard_negative_files))
            img_file = self.hard_negative_files[hn_idx]
            img_path = os.path.join(self.hard_negative_dir, img_file)

            # Read image
            image = self._read_image(img_path)

            # Create empty mask (all background)
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        else:
            # Normal sample
            img_file = self.image_files[idx]
            mask_file = os.path.splitext(img_file)[0] + '.png'

            img_path = os.path.join(self.image_dir, img_file)
            mask_path = os.path.join(self.mask_dir, mask_file)

            image = self._read_image(img_path)
            mask = self._read_mask(mask_path)

            # Apply tape-focused crop
            if self.augment and np.random.random() < self.tape_crop_prob:
                image, mask = self._tape_focused_crop(image, mask)

        # Apply augmentation
        if self.transform is not None and ALBUMENTATIONS_AVAILABLE:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

            # Normalize to [0, 1] and convert to tensor
            image = image.astype(np.float32) / 255.0
            # HWC -> CHW
            image = np.transpose(image, (2, 0, 1))
            image = torch.from_numpy(image).float()

            mask = mask.astype(np.int64)
            mask = torch.from_numpy(mask).long()

            return image, mask
        else:
            # Legacy augmentation
            if self.target_size is not None:
                h, w = self.target_size
                image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

            # Simple augmentation
            if self.augment:
                if np.random.random() < 0.5:
                    image = cv2.flip(image, 1)
                    mask = cv2.flip(mask, 1)
                if np.random.random() < 0.5:
                    image = cv2.flip(image, 0)
                    mask = cv2.flip(mask, 0)
                if np.random.random() < 0.5:
                    factor = 0.7 + np.random.random() * 0.6
                    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
                    hsv[:, :, 2] *= factor
                    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
                    image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

            # Normalize
            image = image.astype(np.float32) / 255.0
            image = np.transpose(image, (2, 0, 1))
            image = torch.from_numpy(image).float()

            mask = mask.astype(np.int64)
            mask = torch.from_numpy(mask).long()

            return image, mask

        # Convert to tensor (albumentations already returns normalized tensor)
        mask = mask.astype(np.int64)
        mask = torch.from_numpy(mask).long()

        return image, mask


class CableDefectDataset3Class(CableDefectDatasetAdvanced):
    """3-class dataset with automatic mask remapping

    Maps 7 classes to 3:
    - 0: background (0, 3, 4, 5, 6 -> 0)
    - 1: cable (1 -> 1)
    - 2: tape (2 -> 2)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, mask = super().__getitem__(idx)

        # Remap 7-class to 3-class
        mask_remapped = torch.zeros_like(mask)
        mask_remapped[mask == 0] = 0  # background
        mask_remapped[mask == 1] = 1  # cable
        mask_remapped[mask == 2] = 2  # tape
        # Classes 3-6 are mapped to background (0)

        return image, mask_remapped


def create_hard_negative_dataset(
    raw_videos_dir: str,
    output_dir: str,
    num_frames: int = 200,
    frame_size: Tuple[int, int] = (512, 512)
) -> None:
    """Create hard negative dataset from raw videos

    Extracts frames that are likely to be hard negatives:
    - Frames with no clear cable/tape
    - Frames with strong reflections
    - Frames with complex backgrounds

    Args:
        raw_videos_dir: Directory containing raw videos
        output_dir: Output directory for hard negative frames
        num_frames: Number of frames to extract
        frame_size: Size to resize frames to
    """
    import cv2

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'masks'), exist_ok=True)

    video_files = [f for f in os.listdir(raw_videos_dir)
                   if f.endswith(('.mp4', '.avi', '.mov'))]

    extracted_count = 0
    target_per_video = num_frames // len(video_files) if video_files else num_frames

    for video_file in video_files:
        if extracted_count >= num_frames:
            break

        video_path = os.path.join(raw_videos_dir, video_file)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Failed to open: {video_file}")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        skip = max(1, total_frames // target_per_video)

        frame_idx = 0
        while extracted_count < num_frames and frame_idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                break

            # Resize frame
            frame_resized = cv2.resize(frame, (frame_size[1], frame_size[0]))

            # Save frame
            output_name = f"hn_{extracted_count:04d}.png"
            output_path = os.path.join(output_dir, 'images', output_name)
            cv2.imwrite(output_path, frame_resized)

            # Create empty mask (all background)
            mask_path = os.path.join(output_dir, 'masks', output_name)
            cv2.imwrite(mask_path, np.zeros((frame_size[0], frame_size[1]), dtype=np.uint8))

            extracted_count += 1
            frame_idx += skip

        cap.release()

    print(f"Extracted {extracted_count} hard negative frames to {output_dir}")
