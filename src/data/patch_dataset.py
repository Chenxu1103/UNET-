"""Patch-based dataset for small defect detection

Crops patches centered on defects to ensure visibility during training.
"""
import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from PIL import Image
import random
import cv2


class PatchDefectDataset(Dataset):
    """Patch-based dataset with defect mining

    Strategy:
    - 50% patches from defect images (centered on defects)
    - 50% patches from normal images
    - Ensures each batch has defect pixels
    """

    def __init__(
        self,
        image_dir,
        mask_dir,
        patch_size=640,
        defect_ratio=0.5,
        augment=True,
        target_size=None
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.patch_size = patch_size
        self.defect_ratio = defect_ratio
        self.augment = augment
        self.target_size = target_size if target_size else patch_size

        # Find all images
        self.images = sorted(list(self.image_dir.glob("*.jpg")))
        print(f"Found {len(self.images)} images")

        # Analyze defect samples
        self.defect_samples = []
        self.normal_samples = []

        defect_ids = [3, 4, 5]  # burr, loose, wrap_uneven

        for img_path in self.images:
            mask_path = self.mask_dir / (img_path.stem + ".png")
            mask = Image.open(mask_path)
            mask_array = np.array(mask)
            if mask_array.ndim == 3:
                mask_array = mask_array[:, :, 0]

            # Check if has defect
            uniq = np.unique(mask_array)
            has_defect = any(cls in defect_ids for cls in uniq)

            if has_defect:
                # Find defect bounding boxes
                defect_masks = []
                for cls in defect_ids:
                    if cls in uniq:
                        defect_masks.append(mask_array == cls)

                combined = np.zeros_like(mask_array, dtype=bool)
                for dm in defect_masks:
                    combined = combined | dm

                # Get bounding box
                rows = np.any(combined, axis=1)
                cols = np.any(combined, axis=0)
                if rows.any() and cols.any():
                    rmin, rmax = np.where(rows)[0][[0, -1]]
                    cmin, cmax = np.where(cols)[0][[0, -1]]

                    self.defect_samples.append({
                        'image': img_path,
                        'mask': mask_path,
                        'bbox': (rmin, rmax, cmin, cmax)
                    })
            else:
                self.normal_samples.append(img_path)

        print(f"  Defect samples: {len(self.defect_samples)}")
        print(f"  Normal samples: {len(self.normal_samples)}")

        # Calculate sampling
        n_defect_patches = int(len(self.images) * defect_ratio)
        n_normal_patches = len(self.images) - n_defect_patches

        self.sample_indices = []
        for i in range(len(self.images)):
            if i < n_defect_patches:
                self.sample_indices.append(('defect', i % len(self.defect_samples)))
            else:
                self.sample_indices.append(('normal', i % len(self.normal_samples)))

        random.shuffle(self.sample_indices)

    def __len__(self):
        return len(self.sample_indices)

    def _crop_patch(self, sample_type):
        """Crop a patch from image/mask"""
        if sample_type == 'defect':
            # Random defect sample
            idx = random.randint(0, len(self.defect_samples) - 1)
            sample = self.defect_samples[idx]

            # Load full image
            img = Image.open(sample['image'])
            mask_full = Image.open(sample['mask'])
            img_array = np.array(img)
            mask_array = np.array(mask_full)
            if mask_array.ndim == 3:
                mask_array = mask_array[:, :, 0]

            # Get bbox center
            rmin, rmax, cmin, cmax = sample['bbox']
            center_r = (rmin + rmax) // 2
            center_c = (cmin + cmax) // 2

            # Get image dimensions
            h, w = img_array.shape[:2]

            # Random jitter around center
            jitter_range = self.patch_size // 2
            new_center_r = center_r + random.randint(-jitter_range, jitter_range)
            new_center_c = center_c + random.randint(-jitter_range, jitter_range)

            # Clip to image bounds
            new_center_r = max(self.patch_size//2, min(h - self.patch_size//2, new_center_r))
            new_center_c = max(self.patch_size//2, min(w - self.patch_size//2, new_center_c))

            # Crop patch
            r1 = new_center_r - self.patch_size // 2
            r2 = r1 + self.patch_size
            c1 = new_center_c - self.patch_size // 2
            c2 = c1 + self.patch_size

            img_patch = img_array[r1:r2, c1:c2]
            mask_patch = mask_array[r1:r2, c1:c2]

        else:
            # Normal sample - random crop
            idx = random.randint(0, len(self.normal_samples) - 1)
            img_path = self.normal_samples[idx]
            mask_path = self.mask_dir / (img_path.stem + ".png")

            img = Image.open(img_path)
            mask_full = Image.open(mask_path)
            img_array = np.array(img)
            mask_array = np.array(mask_full)
            if mask_array.ndim == 3:
                mask_array = mask_array[:, :, 0]

            # Get image dimensions
            h, w = img_array.shape[:2]

            # Random crop
            r1 = random.randint(0, h - self.patch_size)
            c1 = random.randint(0, w - self.patch_size)
            r2 = r1 + self.patch_size
            c2 = c1 + self.patch_size

            img_patch = img_array[r1:r2, c1:c2]
            mask_patch = mask_array[r1:r2, c1:c2]

        return img_patch, mask_patch

    def _augment(self, image, mask):
        """Apply augmentation"""
        # Horizontal flip
        if random.random() > 0.5:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()

        # Vertical flip
        if random.random() > 0.5:
            image = np.flipud(image).copy()
            mask = np.flipud(mask).copy()

        # Rotation (90, 180, 270)
        if random.random() > 0.5:
            k = random.randint(1, 3)
            image = np.rot90(image, k).copy()
            mask = np.rot90(mask, k).copy()

        # Brightness/contrast
        if random.random() > 0.5:
            alpha = random.uniform(0.8, 1.2)
            beta = random.randint(-20, 20)
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

        return image, mask

    def __getitem__(self, idx):
        sample_type, _ = self.sample_indices[idx]

        # Crop patch
        image, mask = self._crop_patch(sample_type)

        # Augment
        if self.augment:
            image, mask = self._augment(image, mask)

        # Resize to target size
        # Handle both int and tuple target_size
        if isinstance(self.target_size, int):
            dst_size = (self.target_size, self.target_size)
        else:
            dst_size = self.target_size

        if dst_size != (self.patch_size, self.patch_size):
            image = cv2.resize(image, dst_size, interpolation=cv2.INTER_LINEAR)
            # CRITICAL: Use NEAREST for masks to preserve labels
            mask = cv2.resize(mask, dst_size, interpolation=cv2.INTER_NEAREST)

        # Convert to tensor
        image = torch.from_numpy(image).float() / 255.0
        image = image.permute(2, 0, 1)  # HWC -> CHW

        mask = torch.from_numpy(mask).long()

        # Convert to binary: defect vs non-defect
        # defect classes: 3(burr), 4(loose), 5(wrap_uneven) -> 1
        # others: 0(bg), 1(cable), 2(tape) -> 0
        binary_mask = torch.zeros_like(mask)
        binary_mask[(mask == 3) | (mask == 4) | (mask == 5)] = 1

        return image, binary_mask
