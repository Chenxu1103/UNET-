from __future__ import annotations
import albumentations as A
from albumentations.pytorch import ToTensorV2


def build_train_tfms(input_h: int, input_w: int):
    return A.Compose([
        A.Resize(input_h, input_w),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussianBlur(p=0.2),
        A.HueSaturationValue(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.1, rotate_limit=8, p=0.5, border_mode=0),
        A.Normalize(),
        ToTensorV2(),
    ])


def build_val_tfms(input_h: int, input_w: int):
    return A.Compose([
        A.Resize(input_h, input_w),
        A.Normalize(),
        ToTensorV2(),
    ])
