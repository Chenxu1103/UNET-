from __future__ import annotations
import os
from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.unetpp import UnetPP
from src.data.dataset import CableSegDataset
from src.data.transforms import build_val_tfms


@dataclass
class EvalArgs:
    model_path: str
    img_dir: str
    mask_dir: str
    encoder: str = "resnet18"
    num_classes: int = 4
    input_size: int = 512
    batch_size: int = 8


@torch.no_grad()
def iou_per_class(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> np.ndarray:
    ious = np.zeros((num_classes,))
    for c in range(num_classes):
        inter = ((pred == c) & (target == c)).sum().item()
        union = ((pred == c) | (target == c)).sum().item()
        if union > 0:
            ious[c] = inter / union
        else:
            ious[c] = 1.0 if (pred != c).all() and (target != c).all() else 0.0
    return ious


def main(a: EvalArgs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UnetPP(a.encoder, a.num_classes).to(device)
    ckpt = torch.load(a.model_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    all_ids = [f for f in os.listdir(a.img_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    ds = CableSegDataset(a.img_dir, a.mask_dir, all_ids, transform=build_val_tfms(a.input_size, a.input_size))
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    all_ious = []
    for img, mask in loader:
        img = img.to(device)
        mask = mask.to(device)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            logits = model(img)
        pred = torch.argmax(logits, dim=1)
        ious = iou_per_class(pred, mask, a.num_classes)
        all_ious.append(ious)

    all_ious = np.array(all_ious)
    miou = all_ious.mean()
    print(f"mIoU: {miou:.4f}")
    for c in range(a.num_classes):
        print(f"  Class {c}: {all_ious[:, c].mean():.4f}")


if __name__ == "__main__":
    args = EvalArgs(
        model_path="runs/unetpp_resnet18/best.pt",
        img_dir="data/images",
        mask_dir="data/masks",
        encoder="resnet18",
        num_classes=4,
        input_size=512,
        batch_size=8,
    )
    main(args)
