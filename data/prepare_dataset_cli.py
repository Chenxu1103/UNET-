import os
import json
import glob
import random
import shutil
import argparse
import numpy as np
import cv2

CLASS_MAP = {
    "background": 0,
    "cable": 1,
    "tape": 2,
    "burr_defect": 3,
    "bulge_defect": 4,
    "loose_defect": 5,
    "damage_defect": 6,
}

def json_to_mask(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    h, w = data["imageHeight"], data["imageWidth"]
    mask = np.zeros((h, w), dtype=np.uint8)

    shapes = data.get("shapes", [])
    def key(s):
        lab = s.get("label","")
        return 1 if "defect" in lab else 0
    shapes = sorted(shapes, key=key)

    for s in shapes:
        lab = s.get("label","")
        if lab not in CLASS_MAP:
            continue
        pts = np.array(s["points"], dtype=np.int32)
        cv2.fillPoly(mask, [pts], CLASS_MAP[lab])
    return mask

def find_image_for_json(jp):
    base = os.path.splitext(os.path.basename(jp))[0]
    for ext in [".jpg",".jpeg",".png",".bmp"]:
        ip = os.path.join(os.path.dirname(jp), base + ext)
        if os.path.exists(ip): return ip
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labelme_dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--val_ratio', type=float, default=0.1)
    ap.add_argument('--test_ratio', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    json_files = glob.glob(os.path.join(args.labelme_dir, "*.json"))
    pairs = []
    for jp in json_files:
        ip = find_image_for_json(jp)
        if ip is None:
            continue
        pairs.append((ip, jp))
    if not pairs:
        raise RuntimeError("No (image,json) pairs found. Ensure json and images are in same folder.")

    random.shuffle(pairs)
    n = len(pairs)
    n_val = int(n * args.val_ratio)
    n_test = int(n * args.test_ratio)
    n_train = n - n_val - n_test
    splits = {
        "train": pairs[:n_train],
        "val": pairs[n_train:n_train+n_val],
        "test": pairs[n_train+n_val:],
    }

    for sp, items in splits.items():
        img_out = os.path.join(args.out, sp, "images")
        msk_out = os.path.join(args.out, sp, "masks")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(msk_out, exist_ok=True)
        for ip, jp in items:
            fname = os.path.basename(ip)
            shutil.copy2(ip, os.path.join(img_out, fname))
            mask = json_to_mask(jp)
            mname = os.path.splitext(fname)[0] + ".png"
            cv2.imwrite(os.path.join(msk_out, mname), mask)

    print(f"Done. train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

if __name__ == "__main__":
    main()
