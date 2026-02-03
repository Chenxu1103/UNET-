import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm

def ahash(img, size=8):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)
    m = g.mean()
    return (g > m).astype(np.uint8).reshape(-1)

def sim(a, b):
    return 1.0 - (np.count_nonzero(a != b) / a.size)

def parse_roi(s):
    if not s:
        return None
    x,y,w,h = map(int, s.split(','))
    return x,y,w,h

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--fps', type=float, default=5.0, help='target extraction fps')
    ap.add_argument('--roi', type=str, default='', help='x,y,w,h crop ROI')
    ap.add_argument('--dedup', type=float, default=0.97, help='hash similarity threshold to skip near-duplicates')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open {args.video}')
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    stride = max(1, int(round(src_fps / args.fps)))

    roi = parse_roi(args.roi)
    last_h = None
    saved = 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    for idx in tqdm(range(total), desc=f"Extract {os.path.basename(args.video)}"):
        ret, frame = cap.read()
        if not ret: break
        if idx % stride != 0:
            continue
        if roi:
            x,y,w,h = roi
            frame = frame[y:y+h, x:x+w]
        hsh = ahash(frame)
        if last_h is not None and sim(hsh, last_h) >= args.dedup:
            continue
        last_h = hsh
        outp = os.path.join(args.out, f"frame_{saved:06d}.jpg")
        cv2.imwrite(outp, frame)
        saved += 1

    cap.release()
    print(f"Saved {saved} frames to {args.out}")

if __name__ == "__main__":
    main()
