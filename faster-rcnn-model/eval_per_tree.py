"""
eval_per_tree.py
================
Treats the 300 images in simulation_test_data/ as a 10-second 30 fps video
and reuses the same video for each of N trees (default 9).

Per-image rule:
    Take the single highest-confidence detection of the target class.
    If that score >= threshold, the image counts as 1 SLF detection;
    otherwise 0. Multiple detections in one image still count as 1.

Per-tree percentage = detected_images / total_images * 100
Final result = mean of per-tree percentages.

Since the same video is used for every tree, every per-tree percentage
is identical and the average equals that single percentage. The per-tree
breakdown is still printed so the format matches the eventual case where
each tree has its own video.
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CLASSES, DEVICE, NUM_CLASSES
from model import create_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--weights",
        default=os.path.join(os.path.dirname(__file__), "outputs", "fasterrcnn.pth"),
    )
    p.add_argument(
        "--input",
        default=os.path.join(os.path.dirname(__file__), "..", "simulation_test_data"),
    )
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--target_class", default="adult")
    p.add_argument("--num_trees", type=int, default=9)
    return p.parse_args()


def main():
    args = parse_args()

    target = args.target_class.strip().lower()
    valid = {c.lower() for c in CLASSES[1:]}
    if target not in valid:
        print(f"[ERROR] Unknown target class '{args.target_class}'. "
              f"Choose from: {', '.join(CLASSES[1:])}")
        sys.exit(1)

    input_dir = os.path.abspath(args.input)
    image_paths = sorted(
        f for ext in ("*.png", "*.jpg", "*.jpeg")
        for f in glob.glob(os.path.join(input_dir, ext))
    )
    total = len(image_paths)
    if total == 0:
        print(f"[ERROR] No images found in {input_dir}")
        sys.exit(1)
    print(f"Found {total} image(s) in {input_dir}")
    print(f"Threshold={args.threshold}  target_class='{args.target_class}'  "
          f"num_trees={args.num_trees}\n")

    print(f"Loading weights from {args.weights} ...")
    model = create_model(num_classes=NUM_CLASSES)
    ckpt = torch.load(args.weights, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE).eval()
    print(f"Model loaded. Running on: {DEVICE}\n")

    detected = 0
    for idx, path in enumerate(image_paths, 1):
        bgr = cv2.imread(path)
        if bgr is None:
            print(f"  [{idx:>3}/{total}] SKIP (cannot read): {os.path.basename(path)}")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(
            np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))
        ).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(tensor)

        scores = outputs[0]["scores"].cpu().numpy()
        labels = outputs[0]["labels"].cpu().numpy()

        # Restrict to target class.
        mask = np.array(
            [labels[i] < len(CLASSES) and CLASSES[labels[i]].lower() == target
             for i in range(len(labels))],
            dtype=bool,
        )
        target_scores = scores[mask]

        if target_scores.size > 0:
            top_score = float(target_scores.max())
        else:
            top_score = 0.0

        hit = top_score >= args.threshold
        if hit:
            detected += 1
        flag = "HIT " if hit else "miss"
        print(f"  [{idx:>3}/{total}] {flag}  top_{target}_score={top_score:.3f}  "
              f"{os.path.basename(path)}")

    pct = detected / total * 100.0

    print("\n" + "=" * 64)
    print("PER-TREE DETECTION RATE  (same video reused per tree)")
    print("=" * 64)
    per_tree = []
    for t in range(1, args.num_trees + 1):
        per_tree.append(pct)
        print(f"  Tree {t}: {detected}/{total} images detected  -> {pct:6.2f}%")
    avg = sum(per_tree) / len(per_tree)
    print("-" * 64)
    print(f"  Detected images per tree : {detected}/{total}")
    print(f"  Per-tree percentage      : {pct:.2f}%")
    print(f"  Average over {args.num_trees} trees      : {avg:.2f}%")
    print("=" * 64)


if __name__ == "__main__":
    main()
