"""
eval_detection_rate.py
======================
Run Faster R-CNN inference on every image in simulation_test_data/ and
report per-class detection counts plus overall detection-rate percentage.

Assumption: each image contains exactly 1 ground-truth instance
(adult spotted lantern fly), so total ground truth = number of images.

Detection rule: an image is counted as "detected" when at least one
prediction with score >= threshold exists for the target class.

Usage (from faster-rcnn-model/):
    python eval_detection_rate.py
    python eval_detection_rate.py --threshold 0.3 --target_class adult
    python eval_detection_rate.py --target_class all
"""

import argparse
import glob
import os
import sys
from collections import defaultdict

import cv2
import numpy as np
import torch

# Allow imports from this directory regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CLASSES, DEVICE, NUM_CLASSES
from model import create_model

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument(
    "--weights",
    default=os.path.join(os.path.dirname(__file__), "outputs", "fasterrcnn.pth"),
    help="Path to model weights (default: outputs/fasterrcnn.pth)",
)
parser.add_argument(
    "--input",
    default=os.path.join(os.path.dirname(__file__), "..", "simulation_test_data"),
    help="Directory of test images (default: ../simulation_test_data)",
)
parser.add_argument(
    "--threshold",
    default=0.25,
    type=float,
    help="Confidence threshold for a detection to count (default: 0.25)",
)
parser.add_argument(
    "--target_class",
    default="adult",
    help=(
        "Class name to count as a 'hit'. Use 'all' to count any detection "
        "regardless of class. Available: " + ", ".join(CLASSES[1:])
    ),
)
parser.add_argument(
    "--imgsz",
    default=None,
    type=int,
    help="Optional square resize before inference (e.g. 640)",
)
args = parser.parse_args()

# ── Validate target class ─────────────────────────────────────────────────────

target_class = args.target_class.strip().lower()
valid_classes = {c.lower() for c in CLASSES[1:]}  # exclude __background__
if target_class != "all" and target_class not in valid_classes:
    print(f"[ERROR] Unknown target class '{args.target_class}'. "
          f"Choose from: all, {', '.join(CLASSES[1:])}")
    sys.exit(1)

# ── Gather images ─────────────────────────────────────────────────────────────

input_dir = os.path.abspath(args.input)
image_paths = sorted(
    f for ext in ("*.png", "*.jpg", "*.jpeg")
    for f in glob.glob(os.path.join(input_dir, ext))
)
total_images = len(image_paths)
if total_images == 0:
    print(f"[ERROR] No images found in {input_dir}")
    sys.exit(1)
print(f"Found {total_images} image(s) in {input_dir}")

# ── Load model ────────────────────────────────────────────────────────────────

print(f"Loading weights from {args.weights} ...")
model = create_model(num_classes=NUM_CLASSES)
checkpoint = torch.load(args.weights, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE).eval()
print(f"Model loaded. Running on: {DEVICE}\n")

# ── Inference loop ────────────────────────────────────────────────────────────

detected_images = 0          # images with >= 1 hit for the target class
per_class_hits = defaultdict(int)   # class_name -> images where it was detected
per_class_box_total = defaultdict(int)  # class_name -> total boxes across all images

for idx, img_path in enumerate(image_paths, 1):
    image_bgr = cv2.imread(img_path)
    if image_bgr is None:
        print(f"  [{idx}/{total_images}] SKIP (cannot read): {os.path.basename(img_path)}")
        continue

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    if args.imgsz:
        image_rgb = cv2.resize(image_rgb, (args.imgsz, args.imgsz))

    tensor = torch.from_numpy(
        np.transpose(image_rgb.astype(np.float32) / 255.0, (2, 0, 1))
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)

    boxes  = outputs[0]["boxes"].cpu().numpy()
    scores = outputs[0]["scores"].cpu().numpy()
    labels = outputs[0]["labels"].cpu().numpy()

    # Apply threshold filter.
    keep = scores >= args.threshold
    boxes  = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    # Tally per-class boxes.
    image_has_hit = False
    seen_classes_this_image = set()
    for label, score in zip(labels, scores):
        class_name = CLASSES[label] if label < len(CLASSES) else "unknown"
        per_class_box_total[class_name] += 1
        if class_name not in seen_classes_this_image:
            per_class_hits[class_name] += 1
            seen_classes_this_image.add(class_name)

        is_target = (
            target_class == "all"
            or class_name.lower() == target_class
        )
        if is_target:
            image_has_hit = True

    if image_has_hit:
        detected_images += 1

    n_det = int(keep.sum())
    flag = "HIT" if image_has_hit else "miss"
    print(f"  [{idx:>3}/{total_images}] {flag}  {n_det} detection(s)  "
          f"{os.path.basename(img_path)}")

# ── Summary ───────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("DETECTION SUMMARY")
print("=" * 60)
print(f"Total images          : {total_images}")
print(f"Confidence threshold  : {args.threshold}")
print(f"Target class          : {args.target_class}")
print()

print("Per-class results (images with ≥1 detection / total boxes):")
for cls in CLASSES[1:]:
    img_hits  = per_class_hits.get(cls, 0)
    box_count = per_class_box_total.get(cls, 0)
    pct = img_hits / total_images * 100
    print(f"  {cls:<25}  {img_hits:>3} / {total_images} images  "
          f"({pct:5.1f}%)   {box_count} total boxes")

print()
detection_pct = detected_images / total_images * 100
print(f"Images with target class detected : {detected_images} / {total_images}")
print(f"Detection rate                    : {detection_pct:.1f}%")
print("=" * 60)
