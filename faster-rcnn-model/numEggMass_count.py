import numpy as np
import cv2
import torch
import os
import argparse
import time
import pathlib

from model import create_model
from config import NUM_CLASSES, DEVICE, CLASSES

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--input",
    required=True,
    type=str,
    help="Path to input video file"
)
parser.add_argument(
    "--output",
    required=True,
    type=str,
    help="Path to output video file"
)
parser.add_argument(
    "--threshold",
    default=0.25,
    type=float,
    help="Detection threshold for egg masses"
)
parser.add_argument(
    "--imgsz",
    default=None,
    type=int,
    help="Resize image before inference (square)"
)
parser.add_argument(
    "--segment_duration",
    default=30,
    type=int,
    help="Length of segment to extract in seconds"
)
args = parser.parse_args()

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = create_model(num_classes=NUM_CLASSES)
checkpoint = torch.load('outputs/best_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model = model.to(device).eval()

def get_egg_mass_count(frame):
    image = frame.copy()
    if args.imgsz is not None:
        image = cv2.resize(image, (args.imgsz, args.imgsz))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    image_input = torch.tensor(
        np.transpose(image, (2, 0, 1)), dtype=torch.float
    ).to(device).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_input)

    outputs = [{k: v.cpu() for k, v in t.items()} for t in outputs]

    if len(outputs[0]['boxes']) == 0:
        return 0

    boxes = outputs[0]['boxes'].numpy()
    scores = outputs[0]['scores'].numpy()
    labels = outputs[0]['labels'].numpy()

    count = 0
    for score, label in zip(scores, labels):
        class_name = CLASSES[label]
        if score >= args.threshold and class_name.lower().replace(' ', '') == "eggmass":
            count += 1
    return count

# Scan video
cap = cv2.VideoCapture(args.input)
if not cap.isOpened():
    raise ValueError(f"Error opening video file {args.input}")

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
segment_length_frames = int(args.segment_duration * fps)
print(f"Video FPS: {fps}, total frames: {total_frames}, segment length: {segment_length_frames}")

egg_mass_counts = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    egg_mass_counts.append(get_egg_mass_count(frame))
cap.release()

cumsum = [0]
for count in egg_mass_counts:
    cumsum.append(cumsum[-1] + count)

best_start = 0
best_total = -1
for i in range(total_frames - segment_length_frames + 1):
    window_sum = cumsum[i + segment_length_frames] - cumsum[i]
    if window_sum > best_total:
        best_total = window_sum
        best_start = i

print(f"Best segment starts at frame {best_start}, total count={best_total}")

# Save segment
cap = cv2.VideoCapture(args.input)
cap.set(cv2.CAP_PROP_POS_FRAMES, best_start)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(
    args.output,
    fourcc,
    fps,
    (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
)

frames_written = 0
while frames_written < segment_length_frames:
    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)
    frames_written += 1

cap.release()
out.release()
print(f"Saved best {args.segment_duration}s segment to {args.output}")
