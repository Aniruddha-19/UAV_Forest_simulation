"""
detector.py
===========
Faster R-CNN inference wrapper for real-time adult SLF detection.

Loads the trained model once at import time and exposes a single function:

    detect_adult_slf(frame_bgr, threshold=0.25)
        → list of (x1, y1, x2, y2) bounding boxes for 'adult' detections

The model and config live in the faster-rcnn-model/ subdirectory relative
to this file.  sys.path is temporarily extended so that model.py and
config.py can be imported without installing them as a package.
"""

import os
import sys

import numpy as np
import torch
from torchvision import transforms as T

# ── Locate the faster-rcnn-model subdirectory ─────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_FRCNN_DIR  = os.path.join(_SCRIPT_DIR, "faster-rcnn-model")
_MODEL_PATH = os.path.join(_FRCNN_DIR, "outputs", "fasterrcnn.pth")

# Temporarily add to sys.path so model.py / config.py can be imported
sys.path.insert(0, _FRCNN_DIR)
try:
    from model  import create_model          # noqa: E402
    from config import NUM_CLASSES, DEVICE, CLASSES  # noqa: E402
finally:
    # Remove the injected path so it doesn't interfere with other imports
    sys.path.pop(0)

# ── Load model once at module import ──────────────────────────────────────────
print(f"[detector] Loading Faster R-CNN from {_MODEL_PATH} …")
_model = create_model(num_classes=NUM_CLASSES)
_checkpoint = torch.load(_MODEL_PATH, map_location=DEVICE)
_model.load_state_dict(_checkpoint["model_state_dict"])
_model.to(DEVICE).eval()
print(f"[detector] Model ready on {DEVICE}. Classes: {CLASSES}")

# ── Pre-built transform (reused every frame) ──────────────────────────────────
_transform = T.Compose([
    T.ToPILImage(),
    T.ToTensor(),
])

_ADULT_CLASS = "adult"


# ── Public API ────────────────────────────────────────────────────────────────

def detect_adult_slf(frame_bgr: np.ndarray,
                     threshold: float = 0.10
                     ) -> list[tuple[int, int, int, int]]:
    """
    Run Faster R-CNN on *frame_bgr* (a BGR uint8 numpy array from OpenCV /
    PyBullet) and return bounding boxes for all 'adult' SLF detections that
    exceed *threshold* confidence.

    Parameters
    ----------
    frame_bgr : H × W × 3 BGR uint8 numpy array
    threshold : confidence threshold (0.0 – 1.0); lowered to 0.10 to compensate
                for the sim-to-real domain gap in synthetic PyBullet renders

    Returns
    -------
    List of (x1, y1, x2, y2) integer bounding boxes, one per detection.
    Empty list when no adult SLF are found.
    """
    # BGR → RGB (model was trained on RGB images)
    rgb    = frame_bgr[:, :, ::-1].copy()
    tensor = _transform(rgb)
    tensor = torch.unsqueeze(tensor, 0)

    with torch.no_grad():
        outputs = _model(tensor.to(DEVICE))

    outputs = [{k: v.to("cpu") for k, v in t.items()} for t in outputs]

    boxes = []
    if len(outputs[0]["boxes"]) == 0:
        return boxes

    scores = outputs[0]["scores"].numpy()
    labels = outputs[0]["labels"].numpy()
    raw_boxes = outputs[0]["boxes"].numpy().astype(int)

    for box, score, label in zip(raw_boxes, scores, labels):
        if score >= threshold and CLASSES[label] == _ADULT_CLASS:
            boxes.append((int(box[0]), int(box[1]), int(box[2]), int(box[3])))

    return boxes
