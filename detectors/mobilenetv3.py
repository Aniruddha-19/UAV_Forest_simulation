"""
MobileNet V3 Large classifier backend (5 classes).

Classifier, not detector — outputs a single class id per frame. Wrapped to
the detector interface by returning a single full-frame box on 'adult'
predictions and an empty list otherwise. The simulation's HUD then reports
the adult-rate as the percentage of 'yes' classifications.
"""

import ast
import os

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torchvision.models import mobilenet_v3_large

_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Standard MobileNet V3 input pipeline: ImageNet normalization on a
# 256-resize -> 224 center-crop tensor.
_PREPROCESS = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_model = None
_threshold = 0.5
_adult_idx = None
_classes = None
_imgsz = None  # in-memory pre-resize, e.g. [3840, 2160]


def _load_labels(label_path: str) -> list[str]:
    with open(label_path, "r") as f:
        raw = f.read()
    label_dict = ast.literal_eval(raw)
    return [label_dict[k] for k in sorted(label_dict)]


def init(name: str, cfg: dict) -> None:
    global _model, _threshold, _adult_idx, _classes, _imgsz
    weights_path = cfg["weights"]
    label_path = cfg.get("label_path",
                         os.path.join(os.path.dirname(weights_path), "label.txt"))
    _threshold = float(cfg.get("threshold", 0.5))
    _imgsz = cfg.get("imgsz")  # [w, h] or null

    _classes = _load_labels(label_path)
    _adult_idx = None
    for i, c in enumerate(_classes):
        if str(c).lower().strip() == "adult":
            _adult_idx = i
            break
    if _adult_idx is None:
        raise RuntimeError(
            f"[detector:{name}] No 'adult' class in {label_path}: {_classes}"
        )

    state_dict = torch.load(weights_path, map_location=_DEVICE, weights_only=False)
    state_dict = {k: v for k, v in state_dict.items() if "fake_quant" not in k}

    model = mobilenet_v3_large(weights=None).to(_DEVICE)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, len(_classes)).to(_DEVICE)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    _model = model
    print(f"[detector:{name}] Ready on {_DEVICE} | classes={_classes} "
          f"| adult_idx={_adult_idx} | threshold={_threshold} | imgsz={_imgsz}")


def detect_adult_slf(frame_bgr, threshold=None):
    if threshold is None:
        threshold = _threshold
    h, w = frame_bgr.shape[:2]

    rgb = frame_bgr[:, :, ::-1].copy()
    pil = Image.fromarray(rgb)
    if _imgsz:
        pil = pil.resize((int(_imgsz[0]), int(_imgsz[1])), Image.BILINEAR)

    tensor = _PREPROCESS(pil).unsqueeze(0).to(_DEVICE)
    with torch.no_grad():
        out = _model(tensor)
    probs = torch.nn.functional.softmax(out[0], dim=0)
    pred_idx = int(torch.argmax(probs).item())
    score = float(probs[pred_idx].item())

    if pred_idx == _adult_idx and score >= threshold:
        return [(0, 0, w, h)]   # full-frame box = "yes, adult SLF"
    return []
