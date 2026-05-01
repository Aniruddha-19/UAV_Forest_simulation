"""Faster R-CNN ResNet-50 FPN v2 backend (UAV-Forest-trained, 6 classes incl. background)."""

import torch
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

CLASSES = ['__background__', 'egg masses', 'instar nymph (1-3)',
           'instar nymph (4)', 'adult', 'Others']
NUM_CLASSES = len(CLASSES)
ADULT_IDX = CLASSES.index('adult')

_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_TRANSFORM = T.Compose([T.ToPILImage(), T.ToTensor()])

_model = None
_threshold = 0.5
_imgsz = None


def _create_model():
    # weights=None avoids the torchvision auto-download — the trained
    # checkpoint provides all parameters.
    model = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    return model


def init(name: str, cfg: dict) -> None:
    global _model, _threshold, _imgsz
    weights_path = cfg["weights"]
    _threshold = float(cfg.get("threshold", 0.5))
    _imgsz = cfg.get("imgsz")
    print(f"[detector:{name}] Loading {weights_path} ...")
    model = _create_model()
    ckpt = torch.load(weights_path, map_location=_DEVICE)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.to(_DEVICE).eval()
    if _imgsz:
        model.transform.min_size = (int(_imgsz),)
        model.transform.max_size = int(_imgsz)
    _model = model
    print(f"[detector:{name}] Ready on {_DEVICE} | adult_idx={ADULT_IDX} "
          f"| threshold={_threshold} | imgsz={_imgsz}")


def detect_adult_slf(frame_bgr, threshold=None):
    if threshold is None:
        threshold = _threshold
    rgb = frame_bgr[:, :, ::-1].copy()
    tensor = _TRANSFORM(rgb).unsqueeze(0)
    with torch.no_grad():
        outputs = _model(tensor.to(_DEVICE))
    out = outputs[0]
    boxes  = out["boxes"].cpu().numpy().astype(int)
    scores = out["scores"].cpu().numpy()
    labels = out["labels"].cpu().numpy()
    keep = (scores >= threshold) & (labels == ADULT_IDX)
    return [tuple(int(v) for v in box) for box in boxes[keep]]
