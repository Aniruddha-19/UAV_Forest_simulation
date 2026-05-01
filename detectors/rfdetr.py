"""
RF-DETR Large backend (Roboflow).

The user fine-tuned the 91-class COCO-pretrained RF-DETR on the SLF dataset
without replacing the classification head, so the checkpoint keeps 91 output
slots but only the first 5 indices were trained for SLF classes:
    0=egg masses, 1=instar nymph (1-3), 2=instar nymph (4), 3=adult, 4=Others
The remaining indices (5..90) carry the original COCO weights and should not
fire on this dataset in practice.
"""

import torch

CLASSES = ['egg masses', 'instar nymph (1-3)', 'instar nymph (4)', 'adult', 'Others']
NUM_CLASSES_TRAINED = len(CLASSES)
NUM_CLASSES_MODEL = 91   # the checkpoint's output head width
ADULT_IDX = CLASSES.index('adult')

_model = None
_threshold = 0.4


def init(name: str, cfg: dict) -> None:
    global _model, _threshold
    from rfdetr import RFDETRLarge   # heavy import; load on demand
    weights_path = cfg["weights"]
    _threshold = float(cfg.get("threshold", 0.4))

    print(f"[detector:{name}] Loading RF-DETR {weights_path} ...")
    _model = RFDETRLarge(num_classes=NUM_CLASSES_MODEL)

    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    _model.model.model.load_state_dict(state_dict, strict=True)
    _model.model.model.eval()
    print(f"[detector:{name}] Ready | adult_idx={ADULT_IDX} (in 91-class output) "
          f"| threshold={_threshold}")


def detect_adult_slf(frame_bgr, threshold=None):
    if threshold is None:
        threshold = _threshold
    rgb = frame_bgr[:, :, ::-1].copy()
    detections = _model.predict(rgb, threshold=threshold)
    if detections is None or len(detections) == 0:
        return []
    out = []
    for box, _score, cid in zip(detections.xyxy,
                                detections.confidence,
                                detections.class_id):
        if int(cid) == ADULT_IDX:
            x1, y1, x2, y2 = map(int, box)
            out.append((x1, y1, x2, y2))
    return out
