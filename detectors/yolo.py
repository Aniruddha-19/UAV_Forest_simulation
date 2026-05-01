"""YOLO11 / YOLO26 backend via Ultralytics. Class index discovered from model.names."""

from ultralytics import YOLO

_model = None
_threshold = 0.25
_imgsz = 640
_adult_idx = None
_device = 0


def init(name: str, cfg: dict) -> None:
    global _model, _threshold, _imgsz, _adult_idx, _device
    weights_path = cfg["weights"]
    _threshold = float(cfg.get("threshold", 0.25))
    _imgsz = int(cfg.get("imgsz", 640))
    _device = cfg.get("device", 0)

    print(f"[detector:{name}] Loading YOLO weights {weights_path} ...")
    _model = YOLO(weights_path)

    names = _model.names if hasattr(_model, "names") else {}
    _adult_idx = None
    for idx, n in names.items():
        if str(n).lower().strip() == "adult":
            _adult_idx = int(idx)
            break
    if _adult_idx is None:
        raise RuntimeError(
            f"[detector:{name}] No 'adult' class in model.names: {names}"
        )
    print(f"[detector:{name}] Ready | adult_idx={_adult_idx} | "
          f"imgsz={_imgsz} | threshold={_threshold} | device={_device}")


def detect_adult_slf(frame_bgr, threshold=None):
    if threshold is None:
        threshold = _threshold
    results = _model.predict(
        source=frame_bgr,
        imgsz=_imgsz,
        conf=threshold,
        device=_device,
        verbose=False,
    )
    out = []
    for result in results:
        if result.boxes is None or len(result.boxes) == 0:
            continue
        boxes  = result.boxes.xyxy.cpu().numpy().astype(int)
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), cid in zip(boxes, cls_ids):
            if cid == _adult_idx:
                out.append((int(x1), int(y1), int(x2), int(y2)))
    return out
