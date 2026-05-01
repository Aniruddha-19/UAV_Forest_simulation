"""
detectors package
=================
Each module in this package implements one detection backend behind a common
interface:

    init(name: str, cfg: dict) -> None
    detect_adult_slf(frame_bgr, threshold=None) -> list[(x1,y1,x2,y2)]

The dispatcher in detector.py picks the active backend by name from
config.json. New backends are added by registering them in _BACKENDS.
"""

import importlib

_BACKENDS = {
    "frcnn":         "detectors.frcnn",
    "fcos":          "detectors.fcos",
    "retinanet_gn":  "detectors.retinanet",
    "retinanet_bn":  "detectors.retinanet",
    "yolo11":        "detectors.yolo",
    "yolo26":        "detectors.yolo",
    "rfdetr":        "detectors.rfdetr",
    "mobilenetv3":   "detectors.mobilenetv3",
}


def load_backend(model_name: str, model_config: dict):
    """Import and initialize the backend module for *model_name*."""
    if model_name not in _BACKENDS:
        raise ValueError(
            f"Unknown detector model '{model_name}'. "
            f"Available: {sorted(_BACKENDS)}"
        )
    module = importlib.import_module(_BACKENDS[model_name])
    module.init(model_name, model_config)
    return module


def available_models() -> list[str]:
    return sorted(_BACKENDS)
