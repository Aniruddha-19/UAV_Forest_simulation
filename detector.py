"""
detector.py
===========
Backend dispatcher. Loads whichever backend is named by
``config['detector']['active_model']`` and routes detect_adult_slf() through
it.

Each backend lives in detectors/<name>.py and exposes:
    init(name: str, cfg: dict) -> None
    detect_adult_slf(frame_bgr, threshold=None) -> list[(x1,y1,x2,y2)]

simulation.py calls init_detector(config) once, after loading the JSON
config, and then uses detect_adult_slf() / get_inference_wait_s() as before.
"""

import os

import numpy as np

from detectors import load_backend

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

_backend = None
_active_model = None
_threshold = 0.5
_inference_wait_s = 0.0


def _resolve_paths(cfg: dict, root: str) -> dict:
    """Resolve relative weights/label_path entries against the project root."""
    out = dict(cfg)
    for key in ("weights", "label_path"):
        v = out.get(key)
        if v and not os.path.isabs(v):
            out[key] = os.path.join(root, v)
    return out


def init_detector(config: dict) -> None:
    """Initialize the active backend named in config['detector']."""
    global _backend, _active_model, _threshold, _inference_wait_s

    det_cfg = config["detector"]
    active = det_cfg["active_model"]
    model_cfg = _resolve_paths(det_cfg["models"][active], _SCRIPT_DIR)

    print(f"[detector] Active model: {active}")
    _active_model = active
    _backend = load_backend(active, model_cfg)
    _threshold = float(model_cfg.get("threshold", 0.5))
    _inference_wait_s = float(model_cfg.get("inference_wait_seconds", 0.0))


def detect_adult_slf(frame_bgr: np.ndarray,
                     threshold: float | None = None
                     ) -> list:
    if _backend is None:
        raise RuntimeError(
            "detector not initialized — call init_detector(config) first")
    return _backend.detect_adult_slf(frame_bgr, threshold)


def get_active_model() -> str | None:
    return _active_model


def get_threshold() -> float:
    return _threshold


def get_inference_wait_s() -> float:
    return _inference_wait_s
