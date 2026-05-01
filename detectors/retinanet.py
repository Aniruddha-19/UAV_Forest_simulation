"""RetinaNet ResNet-50 FPN v2 backend — handles both GroupNorm and BatchNorm variants."""

from functools import partial

import torch
from torchvision import transforms as T
from torchvision.models.detection import retinanet_resnet50_fpn_v2
from torchvision.models.detection.retinanet import (RetinaNetClassificationHead,
                                                     RetinaNetRegressionHead)

CLASSES = ['__background__', 'egg masses', 'instar nymph (1-3)',
           'instar nymph (4)', 'adult', 'Others']
NUM_CLASSES = len(CLASSES)
ADULT_IDX = CLASSES.index('adult')

_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_TRANSFORM = T.Compose([T.ToPILImage(), T.ToTensor()])

_model = None
_threshold = 0.5
_variant = 'gn'


def _create_model(variant: str):
    # weights=None avoids the torchvision auto-download — the trained
    # checkpoint provides all parameters.
    model = retinanet_resnet50_fpn_v2(weights=None, weights_backbone=None)
    num_anchors = model.head.classification_head.num_anchors

    if variant == 'gn':
        norm_layer = partial(torch.nn.GroupNorm, 32)
        model.head.classification_head = RetinaNetClassificationHead(
            in_channels=256, num_anchors=num_anchors,
            num_classes=NUM_CLASSES, norm_layer=norm_layer,
        )
    elif variant == 'bn':
        norm_layer = partial(torch.nn.BatchNorm2d, 256)
        model.head.classification_head = RetinaNetClassificationHead(
            in_channels=256, num_anchors=num_anchors,
            num_classes=NUM_CLASSES, norm_layer=norm_layer,
        )
        model.head.regression_head = RetinaNetRegressionHead(
            in_channels=256, num_anchors=num_anchors,
            norm_layer=norm_layer,
        )
        model.nms_thresh = 0.7
    else:
        raise ValueError(f"unknown retinanet variant: {variant!r}")
    return model


def init(name: str, cfg: dict) -> None:
    global _model, _threshold, _variant
    _variant = 'bn' if name.endswith('_bn') else 'gn'
    weights_path = cfg["weights"]
    _threshold = float(cfg.get("threshold", 0.5))
    print(f"[detector:{name}] Loading {weights_path} (variant={_variant}) ...")
    model = _create_model(_variant)
    ckpt = torch.load(weights_path, map_location=_DEVICE)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.to(_DEVICE).eval()
    _model = model
    print(f"[detector:{name}] Ready on {_DEVICE} | adult_idx={ADULT_IDX} "
          f"| threshold={_threshold}")


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
