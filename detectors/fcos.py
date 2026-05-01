"""
FCOS backend (6 classes incl. background).

Note: the trained checkpoint was produced with a ResNet-101 backbone, not
ResNet-50, so we reconstruct the architecture using torchvision's FPN
extractor on a ResNet-101 trunk to get an exact state_dict match. weights=None
on every component avoids any runtime download — the trained checkpoint
provides every parameter.
"""

from functools import partial

import torch
import torchvision
from torchvision import transforms as T
from torchvision.models.detection import FCOS
from torchvision.models.detection.fcos import FCOSClassificationHead
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.ops import misc as misc_nn_ops
from torchvision.ops.feature_pyramid_network import LastLevelP6P7

CLASSES = ['__background__', 'egg masses', 'instar nymph (1-3)',
           'instar nymph (4)', 'adult', 'Others']
NUM_CLASSES = len(CLASSES)
ADULT_IDX = CLASSES.index('adult')

_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_TRANSFORM = T.Compose([T.ToPILImage(), T.ToTensor()])

_model = None
_threshold = 0.5


def _create_model(min_size: int = 640, max_size: int = 640):
    norm_layer = misc_nn_ops.FrozenBatchNorm2d
    backbone = torchvision.models.resnet101(weights=None, norm_layer=norm_layer)
    backbone = _resnet_fpn_extractor(
        backbone, 3, returned_layers=[2, 3, 4],
        extra_blocks=LastLevelP6P7(256, 256),
    )
    model = FCOS(backbone, num_classes=NUM_CLASSES,
                 min_size=min_size, max_size=max_size)
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head = FCOSClassificationHead(
        in_channels=256,
        num_anchors=num_anchors,
        num_classes=NUM_CLASSES,
        norm_layer=partial(torch.nn.GroupNorm, 32),
    )
    return model


def init(name: str, cfg: dict) -> None:
    global _model, _threshold
    weights_path = cfg["weights"]
    _threshold = float(cfg.get("threshold", 0.5))
    imgsz = int(cfg.get("imgsz") or 640)
    print(f"[detector:{name}] Loading {weights_path} ...")
    model = _create_model(min_size=imgsz, max_size=imgsz)
    ckpt = torch.load(weights_path, map_location=_DEVICE, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.to(_DEVICE).eval()
    _model = model
    print(f"[detector:{name}] Ready on {_DEVICE} | adult_idx={ADULT_IDX} "
          f"| threshold={_threshold} | imgsz={imgsz}")


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
