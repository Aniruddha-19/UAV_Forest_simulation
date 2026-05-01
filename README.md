# UAV Forest Simulation — Adult SLF Detection

A PyBullet-based simulation in which a UAV inspects nine trees, hovers and
orbits each one with the camera trained on the trunk, and runs a detector
on the camera feed to count adult Spotted Lanternfly (SLF). Eight detector
backends are supported and switchable from a single line in
`config.json`.

---

## Quickstart

```bash
# 1. Activate the Python environment that has torch / pybullet / cv2 / ultralytics / rfdetr
source /home/aniruddha/spotLF/bin/activate

# 2. Run the simulation (uses the active detector named in config.json)
python simulation.py
```

Two OpenCV windows open:

- **Camera Feed (30 fps)** — what the drone is looking at right now.
- **Model Output** — the most recent inference result. During TRANSIT it
  shows a "Transiting…" placeholder; the previous tree's frame is not
  held over.

Press **q** in either window to quit.

---

## Switching the detector

Edit `config.json` → `detector.active_model`. Available names:

| name             | architecture                          | adult class idx |
|------------------|---------------------------------------|-----------------|
| `frcnn`          | Faster R-CNN ResNet-50 FPN v2         | 4 (incl. bg)    |
| `fcos`           | FCOS, ResNet-101 backbone             | 4 (incl. bg)    |
| `retinanet_gn`   | RetinaNet ResNet-50 FPN v2 (GroupNorm)| 4 (incl. bg)    |
| `retinanet_bn`   | RetinaNet ResNet-50 FPN v2 (BatchNorm)| 4 (incl. bg)    |
| `yolo11`         | Ultralytics YOLO11                    | 3 (auto-detect) |
| `yolo26`         | Ultralytics YOLO26                    | 3 (auto-detect) |
| `rfdetr`         | Roboflow RF-DETR Large (91-class head)| 3               |
| `mobilenetv3`    | MobileNet V3 Large classifier (5-cls) | 3 (yes/no)      |

Per-model knobs live under `detector.models.<name>`:

```json
"yolo11": {
  "weights": "models/yolo/yolo11.pt",
  "threshold": 0.25,
  "imgsz": 640,
  "device": 0,
  "inference_wait_seconds": 1.0
}
```

- **`weights`** — path to the checkpoint (relative paths resolve to the
  project root).
- **`threshold`** — confidence threshold for a detection to count.
- **`imgsz`** — model input size. Integer for detectors, `[w, h]` for
  MobileNet's pre-resize, `null` to skip.
- **`device`** — `0` for `cuda:0`, `"cpu"` for CPU. (YOLO only.)
- **`inference_wait_seconds`** — minimum seconds per inference cycle. The
  worker pads with `time.sleep` if real GPU inference finishes faster, so
  the cadence is independent of GPU speed. Set to `0` for GPU-paced.

MobileNet V3 is a classifier, not a detector. The backend wraps it: if
softmax says "adult" above threshold it returns one full-frame box; the
simulation's per-frame "yes/no" then becomes the percentage of frames
classified as adult SLF.

---

## Project layout

```
simulation.py          # entry point: physics loop + inference dispatch
camera.py              # PyBullet camera capture + image enhancement
detector.py            # thin dispatcher; init_detector(config) loads the chosen backend
detectors/             # one module per backend
  frcnn.py  fcos.py  retinanet.py  yolo.py  rfdetr.py  mobilenetv3.py
drone_controller.py    # TRANSIT / INSPECT / HOME state machine
environment.py         # ground / trees / SLF panels in PyBullet
logger.py              # detections.csv + saved frames
visualizer.py          # HUD overlay
config.json            # all tuning lives here
models/                # all weights — copy here, never auto-downloaded
faster-rcnn-model/     # legacy training scripts (eval_per_tree.py, eval_detection_rate.py)
simulation_test_data/  # 300 SLF panel images used as textures + offline eval input
logs/<timestamp>/      # per-run output: detections.csv + frames/
```

---

## Tuning runtime behaviour

Most knobs live in `config.json`. Common edits:

- **Drone speed / orbit**: `drone.transit_speed`, `drone.inspect_orbit_speed_deg`.
- **Camera resolution**: `simulation.camera_resolution`.
- **Per-tree topology**: `trees[].position` / `height` / `radius`.
- **Detector pacing**: see *Switching the detector* above.

Every run writes:

- `logs/<YYYYMMDD_HHMMSS>/detections.csv` — one row per inference with detections.
- `logs/<YYYYMMDD_HHMMSS>/frames/frame_NNNNNN.jpg` — the annotated *Model
  Output* frame, saved **once per unique input frame that produced a
  detection** (consecutive identical inputs are not re-saved).

---

## Offline evaluation scripts

Two helpers under `faster-rcnn-model/` run Faster R-CNN inference against
`simulation_test_data/` directly (no PyBullet involved):

```bash
# Per-tree percentage report (treats 300 images as one 10s @ 30fps video,
# replays for each of N trees, averages):
python faster-rcnn-model/eval_per_tree.py \
  --threshold 0.5 --target_class adult --num_trees 9

# Single-pass detection-rate sweep:
python faster-rcnn-model/eval_detection_rate.py --threshold 0.5
```

Both expect the FRCNN weights at `faster-rcnn-model/outputs/fasterrcnn.pth`.

---

## Texture cache

PyBullet's `loadTexture` rejects some JPEG variants (progressive, CMYK,
unusual color profiles). On startup, every `simulation_test_data/*.jpg`
referenced by a panel is re-encoded as a baseline RGB JPEG into
`simulation_test_data/.pybullet_tex_cache/`. The conversion runs once per
image; delete the cache to force a refresh.

---

## Notes on RF-DETR

The RF-DETR backend was fine-tuned with the 91-class COCO head intact
(only indices 0–4 carry SLF training: egg masses / instar 1-3 / instar 4
/ adult / Others). Indices 5–90 still hold the pretrained COCO weights.
Only `adult` (index 3) is counted, but you may occasionally see stray
predictions from other class slots if you ever switch to logging *all*
classes. The first call to `RFDETRLarge(...)` downloads the pretrained
backbone (`rf-detr-large-2026.pth`) into the working directory; this is
inside the upstream library and not currently suppressible.
