#!/usr/bin/env python3
"""
simulation.py  —  Entry point
==============================
Wires together all simulation modules and runs the main loop.

Module responsibilities
-----------------------
  environment.py      — PyBullet world setup, spawning trees / egg masses / drone
  drone_controller.py — TRANSIT → INSPECT → HOME state machine
  camera.py           — frame capture, image processing constants
  visualizer.py       — HUD annotation and OpenCV display
  logger.py           — CSV log, frame saving, console heartbeat

Camera / detection split
------------------------
  TRANSIT / DESCEND / ASCEND / HOME
      PyBullet getCameraImage() renders the 3D scene so the operator can
      watch the drone fly.  No R-CNN inference is run.

  INSPECT
      getCameraImage() is skipped entirely.  Instead, the real bug-photo
      assigned to the egg mass the drone is currently facing is passed
      directly to Faster R-CNN.  As the drone orbits, whichever egg mass
      is closest to the camera's forward direction is displayed and
      detected; the image automatically switches as the drone passes each
      panel position.  R-CNN is also run on every other egg mass image for
      the tree so egg_detected is True even if the drone hasn't reached
      a panel yet.

Usage
-----
    python simulation.py
    python simulation.py --config config.json --fps 5

Controls
--------
    q       — quit from the camera-feed window
    Ctrl-C  — interrupt from the terminal
"""

import argparse
import json
import math
import time

import cv2
import numpy as np
import pybullet as p

from camera           import (IMG_W, IMG_H,
                               build_camera_matrices, capture_frame,
                               enhance_frame, get_camera_vectors)
from detector         import detect_egg_masses
from drone_controller import DroneController
from environment      import build_scene, init_world, spawn_drone
from logger           import SimulationLogger
from visualizer       import annotate, close_windows, show


# ── Configuration loader ──────────────────────────────────────────────────────

def load_config(config_path: str, fps_override: int | None = None) -> dict:
    """Load config.json and optionally override the inference frame rate."""
    with open(config_path) as f:
        config = json.load(f)
    if fps_override is not None:
        config["drone"]["inference_fps"] = fps_override
    return config


# ── Startup banner ────────────────────────────────────────────────────────────

def print_banner(config: dict, n_egg_masses: int, log_dir) -> None:
    orbit_spd   = config["drone"].get("inspect_orbit_speed_deg", 20)
    orbit_r     = config["drone"].get("inspect_radius", 3.0)
    trunk_clr   = config["drone"].get("inspect_trunk_clearance", 1.0)
    inspect_alt = config["drone"].get("inspect_altitude", 2.0)
    print("=" * 60)
    print("  UAV Tree Inspection Simulation")
    print("=" * 60)
    print(f"  Trees          : {len(config['trees'])}")
    print(f"  Egg masses     : {n_egg_masses}")
    print(f"  Transit speed  : {config['drone']['transit_speed']} m/s")
    print(f"  Cruise alt     : {config['drone']['cruise_altitude']} m")
    print(f"  Inspect alt    : {inspect_alt} m  (trunk level)")
    print(f"  Canopy radius  : {orbit_r} m  (clearance beyond canopy)")
    print(f"  Trunk clearance: {trunk_clr} m  (orbit radius = trunk_r + {trunk_clr} m)")
    print(f"  Orbit speed    : {orbit_spd} °/s  "
          f"(360° ≈ {360 / orbit_spd:.0f} s / tree)")
    print(f"  Camera FPS     : {config['drone'].get('camera_fps', 30)}")
    print(f"  Inference FPS  : {config['drone']['inference_fps']}")
    print(f"  Logs           : {log_dir}")
    print("  Press 'q' in the camera window to quit.")
    print("=" * 60)


# ── Visibility window & timing limits ────────────────────────────────────────
# Half-angle (radians) within which an egg mass panel is geometrically in FOV.
_EM_VIS_HALF_ANGLE = math.radians(30.0)

# Maximum continuous seconds to show one egg mass image per viewing.
# Once this expires the panel is suppressed even if still inside the angle
# window (important when orbit slows to 5 °/s on detection — without this
# cap the same image would stay on screen for ~12 s).
_EM_MAX_SHOW_SECS = 2.0

# Maximum number of separate viewings of the same panel per orbit.
# Guards against edge cases where the slow orbit re-enters the FOV window.
_EM_MAX_VIEWS = 2

# Medium warm-brown shown when the camera faces bare bark — not too dark so
# the transition from egg-mass image is not a jarring black flash.
_BARK_FRAME = np.full((IMG_H, IMG_W, 3), (70, 90, 110), dtype=np.uint8)


# ── Angle helpers ─────────────────────────────────────────────────────────────

def _angle_diff(a: float, b: float) -> float:
    """Smallest signed angular difference a − b, wrapped to (−π, π]."""
    return math.atan2(math.sin(a - b), math.cos(a - b))


# ── Main simulation loop ──────────────────────────────────────────────────────

def run(config_path: str, fps_override: int | None = None) -> None:
    config = load_config(config_path, fps_override)

    camera_fps         = config["drone"].get("camera_fps", 30)
    inference_fps      = config["drone"]["inference_fps"]
    dt                 = config["simulation"]["time_step"]
    camera_interval    = max(1, round(1.0 / (camera_fps    * dt)))
    inference_interval = max(1, round(1.0 / (inference_fps * dt)))

    # ── Step 1: Initialise world & build scene ────────────────────────────────
    init_world(config)
    egg_masses_data = build_scene(config)          # each entry has tree_id + image_path
    drone_id        = spawn_drone(config["drone"]["start_position"])

    # ── Step 2: Initialise drone controller ───────────────────────────────────
    controller = DroneController(drone_id, config, config["trees"])

    # ── Step 3: Preload egg mass images & compute facing angles ───────────────
    # For each egg mass, record:
    #   angle — the bearing from the trunk centre to the egg mass panel
    #           (used to pick which image to show based on where drone faces)
    #   bgr   — the real bug photo resized to IMG_W × IMG_H, ready for R-CNN
    #
    # Images are read once at startup; no disk I/O inside the main loop.
    trunk_pos_map = {t["id"]: t["position"] for t in config["trees"]}

    # tree_egg_data[tree_id] = list of {angle, bgr, id} dicts, one per egg mass
    tree_egg_data: dict[str, list[dict]] = {}
    for em in egg_masses_data:
        tp    = trunk_pos_map[em["tree_id"]]
        ex, ey = em["position"][0], em["position"][1]
        angle  = math.atan2(ey - tp[1], ex - tp[0])  # bearing from trunk to panel

        bgr = None
        path = em.get("image_path")
        if path:
            raw = cv2.imread(path)
            if raw is not None:
                bgr = cv2.resize(raw, (IMG_W, IMG_H))
            else:
                print(f"[WARNING] Cannot load egg mass image: {path}")

        tree_egg_data.setdefault(em["tree_id"], []).append({
            "id":    em["id"],
            "angle": angle,
            "bgr":   bgr,
        })

    loaded = sum(1 for ems in tree_egg_data.values()
                 for em in ems if em["bgr"] is not None)
    print(f"[sim] Preloaded {loaded} egg-mass image(s) across "
          f"{len(tree_egg_data)} tree(s).")

    # ── Step 4: Open logger ───────────────────────────────────────────────────
    with SimulationLogger() as log:
        n_egg_masses = sum(len(t.get("egg_masses", [])) for t in config["trees"])
        print_banner(config, n_egg_masses, log.run_dir.resolve())

        physics_step = 0

        # ── Per-orbit egg mass visibility tracking ────────────────────────────
        # Tracking only advances at inference steps — the model never "saw"
        # the frames that were missed while it was busy.
        _inspect_tree_id: str | None       = None
        _em_first_seen:   dict[str, float] = {}
        _em_view_count:   dict[str, int]   = {}
        _em_geo_in_view:  set[str]         = set()

        # ── Camera / inference split state ────────────────────────────────────
        # latest_* is written at every camera step and read at inference steps.
        latest_frame      : np.ndarray  = _BARK_FRAME
        latest_geo_ids    : set[str]    = set()
        latest_facing     : float       = 0.0
        latest_is_inspect : bool        = False

        cam_frames_since_infer: int = 0   # camera frames since last inference
        total_missed_frames   : int = 0   # cumulative dropped frames

        try:
            # ── Main loop ─────────────────────────────────────────────────────
            while not controller.done:

                _step_deadline = time.perf_counter() + dt

                controller.step()
                p.stepSimulation()
                physics_step += 1

                # ── Camera step: capture latest frame (no inference) ──────────
                if physics_step % camera_interval == 0:
                    cam_frames_since_infer += 1
                    _cam_pos, _cam_orn = controller.pose()

                    if (controller.state == DroneController.INSPECT
                            and controller.current_tree is not None):

                        tree_id  = controller.current_tree["id"]
                        egg_data = tree_egg_data.get(tree_id, [])

                        if tree_id != _inspect_tree_id:
                            _inspect_tree_id = tree_id
                            _em_first_seen.clear()
                            _em_view_count.clear()
                            _em_geo_in_view.clear()
                            latest_geo_ids = set()

                        facing             = controller.current_yaw + math.pi
                        latest_facing      = facing
                        latest_is_inspect  = True

                        if egg_data:
                            geo_ids = {
                                egg_data[i]["id"]
                                for i in range(len(egg_data))
                                if abs(_angle_diff(egg_data[i]["angle"], facing))
                                <= _EM_VIS_HALF_ANGLE
                            }
                            latest_geo_ids = geo_ids
                            if geo_ids:
                                best_em = min(
                                    (em for em in egg_data
                                     if em["id"] in geo_ids),
                                    key=lambda em: abs(
                                        _angle_diff(em["angle"], facing)))
                                latest_frame = (best_em["bgr"]
                                                if best_em["bgr"] is not None
                                                else _BARK_FRAME)
                            else:
                                latest_frame   = _BARK_FRAME
                                latest_geo_ids = set()
                        else:
                            latest_frame   = _BARK_FRAME
                            latest_geo_ids = set()

                    else:
                        latest_is_inspect = False
                        fwd     = np.array([math.cos(controller.current_yaw),
                                            math.sin(controller.current_yaw),
                                            0.0])
                        look_at = _cam_pos + fwd * 10.0
                        cam_fwd, cam_up = get_camera_vectors(
                            _cam_orn, _cam_pos, look_at)
                        view, proj   = build_camera_matrices(
                            _cam_pos, cam_fwd, cam_up)
                        latest_frame = enhance_frame(capture_frame(view, proj))
                        latest_geo_ids = set()

                # ── Inference step: R-CNN on the latest captured frame ─────────
                if physics_step % inference_interval == 0:
                    # Frames captured since last inference that weren't processed
                    missed = max(0, cam_frames_since_infer - 1)
                    total_missed_frames    += missed
                    cam_frames_since_infer  = 0

                    drone_pos, _ = controller.pose()
                    rcnn_boxes   = []

                    if (latest_is_inspect
                            and controller.state == DroneController.INSPECT
                            and controller.current_tree is not None
                            and controller.current_tree["id"] == _inspect_tree_id):

                        egg_data = tree_egg_data.get(_inspect_tree_id, [])
                        now      = time.perf_counter()

                        # Step B: detect panel FOV entries at inference granularity.
                        # Panels visible only during missed frames are never counted —
                        # the model never processed those frames.
                        for em_id in latest_geo_ids:
                            if em_id not in _em_geo_in_view:
                                _em_view_count[em_id] = \
                                    _em_view_count.get(em_id, 0) + 1
                                _em_first_seen[em_id] = now
                        _em_geo_in_view = set(latest_geo_ids)

                        # Step C: apply time & count limits
                        in_view = [
                            (i, em)
                            for i, em in enumerate(egg_data)
                            if em["id"] in latest_geo_ids
                            and _em_view_count.get(em["id"], 0) <= _EM_MAX_VIEWS
                            and now - _em_first_seen.get(em["id"], now)
                                <= _EM_MAX_SHOW_SECS
                        ]

                        any_detected = False
                        if in_view:
                            best_i = min(
                                in_view,
                                key=lambda t: abs(
                                    _angle_diff(t[1]["angle"], latest_facing))
                            )[0]
                            for i, em in in_view:
                                if em["bgr"] is None:
                                    continue
                                boxes = detect_egg_masses(em["bgr"])
                                if i == best_i:
                                    rcnn_boxes = boxes
                                if boxes:
                                    any_detected = True

                        controller.egg_detected = any_detected

                    else:
                        controller.egg_detected = False

                    if rcnn_boxes:
                        log.log_rcnn_detection(
                            len(rcnn_boxes), drone_pos, controller.state)

                    # ── Annotate ──────────────────────────────────────────────
                    tree       = controller.current_tree
                    tree_label = tree["id"] if tree else "—"
                    tree_idx   = min(controller.tree_idx + 1,
                                     len(config["trees"]))
                    orbit_pct  = (controller.orbit_progress
                                  if controller.state == DroneController.INSPECT
                                  else 0.0)

                    annotated = annotate(
                        frame        = latest_frame,
                        rcnn_boxes   = rcnn_boxes,
                        drone_pos    = drone_pos,
                        state        = controller.state,
                        tree_label   = tree_label,
                        tree_idx     = tree_idx,
                        total_trees  = len(config["trees"]),
                        orbit_pct    = orbit_pct,
                        orbit_radius = controller.orbit_radius,
                        frame_no     = log.frame_no,
                        saved_no     = log.saved_no,
                        fps          = inference_fps,
                    )

                    if rcnn_boxes:
                        log.save_frame(annotated)

                    if show(annotated):
                        print("\nStopped by user (q).")
                        break

                    log.print_heartbeat(
                        drone_pos, controller.state, tree_label,
                        len(rcnn_boxes), inference_fps)

                    log.tick_frame()

                _remaining = _step_deadline - time.perf_counter()
                if _remaining > 0:
                    time.sleep(_remaining)

        except KeyboardInterrupt:
            print("\nInterrupted (Ctrl-C).")

        finally:
            close_windows()
            p.disconnect()
            log.print_summary(missed_frames=total_missed_frames,
                               sim_time=physics_step * dt)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="UAV Tree Inspection — orbit each tree, save on detection")
    ap.add_argument(
        "--config", default="config.json",
        help="Path to the JSON environment config  (default: config.json)")
    ap.add_argument(
        "--fps", type=int, default=None,
        help="Override inference frame rate in frames-per-second")
    args = ap.parse_args()
    run(args.config, args.fps)
