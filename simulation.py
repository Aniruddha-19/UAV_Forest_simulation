#!/usr/bin/env python3
"""
simulation.py  —  Entry point
==============================
Wires together all simulation modules and runs the main loop.

Module responsibilities
-----------------------
  environment.py      — PyBullet world setup, spawning trees / adult SLF panels / drone
  drone_controller.py — TRANSIT → INSPECT → HOME state machine
  camera.py           — frame capture, image processing constants
  visualizer.py       — HUD annotation and OpenCV display
  logger.py           — CSV log, frame saving, console heartbeat

Camera feed
-----------
  All 100 adult SLF images from simulation_test_data/ are loaded at startup,
  randomly shuffled, and resized to config["simulation"]["camera_resolution"].
  They form a looping 30-fps video feed (FEED_FPS).

  TRANSIT / DESCEND / ASCEND / HOME
      PyBullet getCameraImage() renders the 3D scene (top-down drone view).
      The feed advances in the background but inference is not run.

  INSPECT
      The feed plays.  On each camera step the current feed frame is grabbed
      and fed to the model.  Because inference blocks the loop, the feed
      automatically advances during model execution — this is the
      "waiting-time" pattern: inference time IS the skip interval.
      No frames between consecutive inferences are ever processed.

Usage
-----
    python simulation.py
    python simulation.py --config config.json

Controls
--------
    q       — quit from the camera-feed window
    Ctrl-C  — interrupt from the terminal
"""

import argparse
import glob
import json
import math
import os
import random
import time

import cv2
import numpy as np
import pybullet as p

from camera           import (IMG_W, IMG_H,
                               build_camera_matrices, capture_frame,
                               enhance_frame, get_camera_vectors)
from detector         import detect_adult_slf
from drone_controller import DroneController
from environment      import build_scene, init_world, spawn_drone
from logger           import SimulationLogger
from visualizer       import annotate, close_windows, show


# Feed frame rate — constant, independent of physics timestep
FEED_FPS: float = 30.0


# ── Configuration loader ──────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return json.load(f)


# ── Feed helpers ──────────────────────────────────────────────────────────────

def _load_feed(test_dir: str, cam_w: int, cam_h: int) -> list[np.ndarray]:
    """
    Load every image from *test_dir*, shuffle randomly, resize to
    (cam_w, cam_h) and return as a list ready for time-based playback.
    """
    paths = sorted(
        f for ext in ("*.jpg", "*.jpeg", "*.png")
        for f in glob.glob(os.path.join(test_dir, ext))
    )
    random.shuffle(paths)
    frames = []
    for path in paths:
        img = cv2.imread(path)
        if img is not None:
            frames.append(cv2.resize(img, (cam_w, cam_h)))
    return frames


def _current_feed_frame(feed_frames: list[np.ndarray],
                         feed_start: float) -> np.ndarray:
    """Return the frame that is 'live' right now based on wall-clock time."""
    idx = int((time.perf_counter() - feed_start) * FEED_FPS) % len(feed_frames)
    return feed_frames[idx]


# ── Startup banner ────────────────────────────────────────────────────────────

def print_banner(config: dict, n_panels: int, n_feed: int,
                 cam_w: int, cam_h: int, log_dir) -> None:
    orbit_spd   = config["drone"].get("inspect_orbit_speed_deg", 20)
    orbit_r     = config["drone"].get("inspect_radius", 3.0)
    trunk_clr   = config["drone"].get("inspect_trunk_clearance", 1.0)
    inspect_alt = config["drone"].get("inspect_altitude", 2.0)
    print("=" * 60)
    print("  UAV Tree Inspection Simulation  —  Adult SLF Detection")
    print("=" * 60)
    print(f"  Trees          : {len(config['trees'])}")
    print(f"  3D panels      : {n_panels}")
    print(f"  Feed images    : {n_feed}  (randomly shuffled, looping)")
    print(f"  Feed FPS       : {FEED_FPS:.0f} fps")
    print(f"  Resolution     : {cam_w} × {cam_h}")
    print(f"  Transit speed  : {config['drone']['transit_speed']} m/s")
    print(f"  Cruise alt     : {config['drone']['cruise_altitude']} m")
    print(f"  Inspect alt    : {inspect_alt} m")
    print(f"  Canopy radius  : {orbit_r} m")
    print(f"  Trunk clearance: {trunk_clr} m")
    print(f"  Orbit speed    : {orbit_spd} °/s  "
          f"(360° ≈ {360 / orbit_spd:.0f} s / tree)")
    print(f"  Inference      : waiting-time pattern (no fixed FPS)")
    print(f"  Logs           : {log_dir}")
    print("  Press 'q' in the camera window to quit.")
    print("=" * 60)


# ── Main simulation loop ──────────────────────────────────────────────────────

def run(config_path: str) -> None:
    config   = load_config(config_path)
    dt       = config["simulation"]["time_step"]
    cam_res  = config["simulation"].get("camera_resolution", [640, 480])
    cam_w, cam_h = int(cam_res[0]), int(cam_res[1])
    camera_fps      = config["drone"].get("camera_fps", 30)
    camera_interval = max(1, round(1.0 / (camera_fps * dt)))

    # ── Step 1: Load camera feed ──────────────────────────────────────────────
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _test_dir   = os.path.join(_script_dir, "simulation_test_data")
    feed_frames = _load_feed(_test_dir, cam_w, cam_h)
    if not feed_frames:
        print("[sim] WARNING: no feed images found — inspection display will be blank.")
        feed_frames = [np.zeros((cam_h, cam_w, 3), dtype=np.uint8)]
    print(f"[sim] Feed: {len(feed_frames)} frames @ {cam_w}×{cam_h}, {FEED_FPS:.0f} fps")

    # ── Step 2: Initialise world & build scene ────────────────────────────────
    init_world(config)
    slf_data = build_scene(config)     # spawns 3D panels; image paths not used here
    drone_id = spawn_drone(config["drone"]["start_position"])

    # ── Step 3: Initialise drone controller ───────────────────────────────────
    controller = DroneController(drone_id, config, config["trees"])

    # ── Step 4: Open logger & run ─────────────────────────────────────────────
    with SimulationLogger() as log:
        n_panels = len(slf_data)
        print_banner(config, n_panels, len(feed_frames), cam_w, cam_h,
                     log.run_dir.resolve())

        physics_step = 0
        latest_frame : np.ndarray = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)

        # Feed timing — reset each time the drone enters INSPECT
        _feed_start: float | None = None

        # Running inference statistics
        _infer_count  = 0
        _detect_count = 0

        try:
            while not controller.done:

                _step_deadline = time.perf_counter() + dt

                controller.step()
                p.stepSimulation()
                physics_step += 1

                # ── Combined camera + inference step ──────────────────────────
                if physics_step % camera_interval == 0:

                    drone_pos, drone_orn = controller.pose()
                    rcnn_boxes: list = []

                    if controller.state == DroneController.INSPECT:
                        # ── INSPECT: 30-fps feed + waiting-time inference ──────
                        if _feed_start is None:
                            _feed_start = time.perf_counter()

                        latest_frame = _current_feed_frame(feed_frames, _feed_start)

                        # Run inference — this BLOCKS for T seconds (the waiting time).
                        # The feed index advances automatically during that time, so
                        # the next iteration picks up the frame that is current when
                        # the model finishes.  All frames in between are skipped.
                        rcnn_boxes = detect_adult_slf(latest_frame)
                        _infer_count += 1
                        if rcnn_boxes:
                            _detect_count += 1

                        controller.slf_detected = bool(rcnn_boxes)

                    else:
                        # ── TRANSIT / HOME: PyBullet render, no inference ──────
                        _feed_start = None          # reset so next INSPECT starts fresh
                        controller.slf_detected = False

                        fwd     = np.array([math.cos(controller.current_yaw),
                                            math.sin(controller.current_yaw),
                                            0.0])
                        look_at = drone_pos + fwd * 10.0
                        cam_fwd, cam_up = get_camera_vectors(
                            drone_orn, drone_pos, look_at)
                        view, proj = build_camera_matrices(
                            drone_pos, cam_fwd, cam_up)
                        latest_frame = enhance_frame(capture_frame(view, proj))

                    # ── Log + annotate + display ──────────────────────────────
                    if rcnn_boxes:
                        log.log_rcnn_detection(
                            len(rcnn_boxes), drone_pos, controller.state)

                    tree       = controller.current_tree
                    tree_label = tree["id"] if tree else "—"
                    tree_idx   = min(controller.tree_idx + 1, len(config["trees"]))
                    orbit_pct  = (controller.orbit_progress
                                  if controller.state == DroneController.INSPECT
                                  else 0.0)
                    det_pct    = (_detect_count / _infer_count * 100
                                  if _infer_count else 0.0)

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
                        detect_pct   = det_pct,
                        infer_count  = _infer_count,
                    )

                    if rcnn_boxes:
                        log.save_frame(annotated)

                    if show(annotated):
                        print("\nStopped by user (q).")
                        break

                    log.print_heartbeat(
                        drone_pos, controller.state, tree_label,
                        len(rcnn_boxes), _infer_count, det_pct)

                    log.tick_frame()

                _remaining = _step_deadline - time.perf_counter()
                if _remaining > 0:
                    time.sleep(_remaining)

        except KeyboardInterrupt:
            print("\nInterrupted (Ctrl-C).")

        finally:
            close_windows()
            p.disconnect()
            log.print_summary(sim_time=physics_step * dt,
                               infer_count=_infer_count,
                               detect_count=_detect_count)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="UAV Tree Inspection — adult SLF detection")
    ap.add_argument(
        "--config", default="config.json",
        help="Path to the JSON config  (default: config.json)")
    args = ap.parse_args()
    run(args.config)
