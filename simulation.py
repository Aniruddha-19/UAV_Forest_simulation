#!/usr/bin/env python3
"""
simulation.py  —  Entry point
==============================

Architecture
------------
Main thread  (100 Hz physics loop)
  • controller.step()   — moves the drone each tick
  • p.stepSimulation()  — advances the PyBullet 3D world
  • Every 3rd tick (30 fps):
      Window 1 "Camera Feed"  — what the drone's camera sees right now
      Window 2 "Model Output" — latest inference result

  During INSPECT the camera shows the image of whichever SLF panel the
  drone is currently facing (within ±VIS_HALF_ANGLE degrees).  When the
  drone faces bare bark between panels it sees a plain bark frame.  Not
  every orbit direction has a panel, so the camera sees SLF images only
  part of the time — matching the real scenario where some trees are more
  infested than others.

Inference thread  (daemon)
  • Waits for a frame from the main thread.
  • Runs detect_adult_slf() — takes T seconds (the natural waiting time).
  • Returns (frame, boxes).
  • Main thread sends the next frame only when the worker is free, so all
    frames that arrived during inference are automatically skipped.

Panel distribution
  100 images from simulation_test_data/ are assigned by build_scene():
    tree_1  → 12 panels
    tree_2 … tree_9  → 11 panels each
  Each panel sits at a specific angle and height on its trunk.

Usage
-----
    python simulation.py
    python simulation.py --config config.json
Controls: press 'q' in either window to quit.
"""

import argparse
import glob
import json
import math
import os
import queue
import threading
import time

import cv2
import numpy as np
import pybullet as p

from camera           import (IMG_W, IMG_H,
                               build_camera_matrices, capture_frame,
                               enhance_frame, get_camera_vectors)
from detector         import (detect_adult_slf, get_active_model,
                              get_inference_wait_s, init_detector)
from drone_controller import DroneController
from environment      import build_scene, init_world, spawn_drone
from logger           import SimulationLogger
from visualizer       import annotate, close_windows


# ── Constants ─────────────────────────────────────────────────────────────────

CAMERA_FPS      = 30.0
VIDEO_FPS       = 30.0
WINDOW_FEED     = "Camera Feed  (30 fps)"
WINDOW_MODEL    = "Model Output"

_TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "simulation_test_data")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_video_frames(images_dir: str,
                       cam_w: int, cam_h: int) -> list[np.ndarray]:
    """
    Load every image in *images_dir* (sorted) as a list of BGR ndarrays
    resized to (cam_w, cam_h). Used as the looping camera feed during
    INSPECT — 300 images at 30 fps = a 10 s clip.
    """
    paths = sorted(
        f for ext in ("*.png", "*.jpg", "*.jpeg")
        for f in glob.glob(os.path.join(images_dir, ext))
    )
    frames: list[np.ndarray] = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        if img.shape[:2] != (cam_h, cam_w):
            img = cv2.resize(img, (cam_w, cam_h))
        frames.append(img)
    return frames


# ── Inference thread ──────────────────────────────────────────────────────────

def _inference_worker(frame_q: queue.Queue,
                      result_q: queue.Queue,
                      min_wait_s: float = 0.0) -> None:
    """
    Daemon: block on frame_q, run the model, push (frame, boxes) to result_q.
    Receiving None is the stop signal.

    If *min_wait_s* > 0, the worker pads each inference cycle to that
    minimum duration — useful when you want a fixed inference cadence
    independent of GPU speed.
    """
    while True:
        frame = frame_q.get()
        if frame is None:
            break
        t0 = time.perf_counter()
        boxes = detect_adult_slf(frame)
        elapsed = time.perf_counter() - t0
        if min_wait_s > 0.0 and elapsed < min_wait_s:
            time.sleep(min_wait_s - elapsed)
        # Discard any unconsumed previous result
        try:
            result_q.get_nowait()
        except queue.Empty:
            pass
        result_q.put((frame, boxes))


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ── Banner ────────────────────────────────────────────────────────────────────

def print_banner(config: dict, n_panels: int,
                 cam_w: int, cam_h: int, log_dir) -> None:
    orbit_spd   = config["drone"].get("inspect_orbit_speed_deg", 20)
    inspect_alt = config["drone"].get("inspect_altitude", 2.0)
    trunk_clr   = config["drone"].get("inspect_trunk_clearance", 1.0)
    print("=" * 60)
    print("  UAV Tree Inspection  —  Adult SLF Detection")
    print("=" * 60)
    print(f"  Trees          : {len(config['trees'])}")
    print(f"  SLF panels     : {n_panels}  (scenery only — camera plays video)")
    print(f"  Resolution     : {cam_w} × {cam_h}")
    print(f"  Camera FPS     : {CAMERA_FPS:.0f}")
    print(f"  Inspect alt    : {inspect_alt} m")
    print(f"  Trunk clearance: {trunk_clr} m")
    print(f"  Orbit speed    : {orbit_spd} °/s  "
          f"(360° ≈ {360/orbit_spd:.0f} s / tree)")
    wait_s = get_inference_wait_s()
    model_label = get_active_model() or "?"
    if wait_s > 0:
        print(f"  Detector       : {model_label}  (waiting-time = {wait_s:.2f} s)")
    else:
        print(f"  Detector       : {model_label}  (waiting-time = GPU-paced)")
    print(f"  Window         : {WINDOW_MODEL!r}")
    print(f"  Logs           : {log_dir}")
    print("  Press 'q' in either window to quit.")
    print("=" * 60)


# ── Main loop ─────────────────────────────────────────────────────────────────

def run(config_path: str) -> None:
    config   = load_config(config_path)
    dt       = config["simulation"]["time_step"]
    cam_res  = config["simulation"].get("camera_resolution", [640, 480])
    cam_w, cam_h    = int(cam_res[0]), int(cam_res[1])
    camera_interval = max(1, round(1.0 / (CAMERA_FPS * dt)))

    # Load the active detector backend defined in config['detector'].
    init_detector(config)
    inference_wait_s = get_inference_wait_s()

    # Bark frame shown when the drone faces bare trunk between panels
    _BARK_FRAME = np.full((cam_h, cam_w, 3), (70, 90, 110), dtype=np.uint8)

    # ── Build scene ───────────────────────────────────────────────────────────
    init_world(config)
    slf_data = build_scene(config)        # 12 + 11×8 = 100 panels in PyBullet
    drone_id = spawn_drone(config["drone"]["start_position"])
    controller = DroneController(drone_id, config, config["trees"])

    # ── Preload the 10 s video clip ────────────────────────────────────────────
    # The simulation_test_data/ folder holds 300 images. Loaded once at camera
    # resolution into RAM, they form a 10 s @ 30 fps clip that the drone
    # camera "plays back" during every INSPECT (looping if INSPECT runs
    # longer than 10 s). Inference samples whatever frame is current at the
    # moment its waiting-time cycle ends.
    _video_frames = _load_video_frames(_TEST_DATA_DIR, cam_w, cam_h)
    if not _video_frames:
        raise RuntimeError(
            f"No images found in {_TEST_DATA_DIR} — cannot build video feed.")
    _video_len = len(_video_frames)
    _video_duration_s = _video_len / VIDEO_FPS
    print(f"[sim] Loaded {_video_len} video frames "
          f"({_video_duration_s:.1f} s @ {VIDEO_FPS:.0f} fps, looped per tree).")

    # ── Inference thread ──────────────────────────────────────────────────────
    _frame_q  = queue.Queue(maxsize=1)
    _result_q = queue.Queue(maxsize=1)
    _infer_thread = threading.Thread(
        target=_inference_worker,
        args=(_frame_q, _result_q, inference_wait_s),
        daemon=True, name="inference-worker")
    _infer_thread.start()

    # ── Logger + main loop ────────────────────────────────────────────────────
    with SimulationLogger() as log:
        n_panels = len(slf_data)
        print_banner(config, n_panels, cam_w, cam_h, log.run_dir.resolve())

        physics_step   = 0
        _worker_busy   = False
        _result_frame: np.ndarray | None = None
        _result_boxes: list = []
        _new_result    = False         # True for one display tick per new result
        _last_saved_frame: np.ndarray | None = None  # identity of last saved input
        _infer_count   = 0
        _detect_count  = 0
        _current_frame = _BARK_FRAME   # what the camera sees right now
        _prev_state    = None          # to detect INSPECT entry
        _inspect_step_start = 0        # physics_step at which current INSPECT began

        # Placeholder shown in Window 2 before first inference
        _blank_win2 = _BARK_FRAME.copy()
        cv2.putText(_blank_win2, "Waiting for first inference ...",
                    (10, cam_h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (200, 200, 200), 1)

        # Placeholder shown in Window 2 while the drone is not inspecting
        # (TRANSIT / HOME / DONE), so the last frame of the previous tree
        # does not stay frozen on screen.
        _transit_win2 = _BARK_FRAME.copy()
        cv2.putText(_transit_win2, "Transiting to next tree ...",
                    (10, cam_h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (200, 200, 200), 1)

        try:
            while not controller.done:

                _step_deadline = time.perf_counter() + dt

                # ── Physics ───────────────────────────────────────────────────
                controller.step()
                p.stepSimulation()
                physics_step += 1

                # ── Block A: update camera frame (every physics step) ─────────
                # During INSPECT the camera plays the preloaded video clip
                # exactly ONCE per tree. The clock resets at each INSPECT
                # entry. The moment the index would walk past the last frame
                # we cut INSPECT short via controller.force_inspect_complete()
                # — the drone immediately ASCENDs and moves to the next tree.
                if controller.state == DroneController.INSPECT:
                    if _prev_state != DroneController.INSPECT:
                        _inspect_step_start = physics_step
                    elapsed_s = (physics_step - _inspect_step_start) * dt
                    frame_idx = int(elapsed_s * VIDEO_FPS)
                    if frame_idx >= _video_len:
                        controller.force_inspect_complete()
                    else:
                        _current_frame = _video_frames[frame_idx]
                _prev_state = controller.state

                # ── Block B: inference dispatch (every physics step) ──────────
                # Collect result if worker just finished.
                try:
                    _result_frame, _result_boxes = _result_q.get_nowait()
                    _worker_busy = False
                    _infer_count += 1
                    _new_result = True
                    if _result_boxes:
                        _detect_count += 1
                    controller.slf_detected = bool(_result_boxes)
                    if _result_boxes:
                        drone_pos_log, _ = controller.pose()
                        log.log_rcnn_detection(
                            len(_result_boxes), drone_pos_log, controller.state)
                except queue.Empty:
                    pass

                # Send next frame only when worker is free AND drone inspects.
                # The frame sent is whatever the camera sees RIGHT NOW —
                # all frames skipped while the model was busy are discarded.
                if (not _worker_busy
                        and controller.state == DroneController.INSPECT):
                    _frame_q.put(_current_frame)
                    _worker_busy = True

                elif controller.state != DroneController.INSPECT:
                    _worker_busy = False
                    # Drop the previous tree's last result so Window 2 does
                    # not stay frozen on it during TRANSIT.
                    _result_frame = None
                    _result_boxes = []
                    _new_result = False
                    _last_saved_frame = None
                    controller.slf_detected = False

                # ── Block C: display (30 fps) ─────────────────────────────────
                if physics_step % camera_interval == 0:

                    drone_pos, drone_orn = controller.pose()

                    # Window 1 — live camera view
                    if controller.state == DroneController.INSPECT:
                        live_frame = _current_frame
                    else:
                        fwd     = np.array([math.cos(controller.current_yaw),
                                            math.sin(controller.current_yaw),
                                            0.0])
                        look_at = drone_pos + fwd * 10.0
                        cam_fwd, cam_up = get_camera_vectors(
                            drone_orn, drone_pos, look_at)
                        view, proj = build_camera_matrices(
                            drone_pos, cam_fwd, cam_up)
                        live_frame = enhance_frame(capture_frame(view, proj))

                    tree       = controller.current_tree
                    tree_label = tree["id"] if tree else "—"
                    tree_idx   = min(controller.tree_idx + 1,
                                     len(config["trees"]))
                    orbit_pct  = (controller.orbit_progress
                                  if controller.state == DroneController.INSPECT
                                  else 0.0)
                    det_pct    = (_detect_count / _infer_count * 100
                                  if _infer_count else 0.0)

                    win1 = annotate(
                        frame        = live_frame,
                        rcnn_boxes   = [],          # don't draw boxes on the
                                                    # live frame (they apply
                                                    # to the inferred frame)
                        n_detections = len(_result_boxes),  # but the HUD
                                                    # count should still
                                                    # reflect the latest
                                                    # inference result
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
                    cv2.imshow(WINDOW_FEED, win1)

                    # Window 2 — latest inference result
                    if controller.state != DroneController.INSPECT:
                        win2 = _transit_win2
                    elif _result_frame is not None:
                        win2 = annotate(
                            frame        = _result_frame,
                            rcnn_boxes   = _result_boxes,
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
                        # Save the annotated frame ONCE per *unique input
                        # frame* with detections. Consecutive inferences on
                        # the same panel image (same array identity) are
                        # held on screen but not re-saved.
                        if (_new_result and _result_boxes
                                and _result_frame is not _last_saved_frame):
                            log.save_frame(win2)
                            _last_saved_frame = _result_frame
                        _new_result = False
                    else:
                        win2 = _blank_win2

                    cv2.imshow(WINDOW_MODEL, win2)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\nStopped by user (q).")
                        break

                    log.print_heartbeat(
                        drone_pos, controller.state, tree_label,
                        len(_result_boxes), _infer_count, det_pct)
                    log.tick_frame()

                _remaining = _step_deadline - time.perf_counter()
                if _remaining > 0:
                    time.sleep(_remaining)

        except KeyboardInterrupt:
            print("\nInterrupted (Ctrl-C).")

        finally:
            try:
                _frame_q.put_nowait(None)
            except queue.Full:
                pass
            _infer_thread.join(timeout=3.0)
            close_windows()
            p.disconnect()
            log.print_summary(sim_time=physics_step * dt,
                               infer_count=_infer_count,
                               detect_count=_detect_count)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="UAV SLF Inspection Simulation")
    ap.add_argument("--config", default="config.json")
    args = ap.parse_args()
    run(args.config)
