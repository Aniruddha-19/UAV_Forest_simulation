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
    """Load config.json and optionally override the capture frame rate."""
    with open(config_path) as f:
        config = json.load(f)
    if fps_override is not None:
        config["drone"]["capture_fps"] = fps_override
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
    print(f"  Capture FPS    : {config['drone']['capture_fps']}")
    print(f"  Logs           : {log_dir}")
    print("  Press 'q' in the camera window to quit.")
    print("=" * 60)


# ── Angle helpers ─────────────────────────────────────────────────────────────

def _angle_diff(a: float, b: float) -> float:
    """Smallest signed angular difference a − b, wrapped to (−π, π]."""
    return math.atan2(math.sin(a - b), math.cos(a - b))


# ── Main simulation loop ──────────────────────────────────────────────────────

def run(config_path: str, fps_override: int | None = None) -> None:
    config = load_config(config_path, fps_override)

    capture_fps     = config["drone"]["capture_fps"]
    dt              = config["simulation"]["time_step"]
    steps_per_frame = max(1, round(1.0 / (capture_fps * dt)))

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

        try:
            # ── Main loop ─────────────────────────────────────────────────────
            while not controller.done:

                _step_deadline = time.perf_counter() + dt

                # Advance drone + physics
                controller.step()
                p.stepSimulation()
                physics_step += 1

                # Only run detection / display every N physics steps
                if physics_step % steps_per_frame != 0:
                    _remaining = _step_deadline - time.perf_counter()
                    if _remaining > 0:
                        time.sleep(_remaining)
                    continue

                # ── Step 5: Drone pose ────────────────────────────────────────
                drone_pos, drone_orn = controller.pose()

                # ── Step 6: Frame + detection ─────────────────────────────────
                if (controller.state == DroneController.INSPECT
                        and controller.current_tree is not None):

                    # ── INSPECT: real images → R-CNN, no 3D rendering ─────────
                    tree_id  = controller.current_tree["id"]
                    egg_data = tree_egg_data.get(tree_id, [])

                    if egg_data:
                        # The drone yaw points from the drone TOWARD the trunk,
                        # so the direction FROM the trunk TOWARD the drone
                        # (i.e. which face the camera sees) is yaw + π.
                        facing = controller.current_yaw + math.pi

                        # Pick the egg mass whose bearing from the trunk is
                        # closest to the drone's current facing direction.
                        best_idx = min(
                            range(len(egg_data)),
                            key=lambda i: abs(_angle_diff(
                                egg_data[i]["angle"], facing)))

                        displayed = egg_data[best_idx]
                        frame     = (displayed["bgr"]
                                     if displayed["bgr"] is not None
                                     else np.zeros((IMG_H, IMG_W, 3),
                                                   dtype=np.uint8))

                        # Run R-CNN on every egg mass image for this tree.
                        # Show boxes only for the currently displayed one.
                        any_detected = False
                        rcnn_boxes   = []
                        for i, em in enumerate(egg_data):
                            if em["bgr"] is None:
                                continue
                            boxes = detect_egg_masses(em["bgr"])
                            if i == best_idx:
                                rcnn_boxes = boxes
                            if boxes:
                                any_detected = True

                        controller.egg_detected = any_detected

                    else:
                        frame                   = np.zeros(
                            (IMG_H, IMG_W, 3), dtype=np.uint8)
                        rcnn_boxes              = []
                        controller.egg_detected = False

                else:
                    # ── TRANSIT / DESCEND / ASCEND / HOME: 3D camera ──────────
                    fwd = np.array([math.cos(controller.current_yaw),
                                    math.sin(controller.current_yaw),
                                    0.0])
                    look_at = drone_pos + fwd * 10.0   # horizontal, 10 m ahead

                    cam_fwd, cam_up = get_camera_vectors(
                        drone_orn, drone_pos, look_at)
                    view, proj = build_camera_matrices(
                        drone_pos, cam_fwd, cam_up)

                    raw_frame = capture_frame(view, proj)
                    frame     = enhance_frame(raw_frame)

                    rcnn_boxes              = []
                    controller.egg_detected = False

                if rcnn_boxes:
                    log.log_rcnn_detection(
                        len(rcnn_boxes), drone_pos, controller.state)

                # ── Step 7: Annotate frame ────────────────────────────────────
                tree       = controller.current_tree
                tree_label = tree["id"] if tree else "—"
                tree_idx   = min(controller.tree_idx + 1, len(config["trees"]))
                orbit_pct  = (controller.orbit_progress
                              if controller.state == DroneController.INSPECT
                              else 0.0)

                annotated = annotate(
                    frame        = frame,
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
                    fps          = capture_fps,
                )

                # ── Step 8: Save frame when R-CNN detects egg mass ────────────
                if rcnn_boxes:
                    log.save_frame(annotated)

                # ── Step 9: Display + heartbeat ───────────────────────────────
                if show(annotated):
                    print("\nStopped by user (q).")
                    break

                log.print_heartbeat(
                    drone_pos, controller.state, tree_label,
                    len(rcnn_boxes), capture_fps)

                log.tick_frame()

                _remaining = _step_deadline - time.perf_counter()
                if _remaining > 0:
                    time.sleep(_remaining)

        except KeyboardInterrupt:
            print("\nInterrupted (Ctrl-C).")

        finally:
            close_windows()
            p.disconnect()
            log.print_summary()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="UAV Tree Inspection — orbit each tree, save on detection")
    ap.add_argument(
        "--config", default="config.json",
        help="Path to the JSON environment config  (default: config.json)")
    ap.add_argument(
        "--fps", type=int, default=None,
        help="Override capture frame rate in frames-per-second")
    args = ap.parse_args()
    run(args.config, args.fps)
