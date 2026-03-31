#!/usr/bin/env python3
"""
simulation.py  —  Entry point
==============================
Wires together all simulation modules and runs the main loop.

Module responsibilities
-----------------------
  environment.py      — PyBullet world setup, spawning trees / egg masses / drone
  drone_controller.py — TRANSIT → INSPECT → HOME state machine
  camera.py           — frame capture, image processing, egg mass detection
  visualizer.py       — HUD annotation and OpenCV display
  logger.py           — CSV log, frame saving, console heartbeat

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

import numpy as np
import pybullet as p

from camera           import (build_camera_matrices, capture_and_process,
                               check_egg_in_fov, detect_yellow_blobs,
                               get_camera_vectors)
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
    orbit_spd    = config["drone"].get("inspect_orbit_speed_deg", 20)
    orbit_r      = config["drone"].get("inspect_radius", 3.0)
    trunk_clr    = config["drone"].get("inspect_trunk_clearance", 1.0)
    inspect_alt  = config["drone"].get("inspect_altitude", 2.0)
    print("=" * 60)
    print("  UAV Tree Inspection Simulation")
    print("=" * 60)
    print(f"  Trees          : {len(config['trees'])}")
    print(f"  Egg masses     : {n_egg_masses}")
    print(f"  Transit speed  : {config['drone']['transit_speed']} m/s")
    print(f"  Cruise alt     : {config['drone']['cruise_altitude']} m")
    print(f"  Inspect alt    : {inspect_alt} m  (trunk level)")
    print(f"  Canopy radius  : {orbit_r} m  (clearance beyond canopy during transit)")
    print(f"  Trunk clearance: {trunk_clr} m  (orbit radius = trunk_r + {trunk_clr} m at inspect alt)")
    print(f"  Orbit speed    : {orbit_spd} °/s  "
          f"(360° ≈ {360 / orbit_spd:.0f} s / tree)")
    print(f"  Capture FPS    : {config['drone']['capture_fps']}")
    print(f"  Logs           : {log_dir}")
    print("  Press 'q' in the camera window to quit.")
    print("=" * 60)


# ── Main simulation loop ──────────────────────────────────────────────────────

def run(config_path: str, fps_override: int | None = None) -> None:
    config = load_config(config_path, fps_override)

    capture_fps     = config["drone"]["capture_fps"]
    dt              = config["simulation"]["time_step"]
    steps_per_frame = max(1, round(1.0 / (capture_fps * dt)))

    # ── Step 1: Initialise world & build scene ────────────────────────────────
    init_world(config)
    egg_masses = build_scene(config)                       # spawn trees + eggs
    drone_id   = spawn_drone(config["drone"]["start_position"])

    # ── Step 2: Initialise drone controller ───────────────────────────────────
    controller = DroneController(drone_id, config, config["trees"])

    # ── Step 3: Open logger ───────────────────────────────────────────────────
    with SimulationLogger() as log:
        print_banner(config, len(egg_masses), log.run_dir.resolve())

        physics_step = 0

        try:
            # ── Main loop ─────────────────────────────────────────────────────
            while not controller.done:

                # Advance drone + physics
                controller.step()
                p.stepSimulation()
                physics_step += 1

                # Only process camera every N physics steps
                if physics_step % steps_per_frame != 0:
                    continue

                # ── Step 4: Camera setup ──────────────────────────────────────
                drone_pos, drone_orn = controller.pose()

                # During DESCEND/INSPECT/ASCEND the camera looks at the trunk;
                # during TRANSIT / HOME it looks straight down (nadir).
                look_at = None
                if controller.state in (DroneController.DESCEND,
                                        DroneController.INSPECT,
                                        DroneController.ASCEND):
                    tree = controller.current_tree
                    if tree:
                        look_at = np.array(controller.trunk_look_at(tree))

                cam_fwd, cam_up = get_camera_vectors(
                    drone_orn, drone_pos, look_at)
                view, proj = build_camera_matrices(drone_pos, cam_fwd, cam_up)

                # ── Step 5: Capture + process frame ───────────────────────────
                frame     = capture_and_process(view, proj)
                img_boxes = detect_yellow_blobs(frame)

                # ── Step 5b: Faster R-CNN inference ───────────────────────────
                rcnn_boxes = detect_egg_masses(frame)

                # ── Step 6: Detect egg masses (geometric FOV check) ───────────
                geo_dets = []
                for em in egg_masses:
                    visible, px, py = check_egg_in_fov(
                        drone_pos, cam_fwd, cam_up, em["position"], em["body"])
                    if visible:
                        geo_dets.append((em["id"], px, py))
                        log.log_geometric_detection(
                            em["id"], em["position"],
                            drone_pos, controller.state)

                if img_boxes:
                    log.log_segmentation_detection(
                        len(img_boxes), drone_pos, controller.state)

                if rcnn_boxes:
                    log.log_rcnn_detection(
                        len(rcnn_boxes), drone_pos, controller.state)

                # Slow the orbit whenever an egg mass is geometrically visible
                controller.egg_detected = bool(geo_dets)

                # ── Step 7: Annotate frame ────────────────────────────────────
                tree        = controller.current_tree
                tree_label  = tree["id"] if tree else "—"
                tree_idx    = min(controller.tree_idx + 1,
                                  len(config["trees"]))
                orbit_pct   = (controller.orbit_progress
                               if controller.state == DroneController.INSPECT
                               else 0.0)

                annotated = annotate(
                    frame      = frame,
                    geo_dets   = geo_dets,
                    img_boxes  = img_boxes,
                    rcnn_boxes = rcnn_boxes,
                    drone_pos  = drone_pos,
                    state      = controller.state,
                    tree_label = tree_label,
                    tree_idx   = tree_idx,
                    total_trees= len(config["trees"]),
                    orbit_pct  = orbit_pct,
                    frame_no   = log.frame_no,
                    saved_no   = log.saved_no,
                    fps        = capture_fps,
                )

                # ── Step 8: Save frame only when Faster R-CNN detects egg mass ─
                if rcnn_boxes:
                    log.save_frame(annotated)

                # ── Step 9: Display + heartbeat ───────────────────────────────
                if show(annotated):
                    print("\nStopped by user (q).")
                    break

                log.print_heartbeat(
                    drone_pos, controller.state, tree_label,
                    len(geo_dets), len(img_boxes), len(rcnn_boxes), capture_fps)

                log.tick_frame()

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
