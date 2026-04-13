#!/usr/bin/env python3
"""
dual_drone/simulation_2drones.py
==================================
Two drones inspect the same forest simultaneously.

  Drone 1  — trees[0:4]   cruise altitude 18 m
  Drone 2  — trees[4:9]   cruise altitude 20 m

Different cruise altitudes prevent mid-air crossing during transit legs.
Both drones share one PyBullet world.  Egg-mass detection uses real bug
photos passed directly to Faster R-CNN (no 3-D camera rendering during
INSPECT).  Transit / Descend / Ascend phases show the live 3-D feed.

Logs
----
  logs/drone1/<timestamp>/detections.csv  + frames/
  logs/drone2/<timestamp>/detections.csv  + frames/

Display
-------
  Single 1280×480 side-by-side OpenCV window.
  When a drone finishes its mission, its panel shows "MISSION COMPLETE".

Usage
-----
    python dual_drone/simulation_2drones.py
    python dual_drone/simulation_2drones.py --config config.json --fps 5

Controls
--------
    q       — quit
    Ctrl-C  — interrupt
"""

import argparse
import copy
import json
import math
import os
import sys
import time

import cv2
import numpy as np
import pybullet as p

# ── Shared modules live in the parent directory ───────────────────────────────
_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PARENT)

from camera           import (IMG_W, IMG_H,
                               build_camera_matrices, capture_frame,
                               enhance_frame, get_camera_vectors)
from detector         import detect_egg_masses
from drone_controller import DroneController
from environment      import build_scene, init_world, spawn_drone
from logger           import SimulationLogger
from visualizer       import annotate


# ── Visibility / timing constants (mirrors single-drone simulation.py) ────────
_EM_VIS_HALF_ANGLE = math.radians(30.0)   # ±30° FOV window per egg mass panel
_EM_MAX_SHOW_SECS  = 2.0                  # max seconds to show one panel per viewing
_EM_MAX_VIEWS      = 2                    # max separate viewings per panel per orbit
_BARK_FRAME        = np.full((IMG_H, IMG_W, 3), (70, 90, 110), dtype=np.uint8)
_DONE_FRAME        = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)   # mission-complete panel


# ── Helpers ───────────────────────────────────────────────────────────────────

def _angle_diff(a: float, b: float) -> float:
    """Smallest signed angular difference a − b, wrapped to (−π, π]."""
    return math.atan2(math.sin(a - b), math.cos(a - b))


def load_config(path: str, fps_override=None) -> dict:
    with open(path) as f:
        cfg = json.load(f)
    if fps_override is not None:
        cfg["drone"]["capture_fps"] = fps_override
    return cfg


def preload_egg_images(egg_masses_data: list,
                       tree_ids: set,
                       trunk_pos_map: dict) -> dict:
    """
    Build tree_egg_data[tree_id] = [{id, angle, bgr}, …] for the
    supplied tree_ids, loading and resizing each image from disk once.
    """
    data: dict[str, list] = {}
    for em in egg_masses_data:
        if em["tree_id"] not in tree_ids:
            continue
        tp     = trunk_pos_map[em["tree_id"]]
        angle  = math.atan2(em["position"][1] - tp[1],
                            em["position"][0] - tp[0])
        bgr    = None
        path   = em.get("image_path")
        if path:
            raw = cv2.imread(path)
            if raw is not None:
                bgr = cv2.resize(raw, (IMG_W, IMG_H))
            else:
                print(f"[WARNING] Cannot load: {path}")
        data.setdefault(em["tree_id"], []).append(
            {"id": em["id"], "angle": angle, "bgr": bgr})
    return data


def make_drone_state() -> dict:
    """Fresh per-orbit visibility tracking state for one drone."""
    return {
        "inspect_tree_id": None,
        "em_first_seen":  {},
        "em_view_count":  {},
        "em_geo_in_view": set(),
    }


def process_inspect(controller, tree_egg_data: dict,
                    ds: dict, now: float) -> tuple:
    """
    Detect egg masses for one drone in INSPECT state using real images.

    Returns (frame, rcnn_boxes, any_detected).
    ds is the per-drone state dict and is mutated in place.
    """
    tree_id  = controller.current_tree["id"]
    egg_data = tree_egg_data.get(tree_id, [])

    # Reset per-orbit counters when drone moves to a new tree
    if tree_id != ds["inspect_tree_id"]:
        ds["inspect_tree_id"] = tree_id
        ds["em_first_seen"].clear()
        ds["em_view_count"].clear()
        ds["em_geo_in_view"].clear()

    if not egg_data:
        return _BARK_FRAME, [], False

    facing  = controller.current_yaw + math.pi

    # Geometric FOV check
    geo_ids = {
        egg_data[i]["id"]
        for i in range(len(egg_data))
        if abs(_angle_diff(egg_data[i]["angle"], facing)) <= _EM_VIS_HALF_ANGLE
    }

    # Detect FOV entries → increment view count and record start time
    for em_id in geo_ids:
        if em_id not in ds["em_geo_in_view"]:
            ds["em_view_count"][em_id] = ds["em_view_count"].get(em_id, 0) + 1
            ds["em_first_seen"][em_id] = now
    ds["em_geo_in_view"] = set(geo_ids)

    # Apply time and count limits
    in_view = [
        (i, em) for i, em in enumerate(egg_data)
        if em["id"] in geo_ids
        and ds["em_view_count"].get(em["id"], 0) <= _EM_MAX_VIEWS
        and now - ds["em_first_seen"].get(em["id"], now) <= _EM_MAX_SHOW_SECS
    ]

    if not in_view:
        return _BARK_FRAME, [], False

    # Display the panel most centred in the frame
    best_i, best_em = min(
        in_view, key=lambda t: abs(_angle_diff(t[1]["angle"], facing)))

    frame        = best_em["bgr"] if best_em["bgr"] is not None else _BARK_FRAME
    any_detected = False
    rcnn_boxes   = []
    for i, em in in_view:
        if em["bgr"] is None:
            continue
        boxes = detect_egg_masses(em["bgr"])
        if i == best_i:
            rcnn_boxes = boxes
        if boxes:
            any_detected = True

    return frame, rcnn_boxes, any_detected


def get_transit_frame(controller, drone_pos, drone_orn) -> np.ndarray:
    """Render the 3-D PyBullet camera view for non-INSPECT states."""
    fwd = np.array([math.cos(controller.current_yaw),
                    math.sin(controller.current_yaw),
                    0.0])
    look_at         = drone_pos + fwd * 10.0
    cam_fwd, cam_up = get_camera_vectors(drone_orn, drone_pos, look_at)
    view, proj      = build_camera_matrices(drone_pos, cam_fwd, cam_up)
    return enhance_frame(capture_frame(view, proj))


def annotate_drone(controller, frame, boxes, drone_pos,
                   cfg, log, capture_fps) -> np.ndarray:
    """Wrapper around visualizer.annotate for one drone."""
    tree       = controller.current_tree
    all_trees  = cfg["trees"]
    tree_label = tree["id"] if tree else "—"
    tree_idx   = min(controller.tree_idx + 1, len(all_trees))
    orbit_pct  = (controller.orbit_progress
                  if controller.state == DroneController.INSPECT
                  else 0.0)
    return annotate(
        frame        = frame,
        rcnn_boxes   = boxes,
        drone_pos    = drone_pos,
        state        = controller.state,
        tree_label   = tree_label,
        tree_idx     = tree_idx,
        total_trees  = len(all_trees),
        orbit_pct    = orbit_pct,
        orbit_radius = controller.orbit_radius,
        frame_no     = log.frame_no,
        saved_no     = log.saved_no,
        fps          = capture_fps,
    )


def add_drone_label(frame: np.ndarray, label: str) -> np.ndarray:
    """Overlay a coloured drone-ID banner at the top-centre of the frame."""
    out    = frame.copy()
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cx     = (IMG_W - tw) // 2
    cv2.rectangle(out, (cx - 8, 3), (cx + tw + 8, 30), (20, 20, 20), -1)
    cv2.putText(out, label, (cx, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)
    return out


def draw_done_panel(label: str) -> np.ndarray:
    """Dark panel shown after a drone completes its mission."""
    out = _DONE_FRAME.copy()
    msg = "MISSION COMPLETE"
    (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.putText(out, msg,
                ((IMG_W - tw) // 2, IMG_H // 2 - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 80), 2)
    cv2.putText(out, label,
                ((IMG_W - tw) // 2 + 30, IMG_H // 2 + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 60), 1)
    return out


# ── Main simulation loop ──────────────────────────────────────────────────────

def run(config_path: str, fps_override=None) -> None:
    base_cfg = load_config(config_path, fps_override)

    capture_fps     = base_cfg["drone"]["capture_fps"]
    dt              = base_cfg["simulation"]["time_step"]
    steps_per_frame = max(1, round(1.0 / (capture_fps * dt)))

    # ── Split trees ───────────────────────────────────────────────────────────
    all_trees = base_cfg["trees"]
    if len(all_trees) < 2:
        raise ValueError("Config must have at least 2 trees to split between drones.")

    # Drone 1: trees[0:4]  (trees 1–4 in a 9-tree grid)
    # Drone 2: trees[4:]   (trees 5–9)
    cfg1 = copy.deepcopy(base_cfg)
    cfg1["trees"] = all_trees[0:4]
    cfg1["drone"]["cruise_altitude"] = 18.0   # drone 1 flies at 18 m

    cfg2 = copy.deepcopy(base_cfg)
    cfg2["trees"] = all_trees[4:]
    cfg2["drone"]["cruise_altitude"] = 20.0   # drone 2 flies at 20 m (avoids crossing)

    # ── PyBullet world: spawn all trees + both drones ─────────────────────────
    init_world(base_cfg)
    egg_masses_data = build_scene(base_cfg)
    drone1_id       = spawn_drone(base_cfg["drone"]["start_position"])
    drone2_id       = spawn_drone(base_cfg["drone"]["start_position"])

    # ── Controllers ───────────────────────────────────────────────────────────
    ctrl1 = DroneController(drone1_id, cfg1, cfg1["trees"])
    ctrl2 = DroneController(drone2_id, cfg2, cfg2["trees"])

    # ── Egg-mass images ───────────────────────────────────────────────────────
    trunk_pos_map   = {t["id"]: t["position"] for t in all_trees}
    drone1_tree_ids = {t["id"] for t in cfg1["trees"]}
    drone2_tree_ids = {t["id"] for t in cfg2["trees"]}
    egg_data1 = preload_egg_images(egg_masses_data, drone1_tree_ids, trunk_pos_map)
    egg_data2 = preload_egg_images(egg_masses_data, drone2_tree_ids, trunk_pos_map)

    ds1 = make_drone_state()
    ds2 = make_drone_state()

    # ── Loggers ───────────────────────────────────────────────────────────────
    with SimulationLogger("logs/drone1") as log1, \
         SimulationLogger("logs/drone2") as log2:

        n_em = sum(len(t.get("egg_masses", [])) for t in all_trees)
        print("=" * 60)
        print("  UAV Forest — Dual Drone Inspection")
        print("=" * 60)
        print(f"  Total trees     : {len(all_trees)}"
              f"  (Drone 1: {len(cfg1['trees'])}  Drone 2: {len(cfg2['trees'])})")
        print(f"  Total egg masses: {n_em}")
        print(f"  Drone 1 trees   : {[t['id'] for t in cfg1['trees']]}")
        print(f"  Drone 2 trees   : {[t['id'] for t in cfg2['trees']]}")
        print(f"  Drone 1 altitude: {cfg1['drone']['cruise_altitude']} m")
        print(f"  Drone 2 altitude: {cfg2['drone']['cruise_altitude']} m")
        print(f"  Logs → {log1.run_dir.resolve()}")
        print(f"         {log2.run_dir.resolve()}")
        print("  Press 'q' to quit.")
        print("=" * 60)

        physics_step  = 0
        heartbeat_inv = max(1, int(capture_fps * 5))

        try:
            while not (ctrl1.done and ctrl2.done):
                _deadline = time.perf_counter() + dt

                if not ctrl1.done:
                    ctrl1.step()
                if not ctrl2.done:
                    ctrl2.step()
                p.stepSimulation()
                physics_step += 1

                if physics_step % steps_per_frame != 0:
                    rem = _deadline - time.perf_counter()
                    if rem > 0:
                        time.sleep(rem)
                    continue

                now = time.perf_counter()

                # ── Drone 1 ───────────────────────────────────────────────────
                if ctrl1.done:
                    ann1   = draw_done_panel("DRONE 1")
                    boxes1 = []
                    pos1   = ctrl1.pose()[0]
                else:
                    pos1, orn1 = ctrl1.pose()
                    if (ctrl1.state == DroneController.INSPECT
                            and ctrl1.current_tree is not None):
                        frame1, boxes1, det1 = process_inspect(
                            ctrl1, egg_data1, ds1, now)
                        ctrl1.egg_detected = det1
                    else:
                        frame1 = get_transit_frame(ctrl1, pos1, orn1)
                        boxes1 = []
                        ctrl1.egg_detected = False

                    if boxes1:
                        log1.log_rcnn_detection(len(boxes1), pos1, ctrl1.state)

                    ann1 = annotate_drone(
                        ctrl1, frame1, boxes1, pos1, cfg1, log1, capture_fps)

                ann1 = add_drone_label(ann1, "DRONE 1")
                if boxes1:
                    log1.save_frame(ann1)

                # ── Drone 2 ───────────────────────────────────────────────────
                if ctrl2.done:
                    ann2   = draw_done_panel("DRONE 2")
                    boxes2 = []
                    pos2   = ctrl2.pose()[0]
                else:
                    pos2, orn2 = ctrl2.pose()
                    if (ctrl2.state == DroneController.INSPECT
                            and ctrl2.current_tree is not None):
                        frame2, boxes2, det2 = process_inspect(
                            ctrl2, egg_data2, ds2, now)
                        ctrl2.egg_detected = det2
                    else:
                        frame2 = get_transit_frame(ctrl2, pos2, orn2)
                        boxes2 = []
                        ctrl2.egg_detected = False

                    if boxes2:
                        log2.log_rcnn_detection(len(boxes2), pos2, ctrl2.state)

                    ann2 = annotate_drone(
                        ctrl2, frame2, boxes2, pos2, cfg2, log2, capture_fps)

                ann2 = add_drone_label(ann2, "DRONE 2")
                if boxes2:
                    log2.save_frame(ann2)

                # ── Side-by-side display ──────────────────────────────────────
                combined = np.hstack([ann1, ann2])
                cv2.imshow("Dual Drone Feed  [q = quit]", combined)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\nStopped by user (q).")
                    break

                # ── Heartbeat (every 5 s of rendered time) ────────────────────
                if log1.frame_no % heartbeat_inv == 0:
                    t1 = ctrl1.current_tree["id"] if ctrl1.current_tree else "DONE"
                    t2 = ctrl2.current_tree["id"] if ctrl2.current_tree else "DONE"
                    print(f"  [D1] {log1.elapsed:6.1f}s | {ctrl1.state:8s}"
                          f" | tree={t1:8s} | rcnn={len(boxes1)}")
                    print(f"  [D2] {log2.elapsed:6.1f}s | {ctrl2.state:8s}"
                          f" | tree={t2:8s} | rcnn={len(boxes2)}")

                log1.tick_frame()
                log2.tick_frame()

                rem = _deadline - time.perf_counter()
                if rem > 0:
                    time.sleep(rem)

        except KeyboardInterrupt:
            print("\nInterrupted (Ctrl-C).")

        finally:
            cv2.destroyAllWindows()
            p.disconnect()
            log1.print_summary()
            log2.print_summary()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _default_cfg = os.path.join(_PARENT, "config.json")
    ap = argparse.ArgumentParser(
        description="Dual-drone UAV forest inspection")
    ap.add_argument("--config", default=_default_cfg,
                    help=f"Path to config.json  (default: {_default_cfg})")
    ap.add_argument("--fps", type=int, default=None,
                    help="Override capture FPS")
    args = ap.parse_args()
    run(args.config, args.fps)
