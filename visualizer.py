"""
visualizer.py
=============
Handles all visual output:

  • annotate()  — draws detections, HUD, orbit progress bar, and detection
                  banner onto a processed camera frame
  • show()      — displays the annotated frame in the OpenCV window and
                  checks for the quit key
"""

import cv2
import numpy as np

from camera import IMG_W, IMG_H


# ── Frame annotation ──────────────────────────────────────────────────────────

def annotate(frame:       np.ndarray,
             geo_dets:    list[tuple[str, int, int]],
             img_boxes:   list[tuple[int, int, int, int]],
             rcnn_boxes:  list[tuple[int, int, int, int]],
             drone_pos:   np.ndarray,
             state:       str,
             tree_label:  str,
             tree_idx:    int,
             total_trees: int,
             orbit_pct:   float,
             frame_no:    int,
             saved_no:    int,
             fps:         int) -> np.ndarray:
    """
    Compose all annotations onto a copy of *frame* and return the result.

    Layers (drawn in order):
      1. Geometric detections   — cyan circles with ID labels
      2. Segmentation detections — yellow bounding rectangles
      3. Faster R-CNN detections — red bounding rectangles
      4. Orbit progress bar     — shown only during INSPECT state
      5. HUD text panel         — state, tree, position, frame counters
      6. Detection banner       — bright green banner when RCNN detects egg mass
      7. Legend                 — key for the detection colours

    Parameters
    ----------
    frame       : processed BGR frame from camera.capture_and_process()
    geo_dets    : list of (label, px, py) from camera.check_egg_in_fov()
    img_boxes   : list of (x,y,w,h) bounding boxes from camera.detect_yellow_blobs()
    rcnn_boxes  : list of (x1,y1,x2,y2) from detector.detect_egg_masses()
    drone_pos   : drone world position [x, y, z]
    state       : controller state string ("TRANSIT", "INSPECT", "HOME")
    tree_label  : ID string of the current tree, or "—"
    tree_idx    : 1-based index of the current tree
    total_trees : total number of trees in the scene
    orbit_pct   : fraction of the current orbit completed (0.0 – 1.0)
    frame_no    : total frames rendered so far
    saved_no    : frames saved to disk so far (detection-only subset)
    fps         : configured capture frame rate

    Returns
    -------
    Annotated BGR image (same shape as *frame*).
    """
    out = frame.copy()

    _draw_geometric_detections(out, geo_dets)
    _draw_segmentation_boxes(out, img_boxes)
    _draw_rcnn_boxes(out, rcnn_boxes)
    _draw_orbit_bar(out, state, orbit_pct)
    _draw_hud(out, state, tree_label, tree_idx, total_trees,
               drone_pos, len(geo_dets), len(img_boxes), len(rcnn_boxes),
               frame_no, saved_no, fps)
    _draw_detection_banner(out, rcnn_boxes)
    _draw_legend(out)

    return out


# ── Display ───────────────────────────────────────────────────────────────────

def show(frame: np.ndarray, window_title: str = "Drone Camera Feed") -> bool:
    """
    Display *frame* in an OpenCV window.

    Returns
    -------
    True  if the user pressed 'q' (quit signal)
    False otherwise
    """
    cv2.imshow(window_title, frame)
    return (cv2.waitKey(1) & 0xFF) == ord("q")


def close_windows() -> None:
    """Destroy all OpenCV windows."""
    cv2.destroyAllWindows()


# ── Private drawing helpers ───────────────────────────────────────────────────

def _draw_geometric_detections(img: np.ndarray,
                                geo_dets: list[tuple[str, int, int]]) -> None:
    """
    For each geometrically detected egg mass draw a cyan circle and its
    ID label at the projected image location.
    """
    COLOUR = (0, 210, 210)   # cyan
    for label, px, py in geo_dets:
        cv2.circle(img, (px, py), 14, COLOUR, 2)
        cv2.putText(img, label, (px + 16, py - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOUR, 1)


def _draw_segmentation_boxes(img: np.ndarray,
                              img_boxes: list[tuple[int, int, int, int]]) -> None:
    """
    Draw a yellow bounding rectangle for each blob found by HSV segmentation.
    """
    COLOUR = (0, 220, 255)   # yellow-orange
    for (x, y, w, h) in img_boxes:
        cv2.rectangle(img, (x, y), (x + w, y + h), COLOUR, 1)


def _draw_rcnn_boxes(img: np.ndarray,
                     rcnn_boxes: list[tuple[int, int, int, int]]) -> None:
    """
    Draw a red bounding rectangle for each egg mass found by Faster R-CNN.
    """
    COLOUR = (0, 0, 220)   # red (BGR)
    for (x1, y1, x2, y2) in rcnn_boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), COLOUR, 2)
        cv2.putText(img, "egg mass", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOUR, 1)


def _draw_orbit_bar(img: np.ndarray, state: str, orbit_pct: float) -> None:
    """
    Draw a horizontal progress bar at the bottom of the frame showing how
    much of the current tree's 360° orbit has been completed.
    Only visible during the INSPECT state.
    """
    if state != "INSPECT":
        return

    BAR_COLOUR  = (0, 200, 255)
    BACK_COLOUR = (60, 60, 60)
    bar_x, bar_y = 10, IMG_H - 30
    bar_w = IMG_W - 20
    bar_h = 12

    # Background track
    cv2.rectangle(img,
                  (bar_x, bar_y),
                  (bar_x + bar_w, bar_y + bar_h),
                  BACK_COLOUR, -1)

    # Filled portion
    filled = int(bar_w * min(orbit_pct, 1.0))
    if filled > 0:
        cv2.rectangle(img,
                      (bar_x, bar_y),
                      (bar_x + filled, bar_y + bar_h),
                      BAR_COLOUR, -1)

    # Percentage label above the bar
    cv2.putText(img, f"Orbit  {orbit_pct * 100:.0f}%",
                (bar_x, bar_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, BAR_COLOUR, 1)


def _draw_hud(img: np.ndarray,
              state: str, tree_label: str,
              tree_idx: int, total_trees: int,
              drone_pos: np.ndarray,
              n_geo: int, n_seg: int, n_rcnn: int,
              frame_no: int, saved_no: int,
              fps: int) -> None:
    """
    Draw the heads-up display text panel in the top-left corner.
    Text colour changes between TRANSIT (green) and INSPECT (cyan-blue).
    """
    COLOUR_INSPECT  = (0, 200, 255)
    COLOUR_TRANSIT  = (40, 255, 60)
    colour = COLOUR_INSPECT if state == "INSPECT" else COLOUR_TRANSIT

    hud_lines = [
        f"State  : {state}",
        f"Tree   : {tree_label}  ({tree_idx} / {total_trees})",
        f"FPS    : {fps}",
        f"Frame  : {frame_no}   Saved: {saved_no}",
        (f"Pos    : ({drone_pos[0]:.1f},  {drone_pos[1]:.1f},"
         f"  {drone_pos[2]:.1f}) m"),
        f"Geo    : {n_geo}   Seg: {n_seg}   RCNN: {n_rcnn}",
    ]

    for i, line in enumerate(hud_lines):
        cv2.putText(img, line, (10, 22 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, colour, 1)


def _draw_detection_banner(img: np.ndarray,
                            rcnn_boxes: list) -> None:
    """
    Draw a bright green 'EGG MASS DETECTED' banner in the top-right corner
    when Faster R-CNN finds at least one egg mass.
    """
    if not rcnn_boxes:
        return
    cv2.rectangle(img, (IMG_W - 165, 5), (IMG_W - 5, 30), (0, 200, 0), -1)
    cv2.putText(img, "EGG MASS DETECTED", (IMG_W - 163, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)


def _draw_legend(img: np.ndarray) -> None:
    """Small colour key at the very bottom of the frame."""
    cv2.putText(img,
                "[cyan] = geometric FOV    [yellow] = colour seg    [red] = RCNN",
                (10, IMG_H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)
