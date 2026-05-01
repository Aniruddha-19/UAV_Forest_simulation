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
from detector import get_active_model


# ── Frame annotation ──────────────────────────────────────────────────────────

def annotate(frame:        np.ndarray,
             rcnn_boxes:   list[tuple[int, int, int, int]],
             drone_pos:    np.ndarray,
             state:        str,
             tree_label:   str,
             tree_idx:     int,
             total_trees:  int,
             orbit_pct:    float,
             orbit_radius: float,
             frame_no:     int,
             saved_no:     int,
             detect_pct:   float = 0.0,
             infer_count:  int   = 0,
             n_detections: int | None = None) -> np.ndarray:
    """
    Compose all annotations onto a copy of *frame* and return the result.

    Layers (drawn in order):
      1. Faster R-CNN detections — red bounding rectangles
      2. Orbit progress bar     — shown only during INSPECT state
      3. HUD text panel         — state, tree, position, orbit radius, frame counters
      4. Detection banner       — bright green banner when RCNN detects adult SLF

    Parameters
    ----------
    frame        : processed BGR frame from camera.capture_and_process()
    rcnn_boxes   : list of (x1,y1,x2,y2) from detector.detect_adult_slf()
    drone_pos    : drone world position [x, y, z]
    state        : controller state string ("TRANSIT", "INSPECT", "HOME")
    tree_label   : ID string of the current tree, or "—"
    tree_idx     : 1-based index of the current tree
    total_trees  : total number of trees in the scene
    orbit_pct    : fraction of the current orbit completed (0.0 – 1.0)
    orbit_radius : current orbit radius in metres (shrinks during INSPECT)
    frame_no     : total frames rendered so far
    saved_no     : frames saved to disk so far (detection-only subset)
    fps          : configured capture frame rate

    Returns
    -------
    Annotated BGR image (same shape as *frame*).
    """
    out = frame.copy()

    if n_detections is None:
        n_detections = len(rcnn_boxes)

    _draw_rcnn_boxes(out, rcnn_boxes)
    _draw_orbit_bar(out, state, orbit_pct)
    _draw_hud(out, state, tree_label, tree_idx, total_trees,
               drone_pos, n_detections, orbit_radius,
               frame_no, saved_no, detect_pct, infer_count)
    _draw_detection_banner(out, rcnn_boxes)

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

def _draw_rcnn_boxes(img: np.ndarray,
                     rcnn_boxes: list[tuple[int, int, int, int]]) -> None:
    """
    Draw a red bounding rectangle for each adult SLF found by Faster R-CNN.
    """
    COLOUR = (0, 0, 220)   # red (BGR)
    for (x1, y1, x2, y2) in rcnn_boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), COLOUR, 2)
        cv2.putText(img, "adult SLF", (x1, y1 - 5),
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
              n_rcnn: int, orbit_radius: float,
              frame_no: int, saved_no: int,
              detect_pct: float = 0.0,
              infer_count: int = 0) -> None:
    """
    Draw the heads-up display text panel in the top-left corner.
    Text colour changes between TRANSIT (green) and INSPECT (cyan-blue).
    """
    COLOUR_INSPECT  = (0, 200, 255)
    COLOUR_TRANSIT  = (40, 255, 60)
    colour = COLOUR_INSPECT if state == "INSPECT" else COLOUR_TRANSIT

    radius_line = (f"Radius : {orbit_radius:.2f} m"
                   if state == "INSPECT"
                   else "Radius : —")

    model_label = (get_active_model() or "model").upper()
    hud_lines = [
        f"State  : {state}",
        f"Tree   : {tree_label}  ({tree_idx} / {total_trees})",
        f"Display: 30 fps",
        f"Infer  : waiting-time",
        f"Frame  : {frame_no}   Saved: {saved_no}",
        (f"Pos    : ({drone_pos[0]:.1f},  {drone_pos[1]:.1f},"
         f"  {drone_pos[2]:.1f}) m"),
        radius_line,
        f"{model_label:<7}: {n_rcnn} detection{'s' if n_rcnn != 1 else ''}",
        f"SLF    : {detect_pct:.1f}%  ({infer_count} inferences)",
    ]

    for i, line in enumerate(hud_lines):
        cv2.putText(img, line, (10, 22 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, colour, 1)


def _draw_detection_banner(img: np.ndarray,
                            rcnn_boxes: list) -> None:
    """
    Draw a bright green 'ADULT SLF DETECTED' banner in the top-right corner
    when Faster R-CNN finds at least one adult SLF.
    """
    if not rcnn_boxes:
        return
    cv2.rectangle(img, (IMG_W - 175, 5), (IMG_W - 5, 30), (0, 200, 0), -1)
    cv2.putText(img, "ADULT SLF DETECTED", (IMG_W - 173, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)


def _draw_legend(img: np.ndarray) -> None:
    """Small colour key at the very bottom of the frame."""
    cv2.putText(img,
                "[red] = Faster R-CNN detection",
                (10, IMG_H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)
