"""
logger.py
=========
Manages all file-based output for a simulation run:

  • Creates the run directory  (logs/YYYYMMDD_HHMMSS/)
  • Creates the frames sub-directory
  • Opens and writes the detections CSV log
  • Saves annotated JPEG frames when egg masses are detected
  • Prints the final summary to the terminal

CSV schema
----------
time_s        — seconds elapsed since simulation start
frame         — total rendered frame index
saved_frame   — index within the detection-only saved frames
egg_mass_id   — config ID of the detected egg mass, or blob count label
em_x/y/z     — world position of the egg mass (blank for seg-only rows)
drone_x/y/z  — drone world position at time of detection
state         — controller state ("TRANSIT" / "INSPECT" / "HOME")
detection_method — "geometric_fov" or "colour_segmentation"
"""

import csv
import time
from datetime import datetime
from pathlib import Path

import cv2


class SimulationLogger:
    """
    Handles the run directory, CSV log file, and frame saving for one
    simulation run.  Call open() at the start and close() (or use as a
    context manager) at the end.
    """

    CSV_HEADER = [
        "time_s", "frame", "saved_frame", "egg_mass_id",
        "em_x", "em_y", "em_z",
        "drone_x", "drone_y", "drone_z",
        "state", "detection_method",
    ]

    def __init__(self, base_log_dir: str = "logs"):
        timestamp      = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir   = Path(base_log_dir) / timestamp
        self.frames_dir = self.run_dir / "frames"
        self.csv_path   = self.run_dir / "detections.csv"

        self._csv_file   = None
        self._csv_writer = None
        self._t_start    = None

        # Counters (read-only from outside)
        self.frame_no  = 0   # total frames rendered
        self.saved_no  = 0   # frames saved to disk

    # ── Context manager support ───────────────────────────────────────────────

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.close()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def open(self) -> None:
        """Create directories, open the CSV file, and start the run timer."""
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self._csv_file   = open(self.csv_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(self.CSV_HEADER)
        self._t_start    = time.time()

    def close(self) -> None:
        """Flush and close the CSV file."""
        if self._csv_file:
            self._csv_file.close()
            self._csv_file = None

    # ── Logging methods ───────────────────────────────────────────────────────

    def log_geometric_detection(self,
                                 egg_id:    str,
                                 egg_pos:   list,
                                 drone_pos,
                                 state:     str) -> None:
        """
        Write one row for a geometrically confirmed egg mass sighting.

        Parameters
        ----------
        egg_id    : config ID string of the egg mass
        egg_pos   : [x, y, z] world position of the egg mass
        drone_pos : drone world position (numpy array or list)
        state     : current controller state string
        """
        self._csv_writer.writerow([
            f"{self.elapsed:.3f}",
            self.frame_no,
            self.saved_no,
            egg_id,
            *egg_pos,
            *list(drone_pos),
            state,
            "geometric_fov",
        ])
        self._csv_file.flush()

    def log_rcnn_detection(self,
                           n_boxes:   int,
                           drone_pos,
                           state:     str) -> None:
        """
        Write one row summarising the Faster R-CNN result for a frame.

        Parameters
        ----------
        n_boxes   : number of egg mass boxes found by the RCNN
        drone_pos : drone world position
        state     : current controller state string
        """
        self._csv_writer.writerow([
            f"{self.elapsed:.3f}",
            self.frame_no,
            self.saved_no,
            f"rcnn_{n_boxes}_boxes",
            "", "", "",            # no world coordinates for inference rows
            *list(drone_pos),
            state,
            "faster_rcnn",
        ])
        self._csv_file.flush()

    def log_segmentation_detection(self,
                                    n_blobs:   int,
                                    drone_pos,
                                    state:     str) -> None:
        """
        Write one row summarising the yellow-blob segmentation result for
        a frame (no world coordinates — purely image-level detection).

        Parameters
        ----------
        n_blobs   : number of yellow blobs found in the frame
        drone_pos : drone world position
        state     : current controller state string
        """
        self._csv_writer.writerow([
            f"{self.elapsed:.3f}",
            self.frame_no,
            self.saved_no,
            f"seg_{n_blobs}_blobs",
            "", "", "",            # no world coordinates for seg rows
            *list(drone_pos),
            state,
            "colour_segmentation",
        ])
        self._csv_file.flush()

    def save_frame(self, annotated_frame) -> None:
        """
        Save *annotated_frame* as a JPEG only when egg masses were detected.
        Increments saved_no and returns the file path used.
        """
        path = self.frames_dir / f"frame_{self.saved_no:06d}.jpg"
        cv2.imwrite(str(path), annotated_frame)
        self.saved_no += 1
        return path

    def tick_frame(self) -> None:
        """Call once per rendered frame (regardless of detections)."""
        self.frame_no += 1

    # ── Console output ────────────────────────────────────────────────────────

    def print_heartbeat(self,
                         drone_pos,
                         state:     str,
                         tree_label: str,
                         n_geo:     int,
                         n_seg:     int,
                         n_rcnn:    int,
                         capture_fps: int) -> None:
        """
        Print a one-line status update every 5 seconds of rendered time.
        """
        if self.frame_no % (capture_fps * 5) != 0:
            return
        print(
            f"  t={self.elapsed:6.1f}s | f={self.frame_no:5d} |"
            f" saved={self.saved_no:4d} | {state:8s} |"
            f" tree={tree_label:8s} |"
            f" Pos({drone_pos[0]:5.1f},{drone_pos[1]:5.1f},{drone_pos[2]:4.1f})"
            f" | geo={n_geo} seg={n_seg} rcnn={n_rcnn}"
        )

    def print_summary(self) -> None:
        """Print the end-of-run summary table."""
        elapsed = self.elapsed
        mins, secs = divmod(int(elapsed), 60)
        print("\n" + "=" * 60)
        print("  Simulation complete")
        print("=" * 60)
        print(f"  Total run time : {mins}m {secs:02d}s  ({elapsed:.1f} s)")
        print(f"  Frames total   : {self.frame_no}")
        print(f"  Frames saved   : {self.saved_no}  (RCNN detections only)")
        print(f"  Frames dir     : {self.frames_dir.resolve()}")
        print(f"  Detection log  : {self.csv_path.resolve()}")
        print("=" * 60)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def elapsed(self) -> float:
        """Seconds elapsed since open() was called."""
        return time.time() - self._t_start if self._t_start else 0.0
