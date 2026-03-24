# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Run the simulation:**
```bash
# activate virtualenv first
source /home/boubin-lab/Spott/bin/activate
python simulation.py
python simulation.py --config config.json --fps 5
```

**Dependencies** (all in Spott virtualenv):
```
pybullet  opencv-python  numpy
```

## Architecture

Single-file simulation (`simulation.py`). The drone visits each tree in sequence,
performs a full 360° yaw scan above it, then transits to the next tree.

### Navigation State Machine (`DroneController`)

```
TRANSIT → INSPECT → TRANSIT → INSPECT → … → HOME → DONE
```

- **TRANSIT** — flies directly to the point above the next tree at `cruise_altitude`,
  with 5-ray fan obstacle avoidance blended into the heading.
- **INSPECT** — hovers in place, increments yaw by `inspect_yaw_speed_deg °/s` each step
  until a full 360° is accumulated, then advances to the next tree.
- **HOME** — flies back to `start_position` after all trees are visited.

Speed is slow: `transit_speed = 1.0 m/s`.  One inspection takes ~12 s at 30 °/s.

### Camera

- **Transit**: camera looks straight down (nadir).
- **Inspect**: camera looks at the canopy centre (`look_at` target).
  As the drone yaws, the viewing angle sweeps the full canopy.
- `get_camera_vectors()` returns `(cam_fwd, cam_up)` used by both
  `camera_matrices()` (rendering) and `egg_in_fov()` (geometric check).

### Vision pipeline (per frame)

1. CLAHE contrast enhancement on L channel (LAB colour space)
2. Unsharp-mask sharpening
3. Geometric FOV check — projects each egg mass into image space; logs world XYZ
4. HSV colour segmentation — yellow blobs (HSV [18–38, 100–255, 120–255], min 15 px²)
5. **Frame saved to disk ONLY if at least one detection (geo or seg) is found**

### Configuration (`config.json`)

| Key | Description |
|-----|-------------|
| `trees[].position` | XY ground position, Z=0 |
| `trees[].height` | Trunk height (m) |
| `trees[].radius` | Trunk radius (m); canopy = ×4 |
| `trees[].egg_masses[].position` | World XYZ of each egg mass |
| `drone.start_position` | Take-off / landing position |
| `drone.cruise_altitude` | Flight altitude during transit and inspect (m) |
| `drone.transit_speed` | Translation speed (m/s) |
| `drone.inspect_yaw_speed_deg` | Yaw rate during inspection (°/s) |
| `drone.capture_fps` | Camera capture rate |
| `simulation.time_step` | Physics step (0.01 s = 100 Hz) |

### Key Tuning Constants (top of `simulation.py`)

| Constant | Value | Purpose |
|---|---|---|
| `IMG_W`, `IMG_H` | 640×480 | Camera resolution |
| `CAM_FOV` | 60° | Vertical field of view |
| `CAM_MAX_DIST` | 22 m | Max logging distance |
| `AVOID_RANGE` | 3.0 m | Ray-cast look-ahead |
| `AVOID_WEIGHT` | 1.0 | Avoidance blending |
| `WP_TOL` | 0.40 m | Waypoint acceptance radius |

### Detection CSV Schema

`time_s`, `frame`, `saved_frame`, `egg_mass_id`,
`em_x`, `em_y`, `em_z`, `drone_x`, `drone_y`, `drone_z`,
`state`, `detection_method`

`detection_method` → `geometric_fov` | `colour_segmentation`
