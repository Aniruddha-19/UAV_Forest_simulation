"""
environment.py
==============
Responsible for initialising the PyBullet physics world and spawning
every physical object in the scene: ground plane, trees, adult SLF panels,
and the drone body.

All functions return the PyBullet body ID(s) of the created objects so
the caller can reference them later (e.g. for collision filtering).
"""

import glob
import math
import os

import cv2
import pybullet as p
import pybullet_data

# Adult SLF images loaded from simulation_test_data/.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_TEST_DATA_DIR = os.path.join(_SCRIPT_DIR, "simulation_test_data")
_SLF_IMAGES = sorted(
    f for ext in ("*.png", "*.jpg", "*.jpeg")
    for f in glob.glob(os.path.join(_TEST_DATA_DIR, ext))
)

# PyBullet's loadTexture rejects some JPEG variants (CMYK, progressive,
# unusual color profiles, etc.). We re-encode every texture through OpenCV
# into a cache directory of plain baseline RGB JPEGs that always load.
_TEX_CACHE_DIR = os.path.join(_TEST_DATA_DIR, ".pybullet_tex_cache")
os.makedirs(_TEX_CACHE_DIR, exist_ok=True)


def _normalized_texture_path(src_path: str) -> str | None:
    """
    Return a path to a PyBullet-loadable copy of *src_path*. Cached on disk
    so the conversion runs at most once per image. Returns None if the
    source cannot be read.
    """
    base = os.path.splitext(os.path.basename(src_path))[0]
    out  = os.path.join(_TEX_CACHE_DIR, f"{base}.jpg")
    if os.path.isfile(out):
        return out
    img = cv2.imread(src_path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    cv2.imwrite(out, img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return out
if not _SLF_IMAGES:
    print("[environment] WARNING: no adult SLF images found in "
          "simulation_test_data/. Panels will render as plain yellow.")
else:
    print(f"[environment] Loaded {len(_SLF_IMAGES)} adult SLF image(s) "
          f"from simulation_test_data/")


# ── World initialisation ──────────────────────────────────────────────────────

def init_world(config: dict) -> None:
    """
    Connect to the PyBullet GUI, configure physics parameters, and load
    the flat ground plane.

    Parameters
    ----------
    config : dict
        Full simulation config loaded from config.json.
    """
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, config["simulation"]["gravity"])
    p.setTimeStep(config["simulation"]["time_step"])

    # Position the 3-D viewport for a good overview of the grove
    p.resetDebugVisualizerCamera(
        cameraDistance=35,
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=[0, 0, 0],
    )
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
    p.loadURDF("plane.urdf")


# ── Tree ─────────────────────────────────────────────────────────────────────

def spawn_tree(position: list, height: float, trunk_radius: float) -> tuple[int, int]:
    """
    Spawn a tree composed of two static bodies:
      • a brown cylinder  — the trunk
      • a green sphere    — the canopy

    The canopy radius is four times the trunk radius, and its centre sits
    65 % of a canopy-radius above the trunk top (giving a realistic overlap).

    Parameters
    ----------
    position     : [x, y, z]  ground-level XY position (z is ignored; set to 0)
    height       : trunk height in metres
    trunk_radius : trunk cross-section radius in metres

    Returns
    -------
    (trunk_id, canopy_id) — PyBullet body IDs
    """
    canopy_radius = trunk_radius * 4.0
    canopy_z      = height + canopy_radius * 0.65   # canopy centre height

    # ── Trunk ────────────────────────────────────────────────────────────────
    trunk_col = p.createCollisionShape(
        p.GEOM_CYLINDER, radius=trunk_radius, height=height)
    trunk_vis = p.createVisualShape(
        p.GEOM_CYLINDER, radius=trunk_radius, length=height,
        rgbaColor=[0.38, 0.19, 0.05, 1.0])          # dark brown
    trunk_id = p.createMultiBody(
        baseMass=0,                                  # static object
        baseCollisionShapeIndex=trunk_col,
        baseVisualShapeIndex=trunk_vis,
        basePosition=[position[0], position[1], height / 2])

    # ── Canopy ───────────────────────────────────────────────────────────────
    canopy_col = p.createCollisionShape(p.GEOM_SPHERE, radius=canopy_radius)
    canopy_vis = p.createVisualShape(
        p.GEOM_SPHERE, radius=canopy_radius,
        rgbaColor=[0.08, 0.48, 0.10, 0.92])          # forest green, slightly transparent
    canopy_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=canopy_col,
        baseVisualShapeIndex=canopy_vis,
        basePosition=[position[0], position[1], canopy_z])

    return trunk_id, canopy_id


# ── Adult SLF panel ───────────────────────────────────────────────────────────

def spawn_slf_panel(position: list,
                    trunk_position: list,
                    image_path: str | None = None) -> int:
    """
    Spawn a flat 3-inch-square image panel flush against the trunk surface.

    The panel is a thin box whose face (local Y-Z plane) is perpendicular to
    the radial direction from the trunk axis, so the image is clearly visible
    as the drone orbits the tree.

    Geometry
    --------
      halfExtents = [depth_half, face_half, face_half]
        depth_half = 0.0005 m  (1 mm panel thickness — near-flat decal)
        face_half  = 0.12 m    (half of 24 cm face — >32 px at RCNN min anchor)

      The box's local +X axis is rotated to point radially outward from the
      trunk so the large ±X faces (which carry the texture) face outward and
      inward relative to the trunk.

    Parameters
    ----------
    position       : [x, y, z]  world-space centre of the panel
    trunk_position : [x, y, z]  ground position of the parent trunk (z ignored)
    image_path     : absolute path to the texture image; yellow fallback if
                     None or the file does not exist

    Returns
    -------
    body_id — PyBullet body ID
    """
    face_half  = 0.12     # half of 24 cm face — gives ~90 px at 1 m orbit distance,
                          # safely above Faster R-CNN FPN's 32 px minimum anchor
    depth_half = 0.0005   # half-thickness of the panel (1 mm total depth — near-flat decal)

    # Outward angle from trunk axis → rotation around world Z
    dx    = position[0] - trunk_position[0]
    dy    = position[1] - trunk_position[1]
    angle = math.atan2(dy, dx)
    orn   = p.getQuaternionFromEuler([0.0, 0.0, angle])

    # Thin box: local X = depth (points away from trunk), Y/Z = face
    col = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=[depth_half, face_half, face_half])
    vis = p.createVisualShape(
        p.GEOM_BOX, halfExtents=[depth_half, face_half, face_half],
        rgbaColor=[1.0, 0.95, 0.05, 1.0])           # bright yellow fallback
    body_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=position,
        baseOrientation=orn)

    # Apply image texture when file exists. Route through the cache so
    # PyBullet always sees a plain baseline RGB JPEG it can decode.
    if image_path and os.path.isfile(image_path):
        norm_path = _normalized_texture_path(image_path)
        if norm_path is not None:
            tex_id = p.loadTexture(norm_path)
            p.changeVisualShape(body_id, -1, textureUniqueId=tex_id)

    return body_id


# ── Drone body ────────────────────────────────────────────────────────────────

def spawn_drone(position: list) -> int:
    """
    Spawn the UAV as a flat blue box.

    The drone is kinematically controlled (mass = 0) — its pose is set
    directly every step via resetBasePositionAndOrientation, so no rotor
    physics are simulated.

    Parameters
    ----------
    position : [x, y, z]  initial world position

    Returns
    -------
    drone_id — PyBullet body ID
    """
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.28, 0.28, 0.07])
    vis = p.createVisualShape(
        p.GEOM_BOX, halfExtents=[0.28, 0.28, 0.07],
        rgbaColor=[0.12, 0.12, 0.82, 1.0])           # deep blue
    return p.createMultiBody(
        baseMass=0,                                   # kinematic body
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=position,
        baseOrientation=[0, 0, 0, 1])


# ── Scene builder (convenience wrapper) ───────────────────────────────────────

def _make_panel_layout(n_panels: int,
                       trunk_height: float,
                       n_angles: int = 4) -> list[tuple[float, float]]:
    """
    Return (angle_rad, height_m) pairs for n_panels panels on one trunk.

    Panels are arranged on a grid: n_angles columns evenly spaced around the
    trunk × enough height rows to cover n_panels.  Heights run from 0.35 m up
    to min(2.5 m, trunk_height − 0.30 m) so panels stay within the drone's
    camera view during inspection.
    """
    h_min = 0.35
    h_max = max(h_min + 0.10, min(2.50, trunk_height - 0.30))

    n_heights = math.ceil(n_panels / n_angles)
    if n_heights == 1:
        heights = [h_min]
    else:
        heights = [h_min + (h_max - h_min) * k / (n_heights - 1)
                   for k in range(n_heights)]

    layout: list[tuple[float, float]] = []
    for h in heights:
        for a_i in range(n_angles):
            if len(layout) >= n_panels:
                break
            angle = 2.0 * math.pi * a_i / n_angles
            layout.append((angle, h))

    return layout


def build_scene(config: dict) -> list[dict]:
    """
    Spawn all trees and auto-generate one image panel per image found in
    simulation_test_data/, distributing them evenly across all trees at
    varying heights and angular positions.

    Returns
    -------
    slf_panels : list of dicts, each with keys:
        'id'         : label string  (e.g. "T1-A", "T2-K")
        'tree_id'    : id of the parent tree
        'position'   : [x, y, z] world position of the panel centre
        'body'       : PyBullet body ID of the panel
        'image_path' : absolute path to the assigned image (may be None)
    """
    slf_panels: list[dict] = []

    trees   = config["trees"]
    n_trees = len(trees)
    n_total = len(_SLF_IMAGES)

    if n_total == 0:
        print("[environment] No images found — spawning trees without panels.")

    base  = n_total // n_trees if n_trees else 0
    extra = n_total % n_trees  if n_trees else 0
    # first `extra` trees receive base+1 panels, the rest receive base

    img_idx = 0
    for t_i, tree_cfg in enumerate(trees):
        spawn_tree(
            position     = tree_cfg["position"],
            height       = tree_cfg.get("height", 4.0),
            trunk_radius = tree_cfg.get("radius", 0.30),
        )

        trunk_pos    = tree_cfg["position"]
        trunk_radius = tree_cfg.get("radius", 0.30)
        trunk_height = tree_cfg.get("height", 4.0)
        n_panels     = base + (1 if t_i < extra else 0)

        if n_panels == 0:
            continue

        layout = _make_panel_layout(n_panels, trunk_height)

        for p_i, (angle, height) in enumerate(layout):
            r      = trunk_radius + 0.005
            em_pos = [
                trunk_pos[0] + r * math.cos(angle),
                trunk_pos[1] + r * math.sin(angle),
                height,
            ]
            img_path = _SLF_IMAGES[img_idx] if _SLF_IMAGES else None
            img_idx += 1

            em_id = f"T{t_i + 1}-{chr(65 + p_i)}"
            body  = spawn_slf_panel(
                position       = em_pos,
                trunk_position = trunk_pos,
                image_path     = img_path,
            )
            slf_panels.append({
                "id":         em_id,
                "tree_id":    tree_cfg["id"],
                "position":   em_pos,
                "body":       body,
                "image_path": img_path,
            })

    print(f"[environment] Spawned {len(slf_panels)} panels across "
          f"{n_trees} trees ({base}–{base + (1 if extra else 0)} per tree).")
    return slf_panels
