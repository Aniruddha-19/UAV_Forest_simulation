"""
environment.py
==============
Responsible for initialising the PyBullet physics world and spawning
every physical object in the scene: ground plane, trees, egg masses,
and the drone body.

All functions return the PyBullet body ID(s) of the created objects so
the caller can reference them later (e.g. for collision filtering).
"""

import math
import os
import random

import pybullet as p
import pybullet_data

# Image textures randomly assigned to egg masses (files sit next to this script).
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_EGG_IMAGES = [
    os.path.join(_SCRIPT_DIR, "1.png"),
    os.path.join(_SCRIPT_DIR, "2.png"),
    os.path.join(_SCRIPT_DIR, "3.png"),
    os.path.join(_SCRIPT_DIR, "4.jpg"),
    os.path.join(_SCRIPT_DIR, "5.jpg"),
    os.path.join(_SCRIPT_DIR, "6.jpg"),
]


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


# ── Egg mass ──────────────────────────────────────────────────────────────────

def spawn_egg_mass(position: list,
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
        depth_half = 0.005 m  (1 cm panel thickness, thin dimension)
        face_half  = 0.0381 m (1.5 in — half of the 3-inch face)

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
    face_half  = 0.0381   # half of 3-inch face  (width and height of panel)
    depth_half = 0.005    # half-thickness of the panel (1 cm total depth)

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

    # Apply image texture when file exists
    if image_path and os.path.isfile(image_path):
        tex_id = p.loadTexture(image_path)
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

def build_scene(config: dict) -> list[dict]:
    """
    Spawn all trees and their egg masses from the config.

    Egg mass images cycle through 1.png → 2.png → 3.png → 1.png … across
    every egg mass in the scene.  Each panel is oriented to face outward from
    its parent trunk axis.

    Returns
    -------
    egg_masses : list of dicts, each with keys:
        'id'       – label string from config
        'position' – [x, y, z] world position
        'body'     – PyBullet body ID of the egg mass panel
    """
    egg_masses: list[dict] = []

    for tree_cfg in config["trees"]:
        spawn_tree(
            position     = tree_cfg["position"],
            height       = tree_cfg.get("height", 4.0),
            trunk_radius = tree_cfg.get("radius", 0.30),
        )
        trunk_pos = tree_cfg["position"]
        for em in tree_cfg.get("egg_masses", []):
            img_path = random.choice(_EGG_IMAGES)     # random image per egg mass
            body = spawn_egg_mass(
                position       = em["position"],
                trunk_position = trunk_pos,
                image_path     = img_path,
            )
            egg_masses.append({
                "id":       em["id"],
                "position": em["position"],
                "body":     body,
            })

    return egg_masses
