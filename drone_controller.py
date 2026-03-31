"""
drone_controller.py
====================
Kinematic drone controller implementing a TRANSIT → INSPECT state machine.

State diagram
─────────────
  ┌─────────┐   arrived at      ┌─────────┐   orbit      ┌──────┐   arrived  ┌──────┐
  │ TRANSIT ├──orbit entry────► │ INSPECT ├──complete──► │  ... ├──at home──► │ DONE │
  └─────────┘                   └─────────┘              └──────┘             └──────┘
       ▲                              │
       └───────────────── next tree ──┘

TRANSIT
  The drone flies at cruise altitude to the orbit entry point (east side,
  angle = 0) of the current target tree.  A five-ray obstacle-avoidance fan
  blends repulsion vectors into the forward heading to steer clear of trunks
  and canopies during transit.

INSPECT
  The drone circles the tree at a fixed radius and cruise altitude.
  Position is computed analytically from the orbit angle each step, so no
  avoidance is needed (the radius already keeps the drone outside the canopy).
  The drone body always faces inward toward the tree centre.

HOME
  After all trees are inspected the drone transits back to its start position.
"""

import math

import numpy as np
import pybullet as p


class DroneController:

    # ── State labels ──────────────────────────────────────────────────────────
    TRANSIT = "TRANSIT"   # flying at cruise altitude toward the next tree
    DESCEND = "DESCEND"   # arrived above tree, now descending to inspect altitude
    INSPECT = "INSPECT"   # orbiting the trunk at inspect altitude
    ASCEND  = "ASCEND"    # orbit done, climbing back to cruise altitude
    HOME    = "HOME"      # returning to start position
    DONE    = "DONE"      # mission complete

    # ── Tuning ────────────────────────────────────────────────────────────────
    WP_TOLERANCE  = 0.40   # metres — distance at which a waypoint is considered reached
    AVOID_RANGE   = 3.0    # metres — how far ahead rays are cast for obstacle detection
    AVOID_WEIGHT  = 1.0    # how strongly avoidance repulsion deflects the heading

    # ─────────────────────────────────────────────────────────────────────────

    def __init__(self, drone_id: int, config: dict, trees_cfg: list):
        """
        Parameters
        ----------
        drone_id   : PyBullet body ID of the drone
        config     : full simulation config dict
        trees_cfg  : list of tree config dicts (config["trees"])
        """
        self.drone_id       = drone_id
        self.trees_cfg      = trees_cfg
        self.tree_idx       = 0           # index of the tree currently targeted

        # Physics / timing
        self.dt             = config["simulation"]["time_step"]

        # Drone parameters
        self.cruise_alt     = config["drone"]["cruise_altitude"]
        self.inspect_alt    = config["drone"].get("inspect_altitude", 2.0)     # m  ← trunk height
        self.transit_speed  = config["drone"]["transit_speed"]                 # m/s
        self.inspect_radius   = config["drone"].get("inspect_radius", 3.0)       # m  (canopy clearance during transit)
        self.close_clearance  = config["drone"].get("inspect_trunk_clearance", 1.0)  # m  (trunk clearance during inspection)
        self.orbit_speed      = math.radians(
            config["drone"].get("inspect_orbit_speed_deg", 20))               # rad/s
        self.slow_orbit_speed = math.radians(
            config["drone"].get("inspect_slow_orbit_speed_deg", 5))           # rad/s — used when egg mass detected

        self.home_pos = np.array(config["drone"]["start_position"], dtype=float)

        # Internal state
        self.state          = self.TRANSIT
        self.current_yaw    = 0.0    # drone body yaw in world frame (radians)
        self._orbit_angle   = 0.0    # current angle on the orbit circle (radians)
        self._orbit_accum   = 0.0    # radians completed in the current orbit
        self.egg_detected   = False  # set True by main loop when an egg mass is in view

    # ── Public interface ──────────────────────────────────────────────────────

    @property
    def done(self) -> bool:
        """True once the drone has returned home and the mission is complete."""
        return self.state == self.DONE

    @property
    def current_tree(self) -> dict | None:
        """Config dict of the tree currently being targeted, or None if done."""
        if self.tree_idx < len(self.trees_cfg):
            return self.trees_cfg[self.tree_idx]
        return None

    @property
    def orbit_progress(self) -> float:
        """Fraction of the current orbit completed (0.0 – 1.0)."""
        return self._orbit_accum / (2 * math.pi)

    def _effective_radius(self, tree: dict) -> float:
        """
        Collision-safe orbit radius for a specific tree.

        The canopy is a sphere with radius = trunk_radius × 4  (matches
        spawn_tree in environment.py).  self.inspect_radius is the desired
        clearance *beyond* the canopy edge, so the drone never enters the
        canopy collision shape regardless of tree size.

        effective_radius = canopy_radius + inspect_radius_clearance
        """
        canopy_radius = tree.get("radius", 0.30) * 4.0
        return canopy_radius + self.inspect_radius

    def _close_radius(self, tree: dict) -> float:
        """
        Orbit radius used at inspect altitude — just outside the trunk.
        Much smaller than _effective_radius so the camera is close enough
        for Faster R-CNN to resolve egg masses on the bark.
        """
        return tree.get("radius", 0.30) + self.close_clearance

    def trunk_look_at(self, tree: dict) -> list[float]:
        """
        The point the camera looks at during trunk inspection.
        Aimed at mid-trunk height so egg masses (low on the trunk) are
        centred in the camera frame.
        """
        pos = tree["position"]
        return [pos[0], pos[1], self.inspect_alt * 0.5]

    def orbit_entry_point(self, tree: dict) -> np.ndarray:
        """
        The point the drone flies to (at cruise altitude) before descending
        to inspect altitude.  Placed on the east side of the tree (angle = 0).
        Uses the per-tree effective radius so the entry point is always
        outside the canopy collision sphere.
        """
        tp = tree["position"]
        return np.array(
            [tp[0] + self._effective_radius(tree), tp[1], self.cruise_alt],
            dtype=float)

    def pose(self) -> tuple[np.ndarray, tuple]:
        """Return current (position, orientation) from PyBullet."""
        pos, orn = p.getBasePositionAndOrientation(self.drone_id)
        return np.array(pos, dtype=float), orn

    def step(self) -> None:
        """
        Advance the drone by one simulation time-step.
        Call this once per p.stepSimulation().
        """
        if self.done:
            return

        pos, _ = self.pose()

        if self.state == self.TRANSIT:
            self._step_transit(pos)
        elif self.state == self.DESCEND:
            self._step_descend(pos)
        elif self.state == self.INSPECT:
            self._step_inspect()
        elif self.state == self.ASCEND:
            self._step_ascend(pos)
        elif self.state == self.HOME:
            self._step_home(pos)

    # ── State handlers ────────────────────────────────────────────────────────

    def _step_transit(self, pos: np.ndarray) -> None:
        """
        Fly at cruise altitude toward the orbit entry point above the current
        tree.  When arrived (XY + Z within tolerance) switch to DESCEND.
        If all trees are done, switch to HOME.
        """
        tree = self.current_tree
        if tree is not None:
            target = self.orbit_entry_point(tree)   # cruise altitude, east side
        else:
            target = np.array(
                [self.home_pos[0], self.home_pos[1], self.cruise_alt],
                dtype=float)

        arrived = self._move_toward(pos, target)

        if arrived:
            if tree is not None:
                self.state = self.DESCEND
                print(f"  [DESCEND] Above tree {self.tree_idx + 1}/{len(self.trees_cfg)}"
                      f"  →  id = {tree['id']}")
            else:
                self.state = self.HOME

    def _step_descend(self, pos: np.ndarray) -> None:
        """
        Glide diagonally inward from (effective_radius, cruise_altitude) to
        (close_radius, inspect_altitude).  Moving closer to the trunk as we
        descend puts the camera near enough for Faster R-CNN to resolve egg
        masses on the bark.
        """
        tree    = self.current_tree
        tp      = tree["position"]
        close_r = self._close_radius(tree)
        # Angle = 0 (east side) — matches orbit_entry_point
        target  = np.array(
            [tp[0] + close_r, tp[1], self.inspect_alt],
            dtype=float)

        # Use direct movement — avoidance rays fire against the trunk and push
        # the drone back, preventing convergence to close_radius.
        arrived = self._move_direct(pos, target)

        if arrived:
            self._orbit_angle = 0.0
            self._orbit_accum = 0.0
            self.state = self.INSPECT
            print(f"  [INSPECT]  Orbiting trunk  id = {tree['id']}"
                  f"  alt = {self.inspect_alt} m  r = {close_r:.2f} m")

    def _step_inspect(self) -> None:
        """
        Circle the tree trunk at inspect_altitude.  Position is computed
        analytically from the orbit angle — smooth, no jitter.
        Drone faces inward; camera looks at the trunk at mid-egg-mass height.
        Switches to ASCEND when a full 360° orbit is complete.
        """
        tree = self.current_tree
        tp   = np.array(tree["position"], dtype=float)

        r = self._close_radius(tree)   # orbit close to the trunk at ground level

        # Slow down rotation when an egg mass is actively in view
        speed = self.slow_orbit_speed if self.egg_detected else self.orbit_speed

        delta_angle        = speed * self.dt
        self._orbit_angle += delta_angle
        self._orbit_accum += delta_angle

        x = tp[0] + r * math.cos(self._orbit_angle)
        y = tp[1] + r * math.sin(self._orbit_angle)
        z = self.inspect_alt                          # ← trunk-level altitude

        # Face inward toward the trunk
        yaw = math.atan2(tp[1] - y, tp[0] - x)
        self.current_yaw = yaw
        orn = p.getQuaternionFromEuler([0.0, 0.0, yaw])
        p.resetBasePositionAndOrientation(self.drone_id, [x, y, z], orn)

        if self._orbit_accum >= 2 * math.pi:
            self.state = self.ASCEND
            print(f"  [ASCEND]  Orbit complete — climbing to cruise altitude")

    def _step_ascend(self, pos: np.ndarray) -> None:
        """
        Glide diagonally outward from (close_radius, inspect_altitude) to
        (effective_radius, cruise_altitude).  Moving farther from the trunk as
        we climb gives canopy clearance before the next transit leg.
        """
        tree = self.current_tree
        if tree is not None:
            tp    = np.array(tree["position"], dtype=float)
            eff_r = self._effective_radius(tree)
            # Preserve the orbit angle so the outward path follows the same
            # radial direction as where the orbit finished
            target = np.array([
                tp[0] + eff_r * math.cos(self._orbit_angle),
                tp[1] + eff_r * math.sin(self._orbit_angle),
                self.cruise_alt,
            ], dtype=float)
        else:
            target = np.array([pos[0], pos[1], self.cruise_alt], dtype=float)

        arrived = self._move_toward(pos, target)

        if arrived:
            self.tree_idx += 1
            self.state = self.TRANSIT
            if self.tree_idx < len(self.trees_cfg):
                print(f"  [TRANSIT] → Tree {self.tree_idx + 1}/{len(self.trees_cfg)}")
            else:
                print("  [TRANSIT] → Home")

    def _step_home(self, pos: np.ndarray) -> None:
        """Fly back to the home/start position and mark the mission done."""
        arrived = self._move_toward(pos, self.home_pos)
        if arrived:
            self.state = self.DONE
            print("  [DONE]  Returned home — mission complete.")

    # ── Movement helpers ──────────────────────────────────────────────────────

    def _move_direct(self, pos: np.ndarray, target: np.ndarray) -> bool:
        """
        Move one step toward *target* at transit speed WITHOUT obstacle avoidance.
        Used for DESCEND/ASCEND legs whose paths are geometrically pre-validated
        to be clear of all obstacles, so avoidance rays would only push the drone
        away from the intended close-approach target.

        Returns True when the drone is within WP_TOLERANCE of the target.
        """
        delta = target - pos
        dist  = np.linalg.norm(delta)

        if dist < self.WP_TOLERANCE:
            return True

        forward   = delta / dist
        step_size = min(self.transit_speed * self.dt, dist)
        new_pos   = pos + forward * step_size

        self.current_yaw = math.atan2(float(forward[1]), float(forward[0]))
        orn = p.getQuaternionFromEuler([0.0, 0.0, self.current_yaw])
        p.resetBasePositionAndOrientation(self.drone_id, new_pos.tolist(), orn)
        return False

    def _move_toward(self, pos: np.ndarray, target: np.ndarray) -> bool:
        """
        Move one step toward *target* at transit speed.
        Obstacle avoidance rays deflect the heading when an object is detected.

        Returns True when the drone is within WP_TOLERANCE of the target.
        """
        delta = target - pos
        dist  = np.linalg.norm(delta)

        if dist < self.WP_TOLERANCE:
            return True                          # waypoint reached

        forward   = delta / dist
        avoidance = self._cast_avoidance_rays(pos, forward)

        # Blend forward direction with avoidance repulsion, then normalise
        move_dir = forward + avoidance * self.AVOID_WEIGHT
        norm     = np.linalg.norm(move_dir)
        if norm > 0:
            move_dir /= norm

        # Step forward (never overshoot)
        step_size = min(self.transit_speed * self.dt, dist)
        new_pos   = pos + move_dir * step_size

        # Update yaw to face the direction of travel
        self.current_yaw = math.atan2(float(move_dir[1]), float(move_dir[0]))
        orn = p.getQuaternionFromEuler([0.0, 0.0, self.current_yaw])
        p.resetBasePositionAndOrientation(self.drone_id, new_pos.tolist(), orn)
        return False

    def _cast_avoidance_rays(self,
                              pos: np.ndarray,
                              forward: np.ndarray) -> np.ndarray:
        """
        Cast a fan of five rays ahead of the drone.
        Each ray that hits an obstacle (other than the drone itself) contributes
        a horizontal repulsion vector proportional to how close the hit is.

        Returns a combined avoidance vector (horizontal only — Z = 0).
        """
        avoidance = np.zeros(3, dtype=float)

        # Centre ray + four diagonally offset rays for a wider detection cone
        ray_offsets = [
            np.zeros(3),
            np.array([ 0.25, 0.00, 0.0]),
            np.array([-0.25, 0.00, 0.0]),
            np.array([ 0.00, 0.25, 0.0]),
            np.array([ 0.00,-0.25, 0.0]),
        ]

        for offset in ray_offsets:
            origin  = pos + offset
            ray_end = origin + forward * self.AVOID_RANGE
            hit_id, _, hit_fraction, hit_pos, _ = \
                p.rayTest(origin.tolist(), ray_end.tolist())[0]

            # Ignore misses and hits on the drone's own body
            if hit_id in (-1, self.drone_id):
                continue

            repel = pos - np.array(hit_pos, dtype=float)
            repel[2] = 0.0                        # keep avoidance horizontal
            length = np.linalg.norm(repel)
            if length > 0:
                # Stronger repulsion the closer the hit
                strength   = 1.0 - hit_fraction
                avoidance += (repel / length) * strength

        avoidance[2] = 0.0
        return avoidance
