"""
camera.py
=========
Everything related to the drone's onboard camera:

  • computing camera orientation vectors from drone pose
  • building PyBullet view / projection matrices
  • rendering a raw frame and applying the image-processing pipeline
  • detecting yellow egg masses via HSV colour segmentation
  • geometric FOV check — projects a 3-D world point into image space
    and returns the pixel coordinates if it falls within the frustum
"""

import math

import cv2
import numpy as np
import pybullet as p


# ── Camera constants ──────────────────────────────────────────────────────────

IMG_W       = 640     # rendered image width  (pixels)
IMG_H       = 480     # rendered image height (pixels)
CAM_FOV     = 60.0    # vertical field-of-view (degrees)
CAM_MAX_DIST   = 22.0  # egg masses beyond this distance are ignored (metres)
CAM_HEIGHT_TOL =  2.0  # max vertical separation (m) between drone and egg mass
                        # for a valid detection — enforces height line-of-sight


# ── Camera orientation ────────────────────────────────────────────────────────

def get_camera_vectors(drone_orn,
                       drone_pos: np.ndarray,
                       look_at:   np.ndarray | None
                       ) -> tuple[np.ndarray, np.ndarray]:
    """
    Derive the camera's forward and up unit vectors from the drone's pose.

    During transit (look_at = None)
        The camera points straight down (nadir view).

    During inspect (look_at = canopy centre)
        The camera points toward the canopy centre, so as the drone orbits
        the tree the viewing angle sweeps continuously around the canopy.

    Parameters
    ----------
    drone_orn : PyBullet quaternion (x, y, z, w)
    drone_pos : drone world position as a numpy array
    look_at   : world-space target point, or None for nadir

    Returns
    -------
    (cam_fwd, cam_up) — unit vectors in world space
    """
    rot    = np.array(p.getMatrixFromQuaternion(drone_orn)).reshape(3, 3)
    cam_up = rot @ np.array([0.0, 1.0, 0.0])   # drone body-Y → camera up

    if look_at is not None:
        direction = look_at - drone_pos
        dist      = np.linalg.norm(direction)
        cam_fwd   = (direction / dist
                     if dist > 0
                     else rot @ np.array([0.0, 0.0, -1.0]))
    else:
        cam_fwd = rot @ np.array([0.0, 0.0, -1.0])   # nadir

    return cam_fwd, cam_up


# ── PyBullet matrices ─────────────────────────────────────────────────────────

def build_camera_matrices(drone_pos: np.ndarray,
                          cam_fwd:   np.ndarray,
                          cam_up:    np.ndarray) -> tuple:
    """
    Build PyBullet view and projection matrices for the drone's camera.

    Parameters
    ----------
    drone_pos : camera eye position in world space
    cam_fwd   : unit vector pointing in the direction the camera faces
    cam_up    : unit vector pointing camera-up

    Returns
    -------
    (view_matrix, proj_matrix) — flat lists as returned by PyBullet
    """
    view = p.computeViewMatrix(
        cameraEyePosition    = drone_pos.tolist(),
        cameraTargetPosition = (drone_pos + cam_fwd).tolist(),
        cameraUpVector       = cam_up.tolist(),
    )
    proj = p.computeProjectionMatrixFOV(
        fov     = CAM_FOV,
        aspect  = IMG_W / IMG_H,
        nearVal = 0.1,
        farVal  = 150.0,
    )
    return view, proj


# ── Frame capture & image processing ─────────────────────────────────────────

def capture_and_process(view_matrix, proj_matrix) -> np.ndarray:
    """
    Render a frame from PyBullet and apply the image-processing pipeline:

      Step 1 — Raw render   : getCameraImage → RGBA numpy array
      Step 2 — CLAHE        : contrast-limited adaptive histogram equalisation
                              on the L-channel (LAB colour space)
      Step 3 — Sharpening   : unsharp-mask convolution kernel

    Returns
    -------
    Processed BGR image as a uint8 numpy array (H × W × 3).
    """
    # ── Step 1: render ───────────────────────────────────────────────────────
    _, _, rgba, _, _ = p.getCameraImage(
        IMG_W, IMG_H, view_matrix, proj_matrix,
        renderer=p.ER_TINY_RENDERER)
    bgr = cv2.cvtColor(
        np.array(rgba, dtype=np.uint8)[:, :, :3],
        cv2.COLOR_RGB2BGR)

    # ── Step 2: CLAHE contrast enhancement ───────────────────────────────────
    lab          = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_ch, a, b  = cv2.split(lab)
    clahe        = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_ch         = clahe.apply(l_ch)
    enhanced     = cv2.cvtColor(cv2.merge([l_ch, a, b]), cv2.COLOR_LAB2BGR)

    # ── Step 3: unsharp-mask sharpening ──────────────────────────────────────
    sharpen_kernel = np.array(
        [[ 0.0, -0.4,  0.0],
         [-0.4,  2.6, -0.4],
         [ 0.0, -0.4,  0.0]],
        dtype=np.float32)
    processed = cv2.filter2D(enhanced, -1, sharpen_kernel)

    return processed


# ── Egg mass detection — colour segmentation ──────────────────────────────────

def detect_yellow_blobs(frame_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    Detect yellow egg masses in a BGR frame using HSV colour segmentation.

    Pipeline:
      1. Convert BGR → HSV
      2. Threshold for yellow (H: 18–38, S: 100–255, V: 120–255)
      3. Morphological open  — removes small noise specks
      4. Morphological dilate — fills small holes in blobs
      5. Find contours and return bounding boxes for blobs > 15 px²

    Parameters
    ----------
    frame_bgr : processed BGR image from capture_and_process()

    Returns
    -------
    List of (x, y, w, h) bounding-box tuples, one per detected blob.
    """
    hsv  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,
                       np.array([18, 100, 120]),
                       np.array([38, 255, 255]))

    kernel = np.ones((3, 3), np.uint8)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,   kernel, iterations=1)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return [cv2.boundingRect(c)
            for c in contours
            if cv2.contourArea(c) > 15]


# ── Egg mass detection — geometric FOV check ──────────────────────────────────

def check_egg_in_fov(drone_pos:   np.ndarray,
                     cam_fwd:     np.ndarray,
                     cam_up:      np.ndarray,
                     egg_pos:     list,
                     egg_body_id: int
                     ) -> tuple[bool, int | None, int | None]:
    """
    Determine whether an egg mass is visible to the drone's camera.

    Two-stage check
    ---------------
    1. Geometric frustum check
       Projects the egg mass into image space using the camera basis vectors.
       Returns False immediately if the egg is behind the camera, farther than
       CAM_MAX_DIST, or outside the horizontal/vertical FOV bounds.

    2. Occlusion check  (ray-cast via PyBullet)
       Casts a ray from the drone to the egg mass.  If anything other than the
       egg mass itself is hit first (e.g. canopy sphere, trunk), the egg mass
       is considered occluded and the function returns False.

    Parameters
    ----------
    drone_pos   : camera/drone world position
    cam_fwd     : camera forward unit vector (world space)
    cam_up      : camera up unit vector (world space)
    egg_pos     : [x, y, z] world position of the egg mass
    egg_body_id : PyBullet body ID of the egg mass — used to confirm the ray
                  actually hits the egg and not something in front of it

    Returns
    -------
    (visible, pixel_x, pixel_y)
      visible   — True only when the egg is inside the FOV AND unoccluded
      pixel_x/y — projected image coordinates, or None when not visible
    """
    egg   = np.array(egg_pos, dtype=float)
    delta = egg - drone_pos
    dist  = np.linalg.norm(delta)

    # ── 1. Distance gate ─────────────────────────────────────────────────────
    if dist < 0.1 or dist > CAM_MAX_DIST:
        return False, None, None

    # ── 1b. Height line-of-sight gate ────────────────────────────────────────
    # The drone must be at approximately the same altitude as the egg mass.
    # This prevents detections when the drone flies overhead at cruise altitude
    # (18 m) while egg masses sit low on the trunk (0.4 – 1.8 m).
    if abs(drone_pos[2] - egg[2]) > CAM_HEIGHT_TOL:
        return False, None, None

    # ── 2. Build camera right axis ────────────────────────────────────────────
    cam_rgt = np.cross(cam_fwd, cam_up)
    rlen    = np.linalg.norm(cam_rgt)
    if rlen == 0:
        return False, None, None
    cam_rgt /= rlen

    # ── 3. Project delta onto camera axes ─────────────────────────────────────
    d_fwd = np.dot(delta, cam_fwd)
    if d_fwd <= 0:
        return False, None, None    # egg is behind the camera

    d_rgt = np.dot(delta, cam_rgt)
    d_up  = np.dot(delta, cam_up)

    # ── 4. Frustum check (normalised device coordinates) ─────────────────────
    half_fov_v = math.radians(CAM_FOV / 2)
    half_fov_h = math.atan(math.tan(half_fov_v) * IMG_W / IMG_H)

    ndc_x = d_rgt / (d_fwd * math.tan(half_fov_h))
    ndc_y = d_up  / (d_fwd * math.tan(half_fov_v))

    if not (-1.0 <= ndc_x <= 1.0 and -1.0 <= ndc_y <= 1.0):
        return False, None, None    # outside camera FOV

    # ── 5. Occlusion check — ray from drone to egg mass ───────────────────────
    hit_id, _hit_frac, _hit_frac2, _hit_pos, _hit_norm = \
        p.rayTest(drone_pos.tolist(), egg_pos)[0]

    # hit_id == -1  → nothing in the way (open air, egg collision tiny)
    # hit_id == egg_body_id → ray reached the egg without obstruction
    # anything else → canopy / trunk is blocking the view
    if hit_id not in (-1, egg_body_id):
        return False, None, None    # egg is occluded

    # ── 6. Compute pixel coordinates ──────────────────────────────────────────
    px = int((ndc_x + 1) / 2 * IMG_W)
    py = int((1 - ndc_y) / 2 * IMG_H)   # flip Y — image origin is top-left
    return True, px, py
