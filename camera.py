import math
import numpy as np


def normalized(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-8:
        raise ValueError('Zero-length vector')
    return v / n


def camera_basis(eye, target):
    """Right, up and forward vectors of the camera.

    right = forward x world_up, so with forward = -Z (looking north) the right
    vector is +X: east appears on the right, exactly as in the 2D game.
    """
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    forward = normalized(target - eye)
    right = normalized(np.cross(forward, [0., 1., 0.]))
    up = np.cross(right, forward)
    return right, up, forward


def camera_space(point, eye, target):
    """World point -> camera coordinates (X right, Y up, Z forward)."""
    point = np.asarray(point, dtype=float)
    eye = np.asarray(eye, dtype=float)
    right, up, forward = camera_basis(eye, target)
    d = point - eye
    return np.array([np.dot(d, right),
                     np.dot(d, up),
                     np.dot(d, forward)])


def perspective_project(c, width, height, fov_y_deg=55.0, near=0.1):
    """Camera-space point -> (screen_x, screen_y, depth) or None if clipped."""
    x, y, depth = c
    if depth <= near:
        return None
    focal = (height / 2) / math.tan(math.radians(fov_y_deg) / 2)
    return (width / 2 + focal * x / depth,
            height / 2 - focal * y / depth,
            depth)


def orthographic_project(c, width, height, pixels_per_unit=45.0, near=0.1):
    """Camera-space point -> (screen_x, screen_y, depth); size ignores depth."""
    x, y, depth = c
    if depth <= near:
        return None
    return (width / 2 + pixels_per_unit * x,
            height / 2 - pixels_per_unit * y,
            depth)


def orbit_eye(cx, cz, distance, height, angle_rad):
    """Orbit the camera around (cx, 0, cz) at a fixed height."""
    return np.array([cx + distance * math.cos(angle_rad),
                     height,
                     cz + distance * math.sin(angle_rad)], dtype=float)


def ray_direction(px, py, width, height, fov_y_deg=55.0):
    """Screen pixel -> camera-space direction with Z = 1 (not normalised).

    The inverse of perspective_project(): a click at (px, py) leaves the eye
    along this direction, so the focal length cancels out of the ratio.
    """
    focal = (height / 2) / math.tan(math.radians(fov_y_deg) / 2)
    return np.array([(px - width / 2) / focal,
                     (height / 2 - py) / focal,
                     1.0])


def ground_intersection(eye, target, px, py, width, height, fov_y_deg=55.0,
                        plane_y=0.0, max_t=40.0):
    """World point where the ray through (px, py) meets the plane y = plane_y.

    This is the 3D meaning of a mouse click (handbook 13.2): the click picks a
    world position on the ground instead of a screen position. Returns None when
    the ray points at the sky, behind the camera or past max_t units, so a click
    on the horizon is simply ignored.
    """
    right, up, forward = camera_basis(eye, target)
    local = ray_direction(px, py, width, height, fov_y_deg)
    direction = normalized(local[0] * right + local[1] * up + local[2] * forward)
    if direction[1] > -1e-6:
        return None
    eye = np.asarray(eye, dtype=float)
    t = (plane_y - eye[1]) / direction[1]
    if t <= 0.0 or t > max_t:
        return None
    return eye + t * direction
