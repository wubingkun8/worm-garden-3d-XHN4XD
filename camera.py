import math
import numpy as np


def normalized(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-8:
        raise ValueError('Zero-length vector')
    return v / n


def camera_space(point, eye, target):
    """World point -> camera coordinates (X right, Y up, Z forward)."""
    point = np.asarray(point, dtype=float)
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)

    forward = normalized(target - eye)
    right = normalized(np.cross([0., 1., 0.], forward))
    up = np.cross(forward, right)

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
