import numpy as np

def translation(tx, ty, tz):
    M = np.eye(4)
    M[:3, 3] = [tx, ty, tz]
    return M

def rotation_y(angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[ c, 0, s, 0],
                     [ 0, 1, 0, 0],
                     [-s, 0, c, 0],
                     [ 0, 0, 0, 1]], dtype=float)

def scale(sx, sy, sz):
    return np.diag([sx, sy, sz, 1.0])

def transform_point(M, xyz):
    v = np.append(np.asarray(xyz, dtype=float), 1.0)
    return (M @ v)[:3]

def transform_direction(M, xyz):
    v = np.append(np.asarray(xyz, dtype=float), 0.0)
    return (M @ v)[:3]
