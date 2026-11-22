"""Software rasterizer: near-plane clipping, barycentric fill, Z-buffer.

Everything is computed with NumPy; Pygame only displays the finished image.
"""
import math

import numpy as np

NEAR = 0.1


def clip_near(tri, near=NEAR):
    """Sutherland-Hodgman clip of one camera-space triangle against z = near.

    Returns a list of triangles (3 x 3 floats each). Empty list if fully behind.
    """
    pts = [np.asarray(v, dtype=float) for v in tri]
    visible = []
    for start, end in zip(pts, pts[1:] + pts[:1]):
        inside_start = start[2] >= near
        inside_end = end[2] >= near
        if inside_start != inside_end:
            t = (near - start[2]) / (end[2] - start[2])
            visible.append(start + t * (end - start))
        if inside_end:
            visible.append(end)

    if len(visible) < 3:
        return []
    return [(visible[0], visible[i], visible[i + 1])
            for i in range(1, len(visible) - 1)]


def clip_near_attrs(tri, attrs, near=NEAR):
    """Same clipping as clip_near, but carries per-vertex attributes (UVs).

    A vertex created on a clipped edge gets its attributes linearly
    interpolated; in camera space the edge is a straight segment, so the
    interpolation is exact. Returns (3 points, 3 attributes) tuples.
    """
    pts = [np.asarray(v, dtype=float) for v in tri]
    ats = [np.asarray(a, dtype=float) for a in attrs]
    out_pts, out_ats = [], []
    for i in range(3):
        s, e = pts[i], pts[(i + 1) % 3]
        sa, ea = ats[i], ats[(i + 1) % 3]
        inside_s = s[2] >= near
        inside_e = e[2] >= near
        if inside_s != inside_e:
            t = (near - s[2]) / (e[2] - s[2])
            out_pts.append(s + t * (e - s))
            out_ats.append(sa + t * (ea - sa))
        if inside_e:
            out_pts.append(e)
            out_ats.append(ea)

    if len(out_pts) < 3:
        return []
    return [(out_pts[0], out_pts[i], out_pts[i + 1],
             out_ats[0], out_ats[i], out_ats[i + 1])
            for i in range(1, len(out_pts) - 1)]


def face_normal(a, b, c):
    n = np.cross(b - a, c - a)
    length = np.linalg.norm(n)
    if length < 1e-12:
        return None
    return n / length


def flat_shade(base_rgb, normal, light_dir, ambient=0.25, diffuse=0.75):
    """Ambient + diffuse (Lambert). light_dir points FROM surface TOWARD light."""
    intensity = ambient + diffuse * max(0.0, float(np.dot(normal, light_dir)))
    rgb = np.clip(np.asarray(base_rgb, dtype=float) * intensity, 0, 255)
    return rgb.astype(np.uint8)


class Renderer:
    """RGB framebuffer + per-pixel depth buffer."""

    def __init__(self, width, height, background=(14, 20, 28)):
        self.w = width
        self.h = height
        self.color = np.empty((height, width, 3), dtype=np.uint8)
        self.depth = np.empty((height, width), dtype=np.float64)
        self.background = background
        self.triangles = 0

    def clear(self):
        self.color[:] = self.background
        self.depth.fill(np.inf)
        self.triangles = 0

    @staticmethod
    def _edge(a, b, px, py):
        return (px - a[0]) * (b[1] - a[1]) - (py - a[1]) * (b[0] - a[0])

    def draw_triangle(self, a, b, c, rgb=None, uv=None, texture=None,
                      intensity=1.0):
        """a, b, c are (screen_x, screen_y, camera_depth) tuples.

        Flat colour: pass rgb. Textured: pass uv (three (u, v) pairs), texture
        (H x W x 3 array) and a diffuse intensity. UVs are interpolated
        perspective-correctly with u/z, v/z and 1/z (handbook 10.3), reusing the
        same barycentric weights as the depth buffer.
        """
        self.triangles += 1

        area = self._edge(a, b, c[0], c[1])
        if abs(area) < 1e-9:
            return
        if a[2] <= 0 or b[2] <= 0 or c[2] <= 0:
            return

        x0 = max(0, int(math.floor(min(a[0], b[0], c[0]))))
        x1 = min(self.w - 1, int(math.ceil(max(a[0], b[0], c[0]))))
        y0 = max(0, int(math.floor(min(a[1], b[1], c[1]))))
        y1 = min(self.h - 1, int(math.ceil(max(a[1], b[1], c[1]))))
        if x0 > x1 or y0 > y1:
            return

        px = (np.arange(x0, x1 + 1, dtype=np.float32) + 0.5)[None, :]
        py = (np.arange(y0, y1 + 1, dtype=np.float32) + 0.5)[:, None]

        # Broadcasting instead of meshgrid: same weights, no (h, w) temporaries
        wa = ((px - b[0]) * (c[1] - b[1]) - (py - b[1]) * (c[0] - b[0])) / area
        wb = ((px - c[0]) * (a[1] - c[1]) - (py - c[1]) * (a[0] - c[0])) / area
        wc = ((px - a[0]) * (b[1] - a[1]) - (py - a[1]) * (b[0] - a[0])) / area

        mask = (wa >= -1e-9) & (wb >= -1e-9) & (wc >= -1e-9)
        if not mask.any():
            return

        inv_z = wa / a[2] + wb / b[2] + wc / c[2]
        keep = mask & (inv_z > 0.0)
        if not keep.any():
            return

        z = np.where(keep, 1.0 / np.where(inv_z > 0, inv_z, 1.0), np.inf)

        depth_view = self.depth[y0:y1 + 1, x0:x1 + 1]
        win = keep & (z < depth_view)
        if not win.any():
            return

        depth_view[win] = z[win]

        if texture is None or uv is None:
            if rgb is None:
                raise ValueError("draw_triangle needs rgb or texture+uv")
            self.color[y0:y1 + 1, x0:x1 + 1][win] = rgb
            return

        tex_h, tex_w = texture.shape[:2]
        tex_flat = texture.reshape(-1, 3)
        (ua, va), (ub, vb), (uc, vc) = uv

        # Perspective-correct UV, evaluated only on visible pixels (handbook 10.3)
        iz = inv_z[win]
        waw, wbw, wcw = wa[win], wb[win], wc[win]
        u = (waw * ua / a[2] + wbw * ub / b[2] + wcw * uc / c[2]) / iz
        v = (waw * va / a[2] + wbw * vb / b[2] + wcw * vc / c[2]) / iz

        txi = ((u - np.floor(u)) * tex_w).astype(np.int32)
        tyi = ((v - np.floor(v)) * tex_h).astype(np.int32)
        np.clip(txi, 0, tex_w - 1, out=txi)
        np.clip(tyi, 0, tex_h - 1, out=tyi)

        cols = tex_flat[tyi * tex_w + txi].astype(np.float32)
        if intensity != 1.0:
            cols *= float(intensity)
        self.color[y0:y1 + 1, x0:x1 + 1][win] = np.clip(cols, 0, 255).astype(np.uint8)

    def to_surface(self, pygame):
        rgb = np.ascontiguousarray(self.color, dtype=np.uint8)
        return pygame.image.frombuffer(rgb.tobytes(), (self.w, self.h), "RGB")
