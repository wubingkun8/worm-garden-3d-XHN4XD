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

        px = (np.arange(x0, x1 + 1, dtype=np.float32) + np.float32(0.5))[None, :]
        py = (np.arange(y0, y1 + 1, dtype=np.float32) + np.float32(0.5))[:, None]

        # float32 scalars on purpose: mixing a float32 grid with Python floats
        # promotes every temporary to float64 and doubles the memory traffic
        ax, ay = np.float32(a[0]), np.float32(a[1])
        bx, by = np.float32(b[0]), np.float32(b[1])
        cx, cy = np.float32(c[0]), np.float32(c[1])
        area = np.float32(area)

        # Broadcasting instead of meshgrid: same weights, no (h, w) temporaries.
        # wc = 1 - wa - wb because the barycentric weights always sum to one, so
        # only two edge functions are evaluated here.
        wa = ((px - bx) * (cy - by) - (py - by) * (cx - bx)) / area
        wb = ((px - cx) * (ay - cy) - (py - cy) * (ax - cx)) / area
        wc = np.float32(1.0) - wa - wb

        mask = (wa >= -1e-9) & (wb >= -1e-9) & (wc >= -1e-9)
        if not mask.any():
            return

        # Everything below works on the covered pixels only, so a triangle that
        # covers a small part of a large bounding box pays for the pixels it owns.
        rows, cols = np.nonzero(mask)
        iw = 1.0 / a[2], 1.0 / b[2], 1.0 / c[2]
        wam = wa[rows, cols]
        wbm = wb[rows, cols]
        wcm = wc[rows, cols]

        inv_z = wam * iw[0] + wbm * iw[1] + wcm * iw[2]
        front = inv_z > 0.0
        if not front.any():
            return
        if not front.all():
            rows, cols = rows[front], cols[front]
            wam, wbm, wcm = wam[front], wbm[front], wcm[front]
            inv_z = inv_z[front]

        z = 1.0 / inv_z
        depth_view = self.depth[y0:y1 + 1, x0:x1 + 1]
        win = z < depth_view[rows, cols]
        if not win.any():
            return

        if not win.all():
            rows, cols, z = rows[win], cols[win], z[win]
            wam, wbm, wcm = wam[win], wbm[win], wcm[win]
            inv_z = inv_z[win]

        depth_view[rows, cols] = z
        color_view = self.color[y0:y1 + 1, x0:x1 + 1]

        if texture is None or uv is None:
            if rgb is None:
                raise ValueError("draw_triangle needs rgb or texture+uv")
            color_view[rows, cols] = rgb
            return

        tex_h, tex_w = texture.shape[:2]
        tex_flat = texture.reshape(-1, 3)
        (ua, va), (ub, vb), (uc, vc) = uv

        # Perspective-correct UV on the visible pixels (handbook 10.3): the
        # per-vertex 1/z is folded once into the attribute, then divided back
        # out with the interpolated 1/z, which is the same as interpolating u/z.
        u = (wam * (ua * iw[0]) + wbm * (ub * iw[1]) + wcm * (uc * iw[2])) * z
        v = (wam * (va * iw[0]) + wbm * (vb * iw[1]) + wcm * (vc * iw[2])) * z

        txi = ((u - np.floor(u)) * tex_w).astype(np.int32)
        tyi = ((v - np.floor(v)) * tex_h).astype(np.int32)
        np.clip(txi, 0, tex_w - 1, out=txi)
        np.clip(tyi, 0, tex_h - 1, out=tyi)

        cols_rgb = tex_flat[tyi * tex_w + txi]
        if intensity != 1.0:
            cols_rgb = np.clip(cols_rgb.astype(np.float32) * float(intensity),
                               0, 255).astype(np.uint8)
        color_view[rows, cols] = cols_rgb

    def to_surface(self, pygame):
        rgb = np.ascontiguousarray(self.color, dtype=np.uint8)
        return pygame.image.frombuffer(rgb.tobytes(), (self.w, self.h), "RGB")

    def depth_visual(self, low=2.0, high=98.0):
        """Grey image of the Z-buffer: near surfaces bright, far ones dark.

        The debugging view from handbook 14.3 (F3): empty pixels stay black and
        the rest is stretched over the 2nd..98th percentile of the buffers's own
        depth range, so the walls, the floor and the props are easy to tell
        apart instead of being one flat grey.
        """
        depth = self.depth
        filled = np.isfinite(depth)
        grey = np.zeros(depth.shape, dtype=np.float32)
        if filled.any():
            values = depth[filled]
            lo, hi = np.percentile(values, [low, high])
            if hi - lo < 1e-9:
                hi = lo + 1.0
            span = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
            grey = np.where(filled, 255.0 * (1.0 - span), 0.0)
        image = grey.astype(np.uint8)
        return np.dstack((image, image, image))
