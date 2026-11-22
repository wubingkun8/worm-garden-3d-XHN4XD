"""Tests for M7: texture sampling, materials and the textured rasterizer."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from renderer import Renderer
import textures


def test_nearest_sample_wraps():
    """u,v outside [0,1] must repeat, and u=0.5 must land in the right column."""
    tex = textures.make_stone()
    tex_h, tex_w = tex.shape[:2]
    assert textures.nearest_sample(tex, 0.0, 0.0)[0] == tex[0, 0][0]
    mid = textures.nearest_sample(tex, 0.5, 0.5)
    expected = tex[tex_h // 2, tex_w // 2]
    assert (mid == expected).all()
    # one full tile to the right (and one to the left) must look identical
    assert (textures.nearest_sample(tex, 1.5, 0.5) == mid).all()
    assert (textures.nearest_sample(tex, -0.5, 0.5) == mid).all()


def test_materials_differ():
    """Grass and stone must be different sizes of grey/green, i.e. two materials."""
    grass = textures.get("grass")
    stone = textures.get("stone")
    assert grass.shape == stone.shape == (128, 128, 3)
    grass_mean = grass.mean(axis=(0, 1))
    stone_mean = stone.mean(axis=(0, 1))
    assert grass_mean[1] > grass_mean[0] + 60      # green dominates
    assert abs(stone_mean[0] - stone_mean[1]) < 12  # stone is neutral grey


def test_textured_quad_has_no_cracks():
    """Two textured triangles of one rectangle must fill the whole rectangle."""
    w, h = 300, 300
    r = Renderer(w, h, (0, 0, 0))
    r.clear()
    tex = np.zeros((8, 8, 3), dtype=np.uint8)
    tex[:, :] = (0, 200, 0)

    a, b, c, d = (30.0, 30.0, 5.0), (270.0, 30.0, 5.0), (270.0, 270.0, 5.0), (30.0, 270.0, 5.0)
    uv = [(0, 0), (3, 0), (3, 3), (0, 3)]
    u0, u1, u2, u3 = uv
    r.draw_triangle(a, b, c, uv=(u0, u1, u2), texture=tex, intensity=1.0)
    r.draw_triangle(a, c, d, uv=(u0, u2, u3), texture=tex, intensity=1.0)

    inside = np.zeros((h, w), dtype=bool)
    inside[30:270, 30:270] = True
    holes = (r.color.sum(axis=2) == 0) & inside
    assert int(holes.sum()) == 0


def test_perspective_uv_is_not_linear():
    """Under perspective, equal screen steps must not give equal texture steps.

    A quad strongly tilted away from the camera: with plain screen-space
    interpolation the two halves would get equal u steps; perspective-correct
    interpolation must sample different amounts of texture across the quad.
    """
    w, h = 200, 200
    r = Renderer(w, h, (0, 0, 0))
    r.clear()

    # near edge at z=1, far edge at z=4 -> strong foreshortening
    a = (40.0, 40.0, 1.0)
    b = (160.0, 40.0, 1.0)
    c = (160.0, 160.0, 4.0)
    d = (40.0, 160.0, 4.0)

    # texture with u encoded in red: left half dark, right half bright stripes
    tex = np.zeros((2, 64, 3), dtype=np.uint8)
    tex[:, :32] = (10, 10, 10)
    tex[:, 32:] = (240, 240, 240)

    uv = [(0, 0), (1, 0), (1, 1), (0, 1)]
    r.draw_triangle(a, b, c, uv=(uv[0], uv[1], uv[2]), texture=tex, intensity=1.0)
    r.draw_triangle(a, c, d, uv=(uv[0], uv[2], uv[3]), texture=tex, intensity=1.0)

    # the dark/bright boundary must not sit at exactly 50 % of the width
    row = r.color[100, 40:160]
    bright = (row[:, 0] > 128)
    boundary = np.argmax(bright) if bright.any() else 0
    assert not (60 <= boundary <= 62), f"uv boundary at {boundary}/120 looks linear"


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"{test.__name__}: ok")
    print("texture tests passed")
