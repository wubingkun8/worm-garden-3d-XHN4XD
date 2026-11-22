import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from maze import generate_maze
from geometry import scene_boxes, box_world_vertices
from camera import camera_space, perspective_project, normalized, orbit_eye
from renderer import Renderer, clip_near
import solid_maze as sm


def build_scene(cols=3, rows=3):
    maze, start, finish = generate_maze(cols, rows, seed=7)
    boxes = scene_boxes(maze, cols, rows)
    dist = 1.55 * max(cols, rows)
    eye = orbit_eye(cols / 2.0, rows / 2.0, dist, 0.80 * dist, 0.55)
    target = np.array([cols / 2.0, 0.25, rows / 2.0], dtype=float)
    project = lambda c: perspective_project(c, sm.WIDTH, sm.HEIGHT, 55.0, 0.1)
    return boxes, eye, target, project


def test_draw_order_does_not_matter():
    boxes, eye, target, project = build_scene()
    view_dir = normalized(target - eye)

    near_first = Renderer(sm.WIDTH, sm.HEIGHT, sm.SKY_RGB)
    sm.render_scene(near_first, boxes, eye, target, project, view_dir, False)

    far_first = Renderer(sm.WIDTH, sm.HEIGHT, sm.SKY_RGB)
    sm.render_scene(far_first, boxes, eye, target, project, view_dir, True)

    assert np.array_equal(near_first.color, far_first.color)


def test_near_plane_split():
    tri = [[-0.4, 0.0, 0.05], [0.4, 0.0, 0.5], [0, 0.6, 0.5]]
    assert len(clip_near(tri, near=0.1)) == 2

    behind = [[0.0, 0.0, 0.02], [0.5, 0.0, -0.4], [0.0, 0.5, 0.03]]
    assert len(clip_near(behind, near=0.1)) == 0


def test_occlusion_red_behind_blue():
    """A red cube behind a blue cube must never show through."""
    near = Renderer(200, 200, (0, 0, 0))
    eye = np.array([0.0, 0.0, 6.0])
    target = np.array([0.0, 0.0, 0.0])
    view_dir = normalized(target - eye)
    project = lambda c: perspective_project(c, 200, 200, 55.0, 0.1)

    boxes = [((0.0, 0.0, 2.0), (1.6, 1.6, 1.6)),
             ((0.0, 0.0, 0.0), (1.6, 1.6, 1.6))]

    for reverse in (False, True):
        r = Renderer(200, 200, (0, 0, 0))
        r.clear()
        for idx in ([1, 0] if reverse else [0, 1]):
            center, size = boxes[idx]
            base = (40, 60, 200) if idx == 0 else (200, 40, 40)
            sm.draw_box(r, center, size, base, eye, target, project, view_dir)

        center_pixel = r.color[100, 100]
        blue = float(center_pixel[2]) > float(center_pixel[0])
        assert blue, f"back cube leaked through: {center_pixel}"


if __name__ == "__main__":
    test_draw_order_does_not_matter()
    test_near_plane_split()
    test_occlusion_red_behind_blue()
    print("renderer tests passed")
