"""M5 viewer: solid shaded maze drawn with a NumPy Z-buffer.

Usage:
    python solid_maze.py
    python solid_maze.py --cols 10 --rows 10
    python solid_maze.py --ortho
    python solid_maze.py --reverse                # draw far-to-near instead
    python solid_maze.py --shot out.png
"""
import math
import sys
import time

import numpy as np
import pygame

from maze import generate_maze
from geometry import scene_boxes, box_world_vertices, CUBE_TRIANGLES
from camera import (camera_space, perspective_project,
                    orthographic_project, orbit_eye, normalized)
from renderer import Renderer, clip_near, face_normal, flat_shade

WIDTH, HEIGHT = 900, 700
COLS, ROWS = 3, 3
SEED = 7
FOV_Y = 55.0
NEAR = 0.1

FLOOR_RGB = (58, 122, 66)
WALL_RGB = (150, 152, 148)
SKY_RGB = (14, 20, 28)
LIGHT_DIR = normalized(np.array([-0.45, 0.85, 0.30]))


def arg_value(flag, default):
    if flag in sys.argv:
        return sys.argv[sys.argv.index(flag) + 1]
    return default


def make_projector(perspective, width, height, ortho_scale):
    if perspective:
        return lambda c: perspective_project(c, width, height, FOV_Y, NEAR)
    return lambda c: orthographic_project(c, width, height, ortho_scale, NEAR)


def draw_box(renderer, center, size, base_rgb, eye, target, project, view_dir):
    """Frustum-ish cull, flat lighting, near clip, then rasterize."""
    verts = box_world_vertices(center, size)
    cam = [camera_space(v, eye, target) for v in verts]
    centroid = verts.mean(axis=0)

    for tri in CUBE_TRIANGLES:
        i, j, k = tri
        n = face_normal(verts[i], verts[j], verts[k])
        if n is None:
            continue
        if float(np.dot(n, view_dir)) > 0.0:      # faces the wrong way
            continue

        rgb = flat_shade(base_rgb, n, LIGHT_DIR)
        tri_cam = [cam[i], cam[j], cam[k]]

        for piece in clip_near(tri_cam, NEAR):
            pts = [project(p) for p in piece]
            if any(p is None for p in pts):
                continue
            renderer.draw_triangle(pts[0], pts[1], pts[2], rgb)


def render_scene(renderer, boxes, eye, target, project, view_dir, reverse=False):
    """Draw every box into the framebuffer. Order must not affect the result."""
    renderer.clear()

    depths = []
    for idx, (center, size) in enumerate(boxes):
        centroid = box_world_vertices(center, size).mean(axis=0)
        depths.append((camera_space(centroid, eye, target)[2], idx))
    depths.sort(key=lambda t: t[0], reverse=reverse)

    for depth, idx in depths:
        center, size = boxes[idx]
        base = FLOOR_RGB if idx == 0 else WALL_RGB
        draw_box(renderer, center, size, base,
                 eye, target, project, view_dir)
    return renderer


def main():
    cols = int(arg_value("--cols", COLS))
    rows = int(arg_value("--rows", ROWS))
    shot_path = arg_value("--shot", None)
    perspective = "--ortho" not in sys.argv
    reverse = "--reverse" in sys.argv

    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Worm's Garden 3D - M5 solid maze")
    font = pygame.font.SysFont("consolas", 18)

    maze, start_cell, finish_cell = generate_maze(cols, rows, seed=SEED)
    boxes = scene_boxes(maze, cols, rows)
    ortho_scale = 0.55 * WIDTH / max(cols, rows)

    print(f"maze {cols}x{rows}  start={start_cell}  finish={finish_cell}")
    print(f"boxes: {len(boxes)}  triangles: {len(boxes) * 12}")

    renderer = Renderer(WIDTH, HEIGHT, SKY_RGB)

    dist = 1.25 * max(cols, rows)
    eye_h = 0.80 * max(1.0, dist)
    target = np.array([cols / 2.0, 0.25, rows / 2.0], dtype=float)

    angle = 0.55
    clock = pygame.time.Clock()
    running = True
    frame_ms = 0.0

    while running:
        dt = min(clock.tick(60) / 1000.0, 0.05)
        if shot_path is None:
            angle += 0.35 * dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_p:
                    perspective = not perspective
                elif event.key == pygame.K_r:
                    reverse = not reverse

        eye = orbit_eye(cols / 2.0, rows / 2.0, dist, eye_h, angle)
        view_dir = normalized(target - eye)
        project = make_projector(perspective, WIDTH, HEIGHT, ortho_scale)

        t0 = time.perf_counter()

        render_scene(renderer, boxes, eye, target, project, view_dir, reverse)

        frame = renderer.to_surface(pygame)
        screen.blit(frame, (0, 0))

        frame_ms = (time.perf_counter() - t0) * 1000.0

        mode = "PERSPECTIVE" if perspective else "ORTHOGRAPHIC"
        order = "far->near" if reverse else "near->far"
        info = font.render(
            f"{mode}  |  {cols}x{rows}  |  boxes {len(boxes)}"
            f"  |  tris {renderer.triangles}  |  {frame_ms:.1f} ms"
            f"  |  draw {order}  |  P/R/ESC",
            True, (225, 235, 220))
        screen.blit(info, (12, 10))

        pygame.display.flip()

        if shot_path is not None:
            pygame.image.save(screen, shot_path)
            print(f"saved: {shot_path}  ({frame_ms:.1f} ms, "
                  f"{renderer.triangles} triangles)")
            break

    pygame.quit()
    print("window closed, program finished")


if __name__ == "__main__":
    main()
