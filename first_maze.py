"""M3/M4 debug viewer: floor + maze drawn as wireframe boxes.

Usage:
    python first_maze.py                        # interactive, 3x3
    python first_maze.py --cols 10 --rows 10    # bigger maze
    python first_maze.py --ortho                # start in orthographic
    python first_maze.py --shot out.png         # render one frame and exit
"""
import math
import sys

import numpy as np
import pygame

from maze import generate_maze
from geometry import scene_boxes, box_world_vertices, CUBE_EDGES
from camera import (camera_space, perspective_project,
                    orthographic_project, orbit_eye)

WIDTH, HEIGHT = 900, 700
COLS, ROWS = 3, 3
SEED = 7
FOV_Y = 55.0
NEAR = 0.1


def arg_value(flag, default):
    if flag in sys.argv:
        return sys.argv[sys.argv.index(flag) + 1]
    return default


def project_box(verts_world, eye, target, perspective, ortho_scale):
    """World vertices -> projected screen points (None where behind the camera)."""
    pts = []
    for v in verts_world:
        c = camera_space(v, eye, target)
        if perspective:
            p = perspective_project(c, WIDTH, HEIGHT, FOV_Y, NEAR)
        else:
            p = orthographic_project(c, WIDTH, HEIGHT, ortho_scale, NEAR)
        pts.append(p)
    return pts


def draw_box(screen, pts, color, width=2):
    for i, j in CUBE_EDGES:
        if pts[i] is None or pts[j] is None:
            continue
        pygame.draw.line(screen, color,
                         (int(pts[i][0]), int(pts[i][1])),
                         (int(pts[j][0]), int(pts[j][1])), width)


def main():
    cols = int(arg_value("--cols", COLS))
    rows = int(arg_value("--rows", ROWS))
    shot_path = arg_value("--shot", None)
    perspective = "--ortho" not in sys.argv

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Worm's Garden 3D - M3 wireframe maze")
    font = pygame.font.SysFont("consolas", 18)

    maze, start_cell, finish_cell = generate_maze(cols, rows, seed=SEED)
    boxes = scene_boxes(maze, cols, rows)

    ortho_scale = 0.55 * WIDTH / max(cols, rows)

    print(f"maze {cols}x{rows}  start={start_cell}  finish={finish_cell}")
    print(f"boxes: {len(boxes)}  (1 floor + {len(boxes) - 1} walls)")
    print(f"orthographic scale: {ortho_scale:.1f} px/unit")

    dist = 1.55 * max(cols, rows)
    eye_h = 0.80 * dist
    target = np.array([cols / 2.0, 0.25, rows / 2.0], dtype=float)

    angle = 0.55
    clock = pygame.time.Clock()
    running = True

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

        eye = orbit_eye(cols / 2.0, rows / 2.0, dist, eye_h, angle)

        screen.fill((16, 22, 30))

        # wireframe painter order: draw the farthest boxes first
        draw_list = []
        for k, (center, size) in enumerate(boxes):
            verts = box_world_vertices(center, size)
            depth = camera_space(verts.mean(axis=0), eye, target)[2]
            draw_list.append((depth, k, verts))
        draw_list.sort(key=lambda t: -t[0])

        for depth, k, verts in draw_list:
            pts = project_box(verts, eye, target, perspective, ortho_scale)
            color = (110, 160, 105) if k == 0 else (160, 225, 175)
            draw_box(screen, pts, color, 2)

        mode = "PERSPECTIVE" if perspective else "ORTHOGRAPHIC"
        info = font.render(
            f"{mode}  |  {cols}x{rows}  |  boxes {len(boxes)}"
            f"  |  P: projection  |  ESC: quit",
            True, (215, 230, 210))
        screen.blit(info, (14, 12))

        pygame.display.flip()

        if shot_path is not None:
            pygame.image.save(screen, shot_path)
            print("saved:", shot_path)
            break

    pygame.quit()
    print("window closed, program finished")


if __name__ == "__main__":
    main()
