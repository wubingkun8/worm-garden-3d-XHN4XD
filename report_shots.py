"""Generate the four report screenshots (03, 04, 06, 07) in one run.

Runs without a visible window (SDL dummy driver), so nothing flashes.
Usage:  python report_shots.py
"""
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pygame

from maze import generate_maze
from geometry import scene_boxes, scene_faces, box_world_vertices, CUBE_EDGES
from camera import (camera_space, perspective_project, orthographic_project,
                    orbit_eye, normalized)
from renderer import Renderer
import solid_maze as sm
import textured_maze as tm

W, H = 900, 700
COLS, ROWS = 3, 3
SEED = 7
ANGLE = 0.55
FOV = 55.0
NEAR = 0.1
OUT_DIR = "screenshots"

BG = (16, 22, 30)
FLOOR_WIRE = (110, 160, 105)
WALL_WIRE = (160, 225, 175)


def setup():
    pygame.init()
    pygame.font.init()
    pygame.display.set_mode((W, H))


def camera_for():
    dist = 1.25 * max(COLS, ROWS)
    eye = orbit_eye(COLS / 2.0, ROWS / 2.0, dist, 0.80 * dist, ANGLE)
    target = np.array([COLS / 2.0, 0.25, ROWS / 2.0], dtype=float)
    return eye, target


def ortho_scale():
    return 0.55 * W / max(COLS, ROWS)


def hud(text):
    font = pygame.font.SysFont("consolas", 18)
    return font.render(text, True, (220, 232, 214))


def shoot_wireframe(path, ortho=False):
    maze, _, _ = generate_maze(COLS, ROWS, seed=SEED)
    boxes = scene_boxes(maze, COLS, ROWS)
    eye, target = camera_for()
    scale = ortho_scale()

    screen = pygame.display.get_surface()
    screen.fill(BG)

    items = []
    for k, (center, size) in enumerate(boxes):
        verts = box_world_vertices(center, size)
        depth = camera_space(verts.mean(axis=0), eye, target)[2]
        items.append((depth, k, verts))
    items.sort(key=lambda t: -t[0])

    for depth, k, verts in items:
        pts = []
        for v in verts:
            c = camera_space(v, eye, target)
            if ortho:
                pts.append(orthographic_project(c, W, H, scale, NEAR))
            else:
                pts.append(perspective_project(c, W, H, FOV, NEAR))

        color = FLOOR_WIRE if k == 0 else WALL_WIRE
        for i, j in CUBE_EDGES:
            if pts[i] is None or pts[j] is None:
                continue
            pygame.draw.line(screen, color,
                             (int(pts[i][0]), int(pts[i][1])),
                             (int(pts[j][0]), int(pts[j][1])), 2)

    mode = "ORTHOGRAPHIC" if ortho else "PERSPECTIVE"
    screen.blit(hud(f"{mode}  |  {COLS}x{ROWS}  |  wireframe  |  "
                    f"boxes {len(boxes)}  |  {W}x{H}"), (14, 12))
    pygame.image.save(screen, path)
    print("saved:", path)


def shoot_solid(path, ortho=False):
    maze, _, _ = generate_maze(COLS, ROWS, seed=SEED)
    boxes = scene_boxes(maze, COLS, ROWS)
    eye, target = camera_for()
    view_dir = normalized(target - eye)
    scale = ortho_scale()

    if ortho:
        project = lambda c: orthographic_project(c, W, H, scale, NEAR)
    else:
        project = lambda c: perspective_project(c, W, H, FOV, NEAR)

    renderer = Renderer(W, H, sm.SKY_RGB)
    sm.render_scene(renderer, boxes, eye, target, project, view_dir, False)

    screen = pygame.display.get_surface()
    screen.blit(renderer.to_surface(pygame), (0, 0))

    mode = "ORTHOGRAPHIC" if ortho else "PERSPECTIVE"
    screen.blit(hud(f"{mode}  |  {COLS}x{ROWS}  |  solid + Z-buffer  |  "
                    f"tris {renderer.triangles}  |  {W}x{H}"), (12, 10))
    pygame.image.save(screen, path)
    print("saved:", path, f"({renderer.triangles} triangles drawn)")


def shoot_textured(path, ortho=False, light_index=0, checker=False):
    """M7 frames: textured and lit, rendered through textured_maze.render_scene."""
    maze, _, _ = generate_maze(COLS, ROWS, seed=SEED)
    boxes = scene_boxes(maze, COLS, ROWS)
    faces = scene_faces(boxes)
    eye, target = camera_for()
    view_dir = normalized(target - eye)
    scale = ortho_scale()

    if ortho:
        project = lambda c: orthographic_project(c, W, H, scale, NEAR)
    else:
        project = lambda c: perspective_project(c, W, H, FOV, NEAR)

    renderer = Renderer(W, H, tm.SKY_RGB)
    tm.render_scene(renderer, faces, eye, target, project, view_dir,
                    tm.LIGHT_DIRS[light_index], False, True,
                    "checker" if checker else tm.FLOOR_TEXTURE)

    screen = pygame.display.get_surface()
    screen.blit(renderer.to_surface(pygame), (0, 0))

    mode = "ORTHOGRAPHIC" if ortho else "PERSPECTIVE"
    material = "checker floor" if checker else "textured"
    screen.blit(hud(f"{mode}  |  {COLS}x{ROWS}  |  {material}  |  "
                    f"light {light_index + 1}  |  tris {renderer.triangles}"
                    f"  |  {W}x{H}"), (12, 10))
    pygame.image.save(screen, path)
    print("saved:", path, f"({renderer.triangles} triangles drawn)")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    setup()

    shoot_wireframe(os.path.join(OUT_DIR, "03_maze_wireframe_3x3.png"))
    shoot_wireframe(os.path.join(OUT_DIR, "04_maze_wireframe_3x3_ortho.png"),
                    ortho=True)
    shoot_solid(os.path.join(OUT_DIR, "06_maze_solid_3x3.png"))
    shoot_solid(os.path.join(OUT_DIR, "07_maze_solid_3x3_ortho.png"),
                ortho=True)
    shoot_textured(os.path.join(OUT_DIR, "08_textured_perspective.png"))
    shoot_textured(os.path.join(OUT_DIR, "09_textured_ortho.png"), ortho=True)
    shoot_textured(os.path.join(OUT_DIR, "10_textured_light_rotated.png"),
                   light_index=1)
    shoot_textured(os.path.join(OUT_DIR, "11_textured_checker_floor.png"),
                   checker=True)

    pygame.quit()
    print("done: 8 screenshots in", OUT_DIR)


if __name__ == "__main__":
    main()
