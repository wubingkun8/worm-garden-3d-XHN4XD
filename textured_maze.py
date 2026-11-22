"""M7 viewer: textured, lit maze (Pillow textures + perspective-correct UV).

Usage:
    python textured_maze.py
    python textured_maze.py --cols 10 --rows 10
    python textured_maze.py --ortho
    python textured_maze.py --flat             # no texture, flat colours
    python textured_maze.py --light 1          # second light direction
    python textured_maze.py --checker          # checker floor
    python textured_maze.py --shot out.png

Keys: P projection, R draw order, T texture on/off, L light direction, ESC quit.
"""
import sys
import time

import numpy as np
import pygame

from maze import generate_maze
from geometry import scene_boxes, scene_faces
from camera import (camera_space, perspective_project,
                    orthographic_project, orbit_eye, normalized)
from renderer import Renderer, clip_near_attrs
import textures

WIDTH, HEIGHT = 900, 700
COLS, ROWS = 3, 3
SEED = 7
FOV_Y = 55.0
NEAR = 0.1

SKY_RGB = (14, 20, 28)
FLOOR_RGB = (58, 122, 66)
WALL_RGB = (150, 152, 148)
AMBIENT, DIFFUSE = 0.25, 0.75

FLOOR_TEXTURE = "grass"
WALL_TEXTURE = "stone"

LIGHT_DIRS = [
    normalized(np.array([-0.45, 0.85, 0.30])),   # high, from the west
    normalized(np.array([0.60, 0.28, -0.45])),   # low, from the east
]

TRI_QUADS = ((0, 1, 2), (0, 2, 3))


def arg_value(flag, default):
    if flag in sys.argv:
        return sys.argv[sys.argv.index(flag) + 1]
    return default


def make_projector(perspective, width, height, ortho_scale):
    if perspective:
        return lambda c: perspective_project(c, width, height, FOV_Y, NEAR)
    return lambda c: orthographic_project(c, width, height, ortho_scale, NEAR)


def draw_face(renderer, quad, uv, normal, texture, base_rgb,
              eye, target, project, view_dir, light_dir,
              use_texture=True, bilinear=False):
    """Cull the back face, shade it, clip it, then rasterize two triangles."""
    if float(np.dot(normal, view_dir)) > 0.0:
        return

    intensity = AMBIENT + DIFFUSE * max(0.0, float(np.dot(normal, light_dir)))
    cam = [camera_space(p, eye, target) for p in quad]

    for tri in TRI_QUADS:
        tri_cam = [cam[i] for i in tri]
        tri_uv = [np.asarray(uv[i], dtype=float) for i in tri]

        for piece in clip_near_attrs(tri_cam, tri_uv, NEAR):
            pa, pb, pc, uva, uvb, uvc = piece
            sa, sb, sc = project(pa), project(pb), project(pc)
            if sa is None or sb is None or sc is None:
                continue
            if use_texture:
                renderer.draw_triangle(sa, sb, sc, uv=(uva, uvb, uvc),
                                       texture=texture, intensity=intensity)
            else:
                shaded = np.clip(np.asarray(base_rgb, dtype=float) * intensity,
                                 0, 255).astype(np.uint8)
                renderer.draw_triangle(sa, sb, sc, rgb=shaded)


def render_scene(renderer, faces, eye, target, project, view_dir, light_dir,
                 reverse=False, use_texture=True, floor_texture=None):
    """Draw every face into the framebuffer. Order must not change the result."""
    renderer.clear()

    floor_tex = textures.get(floor_texture or FLOOR_TEXTURE)
    wall_tex = textures.get(WALL_TEXTURE)

    depths = []
    for idx, (box_index, quad, uv, normal) in enumerate(faces):
        depth = camera_space(quad.mean(axis=0), eye, target)[2]
        depths.append((depth, idx))
    depths.sort(key=lambda t: t[0], reverse=reverse)

    for depth, idx in depths:
        box_index, quad, uv, normal = faces[idx]
        is_floor = box_index == 0
        draw_face(renderer, quad, uv, normal,
                  floor_tex if is_floor else wall_tex,
                  FLOOR_RGB if is_floor else WALL_RGB,
                  eye, target, project, view_dir, light_dir,
                  use_texture=use_texture)
    return renderer


def main():
    cols = int(arg_value("--cols", COLS))
    rows = int(arg_value("--rows", ROWS))
    shot_path = arg_value("--shot", None)
    perspective = "--ortho" not in sys.argv
    reverse = "--reverse" in sys.argv
    use_texture = "--flat" not in sys.argv
    light_index = int(arg_value("--light", 0))
    floor_texture = "checker" if "--checker" in sys.argv else FLOOR_TEXTURE

    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Worm's Garden 3D - M7 textured maze")
    font = pygame.font.SysFont("consolas", 18)

    maze, start_cell, finish_cell = generate_maze(cols, rows, seed=SEED)
    boxes = scene_boxes(maze, cols, rows)
    faces = scene_faces(boxes)
    ortho_scale = 0.55 * WIDTH / max(cols, rows)

    print(f"maze {cols}x{rows}  start={start_cell}  finish={finish_cell}")
    print(f"boxes: {len(boxes)}  faces: {len(faces)}  triangles: {len(faces) * 2}")
    print(f"lights: {len(LIGHT_DIRS)}  using light {light_index}")

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
            angle += 0.25 * dt

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
                elif event.key == pygame.K_t:
                    use_texture = not use_texture
                elif event.key == pygame.K_l:
                    light_index = (light_index + 1) % len(LIGHT_DIRS)

        eye = orbit_eye(cols / 2.0, rows / 2.0, dist, eye_h, angle)
        view_dir = normalized(target - eye)
        project = make_projector(perspective, WIDTH, HEIGHT, ortho_scale)

        t0 = time.perf_counter()

        render_scene(renderer, faces, eye, target, project, view_dir,
                     LIGHT_DIRS[light_index], reverse, use_texture, floor_texture)

        screen.blit(renderer.to_surface(pygame), (0, 0))
        frame_ms = (time.perf_counter() - t0) * 1000.0

        mode = "PERSPECTIVE" if perspective else "ORTHOGRAPHIC"
        order = "far->near" if reverse else "near->far"
        material = "textured" if use_texture else "flat colour"
        info = font.render(
            f"{mode}  |  {cols}x{rows}  |  {material}  |  light {light_index + 1}"
            f"  |  tris {renderer.triangles}  |  {frame_ms:.1f} ms"
            f"  |  {WIDTH}x{HEIGHT}", True, (232, 240, 226))
        screen.blit(info, (12, 10))
        hint = font.render("P projection  R order  T texture  L light  ESC quit",
                           True, (150, 172, 160))
        screen.blit(hint, (12, HEIGHT - 26))

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
