"""M8 viewer: walk the worm through the maze with horizontal collision.

Usage:
    python walk_maze.py
    python walk_maze.py --cols 10 --rows 10
    python walk_maze.py --demo            # scripted walk, prints a log
    python walk_maze.py --shot out.png

Keys: W A S D move (world aligned, W = north = -Z), ESC quit.
"""
import sys
import time

import numpy as np
import pygame

from maze import generate_maze, cell_center_3d
from geometry import scene_boxes, scene_faces, sphere_mesh
from camera import (camera_space, perspective_project, normalized)
from renderer import Renderer, clip_near, flat_shade
from physics import Player, movement_direction, wall_aabbs, GROUND_Y
import textures
from textured_maze import draw_face, LIGHT_DIRS

WIDTH, HEIGHT = 900, 700
COLS, ROWS = 3, 3
SEED = 7
FOV_Y = 55.0
NEAR = 0.1

WORM_RGB = (232, 96, 106)
WORM_RINGS, WORM_SEGMENTS = 8, 12

CAMERA_UP = 3.0        # camera height above the worm
CAMERA_BACK = 1.2      # camera distance south of the worm (-Z is north)


def arg_value(flag, default):
    if flag in sys.argv:
        return sys.argv[sys.argv.index(flag) + 1]
    return default


def worm_camera(player):
    eye = np.array([player.x, player.y + CAMERA_UP, player.z + CAMERA_BACK],
                   dtype=float)
    target = np.array([player.x, 0.25, player.z], dtype=float)
    return eye, target


def draw_worm(renderer, player, eye, target, project, view_dir, light_dir):
    """Flat-shaded low-poly sphere, back faces culled like the walls."""
    mesh = sphere_mesh(player.radius, center=(player.x, player.y, player.z),
                       rings=WORM_RINGS, segments=WORM_SEGMENTS)
    for a, b, c, normal in mesh:
        if float(np.dot(normal, view_dir)) > 0.0:
            continue
        rgb = flat_shade(WORM_RGB, normal, light_dir)
        cam = [camera_space(p, eye, target) for p in (a, b, c)]
        for piece in clip_near(cam, NEAR):
            sa, sb, sc = (project(p) for p in piece)
            if sa is None or sb is None or sc is None:
                continue
            renderer.draw_triangle(sa, sb, sc, rgb=rgb)


def render_frame(renderer, faces, player, floor_tex, wall_tex, light_dir):
    """Draw the maze and the worm into the framebuffer; returns milliseconds."""
    eye, target = worm_camera(player)
    view_dir = normalized(target - eye)
    project = lambda c: perspective_project(c, WIDTH, HEIGHT, FOV_Y, NEAR)

    t0 = time.perf_counter()
    renderer.clear()
    for box_index, quad, uv, normal in faces:
        is_floor = box_index == 0
        draw_face(renderer, quad, uv, normal,
                  floor_tex if is_floor else wall_tex,
                  (58, 122, 66) if is_floor else (150, 152, 148),
                  eye, target, project, view_dir, light_dir, True)
    draw_worm(renderer, player, eye, target, project, view_dir, light_dir)
    return (time.perf_counter() - t0) * 1000.0


def draw_hud(screen, font, cols, rows, player, frame_ms, triangles, walked):
    state = "SLIDING" if player.blocked else "free"
    info = (font.render(
        f"M8  |  {cols}x{rows}  |  x {player.x:.2f}  z {player.z:.2f}"
        f"  |  {state}  |  tris {triangles}"
        f"  |  {frame_ms:.0f} ms  |  {WIDTH}x{HEIGHT}", True, (232, 240, 226)))
    screen.blit(info, (12, 10))
    hint = font.render(f"W A S D move (W = north)   ESC quit"
                       f"   walked {walked:.1f} units", True, (150, 172, 160))
    screen.blit(hint, (12, HEIGHT - 26))


def demo(renderer, faces, player, aabbs, cols, rows,
         floor_tex, wall_tex, font, screen):
    """Scripted walk: north into a wall, then diagonal to slide along it.

    Prints a position log so the collision rules can be checked from the
    terminal, and saves two screenshots.
    """
    script = [
        ("north into the wall", np.array([0.0, 0.0, -1.0]), 1.0),
        ("diagonal north-east", np.array([0.7071, 0.0, -0.7071]), 1.6),
        ("east along the corridor", np.array([1.0, 0.0, 0.0]), 1.0),
    ]
    dt = 1.0 / 60.0
    shots = {0: "12_worm_start.png", 1: "13_worm_against_wall.png"}
    walked = 0.0
    t = 0.0

    print(f"{'t':>5}  {'x':>6}  {'z':>6}  state    phase")
    for phase_index, (label, direction, duration) in enumerate(script):
        steps = int(round(duration / dt))
        for step in range(steps):
            before = player.position()
            player.move(direction, dt, aabbs)
            walked += float(np.linalg.norm(player.position() - before))
            t += dt
            if step % 15 == 0:
                state = "SLIDING" if player.blocked else "free"
                print(f"{t:5.2f}  {player.x:6.3f}  {player.z:6.3f}  "
                      f"{state:7}  {label}")

        frame_ms = render_frame(renderer, faces, player, floor_tex, wall_tex,
                                LIGHT_DIRS[0])
        screen.blit(renderer.to_surface(pygame), (0, 0))
        draw_hud(screen, font, cols, rows, player, frame_ms,
                 renderer.triangles, walked)
        pygame.display.flip()
        if phase_index in shots:
            pygame.image.save(screen, f"screenshots/{shots[phase_index]}")
            print(f"saved: screenshots/{shots[phase_index]}")

    print(f"end position ({player.x:.3f}, {player.z:.3f})"
          f"  walked {walked:.2f} units")


def main():
    cols = int(arg_value("--cols", COLS))
    rows = int(arg_value("--rows", ROWS))
    shot_path = arg_value("--shot", None)
    run_demo = "--demo" in sys.argv

    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Worm's Garden 3D - M8 worm movement")
    font = pygame.font.SysFont("consolas", 18)
    clock = pygame.time.Clock()

    maze, start_cell, finish_cell = generate_maze(cols, rows, seed=SEED)
    boxes = scene_boxes(maze, cols, rows)
    faces = scene_faces(boxes)
    aabbs = wall_aabbs(boxes)

    start = cell_center_3d(start_cell[0], start_cell[1])
    player = Player(start[0], start[2])
    floor_tex, wall_tex = textures.get("grass"), textures.get("stone")

    print(f"maze {cols}x{rows}  start={start_cell}  finish={finish_cell}")
    print(f"boxes: {len(boxes)}  walls in collision: {len(aabbs)}")
    print(f"worm: start ({player.x:.2f}, {player.y:.2f}, {player.z:.2f})"
          f"  radius {player.radius}  speed {player.speed}")

    renderer = Renderer(WIDTH, HEIGHT, (14, 20, 28))

    if run_demo:
        demo(renderer, faces, player, aabbs, cols, rows,
             floor_tex, wall_tex, font, screen)
        pygame.quit()
        print("window closed, program finished")
        return

    running = True
    frame_ms, walked = 0.0, 0.0

    while running:
        dt = min(clock.tick(60) / 1000.0, 0.05)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        direction = movement_direction(pygame.key.get_pressed(), pygame)
        before = player.position()
        player.move(direction, dt, aabbs)
        walked += float(np.linalg.norm(player.position() - before))

        frame_ms = render_frame(renderer, faces, player, floor_tex, wall_tex,
                                LIGHT_DIRS[0])
        screen.blit(renderer.to_surface(pygame), (0, 0))
        draw_hud(screen, font, cols, rows, player, frame_ms,
                 renderer.triangles, walked)
        pygame.display.flip()

        if shot_path is not None:
            pygame.image.save(screen, shot_path)
            print(f"saved: {shot_path}")
            break

    pygame.quit()
    print(f"worm finished at ({player.x:.2f}, {player.z:.2f})"
          f"  walked {walked:.2f} units")
    print("window closed, program finished")


if __name__ == "__main__":
    main()
