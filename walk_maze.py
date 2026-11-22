"""M9 viewer: walk the worm, jump over puddles, avoid the birds.

Usage:
    python walk_maze.py
    python walk_maze.py --cols 10 --rows 10
    python walk_maze.py --demo            # scripted jump/hazard log + screenshots
    python walk_maze.py --shot out.png

Keys: W A S D move (world aligned, W = north = -Z), SPACE jump, ESC quit.
"""
import sys
import time

import numpy as np
import pygame

from maze import generate_maze, cell_center_3d
from geometry import scene_boxes, scene_faces, sphere_mesh, disc_mesh
from camera import camera_space, perspective_project, normalized
from renderer import Renderer, clip_near, flat_shade
from physics import (Player, movement_direction, wall_aabbs, GROUND_Y,
                     JUMP_SPEED, GRAVITY)
import hazards
from hazards import make_water_holes, make_birds, in_water, bird_hits, WATER_Y
import textures
from textured_maze import draw_face, LIGHT_DIRS

WIDTH, HEIGHT = 900, 700          # window size
RENDER_W, RENDER_H = WIDTH, HEIGHT  # internal resolution, --res can lower it
COLS, ROWS = 5, 5
SEED = 7
FOV_Y = 55.0
NEAR = 0.1

WORM_RGB = (232, 96, 106)
WORM_RINGS, WORM_SEGMENTS = 8, 12
BIRD_RGB = (24, 26, 38)
BIRD_WING_RGB = (58, 62, 82)
WATER_OUTER_RGB = (5, 35, 75)
WATER_INNER_RGB = (10, 90, 155)

CAMERA_UP = 3.0        # camera height above the floor, so jumps stay visible
CAMERA_BACK = 1.2      # camera distance south of the worm (-Z is north)


def arg_value(flag, default):
    if flag in sys.argv:
        return sys.argv[sys.argv.index(flag) + 1]
    return default


def worm_camera(player):
    eye = np.array([player.x, GROUND_Y + CAMERA_UP, player.z + CAMERA_BACK],
                   dtype=float)
    target = np.array([player.x, 0.25, player.z], dtype=float)
    return eye, target


def draw_sphere(renderer, center, radius, rgb, eye, target, project, view_dir,
                light_dir, rings=8, segments=12):
    """Flat-shaded low-poly sphere with back faces culled like the walls."""
    for a, b, c, normal in sphere_mesh(radius, center, rings, segments):
        if float(np.dot(normal, view_dir)) > 0.0:
            continue
        shaded = flat_shade(rgb, normal, light_dir)
        cam = [camera_space(p, eye, target) for p in (a, b, c)]
        for piece in clip_near(cam, NEAR):
            sa, sb, sc = (project(p) for p in piece)
            if sa is None or sb is None or sc is None:
                continue
            renderer.draw_triangle(sa, sb, sc, rgb=shaded)


def draw_disc(renderer, center, radius, rgb, eye, target, project, view_dir,
              segments=12):
    for a, b, c, normal in disc_mesh(radius, center, segments):
        if float(np.dot(normal, view_dir)) > 0.0:
            continue
        cam = [camera_space(p, eye, target) for p in (a, b, c)]
        for piece in clip_near(cam, NEAR):
            sa, sb, sc = (project(p) for p in piece)
            if sa is None or sb is None or sc is None:
                continue
            renderer.draw_triangle(sa, sb, sc, rgb=rgb)


def draw_bird(renderer, bird, eye, target, project, view_dir, light_dir):
    """Body sphere plus two wings, so a bird does not read as a plain ball."""
    draw_sphere(renderer, bird.position(), bird.radius, BIRD_RGB,
                eye, target, project, view_dir, light_dir,
                rings=6, segments=8)

    up = np.array([0.0, 1.0, 0.0])
    rgb = flat_shade(BIRD_WING_RGB, up, light_dir)
    c = bird.position()
    for side in (-1.0, 1.0):
        tri = (c + np.array([0.0, 0.02, -0.10]),
               c + np.array([0.0, 0.02, 0.10]),
               c + np.array([side * 0.36, -0.07, -0.02]))
        if float(np.dot(up, view_dir)) > 0.0:
            continue
        cam = [camera_space(p, eye, target) for p in tri]
        for piece in clip_near(cam, NEAR):
            sa, sb, sc = (project(p) for p in piece)
            if sa is None or sb is None or sc is None:
                continue
            renderer.draw_triangle(sa, sb, sc, rgb=rgb)


def render_frame(renderer, faces, player, holes, birds, floor_tex, wall_tex,
                 light_dir):
    """Draw maze, puddles, birds and the worm; returns milliseconds."""
    eye, target = worm_camera(player)
    view_dir = normalized(target - eye)
    project = lambda c: perspective_project(c, RENDER_W, RENDER_H, FOV_Y, NEAR)

    t0 = time.perf_counter()
    renderer.clear()
    for box_index, quad, uv, normal in faces:
        is_floor = box_index == 0
        draw_face(renderer, quad, uv, normal,
                  floor_tex if is_floor else wall_tex,
                  (58, 122, 66) if is_floor else (150, 152, 148),
                  eye, target, project, view_dir, light_dir, True)

    for hole in holes:
        draw_disc(renderer, (hole.x, WATER_Y, hole.z), hole.radius(),
                  WATER_OUTER_RGB, eye, target, project, view_dir)
        draw_disc(renderer, (hole.x, WATER_Y + 0.004, hole.z),
                  hole.radius() * 0.72, WATER_INNER_RGB,
                  eye, target, project, view_dir)

    for bird in birds:
        draw_bird(renderer, bird, eye, target, project, view_dir, light_dir)

    draw_sphere(renderer, player.position(), player.radius, WORM_RGB,
                eye, target, project, view_dir, light_dir,
                rings=WORM_RINGS, segments=WORM_SEGMENTS)
    return (time.perf_counter() - t0) * 1000.0


def draw_hud(screen, font, cols, rows, player, frame_ms, triangles, walked,
             message):
    state = "SLIDING" if player.blocked else "free"
    if player.airborne():
        state = "AIR"
    info = (font.render(
        f"M9  |  {cols}x{rows}  |  x {player.x:.2f}  z {player.z:.2f}"
        f"  |  feet {player.feet_y():.2f}  |  {state}"
        f"  |  tris {triangles}  |  {frame_ms:.0f} ms  |  {RENDER_W}x{RENDER_H}",
        True, (232, 240, 226)))
    screen.blit(info, (12, 10))
    hint = font.render("W A S D move (W = north)   SPACE jump   ESC quit"
                       f"   walked {walked:.1f} units", True, (150, 172, 160))
    screen.blit(hint, (12, HEIGHT - 26))
    if message:
        note = font.render(message, True, (255, 214, 120))
        screen.blit(note, (12, 34))


def present(screen, renderer, faces, player, holes, birds, floor_tex, wall_tex,
            font, cols, rows, walked, message):
    frame_ms = render_frame(renderer, faces, player, holes, birds,
                            floor_tex, wall_tex, LIGHT_DIRS[0])
    surf = renderer.to_surface(pygame)
    if (RENDER_W, RENDER_H) != (WIDTH, HEIGHT):
        surf = pygame.transform.scale(surf, (WIDTH, HEIGHT))
    screen.blit(surf, (0, 0))
    draw_hud(screen, font, cols, rows, player, frame_ms, renderer.triangles,
             walked, message)
    pygame.display.flip()
    return frame_ms


def crossable_hole(holes, maze, cols, rows):
    """A puddle with an open south neighbour, so the worm can walk into it."""
    for hole in holes:
        col, row = hole.cell
        if row + 1 < rows and not maze[row][col]["walls"]["S"]:
            return hole, (col, row + 1)
    return (holes[0], None) if holes else (None, None)


def demo(screen, renderer, faces, player, holes, birds, aabbs, maze, cols, rows,
         floor_tex, wall_tex, font):
    """Scripted run: jump apex, drown in a puddle, then jump across it."""
    dt = 1.0 / 60.0
    walked = 0.0
    start = player.position().copy()

    def advance(seconds, direction=None, jump_at=None, shot=None, tag="",
                stop_on_danger=False):
        nonlocal walked
        steps = int(round(seconds / dt))
        danger = False
        for step in range(steps):
            t = step * dt
            if jump_at is not None and abs(t - jump_at) < dt / 2:
                player.jump()
            if direction is not None:
                before = player.position()
                player.move(direction, dt, aabbs)
                walked += float(np.linalg.norm(player.position() - before))
            player.update_vertical(dt)
            if stop_on_danger and in_water(player, holes):
                danger = True
                if shot:
                    present(screen, renderer, faces, player, holes, birds,
                            floor_tex, wall_tex, font, cols, rows, walked, tag)
                    pygame.image.save(screen, f"screenshots/{shot}")
                    print(f"saved: screenshots/{shot}")
                break
            if step % 15 == 0:
                print(f"    t {t:4.2f}  z {player.z:5.3f}  feet {player.feet_y():5.3f}"
                      f"  water {in_water(player, holes)}")
            if shot and step == steps - 1:
                present(screen, renderer, faces, player, holes, birds,
                        floor_tex, wall_tex, font, cols, rows, walked, tag)
                pygame.image.save(screen, f"screenshots/{shot}")
                print(f"saved: screenshots/{shot}")
        return danger

    print("--- 1. jump in place ---")
    advance(0.32, jump_at=0.0, shot="14_worm_jump_apex.png",
            tag="jump apex: world Y, feet off the floor")
    advance(0.40)
    print(f"    landed: feet {player.feet_y():.3f}  y {player.y:.3f}")

    if holes:
        hole, approach = crossable_hole(holes, maze, cols, rows)
        print(f"--- 2. walk into the puddle at ({hole.x:.2f}, {hole.z:.2f}) ---")
        if approach is None:
            print("    no puddle has an open south side, skipping the walk test")
        else:
            south = cell_center_3d(approach[0], approach[1])
            approach_z = hole.z + 1.2      # inside the south neighbour cell
            player.x, player.z = hole.x, approach_z
            player.y, player.vertical_speed = GROUND_Y, 0.0
            drowned = advance(0.35, direction=np.array([0.0, 0.0, -1.0]),
                              shot="15_worm_at_puddle.png",
                              tag="grounded, walking into the puddle",
                              stop_on_danger=True)
            print(f"    drowned: {drowned} at z {player.z:.3f}"
                  f"  -> reset to start")

            print("--- 3. jump across the same puddle ---")
            player.x, player.z = hole.x, approach_z
            player.y, player.vertical_speed = GROUND_Y, 0.0
            drowned = advance(0.40, direction=np.array([0.0, 0.0, -1.0]),
                              jump_at=0.0, stop_on_danger=True)
            print(f"    crossed, drowned: {drowned}"
                  f"  feet {player.feet_y():.3f}  z {player.z:.3f}")
            present(screen, renderer, faces, player, holes, birds, floor_tex,
                    wall_tex, font, cols, rows, walked,
                    "jumped clear over the puddle")
            pygame.image.save(screen, "screenshots/16_worm_jumped_puddle.png")
            print("saved: screenshots/16_worm_jumped_puddle.png")

    player.x, player.z = float(start[0]), float(start[2])
    player.y, player.vertical_speed = GROUND_Y, 0.0
    print(f"end position ({player.x:.3f}, {player.z:.3f})  walked {walked:.2f}")


def main():
    cols = int(arg_value("--cols", COLS))
    rows = int(arg_value("--rows", ROWS))
    shot_path = arg_value("--shot", None)
    run_demo = "--demo" in sys.argv

    global RENDER_W, RENDER_H
    res = arg_value("--res", None)
    if res:
        RENDER_W, RENDER_H = (int(part) for part in res.lower().split("x"))

    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Worm's Garden 3D - M9 jump and hazards")
    font = pygame.font.SysFont("consolas", 18)
    clock = pygame.time.Clock()

    maze, start_cell, finish_cell = generate_maze(cols, rows, seed=SEED)
    boxes = scene_boxes(maze, cols, rows)
    faces = scene_faces(boxes)
    aabbs = wall_aabbs(boxes)
    holes = make_water_holes(cols, rows, start_cell, finish_cell, SEED)
    birds = make_birds(cols, rows, start_cell, finish_cell,
                       {h.cell for h in holes}, SEED, level=1)

    start = cell_center_3d(start_cell[0], start_cell[1])
    player = Player(start[0], start[2])
    floor_tex, wall_tex = textures.get("grass"), textures.get("stone")

    print(f"maze {cols}x{rows}  start={start_cell}  finish={finish_cell}")
    print(f"boxes: {len(boxes)}  walls in collision: {len(aabbs)}")
    print(f"puddles: {len(holes)}  birds: {len(birds)}")
    print(f"worm: start ({player.x:.2f}, {player.y:.2f}, {player.z:.2f})"
          f"  radius {player.radius}  speed {player.speed}"
          f"  jump v0 {JUMP_SPEED} g {GRAVITY}")

    renderer = Renderer(RENDER_W, RENDER_H, (14, 20, 28))

    if run_demo:
        demo(screen, renderer, faces, player, holes, birds, aabbs, maze,
             cols, rows, floor_tex, wall_tex, font)
        pygame.quit()
        print("window closed, program finished")
        return

    running = True
    walked = 0.0
    message = ""

    while running:
        dt = min(clock.tick(60) / 1000.0, 0.05)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    player.jump()

        direction = movement_direction(pygame.key.get_pressed(), pygame)
        before = player.position()
        player.move(direction, dt, aabbs)
        player.update_vertical(dt)
        walked += float(np.linalg.norm(player.position() - before))

        if in_water(player, holes):
            player.x, player.z = float(start[0]), float(start[2])
            player.y, player.vertical_speed = GROUND_Y, 0.0
            message = "splash! back to the start"
        elif any(bird_hits(player, bird) for bird in birds):
            player.x, player.z = float(start[0]), float(start[2])
            player.y, player.vertical_speed = GROUND_Y, 0.0
            message = "bird hit! back to the start"
        else:
            message = ""

        frame_ms = present(screen, renderer, faces, player, holes, birds,
                           floor_tex, wall_tex, font, cols, rows, walked,
                           message)

        if shot_path is not None:
            pygame.image.save(screen, shot_path)
            print(f"saved: {shot_path}  ({frame_ms:.0f} ms per frame)")
            break

    pygame.quit()
    print(f"worm finished at ({player.x:.2f}, {player.z:.2f})"
          f"  walked {walked:.2f} units")
    print("window closed, program finished")


if __name__ == "__main__":
    main()
