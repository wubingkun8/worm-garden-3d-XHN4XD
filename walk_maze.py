"""M10 viewer: walk, jump, collect the four apples, reach the exit, next level.

Usage:
    python walk_maze.py
    python walk_maze.py --cols 10 --rows 10
    python walk_maze.py --demo            # scripted jump/hazard log + screenshots
    python walk_maze.py --shot out.png

Keys: W A S D move (world aligned, W = north = -Z), SPACE jump, F textures,
R restart, ESC quit.
"""
import math
import sys
import time

import numpy as np
import pygame

from maze import generate_maze, cell_center_3d, DIRECTIONS
from geometry import scene_boxes, scene_faces, sphere_mesh, disc_mesh
from camera import (camera_space, camera_basis, perspective_project,
                    normalized, ground_intersection)
from renderer import (Renderer, clip_near, clip_near_attrs, flat_shade)
from physics import (Player, movement_direction, wall_aabbs, GROUND_Y,
                     JUMP_SPEED, GRAVITY)
import particles
from particles import MAX_PARTICLES, MAX_BURST, create_spark, create_firework
import hazards
from hazards import (make_water_holes, make_birds, in_water, bird_hits,
                     nearest_water, open_neighbours, update_birds, WATER_Y)
import apples as apples_module
from apples import (make_apples, collect as collect_apples, remaining,
                    APPLE_RADIUS, APPLE_TOTAL, STEM_RGB, LEAF_RGB)
from progress import Progress, format_clock
import textures
from textured_maze import (draw_face, LIGHT_DIRS, AMBIENT, DIFFUSE, TRI_QUADS,
                           FLOOR_RGB, WALL_RGB)

WIDTH, HEIGHT = 900, 700          # window size
RENDER_W, RENDER_H = 480, 373     # internal resolution while playing
SHOT_W, SHOT_H = 900, 700         # full resolution for screenshots
COLS, ROWS = 12, 12               # playable default; the demo shots stay 5 x 5
DEMO_COLS, DEMO_ROWS = 5, 5
SEED = 7
FOV_Y = 55.0
NEAR = 0.1
# Clip a hair in front of the projection plane: a vertex produced exactly at
# NEAR would be rejected by perspective_project() (it tests depth <= near),
# which used to drop every triangle that straddled the near plane.
CLIP_NEAR = NEAR * 1.02

WORM_RGB = (232, 96, 106)
WORM_EYE_RGB = (246, 244, 238)
WORM_PUPIL_RGB = (26, 26, 36)
WORM_RINGS, WORM_SEGMENTS = 6, 9     # 90 triangles, but tiny on screen
BIRD_RGB = (24, 26, 38)
BIRD_WING_RGB = (58, 62, 82)
BIRD_RINGS, BIRD_SEGMENTS = 4, 6
BIRD_WING_HZ = 3.0                   # wing flap: 0.05 sin(2 pi 3 t)
DISC_SEGMENTS = 10
WATER_OUTER_RGB = (5, 35, 75)
WATER_INNER_RGB = (10, 90, 155)
APPLE_RINGS, APPLE_SEGMENTS = 3, 5   # 36 triangles per apple, small on screen
FLAG_LOCKED_RGB = (196, 46, 52)
FLAG_OPEN_RGB = (86, 196, 110)
FLAG_POLE_RGB = (206, 206, 198)
FLAG_HEIGHT = 0.78
FLAG_HIT_RADIUS = 0.34               # 17 px in check_finish(), divided by CELL
LAWN_RGB = (44, 92, 52)              # the grass ring outside the border walls
APPLE_SHADOW_RGB = (25, 60, 30)      # (25, 60, 30) in the 2D draw_medals()

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


def to_view(points, matrix, eye):
    """World points to camera space with one matmul, instead of one call per vertex."""
    return (np.asarray(points, dtype=float) - eye) @ matrix.T


def draw_sphere(renderer, center, radius, rgb, matrix, eye, project, view_dir,
                light_dir, rings=8, segments=12):
    """Flat-shaded low-poly sphere with back faces culled like the walls."""
    for a, b, c, normal in sphere_mesh(radius, center, rings, segments):
        if float(np.dot(normal, view_dir)) > 0.0:
            continue
        shaded = flat_shade(rgb, normal, light_dir)
        cam = to_view((a, b, c), matrix, eye)
        for piece in clip_near(cam, CLIP_NEAR):
            sa, sb, sc = (project(p) for p in piece)
            if sa is None or sb is None or sc is None:
                continue
            renderer.draw_triangle(sa, sb, sc, rgb=shaded)


def draw_disc(renderer, center, radius, rgb, matrix, eye, project, view_dir,
              segments=None):
    segments = DISC_SEGMENTS if segments is None else segments
    for a, b, c, normal in disc_mesh(radius, center, segments):
        if float(np.dot(normal, view_dir)) > 0.0:
            continue
        cam = to_view((a, b, c), matrix, eye)
        for piece in clip_near(cam, CLIP_NEAR):
            sa, sb, sc = (project(p) for p in piece)
            if sa is None or sb is None or sc is None:
                continue
            renderer.draw_triangle(sa, sb, sc, rgb=rgb)


def draw_bird(renderer, bird, matrix, eye, project, view_dir, light_dir,
              now=0.0):
    """Body sphere plus two wings that flap with a sine wave (handbook 12.2)."""
    draw_sphere(renderer, bird.position(), bird.radius, BIRD_RGB,
                matrix, eye, project, view_dir, light_dir,
                rings=BIRD_RINGS, segments=BIRD_SEGMENTS)

    up = np.array([0.0, 1.0, 0.0])
    rgb = flat_shade(BIRD_WING_RGB, up, light_dir)
    c = bird.position()
    flap = 0.05 * math.sin(2.0 * math.pi * BIRD_WING_HZ * now + bird.phase)
    if float(np.dot(up, view_dir)) > 0.0:
        return
    for side in (-1.0, 1.0):
        tri = (c + np.array([0.0, 0.02, -0.10]),
               c + np.array([0.0, 0.02, 0.10]),
               c + np.array([side * 0.36, -0.07 + flap, -0.02]))
        cam = to_view(tri, matrix, eye)
        for piece in clip_near(cam, CLIP_NEAR):
            sa, sb, sc = (project(p) for p in piece)
            if sa is None or sb is None or sc is None:
                continue
            renderer.draw_triangle(sa, sb, sc, rgb=rgb)


def draw_apple(renderer, apple, matrix, eye, project, view_dir, light_dir,
               now=0.0):
    """A bobbing apple: two-lobed body like the 2D drawing, plus stem and leaf."""
    centre = apple.center(now)
    colour = apple.color()
    # the same soft shadow the 2D drawing puts under every apple, so the apple
    # does not look like it floats
    draw_disc(renderer, (apple.x, 0.004, apple.z), APPLE_RADIUS * 0.9,
              APPLE_SHADOW_RGB, matrix, eye, project, view_dir)
    draw_sphere(renderer, centre, APPLE_RADIUS, colour,
                matrix, eye, project, view_dir, light_dir,
                rings=APPLE_RINGS, segments=APPLE_SEGMENTS)
    draw_sphere(renderer, centre + np.array([0.0, 0.0, 0.10]), APPLE_RADIUS * 0.82,
                colour, matrix, eye, project, view_dir, light_dir,
                rings=APPLE_RINGS, segments=APPLE_SEGMENTS)

    top = centre + np.array([0.0, APPLE_RADIUS * 1.5, 0.0])
    stem = (top, top + np.array([0.0, 0.18, 0.0]),
            top + np.array([0.03, 0.17, 0.0]))
    leaf = (top + np.array([0.0, 0.13, 0.0]),
            top + np.array([0.14, 0.20, 0.0]),
            top + np.array([0.0, 0.20, 0.0]))
    for tri, rgb in ((stem, STEM_RGB), (leaf, LEAF_RGB)):
        cam = to_view(tri, matrix, eye)
        for piece in clip_near(cam, CLIP_NEAR):
            sa, sb, sc = (project(p) for p in piece)
            if sa is None or sb is None or sc is None:
                continue
            shaded = flat_shade(rgb, normalized(np.cross(
                np.array(piece[1]) - np.array(piece[0]),
                np.array(piece[2]) - np.array(piece[0]))), light_dir)
            renderer.draw_triangle(sa, sb, sc, rgb=shaded)


def draw_flag(renderer, cell, unlocked, matrix, eye, project, view_dir,
              light_dir, wave=1.0):
    """The exit: a pole and a flag, red while locked, green when all apples are in."""
    base = cell_center_3d(cell[0], cell[1])
    pole_bottom = base + np.array([0.0, 0.02, 0.0])
    pole_top = base + np.array([0.0, FLAG_HEIGHT, 0.0])
    half = 0.03
    # Two crossed quads instead of one plane: a plane facing east disappears
    # completely when the camera looks north, which is the usual view here.
    for axis in (0, 2):
        offset = np.zeros(3)
        offset[axis] = half
        p0, p1 = pole_bottom - offset, pole_bottom + offset
        p2, p3 = pole_top + offset, pole_top - offset
        for tri in ((p0, p1, p2), (p0, p2, p3)):
            _fill_triangle(renderer, tri, FLAG_POLE_RGB, matrix, eye, project,
                           view_dir, light_dir)

    flag_rgb = FLAG_OPEN_RGB if unlocked else FLAG_LOCKED_RGB
    # wave lets the cloth hang into the open side of the cell: an exit on the
    # east border would otherwise push the cloth into the boundary wall
    tip = pole_top + np.array([0.34 * wave, -0.05, 0.0])
    tail_top = pole_top + np.array([0.0, -0.20, 0.0])
    tail_tip = pole_top + np.array([0.34 * wave, -0.28, 0.0])
    cloth = ((pole_top, tail_top, tip), (tail_top, tail_tip, tip))
    for tri in cloth:
        # both windings, so the cloth is visible from the north as well
        for wound in (tri, (tri[0], tri[2], tri[1])):
            _fill_triangle(renderer, wound, flag_rgb, matrix, eye, project,
                           view_dir, light_dir)


def background_image(width, height, top=(12, 18, 26), bottom=(30, 64, 38)):
    """Clear colour as a vertical ramp: sky at the top, lawn at the bottom.

    A real lawn quad around the maze cost 11 ms per frame (its bounding box is
    the whole screen even when the floor hides it), and a filled ramp is one
    numpy assignment.
    """
    ramp = (np.linspace(0.0, 1.0, height) ** 1.2)[:, None, None]
    top_rgb = np.array(top, dtype=float)[None, None, :]
    bottom_rgb = np.array(bottom, dtype=float)[None, None, :]
    column = (top_rgb + (bottom_rgb - top_rgb) * ramp).astype(np.uint8)
    return np.repeat(column, width, axis=1)


def _fill_triangle(renderer, tri, rgb, matrix, eye, project, view_dir,
                   light_dir):
    normal = normalized(np.cross(np.asarray(tri[1]) - np.asarray(tri[0]),
                                 np.asarray(tri[2]) - np.asarray(tri[0])))
    if float(np.dot(normal, view_dir)) > 0.0:
        return
    shaded = flat_shade(rgb, normal, light_dir)
    cam = to_view(tri, matrix, eye)
    for piece in clip_near(cam, CLIP_NEAR):
        sa, sb, sc = (project(p) for p in piece)
        if sa is None or sb is None or sc is None:
            continue
        renderer.draw_triangle(sa, sb, sc, rgb=shaded)


def prepare_faces(faces):
    """Pre-flatten the static maze into numpy arrays (one off, not per frame)."""
    corners = np.array([[np.asarray(p, dtype=float) for p in quad]
                        for _, quad, _, _ in faces], dtype=float)
    uvs = np.array([np.asarray(uv, dtype=float) for _, _, uv, _ in faces],
                   dtype=float)
    normals = np.array([np.asarray(n, dtype=float) for _, _, _, n in faces],
                       dtype=float)
    is_floor = np.array([box_index == 0 for box_index, _, _, _ in faces],
                        dtype=bool)
    centroids = corners.mean(axis=1)
    return corners, uvs, normals, is_floor, centroids


def camera_matrix(eye, target):
    """3x3 matrix that turns a world offset into camera space in one matmul."""
    right, up, forward = camera_basis(eye, target)
    return np.array([right, up, forward], dtype=float)


def draw_face_fast(renderer, quad, uv, normal, texture, base_rgb, matrix, eye,
                   project, view_dir, light_dir, use_texture=True):
    """Same result as textured_maze.draw_face, without rebuilding the basis."""
    if float(normal @ view_dir) > 0.0:
        return
    intensity = AMBIENT + DIFFUSE * max(0.0, float(normal @ light_dir))
    cam = (quad - eye) @ matrix.T

    shaded = None
    if not use_texture:
        shaded = np.clip(np.asarray(base_rgb, dtype=float) * intensity,
                         0, 255).astype(np.uint8)

    for tri in TRI_QUADS:
        tri_cam = [cam[i] for i in tri]
        tri_uv = [uv[i] for i in tri]
        for piece in clip_near_attrs(tri_cam, tri_uv, NEAR):
            pa, pb, pc, uva, uvb, uvc = piece
            sa, sb, sc = project(pa), project(pb), project(pc)
            if sa is None or sb is None or sc is None:
                continue
            if use_texture:
                renderer.draw_triangle(sa, sb, sc, uv=(uva, uvb, uvc),
                                       texture=texture, intensity=intensity)
            else:
                renderer.draw_triangle(sa, sb, sc, rgb=shaded)


def visible_faces(corners, centroids, normals, eye, target, view_dir,
                  is_floor=None):
    """Indices of faces that can reach the screen, tested in one numpy pass.

    A face is kept when one of its corners falls inside the view frustum (with a
    margin, because an edge can cross the border while the corners sit outside),
    or when its centroid does. Faces that point away from the camera are
    dropped at the same time.

    The floor is one single quad spanning the whole maze, so no corner of it is
    ever inside the frustum when the camera stands in a corner cell, and the
    test above would delete the ground from the picture. It is kept
    unconditionally: the near clipping in draw_face_fast handles the rest.
    """
    right, up, forward = camera_basis(eye, target)
    rel = corners - eye
    cz = rel @ forward
    tan_v = math.tan(math.radians(FOV_Y) / 2.0)
    tan_h = tan_v * RENDER_W / RENDER_H
    margin = 1.3

    inside = ((np.abs(rel @ right) <= cz * tan_h * margin)
              & (np.abs(rel @ up) <= cz * tan_v * margin)
              & (cz > NEAR))
    keep = np.any(inside, axis=1)

    c_rel = centroids - eye
    c_cz = c_rel @ forward
    keep |= ((np.abs(c_rel @ right) <= c_cz * tan_h * margin)
             & (np.abs(c_rel @ up) <= c_cz * tan_v * margin)
             & (c_cz > NEAR))

    if is_floor is not None:
        keep |= is_floor

    keep &= (normals @ view_dir) < 0.0
    return np.nonzero(keep)[0]


def in_frustum(points, matrix, eye, margin=1.2):
    """Which of these world points can appear on screen, in one numpy pass.

    The puddles and the birds used to be drawn whether or not they were in view;
    on a 12 x 12 board that is 180 puddle triangles per frame, most of them
    outside the picture and each paying the fixed per-triangle cost.
    """
    rel = np.asarray(points, dtype=float)
    if rel.size == 0:
        return []
    rel = rel.reshape(-1, 3) - eye
    cam = rel @ matrix.T
    tan_v = math.tan(math.radians(FOV_Y) / 2.0)
    tan_h = tan_v * RENDER_W / RENDER_H
    cz = cam[:, 2]
    return ((np.abs(cam[:, 0]) <= cz * tan_h * margin)
            & (np.abs(cam[:, 1]) <= cz * tan_v * margin)
            & (cz > NEAR))


class Level:
    """Everything one level needs: the maze, its meshes, the hazards and the apples.

    A new level rebuilds all of it, which is what chapter 12.3 asks for
    ("increment level and regenerate the maze, apples, water, and enemies").
    """

    def __init__(self, cols, rows, seed, number=1):
        self.cols, self.rows = cols, rows
        self.number = number
        self.seed = seed
        self.maze, self.start_cell, self.finish_cell = generate_maze(
            cols, rows, seed=seed)
        self.boxes = scene_boxes(self.maze, cols, rows)
        self.faces = scene_faces(self.boxes)
        self.scene = prepare_faces(self.faces)
        self.aabbs = wall_aabbs(self.boxes)
        self.holes = make_water_holes(cols, rows, self.start_cell,
                                      self.finish_cell, seed)
        blocked = {hole.cell for hole in self.holes}
        self.birds = make_birds(cols, rows, self.start_cell, self.finish_cell,
                                blocked, self.maze, seed, number)
        self.apples = make_apples(cols, rows, self.start_cell,
                                  self.finish_cell, blocked, seed)
        self.start = cell_center_3d(*self.start_cell)
        self.finish = cell_center_3d(*self.finish_cell)

    def report(self):
        print(f"maze {self.cols}x{self.rows}  start={self.start_cell}"
              f"  finish={self.finish_cell}")
        print(f"boxes: {len(self.boxes)}  walls in collision: {len(self.aabbs)}")
        print(f"puddles: {len(self.holes)}  birds: {len(self.birds)}"
              f"  apples: {len(self.apples)}"
              f"  (values {[apple.value for apple in self.apples]})")

    def reset_player(self, player):
        player.x, player.z = float(self.start[0]), float(self.start[2])
        player.y, player.vertical_speed = GROUND_Y, 0.0


def render_frame(renderer, level, player, floor_tex, wall_tex, light_dir,
                 use_texture=True, now=0.0, unlocked=False, sparks=None,
                 depth_test=True):
    """Draw the maze, the props and the worm.

    Returns (ms, faces drawn, sparks drawn, sparks occluded).
    """
    eye, target = worm_camera(player)
    view_dir = normalized(target - eye)
    matrix = camera_matrix(eye, target)
    project = lambda c: perspective_project(c, RENDER_W, RENDER_H, FOV_Y, NEAR)

    t0 = time.perf_counter()
    renderer.clear()

    corners, uvs, normals, is_floor, centroids = level.scene
    drawn = visible_faces(corners, centroids, normals, eye, target, view_dir,
                          is_floor)
    # Near to far: the floor and the far walls then fail the depth test early and
    # never pay for texture sampling, which is where most of the frame time goes.
    depths = (centroids[drawn] - eye) @ view_dir
    for index in drawn[np.argsort(depths)]:
        floor = is_floor[index]
        draw_face_fast(renderer, corners[index], uvs[index], normals[index],
                       floor_tex if floor else wall_tex,
                       FLOOR_RGB if floor else WALL_RGB,
                       matrix, eye, project, view_dir, light_dir, use_texture)

    if level.holes:
        seen = in_frustum([(hole.x, WATER_Y, hole.z) for hole in level.holes],
                          matrix, eye)
        for hole, keep in zip(level.holes, seen):
            if not keep:
                continue
            draw_disc(renderer, (hole.x, WATER_Y, hole.z), hole.radius(),
                      WATER_OUTER_RGB, matrix, eye, project, view_dir)
            draw_disc(renderer, (hole.x, WATER_Y + 0.004, hole.z),
                      hole.radius() * 0.72, WATER_INNER_RGB,
                      matrix, eye, project, view_dir)

    if level.birds:
        seen = in_frustum([bird.position() for bird in level.birds], matrix, eye)
        for bird, keep in zip(level.birds, seen):
            if keep:
                draw_bird(renderer, bird, matrix, eye, project, view_dir,
                          light_dir, now)

    if level.apples:
        seen = in_frustum([apple.center(now) for apple in level.apples
                           if not apple.collected], matrix, eye)
        visible = [apple for apple in level.apples if not apple.collected]
        for apple, keep in zip(visible, seen):
            if keep:
                draw_apple(renderer, apple, matrix, eye, project, view_dir,
                           light_dir, now)

    col, row = level.finish_cell
    east_open = not level.maze[row][col]["walls"]["E"]
    draw_flag(renderer, level.finish_cell, unlocked, matrix, eye, project,
              view_dir, light_dir, wave=1.0 if east_open else -1.0)

    draw_sphere(renderer, player.position(), player.radius, WORM_RGB,
                matrix, eye, project, view_dir, light_dir,
                rings=WORM_RINGS, segments=WORM_SEGMENTS)
    # two eyes, so the worm reads as a character and not as another red apple
    centre = player.position()
    for side in (-1.0, 1.0):
        eye_p = centre + np.array([side * 0.085, 0.07, 0.155])
        draw_sphere(renderer, eye_p, 0.042, WORM_EYE_RGB,
                    matrix, eye, project, view_dir, light_dir, rings=3,
                    segments=5)
        draw_sphere(renderer, eye_p + np.array([0.0, 0.0, 0.028]), 0.022,
                    WORM_PUPIL_RGB, matrix, eye, project, view_dir, light_dir,
                    rings=2, segments=4)

    # sparks go last so their depth can be compared with the finished Z-buffer:
    # a firework behind a wall is then simply skipped
    drawn_sparks = hidden_sparks = 0
    if sparks:
        drawn_sparks, hidden_sparks = particles.draw(
            renderer, sparks, matrix, eye, project, CLIP_NEAR,
            depth_test=depth_test)
    return ((time.perf_counter() - t0) * 1000.0, len(drawn),
            drawn_sparks, hidden_sparks)


def draw_hud(screen, font, level, player, frame_ms, triangles, walked,
             message, progress, now, drawn=None, use_texture=True,
             sparks=(0, 0), view="colour"):
    state = "SLIDING" if player.blocked else "free"
    if player.airborne():
        state = "AIR"
    mode = "textured" if use_texture else "flat"
    if view != "colour":
        mode = f"{mode} + {view}"

    # translucent strips: grey text on bright grass is hard to read
    top = pygame.Surface((WIDTH, 62), pygame.SRCALPHA)
    top.fill((8, 14, 10, 150))
    screen.blit(top, (0, 0))
    bottom = pygame.Surface((WIDTH, 92), pygame.SRCALPHA)
    bottom.fill((8, 14, 10, 150))
    screen.blit(bottom, (0, HEIGHT - 92))

    status = font.render(
        f"M11 | LVL {progress.level}  SCORE {progress.score}"
        f"  APPLES {progress.collected}/{APPLE_TOTAL}"
        f"  TIME {progress.clock(now)}",
        True, (255, 214, 120) if progress.level_ready() else (232, 240, 226))
    screen.blit(status, (12, 10))

    cull = f" faces {drawn}/{len(level.faces)}" if drawn is not None else ""
    info = font.render(
        f"{level.cols}x{level.rows}  x {player.x:.2f} z {player.z:.2f}"
        f"  feet {player.feet_y():.2f}  {state}  {mode}",
        True, (150, 172, 160))
    screen.blit(info, (12, HEIGHT - 76))

    perf = font.render(
        f"tris {triangles}{cull}  |  {frame_ms:.0f} ms  {RENDER_W}x{RENDER_H}"
        f"  |  sparks {sparks[0]}/{MAX_PARTICLES}"
        f" (hidden {sparks[1]})  walked {walked:.1f}",
        True, (150, 172, 160))
    screen.blit(perf, (12, HEIGHT - 50))

    hint = font.render("W A S D move   SPACE jump   F firework   T textures"
                       "   F3 depth   R restart   ESC quit",
                       True, (150, 172, 160))
    screen.blit(hint, (12, HEIGHT - 26))
    if message:
        note = font.render(message, True, (255, 214, 120))
        screen.blit(note, (12, 34))


def present(screen, renderer, level, player, floor_tex, wall_tex, font,
            progress, walked, message, use_texture=True, now=0.0, sparks=None,
            view="colour", depth_test=True):
    frame_ms, drawn, spark_drawn, spark_hidden = render_frame(
        renderer, level, player, floor_tex, wall_tex, LIGHT_DIRS[0],
        use_texture, now, progress.level_ready(), sparks, depth_test)
    if view == "depth":
        # F3: show the depth buffer instead of the colour buffer. The particles
        # are already drawn on top of the colour buffer, so they are excluded
        # here on purpose: this view is about the Z-buffer itself.
        rgb = np.ascontiguousarray(renderer.depth_visual(), dtype=np.uint8)
        surf = pygame.image.frombuffer(rgb.tobytes(),
                                       (renderer.w, renderer.h), "RGB")
        spark_drawn = spark_hidden = 0
    else:
        surf = renderer.to_surface(pygame)
    if (renderer.w, renderer.h) != screen.get_size():
        surf = pygame.transform.scale(surf, screen.get_size())
    screen.blit(surf, (0, 0))
    draw_hud(screen, font, level, player, frame_ms, renderer.triangles,
             walked, message, progress, now, drawn, use_texture,
             (spark_drawn, spark_hidden), view)
    pygame.display.flip()
    return frame_ms


def crossable_hole(holes, maze, cols, rows):
    """A puddle with an open south neighbour, so the worm can walk into it."""
    for hole in holes:
        col, row = hole.cell
        if row + 1 < rows and not maze[row][col]["walls"]["S"]:
            return hole, (col, row + 1)
    return (holes[0], None) if holes else (None, None)


def reachable_side(level, cell):
    """A cell next to the target with an open wall between them, or None."""
    col, row = cell
    for name, (dx, dz, _) in DIRECTIONS.items():
        if level.maze[row][col]["walls"][name]:
            continue
        nc, nr = col + dx, row + dz
        if 0 <= nc < level.cols and 0 <= nr < level.rows:
            return (nc, nr)
    return None


def finish_reached(level, player):
    """Same rule as check_finish(): 17 px from the exit cell centre, in XZ."""
    delta = player.position() - level.finish
    return float(math.hypot(delta[0], delta[2])) < FLAG_HIT_RADIUS


def wall_between(level, player, reach=3.0):
    """The nearest wall box that crosses the view axis in front of the worm."""
    best = None
    for x0, x1, z0, z1 in level.aabbs:
        if not (x0 - 0.25 <= player.x <= x1 + 0.25):
            continue
        if not (player.z - reach < z1 <= player.z - 0.2):
            continue
        # z1 is the near face, so the largest z1 is the closest wall
        if best is None or z1 > best[3]:
            best = (x0, x1, z0, z1)
    return best


def demo(screen, renderer, level, player, floor_tex, wall_tex, font, progress,
         holder):
    """Scripted run: wall slide, jump, puddle, then apples and the level rule."""
    dt = 1.0 / 60.0
    state = {"walked": 0.0, "now": 0.0, "collect": False, "sparks": [],
             "view": "colour", "depth_test": True, "free_particles": False}

    def shot(name, tag=""):
        present(screen, renderer, holder[0], player, floor_tex, wall_tex, font,
                progress, state["walked"], tag, True, state["now"],
                sparks=state["sparks"], view=state["view"],
                depth_test=state["depth_test"])
        pygame.image.save(screen, f"screenshots/{name}")
        print(f"saved: screenshots/{name}")

    def animate(seconds):
        """Advance time and the particle pile without moving the worm."""
        for _ in range(int(round(seconds / dt))):
            state["now"] += dt
            particles.update(state["sparks"], dt)

    def advance(seconds, direction=None, jump_at=None, shot_name=None, tag="",
                stop_on_danger=False, stop_on_apple=False):
        nonlocal holder
        steps = int(round(seconds / dt))
        danger = False
        for step in range(steps):
            t = step * dt
            state["now"] += dt
            if jump_at is not None and abs(t - jump_at) < dt / 2:
                player.jump()
            if direction is not None:
                before = player.position()
                player.move(direction, dt, holder[0].aabbs)
                state["walked"] += float(np.linalg.norm(player.position() - before))
            player.update_vertical(dt)
            update_birds(holder[0].birds, holder[0].maze, dt)
            # scenes 1-3 are the M8/M9 evidence: collecting an apple by accident
            # there would leave the score in shot 12-16 at something other than 0
            taken = (collect_apples(holder[0].apples, player, state["now"])
                     if state["collect"] else [])
            if taken:
                progress.collect(taken)
                if shot_name:
                    shot(shot_name, tag.format(score=progress.score,
                                               apples=progress.collected))
                return False
            if stop_on_danger and in_water(player, holder[0].holes):
                danger = True
                if shot_name:
                    shot(shot_name, tag)
                break
        if shot_name and not danger:
            shot(shot_name, tag)
        return danger

    north = np.array([0.0, 0.0, -1.0])
    north_east = np.array([0.7071, 0.0, -0.7071])

    print("--- 1. wall sliding (M8) ---")
    shot("12_worm_start.png", "M8: worm in the start cell")
    advance(1.0, direction=north)
    print(f"    north wall: x {player.x:.3f} z {player.z:.3f}"
          f"  blocked {player.blocked}")
    advance(1.4, direction=north_east, shot_name="13_worm_against_wall.png",
            tag="M8: diagonal input, one axis blocked, the other slides")
    print(f"    after the diagonal: x {player.x:.3f} z {player.z:.3f}"
          f"  blocked {player.blocked}")
    holder[0].reset_player(player)

    print("--- 2. jump in place (M9) ---")
    advance(0.32, jump_at=0.0, shot_name="14_worm_jump_apex.png",
            tag="M9: jump apex, world Y, feet off the floor")
    advance(0.40)
    print(f"    landed: feet {player.feet_y():.3f}  y {player.y:.3f}")

    print("--- 3. puddles and birds (M9) ---")
    hole, approach = crossable_hole(holder[0].holes, holder[0].maze,
                                    holder[0].cols, holder[0].rows)
    if approach is None:
        print("    no puddle with an open south side, skipping the puddle test")
    else:
        approach_z = hole.z + 1.2      # inside the south neighbour cell
        player.x, player.z = hole.x, approach_z
        player.y, player.vertical_speed = GROUND_Y, 0.0
        drowned = advance(0.35, direction=north,
                          shot_name="15_worm_at_puddle.png",
                          tag="M9: grounded, walking into the puddle",
                          stop_on_danger=True)
        print(f"    drowned: {drowned} at z {player.z:.3f}")

        player.x, player.z = hole.x, approach_z
        player.y, player.vertical_speed = GROUND_Y, 0.0
        drowned = advance(0.45, direction=north, jump_at=0.0,
                          stop_on_danger=True)
        print(f"    jump across: drowned {drowned}"
              f"  feet {player.feet_y():.3f}  z {player.z:.3f}")
        shot("16_worm_jumped_puddle.png", "M9: jumped clear over the puddle")

    holder[0].reset_player(player)

    print("--- 4. apples and the exit (M10) ---")
    state["collect"] = True
    # pick the apple cell with the most open sides, so the screenshot shows a
    # corridor instead of the inside of a dead end
    apple = max(holder[0].apples,
                key=lambda a: len(open_neighbours(holder[0].maze, a.cell)))
    side = reachable_side(holder[0], apple.cell)
    if side is not None:
        player.x, player.z = float(side[0] + 0.5), float(side[1] + 0.5)
    else:
        player.x, player.z = apple.x, apple.z + 0.9
    player.y, player.vertical_speed = GROUND_Y, 0.0
    move = np.array([apple.x - player.x, 0.0, apple.z - player.z])
    if float(np.linalg.norm(move)) > 1e-6:
        move = move / float(np.linalg.norm(move))
    # keep the worm upright against the wall that guards the apple
    step = np.array([math.copysign(min(abs(move[0]), 0.7071), move[0]),
                     0.0, math.copysign(min(abs(move[2]), 0.7071), move[2])])
    advance(0.9, direction=step, shot_name="18_apple_collect.png",
            tag="M10: apple collected, score {score} ({apples}/4 apples)")
    print(f"    apples {progress.collected}/{APPLE_TOTAL}"
          f"  score {progress.score}")

    print("--- 5. the exit stays locked without the apples (M10) ---")
    player.x, player.z = float(holder[0].finish[0]), float(holder[0].finish[2] + 1.0)
    player.y, player.vertical_speed = GROUND_Y, 0.0
    for _ in range(20):                     # walk in until the flag is close
        if finish_reached(holder[0], player):
            break
        advance(0.02, direction=north)
    print(f"    standing at the locked exit with {progress.collected}"
          f"/{APPLE_TOTAL} apples")
    capitalize = ("Collect the remaining"
                  f" {progress.missing()} apples to unlock the exit")
    shot("19_exit_locked.png", capitalize)
    # default rule: standing on the locked exit is harmless, the flag stays red
    player.x, player.z = float(holder[0].finish[0]), float(holder[0].finish[2] + 1.0)
    player.y, player.vertical_speed = GROUND_Y, 0.0
    print(f"    walked away from the locked exit without being sent back"
          f" (start is ({holder[0].start[0]:.2f},"
          f" {holder[0].start[2]:.2f}))")

    print("--- 6. all four apples, then the level rule (M10) ---")
    for rest in remaining(holder[0].apples):
        player.x, player.z = rest.x, rest.z
        player.y, player.vertical_speed = GROUND_Y, 0.0
        progress.collect(collect_apples(holder[0].apples, player, state["now"]))
    print(f"    apples {progress.collected}/{APPLE_TOTAL}"
          f"  score {progress.score}  ready {progress.level_ready()}")

    player.x, player.z = float(holder[0].finish[0]), float(holder[0].finish[2] + 1.0)
    player.y, player.vertical_speed = GROUND_Y, 0.0
    reached = False
    for _ in range(20):
        if finish_reached(holder[0], player):
            reached = True
            break
        advance(0.02, direction=north)
    if reached:
        shot("20_exit_unlocked.png",
             "M10: 4/4 apples, the exit flag turns green")
        progress.advance()
        holder[0] = Level(holder[0].cols, holder[0].rows,
                          SEED + progress.level, progress.level)
        holder[0].reset_player(player)
        print(f"    level advanced to {progress.level}"
              f"  score kept at {progress.score}"
              f"  apples regenerated: {len(holder[0].apples)}")
        state["now"] += 0.2
        shot("21_level_2.png", f"M10: level {progress.level} maze and apples")
    else:
        print("    the worm did not reach the exit cell, no level-up shot")

    print("--- 7. sparks and fireworks (M11) ---")
    holder[0].reset_player(player)
    particles.clear(state["sparks"])
    mouth = player.position() + np.array([0.0, 0.55, -1.10])
    # spawn rule 1 (handbook 13.2, required): F launches the burst from the
    # worm's own 3D position
    create_firework(state["sparks"], mouth, np.random.default_rng(11), 110)
    create_spark(state["sparks"], mouth, np.random.default_rng(12), 40)
    animate(0.20)
    shot("25_fireworks_worm.png",
         "M11: a firework launched from the worm's 3D position")
    print(f"    sparks alive: {particles.total(state['sparks'])}")

    print("--- 8. a firework behind a wall (M11) ---")
    particles.clear(state["sparks"])
    box = wall_between(holder[0], player)
    if box is None:
        print("    no wall in front of the worm, skipping the occlusion test")
    else:
        behind = np.array([player.x, 0.25, box[2] - 0.30])
        in_front = np.array([player.x + 0.62, 0.06, player.z - 0.45])
        create_firework(state["sparks"], behind, np.random.default_rng(21), 90)
        create_spark(state["sparks"], in_front, np.random.default_rng(22), 30)
        animate(0.10)
        shot("26_particles_occluded.png",
             "M11: the burst behind the wall is depth-tested out")
        # the same frame with the test switched off: the beads come through
        state["depth_test"] = False
        shot("27_particles_through_wall.png",
             "M11: same frame without the depth test")
        state["depth_test"] = True
        print(f"    sparks in the pile: {particles.total(state['sparks'])}")

    print("--- 9. the Z-buffer view (M11) ---")
    particles.clear(state["sparks"])
    state["view"] = "depth"
    shot("28_depth_buffer.png",
         "M11: F3 depth view, near surfaces bright, empty pixels black")
    state["view"] = "colour"

    holder[0].reset_player(player)
    print(f"end position ({player.x:.3f}, {player.z:.3f})"
          f"  walked {state['walked']:.2f}  score {progress.score}")


def main():
    run_demo = "--demo" in sys.argv
    shot_path = arg_value("--shot", None)

    # The scripted shots are composed inside a 5 x 5 maze, so --demo keeps that
    # size unless the caller asks for another one explicitly.
    default_cols = DEMO_COLS if run_demo else COLS
    default_rows = DEMO_ROWS if run_demo else ROWS
    cols = int(arg_value("--cols", default_cols))
    rows = int(arg_value("--rows", default_rows))
    use_texture = "--flat" not in sys.argv
    # default: the locked exit is harmless. --strict-finish keeps the source
    # reset_ball() behaviour for a side-by-side comparison in the report.
    strict_finish = "--strict-finish" in sys.argv

    global RENDER_W, RENDER_H, WIDTH, HEIGHT
    global WORM_RINGS, WORM_SEGMENTS, BIRD_RINGS, BIRD_SEGMENTS, DISC_SEGMENTS
    global APPLE_RINGS, APPLE_SEGMENTS
    res = arg_value("--res", None)
    if res:
        RENDER_W, RENDER_H = (int(part) for part in res.lower().split("x"))
    elif shot_path is not None or run_demo:
        # screenshots are single frames: spend the time on a smoother mesh
        RENDER_W, RENDER_H = SHOT_W, SHOT_H
        WORM_RINGS, WORM_SEGMENTS = 8, 12
        BIRD_RINGS, BIRD_SEGMENTS = 6, 8
        DISC_SEGMENTS = 18
        APPLE_RINGS, APPLE_SEGMENTS = 6, 9

    # --no-occlusion is the deliberate "wrong" build used for the before/after
    # particle screenshot: sparks then draw straight through walls
    depth_test = "--no-occlusion" not in sys.argv

    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("Worm's Garden 3D - M11 particles, HUD, quality")
    font = pygame.font.SysFont("consolas", 18)
    clock = pygame.time.Clock()

    level = Level(cols, rows, SEED, 1)
    player = Player(level.start[0], level.start[2])
    floor_tex, wall_tex = textures.get("grass"), textures.get("stone")
    now = time.perf_counter()
    # the scripted demo counts from zero, so the HUD clock shows the run time
    progress = Progress(0.0 if run_demo else now)
    holder = [level]

    level.report()
    print(f"worm: start ({player.x:.2f}, {player.y:.2f}, {player.z:.2f})"
          f"  radius {player.radius}  speed {player.speed}"
          f"  jump v0 {JUMP_SPEED} g {GRAVITY}")

    renderer = Renderer(RENDER_W, RENDER_H,
                        background_image(RENDER_W, RENDER_H))

    if run_demo:
        demo(screen, renderer, level, player, floor_tex, wall_tex, font,
             progress, holder)
        pygame.quit()
        print("window closed, program finished")
        return

    running = True
    walked = 0.0
    message = ""
    sparks = []                   # M11 particle pile, capped in particles.py
    view = "colour"               # F3 switches to the Z-buffer view
    rng = np.random.default_rng(SEED)

    while running:
        dt = min(clock.tick(60) / 1000.0, 0.05)
        now = time.perf_counter()
        anim = progress.elapsed(now)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE and shot_path is None:
                new_w, new_h = max(400, event.w), max(320, event.h)
                # keep the pixel budget of the framebuffer and just bend it to
                # the new aspect ratio (handbook 14.2)
                budget = float(RENDER_W * RENDER_H)
                aspect = new_w / float(new_h)
                RENDER_H = max(240, min(760,
                                        int(round(math.sqrt(budget / aspect)))))
                RENDER_W = max(320, min(1100, int(round(RENDER_H * aspect))))
                WIDTH, HEIGHT = new_w, new_h
                renderer = Renderer(RENDER_W, RENDER_H,
                                    background_image(RENDER_W, RENDER_H))
                screen = pygame.display.set_mode((WIDTH, HEIGHT),
                                                 pygame.RESIZABLE)
                message = (f"resized: {WIDTH}x{HEIGHT} window,"
                           f" {RENDER_W}x{RENDER_H} framebuffer")
                print(message)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    player.jump()
                elif event.key == pygame.K_f:
                    # handbook 13.2: the required 3D spawn rule, a burst at the
                    # worm's own position
                    mouth = player.position() + np.array([0.0, 0.42, 0.0])
                    create_firework(sparks, mouth, rng, MAX_BURST)
                    create_spark(sparks, mouth, rng, 20)
                elif event.key == pygame.K_t:
                    use_texture = not use_texture
                elif event.key == pygame.K_F3:
                    view = "colour" if view == "depth" else "depth"
                    message = (f"view: {view}"
                               + (" (Z-buffer, handbook 14.3)" if view == "depth"
                                  else ""))
                elif event.key == pygame.K_r:
                    progress.reset(now)
                    holder[0] = Level(cols, rows, SEED, 1)
                    holder[0].reset_player(player)
                    walked = 0.0
                    particles.clear(sparks)
                    message = "restart: level 1, score 0"
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # spawn rule 2 (extension in handbook 13.2): the click defines a
                # ray, the ray meets the ground plane, the burst lands there
                eye, target = worm_camera(player)
                mx = event.pos[0] * RENDER_W / float(WIDTH)
                my = event.pos[1] * RENDER_H / float(HEIGHT)
                hit = ground_intersection(eye, target, mx, my,
                                          RENDER_W, RENDER_H, FOV_Y)
                if hit is None:
                    message = "that click points too far away, no ground hit"
                elif not (0.0 <= hit[0] <= cols and 0.0 <= hit[2] <= rows):
                    message = "that point is outside the lawn"
                else:
                    create_firework(sparks, hit + np.array([0.0, 0.45, 0.0]),
                                    rng, MAX_BURST)
                    message = (f"firework on the ground at"
                               f" x {hit[0]:.2f} z {hit[2]:.2f}")

        level = holder[0]
        direction = movement_direction(pygame.key.get_pressed(), pygame)
        before = player.position()
        player.move(direction, dt, level.aabbs)
        player.update_vertical(dt)
        walked += float(np.linalg.norm(player.position() - before))
        update_birds(level.birds, level.maze, dt)
        particles.update(sparks, dt)

        taken = collect_apples(level.apples, player, anim)
        gained = sum(apple.value for apple in taken)
        if taken:
            progress.collect(taken)
            message = (f"+{gained} apple"
                       f"  ({progress.collected}/{APPLE_TOTAL},"
                       f" score {progress.score})")

        if in_water(player, level.holes):
            level.reset_player(player)
            message = "splash! back to the start"
        elif any(bird_hits(player, bird) for bird in level.birds):
            level.reset_player(player)
            message = "bird hit! back to the start"
        elif finish_reached(level, player):
            if progress.level_ready():
                progress.advance()
                holder[0] = Level(cols, rows, SEED + progress.level,
                                  progress.level)
                holder[0].reset_player(player)
                level = holder[0]
                message = (f"level {progress.level}!  score {progress.score}"
                           f" carried over, 4 new apples")
            elif strict_finish:
                level.reset_player(player)
                message = ("exit locked: collect the remaining"
                           f" {progress.missing()} apples")
            else:
                # the exit sits in a corridor, so walking over it must not
                # punish the player: the flag stays locked and only the notice
                # fires (handbook 12.3 allows this, it has to be documented)
                message = ("exit locked: collect the remaining"
                           f" {progress.missing()} apples to open it")
        elif not taken:
            near = nearest_water(player, level.holes)
            message = ("SPACE jumps the puddle" if near is not None
                       and near < 1.1 and not player.airborne() else "")

        frame_ms = present(screen, renderer, level, player, floor_tex, wall_tex,
                           font, progress, walked, message, use_texture, anim,
                           sparks=sparks, view=view, depth_test=depth_test)

        if shot_path is not None:
            pygame.image.save(screen, shot_path)
            print(f"saved: {shot_path}  ({frame_ms:.0f} ms per frame)")
            break

    pygame.quit()
    print(f"worm finished at ({player.x:.2f}, {player.z:.2f})"
          f"  walked {walked:.2f} units  level {progress.level}"
          f"  score {progress.score}")
    print("window closed, program finished")


if __name__ == "__main__":
    main()
