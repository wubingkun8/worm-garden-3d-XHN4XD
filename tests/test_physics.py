"""Tests for M8: world-aligned movement and circle-vs-AABB wall collision."""
import os
import sys

import numpy as np
import pygame

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geometry import wall_box, scene_boxes
from maze import generate_maze
from physics import (Player, movement_direction, circle_hits_xz, wall_aabbs,
                     WORM_RADIUS, WORM_SPEED, GROUND_Y)


def keys_down(*names):
    """Minimal stand-in for pygame.key.get_pressed()."""
    state = {key: False for key in (pygame.K_w, pygame.K_a, pygame.K_s, pygame.K_d)}
    for name in names:
        state[getattr(pygame, "K_" + name)] = True
    return state


def single_wall():
    """One north wall of cell (0, 0): a thin box across z = 0, 0.16 thick."""
    center, size = wall_box(0, 0, "N")
    return (center[0] - size[0] / 2, center[0] + size[0] / 2,
            center[2] - size[2] / 2, center[2] + size[2] / 2)


def test_diagonal_input_is_normalised():
    """W+D must be a unit vector, not 1.41 units, or diagonals would be faster."""
    forward_right = movement_direction(keys_down("w", "d"), pygame)
    assert abs(np.linalg.norm(forward_right) - 1.0) < 1e-12
    assert abs(forward_right[0] - 0.7071067811865476) < 1e-12
    assert abs(forward_right[2] + 0.7071067811865476) < 1e-12

    north = movement_direction(keys_down("w"), pygame)
    assert abs(north[2] + 1.0) < 1e-12, "W must be north (-Z)"

    still = movement_direction(keys_down(), pygame)
    assert np.allclose(still, 0.0)


def test_speed_is_framerate_independent():
    """One second of movement must cover the same distance at 60 and 200 fps."""
    coarse = Player(1.0, 1.0)
    fine = Player(1.0, 1.0)
    direction = np.array([1.0, 0.0, 0.0])

    for _ in range(60):
        coarse.move(direction, 1.0 / 60.0, [])
    for _ in range(200):
        fine.move(direction, 1.0 / 200.0, [])

    assert abs(coarse.x - fine.x) < 1e-9
    assert abs(coarse.x - (1.0 + WORM_SPEED)) < 1e-9


def test_worm_stops_at_wall_surface():
    """Walking into a wall must stop exactly one radius short of its face."""
    box = single_wall()
    player = Player(0.5, 1.0)
    for _ in range(120):
        player.move(np.array([0.0, 0.0, -1.0]), 1.0 / 60.0, [box])

    gap = player.z - box[3]
    assert gap > WORM_RADIUS - 1e-6, f"penetrated the wall, gap {gap}"
    assert gap < WORM_RADIUS + 0.06, f"stopped too far away, gap {gap}"


def test_worm_slides_along_wall():
    """Blocked on Z, the worm must keep its X component (handbook 11.1)."""
    box = single_wall()
    player = Player(0.5, 1.0)
    direction = np.array([1.0, 0.0, -1.0]) / np.sqrt(2.0)
    for _ in range(12):
        player.move(direction, 1.0 / 60.0, [box])

    assert player.z - box[3] > WORM_RADIUS - 1e-6, "the worm pushed into the wall"
    assert player.x > 0.9, f"the worm stopped sliding, x = {player.x}"


def test_no_tunnelling_with_long_frames():
    """A long frame must not jump through a 0.16 thick wall."""
    box = single_wall()
    player = Player(0.5, 1.0)
    player.move(np.array([0.0, 0.0, -1.0]), 0.5, [box])   # 2.3 units in one go
    assert player.z - box[3] > WORM_RADIUS - 1e-6, "tunnelled through the wall"


def test_collision_footprint_matches_mesh():
    """The horizontal AABB must be exactly the 3D wall footprint (0.16 thick)."""
    maze, _, _ = generate_maze(3, 3, seed=7)
    boxes = scene_boxes(maze, 3, 3)
    aabbs = wall_aabbs(boxes)
    assert len(aabbs) == len(boxes) - 1, "every wall box must get one footprint"

    thin = [abs(box[1] - box[0]) for box in aabbs if abs(box[1] - box[0]) < 0.5]
    assert thin, "no thin walls found"
    assert all(abs(width - 0.16) < 1e-9 for width in thin)


def test_start_cell_is_free():
    """The worm must not spawn inside a wall."""
    maze, start, _ = generate_maze(3, 3, seed=7)
    boxes = scene_boxes(maze, 3, 3)
    aabbs = wall_aabbs(boxes)
    from maze import cell_center_3d
    centre = cell_center_3d(start[0], start[1])
    player = Player(centre[0], centre[2])
    assert not player.collides(player.x, player.z, aabbs)
    assert player.y == GROUND_Y


def test_east_is_right_on_screen():
    """Looking north, +X must land on the right half of the screen (A/D not swapped)."""
    from camera import camera_space, perspective_project, camera_basis
    eye = np.array([0.5, 3.22, 3.70])
    target = np.array([0.5, 0.25, 2.50])
    right, _, forward = camera_basis(eye, target)
    assert np.allclose(right, [1.0, 0.0, 0.0]), "east must be the camera right vector"
    assert forward[2] < 0, "forward must point north (-Z)"

    here = perspective_project(camera_space(np.array([0.5, 0.0, 2.5]), eye, target), 900, 700)
    east = perspective_project(camera_space(np.array([1.5, 0.0, 2.5]), eye, target), 900, 700)
    north = perspective_project(camera_space(np.array([0.5, 0.0, 1.5]), eye, target), 900, 700)
    assert east[0] > here[0] + 50, "+X must move right on screen"
    assert north[1] < here[1], "-Z must move up on screen"


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"{test.__name__}: ok")
    print("physics tests passed")
