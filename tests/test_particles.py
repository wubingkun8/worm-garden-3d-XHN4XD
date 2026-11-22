"""M11 tests: particle bounds, lifetimes, 3D motion, occlusion and spawn rules."""
import math
import sys

import numpy as np

sys.path.insert(0, ".")
import particles
from camera import ground_intersection, camera_basis, normalized, camera_space


def test_cap_is_bounded():
    pile = []
    for _ in range(40):
        particles.create_firework(pile, np.array([1.0, 0.5, 2.0]),
                                  np.random.default_rng(1), 120)
    assert len(pile) == particles.MAX_PARTICLES, len(pile)
    print(f"  cap: {particles.MAX_PARTICLES} particles kept out of 4800")


def test_lifetime_removes_particles():
    pile = []
    rng = np.random.default_rng(2)
    particles.create_spark(pile, np.array([0.0, 0.5, 0.0]), rng, 60)
    first = len(pile)
    assert 0 < first <= 60
    for _ in range(200):                       # 200 frames = 3.3 s > max life 1.0
        particles.update(pile, 1.0 / 60.0)
    assert not pile, f"{len(pile)} particles survived their lifetime"
    print(f"  lifetime: {first} sparks gone after 3.3 s")


def test_burst_does_not_share_state():
    """Every bead owns its own pos and vel arrays.

    np.asarray() on an array returns the same object, so a burst built with it
    moved as a single point; this test keeps that regression out.
    """
    pile = []
    origin = np.array([1.0, 0.5, 2.0])
    particles.create_firework(pile, origin, np.random.default_rng(3), 60)
    assert not np.shares_memory(pile[0]["pos"], pile[1]["pos"])
    assert not np.shares_memory(pile[0]["pos"], origin)
    before = np.array([p["pos"].copy() for p in pile])
    particles.update(pile, 1.0 / 60.0)
    moved = np.array([p["pos"] for p in pile])
    spread = float(np.abs(moved - before).max())
    # beads must fan out, not stay in one place
    assert float(np.abs(moved - moved[0]).max()) > 0.1, spread
    print(f"  independence: 60 beads spread over"
          f" {float(np.abs(moved - moved[0]).max()):.2f} units after one frame")


def test_motion_is_three_dimensional():
    particle = particles.make_particle(np.array([0.0, 1.0, 0.0]),
                                       np.array([1.0, 2.0, 3.0]),
                                       2.0, 3, (255, 0, 0))
    pile = [particle]
    dt = 1.0 / 60.0
    particles.update(pile, dt)
    # the Y component has to change: a 2D particle would keep y at 0
    assert particle["pos"][0] > 0.0
    assert particle["pos"][2] > 0.0
    assert particle["vel"][1] < 2.0, particle["vel"][1]
    total = 0.0
    for _ in range(60):
        particles.update(pile, dt)
        total += 1.0
    assert particle["pos"][1] != 1.0
    print(f"  motion: spark moved to {np.round(particle['pos'], 3)}"
          f" after {total:.0f} frames")


def test_floor_stops_the_spark():
    particle = particles.make_particle(np.array([0.0, 0.3, 0.0]),
                                       np.array([0.0, -8.0, 0.0]),
                                       2.0, 3, (255, 0, 0))
    pile = [particle]
    lowest = particle["pos"][1]
    for _ in range(30):
        particles.update(pile, 1.0 / 60.0)
        lowest = min(lowest, particle["pos"][1])
    # not in the 2D source: a 3D spark must not fall through the lawn
    assert lowest >= particles.FLOOR_Y - 1e-9, lowest
    print(f"  floor: a falling spark never went below y = {lowest:.3f}")


def block_screen(renderer, depth):
    """Fill the whole framebuffer with one triangle at a constant depth."""
    w, h = renderer.w, renderer.h
    renderer.draw_triangle((0.0, 0.0, depth), (w - 1.0, 0.0, depth),
                           (w - 1.0, h - 1.0, depth), rgb=(120, 120, 120))
    renderer.draw_triangle((0.0, 0.0, depth), (w - 1.0, h - 1.0, depth),
                           (0.0, h - 1.0, depth), rgb=(120, 120, 120))


def test_wall_hides_particles():
    from renderer import Renderer
    renderer = Renderer(64, 64, background=(0, 0, 0))
    renderer.clear()
    wall = 2.0
    block_screen(renderer, wall)

    eye = np.array([0.0, 0.0, 0.0])
    target = np.array([0.0, 0.0, 1.0])
    right, up, forward = camera_basis(eye, target)
    matrix = np.vstack([right, up, forward])
    project = lambda c: (32 + 30 * c[0] / c[2], 32 - 30 * c[1] / c[2], c[2])

    behind = [particles.make_particle(np.array([0.0, 0.0, 4.0]),
                                      np.array([0.0, 0.0, 1.0]), 1.0, 3,
                                      (255, 200, 0))]
    in_front = [particles.make_particle(np.array([0.0, 0.0, 1.0]),
                                        np.array([0.0, 0.0, 1.0]), 1.0, 3,
                                        (255, 200, 0))]

    before = renderer.color.copy()
    drawn_behind, hidden_behind = particles.draw(renderer, behind, matrix, eye,
                                                 project)
    assert (drawn_behind, hidden_behind) == (0, 1), (drawn_behind, hidden_behind)
    assert np.array_equal(renderer.color, before), \
        "the hidden particle changed the image"

    drawn_front, hidden_front = particles.draw(renderer, in_front, matrix, eye,
                                               project)
    assert (drawn_front, hidden_front) == (1, 0), (drawn_front, hidden_front)
    assert not np.array_equal(renderer.color, before), \
        "the near particle drew nothing"

    # the debugging switch paints the same particle straight through the wall
    drawn_cheat, _ = particles.draw(renderer, behind, matrix, eye, project,
                                    depth_test=False)
    assert drawn_cheat == 1
    print("  occlusion: 1 behind the wall hidden, 1 in front drawn,"
          " depth_test=False draws through")


def test_spawn_rule_worm_position():
    """Rule 1 (handbook 13.2): the burst starts at the worm's 3D position."""
    worm = np.array([2.5, 0.22, 3.5])
    origin = worm + np.array([0.0, 0.42, 0.0])
    pile = []
    particles.create_firework(pile, origin, np.random.default_rng(5), 100)
    starts = np.array([p["pos"] for p in pile])
    assert np.allclose(starts, origin), "a firework bead started somewhere else"
    assert all(p["pos"].shape == (3,) for p in pile)
    print(f"  spawn rule 1: {len(pile)} beads started at"
          f" ({origin[0]:.2f}, {origin[1]:.2f}, {origin[2]:.2f})")


def test_spawn_rule_mouse_ray():
    """Rule 2 (extension): the click ray meets y = 0, and only then."""
    eye = np.array([2.5, 3.22, 4.7])
    target = np.array([2.5, 0.25, 3.5])

    # a pixel below the horizon must hit the ground in front of the camera
    hit = ground_intersection(eye, target, 320.0, 260.0, 480, 373, 55.0)
    assert hit is not None, "a click on the lawn produced no ground point"
    assert abs(hit[1]) < 1e-9, hit[1]
    direction = normalized(hit - eye)
    assert direction[1] < 0.0
    # the hit lies further away than the worm, along the view direction
    distance = float(np.linalg.norm(hit - eye))
    assert distance > 1.0

    # a pixel near the top edge of this steep camera points almost at the
    # horizon, so its ground hit is far away: max_t has to reject it
    far = ground_intersection(eye, target, 320.0, -40.0, 480, 373, 55.0,
                              max_t=4.0)
    assert far is None, "the max_t limit did not reject a far ground hit"

    # the same pixel in camera space really is the inverse of the projection
    focal = (373 / 2) / math.tan(math.radians(55.0) / 2)
    from camera import perspective_project, ray_direction
    local = ray_direction(320.0, 260.0, 480, 373, 55.0)
    screen = perspective_project(local, 480, 373, 55.0)
    assert screen is not None
    assert abs(screen[0] - 320.0) < 1e-6 and abs(screen[1] - 260.0) < 1e-6
    print(f"  spawn rule 2: click at (320, 260) hit the ground at"
          f" ({hit[0]:.2f}, 0.00, {hit[2]:.2f}), a far click rejected")


def test_colour_values_match_the_source():
    assert particles.SPARK_RGB == (255, 180, 30)
    assert particles.FIREWORK_COLORS == ((40, 140, 255), (255, 80, 150),
                                         (170, 70, 255))
    # 30-150 px/s on a 50 px cell is 0.6-3.0 world units per second
    assert abs(particles.SPARK_SPEED[0] - 0.6) < 1e-9
    assert abs(particles.SPARK_SPEED[1] - 3.0) < 1e-9
    print("  colours and speeds match Start2Dto3D.py")


test_cap_is_bounded()
test_lifetime_removes_particles()
test_burst_does_not_share_state()
test_motion_is_three_dimensional()
test_floor_stops_the_spark()
test_wall_hides_particles()
test_spawn_rule_worm_position()
test_spawn_rule_mouse_ray()
test_colour_values_match_the_source()
print("particle tests passed")
