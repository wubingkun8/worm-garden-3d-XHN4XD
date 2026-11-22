"""Water holes and birds with explicit height-aware rules (handbook 11.4).

Both hazards come from the 2D game and keep its numbers, divided by CELL = 50:

    water hole radius  13 px -> 0.26 units, 9 holes per maze
    bird radius        10 px -> 0.20 units, 75 px/s -> 1.5 units/s
    bird count          min(2 + level, 8)

The rules are deliberately different from "distance on the floor":

    water  the worm drowns when its feet are below WATER_SAFE_CLEARANCE and its
           XZ centre is within the combined radius. Above that clearance it
           passes over the hole, which is the 3D form of the 2D code's
           "if jump_timer > 0: return False".
    birds  the birds hover at 0.75 units, above the worm's walking height, so a
           grounded worm walks underneath them. The test is a 3D sphere
           distance, so jumping into a bird hits it, and there is no way to fly
           over one with a 0.512 unit jump.
"""
import random

import numpy as np

from maze import cell_center_3d
from physics import WORM_RADIUS

WATER_HOLE_COUNT = 9
WATER_HOLE_RADIUS = 13 / 50.0     # 0.26 world units, the source value
WATER_DISC_RADIUS = 0.42          # drawn disc: fits between the 0.16 thick walls
WATER_Y = 0.006                   # a few millimetres above the floor plane
WATER_SAFE_CLEARANCE = 0.30       # feet above the floor needed to skim a hole

BIRD_RADIUS = 10 / 50.0           # 0.20 world units
BIRD_HEIGHT = 0.75                # centre height: above head, below wall top
BIRD_SPEED = 75 / 50.0            # 1.5 world units per second


def water_cells(cols, rows, start_cell, finish_cell, seed=7):
    """Puddle cells, start and finish excluded, like generate_water_holes().

    The 2D game always uses nine holes on a 23x15 maze. On the small test
    mazes nine holes would cover every free cell, so the count is capped at
    half of the available cells as well.
    """
    available = [(x, y) for y in range(rows) for x in range(cols)
                 if (x, y) != start_cell and (x, y) != finish_cell]
    rng = random.Random(seed + 101)
    rng.shuffle(available)
    count = min(WATER_HOLE_COUNT, len(available) // 2)
    return available[:count]


class WaterHole:
    """A puddle: a disc on the floor, plus the wave phase of the 2D original."""

    def __init__(self, cell, seed=7):
        self.cell = cell
        self.center = cell_center_3d(cell[0], cell[1])
        self.x = float(self.center[0])
        self.z = float(self.center[2])
        self.phase = random.Random(seed + cell[0] * 31 + cell[1] * 17).uniform(
            0.0, 2.0 * np.pi)

    def radius(self):
        """Radius of the drawn puddle: what the player actually sees."""
        return WATER_DISC_RADIUS


def make_water_holes(cols, rows, start_cell, finish_cell, seed=7):
    return [WaterHole(cell, seed) for cell
            in water_cells(cols, rows, start_cell, finish_cell, seed)]


def in_water(player, holes):
    """True when the worm drowns: feet low AND its centre inside the puddle.

    The test uses the radius of the disc that is actually drawn, so the water
    that kills the worm is exactly the water on screen (handbook 11.4 asks for
    visual and logical consistency). The source formula
    BALL_RADIUS + WATER_HOLE_RADIUS (11 + 13 px = 0.48 units) would drown the
    worm while it still looked dry, because the drawn ellipse is only 34 x 24 px.
    """
    if player.feet_y() >= WATER_SAFE_CLEARANCE:
        return False
    for hole in holes:
        dx, dz = player.x - hole.x, player.z - hole.z
        if dx * dx + dz * dz < WATER_DISC_RADIUS * WATER_DISC_RADIUS:
            return True
    return False


class Bird:
    """A patrolling bird, hovering at BIRD_HEIGHT above its cell centre."""

    def __init__(self, cell, target, rng):
        self.cell = cell
        self.target = target
        self.center = cell_center_3d(cell[0], cell[1], height=BIRD_HEIGHT)
        self.x = float(self.center[0])
        self.y = float(self.center[1])
        self.z = float(self.center[2])
        self.radius = BIRD_RADIUS
        self.speed = BIRD_SPEED + rng.uniform(-0.2, 0.4)

    def position(self):
        return np.array([self.x, self.y, self.z], dtype=float)


def make_birds(cols, rows, start_cell, finish_cell, blocked_cells,
               seed=7, level=1):
    """Birds on free cells, min(2 + level, 8) of them, as in generate_enemies()."""
    available = [(x, y) for y in range(rows) for x in range(cols)
                 if (x, y) != start_cell
                 and (x, y) != finish_cell
                 and (x, y) not in blocked_cells]
    rng = random.Random(seed + 202)
    rng.shuffle(available)
    count = min(2 + level, 8, len(available))
    return [Bird(cell, cell, rng) for cell in available[:count]]


def bird_hits(player, bird):
    """3D sphere distance: walking under is safe, jumping into one is not."""
    dx = player.x - bird.x
    dy = player.y - bird.y
    dz = player.z - bird.z
    reach = player.radius + bird.radius
    return dx * dx + dy * dy + dz * dz < reach * reach
