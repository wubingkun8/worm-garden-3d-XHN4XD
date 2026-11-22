"""Apples (the source code calls them medals) and their collection rule (M10).

Everything here comes from the 2D game, divided by CELL = 50:

    four apples per level          3 x SILVER_VALUE (50) + 1 x GOLD_VALUE (100)
    placement                      free cells only: no start, no finish, no water
    collection radius              BALL_RADIUS + 12 px = 23 px = 0.46 units
    bobbing                        y(t) = y0 + 0.05 sin(2 pi 1.5 t + phase)

The bob is part of the collider, not only of the picture: the apple that the
player sees at the top of its arc is the apple that gets collected, which is what
handbook 11.4 asks for ("make bobbing motion visually and logically consistent").
"""
import math
import random

import numpy as np

from maze import cell_center_3d

APPLE_TOTAL = 4
SILVER_VALUE = 50
GOLD_VALUE = 100
REWARDS = (("silver", SILVER_VALUE),
           ("silver", SILVER_VALUE),
           ("silver", SILVER_VALUE),
           ("gold", GOLD_VALUE))

APPLE_RADIUS = 0.18          # the source draws two lobes of radius 9 px = 0.18
APPLE_HOVER = 0.28           # handbook 12.1: cell_center_3d(cell, height=0.28)
COLLECT_RADIUS = 0.46        # BALL_RADIUS 0.22 + 12 px pick radius / 50
BOB_AMPLITUDE = 0.05         # handbook 12.2
BOB_FREQUENCY = 1.5          # handbook 12.2: 1.5 Hz

SILVER_RGB = (220, 35, 40)   # the source draws "silver" apples red
GOLD_RGB = (255, 190, 20)
STEM_RGB = (74, 44, 30)
LEAF_RGB = (66, 140, 60)


class Apple:
    """One apple: a cell, a value, and a bobbing height above the floor."""

    def __init__(self, cell, kind, value, phase=0.0):
        self.cell = cell
        self.kind = kind
        self.value = value
        self.collected = False
        centre = cell_center_3d(cell[0], cell[1])
        self.x = float(centre[0])
        self.z = float(centre[2])
        self.phase = phase
        self.base_y = APPLE_HOVER

    def height(self, now):
        """Current centre height, the same formula the picture uses."""
        return (self.base_y
                + BOB_AMPLITUDE * math.sin(2.0 * math.pi * BOB_FREQUENCY * now
                                           + self.phase))

    def center(self, now=0.0):
        return np.array([self.x, self.height(now), self.z], dtype=float)

    def color(self):
        return SILVER_RGB if self.kind == "silver" else GOLD_RGB


def apple_cells(cols, rows, start_cell, finish_cell, blocked_cells, seed=7):
    """Four free cells, shuffled, water and the start/finish excluded.

    The exit cell and its four neighbours are also excluded: the flag sits in
    the middle of a corridor, and an apple right next to it would force the
    player to walk over the (locked) exit to collect it.
    """
    near_exit = {(finish_cell[0], finish_cell[1]),
                 (finish_cell[0] + 1, finish_cell[1]),
                 (finish_cell[0] - 1, finish_cell[1]),
                 (finish_cell[0], finish_cell[1] + 1),
                 (finish_cell[0], finish_cell[1] - 1)}
    available = [(x, y) for y in range(rows) for x in range(cols)
                 if (x, y) != start_cell
                 and (x, y) not in near_exit
                 and (x, y) not in blocked_cells]
    rng = random.Random(seed + 303)
    rng.shuffle(available)
    return available[:APPLE_TOTAL]


def make_apples(cols, rows, start_cell, finish_cell, blocked_cells, seed=7):
    cells = apple_cells(cols, rows, start_cell, finish_cell, blocked_cells, seed)
    rng = random.Random(seed + 404)
    return [Apple(cell, kind, value, rng.uniform(0.0, 2.0 * math.pi))
            for cell, (kind, value) in zip(cells, REWARDS)]


def collect(apples, player, now=0.0):
    """Mark and return the apples whose 3D distance is inside the collect radius."""
    taken = []
    for apple in apples:
        if apple.collected:
            continue
        delta = player.position() - apple.center(now)
        if float(np.linalg.norm(delta)) < COLLECT_RADIUS:
            apple.collected = True
            taken.append(apple)
    return taken


def remaining(apples):
    return [apple for apple in apples if not apple.collected]


def total_value(apples):
    return sum(apple.value for apple in apples)
