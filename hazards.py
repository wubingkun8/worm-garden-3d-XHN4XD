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
import math
import random

import numpy as np

from maze import cell_center_3d, DIRECTIONS
from physics import WORM_RADIUS

WATER_HOLE_COUNT = 9
WATER_HOLE_RADIUS = 13 / 50.0     # 0.26 world units, the source value
WATER_DISC_RADIUS = 0.34          # drawn puddle: small enough to leave a route
WATER_Y = 0.006                   # a few millimetres above the floor plane
WATER_SAFE_CLEARANCE = 0.30       # feet above the floor needed to skim a hole

BIRD_RADIUS = 10 / 50.0           # 0.20 world units
BIRD_HEIGHT = 0.75                # centre height: above head, below wall top
BIRD_SPEED = 75 / 50.0            # 1.5 world units per second
ARRIVE_EPS = 2 / 50.0             # 2 px in the source: snap to the cell centre


def water_cells(cols, rows, start_cell, finish_cell, seed=7):
    """Puddle cells, start and finish excluded, like generate_water_holes().

    The 2D game always uses nine holes on a 23x15 maze, which is 2.6 % of the
    cells. A fixed nine on a 5x5 test maze would be 36 % and would wall the
    player in, so the count follows the area instead: one hole per nine cells,
    at least two, at most nine.
    """
    available = [(x, y) for y in range(rows) for x in range(cols)
                 if (x, y) != start_cell and (x, y) != finish_cell]
    rng = random.Random(seed + 101)
    rng.shuffle(available)
    count = min(WATER_HOLE_COUNT, max(2, round(cols * rows / 9)))
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


def nearest_water(player, holes):
    """Centre distance to the closest puddle, or None when there is none."""
    best = None
    for hole in holes:
        d = math.hypot(player.x - hole.x, player.z - hole.z)
        if best is None or d < best:
            best = d
    return best


def open_neighbours(maze, cell):
    """Cells reachable from this cell through a missing wall (get_available_neighbors)."""
    col, row = cell
    rows, cols = len(maze), len(maze[0])
    out = []
    for name, (dx, dz, _) in DIRECTIONS.items():
        if maze[row][col]["walls"][name]:
            continue
        nc, nr = col + dx, row + dz
        if 0 <= nc < cols and 0 <= nr < rows:
            out.append((nc, nr))
    return out


def update_birds(birds, maze, dt):
    for bird in birds:
        bird.update(dt, maze)


class Bird:
    """A bird that patrols open corridors, hovering at BIRD_HEIGHT (M10).

    The 2D original walks the bird from cell centre to cell centre and only ever
    steps through a side without a wall (get_available_neighbors()), avoiding the
    cell it came from. That rule is kept here, so the bird in 3D never crosses a
    closed wall either.
    """

    def __init__(self, cell, target, rng):
        self.cell = cell
        self.target = target or cell
        self.previous = cell
        self.center = cell_center_3d(cell[0], cell[1], height=BIRD_HEIGHT)
        self.x = float(self.center[0])
        self.y = float(self.center[1])
        self.z = float(self.center[2])
        self.radius = BIRD_RADIUS
        self.speed = BIRD_SPEED + rng.uniform(-0.2, 0.4)
        self.phase = rng.uniform(0.0, 2.0 * np.pi)
        self.rng = rng

    def position(self):
        return np.array([self.x, self.y, self.z], dtype=float)

    def update(self, dt, maze):
        """Walk towards the target cell centre, then pick a new open neighbour."""
        goal = cell_center_3d(self.target[0], self.target[1], height=BIRD_HEIGHT)
        delta = goal - self.position()
        distance = float(np.linalg.norm(delta))
        step = self.speed * dt
        if distance <= max(step, ARRIVE_EPS):
            self.x, self.z = float(goal[0]), float(goal[2])
            self.previous, self.cell = self.cell, self.target
            self._choose_next(maze)
            return
        self.x += float(delta[0] / distance * step)
        self.z += float(delta[2] / distance * step)

    def _choose_next(self, maze):
        """Next open neighbour, never straight back where the bird came from."""
        neighbours = open_neighbours(maze, self.cell)
        if not neighbours:
            self.target = self.cell
            return
        alternatives = [cell for cell in neighbours if cell != self.previous]
        self.target = self.rng.choice(alternatives or neighbours)


def make_birds(cols, rows, start_cell, finish_cell, blocked_cells, maze,
               seed=7, level=1):
    """Birds on free cells, min(2 + level, 8) of them, as in generate_enemies().

    The count is also capped by the size of the maze (one bird per eight free
    cells, never less than one): the source rule assumes a 23 x 15 board, and
    three birds on a 3 x 3 board would leave a nine cell maze unplayable.
    """
    available = [(x, y) for y in range(rows) for x in range(cols)
                 if (x, y) != start_cell
                 and (x, y) != finish_cell
                 and (x, y) not in blocked_cells]
    rng = random.Random(seed + 202)
    rng.shuffle(available)
    by_area = max(1, round(cols * rows / 8))
    count = min(2 + level, 8, by_area, len(available))
    birds = []
    for cell in available[:count]:
        neighbours = open_neighbours(maze, cell)
        if not neighbours:
            continue
        birds.append(Bird(cell, rng.choice(neighbours), rng))
    return birds


def bird_hits(player, bird):
    """3D sphere distance: walking under is safe, jumping into one is not."""
    dx = player.x - bird.x
    dy = player.y - bird.y
    dz = player.z - bird.z
    reach = player.radius + bird.radius
    return dx * dx + dy * dy + dz * dz < reach * reach
