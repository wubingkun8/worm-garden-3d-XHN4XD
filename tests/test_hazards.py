"""Tests for the jump and the height-aware hazard rules (handbook 11.3, 11.4)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maze import generate_maze
from geometry import scene_boxes, maze_wall_boxes, WALL_H
from physics import (Player, GROUND_Y, WORM_RADIUS, JUMP_SPEED, GRAVITY,
                     wall_aabbs, circle_hits_xz)
from hazards import (make_water_holes, make_birds, in_water, bird_hits,
                     WATER_SAFE_CLEARANCE, WATER_DISC_RADIUS, BIRD_HEIGHT)


def jump_profile(dt=1.0 / 120.0, seconds=1.2):
    """Run one jump and return (peak feet height, flight time)."""
    player = Player(0.5, 0.5)
    assert player.jump()
    peak, flight, landed = 0.0, None, False
    t = 0.0
    while t < seconds:
        player.update_vertical(dt)
        t += dt
        peak = max(peak, player.feet_y())
        if not landed and player.feet_y() <= 1e-9 and player.vertical_speed <= 0.0:
            landed = True
            flight = t
    return peak, flight, player


maze, start_cell, finish_cell = generate_maze(5, 5, seed=7)
boxes = scene_boxes(maze, 5, 5)
aabbs = wall_aabbs(boxes)
holes = make_water_holes(5, 5, start_cell, finish_cell, seed=7)
birds = make_birds(5, 5, start_cell, finish_cell, {h.cell for h in holes},
                   seed=7, level=1)

# 1. the ideal arc of handbook 11.3: rise 0.512, flight 0.64
peak, flight, player = jump_profile()
assert abs(peak - 0.512) < 0.03, peak
assert flight is not None and abs(flight - 0.64) < 0.06, flight

# 2. the jump moves world Y, not a screen offset, and does not move X/Z
assert player.feet_y() == 0.0, player.feet_y()
assert abs(player.x - 0.5) < 1e-12 and abs(player.z - 0.5) < 1e-12
assert abs(player.y - GROUND_Y) < 1e-12, player.y

# 3. a jump cannot clear a 0.75 high wall: the feet peak at 0.512
test_player = Player(0.5, 0.5)
assert test_player.jump()
for _ in range(38):                       # ~0.32 s, the apex
    test_player.update_vertical(1.0 / 120.0)
wall_center, wall_size = maze_wall_boxes(maze, 5, 5)[0]
wx, _, wz = wall_center
assert test_player.feet_y() < WALL_H
assert test_player.collides(wx, wz, aabbs), "airborne worm passed through a wall"

# 4. grounded in a puddle drowns, but skimming above the clearance does not
assert holes, "expected at least one puddle"
hole = holes[0]
drowner = Player(hole.x, hole.z)
assert in_water(drowner, holes), "grounded worm standing in water survived"
drowner.y = GROUND_Y + WATER_SAFE_CLEARANCE + 0.04
assert not in_water(drowner, holes), "worm above the clearance still drowned"
# its feet, not its centre, decide: the same Y inside the hole is safe only if high
skimmer = Player(hole.x + WATER_DISC_RADIUS * 0.9, hole.z)
skimmer.y = GROUND_Y + WATER_SAFE_CLEARANCE - 0.02
assert in_water(skimmer, holes), "feet just below the clearance survived"

# 5. birds are a 3D test: walking under is safe, jumping into one is not
bird = birds[0]
walker = Player(bird.x, bird.z)
assert not bird_hits(walker, bird), "grounded worm hit a bird flying above it"
walker.y = GROUND_Y + 0.512
assert bird_hits(walker, bird), "jumping into a bird did not hit it"
assert BIRD_HEIGHT > WORM_RADIUS + 0.3

# 6. hazards keep the source placement rules
assert start_cell not in {h.cell for h in holes}
assert finish_cell not in {h.cell for h in holes}
assert start_cell not in {b.cell for b in birds}
assert finish_cell not in {b.cell for b in birds}
assert not ({b.cell for b in birds} & {h.cell for h in holes})

# 7. birds scale with the level, capped at eight, as in generate_enemies()
assert len(birds) == min(2 + 1, 8, 23)
assert len(make_birds(5, 5, start_cell, finish_cell,
                      {h.cell for h in holes}, seed=7, level=6)) <= 8

print("hazard tests passed")
