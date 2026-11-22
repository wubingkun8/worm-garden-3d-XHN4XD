"""Tests for the apples, the bobbing, the score and the level rule (handbook M10)."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maze import generate_maze
from physics import Player, GROUND_Y
from hazards import make_water_holes
from apples import (APPLE_TOTAL, GOLD_VALUE, SILVER_VALUE, COLLECT_RADIUS,
                    BOB_AMPLITUDE, BOB_FREQUENCY, make_apples, collect,
                    remaining, total_value, SILVER_RGB, GOLD_RGB)
from progress import Progress, format_clock

cols, rows, seed = 12, 12, 7
maze, start_cell, finish_cell = generate_maze(cols, rows, seed=seed)
holes = make_water_holes(cols, rows, start_cell, finish_cell, seed)
blocked = {hole.cell for hole in holes}
apples = make_apples(cols, rows, start_cell, finish_cell, blocked, seed)

# 1. four apples: three silver and one gold, exactly as REWARDS in the source
assert len(apples) == APPLE_TOTAL == 4, len(apples)
values = sorted(apple.value for apple in apples)
assert values == [SILVER_VALUE] * 3 + [GOLD_VALUE], values
assert total_value(apples) == 250, total_value(apples)

# 2. never on the start cell, the exit cell or a puddle
cells = {apple.cell for apple in apples}
assert start_cell not in cells and finish_cell not in cells
assert not (cells & blocked), "an apple sits in a puddle"
assert len(cells) == 4, "two apples share a cell"

# ...and never in the exit cell or next to it: walking the corridor into the
# locked flag must not be part of any apple route
for col, row in cells:
    assert abs(col - finish_cell[0]) + abs(row - finish_cell[1]) > 1, \
        f"apple at {(col, row)} touches the exit at {finish_cell}"

# 3. the bob is the handbook formula, and the collider follows it
apple = apples[0]
assert abs(apple.height(0.0) - (apple.base_y
            + BOB_AMPLITUDE * math.sin(apple.phase))) < 1e-12
quarter = (math.pi / 2.0 - apple.phase) / (2.0 * math.pi * BOB_FREQUENCY)
assert abs(apple.height(quarter) - (apple.base_y + BOB_AMPLITUDE)) < 1e-9
three_quarter = quarter + 0.5 / BOB_FREQUENCY
assert abs(apple.height(three_quarter) - (apple.base_y - BOB_AMPLITUDE)) < 1e-9
assert abs(apple.base_y - 0.28) < 1e-12, "handbook 12.1: height 0.28"

# 4. the collection radius is BALL_RADIUS + 12 px = 0.46, measured in 3D
player = Player(apple.x, apple.z)
player.y = apple.center(0.0)[1]
assert collect(apples, player, 0.0), "standing on an apple must collect it"
assert len(remaining(apples)) == 3

edge = Player(apple.x + COLLECT_RADIUS + 0.02, apple.z)
edge.y = apples[1].center(0.0)[1]
assert not collect([apples[1]], edge, 0.0), "collected from too far away"

# 5. a second pass collects nothing twice (score cannot double count)
assert collect(apples, player, 0.0) == []

# 6. the level only advances when all four are in, and the score persists
progress = Progress(now=0.0)
progress.collect([apples[0]])
assert progress.score == SILVER_VALUE and progress.collected == 1
assert not progress.level_ready() and progress.missing() == 3

for rest in remaining(apples):
    progress.collect([rest])
assert progress.level_ready(), "all four apples must unlock the exit"
assert progress.score == 250, progress.score

level_before, score_before = progress.level, progress.score
progress.advance()
assert progress.level == level_before + 1
assert progress.collected == 0, "apples must reset per level"
assert progress.score == score_before, "the score must survive a new level"
assert progress.levels_cleared == 1

# 7. R restarts the run: level 1, score 0, clock restarted
progress.reset(120.0)
assert progress.level == 1 and progress.score == 0 and progress.collected == 0
assert progress.elapsed(125.0) == 5.0, progress.elapsed(125.0)
assert format_clock(5.0) == "00:05"
assert format_clock(3599.0) == "59:59"

# 8. colours: the source draws the "silver" apples red, the gold one gold
assert apples[0].kind == "silver" and apples[0].color() == SILVER_RGB
assert apples[-1].kind == "gold" and apples[-1].color() == GOLD_RGB

# 9. apples must be reachable: their cell is not blocked by walls all around
for apple in apples:
    col, row = apple.cell
    open_sides = [name for name, (_, _) in
                  {"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}.items()
                  if not maze[row][col]["walls"][name]]
    assert open_sides, f"apple at {apple.cell} is walled in on every side"

# 10. apples sit at the cell centre, on the floor, ready for the height rule
for apple in apples:
    assert abs(apple.x - (apple.cell[0] + 0.5)) < 1e-12
    assert abs(apple.z - (apple.cell[1] + 0.5)) < 1e-12

print("apple tests passed")
