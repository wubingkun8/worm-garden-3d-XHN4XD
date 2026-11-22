import numpy as np

from transforms import translation, scale, transform_point

# Cube centred at the origin, side length 1 (handbook section 6.1)
CUBE_VERTICES = np.array([
    [-.5, -.5, -.5], [.5, -.5, -.5], [.5, .5, -.5], [-.5, .5, -.5],
    [-.5, -.5, .5], [.5, -.5, .5], [.5, .5, .5], [-.5, .5, .5],
], dtype=float)

CUBE_TRIANGLES = [
    (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
    (0, 1, 5), (0, 5, 4), (3, 6, 2), (3, 7, 6),
    (1, 2, 6), (1, 6, 5), (0, 4, 7), (0, 7, 3),
]

CUBE_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]

# World units, scaled from the original pixels by CELL = 50
WALL_H = 0.75          # 0.75 units high
WALL_T = 8 / 50         # 0.16 units thick


def wall_box(col, row, direction):
    """Centre and size of one wall prism in world space (handbook section 6.2)."""
    if direction == 'N':
        return ((col + .5, WALL_H / 2, row),
                (1.0, WALL_H, WALL_T))
    if direction == 'W':
        return ((col, WALL_H / 2, row + .5),
                (WALL_T, WALL_H, 1.0))
    if direction == 'E':
        return ((col + 1, WALL_H / 2, row + .5),
                (WALL_T, WALL_H, 1.0))
    if direction == 'S':
        return ((col + .5, WALL_H / 2, row + 1),
                (1.0, WALL_H, WALL_T))
    raise ValueError('Unknown direction')


def maze_wall_boxes(maze, cols, rows):
    """Every wall prism built from the original N/E/S/W flags.

    Same duplicate-avoiding strategy as draw_maze(): only north and west walls
    for all cells, plus east of the last column and south of the last row.
    """
    boxes = []
    for row in range(rows):
        for col in range(cols):
            walls = maze[row][col]["walls"]
            for side in ('N', 'W'):
                if walls[side]:
                    boxes.append(wall_box(col, row, side))
            if col == cols - 1 and walls['E']:
                boxes.append(wall_box(col, row, 'E'))
            if row == rows - 1 and walls['S']:
                boxes.append(wall_box(col, row, 'S'))
    return boxes


def box_world_vertices(center, size):
    """Model matrix -> 8 world-space corners of a box."""
    M = translation(*center) @ scale(*size)
    return np.array([transform_point(M, v) for v in CUBE_VERTICES])


def floor_box(cols, rows):
    """A thin slab whose top face sits exactly at Y = 0."""
    return ((cols / 2.0, -0.01, rows / 2.0),
            (float(cols), 0.02, float(rows)))


def scene_boxes(maze, cols, rows):
    """Floor first, then every maze wall."""
    boxes = [floor_box(cols, rows)]
    boxes.extend(maze_wall_boxes(maze, cols, rows))
    return boxes
