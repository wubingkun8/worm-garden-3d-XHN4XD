import random
import numpy as np

# Original 2D game constants
ORIGIN_X, ORIGIN_Y, CELL = 40, 95, 50

# Start small for renderer debugging (handbook: 3x3 -> 10x10 -> 23x15)
COLS_DEFAULT, ROWS_DEFAULT = 3, 3

DIRECTIONS = {
    "N": (0, -1, "S"),
    "E": (1, 0, "W"),
    "S": (0, 1, "N"),
    "W": (-1, 0, "E"),
}


def cell_center_3d(col, row, height=0.0):
    """Maze cell (col, row) -> world-space center. X = east, Y = up, Z = south."""
    return np.array([col + 0.5, height, row + 0.5], dtype=float)


def old_screen_to_world(pos2d, height=0.22):
    """Convert an OLD 2D screen pixel position into a world position."""
    old_x, old_y = pos2d
    return np.array([(old_x - ORIGIN_X) / CELL,
                     height,
                     (old_y - ORIGIN_Y) / CELL], dtype=float)


def generate_maze(cols=COLS_DEFAULT, rows=ROWS_DEFAULT, seed=None):
    """Same depth-first backtracking algorithm as the original 2D game.

    Returns (maze, start_cell, finish_cell). Maze is indexed as maze[row][col]
    and each cell keeps the original 'walls' flags N/E/S/W plus 'visited'.
    """
    if seed is not None:
        random.seed(seed)

    maze = [
        [
            {
                "walls": {"N": True, "E": True, "S": True, "W": True},
                "visited": False,
            }
            for _ in range(cols)
        ]
        for _ in range(rows)
    ]

    start_cell = (0, rows // 2)
    finish_cell = (cols - 1, rows // 2)

    stack = [start_cell]
    sx, sy = start_cell
    maze[sy][sx]["visited"] = True

    while stack:
        x, y = stack[-1]
        neighbors = []

        for direction, (dx, dy, opposite) in DIRECTIONS.items():
            nx = x + dx
            ny = y + dy
            if (0 <= nx < cols and 0 <= ny < rows
                    and not maze[ny][nx]["visited"]):
                neighbors.append((nx, ny, direction, opposite))

        if neighbors:
            nx, ny, direction, opposite = random.choice(neighbors)
            maze[y][x]["walls"][direction] = False
            maze[ny][nx]["walls"][opposite] = False
            maze[ny][nx]["visited"] = True
            stack.append((nx, ny))
        else:
            stack.pop()

    # entrance and exit openings (same as the original game)
    maze[start_cell[1]][start_cell[0]]["walls"]["W"] = False
    maze[finish_cell[1]][finish_cell[0]]["walls"]["E"] = False

    return maze, start_cell, finish_cell


if __name__ == "__main__":
    for col, row in [(2, 1), (0, 7), (22, 14)]:
        print(f"cell ({col},{row}) -> {cell_center_3d(col, row)}")
    print("start position:", old_screen_to_world((65, 470)))

    maze, start, finish = generate_maze(3, 3, seed=7)
    print(f"\n3x3 maze  start={start}  finish={finish}")
    for row in range(3):
        for col in range(3):
            print(f"  cell ({col},{row}) {maze[row][col]['walls']}")
