import math

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

# Six quads of a cube, counter-clockwise seen from outside (so normals point out)
BOX_FACES = [
    (1, 0, 3, 2),   # -Z
    (4, 5, 6, 7),   # +Z
    (1, 2, 6, 5),   # +X
    (0, 4, 7, 3),   # -X
    (3, 7, 6, 2),   # +Y (top)
    (0, 1, 5, 4),   # -Y (bottom)
]

TILES_PER_UNIT = 1.0   # one texture tile covers one world unit

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
    """A thin slab whose top face sits exactly at Y = 0.

    The slab stops at the inner face of the boundary walls. Extending it under
    the walls produced a one-pixel grass seam along the wall silhouette, because
    the slab edge and the wall face met at almost the same depth.
    """
    inset = WALL_T
    return ((cols / 2.0, -0.01, rows / 2.0),
            (float(cols) - 2 * inset, 0.02, float(rows) - 2 * inset))


def scene_boxes(maze, cols, rows):
    """Floor first, then every maze wall."""
    boxes = [floor_box(cols, rows)]
    boxes.extend(maze_wall_boxes(maze, cols, rows))
    return boxes


def face_uv(quad, normal, tiles=TILES_PER_UNIT):
    """UV corners for one quad, in world units, so the texture tiles once per unit.

    Vertical faces: u runs along the horizontal direction of the face, v runs
    downwards from its top edge, so bricks and grass stay upright. Horizontal
    faces use X and Z directly.
    """
    if abs(normal[1]) < 0.9:
        tangent = np.cross(np.array([0.0, 1.0, 0.0]), normal)
        length = np.linalg.norm(tangent)
        tangent = tangent / length if length > 1e-12 else np.array([1.0, 0.0, 0.0])
        u = quad @ tangent
        v = quad[:, 1].max() - quad[:, 1]
    else:
        u = quad[:, 0]
        v = quad[:, 2]
    return np.stack([u, v], axis=1) * tiles


def box_face_quads(center, size, tiles=TILES_PER_UNIT):
    """List of (4 world corners, 4 UV corners, outward normal) for one box."""
    verts = box_world_vertices(center, size)
    faces = []
    for indices in BOX_FACES:
        quad = verts[list(indices)]
        normal = np.cross(quad[1] - quad[0], quad[3] - quad[0])
        length = np.linalg.norm(normal)
        if length < 1e-12:
            continue
        normal = normal / length
        faces.append((quad, face_uv(quad, normal, tiles), normal))
    return faces


def scene_faces(boxes, tiles=TILES_PER_UNIT):
    """Flatten a box list into textured faces (index 0 keeps the floor separate)."""
    faces = []
    for box_index, (center, size) in enumerate(boxes):
        for quad, uv, normal in box_face_quads(center, size, tiles):
            faces.append((box_index, quad, uv, normal))
    return faces


def sphere_mesh(radius, center=(0.0, 0.0, 0.0), rings=8, segments=12):
    """Latitude/longitude sphere as triangles whose winding faces outwards.

    Returns a list of (a, b, c, normal) with a, b, c in world space. Low
    polygonal on purpose: flat shading makes the facets visible, which keeps the
    polygon count predictable for the software rasterizer.
    """
    cx, cy, cz = center
    rings = max(3, rings)
    segments = max(3, segments)

    grid = []
    for i in range(rings + 1):
        phi = math.pi * i / rings
        ring_y = math.cos(phi) * radius
        ring_r = math.sin(phi) * radius
        row = []
        for j in range(segments):
            theta = 2.0 * math.pi * j / segments
            row.append(np.array([cx + ring_r * math.cos(theta),
                                 cy + ring_y,
                                 cz + ring_r * math.sin(theta)], dtype=float))
        grid.append(row)

    triangles = []
    # top and bottom caps: the pole is one vertex, so no degenerate triangles
    for j in range(segments):
        triangles.append((grid[0][0],
                          grid[1][(j + 1) % segments],
                          grid[1][j]))
        triangles.append((grid[rings][0],
                          grid[rings - 1][j],
                          grid[rings - 1][(j + 1) % segments]))

    for i in range(1, rings - 1):
        for j in range(segments):
            a = grid[i][j]
            b = grid[i][(j + 1) % segments]
            c = grid[i + 1][(j + 1) % segments]
            d = grid[i + 1][j]
            triangles.append((a, b, c))
            triangles.append((a, c, d))

    out = []
    centre = np.array([cx, cy, cz], dtype=float)
    for a, b, c in triangles:
        normal = np.cross(b - a, c - a)
        length = np.linalg.norm(normal)
        if length < 1e-12:
            continue
        normal = normal / length
        if float(np.dot(normal, (a + b + c) / 3.0 - centre)) < 0.0:
            b, c = c, b
            normal = -normal
        out.append((a, b, c, normal))
    return out


def disc_mesh(radius, center=(0.0, 0.0, 0.0), segments=20):
    """Flat triangle fan on the X/Z plane, facing up (+Y).

    Used for the water holes: a disc a few millimetres above the floor reads as
    a puddle from the follow camera, and the fan keeps the triangle count low.
    """
    cx, cy, cz = center
    segments = max(3, segments)
    rim = []
    for j in range(segments):
        theta = 2.0 * math.pi * j / segments
        rim.append(np.array([cx + radius * math.cos(theta), cy,
                             cz + radius * math.sin(theta)], dtype=float))

    out = []
    up = np.array([0.0, 1.0, 0.0])
    for j in range(segments):
        a = np.array([cx, cy, cz], dtype=float)
        b = rim[j]
        c = rim[(j + 1) % segments]
        # wound so that cross(b-a, c-a) points up: rim goes clockwise seen from above
        out.append((a, c, b, up))
    return out
