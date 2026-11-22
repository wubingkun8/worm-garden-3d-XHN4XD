import pygame
import numpy as np
import random
import math
import time

# =========================================================
# INITIALIZATION
# =========================================================

pygame.init()

WIDTH, HEIGHT = 1250, 900
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Worm's Garden Adventure")

clock = pygame.time.Clock()
FPS = 60

# Colors
BLACK = (10, 10, 16)
WHITE = (255, 255, 255)
RED = (255, 0, 0)

GRASS_BASE = (35, 105, 48)
GRASS_DARK = (29, 88, 40)
GRASS_LIGHT = (49, 125, 57)
GRASS_BLADE = (75, 145, 72)

STONE = (112, 116, 119)
STONE_DARK = (55, 59, 62)
STONE_LIGHT = (175, 178, 178)

SILVER = (195, 205, 215)
SILVER_LIGHT = (240, 245, 255)
GOLD = (255, 195, 30)
GOLD_LIGHT = (255, 235, 120)

START_GREEN = (40, 220, 75)
CHECKER_BLACK = (15, 15, 15)
CHECKER_WHITE = (245, 245, 245)

WATER_DARK = (5, 35, 75)
WATER_BLUE = (10, 90, 155)
WATER_LIGHT = (55, 175, 220)
WATER_HIGHLIGHT = (150, 225, 255)

FIREWORK_COLORS = [
    (40, 140, 255),
    (255, 80, 150),
    (170, 70, 255)
]

# Worm colors
WORM_GREEN = (75, 195, 65)
WORM_DARK = (25, 90, 25)
WORM_LIGHT = (125, 225, 95)

# Bird colors
BIRD_BODY = (15, 15, 22)
BIRD_WING = (35, 35, 48)
BIRD_BEAK = (255, 175, 30)

# =========================================================
# GAME SETTINGS
# =========================================================

BALL_RADIUS = 11
BALL_DIAMETER = BALL_RADIUS * 2
BALL_SPEED = 230

# Jump settings
JUMP_DURATION = 0.55
JUMP_COOLDOWN = 0.65
JUMP_HEIGHT = 30

jump_timer = 0
jump_cooldown = 0

# Water settings
WATER_HOLE_COUNT = 9
WATER_HOLE_RADIUS = 13
water_holes = []
water_hole_cells = set()

ENEMY_RADIUS = 10
ENEMY_SPEED = 75

COLS = 23
ROWS = 15

CELL = 50
WALL_THICKNESS = 8

MIN_CORRIDOR_WIDTH = 30
CORRIDOR_WIDTH = CELL - WALL_THICKNESS

if CORRIDOR_WIDTH < MIN_CORRIDOR_WIDTH:
    raise ValueError("Maze corridors are too narrow!")

if CORRIDOR_WIDTH < BALL_DIAMETER:
    raise ValueError("The worm is too large for the corridors!")

ORIGIN_X = 40
ORIGIN_Y = 95

MAZE_WIDTH = COLS * CELL
MAZE_HEIGHT = ROWS * CELL

START_CELL = (0, ROWS // 2)
FINISH_CELL = (COLS - 1, ROWS // 2)

START_POSITION = np.array([
    ORIGIN_X + CELL / 2,
    ORIGIN_Y + START_CELL[1] * CELL + CELL / 2
], dtype=float)

position = START_POSITION.copy()

# Rewards
SILVER_VALUE = 50
GOLD_VALUE = 100
TOTAL_MEDALS = 4

# Particles
MAX_PARTICLES = 250
SPARK_INTERVAL = 0.015
spark_timer = 0

particles = []

# Progress
score = 0
collected_medals = 0
level = 1
game_won = False

start_time = time.time()
finish_time = 0

# Fonts
font = pygame.font.SysFont("Arial", 22)
small_font = pygame.font.SysFont("Arial", 17)
title_font = pygame.font.SysFont("Arial", 30, bold=True)

# =========================================================
# GRASS BACKGROUND
# =========================================================

random.seed(12)

grass_details = []

for _ in range(1400):
    grass_details.append((
        random.randint(0, WIDTH - 1),
        random.randint(0, HEIGHT - 1),
        random.choice([
            GRASS_DARK,
            GRASS_LIGHT,
            GRASS_BLADE
        ]),
        random.randint(3, 7),
        random.randint(-2, 2)
    ))

random.seed()


def draw_grass_background():

    screen.fill(GRASS_BASE)

    for gx, gy, color, length, lean in grass_details:
        pygame.draw.line(
            screen,
            color,
            (gx, gy),
            (gx + lean, gy - length),
            1
        )


# =========================================================
# PARTICLE SYSTEM
# =========================================================

def add_particle(particle):

    particles.append(particle)

    if len(particles) > MAX_PARTICLES:
        del particles[:len(particles) - MAX_PARTICLES]


def create_spark(pos):

    angle = random.uniform(0, 2 * math.pi)
    speed = random.uniform(30, 150)

    velocity = np.array([
        math.cos(angle) * speed,
        math.sin(angle) * speed
    ])

    add_particle({
        "pos": pos.copy(),
        "vel": velocity,
        "life": random.uniform(0.4, 1.0),
        "max_life": 1.0,
        "size": random.randint(2, 4),
        "color": (255, 180, 30)
    })


def create_firework(pos):

    for _ in range(random.randint(70, 110)):

        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(100, 350)

        velocity = np.array([
            math.cos(angle) * speed,
            math.sin(angle) * speed
        ])

        life = random.uniform(0.8, 1.6)

        add_particle({
            "pos": np.array(pos, dtype=float),
            "vel": velocity,
            "life": life,
            "max_life": life,
            "size": random.randint(2, 5),
            "color": random.choice(FIREWORK_COLORS)
        })


def update_particles(dt):

    alive_particles = []

    for particle in particles:

        particle["pos"] += particle["vel"] * dt
        particle["vel"] *= 0.97 ** (dt * FPS)
        particle["life"] -= dt

        if particle["life"] > 0:
            alive_particles.append(particle)

    particles[:] = alive_particles


def draw_particles():

    for particle in particles:

        alpha = max(
            0,
            particle["life"] / particle["max_life"]
        )

        color = tuple(
            int(c * alpha)
            for c in particle["color"]
        )

        pygame.draw.circle(
            screen,
            color,
            particle["pos"].astype(int),
            max(1, int(particle["size"] * alpha))
        )


# =========================================================
# MAZE GENERATION
# =========================================================

DIRECTIONS = {
    "N": (0, -1, "S"),
    "E": (1, 0, "W"),
    "S": (0, 1, "N"),
    "W": (-1, 0, "E")
}

maze = []


def generate_maze():

    global maze

    maze = [
        [
            {
                "walls": {
                    "N": True,
                    "E": True,
                    "S": True,
                    "W": True
                },
                "visited": False
            }
            for _ in range(COLS)
        ]
        for _ in range(ROWS)
    ]

    stack = [START_CELL]

    sx, sy = START_CELL
    maze[sy][sx]["visited"] = True

    while stack:

        x, y = stack[-1]
        neighbors = []

        for direction, (dx, dy, opposite) in DIRECTIONS.items():

            nx = x + dx
            ny = y + dy

            if (
                0 <= nx < COLS
                and 0 <= ny < ROWS
                and not maze[ny][nx]["visited"]
            ):
                neighbors.append(
                    (nx, ny, direction, opposite)
                )

        if neighbors:

            nx, ny, direction, opposite = random.choice(neighbors)

            maze[y][x]["walls"][direction] = False
            maze[ny][nx]["walls"][opposite] = False

            maze[ny][nx]["visited"] = True

            stack.append((nx, ny))

        else:
            stack.pop()

    maze[START_CELL[1]][START_CELL[0]]["walls"]["W"] = False
    maze[FINISH_CELL[1]][FINISH_CELL[0]]["walls"]["E"] = False


def cell_center(cell):

    x, y = cell

    return np.array([
        ORIGIN_X + x * CELL + CELL / 2,
        ORIGIN_Y + y * CELL + CELL / 2
    ], dtype=float)


# =========================================================
# WATER HOLES
# =========================================================

def generate_water_holes():

    global water_holes, water_hole_cells

    water_holes = []
    water_hole_cells = set()

    available_cells = [
        (x, y)
        for y in range(ROWS)
        for x in range(COLS)
        if (x, y) != START_CELL
        and (x, y) != FINISH_CELL
    ]

    random.shuffle(available_cells)

    selected_cells = available_cells[:WATER_HOLE_COUNT]

    for cell in selected_cells:

        water_hole_cells.add(cell)

        water_holes.append({
            "pos": cell_center(cell),
            "phase": random.uniform(0, 2 * math.pi)
        })


def draw_water_holes():

    current_time = pygame.time.get_ticks() / 1000

    for hole in water_holes:

        x, y = hole["pos"].astype(int)

        ripple = math.sin(
            current_time * 3 + hole["phase"]
        ) * 2

        pygame.draw.ellipse(
            screen,
            (5, 25, 45),
            (x - 17, y - 12, 34, 24)
        )

        pygame.draw.ellipse(
            screen,
            WATER_DARK,
            (x - 15, y - 10, 30, 20)
        )

        pygame.draw.ellipse(
            screen,
            WATER_BLUE,
            (x - 12, y - 8, 24, 16)
        )

        pygame.draw.ellipse(
            screen,
            WATER_LIGHT,
            (
                int(x - 9 - ripple / 2),
                int(y - 5 - ripple / 2),
                int(18 + ripple),
                int(10 + ripple)
            ),
            1
        )

        pygame.draw.line(
            screen,
            WATER_HIGHLIGHT,
            (x - 6, y - 3),
            (x - 1, y - 4),
            2
        )


def check_water_collision():

    if jump_timer > 0:
        return False

    for hole in water_holes:

        distance = np.linalg.norm(
            position - hole["pos"]
        )

        if distance < BALL_RADIUS + WATER_HOLE_RADIUS:

            reset_ball()
            return True

    return False


# =========================================================
# MAZE WALL COLLISION
# =========================================================

def get_wall_rectangles():

    walls = []

    half = WALL_THICKNESS // 2
    thickness = WALL_THICKNESS

    for y in range(ROWS):
        for x in range(COLS):

            px = ORIGIN_X + x * CELL
            py = ORIGIN_Y + y * CELL

            cell = maze[y][x]["walls"]

            if cell["N"]:

                walls.append(
                    pygame.Rect(
                        px - half,
                        py - half,
                        CELL + thickness,
                        thickness
                    )
                )

            if cell["W"]:

                walls.append(
                    pygame.Rect(
                        px - half,
                        py - half,
                        thickness,
                        CELL + thickness
                    )
                )

            if cell["E"] and x == COLS - 1:

                walls.append(
                    pygame.Rect(
                        px + CELL - half,
                        py - half,
                        thickness,
                        CELL + thickness
                    )
                )

            if cell["S"] and y == ROWS - 1:

                walls.append(
                    pygame.Rect(
                        px - half,
                        py + CELL - half,
                        CELL + thickness,
                        thickness
                    )
                )

    return walls


wall_rectangles = []


def circle_hits_wall(pos, radius):

    for wall in wall_rectangles:

        closest_x = max(
            wall.left,
            min(pos[0], wall.right)
        )

        closest_y = max(
            wall.top,
            min(pos[1], wall.bottom)
        )

        dx = pos[0] - closest_x
        dy = pos[1] - closest_y

        if dx * dx + dy * dy < radius * radius:
            return True

    return False


def move_player(direction, dt):

    global position

    movement = direction * BALL_SPEED * dt

    max_step = max(2, WALL_THICKNESS / 2)

    steps = max(
        1,
        math.ceil(
            max(abs(movement[0]), abs(movement[1])) / max_step
        )
    )

    step_x = movement[0] / steps
    step_y = movement[1] / steps

    for _ in range(steps):

        new_x = position.copy()
        new_x[0] += step_x

        if not circle_hits_wall(new_x, BALL_RADIUS):
            position[0] = new_x[0]

        new_y = position.copy()
        new_y[1] += step_y

        if not circle_hits_wall(new_y, BALL_RADIUS):
            position[1] = new_y[1]


# =========================================================
# DRAW MAZE
# =========================================================

def draw_maze():

    pygame.draw.rect(
        screen,
        (43, 96, 49),
        (
            ORIGIN_X,
            ORIGIN_Y,
            MAZE_WIDTH,
            MAZE_HEIGHT
        )
    )

    for y in range(ROWS):
        for x in range(COLS):

            px = ORIGIN_X + x * CELL
            py = ORIGIN_Y + y * CELL

            cell = maze[y][x]["walls"]

            pygame.draw.circle(
                screen,
                (48, 105, 53),
                (px + 12, py + 12),
                2
            )

            for direction in ["N", "W"]:

                if cell[direction]:

                    if direction == "N":
                        start = (px, py)
                        end = (px + CELL, py)
                    else:
                        start = (px, py)
                        end = (px, py + CELL)

                    pygame.draw.line(
                        screen,
                        STONE_DARK,
                        start,
                        end,
                        WALL_THICKNESS
                    )

                    pygame.draw.line(
                        screen,
                        STONE,
                        start,
                        end,
                        WALL_THICKNESS - 2
                    )

                    pygame.draw.line(
                        screen,
                        STONE_LIGHT,
                        start,
                        end,
                        2
                    )

            if x == COLS - 1 and cell["E"]:

                start = (px + CELL, py)
                end = (px + CELL, py + CELL)

                pygame.draw.line(
                    screen,
                    STONE_DARK,
                    start,
                    end,
                    WALL_THICKNESS
                )

                pygame.draw.line(
                    screen,
                    STONE,
                    start,
                    end,
                    WALL_THICKNESS - 2
                )

            if y == ROWS - 1 and cell["S"]:

                start = (px, py + CELL)
                end = (px + CELL, py + CELL)

                pygame.draw.line(
                    screen,
                    STONE_DARK,
                    start,
                    end,
                    WALL_THICKNESS
                )

                pygame.draw.line(
                    screen,
                    STONE,
                    start,
                    end,
                    WALL_THICKNESS - 2
                )


# =========================================================
# START AND FINISH FLAGS
# =========================================================

def draw_flag(x, y, color, label):

    pole_top = y - 28
    pole_bottom = y + 28

    # Pole
    pygame.draw.line(
        screen,
        (90, 65, 35),
        (x, pole_top),
        (x, pole_bottom),
        4
    )

    # Pole top
    pygame.draw.circle(
        screen,
        GOLD,
        (x, pole_top),
        4
    )

    # Flag
    wave = math.sin(pygame.time.get_ticks() / 180) * 3

    flag_points = [
        (x + 2, pole_top + 3),
        (x + 22, pole_top + 9 + wave),
        (x + 2, pole_top + 18)
    ]

    pygame.draw.polygon(
        screen,
        color,
        flag_points
    )

    pygame.draw.polygon(
        screen,
        WHITE,
        flag_points,
        1
    )

    # Label
    label_text = small_font.render(
        label,
        True,
        color
    )

    screen.blit(
        label_text,
        (x - label_text.get_width() // 2, pole_bottom + 3)
    )



# =========================================================
# START AND FINISH FLAGS
# =========================================================

def draw_start_flag(x, y):

    pole_top = y - 28
    pole_bottom = y + 28

    # Wooden pole
    pygame.draw.line(
        screen,
        (90, 65, 35),
        (x, pole_top),
        (x, pole_bottom),
        4
    )

    # Gold pole top
    pygame.draw.circle(
        screen,
        GOLD,
        (x, pole_top),
        4
    )

    # Animated green flag
    wave = math.sin(pygame.time.get_ticks() / 180) * 3

    flag_points = [
        (x + 2, pole_top + 3),
        (x + 22, pole_top + 9 + wave),
        (x + 2, pole_top + 18)
    ]

    pygame.draw.polygon(
        screen,
        START_GREEN,
        flag_points
    )

    pygame.draw.polygon(
        screen,
        WHITE,
        flag_points,
        1
    )

    # START label
    label = small_font.render(
        "START",
        True,
        START_GREEN
    )

    screen.blit(
        label,
        (x - label.get_width() // 2, pole_bottom + 3)
    )


def draw_finish_flag(x, y):

    pole_top = y - 28
    pole_bottom = y + 28

    # Pole
    pygame.draw.line(
        screen,
        (90, 65, 35),
        (x, pole_top),
        (x, pole_bottom),
        4
    )

    pygame.draw.circle(
        screen,
        GOLD,
        (x, pole_top),
        4
    )

    # Checkered flag dimensions
    flag_width = 28
    flag_height = 24
    square_size = 6

    # Animated waving effect
    wave = math.sin(pygame.time.get_ticks() / 180) * 2

    for row in range(4):
        for col in range(5):

            color = (
                CHECKER_WHITE
                if (row + col) % 2 == 0
                else CHECKER_BLACK
            )

            offset = int(
                math.sin(
                    pygame.time.get_ticks() / 180 + col * 0.5
                ) * 2
            )

            rect = pygame.Rect(
                x + 2 + col * square_size,
                pole_top + 3 + row * square_size + offset,
                square_size + 1,
                square_size + 1
            )

            pygame.draw.rect(
                screen,
                color,
                rect
            )

    # Subtle white border
    pygame.draw.rect(
        screen,
        WHITE,
        (
            x + 2,
            pole_top + 3,
            flag_width,
            flag_height
        ),
        1
    )

    # EXIT label
    label = small_font.render(
        "FINISH",
        True,
        CHECKER_WHITE
    )

    screen.blit(
        label,
        (x - label.get_width() // 2, pole_bottom + 3)
    )


def draw_start_and_finish():

    start_x = ORIGIN_X + 5
    start_y = ORIGIN_Y + START_CELL[1] * CELL + CELL // 2

    finish_x = ORIGIN_X + COLS * CELL - 5
    finish_y = ORIGIN_Y + FINISH_CELL[1] * CELL + CELL // 2

    draw_start_flag(
        start_x,
        start_y
    )

    draw_finish_flag(
        finish_x,
        finish_y
    )



# =========================================================
# PLAYER WORM
# =========================================================

def draw_ball():

    if jump_timer > 0:

        jump_progress = 1 - (jump_timer / JUMP_DURATION)

        jump_height = math.sin(
            jump_progress * math.pi
        ) * JUMP_HEIGHT

    else:
        jump_height = 0

    x = int(position[0])
    y = int(position[1] - jump_height)

    # Ground shadow
    shadow_size = max(
        4,
        int(13 - jump_height * 0.12)
    )

    pygame.draw.ellipse(
        screen,
        (15, 45, 20),
        (
            x - shadow_size,
            int(position[1]) + 5,
            shadow_size * 2,
            max(3, shadow_size // 2)
        )
    )

    # Worm body segments
    body_segments = [
        (-15, 4),
        (-10, 2),
        (-5, 1),
        (0, 0)
    ]

    segment_colors = [
        (35, 125, 35),
        (45, 155, 45),
        (55, 175, 50),
        WORM_GREEN
    ]

    for i, (dx, dy) in enumerate(body_segments):

        radius = 7 if i < 3 else 10

        pygame.draw.circle(
            screen,
            segment_colors[i],
            (x + dx, y + dy),
            radius
        )

        pygame.draw.circle(
            screen,
            WORM_DARK,
            (x + dx, y + dy),
            radius,
            1
        )

    # Worm head
    pygame.draw.circle(
        screen,
        WORM_GREEN,
        (x + 7, y - 1),
        10
    )

    # Eyes
    pygame.draw.circle(screen, WHITE, (x + 10, y - 5), 3)
    pygame.draw.circle(screen, BLACK, (x + 11, y - 5), 1)

    pygame.draw.circle(screen, WHITE, (x + 4, y - 5), 3)
    pygame.draw.circle(screen, BLACK, (x + 5, y - 5), 1)

    # Smile
    pygame.draw.arc(
        screen,
        WORM_DARK,
        (x + 4, y - 1, 9, 7),
        0,
        math.pi,
        1
    )


def reset_ball():

    global spark_timer, jump_timer, jump_cooldown

    position[:] = START_POSITION
    spark_timer = 0

    jump_timer = 0
    jump_cooldown = 0


def start_jump():

    global jump_timer, jump_cooldown

    if jump_timer <= 0 and jump_cooldown <= 0:

        jump_timer = JUMP_DURATION
        jump_cooldown = JUMP_COOLDOWN


# =========================================================
# BIRDS
# =========================================================

enemies = []


def get_available_neighbors(cell):

    x, y = cell
    neighbors = []

    for direction, (dx, dy, opposite) in DIRECTIONS.items():

        if not maze[y][x]["walls"][direction]:

            nx = x + dx
            ny = y + dy

            if 0 <= nx < COLS and 0 <= ny < ROWS:
                neighbors.append((nx, ny))

    return neighbors


def generate_enemies():

    global enemies

    enemies = []

    # More birds with every level, maximum 8
    enemy_count = min(2 + level, 8)

    available_cells = [
        (x, y)
        for y in range(ROWS)
        for x in range(COLS)
        if (x, y) != START_CELL
        and (x, y) != FINISH_CELL
        and (x, y) not in water_hole_cells
    ]

    random.shuffle(available_cells)

    for cell in available_cells[:enemy_count]:

        neighbors = get_available_neighbors(cell)

        if not neighbors:
            continue

        enemies.append({
            "cell": cell,
            "target": random.choice(neighbors),
            "previous": cell,
            "pos": cell_center(cell),
            "speed": ENEMY_SPEED + random.uniform(-10, 20)
        })


def update_enemies(dt):

    for enemy in enemies:

        current = enemy["pos"]
        target = cell_center(enemy["target"])

        direction = target - current
        distance = np.linalg.norm(direction)

        if distance <= 2:

            enemy["pos"] = target.copy()
            enemy["previous"] = enemy["cell"]
            enemy["cell"] = enemy["target"]

            neighbors = get_available_neighbors(enemy["cell"])

            if neighbors:

                alternatives = [
                    n for n in neighbors
                    if n != enemy["previous"]
                ]

                choices = alternatives if alternatives else neighbors

                enemy["target"] = random.choice(choices)

        else:

            direction /= distance

            movement = enemy["speed"] * dt

            if movement >= distance:
                enemy["pos"] = target.copy()
            else:
                enemy["pos"] += direction * movement


def draw_enemies():

    current_time = pygame.time.get_ticks() / 1000

    for enemy in enemies:

        x, y = enemy["pos"].astype(int)

        # Animated wings
        wing_offset = int(
            math.sin(current_time * 9 + x) * 3
        )

        # Bird body
        pygame.draw.ellipse(
            screen,
            BIRD_BODY,
            (x - 10, y - 6, 20, 14)
        )

        # Left wing
        pygame.draw.ellipse(
            screen,
            BIRD_WING,
            (x - 13, y - 10 - wing_offset, 13, 9)
        )

        # Right wing
        pygame.draw.ellipse(
            screen,
            BIRD_WING,
            (x, y - 10 + wing_offset, 13, 9)
        )

        # Head
        pygame.draw.circle(
            screen,
            BIRD_BODY,
            (x + 8, y - 5),
            7
        )

        # Beak
        pygame.draw.polygon(
            screen,
            BIRD_BEAK,
            [
                (x + 13, y - 5),
                (x + 21, y - 2),
                (x + 13, y)
            ]
        )

        # Eye
        pygame.draw.circle(
            screen,
            WHITE,
            (x + 10, y - 7),
            2
        )

        pygame.draw.circle(
            screen,
            BLACK,
            (x + 11, y - 7),
            1
        )

        # Tail feathers
        pygame.draw.polygon(
            screen,
            BIRD_WING,
            [
                (x - 9, y),
                (x - 17, y - 5),
                (x - 14, y + 3)
            ]
        )


def check_enemy_collision():

    for enemy in enemies:

        distance = np.linalg.norm(
            position - enemy["pos"]
        )

        if distance < BALL_RADIUS + ENEMY_RADIUS:

            reset_ball()
            return True

    return False


# =========================================================
# APPLES
# =========================================================

medals = []


def generate_medals():

    global medals

    medals = []

    available_cells = []

    for y in range(ROWS):
        for x in range(COLS):

            cell = (x, y)

            if cell in [START_CELL, FINISH_CELL]:
                continue

            if cell in water_hole_cells:
                continue

            available_cells.append(cell)

    random.shuffle(available_cells)

    rewards = [
        ("silver", SILVER_VALUE),
        ("silver", SILVER_VALUE),
        ("silver", SILVER_VALUE),
        ("gold", GOLD_VALUE)
    ]

    for cell, reward in zip(available_cells, rewards):

        medal_type, value = reward

        medals.append({
            "pos": cell_center(cell),
            "type": medal_type,
            "value": value,
            "collected": False
        })


def update_medals():

    global score, collected_medals

    for medal in medals:

        if medal["collected"]:
            continue

        distance = np.linalg.norm(
            position - medal["pos"]
        )

        if distance < BALL_RADIUS + 12:

            medal["collected"] = True

            score += medal["value"]
            collected_medals += 1


def draw_medals():

    current_time = pygame.time.get_ticks() / 1000

    for medal in medals:

        if medal["collected"]:
            continue

        x, y = medal["pos"].astype(int)

        bob = int(
            math.sin(current_time * 3 + x) * 2
        )

        y += bob

        if medal["type"] == "silver":

            apple_color = (220, 35, 40)
            apple_light = (255, 100, 100)

        else:

            apple_color = (255, 190, 20)
            apple_light = (255, 240, 100)

        # Apple shadow
        pygame.draw.ellipse(
            screen,
            (25, 60, 30),
            (x - 12, y + 9, 25, 7)
        )

        # Apple body
        pygame.draw.circle(
            screen,
            apple_color,
            (x - 5, y),
            9
        )

        pygame.draw.circle(
            screen,
            apple_color,
            (x + 5, y),
            9
        )

        pygame.draw.ellipse(
            screen,
            apple_color,
            (x - 11, y - 4, 22, 17)
        )

        # Apple highlight
        pygame.draw.circle(
            screen,
            apple_light,
            (x - 5, y - 5),
            3
        )

        # Stem
        pygame.draw.line(
            screen,
            (90, 55, 25),
            (x, y - 9),
            (x + 1, y - 15),
            3
        )

        # Leaf
        pygame.draw.ellipse(
            screen,
            (50, 160, 55),
            (x + 1, y - 16, 9, 5)
        )


# =========================================================
# LEVEL SYSTEM
# =========================================================

def generate_level():

    global collected_medals, spark_timer
    global wall_rectangles, start_time
    global particles

    reset_ball()

    collected_medals = 0
    spark_timer = 0

    # Clear leftover particles from the previous level
    particles.clear()

    # Generate an entirely new maze
    generate_maze()

    wall_rectangles = get_wall_rectangles()

    # Randomize all level objects
    generate_water_holes()
    generate_medals()
    generate_enemies()

    # Start a fresh timer for this level
    start_time = time.time()


def complete_level():

    global level

    level += 1

    generate_level()


# =========================================================
# FINISH
# =========================================================

def check_finish():

    if np.linalg.norm(
        position - cell_center(FINISH_CELL)
    ) < 17:

        if collected_medals == TOTAL_MEDALS:

            # Automatically advance to the next level
            complete_level()

        else:

            # Not enough apples yet
            reset_ball()


# =========================================================
# USER INTERFACE
# =========================================================

def draw_ui():

    pygame.draw.rect(
        screen,
        (20, 65, 32),
        (0, 0, WIDTH, 58)
    )

    pygame.draw.line(
        screen,
        (80, 150, 75),
        (0, 58),
        (WIDTH, 58),
        2
    )

    level_text = font.render(
        f"LEVEL: {level}",
        True,
        WHITE
    )

    screen.blit(level_text, (15, 15))

    score_text = font.render(
        f"SCORE: {score}",
        True,
        GOLD
    )

    screen.blit(score_text, (190, 15))

    apple_text = font.render(
        f"APPLES: {collected_medals}/{TOTAL_MEDALS}",
        True,
        (255, 130, 130)
    )

    screen.blit(apple_text, (390, 15))

    elapsed = time.time() - start_time

    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    timer_text = font.render(
        f"TIME: {minutes:02}:{seconds:02}",
        True,
        WHITE
    )

    screen.blit(timer_text, (650, 15))

    instructions = small_font.render(
        "WASD / ARROWS: MOVE     SPACE: JUMP     CLICK: FIREWORKS",
        True,
        WHITE
    )

    screen.blit(
        instructions,
        (
            WIDTH // 2 - instructions.get_width() // 2,
            HEIGHT - 28
        )
    )

    if collected_medals < TOTAL_MEDALS:

        message = small_font.render(
            "Collect all 4 apples to unlock the red exit flag!",
            True,
            GOLD
        )

    else:

        message = small_font.render(
            "EXIT UNLOCKED! Reach the red flag to advance!",
            True,
            (100, 255, 130)
        )

    screen.blit(
        message,
        (
            WIDTH // 2 - message.get_width() // 2,
            65
        )
    )


# =========================================================
# START GAME
# =========================================================

generate_level()


# =========================================================
# MAIN GAME LOOP
# =========================================================

running = True

while running:

    dt = min(clock.tick(FPS) / 1000, 0.05)

    # -----------------------------------------------------
    # EVENTS
    # -----------------------------------------------------

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.MOUSEBUTTONDOWN:

            if event.button == 1:
                create_firework(event.pos)

        if event.type == pygame.KEYDOWN:

            if event.key == pygame.K_SPACE:
                start_jump()

            # R can restart the current run at any time
            if event.key == pygame.K_r:

                score = 0
                level = 1

                generate_level()

    # -----------------------------------------------------
    # GAME UPDATE
    # -----------------------------------------------------

    # Update jump timers
    if jump_timer > 0:
        jump_timer = max(0, jump_timer - dt)

    if jump_cooldown > 0:
        jump_cooldown = max(0, jump_cooldown - dt)

    keys = pygame.key.get_pressed()

    direction = np.array([0.0, 0.0])

    if keys[pygame.K_w] or keys[pygame.K_UP]:
        direction[1] -= 1

    if keys[pygame.K_s] or keys[pygame.K_DOWN]:
        direction[1] += 1

    if keys[pygame.K_a] or keys[pygame.K_LEFT]:
        direction[0] -= 1

    if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
        direction[0] += 1

    moving = np.linalg.norm(direction) > 0

    if moving:

        direction /= np.linalg.norm(direction)

        move_player(direction, dt)

        spark_timer += dt

        while spark_timer >= SPARK_INTERVAL:

            create_spark(position)
            spark_timer -= SPARK_INTERVAL

    else:
        spark_timer = 0

    # Check water before checking birds
    check_water_collision()

    update_enemies(dt)

    check_enemy_collision()

    update_medals()

    check_finish()

    update_particles(dt)

    # -----------------------------------------------------
    # RENDER
    # -----------------------------------------------------

    draw_grass_background()

    draw_maze()

    draw_water_holes()

    draw_start_and_finish()

    draw_medals()

    draw_enemies()

    draw_particles()

    draw_ball()

    draw_ui()

    pygame.display.flip()


pygame.quit()