"""M11 (handbook chapter 13): the 3D particle system for sparks and fireworks.

The 2D game kept particles in plain dictionaries with 2-component pos/vel and
drew them straight onto the Pygame surface. The dictionary survives unchanged
and only the arrays grow a third component, which is exactly the conversion
chapter 13.1 asks for:

* pos and vel are 3-component arrays,
* gravity and damping are applied per second, so motion does not depend on FPS,
* every visible particle is projected through the same camera as the maze,
* particles are drawn after the opaque geometry and their depth is compared with
  the Z-buffer, so a firework behind a wall cannot shine through it.

The source draws with `pygame.draw.circle(screen, color, pos, int(size*alpha))`
and kills a particle when `life <= 0`; both rules are kept, with `size`
interpreted as pixels on the reference 900 px frame and scaled by 1/depth, and
the whole pile capped at MAX_PARTICLES (the source capped at 250).
"""
import math

import numpy as np

MAX_PARTICLES = 900        # source: 250; a 3D burst holds more visible beads
MAX_BURST = 120            # source: 70-110 per firework
GRAVITY = -4.0             # handbook 13.1 example value, world units / s^2
DAMPING = 0.985            # per 1/60 s, source: 0.97 ** (dt * FPS)
GROUND_BOUNCE = 0.4        # 3D only: the floor is at y = 0, the source had none
FLOOR_Y = 0.02

SPARK_RGB = (255, 180, 30)                   # source create_spark() colour
FIREWORK_COLORS = ((40, 140, 255),           # source FIREWORK_COLORS
                   (255, 80, 150),
                   (170, 70, 255))

# The source measured speed in pixels per second on a 23 x 15 maze with
# CELL = 50 px, so 1 world unit per second is 50 px/s.
SOURCE_PX_PER_UNIT = 50.0
SPARK_SPEED = (30.0 / SOURCE_PX_PER_UNIT, 150.0 / SOURCE_PX_PER_UNIT)
FIREWORK_SPEED = (100.0 / SOURCE_PX_PER_UNIT, 350.0 / SOURCE_PX_PER_UNIT)

# Apparent size: the source `size` is a pixel radius on the 900 px reference
# frame, shrunk by 1/depth so far sparks are small (handbook 13.1).
REFERENCE_WIDTH = 900.0
REFERENCE_DEPTH = 3.0
PARTICLE_GAIN = 1.35


def make_particle(pos, vel, life, size, color):
    """The source particle dictionary, with 3-component pos and vel.

    Both arrays are copied on purpose: np.asarray() would return the caller's
    array for an array input, and a whole burst would then share one position
    and move as a single point.
    """
    return {"pos": np.array(pos, dtype=float),
            "vel": np.array(vel, dtype=float),
            "life": float(life),
            # the source used max_life = 1.0 for sparks, which made every spark
            # pop in at 40-100 % opacity; starting from full life fades better
            "max_life": float(life),
            "size": int(size),
            "color": tuple(int(c) for c in color)}


def add_particle(pile, particle):
    """Source add_particle(): oldest particles are dropped above the cap."""
    pile.append(particle)
    if len(pile) > MAX_PARTICLES:
        del pile[:len(pile) - MAX_PARTICLES]
    return particle


def create_spark(pile, pos, rng, count=1, speed=SPARK_SPEED):
    """Source create_spark(), converted to 3D.

    The 2D velocity is (cos a, sin a) * speed on the screen plane. In 3D the
    same circle lies on the XZ plane and a small positive Y component lifts the
    spark off the floor, which is the only new term.
    """
    for _ in range(count):
        angle = rng.uniform(0.0, 2.0 * math.pi)
        spd = rng.uniform(speed[0], speed[1])
        vel = np.array([math.cos(angle) * spd,
                        rng.uniform(0.4, 1.6),
                        math.sin(angle) * spd])
        # 0.4-1.0 s and 2-4 px, straight from the source
        add_particle(pile, make_particle(
            pos, vel, rng.uniform(0.4, 1.0), rng.integers(2, 5), SPARK_RGB))
    return pile


def create_firework(pile, pos, rng, count=None):
    """Source create_firework(), converted to a sphere instead of a 2D disc.

    A 2D firework spreads in a circle; a 3D one spreads over a sphere, so the
    burst looks the same from any camera position. Directions are sampled
    uniformly on the sphere (phi from a uniform cosine, not from a uniform
    angle, or the beads would bunch at the poles).
    """
    if count is None:
        count = int(rng.integers(70, MAX_BURST + 1))
    for _ in range(count):
        theta = rng.uniform(0.0, 2.0 * math.pi)
        phi = math.acos(float(rng.uniform(-1.0, 1.0)))
        spd = rng.uniform(FIREWORK_SPEED[0], FIREWORK_SPEED[1])
        direction = np.array([math.sin(phi) * math.cos(theta),
                              math.cos(phi),
                              math.sin(phi) * math.sin(theta)])
        # life 0.8-1.6 s and 2-5 px, straight from the source
        add_particle(pile, make_particle(
            pos, direction * spd, rng.uniform(0.8, 1.6), rng.integers(2, 6),
            FIREWORK_COLORS[int(rng.integers(0, len(FIREWORK_COLORS)))]))
    return pile


def update(pile, dt):
    """Source update_particles(), with 3D gravity, damping and a floor."""
    gravity = np.array([0.0, GRAVITY, 0.0])
    damp = DAMPING ** (dt * 60.0)
    alive = []
    for particle in pile:
        particle["vel"] += gravity * dt
        particle["vel"] *= damp
        particle["pos"] += particle["vel"] * dt
        particle["life"] -= dt
        if particle["pos"][1] < FLOOR_Y:
            # not in the source: a 3D spark that falls through the lawn looks
            # broken, so it bounces once and settles
            particle["pos"][1] = FLOOR_Y
            particle["vel"][1] = -particle["vel"][1] * GROUND_BOUNCE
        if particle["life"] > 0.0:
            alive.append(particle)
    pile[:] = alive
    return pile


def clear(pile):
    pile[:] = []
    return pile


def radius_px(particle, alpha, width):
    """Apparent radius: source size, shrunk by 1/depth (handbook 13.1)."""
    depth_scale = REFERENCE_DEPTH / max(1e-3, particle["_depth"])
    return max(1, int(round(particle["size"] * PARTICLE_GAIN * alpha
                            * (width / REFERENCE_WIDTH) * depth_scale)))


def draw(renderer, pile, matrix, eye, project, near=0.1, occl_tol=1e-6,
         depth_test=True):
    """Additive sprites, depth-tested against the Z-buffer (handbook 13.1).

    Returns (drawn, occluded) so the HUD and the tests can show that the
    occlusion test really ran: a particle is skipped when the nearest opaque
    surface in a 3x3 window around its pixel is closer to the camera than the
    particle itself.

    depth_test=False is the debugging switch used for the before/after
    screenshot: the same particles are then drawn straight through the walls.
    """
    colour = renderer.color
    zbuf = renderer.depth
    height, width = zbuf.shape
    drawn = occluded = 0

    for particle in pile:
        cam = (particle["pos"] - eye) @ matrix.T
        if cam[2] <= near:
            continue
        projected = project(cam)
        if projected is None:
            continue
        sx, sy, z = projected
        x, y = int(sx), int(sy)
        if x < 0 or y < 0 or x >= width or y >= height:
            continue
        # nearest opaque surface under the sprite: if that is closer than the
        # particle, a wall is in front of it
        front = zbuf[max(0, y - 1):min(height, y + 2),
                     max(0, x - 1):min(width, x + 2)].min()
        if depth_test and z > front + occl_tol:
            occluded += 1
            continue

        particle["_depth"] = z
        alpha = max(0.0, particle["life"] / particle["max_life"])
        r = radius_px(particle, alpha, width)
        tint = (np.asarray(particle["color"], dtype=np.float32)
                * (0.9 * alpha)).astype(np.int16)
        y0, y1 = max(0, y - r), min(height, y + r + 1)
        x0, x1 = max(0, x - r), min(width, x + r + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        window = colour[y0:y1, x0:x1]
        # additive blending: 255 + tint is clipped, not wrapped
        window[:] = np.minimum(window.astype(np.int16) + tint, 255).astype(
            np.uint8)
        drawn += 1
    return drawn, occluded


def total(pile):
    return len(pile)
