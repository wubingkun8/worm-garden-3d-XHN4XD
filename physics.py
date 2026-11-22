"""Worm movement and horizontal collision (handbook chapter 11).

Movement is world-aligned on the X/Z plane: W is north (-Z), S is south (+Z).
The 2D game used 230 px/s and an 11 px radius, so after dividing by CELL = 50
the worm moves 4.6 units/s and has radius 0.22.
"""
import math

import numpy as np

from geometry import WALL_H

WORM_RADIUS = 11 / 50.0     # 0.22 world units
WORM_SPEED = 230 / 50.0     # 4.6 world units per second
GROUND_Y = WORM_RADIUS      # sphere centre when resting on the floor
MAX_STEP = 0.05             # longest single test step, so nothing tunnels

# Jump, handbook 11.3: v0 = 3.2, g = 10, flight ~0.64 s, rise ~0.512 units
GRAVITY = 10.0
JUMP_SPEED = 3.2
JUMP_COOLDOWN = 0.65        # the 2D source used the same cooldown
VERTICAL_STEP = 0.02        # integrate the arc in small steps, long frames included


def movement_direction(keys, pygame_module):
    """Normalised X/Z direction from WASD; diagonal input is not faster."""
    dx = float(keys[pygame_module.K_d]) - float(keys[pygame_module.K_a])
    dz = float(keys[pygame_module.K_s]) - float(keys[pygame_module.K_w])
    direction = np.array([dx, 0.0, dz], dtype=float)
    length = np.linalg.norm(direction)
    return direction / length if length else direction


def circle_hits_xz(px, pz, radius, box):
    """Circle against a rectangular XZ footprint (handbook 11.2).

    box = (min_x, max_x, min_z, max_z)
    """
    x0, x1, z0, z1 = box
    nearest_x = max(x0, min(px, x1))
    nearest_z = max(z0, min(pz, z1))
    dx, dz = px - nearest_x, pz - nearest_z
    return dx * dx + dz * dz < radius * radius


def wall_aabbs(boxes, skip_floors=True):
    """XZ footprints of the wall boxes, matching the 3D mesh exactly.

    Boxes come straight from geometry.scene_boxes, so the 0.16 unit wall
    thickness is identical in the renderer and in the collision test.
    """
    out = []
    for index, (center, size) in enumerate(boxes):
        if skip_floors and index == 0:
            continue
        cx, _, cz = center
        sx, _, sz = size
        out.append((cx - sx / 2.0, cx + sx / 2.0, cz - sz / 2.0, cz + sz / 2.0))
    return out


class Player:
    """The worm: X/Z position, ground height, radius and speed."""

    def __init__(self, x, z, radius=WORM_RADIUS, speed=WORM_SPEED,
                 ground_y=GROUND_Y):
        self.x = float(x)
        self.z = float(z)
        self.y = float(ground_y)
        self.radius = float(radius)
        self.speed = float(speed)
        self.ground_y = float(ground_y)
        self.vertical_speed = 0.0      # world units per second, +Y is up
        self.jump_cooldown = 0.0
        self.blocked = False

    def position(self):
        return np.array([self.x, self.y, self.z], dtype=float)

    def feet_y(self):
        """Height of the bottom of the sphere: the value hazard rules use."""
        return self.y - self.radius

    def airborne(self):
        return self.y > self.ground_y + 1e-9 or self.vertical_speed > 0.0

    def jump(self):
        """Start a jump if grounded and the cooldown has expired."""
        if self.airborne() or self.jump_cooldown > 0.0:
            return False
        self.vertical_speed = JUMP_SPEED
        self.jump_cooldown = JUMP_COOLDOWN
        return True

    def update_vertical(self, dt):
        """Euler integration of the jump arc, clamped to the ground plane.

        The arc is stepped in VERTICAL_STEP slices so a long frame cannot make
        the worm jump straight through the floor or overshoot its apex.
        """
        self.jump_cooldown = max(0.0, self.jump_cooldown - dt)
        if not self.airborne() and self.vertical_speed == 0.0:
            self.y = self.ground_y
            return

        slices = max(1, int(math.ceil(dt / VERTICAL_STEP)))
        step = dt / slices
        for _ in range(slices):
            self.vertical_speed -= GRAVITY * step
            self.y += self.vertical_speed * step
            if self.y <= self.ground_y:
                self.y = self.ground_y
                self.vertical_speed = 0.0
                break

    def collides(self, x, z, aabbs, y=None):
        """Horizontal footprint plus a height test against the wall body.

        A jump only clears a wall if the worm's feet rise above WALL_H; with
        v0 = 3.2 the feet peak at 0.512 and the walls are 0.75 high, so in
        practice the worm is always blocked. The rule is still explicit here.
        """
        feet = self.feet_y() if y is None else float(y) - self.radius
        if feet >= WALL_H:
            return False
        return any(circle_hits_xz(x, z, self.radius, box) for box in aabbs)

    def move(self, direction, dt, aabbs):
        """Advance by speed*dt, testing X and Z separately so the worm slides."""
        dx = float(direction[0]) * self.speed * dt
        dz = float(direction[2]) * self.speed * dt
        distance = math.hypot(dx, dz)
        steps = max(1, int(math.ceil(distance / MAX_STEP)))
        step_x, step_z = dx / steps, dz / steps

        hit_wall = False
        for _ in range(steps):
            if step_x:
                if self.collides(self.x + step_x, self.z, aabbs):
                    hit_wall = True
                else:
                    self.x += step_x
            if step_z:
                if self.collides(self.x, self.z + step_z, aabbs):
                    hit_wall = True
                else:
                    self.z += step_z

        self.blocked = hit_wall
        return self.position()
