"""Procedural textures for M7, generated with Pillow.

Every texture is deterministic (fixed seed), so the same PNG is produced on
every machine. The first run writes assets/<name>.png; later runs load it.

Handbook 10.1 uses 128 x 128, 900 grass blades, seed 12 — kept identical here.
"""
import os
import random

import numpy as np
from PIL import Image, ImageDraw

SIZE = 128
ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def make_grass(seed=12, size=SIZE):
    """Green noise: 900 short strokes on three green tones (handbook 10.1)."""
    rng = random.Random(seed)
    img = Image.new("RGB", (size, size), (38, 115, 52))
    draw = ImageDraw.Draw(img)
    for _ in range(900):
        x, y = rng.randrange(size), rng.randrange(size)
        tone = rng.choice([(30, 90, 42), (54, 130, 63), (72, 148, 78)])
        draw.line((x, y, x + 1, y - 3), fill=tone)
    return np.asarray(img)


def make_stone(seed=21, size=SIZE):
    """Grey brick wall: offset rows of bricks plus per-pixel grain."""
    rng = random.Random(seed)
    img = Image.new("RGB", (size, size), (96, 96, 92))
    draw = ImageDraw.Draw(img)
    brick_h, brick_w = 16, 32
    for row, y in enumerate(range(0, size, brick_h)):
        offset = 0 if row % 2 == 0 else brick_w // 2
        for x in range(-brick_w, size, brick_w):
            x0 = x + offset
            grey = rng.randrange(138, 176)
            draw.rectangle(
                (x0 + 1, y + 1, x0 + brick_w - 2, y + brick_h - 2),
                fill=(grey, grey, max(0, grey - 6))
            )
    for _ in range(2000):
        x, y = rng.randrange(size), rng.randrange(size)
        delta = rng.randrange(-14, 15)
        px = img.getpixel((x, y))
        img.putpixel((x, y), tuple(max(0, min(255, c + delta)) for c in px))
    return np.asarray(img)


def make_checker(seed=0, size=SIZE, squares=8):
    """Black/white checkerboard: makes perspective UV distortion obvious."""
    Image.init()
    img = Image.new("RGB", (size, size), (235, 235, 230))
    draw = ImageDraw.Draw(img)
    step = size // squares
    for row in range(squares):
        for col in range(squares):
            if (row + col) % 2 == 0:
                draw.rectangle((col * step, row * step,
                                (col + 1) * step - 1, (row + 1) * step - 1),
                               fill=(28, 28, 30))
    return np.asarray(img)


FACTORIES = {
    "grass": make_grass,
    "stone": make_stone,
    "checker": make_checker,
}

_CACHE = {}


def texture_path(name):
    return os.path.join(ASSET_DIR, name + ".png")


def load_texture(name, factory):
    """Load assets/<name>.png, or generate and save it on first use."""
    path = texture_path(name)
    if os.path.exists(path):
        return np.asarray(Image.open(path).convert("RGB"))
    os.makedirs(ASSET_DIR, exist_ok=True)
    tex = factory()
    Image.fromarray(tex).save(path)
    print("generated:", path)
    return tex


def get(name):
    """Cached texture lookup by name (grass / stone / checker)."""
    if name not in _CACHE:
        _CACHE[name] = load_texture(name, FACTORIES[name])
    return _CACHE[name]


def nearest_sample(tex, u, v):
    """Nearest-neighbour lookup with u,v repeated outside [0,1] (handbook 10.2)."""
    tex_h, tex_w = tex.shape[:2]
    tx = np.clip((np.mod(u, 1.0) * tex_w).astype(np.int32), 0, tex_w - 1)
    ty = np.clip((np.mod(v, 1.0) * tex_h).astype(np.int32), 0, tex_h - 1)
    return tex[ty, tx]


def bilinear_sample(tex, u, v):
    """Bilinear filtering: blends four neighbouring texels (Part B candidate)."""
    tex_h, tex_w = tex.shape[:2]
    fx = np.mod(u, 1.0) * tex_w - 0.5
    fy = np.mod(v, 1.0) * tex_h - 0.5
    x0 = np.floor(fx).astype(np.int32)
    y0 = np.floor(fy).astype(np.int32)
    ax = (fx - x0)[..., None]
    ay = (fy - y0)[..., None]
    x0m, x1m = x0 % tex_w, (x0 + 1) % tex_w
    y0m, y1m = y0 % tex_h, (y0 + 1) % tex_h
    c00 = tex[y0m, x0m].astype(np.float32)
    c10 = tex[y0m, x1m].astype(np.float32)
    c01 = tex[y1m, x0m].astype(np.float32)
    c11 = tex[y1m, x1m].astype(np.float32)
    top = c00 * (1 - ax) + c10 * ax
    bottom = c01 * (1 - ax) + c11 * ax
    return top * (1 - ay) + bottom * ay


if __name__ == "__main__":
    for name in FACTORIES:
        tex = get(name)
        print(f"{name}: {tex.shape[1]}x{tex.shape[0]}  mean RGB {tex.mean(axis=(0, 1)).round(1)}")
