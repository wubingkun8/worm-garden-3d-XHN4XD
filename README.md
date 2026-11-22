# Worm's Garden Adventure 3D

**Student:** Wu Bingkun · **Neptun:** XHN4XD
**Course:** Computer Graphics Programming (GEIAL31L-B2a), University of Miskolc

A software 3D renderer built from scratch: no OpenGL, no GPU, every pixel is
computed on the CPU with numpy. The 2D game `Start2Dto3D.py` is the provided
baseline; the 3D modules below are my own implementation.

## Installation

```
pip install numpy pillow pygame-ce
```

Tested with Python 3.14.7, numpy 2.5.3, Pillow 12.3.0, pygame-ce 2.5.8
(SDL 2.32.10), on Windows 11.

## Run

| Command | What it shows |
|---|---|
| `python Start2Dto3D.py` | the original 2D game (baseline, provided) |
| `python first_cube.py` | rotating wireframe cube, `P` switches projection |
| `python first_maze.py` | wireframe 3D maze (`--cols N --rows M --ortho --shot FILE`) |
| `python solid_maze.py` | solid 3D maze with Z-buffer (`--cols --rows --ortho --reverse --shot`) |
| `python textured_maze.py` | textured and lit maze (`--cols --rows --ortho --flat --light N --checker --shot`) |
| `python walk_maze.py` | M11: walk with WASD, jump with SPACE, collect four apples, launch fireworks, unlock the exit, next level (12×12 default, `--cols --rows --res --flat --demo --shot --no-occlusion`) |
| `python walk_maze.py --demo` | scripted run: wall slide, jump apex, drown, jump a puddle, collect an apple, locked exit, level 2, a firework, a hidden firework, the Z-buffer view (writes screenshots 12–16 and 18–21 and 25–28, stays 5×5) |
| `python walk_maze.py --flat` | same scene without textures, about 25–30 % faster on slow machines |
| `python walk_maze.py --strict-finish` | the original `reset_ball()` rule: touching the locked exit sends you back to the start |
| `python walk_maze.py --no-occlusion` | debugging build: particles ignore the Z-buffer and shine through walls (before/after evidence) |
| `python walk_maze.py --res 420x330` | render below the play default when a machine is too slow |
| `python report_shots.py` | regenerates screenshots 03, 04 and 06–11 at 900×700 |
| `python first_maze.py --cols 10 --rows 10 --shot screenshots\05_maze_wireframe_10x10.png` | 10×10 scalability check (120 boxes) |
| `python tests\test_math.py` | coordinate and matrix tests |
| `python tests\test_renderer.py` | depth-buffer, clipping and draw-order tests |
| `python tests\test_textures.py` | texture sampling, materials and perspective UV tests |
| `python tests\test_physics.py` | movement, frame-rate independence and wall collision tests |
| `python tests\test_hazards.py` | jump arc, wall clearing, water and bird height rules |
| `python tests\test_apples.py` | apple placement, bobbing, score and level rules |
| `python tests\test_particles.py` | particle cap, lifetimes, 3D motion, wall occlusion, both spawn rules |

Full original maze layout:

```
python solid_maze.py --cols 23 --rows 15
python textured_maze.py --cols 23 --rows 15
python walk_maze.py --cols 23 --rows 15
```

## Controls

| Key | Action |
|---|---|
| `W A S D` | move the worm on the X/Z plane (W is north, -Z) |
| `SPACE` | jump (only when grounded and off cooldown) |
| `F` | launch a firework at the worm's 3D position (`walk_maze.py`) |
| mouse left button | launch a firework where the click ray meets the ground (`walk_maze.py`) |
| `T` | toggle textures on/off while playing (also the fast mode) |
| `F3` | show the Z-buffer instead of the colour buffer (`walk_maze.py`) |
| `R` | restart the run: level 1, score 0, new apples, no particles (`walk_maze.py`) |
| `P` | toggle between perspective and orthographic projection |
| `R` | toggle draw order near→far / far→near (`solid_maze.py` only) |
| `L` | cycle the light direction (textured viewer only) |
| `ESC` | quit |

Movement follows handbook chapter 11: world-aligned keys, normalised diagonal
input, 4.6 units/s and a 0.22 radius, with X and Z resolved separately so the
worm slides along walls instead of stopping dead.

`walk_maze.py` starts in a 12×12 maze, which is 2.4 times the corridor length of
the 5×5 practice maze in both directions; `--cols` and `--rows` change it, and
`--demo` deliberately stays on the 5×5 board so the scripted shots keep the
composition described below. The scene is drawn at an internal 480×373 and
scaled up to the 900×700 window; `--shot` and `--demo` render at the full
900×700 for the report images, and `--res WxH` overrides either.

## Jump and hazards

The jump is a real physics arc in world Y (handbook 11.3), not a screen offset:

```
v0 = 3.2 units/s     g = 10 units/s^2
flight = 2v0/g = 0.64 s      rise = v0^2/(2g) = 0.512 units
```

The arc is integrated with Euler steps of at most 0.02 s, so a long frame
cannot pass through the floor; on landing Y is clamped to 0.22 and the vertical
speed is cleared.

Collision decisions use height (handbook 11.4):

* **Walls** — blocking needs horizontal overlap *and* feet below the 0.75 wall
  top. The feet peak at 0.512, so a jump never clears a wall, and the rule is
  still explicit instead of being assumed.
* **Puddles** — nine holes on the original 23×15 maze is 2.6 % of the cells, so
  the count follows the area instead (one per nine cells, at least two, at most
  nine: three on a 5×5, nine on a 10×10). The worm drowns when its feet are
  below 0.30 units and its centre is inside the drawn disc (radius 0.34), which
  is the 3D form of the 2D code's `if jump_timer > 0: return False`. Above that
  clearance it skims over: jumping across a puddle keeps the feet at about 0.39
  while the centre crosses the disc.
* **Birds** — a 3D sphere test with summed radii 0.42. The birds hover at 0.75,
  so a grounded worm walks underneath them and a jump hits them; flying over
  one is impossible with a 0.512 unit jump. Their count is `min(2 + level, 8)`
  as in `generate_enemies()`, also capped by one bird per eight cells, which
  leaves three on a 5×5 and a single one on a 3×3.
* Drowning or hitting a bird resets the worm to the start cell, like
  `reset_ball()` in the original game. Standing next to a puddle shows a hint
  that SPACE jumps it.

`python walk_maze.py --demo` prints the height of the feet every quarter
second while it script-walks these three cases, so the rules can be checked
without playing.

Original 2D game: `W A S D` to move, collect the apples, avoid water holes and the bird.

## Apples, score and levels (M10)

Chapter 12 of the handbook turns the 2D medals into apples. The rules are taken
from the source code and scaled by `CELL = 50`:

* **Four apples per level.** Three are "silver" and worth 50, one is "gold" and
  worth 100, so a full level is 250 points (`REWARDS` in `apples.py`). The
  source draws the silver apples red, so a red apple is a 50 pointer and the
  yellow one is the 100 pointer.
* **Placement.** Every cell except the start cell, the exit cell and a puddle is
  collected and shuffled with a seeded `random.Random`, then the first four are
  taken — the same rule as `generate_medals()`.
* **The apples float.** Height is `y(t) = 0.28 + 0.05 sin(2 pi 1.5 t + phase)`
  (handbook 12.2). The drawn position and the collision body use the *same*
  `Apple.center(t)`, so the apple you see at the top of its arc is the apple you
  collect.
* **Collection** is a 3D distance test, `|worm - apple| < 0.46`, i.e. the 0.22
  body radius plus the 12 px (0.24) pick radius of the 2D game. Collected apples
  are removed from the frame, and their value is added to the score once.
* **The exit stays locked** until four apples are in. The flag is red and the
  exit cell is a normal corridor cell, so **walking over it costs nothing**: the
  HUD says how many apples are missing and the player keeps moving. Reaching it
  with four apples turns the flag green and starts the next level: new maze, new
  puddles, new birds, four new apples, while **score and clock carry over**.
  Handbook 12.3 keeps the original "sent back to the start" behaviour as an
  option; the friendly variant is the default here because the flag often sits
  in the middle of a corridor, and being bounced to the start while looking for
  apples is not a puzzle, just a nuisance. `--strict-finish` restores
  `reset_ball()` for a side-by-side comparison.
* **Apples avoid the exit and its four neighbours**, so no apple ever forces the
  player to walk into the flag to pick it up.
* **The birds patrol the corridors** like `update_enemies()`: a bird walks from
  cell centre to cell centre, only ever through a side without a wall, and
  never back into the cell it came from. At 1.5 units/s a bird therefore cannot
  pass through a wall.
* **`R` restarts the run**: level 1, score 0, clock back to 00:00.

`python tests\test_apples.py` checks the placement rules, the bob formula, the
0.46 collection radius, the 250 point total, the level rule and the persistence
of the score.

`python walk_maze.py --demo` script-walks all of it and writes the evidence
shots 18–21.

## Particles, fireworks and the HUD (M11)

Chapter 13 converts the 2D particle system. The source kept particles in a
dictionary with 2-component `pos`/`vel` and drew them with
`pygame.draw.circle`; the dictionary is unchanged and only the arrays grow a
third component (`particles.py`):

* **Motion.** `vel.y += -4.0 * dt` (the handbook's example gravity),
  `vel *= 0.985 ** (60 dt)` (the source's `0.97 ** (dt * FPS)` damping) and
  `pos += vel * dt`. Speeds keep the source numbers converted at 50 px = 1 unit:
  sparks 30–150 px/s → 0.6–3.0 units/s, firework beads 100–350 px/s → 2.0–7.0.
* **Lifetimes are bounded.** A spark lives 0.4–1.0 s, a firework bead 0.8–1.6 s;
  `update()` drops everything whose `life <= 0`, so nothing leaks.
* **The count is capped** at `MAX_PARTICLES = 900` (the source capped at 250);
  `add_particle()` deletes the oldest beads first, so a key held down cannot
  grow the pile. `MAX_BURST = 120` beads per firework.
* **A 2D circle becomes a 3D sphere.** The source spread beads on a circle; a
  3D burst samples directions uniformly on a sphere (`phi` from a uniform
  cosine, otherwise the beads bunch at the poles), so the burst looks the same
  from any camera position.
* **Every bead is projected through the maze camera**, and its apparent radius
  is the source pixel size scaled by `1/depth` (handbook 13.1).
* **Particles are depth-tested.** They are drawn after the opaque geometry;
  each bead compares its camera depth with the Z-buffer in a 3×3 window around
  its pixel and is skipped when a nearer surface is there, so a firework behind
  a wall cannot shine through it. `--no-occlusion` turns the test off on
  purpose, which is how the before/after pair 26/27 was produced.
* **Two documented spawn rules** (handbook 13.2):
  1. `F` launches a burst at the worm's own 3D position, `worm + (0, 0.42, 0)`. The
     required simplification.
  2. A left click defines a ray from the eye through that pixel
     (`camera.ray_direction`, the inverse of `perspective_project`); the ray meets
     the plane `y = 0` at `t = -eye.y / dir.y`, and the burst starts 0.45 units
     above that point. Clicks that point too far (`t > 40`) or outside the lawn
     are rejected with a HUD note instead of spawning a firework in the void.
* **HUD** (handbook 14.1) stays in screen space: level, score, apples, clock,
  position, feet height, slide state, texture mode, triangles, faces culled,
  frame ms, the real framebuffer size, the live particle count and the number of
  beads the depth test hid this frame.
* **The window is resizable** (handbook 14.2). A resize keeps the framebuffer's
  pixel budget and bends it to the new aspect ratio, rebuilds the renderer and
  the background gradient, and the HUD follows the new height.
* **`F3` shows the Z-buffer** (handbook 14.3): near surfaces bright, far ones
  dark, empty pixels black, stretched over the frame's own 2–98 % depth range.

Measured cost of the particles themselves, 450 beads alive, same camera, full
frame including the HUD and the upscale: **+2.1 ms at 420×330, +2.8 ms at
480×373, +3.2 ms at 560×440, +1.6 ms at 900×700** — about 5–10 % of the frame,
because each bead only touches a 3×3–7×7 window. The cap is what keeps that
number bounded.

`python tests\test_particles.py` checks the cap, the lifetimes, the 3D motion,
the floor bounce, the wall occlusion (a bead behind a wall changes no pixel,
a bead in front does), that a burst does not share one position array, and both
spawn rules including the rejected far click.

`python walk_maze.py --demo` writes 25 (burst at the worm), 26 (the same burst
hidden by a wall, `hidden 69` in the HUD), 27 (the same frame without the depth
test) and 28 (the Z-buffer view).

## World conventions

* One maze cell = **1 world unit**. The 2D cell was 50 px, so every pixel length
  and speed is divided by 50 (ball radius 11 px → 0.22, speed 230 px/s → 4.6).
* Axes: **X = east, Y = up, Z = south**.
* The 2D screen space had y growing downwards, so screen y becomes world **Z**,
  not world Y.
* The 2D player position included the origin offset `ORIGIN_X=40`, `ORIGIN_Y=95`;
  it is subtracted first, then divided by 50: pixel (65, 470) → world (0.5, 0.22, 7.5).
* Cell centres are at `(col + 0.5, 0, row + 0.5)`; walls are 0.75 high and 0.16 thick.
* Camera basis: `right = forward × world_up`, so looking north (-Z) puts east
  (+X) on the right of the screen, matching the 2D game and the W/A/S/D keys.

## Project layout

```
Start2Dto3D.py     provided 2D game (baseline)
transforms.py      4x4 matrices, point and direction transforms
maze.py            DFS maze generation with N/E/S/W wall flags, 2D->3D conversion
geometry.py        cube mesh, wall and floor boxes, world vertices, face UVs
camera.py          camera basis, perspective and orthographic projection
renderer.py        clipping, back-face culling, flat shading, rasterizer, Z-buffer,
                   perspective-correct texture mapping
textures.py        procedural grass / stone / checker textures (Pillow)
assets/            grass.png, stone.png, checker.png generated by textures.py
first_cube.py      M2/M4 mini-lab: rotating cube, P toggles projection
first_maze.py      M3/M4 viewer: wireframe maze
solid_maze.py      M5 viewer: solid maze, shading and depth
textured_maze.py   M7 viewer: textured and lit maze, T and L keys
physics.py         worm movement, jump, circle-vs-AABB collision, substep anti-tunnelling
hazards.py         puddles and birds with the height-aware rules of chapter 11.4
apples.py          four apples per level: values, placement, bobbing, collection
progress.py        level, score and clock, and the rule for unlocking the exit
particles.py       M11 3D sparks and fireworks, depth-tested against the Z-buffer
walk_maze.py       M11 viewer: walk, jump, hazards, apples, levels, particles, --demo log
report_shots.py    batch script that regenerates the report screenshots
tests/             numerical, renderer, texture, physics, hazard, apple and particle tests
screenshots/       evidence images, all 900x700 internal resolution
```

## Screenshots

| File | Content |
|---|---|
| `00_2d_original.png` | the provided 2D game running (baseline for the conversion) |
| `01_cube_perspective.png` | wireframe cube, perspective projection |
| `02_cube_ortho.png` | the same cube, orthographic projection |
| `03_maze_wireframe_3x3.png` | 3×3 maze wireframe, perspective |
| `04_maze_wireframe_3x3_ortho.png` | the same maze and camera, orthographic |
| `05_maze_wireframe_10x10.png` | 10×10 maze wireframe, 120 boxes (scalability check) |
| `06_maze_solid_3x3.png` | solid surfaces, lighting and Z-buffer, perspective |
| `07_maze_solid_3x3_ortho.png` | solid surfaces, same camera, orthographic |
| `08_textured_perspective.png` | stone walls and grass floor, perspective, light 1 |
| `09_textured_ortho.png` | the same textured scene and camera, orthographic |
| `10_textured_light_rotated.png` | same scene with the light moved to 2: walls change brightness |
| `11_textured_checker_floor.png` | checker floor: perspective-correct UV has no linear distortion |
| `12_worm_start.png` | M8: the worm in its start cell, follow camera, textured maze |
| `13_worm_against_wall.png` | M8: diagonal input, one axis blocked, the other slides along the wall |
| `14_worm_jump_apex.png` | M9: jump at its apex, feet 0.49 above the floor, HUD reports AIR |
| `15_worm_at_puddle.png` | M9: grounded worm walking into a puddle (drowns on contact) |
| `16_worm_jumped_puddle.png` | M9: the same puddle crossed with a jump, feet 0.39 while over the water |
| `17_walk_12x12.png` | the playable 12×12 board: 9 puddles, 3 birds, 167 walls |
| `18_apple_collect.png` | M10: an apple collected, HUD shows APPLES 1/4 and SCORE 50 |
| `19_exit_locked.png` | M10: standing on the locked exit with 1/4 apples, red flag, missing-apple hint, no reset |
| `20_exit_unlocked.png` | M10: 4/4 apples, SCORE 250, the exit flag turns green |
| `21_level_2.png` | M10: level 2 regenerated, score 250 kept, APPLES 0/4 |
| `23_flat_m10.png` | M10 with textures off (`T`): flat shading, the fast debug mode |
| `24_5x5_m10.png` | M10 gameplay scene: worm, apples, puddles and the jump hint on 5×5 |
| `25_fireworks_worm.png` | M11: a 3D firework launched from the worm's position, 107 beads drawn, 21 already hidden by a wall |
| `26_particles_occluded.png` | M11: a burst behind a wall, HUD reports `hidden 69`, nothing shines through |
| `27_particles_through_wall.png` | M11: the same frame with `--no-occlusion`, the beads paint over the wall |
| `28_depth_buffer.png` | M11: `F3` Z-buffer view, near surfaces bright, empty pixels black |

All of them are reproducible: fixed seed 7, fixed camera angle 0.55, internal
render resolution 900×700 (stated in the on-screen HUD of every image).

## Tests

134 assertions in total (7 files):

```
python tests\test_math.py        ->  all tests passed
python tests\test_renderer.py    ->  renderer tests passed
python tests\test_textures.py    ->  texture tests passed
python tests\test_physics.py     ->  physics tests passed
python tests\test_hazards.py     ->  hazard tests passed
python tests\test_apples.py      ->  apple tests passed
python tests\test_particles.py   ->  particle tests passed
```

* `test_math.py` — cell centre conversion, old 2D pixel to world conversion,
  matrix translation, Y rotation, and point/direction transform.
* `test_renderer.py` — reversing the draw order gives an identical colour buffer
  (so visibility comes from the Z-buffer, not from draw order); one triangle
  crossing the near plane is clipped into two; a red box behind a blue box never
  shows through.
* `test_textures.py` — nearest sampling repeats correctly in both directions;
  grass and stone are different materials; a textured rectangle has no cracks
  between its two triangles; the half-way point of a strongly foreshortened quad
  is not at 50 % of the screen width, which proves the UV interpolation is
  perspective-correct and not linear.
* `test_physics.py` — W+D is a unit vector (diagonals are not faster); one second
  of walking covers the same distance at 60 and 200 fps; the worm stops exactly
  one radius short of a wall face; a blocked axis does not stop the other axis
  (wall sliding); a 0.5 s frame does not tunnel through a 0.16 thick wall; the
  collision footprint is exactly the mesh footprint; the worm does not spawn
  inside a wall; looking north puts +X on the right half of the screen, so D
  cannot be swapped with A.
* `test_hazards.py` — the jump peaks at 0.512 and lands after 0.64 s (the
  handbook values); the jump only moves world Y, never X/Z; an airborne worm is
  still blocked by a 0.75 wall; a grounded worm in a puddle drowns while one
  above the 0.30 clearance does not; a grounded worm walks under a bird but
  hits it at the apex of a jump; puddles and birds keep the source placement
  rules (start, finish and water cells excluded), birds cap at eight, and both
  counts shrink with the size of the maze; the "SPACE jumps the puddle" hint
  measures the true distance to the nearest puddle and stays quiet with none;
  and thirty simulated seconds of bird patrol: every cell change goes through a
  side without a wall, a bird never turns straight back, and no bird ever moves
  faster than its own speed.
* `test_apples.py` — four apples per level, three silver (50) and one gold (100)
  for 250 points; never on the start cell, the exit or a puddle, never two on
  one cell; the bob follows `0.28 + 0.05 sin(2 pi 1.5 t)` at its peak and trough
  and the collider uses the same curve; standing on an apple collects it while
  0.48 units away does not; a second pass never collects an apple twice; the
  level only advances with four apples, the score survives `advance()` while
  the apple count resets; `R` (`reset()`) clears the level, the score and the
  clock; silver apples are drawn red and the gold one yellow; no apple is
  walled in on all four sides.
* `test_particles.py` — the pile never exceeds 900 beads even after 4800 are
  spawned; nothing survives its lifetime; a burst owns one `pos`/`vel` array per
  bead (a shared array would move the whole firework as one point); a spark's
  motion has all three components and gravity pulls it down; a falling bead
  never drops below the floor; a bead behind a wall fills no pixel while a bead
  in front of it does, and `depth_test=False` draws through on purpose; spawn
  rule 1 starts every bead at the worm's position; spawn rule 2's ground
  intersection lies on `y = 0`, is the exact inverse of `perspective_project`,
  and rejects a click whose hit is past `max_t`.

## Progress

- [x] M1–M2 coordinates, vectors, matrices, numerical tests
- [x] M3–M4 geometry, camera, perspective and orthographic projection
- [x] M5 solid surfaces, depth buffer, draw-order independence
- [x] M6 maze conversion, wall flags match the original 2D data
- [x] M7 lighting and textures: stone and grass materials, perspective-correct UV
- [x] M8 worm movement and collision: WASD on X/Z, sliding along walls, seven physics tests
- [x] M9 jump and hazards: 0.512 unit arc in world Y, height-aware water and birds
- [x] M10 apples, score, levels: four bobbing apples, 250 points per level, birds patrol the corridors, locked exit
- [x] M11 particles and quality: 3D sparks and fireworks, depth-tested against the Z-buffer, resizable window, F3 debug view
- [ ] M12 extensions, report, demo recording

## Performance

Measured on the machine listed above with the software rasterizer, textured and
lit, standing in the start cell of a 12×12 maze with four apples, three birds and
the exit flag on screen. "Frame" is the full present path (render, HUD, upscale
to the 900×700 window), so it is the number the player actually feels.

| Scene | Internal res | Faces drawn | Frame (no particles) | Frame (450 beads) | Frames/s |
|---|---|---|---|---|---|
| 12×12 play default | 480×373 | 41 / 1008 | 22.7 ms | 25.5 ms | ~39 |
| same, `T` (flat) | 480×373 | 41 / 1008 | 16.8 ms | 19.8 ms | ~51 |
| `--res 560x440` | 560×440 | 41 / 1008 | 27.0 ms | 30.2 ms | ~33 |
| `--res 420x330` | 420×330 | 41 / 1008 | 18.9 ms | 21.0 ms | ~48 |
| screenshots (smoother mesh, no particles) | 900×700 | 48 / 1008 | 65 ms | – | – |

The frame time follows the pixel count, not the triangle count: 420→480→560→900
wide costs 19→23→27→57 ms because the floor and the far walls are large
triangles that touch nearly every pixel. 450 live particles add 1.6–3.2 ms
depending on the resolution, i.e. 5–10 % of a frame.

1. **Frustum culling in one numpy pass.** The static maze is flattened into flat
   arrays once, then every corner and centroid is tested against the frustum in
   a single matmul, dropping 96 % of the faces before any pixel work.
2. **The floor quad is never culled.** It spans the whole maze, so in a corner
   cell none of its corners is inside the frustum and the culling deleted the
   ground from the picture. It is now kept unconditionally and the near-plane
   clipping handles the geometry.
3. **Puddles, birds, apples and the flag are culled too**, so the props cost
   about 4 ms together instead of being submitted whether or not they are on
   screen.
4. **The camera basis is built once per frame**, not once per vertex: the worm,
   the bird and the apple meshes are transformed with one matmul each, the
   sphere mesh is generated once and only scaled and translated afterwards, and
   `draw_triangle` does its arithmetic on the covered pixels instead of on the
   whole bounding box.
5. **Props are low-poly on purpose**: an apple is two 3-by-5 spheres, the worm
   has two 3-by-5 eyes, so 245 triangles are submitted per frame instead of 700.
6. **Clipping is 2 % in front of the projection plane.** A vertex produced
   exactly at `NEAR` was rejected by `perspective_project()` (`depth <= near`),
   which silently dropped every triangle that straddled the near plane — that is
   how the lawn around the maze and parts of the close walls used to disappear.
7. **The clear colour is a vertical ramp**, sky above, lawn below, computed once.
8. **Particles pay for their own pixels only**, and they are depth-tested with a
   single 3×3 minimum from the Z-buffer instead of a ray test, so 450 beads cost
   1.6–3.2 ms and the pile is capped at 900.

The remaining cost is the per-pixel fill of the floor and the far walls. The
resolution switches (`--res`, 480×373 default) and flat shading (`T`) are the
levers that change the frame time most, and screenshots are still rendered at the
full 900×700 with a smoother mesh because a single frame can afford it.

## Credits

* `Start2Dto3D.py` and the project specification are course material for
  GEIAL31L-B2a, Institute of Information Science, University of Miskolc.
* All other code in this repository was written by me. No third-party 3D
  library is used; textures are generated procedurally with Pillow.
