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
| `python walk_maze.py` | M8/M9: walk with WASD, jump with SPACE, puddles and birds (`--cols --rows --res --demo --shot`) |
| `python walk_maze.py --demo` | scripted jump: apex, drown in a puddle, jump across it (writes screenshots 14–16) |
| `python report_shots.py` | regenerates screenshots 03, 04 and 06–11 at 900×700 |
| `python first_maze.py --cols 10 --rows 10 --shot screenshots\05_maze_wireframe_10x10.png` | 10×10 scalability check (120 boxes) |
| `python tests\test_math.py` | coordinate and matrix tests |
| `python tests\test_renderer.py` | depth-buffer, clipping and draw-order tests |
| `python tests\test_textures.py` | texture sampling, materials and perspective UV tests |
| `python tests\test_physics.py` | movement, frame-rate independence and wall collision tests |
| `python tests\test_hazards.py` | jump arc, wall clearing, water and bird height rules |

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
| `P` | toggle between perspective and orthographic projection |
| `R` | toggle draw order near→far / far→near (solid viewer only) |
| `T` | toggle textures on/off (textured and walk viewers) |
| `L` | cycle the light direction (textured viewer only) |
| `ESC` | quit |

Movement follows handbook chapter 11: world-aligned keys, normalised diagonal
input, 4.6 units/s and a 0.22 radius, with X and Z resolved separately so the
worm slides along walls instead of stopping dead.

`walk_maze.py` also accepts `--res 720x560` to render at a lower internal
resolution and scale the image up, and `--cols 3 --rows 3` for the smoothest
frame rate (the software rasterizer costs roughly 0.1–0.2 ms per triangle).

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
* **Puddles** — the worm drowns when its feet are below 0.30 units and its
  centre is inside the disc that is drawn (radius 0.42), which is the 3D form of
  the 2D code's `if jump_timer > 0: return False`. Above that clearance it
  skims over: jumping across a puddle keeps the feet at about 0.45 while the
  centre crosses the disc.
* **Birds** — a 3D sphere test with summed radii 0.42. The birds hover at 0.75,
  so a grounded worm walks underneath them and a jump hits them; flying over
  one is impossible with a 0.512 unit jump.
* Drowning or hitting a bird resets the worm to the start cell, like
  `reset_ball()` in the original game.

`python walk_maze.py --demo` prints the height of the feet every quarter
second while it script-walks these three cases, so the rules can be checked
without playing.

Original 2D game: `W A S D` to move, collect the apples, avoid water holes and the bird.

## World conventions

* One maze cell = **1 world unit**. The 2D cell was 50 px, so every pixel length
  and speed is divided by 50 (ball radius 11 px → 0.22, speed 230 px/s → 4.6).
* Axes: **X = east, Y = up, Z = south**.
* The 2D screen space had y growing downwards, so screen y becomes world **Z**,
  not world Y.
* The 2D player position included the origin offset `ORIGIN_X=40`, `ORIGIN_Y=95`;
  it is subtracted first, then divided by 50: pixel (65, 470) → world (0.5, 0.22, 7.5).
* Cell centres are at `(col + 0.5, 0, row + 0.5)`; walls are 0.75 high and 0.16 thick.

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
walk_maze.py       M8/M9 viewer: walk, jump, puddles, birds, follow camera, --demo log
report_shots.py    batch script that regenerates the report screenshots
tests/             numerical, renderer, texture, physics and hazard tests
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
| `13_worm_against_wall.png` | M8: worm pressed into the north-east corner, HUD reports SLIDING |
| `14_worm_jump_apex.png` | M9: jump at its apex, feet 0.49 above the floor, HUD reports AIR |
| `15_worm_at_puddle.png` | M9: grounded worm walking into a puddle (drowns on contact) |
| `16_worm_jumped_puddle.png` | M9: the same puddle crossed with a jump, feet 0.45 while over the water |

All of them are reproducible: fixed seed 7, fixed camera angle 0.55, internal
render resolution 900×700 (stated in the on-screen HUD of every image).

## Tests

```
python tests\test_math.py        ->  all tests passed
python tests\test_renderer.py    ->  renderer tests passed
python tests\test_textures.py    ->  texture tests passed
python tests\test_physics.py     ->  physics tests passed
python tests\test_hazards.py     ->  hazard tests passed
```

Twenty-six assertions in total:

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
  inside a wall.
* `test_hazards.py` — the jump peaks at 0.512 and lands after 0.64 s (the
  handbook values); the jump only moves world Y, never X/Z; an airborne worm is
  still blocked by a 0.75 wall; a grounded worm in a puddle drowns while one
  above the 0.30 clearance does not; a grounded worm walks under a bird but
  hits it at the apex of a jump; puddles and birds keep the source placement
  rules (start, finish and water cells excluded) and birds cap at eight.

## Progress

- [x] M1–M2 coordinates, vectors, matrices, numerical tests
- [x] M3–M4 geometry, camera, perspective and orthographic projection
- [x] M5 solid surfaces, depth buffer, draw-order independence
- [x] M6 maze conversion, wall flags match the original 2D data
- [x] M7 lighting and textures: stone and grass materials, perspective-correct UV
- [x] M8 worm movement and collision: WASD on X/Z, sliding along walls, seven physics tests
- [x] M9 jump and hazards: 0.512 unit arc in world Y, height-aware water and birds
- [ ] M10 apples, birds, level rules
- [ ] M11 particles, HUD, quality
- [ ] M12 extensions, report, demo recording

## Performance

Measured on one core of the machine listed above, 900×700 internal resolution,
software rasterizer, textured and lit:

| Scene | Triangles | Frame time |
|---|---|---|
| 3×3 maze, walk view | 384 | 82 ms |
| 5×5 maze, walk view | ~700 | 180 ms |
| 10×10 maze, walk view | 1386 | 220 ms |
| 10×10 with `--res 600x466` | 1386 | 182 ms |

The cost is dominated by per-triangle numpy work (vertex transform, culling,
near clipping), not by pixels, which is why lowering the resolution saves little.
Batching the vertex transform is the milestone 11 improvement.

## Credits

* `Start2Dto3D.py` and the project specification are course material for
  GEIAL31L-B2a, Institute of Information Science, University of Miskolc.
* All other code in this repository was written by me. No third-party 3D
  library is used; textures are generated procedurally with Pillow.
