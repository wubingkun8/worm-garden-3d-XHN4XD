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
| `python report_shots.py` | regenerates screenshots 03, 04, 06, 07 at 900×700 |
| `python first_maze.py --cols 10 --rows 10 --shot screenshots\05_maze_wireframe_10x10.png` | 10×10 scalability check (120 boxes) |
| `python tests\test_math.py` | coordinate and matrix tests |
| `python tests\test_renderer.py` | depth-buffer, clipping and draw-order tests |

Full original maze layout:

```
python solid_maze.py --cols 23 --rows 15
```

## Controls

| Key | Action |
|---|---|
| `P` | toggle between perspective and orthographic projection |
| `R` | toggle draw order near→far / far→near (solid viewer only) |
| `ESC` | quit |

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
geometry.py        cube mesh, wall and floor boxes, world vertices
camera.py          camera basis, perspective and orthographic projection
renderer.py        clipping, back-face culling, flat shading, rasterizer, Z-buffer
first_cube.py      M2/M4 mini-lab: rotating cube, P toggles projection
first_maze.py      M3/M4 viewer: wireframe maze
solid_maze.py      M5 viewer: solid maze, shading and depth
report_shots.py    batch script that regenerates the report screenshots
tests/             numerical and renderer tests
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

All of them are reproducible: fixed seed 7, fixed camera angle 0.55, internal
render resolution 900×700 (stated in the on-screen HUD of every image).

## Tests

```
python tests\test_math.py        ->  all tests passed
python tests\test_renderer.py    ->  renderer tests passed
```

Eight assertions in total:

* `test_math.py` — cell centre conversion, old 2D pixel to world conversion,
  matrix translation, Y rotation, and point/direction transform.
* `test_renderer.py` — reversing the draw order gives an identical colour buffer
  (so visibility comes from the Z-buffer, not from draw order); one triangle
  crossing the near plane is clipped into two; a red box behind a blue box never
  shows through.

## Progress

- [x] M1–M2 coordinates, vectors, matrices, numerical tests
- [x] M3–M4 geometry, camera, perspective and orthographic projection
- [x] M5 solid surfaces, depth buffer, draw-order independence
- [x] M6 maze conversion, wall flags match the original 2D data
- [ ] M7 lighting and textures
- [ ] M8–M10 movement, jumping, hazards, birds, apples, levels
- [ ] M11 particles, HUD, quality
- [ ] M12 extensions, report, demo recording

## Credits

* `Start2Dto3D.py` and the project specification are course material for
  GEIAL31L-B2a, Institute of Information Science, University of Miskolc.
* All other code in this repository was written by me. No third-party 3D
  library is used; textures are generated procedurally with Pillow.
