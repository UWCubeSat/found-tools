# Render

The `render` tool generates synthetic Earth-limb test imagery for `EarthLim`
and similar attitude-determination tools in [found](https://github.com/UWCubeSat/found).
It computes Sun/camera geometry for a given date, position, and attitude,
then drives a headless Blender scene to produce a physically-plausible
rendered image: a textured Earth with a cloud layer, a Sun lamp along the
computed Sun vector, and an atmosphere shader for limb glow and terminator
softening.

## How it works

1. `found_tools.render.geometry` computes the Sun direction and the
   ECI->ECEF rotation for the requested UTC date using closed-form
   analytical formulas (no SPICE kernel or IERS data download required).
2. `found_tools.render.scene` packages the Sun vector, camera intrinsics
   (reusing `found_tools.utils.Camera`), camera pose, and date-appropriate
   texture filenames into a scene JSON file.
3. `found_tools.render.main` (the `found_tools_render` CLI) writes the
   scene JSON and, if `--render` is passed, invokes Blender in headless
   mode (`blender --background --python blender_scene.py`) to build and
   render the scene.
4. `found_tools.render.blender_scene` is the only module that imports
   `bpy`. It runs inside Blender's own bundled Python interpreter (not the
   project's virtualenv -- see "Rendering" below) and is not covered by
   the unit test suite.

## Textures

This tool does not bundle texture assets. Download them yourself into a
directory and pass it via `--texture-dir`:

- **Earth color**: [NASA Blue Marble Next Generation](https://visibleearth.nasa.gov/collection/1484/blue-marble),
  one mosaic per month (`world.<year><month>.3x5400x2700.jpg`). The tool
  picks the file matching the requested date's month.
- **Clouds**: a generic cloud layer texture (e.g. the Blue Marble cloud
  mosaic), saved as `cloud_combined_2048.jpg`.
  **Caveat**: NASA does not publish a month-by-month cloud product, so the
  same cloud texture is used regardless of date. It gives the render a
  plausible cloud layer, but it will not match real weather for the
  requested date.

## Usage

Compute geometry and inspect the scene JSON without needing Blender
installed:

```bash
found_tools_render \
  --date 2026-03-20T12:00:00 \
  --position 7000000 0 0 \
  --attitude 45 0 0 \
  --focal-length 0.05 \
  --pixel-pitch 5e-6 \
  --x-resolution 1920 \
  --y-resolution 1080 \
  --texture-dir ./textures \
  --output render.png
```

### Rendering

`bpy` (Blender as a Python module) is not published for this project's
Python version, so it is not a dependency of `found-tools`. Instead, add
`--render` and point `--blender-executable` at a real Blender install; the
CLI shells out to Blender's own bundled Python to run
`blender_scene.py`:

```bash
found_tools_render \
  --date 2026-03-20T12:00:00 \
  --position 7000000 0 0 \
  --attitude 45 0 0 \
  --focal-length 0.05 \
  --pixel-pitch 5e-6 \
  --x-resolution 1920 \
  --y-resolution 1080 \
  --texture-dir ./textures \
  --output render.png \
  --render \
  --blender-executable /path/to/blender
```

## Flags

- `--date`: UTC date/time to render, ISO 8601. Drives the Sun vector and
  the Blue Marble texture month.
- `--position`: Camera position in the ECEF frame, meters.
- `--attitude`: Camera attitude (ECEF->camera) as RA/DE/ROLL degrees, in
  the same convention as the `calibrate` tool.
- `--focal-length`, `--pixel-pitch`, `--x-resolution`, `--y-resolution`:
  Camera intrinsics, matched to the real camera model under test.
- `--texture-dir`: Directory containing the downloaded textures.
- `--output`: Path to write the rendered PNG to.
- `--scene-file`: Where to write the intermediate scene JSON (defaults
  next to `--output`).
- `--blender-executable`: Path to the Blender executable (default:
  `blender` on `PATH`).
- `--render`: Actually invoke Blender. Without it, only the scene JSON is
  written.
