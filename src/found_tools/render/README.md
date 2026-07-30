# Render

The `render` tool generates synthetic Earth-limb test imagery for `EarthLim`
and similar attitude-determination tools in [found](https://github.com/UWCubeSat/found).
It computes Sun/camera geometry for a given date, position, and attitude,
then drives a headless Blender scene to produce a physically-plausible
rendered image: a WGS84-ellipsoid Earth with a cloud layer, a Sun lamp along
the computed Sun vector, and a sun-facing atmosphere glow shell for limb
glow and terminator softening.

The Earth mesh (and the concentric cloud/atmosphere shells) are built as
WGS84 ellipsoids -- equatorial radius 6,378,137 m, polar radius
6,356,752.314245 m (flattening ~1/298.257) -- not spheres. The ~21 km
difference is small next to Earth's radius but matters for accurate
limb-fitting.

## How it works

1. `found_tools.render.geometry` computes the Sun direction and the
   ECI->ECEF rotation for the requested UTC date using closed-form
   analytical formulas (no SPICE kernel or IERS data download required).
2. `found_tools.render.scene` packages the Sun vector, camera intrinsics
   (reusing `found_tools.utils.Camera`), camera pose, and date-appropriate
   texture filenames into a scene JSON file.
3. `found_tools.render.main` (the `found_tools_render` CLI) writes the
   scene JSON and, if `--render` is passed, calls
   `found_tools.render.blender_scene.render_scene` in process to build and
   render the scene.
4. `found_tools.render.blender_scene` is the only module that imports
   `bpy` (Blender as a Python module, a project dependency). It can also be
   run standalone inside a full Blender install
   (`blender --background --python blender_scene.py`) if you need Blender's
   own bundled interpreter instead.

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
  --position 25000000 0 0 \
  --attitude 180 0 0 \
  --focal-length 0.008 \
  --pixel-pitch 5e-6 \
  --x-resolution 1920 \
  --y-resolution 1080 \
  --texture-dir ./textures \
  --output render.png
```

### Rendering

`bpy` (Blender as a Python module) is a project dependency, so add
`--render` to actually build and render the scene -- no separate Blender
install is required:

```bash
found_tools_render \
  --date 2026-03-20T12:00:00 \
  --position 25000000 0 0 \
  --attitude 180 0 0 \
  --focal-length 0.008 \
  --pixel-pitch 5e-6 \
  --x-resolution 1920 \
  --y-resolution 1080 \
  --texture-dir ./textures \
  --output render.png \
  --render
```

### Choosing camera parameters

`--position`, `--focal-length`, `--pixel-pitch`, and `--x/y-resolution`
must be consistent with each other, or the render can come out looking
like an unrecognizable, blurry close-up instead of a visible Earth disk.
The camera's field of view is set by focal length and sensor size
(`resolution * pixel_pitch`), same as a real camera; if that FOV is much
narrower than the Earth's angular size at the requested `--position`
(`2 * asin(6378137 / distance_from_earth_center_m)`), the camera is
effectively a telephoto lens pointed at a patch of the surface a few
kilometers wide, magnified to fill the whole frame -- the "blurry mess"
you'd expect from zooming a real photo in far past its resolution. Widen
the lens (shorter `--focal-length` and/or larger `--pixel-pitch`), move
`--position` farther out, or both, until the FOV comfortably exceeds the
Earth's angular size at that distance. The example above (25,000 km
altitude, an 8 mm-equivalent lens) puts the full Earth disk in frame with
margin.

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
- `--render`: Actually build and render the scene with `bpy`. Without it,
  only the scene JSON is written.
- `--no-atmosphere`: Skip the atmosphere limb-glow shell, rendering just
  the Earth, clouds, and lighting. The Earth's WGS84 shape and the black
  deep-space background are unaffected.
