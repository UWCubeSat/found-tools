import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from found_tools.calibrate.transform import Attitude
from found_tools.render.blender_scene import render_scene
from found_tools.render.scene import build_scene, write_scene
from found_tools.utils._camera import Camera

logging.basicConfig(level=logging.DEBUG, format="[%(levelname)s]: %(message)s")


def parse_args() -> argparse.Namespace:  # pragma: no cover
    """Parses arguments for the render tool."""
    parser = argparse.ArgumentParser(
        description=(
            "Generates a synthetic Earth-limb test image (for EarthLim and "
            "similar attitude-determination tools) by computing Sun/camera "
            "geometry and rendering it in headless Blender."
        )
    )

    parser.add_argument(
        "--date",
        required=True,
        help="UTC date/time to render, ISO 8601 (e.g. 2026-03-20T12:00:00).",
    )
    parser.add_argument(
        "--position",
        nargs=3,
        type=float,
        required=True,
        metavar=("X", "Y", "Z"),
        help="Camera position in the ECEF frame, meters.",
    )
    parser.add_argument(
        "--attitude",
        nargs=3,
        type=float,
        required=True,
        metavar=("RA", "DE", "ROLL"),
        help="Camera attitude (ECEF->camera) as RA/DE/ROLL degrees, in the "
        "same convention as the calibrate tool.",
    )
    parser.add_argument(
        "--focal-length", type=float, required=True, help="Focal length, meters."
    )
    parser.add_argument(
        "--pixel-pitch", type=float, required=True, help="Pixel pitch, meters."
    )
    parser.add_argument(
        "--x-resolution", type=int, required=True, help="Sensor width, pixels."
    )
    parser.add_argument(
        "--y-resolution", type=int, required=True, help="Sensor height, pixels."
    )
    parser.add_argument(
        "--texture-dir",
        required=True,
        help="Directory containing the downloaded Blue Marble and cloud "
        "layer textures (see the render tool README).",
    )
    parser.add_argument(
        "--output", required=True, help="Path to write the rendered PNG to."
    )
    parser.add_argument(
        "--scene-file",
        default=None,
        help="Where to write the intermediate scene JSON. Defaults next to --output.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the scene with Blender's bpy module. If omitted, only "
        "the scene JSON is written (useful for inspecting geometry without "
        "waiting on a render).",
    )
    parser.add_argument(
        "--no-atmosphere",
        action="store_true",
        help="Skip the atmosphere limb-glow shell, rendering just the Earth, "
        "clouds, and lighting. The Earth's shape and the black deep-space "
        "background are unaffected.",
    )

    return parser.parse_args()


def build_scene_from_args(args: argparse.Namespace) -> dict:
    """Builds the scene description dict from parsed CLI arguments.

    Args:
        args: Parsed CLI arguments (see :func:`parse_args`).

    Returns:
        dict: The scene description, as produced by
            :func:`found_tools.render.scene.build_scene`.
    """
    when = datetime.fromisoformat(args.date)
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)

    camera = Camera(
        focal_length=args.focal_length,
        x_pixel_pitch=args.pixel_pitch,
        x_resolution=args.x_resolution,
        y_resolution=args.y_resolution,
    )
    attitude = Attitude(*args.attitude).to_dcm()
    position = np.array(args.position, dtype=np.float64)

    return build_scene(
        when=when,
        camera=camera,
        camera_position_ecef_m=position,
        camera_attitude_ecef_to_camera=attitude,
        texture_dir=args.texture_dir,
        atmosphere_glow_enabled=not args.no_atmosphere,
    )


def resolve_scene_file(args: argparse.Namespace) -> Path:
    """Resolves the path the scene JSON should be written to.

    Args:
        args: Parsed CLI arguments (see :func:`parse_args`).

    Returns:
        Path: The scene JSON output path.
    """
    if args.scene_file is not None:
        return Path(args.scene_file)
    return Path(args.output).with_suffix(".scene.json")


def main():  # pragma: no cover
    args = parse_args()
    scene = build_scene_from_args(args)
    write_scene(scene, resolve_scene_file(args))

    if args.render:
        render_scene(scene, Path(args.output))


if __name__ == "__main__":  # pragma: no cover
    main()
