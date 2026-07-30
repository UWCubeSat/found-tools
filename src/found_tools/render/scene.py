"""Builds a JSON-serializable scene description for the Blender renderer.

This module contains only pure, testable logic: it computes the Sun vector
and camera pose/intrinsics and packages them into a plain dict. The dict is
handed off to ``blender_scene.py`` (run inside Blender's own Python
interpreter, see the render tool README) which is the only part of this
tool that imports ``bpy``.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from found_tools.calibrate.transform import DCM
from found_tools.render.constants import (
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
)
from found_tools.render.geometry import sun_vector_ecef
from found_tools.render.textures import blue_marble_filename, cloud_layer_filename
from found_tools.utils._camera import Camera


def build_scene(
    when: datetime,
    camera: Camera,
    camera_position_ecef_m: np.ndarray,
    camera_attitude_ecef_to_camera: DCM,
    image_width: int = DEFAULT_IMAGE_WIDTH,
    image_height: int = DEFAULT_IMAGE_HEIGHT,
    texture_dir: str | None = None,
) -> dict:
    """Builds a JSON-serializable scene description.

    Args:
        when: UTC datetime the image should depict (drives the Sun vector
            and the Blue Marble texture month).
        camera: Camera intrinsics to reproduce in Blender.
        camera_position_ecef_m: Shape (3,) spacecraft position in the ECEF
            frame, in meters.
        camera_attitude_ecef_to_camera: Rotation from the ECEF frame to the
            camera frame (x boresight), as produced by the calibrate tool's
            attitude propagation.
        image_width: Rendered image width in pixels.
        image_height: Rendered image height in pixels.
        texture_dir: Optional directory containing downloaded texture
            assets; stored in the scene so blender_scene.py can resolve
            texture files without recomputing filenames.

    Returns:
        dict: A JSON-serializable scene description.
    """
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)

    sun_vector = sun_vector_ecef(when)
    position = np.asarray(camera_position_ecef_m, dtype=np.float64)
    rotation_matrix = camera_attitude_ecef_to_camera.rotation.as_matrix()

    return {
        "date": when.astimezone(UTC).isoformat(),
        "sun_vector_ecef": sun_vector.tolist(),
        "camera": {
            "focal_length_m": camera.focal_length,
            "x_pixel_pitch_m": camera.x_pixel_pitch,
            "y_pixel_pitch_m": camera.y_pixel_pitch,
            "x_resolution": camera.x_resolution,
            "y_resolution": camera.y_resolution,
            "x_center": camera.x_center,
            "y_center": camera.y_center,
            "position_ecef_m": position.tolist(),
            "rotation_ecef_to_camera": rotation_matrix.tolist(),
        },
        "earth": {
            "blue_marble_filename": blue_marble_filename(when),
            "cloud_layer_filename": cloud_layer_filename(),
            "texture_dir": texture_dir,
        },
        "image": {
            "width": image_width,
            "height": image_height,
        },
    }


def write_scene(scene: dict, output_path: Path) -> Path:
    """Writes a scene description dict to a JSON file.

    Args:
        scene: The scene description, as produced by :func:`build_scene`.
        output_path: Path to write the scene JSON to.

    Returns:
        Path: The path the scene was written to.
    """
    output_path = Path(output_path)
    output_path.write_text(json.dumps(scene, indent=2))
    return output_path
