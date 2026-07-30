import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from found_tools.calibrate.transform import DCM
from found_tools.render.scene import build_scene, write_scene
from found_tools.utils._camera import Camera

WHEN = datetime(2026, 3, 20, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
def camera() -> Camera:
    return Camera(0.05, 5e-6, 1920, 1080)


def test_build_scene_contains_expected_top_level_keys(camera):
    scene = build_scene(
        when=WHEN,
        camera=camera,
        camera_position_ecef_m=np.array([7_000_000.0, 0.0, 0.0]),
        camera_attitude_ecef_to_camera=DCM(),
        texture_dir="/textures",
    )

    assert set(scene.keys()) == {
        "date",
        "sun_vector_ecef",
        "camera",
        "earth",
        "image",
    }
    assert scene["date"] == WHEN.isoformat()
    assert scene["earth"]["blue_marble_filename"] == "world.200403.3x5400x2700.jpg"
    assert scene["earth"]["texture_dir"] == "/textures"
    assert scene["image"] == {"width": 1920, "height": 1080}


def test_build_scene_normalizes_naive_datetime(camera):
    naive_when = datetime(2026, 3, 20, 12, 0, 0)  # noqa: DTZ001 -- deliberately naive

    scene = build_scene(
        when=naive_when,
        camera=camera,
        camera_position_ecef_m=np.array([7_000_000.0, 0.0, 0.0]),
        camera_attitude_ecef_to_camera=DCM(),
    )

    assert scene["date"] == WHEN.isoformat()


def test_build_scene_camera_fields_match_input(camera):
    position = np.array([1.0, 2.0, 3.0])

    scene = build_scene(
        when=WHEN,
        camera=camera,
        camera_position_ecef_m=position,
        camera_attitude_ecef_to_camera=DCM(),
    )

    cam = scene["camera"]
    assert cam["focal_length_m"] == camera.focal_length
    assert cam["x_resolution"] == camera.x_resolution
    assert cam["y_resolution"] == camera.y_resolution
    assert cam["position_ecef_m"] == pytest.approx([1.0, 2.0, 3.0])
    assert np.array(cam["rotation_ecef_to_camera"]).shape == (3, 3)


def test_build_scene_is_json_serializable(camera):
    scene = build_scene(
        when=WHEN,
        camera=camera,
        camera_position_ecef_m=np.array([7_000_000.0, 0.0, 0.0]),
        camera_attitude_ecef_to_camera=DCM(),
    )

    # Should not raise.
    json.dumps(scene)


def test_write_scene_writes_valid_json(tmp_path: Path, camera):
    scene = build_scene(
        when=WHEN,
        camera=camera,
        camera_position_ecef_m=np.array([7_000_000.0, 0.0, 0.0]),
        camera_attitude_ecef_to_camera=DCM(),
    )
    output_path = tmp_path / "scene.json"

    result_path = write_scene(scene, output_path)

    assert result_path == output_path
    assert json.loads(output_path.read_text()) == scene
