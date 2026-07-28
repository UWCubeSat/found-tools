import argparse
from datetime import datetime, timezone
from pathlib import Path

import pytest

from found_tools.render.main import build_scene_from_args, resolve_scene_file


def make_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        date="2026-03-20T12:00:00",
        position=[7_000_000.0, 0.0, 0.0],
        attitude=[45.0, 0.0, 0.0],
        focal_length=0.05,
        pixel_pitch=5e-6,
        x_resolution=1920,
        y_resolution=1080,
        texture_dir="/textures",
        output="render.png",
        scene_file=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_build_scene_from_args_produces_expected_scene():
    scene = build_scene_from_args(make_args())

    assert scene["date"] == datetime(2026, 3, 20, 12, tzinfo=timezone.utc).isoformat()
    assert scene["camera"]["x_resolution"] == 1920
    assert scene["camera"]["y_resolution"] == 1080
    assert scene["camera"]["position_ecef_m"] == pytest.approx([7_000_000.0, 0.0, 0.0])
    assert scene["earth"]["texture_dir"] == "/textures"


def test_build_scene_from_args_defaults_naive_date_to_utc():
    scene = build_scene_from_args(make_args(date="2026-03-20T12:00:00"))

    assert scene["date"].endswith("+00:00")


def test_resolve_scene_file_uses_explicit_scene_file():
    args = make_args(scene_file="custom.json")

    assert resolve_scene_file(args) == Path("custom.json")


def test_resolve_scene_file_defaults_next_to_output():
    args = make_args(output="render.png", scene_file=None)

    assert resolve_scene_file(args) == Path("render.scene.json")
