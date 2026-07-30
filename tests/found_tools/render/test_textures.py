from datetime import UTC, datetime
from pathlib import Path

import pytest

from found_tools.render.constants import CLOUD_LAYER_FILENAME
from found_tools.render.textures import (
    blue_marble_filename,
    cloud_layer_filename,
    resolve_texture_path,
)


def test_blue_marble_filename_matches_month():
    when = datetime(2026, 3, 20, tzinfo=UTC)
    assert blue_marble_filename(when) == "world.200403.3x5400x2700.jpg"


def test_blue_marble_filename_covers_all_months():
    for month in range(1, 13):
        when = datetime(2026, month, 1, tzinfo=UTC)
        filename = blue_marble_filename(when)
        assert filename.startswith("world.")
        assert filename.endswith(".jpg")


def test_cloud_layer_filename_is_generic():
    assert cloud_layer_filename() == CLOUD_LAYER_FILENAME


def test_resolve_texture_path_returns_existing_file(tmp_path: Path):
    texture_file = tmp_path / "world.202603.3x5400x2700.jpg"
    texture_file.write_bytes(b"fake-image-data")

    resolved = resolve_texture_path(tmp_path, texture_file.name)

    assert resolved == texture_file


def test_resolve_texture_path_raises_when_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        resolve_texture_path(tmp_path, "does-not-exist.jpg")
