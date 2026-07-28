"""Texture selection helpers for the render tool.

Caveat: NASA's Blue Marble Next Generation is published as one true-color
mosaic per month, so :func:`blue_marble_filename` can match the requested
date's season/vegetation state closely. There is no equivalent month-by-month
public cloud product, so :func:`cloud_layer_filename` always returns the same
generic cloud layer regardless of date -- it will not reflect real
weather/cloud cover for the requested date, only give the scene a
plausible-looking cloud layer.
"""

from datetime import datetime
from pathlib import Path

from found_tools.render.constants import (
    BLUE_MARBLE_MONTH_FILENAMES,
    CLOUD_LAYER_FILENAME,
)


def blue_marble_filename(when: datetime) -> str:
    """Selects the Blue Marble Next Generation texture filename for a date.

    Args:
        when: The UTC datetime whose month should be used to pick a
            date-appropriate true-color Earth texture.

    Returns:
        str: The Blue Marble texture filename for that month.
    """
    return BLUE_MARBLE_MONTH_FILENAMES[when.month]


def cloud_layer_filename() -> str:
    """Returns the generic cloud layer texture filename.

    Returns:
        str: The cloud layer texture filename (not date-specific; see
            module docstring caveat).
    """
    return CLOUD_LAYER_FILENAME


def resolve_texture_path(texture_dir: Path, filename: str) -> Path:
    """Resolves a texture filename against a texture directory.

    Args:
        texture_dir: Directory containing downloaded texture assets.
        filename: The texture filename to resolve.

    Returns:
        Path: The resolved path to the texture file.

    Raises:
        FileNotFoundError: If the texture file does not exist under
            texture_dir.
    """
    path = Path(texture_dir) / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"Texture '{filename}' not found in '{texture_dir}'. "
            "Download Blue Marble Next Generation and a cloud layer texture "
            "into this directory (see the render tool README)."
        )
    return path
