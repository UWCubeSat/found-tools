"""Core math utilities for the edge-point generator.

The implementation follows the perspective projection model described in
common camera-projection references: a world-space point is rotated into the
camera frame and intersected with the film plane at ``z = focal_length``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

from scipy.spatial.transform import Rotation

Vector3 = Tuple[float, float, float]
Quaternion = Tuple[float, float, float, float]

_EPSILON = 1e-12


@dataclass(frozen=True)
class FilmEdgePoint:
    """Represents a precise (non-integer) coordinate on the film plane."""

    x: float
    y: float


def _coerce_vector(
    values: Sequence[float], expected_length: int, name: str
) -> Tuple[float, ...]:
    """Validate and coerce a numeric sequence into a tuple of floats."""

    try:
        vector = tuple(float(component) for component in values)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of numbers") from exc

    if len(vector) != expected_length:
        raise ValueError(f"{name} must contain {expected_length} values")

    return vector


def generate_edge_point(
    point: Sequence[float],
    quaternion: Sequence[float],
    focal_length: float = 1.0,
) -> FilmEdgePoint:
    """Project a world-space point into film-plane coordinates.

    Args:
        point: The (x, y, z) coordinate of the point in world/cartesian space.
        quaternion: Camera orientation as (x, y, z, w) quaternion rotating the
            camera frame into the world frame.
        focal_length: Distance between camera origin and film plane. Units are
            preserved in the output film coordinates.

    Returns:
        FilmEdgePoint: The projected film-plane coordinate (x_f, y_f).

    Raises:
        ValueError: If the inputs are malformed, the focal length is invalid, or
            the point lies behind the camera/film plane.
    """

    if focal_length <= 0:
        raise ValueError("focal_length must be positive")

    world_point = _coerce_vector(point, 3, "point")
    quat = _coerce_vector(quaternion, 4, "quaternion")

    rotation = Rotation.from_quat(quat)
    camera_point = rotation.inv().apply(world_point)

    z_cam = float(camera_point[2])
    if z_cam <= _EPSILON:
        raise ValueError("Point projects behind the camera or onto the film plane")

    scale = focal_length / z_cam
    x_film = float(camera_point[0]) * scale
    y_film = float(camera_point[1]) * scale

    return FilmEdgePoint(x=x_film, y=y_film)
