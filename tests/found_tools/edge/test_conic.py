import numpy as np
import pytest

from found_tools.edge.conic import (
    _shape_matrix_from_axes,
    conic_matrix_to_coeffs,
    generate_camera_conic,
    generate_pixel_conic,
)
from found_tools.utils._camera import Camera


@pytest.fixture
def sphere_camera_setup():
    """Unit sphere imaged by a 2×2, 45° FOV, focal-length-1 camera at (3,0,0)."""

    cam = Camera(focal_length=1, x_pixel_pitch=1, x_resolution=2)

    # Camera at world (3,0,0), spun 180° around z to face the sphere
    tpc = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
    rw = np.array([3.0, 0.0, 0.0])

    return cam, tpc, rw


def test_unit_sphere(sphere_camera_setup):
    """Unit sphere viewed on-axis produces a circle (equal diag, zero cross-term)."""
    cam, tpc, rw = sphere_camera_setup
    shape_matrix = np.eye(3)  # Unit sphere has shape matrix diag(1, 1, 1).

    cam_conic = generate_camera_conic(rw, shape_matrix, tpc)
    pixel_conic = generate_pixel_conic(cam_conic, cam)
    conic_coeffs = conic_matrix_to_coeffs(pixel_conic)

    """Circle with radius sqrt(2)/4, centred at (1, 1) in pixel coordinates"""
    expected_coeffs = np.array([1.0, 0.0, 1.0, -2.0, -2.0, 15 / 8])

    assert conic_coeffs == pytest.approx(expected_coeffs, abs=1e-9)


def test_bigger_sphere(sphere_camera_setup):
    """Unit sphere viewed on-axis produces a circle (equal diag, zero cross-term)."""
    cam, tpc, rw = sphere_camera_setup
    shape_matrix = _shape_matrix_from_axes(
        [2, 2, 2]
    )  # Unit sphere has shape matrix diag(1, 1, 1).

    cam_conic = generate_camera_conic(rw, shape_matrix, tpc)
    pixel_conic = generate_pixel_conic(cam_conic, cam)
    conic_coeffs = conic_matrix_to_coeffs(pixel_conic)

    """Circle with radius 2/sqrt(5), centred at (1, 1) in pixel coordinates"""
    expected_coeffs = np.array([1.0, 0.0, 1.0, -2.0, -2.0, 6 / 5])

    assert conic_coeffs == pytest.approx(expected_coeffs, abs=1e-9)


def test_ellipse(sphere_camera_setup):
    """Circle is centred in the image: equal linear terms in x and y."""
    cam, tpc, rc = sphere_camera_setup
    shape_matrix = _shape_matrix_from_axes(
        [1, 2, 1]
    )  # Ellipsoid with z-axis half the radius.

    cam_conic = generate_camera_conic(rc, shape_matrix, tpc)
    pixel_conic = generate_pixel_conic(cam_conic, cam)
    conic_coeffs = conic_matrix_to_coeffs(pixel_conic)

    """Ellipse with x-radius sqrt(2)/4 and y-radius sqrt(2)/2, centred at (1, 1) in pixel coordinates"""
    expected_coeffs = np.array([1.0, 0.0, 4, -2.0, -8.0, 9 / 2])

    assert conic_coeffs == pytest.approx(expected_coeffs, abs=1e-9)
