"""Tests for limb.simulation.metadata.state."""

import unittest

import numpy as np

from limb.simulation.metadata.state import (
    generate_uniform_directions,
    generate_satellite_state,
)
from limb.utils._camera import Camera, focal_length_from_fov

# Match README example: --fovs 70 --resolutions 1024 --distances 6800000
# --num-positions-per-point 2 --num-spins-per-position 2 --num-radials-per-spin 4
PIXEL_PITCH = 5e-6
FOV_DEG = 70
RESOLUTION = 1024
DISTANCE = 6800000.0
NUM_POSITIONS_PER_POINT = 2
NUM_SPINS_PER_POSITION = 2
NUM_RADIALS_PER_SPIN = 4


class TestGenerateUniformDirections(unittest.TestCase):
    def test_zero_returns_empty(self):
        out = generate_uniform_directions(0)
        self.assertEqual(out.shape, (0, 3))

    def test_one_returns_one_unit_vector(self):
        out = generate_uniform_directions(1)
        self.assertEqual(out.shape, (1, 3))
        np.testing.assert_allclose(np.linalg.norm(out[0]), 1.0)

    def test_multiple_returns_unit_vectors(self):
        out = generate_uniform_directions(10)
        self.assertEqual(out.shape, (10, 3))
        norms = np.linalg.norm(out, axis=1)
        np.testing.assert_allclose(norms, 1.0)


class TestGenerateSatelliteState(unittest.TestCase):
    def test_returns_positions_and_orientations(self):
        # WGS84 ellipsoid (README default semi-axes)
        shape_matrix = np.diag(
            [1.0 / 6378137.0**2, 1.0 / 6378137.0**2, 1.0 / 6356752.0**2]
        )
        earth_directions = np.array([[1.0, 0.0, 0.0]])
        focal_length = focal_length_from_fov(FOV_DEG, RESOLUTION, PIXEL_PITCH)
        camera = Camera(
            focal_length=focal_length,
            x_pixel_pitch=PIXEL_PITCH,
            y_pixel_pitch=PIXEL_PITCH,
            x_resolution=RESOLUTION,
            y_resolution=RESOLUTION,
        )
        positions, orientations = generate_satellite_state(
            shape_matrix,
            earth_directions,
            DISTANCE,
            camera,
            NUM_POSITIONS_PER_POINT,
            NUM_SPINS_PER_POSITION,
            NUM_RADIALS_PER_SPIN,
        )
        n_orient = (
            len(earth_directions)
            * NUM_POSITIONS_PER_POINT
            * NUM_SPINS_PER_POSITION
            * NUM_RADIALS_PER_SPIN
        )
        self.assertEqual(positions.shape[0], n_orient)
        self.assertEqual(positions.shape[1], 3)
        self.assertEqual(orientations.shape[0], n_orient)
        self.assertEqual(orientations.shape[1], 3)
        self.assertEqual(orientations.shape[2], 3)
        self.assertTrue(np.isfinite(positions).all())
        self.assertTrue(np.isfinite(orientations).all())
        for r in orientations:
            np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-5)

    def test_multiple_earth_directions(self):
        shape_matrix = np.diag([1.0 / 100.0**2, 1.0 / 100.0**2, 1.0 / 100.0**2])
        earth_directions = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        distance = 200.0
        focal_length = focal_length_from_fov(FOV_DEG, RESOLUTION, PIXEL_PITCH)
        camera = Camera(
            focal_length=focal_length,
            x_pixel_pitch=PIXEL_PITCH,
            y_pixel_pitch=PIXEL_PITCH,
            x_resolution=RESOLUTION,
            y_resolution=RESOLUTION,
        )
        positions, orientations = generate_satellite_state(
            shape_matrix,
            earth_directions,
            distance,
            camera,
            NUM_POSITIONS_PER_POINT,
            NUM_SPINS_PER_POSITION,
            NUM_RADIALS_PER_SPIN,
        )
        n_orient = (
            len(earth_directions)
            * NUM_POSITIONS_PER_POINT
            * NUM_SPINS_PER_POSITION
            * NUM_RADIALS_PER_SPIN
        )
        self.assertEqual(positions.shape[0], n_orient)
        self.assertEqual(orientations.shape[0], positions.shape[0])


if __name__ == "__main__":
    unittest.main()
