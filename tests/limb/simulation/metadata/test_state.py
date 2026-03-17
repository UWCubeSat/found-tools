"""Tests for limb.simulation.metadata.state."""

import unittest

import numpy as np

from limb.simulation.metadata.state import (
    generate_uniform_directions,
    generate_satellite_state,
)
from limb.utils._camera import Camera


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
        shape_matrix = np.diag(
            [1.0 / 6378137.0**2, 1.0 / 6378137.0**2, 1.0 / 6356752.0**2]
        )
        earth_directions = np.array([[1.0, 0.0, 0.0]])
        distance = 6800000.0
        camera = Camera(
            focal_length=0.035,
            x_pixel_pitch=5e-6,
            x_resolution=64,
            y_resolution=64,
        )
        num_positions = 2
        num_spins = 2
        num_radials = 2
        positions, orientations = generate_satellite_state(
            shape_matrix,
            earth_directions,
            distance,
            camera,
            num_positions,
            num_spins,
            num_radials,
        )
        n_orient = num_positions * num_spins * num_radials
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
        camera = Camera(
            focal_length=0.035,
            x_pixel_pitch=5e-6,
            x_resolution=64,
            y_resolution=64,
        )
        positions, orientations = generate_satellite_state(
            shape_matrix,
            earth_directions,
            distance,
            camera,
            numSatellitePositions=2,
            numImageSpins=1,
            numImageRadials=1,
        )
        self.assertGreater(positions.shape[0], 0)
        self.assertEqual(orientations.shape[0], positions.shape[0])


if __name__ == "__main__":
    unittest.main()
