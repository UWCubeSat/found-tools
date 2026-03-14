import unittest

import numpy as np

from limb.simulation.metadata.state import generateSatelliteState
from limb.utils._camera import Camera


class StateTest(unittest.TestCase):
    def test_generate_satellite_state_with_zero_edge_angle(self):
        shape_matrix = np.diag([1.0 / 100.0, 1.0 / 100.0, 1.0 / 100.0])
        earth_direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        distance = 15.0

        camera = Camera(
            focalLength=0.035,
            xPixelPitch=5e-6,
            xResolution=1280,
            yResolution=1024,
            edgeAngleMode="zero",
        )

        num_positions = 4
        num_orientations = 3
        positions, orientations = generateSatelliteState(
            shape_matrix,
            earth_direction,
            distance,
            camera,
            num_positions,
            num_orientations,
        )

        self.assertEqual((num_positions, 3), positions.shape)
        self.assertEqual((num_orientations, num_positions, 3, 3), orientations.shape)
        self.assertTrue(np.isfinite(positions).all())
        self.assertTrue(np.isfinite(orientations).all())

        # Each orientation should be a valid rotation matrix.
        for r in orientations.reshape(-1, 3, 3):
            self.assertTrue(np.allclose(r @ r.T, np.eye(3), atol=1e-6))

    def test_generate_satellite_state_with_wgs84_earth_geometry(self):
        # WGS84 ellipsoid semi-axes
        a, b, c = 6378137.0, 6378137.0, 6356752.31424518
        shape_matrix = np.diag([1.0 / (a * a), 1.0 / (b * b), 1.0 / (c * c)])
        earth_direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        distance = 6800000.0

        camera = Camera(
            focalLength=0.035,
            xPixelPitch=5e-6,
            xResolution=1280,
            yResolution=1024,
            edgeOffset=80,
            edgeAngleMode="zero",
        )

        num_positions = 4
        num_orientations = 2
        positions, orientations = generateSatelliteState(
            shape_matrix,
            earth_direction,
            distance,
            camera,
            num_positions,
            num_orientations,
        )

        self.assertEqual((num_positions, 3), positions.shape)
        self.assertEqual((num_orientations, num_positions, 3, 3), orientations.shape)
        self.assertTrue(np.isfinite(positions).all())
        self.assertTrue(np.isfinite(orientations).all())

        # Each orientation should be a valid rotation matrix.
        for r in orientations.reshape(-1, 3, 3):
            self.assertTrue(np.allclose(r @ r.T, np.eye(3), atol=1e-6))

        # All satellite positions should be at approximately the same distance from origin.
        distances = np.linalg.norm(positions, axis=1)
        self.assertTrue(np.allclose(distances, distance, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
