"""Tests for limb.simulation.analysis.metrics."""

import unittest

import numpy as np

from limb.simulation.analysis.metrics import apparent_radius_pixels
from limb.utils._camera import Camera


class TestApparentRadiusPixels(unittest.TestCase):
    """Test that with focal_length=1 and pixel_pitch=1, output is radius/distance."""

    def test_apparent_radius_equals_radius_over_distance_when_f_and_pixel_size_one(
        self,
    ):
        # When focal_length=1 and pixel_pitch=1, fx=1 so apparent radius in pixels
        # equals R/sqrt(d²-R²), which approximates R/d for small R/d.
        focal_length = 1.0
        pixel_pitch = 1.0
        camera = Camera(
            focal_length=focal_length,
            x_pixel_pitch=pixel_pitch,
            y_pixel_pitch=pixel_pitch,
            x_resolution=64,
            y_resolution=64,
        )
        R = 1.0
        d = 10.0
        position_camera = np.array([d, 0.0, 0.0], dtype=np.float64)

        result = apparent_radius_pixels(position_camera, R, camera.calibration_matrix)

        expected_ratio = R / d
        # Exact formula is R/sqrt(d²-R²); for d=10, R=1 that's 1/sqrt(99) ≈ 0.1005
        exact = R / np.sqrt(d * d - R * R)
        self.assertAlmostEqual(result, exact, places=10)
        # With f=p=1, result should equal radius/distance in the small-angle limit
        self.assertAlmostEqual(result, expected_ratio, delta=0.002)
