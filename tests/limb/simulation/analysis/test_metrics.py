"""Tests for limb.simulation.analysis.metrics."""

import unittest

import numpy as np
import pandas as pd

from limb.simulation.analysis.metrics import apparent_radius_pixels, fill_pixel_metrics
from limb.simulation.metadata.orchestrate import _fill_setup, initialize_sim_df
from limb.utils._camera import Camera


class TestFillPixelMetrics(unittest.TestCase):
    """Test that fill_pixel_metrics fills true_* columns correctly."""

    def test_fills_true_centroid_and_apparent_radius(self):
        df = initialize_sim_df()
        camera = Camera(
            focal_length=0.035,
            x_pixel_pitch=5e-6,
            x_resolution=64,
            y_resolution=64,
        )
        sat_position = np.array([6800000.0, 0.0, 0.0])
        semi_axes = np.array([6378137.0, 6378137.0, 6356752.0])
        quat = np.array([0.0, 0.0, 0.0, 1.0])
        # df = _fill_setup(df, camera, sat_position, semi_axes, quat)
        df = pd.read_csv("/Users/jlando/Documents/GitHub/found-tools/sim_metadata.csv", index_col=0)

        result = fill_pixel_metrics(df)

        self.assertEqual(len(result), 1)
        row = result.iloc[0]
        self.assertFalse(
            np.isnan(row["true_x_centroid"]),
            "true_x_centroid should be filled",
        )
        self.assertFalse(
            np.isnan(row["true_y_centroid"]),
            "true_y_centroid should be filled",
        )
        self.assertFalse(
            np.isnan(row["true_r_apparent"]),
            "true_r_apparent should be filled",
        )
        self.assertGreater(
            row["true_r_apparent"], 0, "true_r_apparent should be positive"
        )
        self.assertGreaterEqual(
            row["true_x_centroid"],
            0,
            "true_x_centroid should be within image",
        )
        self.assertLess(
            row["true_x_centroid"],
            camera.x_resolution,
            "true_x_centroid should be within image",
        )
        self.assertGreaterEqual(
            row["true_y_centroid"],
            0,
            "true_y_centroid should be within image",
        )
        self.assertLess(
            row["true_y_centroid"],
            camera.y_resolution,
            "true_y_centroid should be within image",
        )
        # Identity quat, sat on +x: centroid should be at principal point (32, 32)
        self.assertAlmostEqual(row["true_x_centroid"], camera.x_center, delta=0.5)
        self.assertAlmostEqual(row["true_y_centroid"], camera.y_center, delta=0.5)

    def test_returns_copy_does_not_mutate_input(self):
        df = initialize_sim_df()
        df = _fill_setup(
            df,
            Camera(
                focal_length=0.035, x_pixel_pitch=5e-6, x_resolution=64, y_resolution=64
            ),
            np.array([6800000.0, 0.0, 0.0]),
            np.array([6378137.0, 6378137.0, 6356752.0]),
            np.array([0.0, 0.0, 0.0, 1.0]),
        )
        result = fill_pixel_metrics(df)

        self.assertIsNot(result, df)
        self.assertTrue(
            np.isnan(df["true_x_centroid"].iloc[0]), "input df should be unchanged"
        )
        self.assertFalse(
            np.isnan(result["true_x_centroid"].iloc[0]),
            "result should have filled metrics",
        )


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
