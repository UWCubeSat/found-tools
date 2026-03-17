"""Tests for limb.simulation.metadata.orchestrate."""

import tempfile
import unittest

import numpy as np
import pandas as pd

from limb.simulation.metadata.orchestrate import (
    initialize_sim_df,
    _fill_setup,
    _conic_from_row,
    points_from_row,
    _calculate_conic_coeffs,
    _setup_simulation,
    setup_simulation,
)
from limb.utils._camera import Camera


class TestInitializeSimDf(unittest.TestCase):
    def test_returns_empty_dataframe_with_columns(self):
        df = initialize_sim_df()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 0)
        for col in [
            "true_pos_x",
            "true_pos_y",
            "true_pos_z",
            "qx",
            "qy",
            "qz",
            "qw",
            "shape_axis_a",
            "shape_axis_b",
            "shape_axis_c",
            "cam_focal_length",
            "cam_x_resolution",
            "cam_y_resolution",
            "true_x_centroid",
            "true_y_centroid",
            "true_r_apparent",
        ]:
            self.assertIn(col, df.columns)


class TestFillSetup(unittest.TestCase):
    def test_adds_one_row(self):
        df = initialize_sim_df()
        camera = Camera(
            focal_length=0.035,
            x_pixel_pitch=5e-6,
            x_resolution=64,
            y_resolution=64,
        )
        sat_position = np.array([6800000.0, 0.0, 0.0])
        semi_axes = np.array([6378137.0, 6378137.0, 6356752.0])
        quat = np.array([0.0, 0.0, 0.0, 1.0])  # identity
        out = _fill_setup(df, camera, sat_position, semi_axes, quat)
        self.assertEqual(len(out), 1)
        self.assertEqual(out["true_pos_x"].iloc[0], 6800000.0)
        self.assertEqual(out["cam_x_resolution"].iloc[0], 64)


class TestConicFromRow(unittest.TestCase):
    def test_conic_from_row_shape(self):
        df = initialize_sim_df()
        camera = Camera(
            focal_length=0.035,
            x_pixel_pitch=5e-6,
            x_resolution=64,
            y_resolution=64,
        )
        row = _fill_setup(
            df,
            camera,
            np.array([6800000.0, 0.0, 0.0]),
            np.array([6378137.0, 6378137.0, 6356752.0]),
            np.array([0.0, 0.0, 0.0, 1.0]),
        ).iloc[0]
        conic = _conic_from_row(row)
        self.assertEqual(conic.shape, (3, 3))


class TestPointsFromRow(unittest.TestCase):
    def test_points_from_row_returns_array(self):
        df = initialize_sim_df()
        camera = Camera(
            focal_length=0.035,
            x_pixel_pitch=5e-6,
            x_resolution=64,
            y_resolution=64,
        )
        row = _fill_setup(
            df,
            camera,
            np.array([6800000.0, 0.0, 0.0]),
            np.array([6378137.0, 6378137.0, 6356752.0]),
            np.array([0.0, 0.0, 0.0, 1.0]),
        ).iloc[0]
        points = points_from_row(row)
        self.assertIsInstance(points, np.ndarray)
        self.assertEqual(points.ndim, 2)
        if points.size > 0:
            self.assertEqual(points.shape[1], 2)

    def test_points_from_row_truncate(self):
        """Truncate controls decimal places of returned point coordinates."""
        df = initialize_sim_df()
        camera = Camera(
            focal_length=0.035,
            x_pixel_pitch=5e-6,
            x_resolution=64,
            y_resolution=64,
        )
        row = _fill_setup(
            df,
            camera,
            np.array([6800000.0, 0.0, 0.0]),
            np.array([6378137.0, 6378137.0, 6356752.0]),
            np.array([0.0, 0.0, 0.0, 1.0]),
        ).iloc[0]
        points_0 = points_from_row(row, truncate=0)
        if points_0.size > 0:
            self.assertTrue(
                np.all(points_0 == np.round(points_0)),
                msg="truncate=0 should yield integer pixel coordinates",
            )
        points_2 = points_from_row(row, truncate=2)
        if points_2.size > 0:
            self.assertTrue(
                np.allclose(points_2, np.round(points_2, 2)),
                msg="truncate=2 should yield coordinates with at most 2 decimal places",
            )


class TestCalculateConicCoeffs(unittest.TestCase):
    def test_from_dataframe(self):
        df = initialize_sim_df()
        camera = Camera(
            focal_length=0.035,
            x_pixel_pitch=5e-6,
            x_resolution=64,
            y_resolution=64,
        )
        for _ in range(2):
            df = _fill_setup(
                df,
                camera,
                np.array([6800000.0, 0.0, 0.0]),
                np.array([6378137.0, 6378137.0, 6356752.0]),
                np.array([0.0, 0.0, 0.0, 1.0]),
            )
        result = _calculate_conic_coeffs(df)
        self.assertIsInstance(result, dict)
        self.assertIn((64, 64), result)
        row_indices, coeffs, K, rc = result[(64, 64)]
        self.assertEqual(len(row_indices), 2)
        self.assertEqual(coeffs.shape[0], 2)
        self.assertEqual(coeffs.shape[1], 6)
        self.assertEqual(K.shape[0], 2)
        self.assertEqual(K.shape[1], 3)
        self.assertEqual(K.shape[2], 3)
        self.assertEqual(rc.shape[0], 2)
        self.assertEqual(rc.shape[1], 3)

    def test_from_csv_path(self):
        df = initialize_sim_df()
        camera = Camera(
            focal_length=0.035,
            x_pixel_pitch=5e-6,
            x_resolution=32,
            y_resolution=32,
        )
        df = _fill_setup(
            df,
            camera,
            np.array([6800000.0, 0.0, 0.0]),
            np.array([6378137.0, 6378137.0, 6356752.0]),
            np.array([0.0, 0.0, 0.0, 1.0]),
        )
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            df.to_csv(path, index=True)
            result = _calculate_conic_coeffs(path)
            self.assertIn((32, 32), result)
            row_indices, coeffs, K, rc = result[(32, 32)]
            self.assertEqual(len(row_indices), 1)
            self.assertEqual(coeffs.shape, (1, 6))
        finally:
            import os

            os.unlink(path)


class TestSetupSimulation(unittest.TestCase):
    def test_setup_simulation_writes_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            import os

            path = os.path.join(tmpdir, "sim.csv")
            setup_simulation(
                semi_axes=[6378137.0, 6378137.0, 6356752.0],
                fovs=[70.0],
                resolutions=[32],
                distances=[6800000.0],
                num_earth_points=1,
                num_positions_per_point=2,
                num_spins_per_position=2,
                num_radials_per_spin=2,
                output_path=path,
            )
            self.assertTrue(os.path.isfile(path))
            df = pd.read_csv(path, index_col=0)
            self.assertGreater(len(df), 0)

    def test_setup_simulation_in_memory(self):
        df = _setup_simulation(
            semi_axes=[6378137.0, 6378137.0, 6356752.0],
            fovs=[70.0],
            resolutions=[32],
            distances=[6800000.0],
            num_earth_points=1,
            num_positions_per_point=2,
            num_spins_per_position=2,
            num_radials_per_spin=2,
        )
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        self.assertIn("true_pos_x", df.columns)


if __name__ == "__main__":
    unittest.main()
