"""Tests for limb.simulation.edge.conic."""

import unittest

import numpy as np

from limb.simulation.edge.conic import (
    _shape_matrix_from_axes,
    generate_camera_conic,
    generate_pixel_conic,
    _conic_matrix_to_coeffs,
    solve_general_conic,
    sample_conic_at_all_rows_columns,
    sort_points_polar_order,
    generate_edge_points,
    add_point_noise,
)
from limb.utils._camera import Camera


class TestShapeMatrixFromAxes(unittest.TestCase):
    def test_diagonal(self):
        M = _shape_matrix_from_axes([1.0, 2.0, 3.0])
        np.testing.assert_allclose(M, np.diag([1.0, 1 / 4.0, 1 / 9.0]))


class TestGenerateCameraConic(unittest.TestCase):
    def test_returns_3x3(self):
        rc = np.array([10.0, 0.0, 0.0])
        shape = np.diag([1.0, 1.0, 1.0])
        tpc = np.eye(3)
        c = generate_camera_conic(rc, shape, tpc)
        self.assertEqual(c.shape, (3, 3))
        np.testing.assert_allclose(c, c.T)


class TestGeneratePixelConic(unittest.TestCase):
    def test_normalized_leading_coeff(self):
        c = np.eye(3)
        camera = Camera(
            focal_length=0.035,
            x_pixel_pitch=5e-6,
            x_resolution=64,
            y_resolution=64,
        )
        out = generate_pixel_conic(c, camera)
        self.assertAlmostEqual(out[0, 0], 1.0)


class TestConicMatrixToCoeffs(unittest.TestCase):
    def test_shape_and_values(self):
        conic = np.array(
            [[1, 0.5, 0.2], [0.5, 2, 0.3], [0.2, 0.3, 3]], dtype=np.float64
        )
        coeffs = _conic_matrix_to_coeffs(conic)
        self.assertEqual(coeffs.shape, (6,))
        self.assertEqual(coeffs[0], 1.0)
        self.assertEqual(coeffs[1], 1.0)  # 2*0.5
        self.assertEqual(coeffs[2], 2.0)


class TestSolveGeneralConic(unittest.TestCase):
    def test_solve_y_mode_invalid_raises(self):
        with self.assertRaises(ValueError):
            solve_general_conic(1, 0, 1, 0, 0, -1, 0.5, "solve_z")

    def test_solve_y_linear(self):
        # P=0: Q*y + R = 0 -> y = -R/Q. Use 2y - 4 = 0 -> y=2, so e=-4, f=8, v0=0.
        sols = solve_general_conic(0, 0, 0, 0, -4, 8, 0.0, "solve_y")
        self.assertIsNotNone(sols)
        self.assertEqual(len(sols), 1)
        self.assertAlmostEqual(sols[0], 2.0)

    def test_solve_y_no_solution(self):
        # P*y^2 + ... with delta < 0
        sols = solve_general_conic(1, 0, 1, 0, 0, 1, 0.0, "solve_y")
        self.assertIsNone(sols)

    def test_solve_y_double_root(self):
        # (y-1)^2 = y^2 - 2y + 1 = 0: P=c=1, Q=e=-2, R=f=1 with v0=0.
        sols = solve_general_conic(0, 0, 1, 0, -2, 1, 0.0, "solve_y")
        self.assertIsNotNone(sols)
        self.assertEqual(len(sols), 1)
        self.assertAlmostEqual(sols[0], 1.0)

    def test_solve_y_two_roots(self):
        # y^2 - 1 = 0 at x=0: a*0 + d*0 + f = -1, c=1, e=0 -> y^2 - 1 = 0
        sols = solve_general_conic(0, 0, 1, 0, 0, -1, 0.0, "solve_y")
        self.assertIsNotNone(sols)
        self.assertEqual(len(sols), 2)
        self.assertAlmostEqual(sols[0], 1.0)
        self.assertAlmostEqual(sols[1], -1.0)

    def test_solve_x_mode(self):
        sols = solve_general_conic(1, 0, 0, 0, 0, -1, 0.0, "solve_x")
        self.assertIsNotNone(sols)
        self.assertEqual(len(sols), 2)

    def test_solve_y_degenerate_P_and_Q_zero(self):
        sols = solve_general_conic(0, 0, 0, 0, 0, 1, 0.5, "solve_y")
        self.assertIsNone(sols)

    def test_custom_eps(self):
        sols = solve_general_conic(1, 0, 0, 0, -2, 1, 0.0, "solve_y", eps=1e-12)
        self.assertIsNotNone(sols)


class TestSampleConicAtAllRowsColumns(unittest.TestCase):
    def test_empty_when_conic_outside(self):
        camera = Camera(
            focal_length=0.035,
            x_pixel_pitch=5e-6,
            x_resolution=16,
            y_resolution=16,
        )
        # conic that has no intersection with image (e.g. circle far away)
        conic = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1e10]], dtype=np.float64)
        rc = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        points = sample_conic_at_all_rows_columns(conic, camera, rc)
        self.assertIsInstance(points, np.ndarray)
        self.assertEqual(points.ndim, 2)
        if points.size > 0:
            self.assertEqual(points.shape[1], 2)

    def test_returns_in_bounds_points(self):
        camera = Camera(
            focal_length=0.035,
            x_pixel_pitch=5e-6,
            x_resolution=32,
            y_resolution=32,
        )
        # ellipse-like conic passing through image: x^2/100 + y^2/100 - 1 shifted to center
        # Use a conic that actually crosses the image
        conic = np.array(
            [
                [1.0, 0.0, -16.0],
                [0.0, 1.0, -16.0],
                [-16.0, -16.0, 16.0**2 + 16.0**2 - 50.0],
            ],
            dtype=np.float64,
        )
        rc = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        points = sample_conic_at_all_rows_columns(conic, camera, rc)
        if points.size > 0:
            self.assertTrue((points[:, 0] >= 0).all())
            self.assertTrue((points[:, 0] < camera.x_resolution).all())
            self.assertTrue((points[:, 1] >= 0).all())
            self.assertTrue((points[:, 1] < camera.y_resolution).all())


class TestSortPointsPolarOrder(unittest.TestCase):
    def test_empty_unchanged(self):
        pts = np.empty((0, 2))
        out = sort_points_polar_order(pts)
        self.assertEqual(out.size, 0)

    def test_single_point(self):
        pts = np.array([[1.0, 1.0]])
        out = sort_points_polar_order(pts)
        np.testing.assert_allclose(out, pts)

    def test_sorts_by_angle(self):
        pts = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.float64)
        out = sort_points_polar_order(pts)
        self.assertEqual(out.shape, (4, 2))
        angles = np.arctan2(out[:, 1], out[:, 0])
        self.assertTrue(np.all(np.diff(angles) >= -1e-9))


class TestGenerateEdgePoints(unittest.TestCase):
    def test_returns_array(self):
        pos = np.array([6800000.0, 0.0, 0.0])
        shape = np.diag([1.0 / 6378137**2, 1.0 / 6378137**2, 1.0 / 6356752**2])
        tpc = np.eye(3)
        camera = Camera(
            focal_length=0.035,
            x_pixel_pitch=5e-6,
            x_resolution=64,
            y_resolution=64,
        )
        points = generate_edge_points(pos, shape, tpc, camera)
        self.assertIsInstance(points, np.ndarray)
        self.assertEqual(points.ndim, 2)
        if points.size > 0:
            self.assertEqual(points.shape[1], 2)

    def test_with_point_noise(self):
        """Cover add_point_noise path inside generate_edge_points."""
        pos = np.array([6800000.0, 0.0, 0.0])
        shape = np.diag([1.0 / 6378137**2, 1.0 / 6378137**2, 1.0 / 6356752**2])
        tpc = np.eye(3)
        camera = Camera(
            focal_length=0.035,
            x_pixel_pitch=5e-6,
            x_resolution=64,
            y_resolution=64,
        )
        points = generate_edge_points(
            pos,
            shape,
            tpc,
            camera,
            gaussian_sigma=0.5,
            rng=np.random.default_rng(42),
        )
        self.assertIsInstance(points, np.ndarray)
        self.assertEqual(points.ndim, 2)


class TestAddPointNoise(unittest.TestCase):
    def test_empty_points(self):
        camera = Camera(
            focal_length=0.035,
            x_pixel_pitch=5e-6,
            x_resolution=64,
            y_resolution=64,
        )
        out = add_point_noise(np.empty((0, 2)), camera)
        self.assertEqual(out.shape, (0, 2))

    def test_invalid_shape_raises(self):
        camera = Camera(
            focal_length=0.035,
            x_pixel_pitch=5e-6,
            x_resolution=64,
            y_resolution=64,
        )
        with self.assertRaises(ValueError):
            add_point_noise(np.array([[1, 2, 3]]), camera)

    def test_gaussian_sigma_scalar(self):
        camera = Camera(
            focal_length=0.035,
            x_pixel_pitch=5e-6,
            x_resolution=64,
            y_resolution=64,
        )
        pts = np.array([[32.0, 32.0]])
        rng = np.random.default_rng(42)
        out = add_point_noise(pts, camera, gaussian_sigma=0.5, rng=rng)
        self.assertEqual(out.shape[0], 1)
        self.assertEqual(out.shape[1], 2)

    def test_gaussian_sigma_tuple(self):
        camera = Camera(
            focal_length=0.035,
            x_pixel_pitch=5e-6,
            x_resolution=64,
            y_resolution=64,
        )
        pts = np.array([[32.0, 32.0]])
        out = add_point_noise(
            pts, camera, gaussian_sigma=(0.1, 0.2), rng=np.random.default_rng(42)
        )
        self.assertEqual(out.shape[1], 2)

    def test_n_false_points(self):
        camera = Camera(
            focal_length=0.035,
            x_pixel_pitch=5e-6,
            x_resolution=64,
            y_resolution=64,
        )
        out = add_point_noise(
            np.empty((0, 2)), camera, n_false_points=5, rng=np.random.default_rng(42)
        )
        self.assertEqual(out.shape[0], 5)
        self.assertEqual(out.shape[1], 2)


if __name__ == "__main__":
    unittest.main()
