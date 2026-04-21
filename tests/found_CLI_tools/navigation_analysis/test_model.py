import math
import unittest

import numpy as np

from found_CLI_tools.navigation_analysis.model import (
    build_camera_frame_basis,
    build_geometry_matrix,
    compute_parametric_covariance,
)


class ParametricCovarianceModelTest(unittest.TestCase):
    def test_build_camera_frame_basis_identity_case(self):
        basis = build_camera_frame_basis([0.0, 0.0, 2.0], [0.0, 1.0, 0.0])

        np.testing.assert_allclose(basis, np.eye(3), atol=1e-12)

    def test_build_geometry_matrix_matches_excerpt(self):
        geometry_matrix, denominator_d, theta_max_rad = build_geometry_matrix(
            body_radius=1.0,
            slant_range=2.0,
            theta_max_deg=70.0,
        )

        expected_denominator_d = 0.05961876006889755
        expected_geometry_matrix = np.array(
            [
                [20.492383185831876, 0.0, 27.30005388937718],
                [0.0, 2.221391245288503, 0.0],
                [27.30005388937718, 0.0, 38.82479976410113],
            ]
        )

        self.assertAlmostEqual(theta_max_rad, math.radians(70.0))
        self.assertAlmostEqual(denominator_d, expected_denominator_d)
        np.testing.assert_allclose(geometry_matrix, expected_geometry_matrix, atol=1e-12)

    def test_compute_parametric_covariance_identity_basis(self):
        covariance, diagnostics = compute_parametric_covariance(
            s_c=[0.0, 0.0, 2.0],
            u_c=[0.0, 1.0, 0.0],
            body_radius=1.0,
            theta_max_deg=70.0,
            sigma_x=0.5,
            num_points=4,
        )

        expected_covariance = np.array(
            [
                [8.345389690705876, 0.0, 11.117769281302005],
                [0.0, 0.9046471281231287, 0.0],
                [11.117769281302005, 0.0, 15.811147037258918],
            ]
        )

        np.testing.assert_allclose(diagnostics.basis_x_to_camera, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(covariance, expected_covariance, atol=1e-12)

    def test_compute_parametric_covariance_rejects_invalid_geometry(self):
        with self.assertRaises(ValueError):
            compute_parametric_covariance(
                s_c=[0.0, 0.0, 1.0],
                u_c=[0.0, 1.0, 0.0],
                body_radius=1.0,
                theta_max_deg=70.0,
                sigma_x=0.5,
                num_points=4,
            )
