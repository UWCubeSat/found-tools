import unittest

import numpy as np

from found_CLI_tools.navigation_analysis.generic import (
    compute_H_matrix,
    compute_P_sc,
    compute_residuals,
    compute_sigma2_list,
)


def recompute_mc(sc):
    s0, s1, s2 = np.asarray(sc, dtype=float)
    return np.array(
        [
            [2.0 + 0.7 * s0, 0.2 * s1, 0.1 * s2],
            [0.2 * s1, 3.0 + 0.5 * s1, 0.15 * s0],
            [0.1 * s2, 0.15 * s0, 4.0 + 0.4 * s2],
        ],
        dtype=float,
    )


class GenericCovarianceTest(unittest.TestCase):
    def setUp(self):
        self.sc_hat = np.array([1.0, 2.0, 3.0])
        self.x_list = [
            np.array([1.0, 0.0, 1.0]),
            np.array([0.0, 1.0, 1.0]),
            np.array([1.0, 1.0, 1.0]),
            np.array([1.0, -1.0, 1.0]),
        ]

    def test_residuals_are_computed(self):
        residuals, conic_matrix = compute_residuals(self.sc_hat, self.x_list, recompute_mc)

        self.assertEqual(residuals.shape, (len(self.x_list),))
        self.assertEqual(conic_matrix.shape, (3, 3))
        self.assertTrue(np.all(np.isfinite(residuals)))

    def test_sigma2_list_is_positive(self):
        _, conic_matrix = compute_residuals(self.sc_hat, self.x_list, recompute_mc)

        sigma2_list = compute_sigma2_list(conic_matrix, self.x_list, sigma_x=0.5)

        self.assertEqual(sigma2_list.shape, (len(self.x_list),))
        self.assertTrue(np.all(sigma2_list > 0.0))

    def test_h_matrix_has_expected_shape(self):
        h_matrix = compute_H_matrix(self.sc_hat, self.x_list, recompute_mc, eps=1e-5)

        self.assertEqual(h_matrix.shape, (len(self.x_list), 3))
        self.assertTrue(np.any(np.abs(h_matrix) > 0.0))

    def test_covariance_and_diagnostics_return(self):
        covariance, diagnostics = compute_P_sc(
            self.sc_hat,
            self.x_list,
            recompute_mc,
            sigma_x=0.5,
            eps=1e-5,
            return_diagnostics=True,
        )

        self.assertEqual(covariance.shape, (3, 3))
        self.assertTrue(np.allclose(covariance, covariance.T))
        self.assertEqual(diagnostics.jacobian.shape, (len(self.x_list), 3))
        self.assertEqual(diagnostics.fisher_information.shape, (3, 3))
        self.assertTrue(np.isfinite(diagnostics.fisher_condition_number))

    def test_singular_fisher_raises(self):
        def degenerate_mc(sc):
            sc = np.asarray(sc, dtype=float)
            return np.array(
                [
                    [1.0 + sc[0], 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=float,
            )

        with self.assertRaises(np.linalg.LinAlgError):
            compute_P_sc(self.sc_hat, self.x_list, degenerate_mc, sigma_x=0.5, eps=1e-5)
