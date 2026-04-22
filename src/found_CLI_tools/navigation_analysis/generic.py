
import argparse

import numpy as np

def compute_residuals(sc, x_list, recompute_Mc):
    """
    Compute residuals e_i = x_i^T M_c(sc) x_i for all horizon points.
    """
    Mc = recompute_Mc(sc)  # 3x3 conic matrix at this sc
    e = []
    for x in x_list:
        e_i = float(x.T @ Mc @ x)
        e.append(e_i)
    return np.array(e), Mc


def compute_sigma2_list(Mc, x_list, sigma_x):
    """
    Compute residual variances sigma_i^2 = 4 x_i^T M_c R_x M_c x_i
    assuming R_x = sigma_x^2 * I_2 on the first two components.
    """
    sigma2_list = []
    R_x = sigma_x**2 * np.eye(2)

    for x in x_list:
        # Only first two components are noisy; build 3x3 R_xbar
        R_xbar = np.zeros((3, 3))
        R_xbar[:2, :2] = R_x

        sigma2_i = 4.0 * float(x.T @ Mc @ R_xbar @ Mc @ x)
        sigma2_list.append(sigma2_i)

    return np.array(sigma2_list)


def compute_H_matrix(sc_hat, x_list, recompute_Mc, eps=1e-4):
    """
    Finite-difference Jacobians H_i = d e_i / d s_c (1x3) for each point.
    Returns an array H of shape (n, 3).
    """
    sc_hat = np.asarray(sc_hat).reshape(3)
    n = len(x_list)
    H = np.zeros((n, 3))

    # Baseline residuals at sc_hat
    e0, _ = compute_residuals(sc_hat, x_list, recompute_Mc)

    for k in range(3):
        sc_pert = sc_hat.copy()
        sc_pert[k] += eps

        e_pert, _ = compute_residuals(sc_pert, x_list, recompute_Mc)

        # Finite difference derivative for each residual
        de = (e_pert - e0) / eps
        H[:, k] = de

    return H


def compute_P_sc(sc_hat, x_list, recompute_Mc, sigma_x, eps=1e-4):
    """
    Main function: compute covariance P_sc for the generic CRA OpNav case.

    Parameters
    ----------
    sc_hat : array-like, shape (3,)
        Estimated relative position vector s_c (from CRA).
    x_list : list of np.array, each shape (3,1) or (3,)
        Homogeneous image points on the horizon.
    recompute_Mc : callable
        Function recompute_Mc(sc) -> 3x3 conic matrix M_c for given s_c.
        Internally, this can use your C and A_C.
    sigma_x : float
        Standard deviation of image-plane noise (same for all points).
    eps : float
        Finite-difference step for Jacobian computation.

    Returns
    -------
    P_sc : np.ndarray, shape (3,3)
        Covariance matrix of the position estimate s_c.
    """
    sc_hat = np.asarray(sc_hat).reshape(3)

    # 1) Residuals and conic at the solution
    e0, Mc = compute_residuals(sc_hat, x_list, recompute_Mc)

    # 2) Residual variances sigma_i^2
    sigma2_list = compute_sigma2_list(Mc, x_list, sigma_x)

    # 3) Jacobians H_i = d e_i / d s_c
    H = compute_H_matrix(sc_hat, x_list, recompute_Mc, eps=eps)

    # 4) Build Fisher information matrix: I = sum_i H_i^T (1/sigma_i^2) H_i
    I = np.zeros((3, 3))
    for i in range(len(x_list)):
        Hi = H[i, :].reshape(1, 3)          # 1x3
        w = 1.0 / sigma2_list[i]
        I += w * (Hi.T @ Hi)                # 3x3

    # 5) Covariance is inverse of information
    P_sc = np.linalg.inv(I)
    return P_sc

