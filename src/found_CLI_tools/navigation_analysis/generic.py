"""Generic covariance model for horizon-based optical navigation.

This module implements the residual/Jacobian/Fisher-information path used by the
Christian generic case. The model is deliberately callback-driven: the caller
supplies ``recompute_Mc(sc)`` so the covariance can be evaluated for arbitrary
conic geometry.

THIS IS TERRIBLE ONLY FOR POINTING OUT PROBLEMS IN UNDERSTANDING THE MODEL. DO NOT USE THIS IN PRODUCTION CODE.
"""

from dataclasses import dataclass
from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from .constants import DEFAULT_FISHER_CONDITION_LIMIT

Vector = NDArray[np.float64]
Matrix = NDArray[np.float64]
VectorLike = Sequence[float] | NDArray[np.float64]


@dataclass(frozen=True)
class GenericCovarianceDiagnostics:
    """Intermediate values used to build the generic covariance."""

    position_estimate: Vector
    residuals: Vector
    residual_variances: Vector
    jacobian: Matrix
    fisher_information: Matrix
    fisher_condition_number: float
    conic_matrix: Matrix


def _as_vector3(vector: VectorLike) -> Vector:
    array = np.asarray(vector, dtype=float).reshape(-1)
    if array.shape != (3,):
        raise ValueError("Expected a 3-vector")
    return array


def _as_matrix3(matrix: NDArray[np.float64] | Sequence[Sequence[float]]) -> Matrix:
    array = np.asarray(matrix, dtype=float)
    if array.shape != (3, 3):
        raise ValueError("Expected a 3x3 matrix")
    return array


def _as_point_list(x_list: Sequence[VectorLike]) -> list[Vector]:
    points = [_as_vector3(point) for point in x_list]
    if not points:
        raise ValueError("x_list must not be empty")
    return points


def compute_residuals(
    sc: VectorLike,
    x_list: Sequence[VectorLike],
    recompute_Mc: Callable[[Vector], NDArray[np.float64] | Sequence[Sequence[float]]],
) -> tuple[Vector, Matrix]:
    """Compute residuals ``e_i = x_i^T M_c(sc) x_i`` for all horizon points."""

    sc_vec = _as_vector3(sc)
    points = _as_point_list(x_list)
    conic_matrix = _as_matrix3(recompute_Mc(sc_vec))

    residuals = np.array([float(point.T @ conic_matrix @ point) for point in points], dtype=float)
    return residuals, conic_matrix


def compute_sigma2_list(
    conic_matrix: NDArray[np.float64] | Sequence[Sequence[float]],
    x_list: Sequence[VectorLike],
    sigma_x: float,
) -> Vector:
    """Compute residual variances ``sigma_i^2 = 4 x_i^T M_c R_xbar M_c x_i``."""

    if sigma_x < 0.0:
        raise ValueError("sigma_x must be non-negative")

    points = _as_point_list(x_list)
    conic_matrix = _as_matrix3(conic_matrix)
    lifted_noise = np.zeros((3, 3), dtype=float)
    lifted_noise[:2, :2] = sigma_x**2 * np.eye(2)

    sigma2_list = np.array(
        [4.0 * float(point.T @ conic_matrix @ lifted_noise @ conic_matrix @ point) for point in points],
        dtype=float,
    )

    if np.any(sigma2_list <= 0.0):
        raise ValueError("Residual variances must be positive")

    return sigma2_list


# This is incorrect. We have sc_hat which equals the actual sc plus some noise
# delta sc. We are treating delta sc as the perturbation. The H_i falls out from
# there.
def compute_H_matrix(
    sc_hat: VectorLike,
    x_list: Sequence[VectorLike],
    recompute_Mc: Callable[[Vector], NDArray[np.float64] | Sequence[Sequence[float]]],
    eps: float = 1e-4,
) -> Matrix:
    """Compute finite-difference Jacobians ``H_i = d e_i / d s_c`` for each point."""

    if eps <= 0.0:
        raise ValueError("eps must be positive")

    sc_vec = _as_vector3(sc_hat)
    points = _as_point_list(x_list)
    jacobian = np.zeros((len(points), 3), dtype=float)

    residuals_0, _ = compute_residuals(sc_vec, points, recompute_Mc)

    for axis_index in range(3):
        perturbed = sc_vec.copy()
        perturbed[axis_index] += eps
        residuals_perturbed, _ = compute_residuals(perturbed, points, recompute_Mc)
        jacobian[:, axis_index] = (residuals_perturbed - residuals_0) / eps

    return jacobian

# We do not need recompute_Mc. The Mc does not need to be recomputed as the
# perturbations have *already* happened; that is the delta Sc term.
def compute_P_sc(
    sc_hat: VectorLike,
    x_list: Sequence[VectorLike],
    recompute_Mc: Callable[[Vector], NDArray[np.float64] | Sequence[Sequence[float]]],
    sigma_x: float,
    eps: float = 1e-4,
    *,
    return_diagnostics: bool = False,
    fisher_condition_limit: float = DEFAULT_FISHER_CONDITION_LIMIT,
) -> Matrix | tuple[Matrix, GenericCovarianceDiagnostics]:
    """Compute the generic covariance matrix using a residual-based Fisher model."""

    sc_vec = _as_vector3(sc_hat)
    points = _as_point_list(x_list)

    # Conic matrix is just M_c
    residuals, conic_matrix = compute_residuals(sc_vec, points, recompute_Mc)
    sigma2_list = compute_sigma2_list(conic_matrix, points, sigma_x)
    jacobian = compute_H_matrix(sc_vec, points, recompute_Mc, eps=eps)

    fisher_information = np.zeros((3, 3), dtype=float)
    for row_index in range(len(points)):
        row = jacobian[row_index, :].reshape(1, 3)
        fisher_information += (row.T @ row) / sigma2_list[row_index]

    fisher_condition_number = float(np.linalg.cond(fisher_information))
    if not np.isfinite(fisher_condition_number) or fisher_condition_number > fisher_condition_limit:
        raise np.linalg.LinAlgError(
            "Fisher information matrix is singular or ill-conditioned"
        )

    covariance = np.linalg.inv(fisher_information)

    diagnostics = GenericCovarianceDiagnostics(
        position_estimate=sc_vec,
        residuals=residuals,
        residual_variances=sigma2_list,
        jacobian=jacobian,
        fisher_information=fisher_information,
        fisher_condition_number=fisher_condition_number,
        conic_matrix=conic_matrix,
    )

    if return_diagnostics:
        return covariance, diagnostics
    return covariance


