"""Parametric covariance model from the Christian OpNav excerpt."""

from dataclasses import dataclass
from math import radians

import numpy as np
from numpy.typing import NDArray

Vector3 = NDArray[np.float64]
Matrix3 = NDArray[np.float64]


@dataclass(frozen=True)
class ParametricCovarianceDiagnostics:
    """Intermediate values used to construct the covariance matrix."""

    slant_range: float
    body_radius: float
    theta_max_deg: float
    theta_max_rad: float
    denominator_d: float
    scale_factor: float
    basis_x_to_camera: Matrix3
    geometry_matrix: Matrix3
    camera_frame_covariance: Matrix3

# Converts a list or tuple to a numpy array and checks that it's a 3-vector
def _as_vector3(vector: NDArray[np.float64] | list[float] | tuple[float, ...]) -> Vector3:
    array = np.asarray(vector, dtype=float)
    if array.shape != (3,):
        raise ValueError("Expected a 3-vector")
    return array

# Custom normalization function that raises an error for zero vectors
def _normalize(vector: Vector3, label: str) -> Vector3:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError(f"{label} must be non-zero")
    return vector / norm


def build_camera_frame_basis(s_c: NDArray[np.float64] | list[float] | tuple[float, ...], u_c: NDArray[np.float64] | list[float] | tuple[float, ...]) -> Matrix3:
    """Build the camera-frame basis matrix T_C^X from the excerpt."""

    position_vector = _as_vector3(s_c)
    sunlight_vector = _as_vector3(u_c)

    # computing e1, e2, e3 according to the excerpt
    # e3 is the camera boresight
    # e1 is the cross product of e3 & sun vector
    # e2 is the cross product of e3 & e1
    e3 = _normalize(position_vector, "s_c")
    unit_sunlight = _normalize(sunlight_vector, "u_c")

    e1 = _normalize(np.cross(unit_sunlight, e3), "u_c x e3")
    e2 = _normalize(np.cross(e3, e1), "e3 x e1")

    return np.column_stack((e1, e2, e3))


def build_geometry_matrix(
    body_radius: float,
    slant_range: float,
    theta_max_deg: float,
) -> tuple[Matrix3, float, float]:
    """Build the symmetric geometry matrix from Eq. 9.168 and D from Eq. 9.169."""

    if body_radius <= 0.0:
        raise ValueError("body_radius must be positive")
    if slant_range <= body_radius:
        raise ValueError("slant_range must be greater than body_radius")
    if theta_max_deg <= 0.0:
        raise ValueError("theta_max_deg must be positive")

    theta_max_rad = radians(theta_max_deg)
    delta = slant_range**2 - body_radius**2
    denominator_d = (
        theta_max_rad / 4.0 * (2.0 * theta_max_rad + np.sin(2.0 * theta_max_rad))
        - np.sin(theta_max_rad) ** 2
    )
    if denominator_d == 0.0:
        raise ValueError("Degenerate geometry denominator")

    # Represents the covariance between the x and z directions in the camera frame.
    off_diagonal = np.sqrt(delta) / (denominator_d * body_radius) * np.sin(theta_max_rad)
    
    # The geometry matrix is symmetric with the form:
    # [ A, 0, B ]
    # [ 0, C, 0 ]
    # [ B, 0, D ]
    # A, C, D are the variances of each camera frame axis (x, y, z)
    geometry_matrix = np.array(
        [
            [theta_max_rad / denominator_d, 0.0, off_diagonal],
            [0.0, 4.0 / (2.0 * theta_max_rad - np.sin(2.0 * theta_max_rad)), 0.0],
            [
                off_diagonal,
                0.0,
                delta * (2.0 * theta_max_rad + np.sin(2.0 * theta_max_rad))
                / (4.0 * denominator_d * body_radius**2),
            ],
        ],
        dtype=float,
    )
    return geometry_matrix, denominator_d, theta_max_rad


def compute_parametric_covariance(
    s_c: NDArray[np.float64] | list[float] | tuple[float, ...],
    u_c: NDArray[np.float64] | list[float] | tuple[float, ...],
    body_radius: float,
    theta_max_deg: float,
    sigma_x: float,
    num_points: int,
) -> tuple[Matrix3, ParametricCovarianceDiagnostics]:
    """Compute the parametric covariance matrix from the Christian textbook model."""

    if sigma_x < 0.0:
        raise ValueError("sigma_x must be non-negative")
    if num_points < 1:
        raise ValueError("num_points must be at least 1")

    position_vector = _as_vector3(s_c)
    sunlight_vector = _as_vector3(u_c)
    slant_range = float(np.linalg.norm(position_vector))

    basis_x_to_camera = build_camera_frame_basis(position_vector, sunlight_vector)
    geometry_matrix, denominator_d, theta_max_rad = build_geometry_matrix(
        body_radius=body_radius,
        slant_range=slant_range,
        theta_max_deg=theta_max_deg,
    )
    
    delta = slant_range**2 - body_radius**2
    scale_factor = sigma_x**2 * slant_range**4 * theta_max_rad / (num_points * delta)
    camera_frame_covariance = scale_factor * geometry_matrix
    covariance = basis_x_to_camera @ camera_frame_covariance @ basis_x_to_camera.T

    # Intermediate values to help with debugging
    diagnostics = ParametricCovarianceDiagnostics(
        slant_range=slant_range,
        body_radius=body_radius,
        theta_max_deg=theta_max_deg,
        theta_max_rad=theta_max_rad,
        denominator_d=denominator_d,
        scale_factor=scale_factor,
        basis_x_to_camera=basis_x_to_camera,
        geometry_matrix=geometry_matrix,
        camera_frame_covariance=camera_frame_covariance,
    )
    return covariance, diagnostics
