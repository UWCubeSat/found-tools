"""Conic projection

Provides functions to build and sample the projected ellipsoid (horizon)
conic in pixel coordinates, given a camera pose and an ellipsoid model.
"""

import numpy as np

from found_tools.utils._camera import Camera


def _shape_matrix_from_axes(semi_axes: list[float]) -> np.ndarray:
    a, b, c = semi_axes
    return np.diag([1.0 / (a * a), 1.0 / (b * b), 1.0 / (c * c)])


def generate_camera_conic(
    rc: np.ndarray,
    shape_matrix: np.ndarray,
    tpc: np.ndarray,
) -> np.ndarray:
    """Build the conic locus matrix in camera coordinates.

    Transforms the world-frame ellipsoid into the camera frame and constructs
    the 3×3 symmetric matrix C representing the horizon conic seen from *rc*.

    Args:
        rc:          Camera position vector in camera coordinates, shape (3,).
        shape_matrix: Ellipsoid defining matrix in world coordinates,
                     shape (3, 3). Diagonal entries are 1/a² for each
                     semi-axis.
        orientation: Rotation matrix transforming world coordinates to camera
                     coordinates (TPC), shape (3, 3).

    Returns:
        C: 3×3 symmetric NumPy array representing the conic locus in camera
           coordinates.
    """
    tcp = tpc.T
    ac = tpc @ shape_matrix @ tcp
    c = ac @ np.outer(rc, rc) @ ac - (rc @ ac @ rc * np.eye(3) - np.eye(3)) @ ac
    return c


def generate_pixel_conic(c: np.ndarray, camera: Camera) -> np.ndarray:
    """Project the camera-space conic into pixel coordinates.

    Applies the inverse intrinsics transform K⁻¹ so the resulting matrix
    represents the conic directly in pixel coordinates::

        C[0,0]x² + 2C[0,1]xy + C[1,1]y²
            + 2C[0,2]x + 2C[1,2]y + C[2,2] = 0

    The matrix is normalised so that the leading coefficient equals 1.

    Args:
        c:    3×3 symmetric conic matrix in camera (metric) coordinates.
        camera: :class:`~found_CLI_tools.cameraGeometry.Camera` object
                supplying intrinsics and resolution.

    Returns:
        calibratedC: 3×3 symmetric conic matrix in pixel coordinates,
                     normalised so calibratedC[0, 0] == 1.
    """
    k_inv = camera.inverse_calibration_matrix
    calibrated_c = k_inv.T @ c @ k_inv
    return calibrated_c / calibrated_c[0, 0]


def conic_matrix_to_coeffs(conic: np.ndarray) -> np.ndarray:
    # Conic matrix C corresponds to Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0.
    return np.array(
        [
            conic[0, 0],
            2.0 * conic[0, 1],
            conic[1, 1],
            2.0 * conic[0, 2],
            2.0 * conic[1, 2],
            conic[2, 2],
        ],
        dtype=np.float64,
    )
