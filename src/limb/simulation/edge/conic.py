"""Conic projection and sampling for horizon edge detection.

Provides functions to build and sample the projected ellipsoid (horizon)
conic in pixel coordinates, given a camera pose and an ellipsoid model.

Typical usage::

    import numpy as np
    from found_CLI_tools.cameraGeometry import Camera, sphericalToRotationDegrees
    from found_CLI_tools.generatePoints import generateEdgePoints

    Ap = np.diag([1/6378.1366**2, 1/6378.1366**2, 1/6356.7519**2])
    cam = Camera(focal_length, pixel_size, x_res, y_res)
    TPC = sphericalToRotationDegrees(ra, dec, roll)
    rc = TPC @ rp                  # world position → camera frame
    points = generateEdgePoints(rc, Ap, TPC, cam)
"""

import numpy as np

from limb.utils._camera import Camera

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


def _conic_matrix_to_coeffs(conic: np.ndarray) -> np.ndarray:
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
        dtype=np.float32,
    )


def solve_general_conic(
    a: float,
    b: float,
    c: float,
    d: float,
    e: float,
    f: float,
    v0: float,
    mode: str,
    *,
    eps: float | None = None,
) -> tuple[float, ...] | None:
    """Solve ax²+bxy+cy²+dx+ey+f = 0 for the unknown variable given v0.

    Algorithm: SolveGeneralConic — classify and solve quadratic in one variable.

    Args:
        a, b, c, d, e, f: Conic coefficients (ax²+bxy+cy²+dx+ey+f = 0).
        v0: Known value (x when mode='solve_y', y when mode='solve_x').
        mode: 'solve_y' to solve for y given x=v0; 'solve_x' to solve for x given y=v0.
        eps: Tolerance for near-zero (default: 1e-10).

    Returns:
        Tuple of 0, 1, or 2 real values for the unknown; None if no real solution
        or degenerate (e.g. whole line satisfies).
    """
    if eps is None:
        eps = 1e-10

    if mode == "solve_y":
        P, Q, R = c, b * v0 + e, a * v0**2 + d * v0 + f
    elif mode == "solve_x":
        P, Q, R = a, b * v0 + d, c * v0**2 + e * v0 + f
    else:
        raise ValueError("mode must be 'solve_y' or 'solve_x'")

    if abs(P) < eps:
        if abs(Q) < eps:
            return None  # 0=0 (ALL) or 0=R≠0 (NONE) — no finite set to return
        return (-R / Q,)
    delta = Q * Q - 4 * P * R
    if delta < -eps:
        return None
    if abs(delta) <= eps:
        return (-Q / (2 * P),)
    sqrt_d = np.sqrt(delta)
    return ((-Q + sqrt_d) / (2 * P), (-Q - sqrt_d) / (2 * P))


def sample_conic_at_all_rows_columns(conic: np.ndarray, camera: Camera) -> np.ndarray:
    """Solve conic at every row and column (step 1); return in-image points.

    Steps x from 0 to x_resolution - 1 and, for each x, solves for y.
    Steps y from 0 to y_resolution - 1 and, for each y, solves for x.
    Only points with 0 ≤ x < x_resolution and 0 ≤ y < y_resolution are returned.

    Args:
        conic: 3×3 symmetric conic matrix in pixel coordinates.
        camera: Camera supplying x_resolution and y_resolution.

    Returns:
        points: (N, 2) array of (x, y) pixel coordinates on the conic inside the image.
    """
    a, b, c, d, e, f = _conic_matrix_to_coeffs(conic)
    points: list[list[float]] = []
    for x in range(camera.x_resolution):
        sols = solve_general_conic(a, b, c, d, e, f, float(x), "solve_y")
        if sols is not None:
            for y in sols:
                if 0 <= y < camera.y_resolution:
                    points.append([float(x), float(y)])
    for y in range(camera.y_resolution):
        sols = solve_general_conic(a, b, c, d, e, f, float(y), "solve_x")
        if sols is not None:
            for x in sols:
                if 0 <= x < camera.x_resolution:
                    points.append([float(x), float(y)])
    return np.array(points, dtype=np.float64) if points else np.empty((0, 2), dtype=np.float64)


def generate_edge_points(
    pos: np.ndarray,
    shape_matrix: np.ndarray,
    orientation: np.ndarray,
    cam: Camera,
) -> np.ndarray:
    """Generate points on the projected horizon ellipse in pixel coordinates.

    Builds the conic locus from the given pose, projects it into pixel
    coordinates, then solves at every row and column (step 1) using the
    general conic algorithm. Only points inside the image
    (0 ≤ x < x_resolution, 0 ≤ y < y_resolution) are returned.

    Args:
        pos:         Satellite position in camera coordinates, shape (3,).
        shape_matrix: Ellipsoid defining matrix in world coordinates,
                     shape (3, 3). Diagonal entries are 1/a² for each
                     semi-axis.
        orientation: Rotation matrix from world to camera coordinates
                     (TPC), shape (3, 3).
        cam:         Camera object supplying intrinsics and resolution.

    Returns:
        points: NumPy array of (x, y) pixel coordinates on the conic that
            lie within the image, shape (M, 2).
    """
    cam_conic = generate_camera_conic(pos, shape_matrix, orientation)
    pixel_conic = generate_pixel_conic(cam_conic, cam)
    return sample_conic_at_all_rows_columns(pixel_conic, cam)


# Backward-compatible aliases for previous camelCase API names.
generateCameraConic = generate_camera_conic
generatePixelConic = generate_pixel_conic
solveGeneralConic = solve_general_conic
sampleConicAtAllRowsColumns = sample_conic_at_all_rows_columns
generateEdgePoints = generate_edge_points
