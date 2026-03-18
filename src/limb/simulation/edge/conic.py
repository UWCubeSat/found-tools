"""Conic projection and sampling for horizon edge detection.

Provides functions to build and sample the projected ellipsoid (horizon)
conic in pixel coordinates, given a camera pose and an ellipsoid model.

Typical usage::

    import numpy as np
    from limb.utils._camera import Camera
    from limb.simulation.edge.conic import generate_edge_points, generate_camera_conic

    Ap = np.diag([1/6378.1366**2, 1/6378.1366**2, 1/6356.7519**2])
    cam = Camera(focal_length, pixel_size, x_res, y_res)
    rc = TPC @ rp                  # world position → camera frame
    points = generate_edge_points(rc, Ap, TPC, cam)
"""

from typing import Optional

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


def _point_on_visible_arc(
    x: float, y: float, rc: np.ndarray, k_inv: np.ndarray
) -> bool:
    """True if the ray through pixel (x, y) points away from Earth (sky side of limb).

    Matches render logic: keep points where dot(ray, rc) <= 0.
    """
    ray = k_inv @ np.array([x, y, 1.0], dtype=np.float64)
    return float(np.dot(ray, -rc)) >= 0.0


def sample_conic_at_all_rows_columns(
    conic: np.ndarray, camera: Camera, rc: np.ndarray
) -> np.ndarray:
    """Solve conic at every row and column (step 1); return in-image points on the visible arc.

    Steps x from 0.5 to x_resolution - 0.5 and y from 0.5 to y_resolution - 0.5 (pixel
    centers), step 1. For each x, solves for y; for each y, solves for x.
    Only points with 0 ≤ x < x_resolution and 0 ≤ y < y_resolution are returned.
    Points are filtered to the visible horizon arc (sky side): same as render, only
    points where the pixel ray dot rc <= 0 are kept.

    Args:
        conic: 3×3 symmetric conic matrix in pixel coordinates.
        camera: Camera supplying x_resolution, y_resolution, and inverse calibration.
        rc: Camera position in camera frame (vector to Earth), shape (3,).

    Returns:
        points: (N, 2) array of (x, y) pixel coordinates on the conic inside the image.
    """
    a, b, c, d, e, f = _conic_matrix_to_coeffs(conic)
    k_inv = camera.inverse_calibration_matrix
    points: list[list[float]] = []
    for x in np.arange(0.5, camera.x_resolution, 1.0):
        sols = solve_general_conic(a, b, c, d, e, f, float(x), "solve_y")
        if sols is not None:
            for y in sols:
                if 0 <= y < camera.y_resolution and _point_on_visible_arc(
                    float(x), float(y), rc, k_inv
                ):
                    points.append([float(x), float(y)])
    for y in np.arange(0.5, camera.y_resolution, 1.0):
        sols = solve_general_conic(a, b, c, d, e, f, float(y), "solve_x")
        if sols is not None:
            for x in sols:
                if 0 <= x < camera.x_resolution and _point_on_visible_arc(
                    float(x), float(y), rc, k_inv
                ):
                    points.append([float(x), float(y)])
    return (
        np.array(points, dtype=np.float64)
        if points
        else np.empty((0, 2), dtype=np.float64)
    )


def sort_points_polar_order(points: np.ndarray) -> np.ndarray:
    """Sort (N, 2) pixel points by angle around their centroid for drawing a closed contour.

    Args:
        points: (N, 2) array of (x, y) coordinates.

    Returns:
        Points sorted by angle (radians) from centroid, shape (N, 2).
        Returns a copy if N > 0, or the same empty array if N == 0.
    """
    if points.size == 0:
        return points
    cx = float(np.mean(points[:, 0]))
    cy = float(np.mean(points[:, 1]))
    angles = np.arctan2(points[:, 1] - cy, points[:, 0] - cx)
    order = np.argsort(angles)
    return points[order].copy()


def generate_edge_points(
    pos: np.ndarray,
    shape_matrix: np.ndarray,
    orientation: np.ndarray,
    cam: Camera,
    gaussian_sigma: Optional[float | tuple[float, float]] = None,
    n_false_points: int = 0,
    truncate: int = 0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generate points on the projected horizon ellipse in pixel coordinates.

    Builds the conic locus from the given pose, projects it into pixel
    coordinates, then solves at every row and column (step 1) using the
    general conic algorithm. Only points inside the image
    (0 ≤ x < x_resolution, 0 ≤ y < y_resolution) are returned.

    Optionally applies point noise and/or adds false points; both can be
    used at the same time (noisy edge points first, then false points).

    Args:
        pos:         Satellite position in camera coordinates, shape (3,).
        shape_matrix: Ellipsoid defining matrix in world coordinates,
                     shape (3, 3). Diagonal entries are 1/a² for each
                     semi-axis.
        orientation: Rotation matrix from world to camera coordinates
                     (TPC), shape (3, 3).
        cam:         Camera object supplying intrinsics and resolution.
        gaussian_sigma: If set, add zero-mean Gaussian noise to each edge
            point. Use a float for same sigma in x and y, or (sigma_x, sigma_y).
        n_false_points: Number of extra points to add uniformly at random
            inside the image.
        truncate: Number of decimal places for point coordinates (e.g. 0 for
            integer pixels, 2 for two decimals). Applied to all returned points.
        rng: Random generator for reproducibility. If None, uses default.

    Returns:
        points: NumPy array of (x, y) pixel coordinates on the conic that
            lie within the image, shape (M, 2). If noise/false points are
            used, order is perturbed edge points (in bounds) then false points.
    """
    cam_conic = generate_camera_conic(pos, shape_matrix, orientation)
    pixel_conic = generate_pixel_conic(cam_conic, cam)
    points = sample_conic_at_all_rows_columns(pixel_conic, cam, pos)
    points = sort_points_polar_order(points)
    if gaussian_sigma is not None or n_false_points > 0:
        points = add_point_noise(
            points,
            cam,
            gaussian_sigma=gaussian_sigma,
            n_false_points=n_false_points,
            truncate=truncate,
            rng=rng,
        )
    else:
        points = np.round(points, decimals=truncate)
    return points


def add_point_noise(
    points: np.ndarray,
    camera: Camera,
    gaussian_sigma: Optional[float | tuple[float, float]] = None,
    n_false_points: int = 0,
    truncate: int = 0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Add Gaussian noise in x/y and/or random false points; return only in-image points.

    You can use gaussian_sigma and n_false_points at the same time: noisy edge
    points (that stay in bounds) are returned first, then false points.
    Each input point can be perturbed by independent Gaussian noise. Optionally,
    additional points are drawn uniformly at random in the image. Any point
    (original+noise or false) that falls outside the image is dropped.

    Args:
        points: (N, 2) array of (x, y) pixel coordinates.
        camera: Camera supplying x_resolution and y_resolution (image bounds).
        gaussian_sigma: If set, add zero-mean Gaussian noise to each point.
            Use a float for the same sigma in x and y, or (sigma_x, sigma_y).
        n_false_points: Number of extra points to add uniformly at random
            inside the image.
        truncate: Number of decimal places for point coordinates (e.g. 0 for
            integer pixels, 2 for two decimals). Applied to all returned points.
        rng: Random generator for reproducibility. If None, uses default generator.

    Returns:
        (M, 2) array of (x, y) coordinates; all rows satisfy
        0 <= x < width and 0 <= y < height. Order: perturbed true points
        (that remain in bounds) first, then false points.
    """
    width = camera.x_resolution
    height = camera.y_resolution
    rng = rng if rng is not None else np.random.default_rng()

    out_list: list[np.ndarray] = []

    if points.size > 0:
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError("points must have shape (N, 2)")
        if gaussian_sigma is not None:
            if isinstance(gaussian_sigma, (int, float)):
                sx, sy = float(gaussian_sigma), float(gaussian_sigma)
            else:
                sx, sy = float(gaussian_sigma[0]), float(gaussian_sigma[1])
            noise = rng.normal(0, (sx, sy), size=pts.shape)
            pts = pts + noise
        in_bounds = (
            (pts[:, 0] >= 0)
            & (pts[:, 0] < width)
            & (pts[:, 1] >= 0)
            & (pts[:, 1] < height)
        )
        kept = np.round(pts[in_bounds], decimals=truncate)
        if kept.size > 0:
            out_list.append(kept)

    if n_false_points > 0:
        x_false = rng.uniform(0, width, size=n_false_points)
        y_false = rng.uniform(0, height, size=n_false_points)
        out_list.append(
            np.round(np.column_stack([x_false, y_false]), decimals=truncate)
        )

    if not out_list:
        return np.empty((0, 2), dtype=np.float64)
    return np.vstack(out_list)
