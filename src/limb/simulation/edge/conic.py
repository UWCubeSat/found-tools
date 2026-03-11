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
    points = generateEdgePoints(rc, Ap, TPC, num_points, cam)
"""

import numpy as np

from limb.utils._camera import Camera


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
    k_inv = camera.inverseCalibrationMatrix_
    calibrated_c = k_inv.T @ c @ k_inv
    return calibrated_c / calibrated_c[0, 0]


def solve_quadratic_y(matrix_a: np.ndarray, x_val: float):
    """Solve for y in the quadratic form [x, y, 1] A [x, y, 1]ᵀ = 0.

    Args:
        matrix_a: 3×3 NumPy array representing the symmetric quadratic form.
        x_val:    The known x-coordinate.

    Returns:
        A tuple of real roots (y1, y2) if they exist, or None if the roots
        are complex or no solution exists.
    """
    a = matrix_a[1, 1]
    b = (matrix_a[0, 1] + matrix_a[1, 0]) * x_val + (matrix_a[1, 2] + matrix_a[2, 1])
    c = matrix_a[0, 0] * x_val**2 + (matrix_a[0, 2] + matrix_a[2, 0]) * x_val + matrix_a[2, 2]

    if np.isclose(a, 0):
        if np.isclose(b, 0):
            return None
        return (-c / b,)

    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        return None
    if np.isclose(discriminant, 0):
        return (-b / (2 * a),)

    sqrt_d = np.sqrt(discriminant)
    return ((-b + sqrt_d) / (2 * a), (-b - sqrt_d) / (2 * a))


def solve_conic(conic: np.ndarray, x_vals: np.ndarray) -> np.ndarray:
    """Sample y-coordinates on a conic for an array of x-values.

    For each x, solves the quadratic form [x, y, 1] A [x, y, 1]ᵀ = 0 for y.
    When two real roots exist the one with the smaller absolute value is taken;
    rows where no real solution exists are filled with NaN.

    Args:
        conic:  3×3 symmetric conic matrix in pixel coordinates.
        x_vals:  1-D array of x pixel coordinates, shape (N,).

    Returns:
        points: Array of (x, y) pixel coordinates, shape (N, 2).
                Rows where no real solution exists contain NaN.
    """
    points = np.full((len(x_vals), 2), np.nan)
    for i, x in enumerate(x_vals):
        roots = solve_quadratic_y(conic, x)
        if roots is not None:
            points[i] = [x, min(roots, key=abs)]
    return points


def generate_edge_points(
    pos: np.ndarray,
    shape_matrix: np.ndarray,
    orientation: np.ndarray,
    num_points: int,
    cam: Camera,
) -> np.ndarray:
    """Generate points on the projected horizon ellipse in pixel coordinates.

    Builds the conic locus from the given pose, projects it into pixel
    coordinates using the camera's intrinsics, then samples it at evenly
    spaced x-values across the image width.

    Args:
        pos:         Satellite position in camera coordinates, shape (3,).
        shape_matrix: Ellipsoid defining matrix in world coordinates,
                     shape (3, 3). Diagonal entries are 1/a² for each
                     semi-axis.
        orientation: Rotation matrix from world to camera coordinates
                     (TPC), shape (3, 3).
        num_points:   Number of points to sample along the conic.
        cam:         :class:`~found_CLI_tools.cameraGeometry.Camera` object
                     supplying intrinsics and resolution.

    Returns:
        points: NumPy array of (x, y) pixel coordinates on the conic,
            shape (num_points, 2). Rows with no real solution contain NaN.
    """
    k_inv = cam.inverseCalibrationMatrix_
    cam_conic = generate_camera_conic(pos, shape_matrix, orientation)
    pixel_conic = generate_pixel_conic(cam_conic, k_inv)
    x_points = np.linspace(0, cam.xResolution_ - 1, num_points)
    return solve_conic(pixel_conic, x_points)


# Backward-compatible aliases for previous camelCase API names.
generateCameraConic = generate_camera_conic
generatePixelConic = generate_pixel_conic
solveQuadraticY = solve_quadratic_y
solveConic = solve_conic
generateEdgePoints = generate_edge_points
