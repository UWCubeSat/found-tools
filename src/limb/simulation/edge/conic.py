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


def generateCameraConic(
    rc: np.ndarray,
    shapeMatrix: np.ndarray,
    orientation: np.ndarray,
) -> np.ndarray:
    """Build the conic locus matrix in camera coordinates.

    Transforms the world-frame ellipsoid into the camera frame and constructs
    the 3×3 symmetric matrix C representing the horizon conic seen from *rc*.

    Args:
        rc:          Camera position vector in camera coordinates, shape (3,).
        shapeMatrix: Ellipsoid defining matrix in world coordinates,
                     shape (3, 3). Diagonal entries are 1/a² for each
                     semi-axis.
        orientation: Rotation matrix transforming world coordinates to camera
                     coordinates (TPC), shape (3, 3).

    Returns:
        C: 3×3 symmetric NumPy array representing the conic locus in camera
           coordinates.
    """
    tcp = orientation.T
    ac = orientation @ shapeMatrix @ tcp
    c = ac @ np.outer(rc, rc) @ ac - (rc @ ac @ rc * np.eye(3) - np.eye(3)) @ ac
    return c


def generatePixelConic(c: np.ndarray, kInv: np.ndarray) -> np.ndarray:
    """Project the camera-space conic into pixel coordinates.

    Applies the inverse intrinsics transform K⁻¹ so the resulting matrix
    represents the conic directly in pixel coordinates::

        C[0,0]x² + 2C[0,1]xy + C[1,1]y²
            + 2C[0,2]x + 2C[1,2]y + C[2,2] = 0

    The matrix is normalised so that the leading coefficient equals 1.

    Args:
        c:    3×3 symmetric conic matrix in camera (metric) coordinates.
        kInv: 3×3 inverse intrinsics matrix (K⁻¹).

    Returns:
        calibratedC: 3×3 symmetric conic matrix in pixel coordinates,
                     normalised so calibratedC[0, 0] == 1.
    """
    calibratedC = kInv.T @ c @ kInv
    return calibratedC / calibratedC[0, 0]


def solveQuadraticY(matrixA: np.ndarray, xVal: float):
    """Solve for y in the quadratic form [x, y, 1] A [x, y, 1]ᵀ = 0.

    Args:
        matrixA: 3×3 NumPy array representing the symmetric quadratic form.
        xVal:    The known x-coordinate.

    Returns:
        A tuple of real roots (y1, y2) if they exist, or None if the roots
        are complex or no solution exists.
    """
    a = matrixA[1, 1]
    b = (matrixA[0, 1] + matrixA[1, 0]) * xVal + (matrixA[1, 2] + matrixA[2, 1])
    c = matrixA[0, 0] * xVal**2 + (matrixA[0, 2] + matrixA[2, 0]) * xVal + matrixA[2, 2]

    if np.isclose(a, 0):
        if np.isclose(b, 0):
            return None
        return (-c / b,)

    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        return None
    if np.isclose(discriminant, 0):
        return (-b / (2 * a),)

    sqrtD = np.sqrt(discriminant)
    return ((-b + sqrtD) / (2 * a), (-b - sqrtD) / (2 * a))


def solveConic(conic: np.ndarray, xVals: np.ndarray) -> np.ndarray:
    """Sample y-coordinates on a conic for an array of x-values.

    For each x, solves the quadratic form [x, y, 1] A [x, y, 1]ᵀ = 0 for y.
    When two real roots exist the one with the smaller absolute value is taken;
    rows where no real solution exists are filled with NaN.

    Args:
        conic:  3×3 symmetric conic matrix in pixel coordinates.
        xVals:  1-D array of x pixel coordinates, shape (N,).

    Returns:
        points: Array of (x, y) pixel coordinates, shape (N, 2).
                Rows where no real solution exists contain NaN.
    """
    points = np.full((len(xVals), 2), np.nan)
    for i, x in enumerate(xVals):
        roots = solveQuadraticY(conic, x)
        if roots is not None:
            points[i] = [x, min(roots, key=abs)]
    return points


def generateEdgePoints(
    pos: np.ndarray,
    shapeMatrix: np.ndarray,
    orientation: np.ndarray,
    numPoints: int,
    cam: Camera,
) -> np.ndarray:
    """Generate points on the projected horizon ellipse in pixel coordinates.

    Builds the conic locus from the given pose, projects it into pixel
    coordinates using the camera's intrinsics, then samples it at evenly
    spaced x-values across the image width.

    Args:
        pos:         Satellite position in camera coordinates, shape (3,).
        shapeMatrix: Ellipsoid defining matrix in world coordinates,
                     shape (3, 3). Diagonal entries are 1/a² for each
                     semi-axis.
        orientation: Rotation matrix from world to camera coordinates
                     (TPC), shape (3, 3).
        numPoints:   Number of points to sample along the conic.
        cam:         :class:`~found_CLI_tools.cameraGeometry.Camera` object
                     supplying intrinsics and resolution.

    Returns:
        points: NumPy array of (x, y) pixel coordinates on the conic,
                shape (numPoints, 2). Rows with no real solution contain NaN.
    """
    kInv = cam.inverseCalibrationMatrix_
    camConic = generateCameraConic(pos, shapeMatrix, orientation)
    pixelConic = generatePixelConic(camConic, kInv)
    xPoints = np.linspace(0, cam.xResolution_ - 1, numPoints)
    return solveConic(pixelConic, xPoints)
