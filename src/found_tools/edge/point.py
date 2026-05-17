
import numpy as np

from found_tools.utils._camera import Camera
    
def _solve_conic(
    cc: np.ndarray,
    x: float | None = None,
    y: float | None = None,
    eps: float | None = None,
) -> list[np.ndarray]:
    """Solve ax²+bxy+cy²+dx+ey+f = 0 for the unknown variable; return all real solutions as [x, y] arrays.

    Classifies and solves the quadratic (or linear) equation in one variable
    obtained by substituting the known coordinate. Returns all real roots
    without any visibility filtering.

    Args:
        cc:  Conic coefficients [a, b, c, d, e, f] for ax²+bxy+cy²+dx+ey+f = 0,
             shape (6,) or flattenable to (6,).
        x:   Known x value; provide to solve for y.
        y:   Known y value; provide to solve for x.
        eps: Tolerance for near-zero discriminant / leading coefficient
             (default: 1e-10).

    Preconditions:
        Exactly one of x or y must be provided (not both, not neither).

    Returns:
        List of 0, 1, or 2 arrays of shape (2,), each containing [x, y] for a
        real intersection point. Empty when there is no real solution or the
        equation is degenerate (e.g. whole line satisfies).
    """
    assert (x is None) != (y is None), "Provide exactly one of x or y"

    if eps is None:
        eps = 1e-10

    a, b, c, d, e, f = cc.flatten()

    if x is not None:
        P, Q, R = c, b * x + e, a * x**2 + d * x + f
    else:
        P, Q, R = a, b * y + d, c * y**2 + e * y + f

    def _make(val: float) -> np.ndarray:
        px, py = (x, val) if x is not None else (val, y)
        return np.array([px, py], dtype=np.float64)

    if abs(P) < eps:
        if abs(Q) < eps:
            return []  # 0=0 (ALL) or 0=R≠0 (NONE) — no finite set to return
        return [_make(-R / Q)]

    delta = Q * Q - 4 * P * R
    if delta < -eps:
        return []
    if abs(delta) <= eps:
        return [_make(-Q / (2 * P))]

    sqrt_d = np.sqrt(delta)
    return [_make((-Q + sqrt_d) / (2 * P)), _make((-Q - sqrt_d) / (2 * P))]


def solve_point(
    cc: np.ndarray,
    rc: np.ndarray,
    camera: Camera,
    x: float | None = None,
    y: float | None = None,
    eps: float | None = None,
) -> list[np.ndarray]:
    """Solve ax²+bxy+cy²+dx+ey+f = 0 and return visible in-image intersections.

    Delegates to _solve_conic for the quadratic solve, then keeps only points
    that lie within the camera image bounds and on the visible (sky-facing) arc.

    Args:
        cc:     Conic coefficients [a, b, c, d, e, f] for ax²+bxy+cy²+dx+ey+f = 0,
                shape (6,) or flattenable to (6,).
        rc:     Camera position in camera frame, shape (3,). Used to determine
                which roots lie on the near (visible) arc.
        camera: Camera object supplying x_resolution, y_resolution, and
                inverse_calibration_matrix for bounds checking and visibility.
        x:      Known x value; provide to solve for y.
        y:      Known y value; provide to solve for x.
        eps:    Tolerance for near-zero discriminant / leading coefficient
                (default: 1e-10).

    Preconditions:
        Exactly one of x or y must be provided (not both, not neither).

    Returns:
        List of 0, 1, or 2 arrays of shape (2,), each containing [x, y] for a
        point on the visible arc within the image bounds.
    """
    k_inv = camera.inverse_calibration_matrix
    return [
        pt for pt in _solve_conic(cc, x=x, y=y, eps=eps)
        if (
            0 <= float(pt[0]) < camera.x_resolution
            and 0 <= float(pt[1]) < camera.y_resolution
            and _point_on_visible_arc(float(pt[0]), float(pt[1]), rc, k_inv)
        )
    ]

def _point_on_visible_arc(
    x: float, y: float, rc: np.ndarray, k_inv: np.ndarray
) -> bool:
    """Return True if pixel (x, y) lies on the visible (sky-facing) arc of the horizon.

    The projected conic has two arcs: the near (visible) limb facing the camera
    and the far arc hidden behind the body. This test selects the near arc by
    checking which side of the limb the pixel's ray falls on.

    Geometric rule: unproject the pixel to a camera-frame ray direction, then
    check ray · (−rc) ≥ 0. Because rc points from the camera origin toward
    Earth's center, −rc points toward open sky. A non-negative dot product means
    the ray and the sky vector agree within 90°, so the pixel is on the
    observable side of the limb.

    Args:
        x:     Pixel x-coordinate (column).
        y:     Pixel y-coordinate (row).
        rc:    Camera position in camera frame, shape (3,). Points from the
               camera origin toward the body center; its magnitude equals the
               camera-to-center distance.
        k_inv: Inverse camera calibration matrix (3×3). Maps homogeneous pixel
               coordinates to un-normalised camera-frame ray directions.

    Returns:
        True when the ray through (x, y) points away from Earth (visible arc);
        False when it points toward the far arc, occluded by the body.
    """
    ray = k_inv @ np.array([x, y, 1.0], dtype=np.float64)
    return float(np.dot(ray, -rc)) >= 0.0

def sort_points_polar_order(points: np.ndarray) -> np.ndarray:
    """Sort (N, 2) pixel points by angle around their centroid for drawing a closed contour.

    Args:
        points: (N, 2) array of (x, y) coordinates.

    Returns:
        Points sorted by angle (radians) from centroid, shape (N, 2).
        Returns a copy if N > 0, or the same empty array if N == 0.
    """
    if points.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    pts = np.asarray(points, dtype=np.float64)
    cx = float(np.mean(pts[:, 0]))
    cy = float(np.mean(pts[:, 1]))
    angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    order = np.argsort(angles)
    return pts[order].copy()

def _limit_points_per_pixel(
    points: list[np.ndarray], edge_per_pixel: int
) -> list[np.ndarray]:
    """Keep at most edge_per_pixel points per pixel cell, discarding later arrivals.

    A pixel cell is identified by (floor(x), floor(y)). Points are processed
    in order; once a cell has edge_per_pixel entries, further points in that
    cell are dropped.

    Args:
        points:         List of shape-(2,) arrays, each [x, y].
        edge_per_pixel: Maximum number of points to retain per pixel cell.

    Returns:
        Filtered list preserving original order.
    """
    counts: dict[tuple[int, int], int] = {}
    keep: list[np.ndarray] = []
    for pt in points:
        key = (int(np.floor(pt[0])), int(np.floor(pt[1])))
        if counts.get(key, 0) < edge_per_pixel:
            counts[key] = counts.get(key, 0) + 1
            keep.append(pt)
    return keep


def generate_edge_points(
    cc: np.ndarray, camera: Camera, rc: np.ndarray, center: float = 0.5, step: float = 1.0, edge_per_pixel: int = 1
) -> np.ndarray:
    """Sample the conic at every row and column; return visible in-image points.

    Scans x from center to x_resolution and y from center to y_resolution in
    increments of step. For each x, solves for y; for each y, solves for x.
    Results are filtered to the visible arc and image bounds by solve_point,
    then deduplicated to at most edge_per_pixel points per pixel cell.

    Args:
        cc:            Conic coefficients [a, b, c, d, e, f], shape (6,) or
                       flattenable to (6,).
        camera:        Camera supplying resolution and inverse calibration.
        rc:            Camera position in camera frame, shape (3,).
        center:        Starting offset for row/column sampling (default 0.5,
                       i.e. pixel centers).
        step:          Sampling increment in pixels (default 1.0).
        edge_per_pixel: Maximum number of edge points to keep per pixel cell
                       (default 1).

    Returns:
        (N, 2) array of (x, y) pixel coordinates on the visible arc.
    """
    points: list[np.ndarray] = []
    for x in np.arange(center, camera.x_resolution, step):
        points.extend(solve_point(cc, rc, camera, x=float(x)))
    for y in np.arange(center, camera.y_resolution, step):
        points.extend(solve_point(cc, rc, camera, y=float(y)))
    points = _limit_points_per_pixel(points, edge_per_pixel)
    if not points:
        return np.empty((0, 2), dtype=np.float64)
    return np.array(points, dtype=np.float64)