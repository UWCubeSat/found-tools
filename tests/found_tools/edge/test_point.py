import numpy as np
import pytest

from found_tools.edge.point import (
    _limit_points_per_pixel,
    _point_on_visible_arc,
    _solve_conic,
    generate_edge_points,
    solve_point,
    sort_points_polar_order,
)
from found_tools.utils._camera import Camera


@pytest.fixture
def camera():
    return Camera(0.01, 1e-5, 1000, 1000)


# Circle centred at (500, 500) with radius 100 in pixel space:
#   (x-500)² + (y-500)² = 10000  →  x²+y²-1000x-1000y+490000 = 0
CC_CIRCLE = np.array([1.0, 0.0, 1.0, -1000.0, -1000.0, 490000.0])

# rc = [-1,0,0]: Earth is in the -x camera direction, so -rc = [1,0,0]
# (pointing toward sky). Rays through the circle at x=500 have a positive
# dot product with [1,0,0], so both solutions are on the visible arc.
RC_VISIBLE = np.array([-1.0, 0.0, 0.0])
RC_OCCLUDED = np.array([1.0, 0.0, 0.0])


def _eval_conic(cc: np.ndarray, pt: np.ndarray) -> float:
    a, b, c, d, e, f = cc
    x, y = float(pt[0]), float(pt[1])
    return a * x**2 + b * x * y + c * y**2 + d * x + e * y + f


# ── _solve_conic ────────────────────────────────────────────────────────────


def test_solve_conic_no_real_solution():
    # x² + y² + 1 = 0 has no real solutions
    cc = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0])
    assert _solve_conic(cc, x=0.0) == []


def test_solve_conic_tangent():
    # y² + 2y + 1 = 0  →  (y+1)² = 0  →  y = -1 (double root)
    cc = np.array([0.0, 0.0, 1.0, 0.0, 2.0, 1.0])
    result = _solve_conic(cc, x=0.0)
    assert len(result) == 1
    assert result[0] == pytest.approx(np.array([0.0, -1.0]), abs=1e-9)


def test_solve_conic_two_solutions():
    # x² + y² - 1 = 0 at x=0  →  y = ±1
    cc = np.array([1.0, 0.0, 1.0, 0.0, 0.0, -1.0])
    result = _solve_conic(cc, x=0.0)
    assert len(result) == 2
    ys = sorted(float(pt[1]) for pt in result)
    assert ys == pytest.approx([-1.0, 1.0], abs=1e-9)


def test_solve_conic_linear_p_zero_q_nonzero():
    # x + y² = 0  (a=0, b=0, c=1, d=1, e=0, f=0).
    # Solving for x at y=2: P=a=0, Q=d=1, R=c*4=4  →  x = -R/Q = -4.
    cc = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    result = _solve_conic(cc, y=2.0)
    assert len(result) == 1
    assert result[0] == pytest.approx(np.array([-4.0, 2.0]), abs=1e-9)


# ── solve_point ─────────────────────────────────────────────────────────────


def test_solve_point_satisfies_conic(camera):
    # Circle at (500,500) r=100. At x=500: y=400 and y=600 (both in bounds).
    pts = solve_point(CC_CIRCLE, RC_VISIBLE, camera, x=500.0)
    assert len(pts) == 2
    for pt in pts:
        assert _eval_conic(CC_CIRCLE, pt) == pytest.approx(0.0, abs=1e-6)


def test_solve_point_filters_out_of_bounds(camera):
    # x² + y² = 1_000_000 (radius 1000). At x=0: y = ±1000, both outside [0, 1000).
    cc = np.array([1.0, 0.0, 1.0, 0.0, 0.0, -1_000_000.0])
    assert solve_point(cc, RC_VISIBLE, camera, x=0.0) == []


def test_solve_point_filters_not_visible(camera):
    # Same in-bounds circle, but rc flipped → dot products negative → not visible.
    assert solve_point(CC_CIRCLE, RC_OCCLUDED, camera, x=500.0) == []


# ── sort_points_polar_order ──────────────────────────────────────────────────


def test_sort_polar_order_four_corners():
    # Cardinal points around the origin; centroid is (0, 0).
    points = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    sorted_pts = sort_points_polar_order(points)
    cx = float(np.mean(sorted_pts[:, 0]))
    cy = float(np.mean(sorted_pts[:, 1]))
    angles = np.arctan2(sorted_pts[:, 1] - cy, sorted_pts[:, 0] - cx)
    assert np.all(np.diff(angles) >= 0)


def test_sort_polar_order_empty():
    result = sort_points_polar_order(np.empty((0, 2)))
    assert result.shape == (0, 2)


# ── _limit_points_per_pixel ──────────────────────────────────────────────────


def test_limit_per_pixel_keeps_first():
    # Two points in pixel (0, 0); only the first should be kept.
    pts = [np.array([0.1, 0.1]), np.array([0.2, 0.2])]
    result = _limit_points_per_pixel(pts, edge_per_pixel=1)
    assert len(result) == 1
    assert np.array_equal(result[0], pts[0])


def test_limit_per_pixel_allows_two():
    pts = [np.array([0.1, 0.1]), np.array([0.2, 0.2])]
    result = _limit_points_per_pixel(pts, edge_per_pixel=2)
    assert len(result) == 2


def test_limit_per_pixel_different_pixels():
    # Each point is in a distinct pixel; all should be kept with limit=1.
    pts = [np.array([0.5, 0.5]), np.array([1.5, 1.5]), np.array([2.5, 2.5])]
    result = _limit_points_per_pixel(pts, edge_per_pixel=1)
    assert len(result) == 3


# ── _point_on_visible_arc ───────────────────────────────────────────────────


def test_point_on_visible_arc_hyperbola_branches():
    # Hyperbola y² - x² = 1. At y=2: x = ±√3.
    # k_inv=I so the unprojected ray equals [x, y, 1].
    # rc=[1,0,-1], sky direction -rc=[-1,0,1].
    # dot([x,y,1],[-1,0,1]) = -x+1
    # Right branch x=+√3: -√3+1 < 0 → not visible
    # Left  branch x=-√3: +√3+1 > 0 → visible
    k_inv = np.eye(3)
    rc = np.array([1.0, 0.0, -1.0])
    assert not _point_on_visible_arc(np.sqrt(3), 2.0, rc, k_inv)
    assert _point_on_visible_arc(-np.sqrt(3), 2.0, rc, k_inv)


# ── _solve_conic degenerate (P=0, Q=0) ──────────────────────────────────────


def test_solve_conic_degenerate_pq_zero():
    # y² - 1 = 0  (c=1, f=-1, all others zero).
    # Solving for x (fixed y): P=a=0, Q=b*y+d=0, R=c*y²+e*y+f.
    # At y=1: R=0 → degenerate whole-line → returns [].
    cc = np.array([0.0, 0.0, 1.0, 0.0, 0.0, -1.0])
    assert _solve_conic(cc, y=1.0) == []


# ── generate_edge_points ─────────────────────────────────────────────────────

# Circle centred at (-500, -500) with radius 100 — entirely outside the 1000×1000
# image. No scan line inside the image intersects this circle.
# (x+500)² + (y+500)² = 10000  →  x²+y²+1000x+1000y+490000 = 0
CC_OUTSIDE = np.array([1.0, 0.0, 1.0, 1000.0, 1000.0, 490000.0])


def test_generate_edge_points_no_intersections(camera):
    result = generate_edge_points(CC_OUTSIDE, camera, RC_VISIBLE)
    assert result.shape == (0, 2)


def test_generate_edge_points_returns_valid_points(camera):
    # Circle at (500, 500) r=100 intersects every scan line from x=400 to x=600.
    result = generate_edge_points(CC_CIRCLE, camera, RC_VISIBLE)
    assert result.ndim == 2 and result.shape[1] == 2
    assert len(result) > 0
    for pt in result:
        assert _eval_conic(CC_CIRCLE, pt) == pytest.approx(0.0, abs=1e-6)
