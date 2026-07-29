import numpy as np
import pytest

from found_tools.edge.noise import (
    _add_gaussian_noise,
    _downsample,
    _false_points,
    _filter_in_bounds,
    _truncate,
    add_point_noise,
)
from found_tools.utils._camera import Camera


@pytest.fixture
def camera():
    return Camera(0.01, 1e-5, 1000, 1000)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


# ── _downsample ──────────────────────────────────────────────────────────────


def test_downsample_reduces_count(rng):
    pts = np.arange(20, dtype=np.float64).reshape(10, 2)
    result = _downsample(pts, max_points=5, rng=rng)
    assert result.shape == (5, 2)


def test_downsample_unchanged_when_small(rng):
    pts = np.arange(6, dtype=np.float64).reshape(3, 2)
    result = _downsample(pts, max_points=5, rng=rng)
    assert np.array_equal(result, pts)


def test_downsample_reproducible():
    pts = np.arange(20, dtype=np.float64).reshape(10, 2)
    r1 = _downsample(pts, max_points=5, rng=np.random.default_rng(0))
    r2 = _downsample(pts, max_points=5, rng=np.random.default_rng(0))
    assert np.array_equal(r1, r2)


# ── _add_gaussian_noise ──────────────────────────────────────────────────────


def test_add_gaussian_noise_zero_sigma(rng):
    pts = np.array([[100.0, 200.0], [300.0, 400.0]])
    result = _add_gaussian_noise(pts, sigma=0.0, rng=rng)
    assert result == pytest.approx(pts, abs=1e-12)


def test_add_gaussian_noise_changes_values():
    pts = np.array([[100.0, 200.0], [300.0, 400.0]])
    result = _add_gaussian_noise(pts, sigma=10.0, rng=np.random.default_rng(7))
    assert not np.array_equal(result, pts)


# ── _filter_in_bounds ────────────────────────────────────────────────────────


def test_filter_in_bounds_removes_outside(camera):
    pts = np.array(
        [
            [-1.0, 500.0],  # x < 0
            [500.0, 500.0],  # in bounds
            [500.0, -1.0],  # y < 0
            [999.9, 999.9],  # in bounds
            [1000.0, 500.0],  # x == width (exclusive)
        ]
    )
    result = _filter_in_bounds(pts, camera)
    assert result.shape == (2, 2)
    assert np.array_equal(result[0], [500.0, 500.0])
    assert np.array_equal(result[1], [999.9, 999.9])


def test_filter_in_bounds_all_pass(camera):
    pts = np.array([[0.0, 0.0], [500.0, 500.0], [999.0, 999.0]])
    assert np.array_equal(_filter_in_bounds(pts, camera), pts)


# ── _false_points ────────────────────────────────────────────────────────────


def test_false_points_count(camera, rng):
    result = _false_points(10, camera, rng)
    assert result.shape == (10, 2)


def test_false_points_in_bounds(camera, rng):
    result = _false_points(200, camera, rng)
    assert np.all((result[:, 0] >= 0) & (result[:, 0] < camera.x_resolution))
    assert np.all((result[:, 1] >= 0) & (result[:, 1] < camera.y_resolution))


# ── _truncate ────────────────────────────────────────────────────────────────


def test_truncate_to_integer():
    pts = np.array([[1.7, 2.3], [3.5, 4.4]])
    result = _truncate(pts, decimals=0)
    assert result == pytest.approx(np.round(pts, 0), abs=1e-9)
    assert result.dtype == np.float64


def test_truncate_to_two_decimals():
    pts = np.array([[1.23456, 7.89012]])
    result = _truncate(pts, decimals=2)
    assert result[0, 0] == pytest.approx(1.23, abs=1e-9)
    assert result[0, 1] == pytest.approx(7.89, abs=1e-9)


# ── add_point_noise ──────────────────────────────────────────────────────────


def test_add_point_noise_gaussian_only(camera):
    pts = np.full((20, 2), 500.0)
    result = add_point_noise(
        pts, camera, gaussian_sigma=1.0, rng=np.random.default_rng(0)
    )
    assert result.ndim == 2 and result.shape[1] == 2
    assert np.all((result[:, 0] >= 0) & (result[:, 0] < camera.x_resolution))
    assert np.all((result[:, 1] >= 0) & (result[:, 1] < camera.y_resolution))


def test_add_point_noise_false_only(camera):
    pts = np.empty((0, 2))
    result = add_point_noise(
        pts, camera, n_false_points=5, rng=np.random.default_rng(0)
    )
    assert result.shape == (5, 2)


def test_add_point_noise_combined(camera):
    # sigma=0 keeps all 3 true points in bounds; 2 false appended → 5 total
    pts = np.full((3, 2), 500.0)
    result = add_point_noise(
        pts, camera, gaussian_sigma=0.0, n_false_points=2, rng=np.random.default_rng(0)
    )
    assert result.shape == (5, 2)


def test_add_point_noise_downsample(camera):
    # sigma=0 keeps all 3 true points in bounds; 2 false appended → 5 total
    pts = np.full((3, 2), 500.0)
    result = add_point_noise(
        pts,
        camera,
        gaussian_sigma=0.0,
        n_false_points=2,
        max_points=3,
        rng=np.random.default_rng(0),
    )
    assert result.shape == (3, 2)


def test_add_point_noise_empty_input(camera):
    pts = np.empty((0, 2))
    result = add_point_noise(
        pts, camera, n_false_points=3, rng=np.random.default_rng(0)
    )
    assert result.shape == (3, 2)


def test_add_point_noise_invalid_shape(camera):
    with pytest.raises(ValueError):
        add_point_noise(np.ones((3, 3)), camera)
