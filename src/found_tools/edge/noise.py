import numpy as np

from found_tools.utils._camera import Camera


def _downsample(
    points: np.ndarray,
    max_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Randomly downsample to at most max_points rows; return unchanged if already small enough.

    Args:
        points:     (N, 2) float64 array.
        max_points: Maximum rows to keep.
        rng:        Random generator for reproducibility.

    Returns:
        (M, 2) array where M = min(N, max_points).
    """
    if points.shape[0] <= max_points:
        return points
    idx = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[idx]


def _add_gaussian_noise(
    points: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add independent zero-mean Gaussian noise to each point.

    Args:
        points: (N, 2) array of (x, y) coordinates.
        sigma:  Noise standard deviation. A single float applies equal sigma to
                both axes; a (sigma_x, sigma_y) tuple uses per-axis values.
        rng:    Random generator for reproducibility.

    Returns:
        (N, 2) array with noise added.
    """
    return points + rng.normal(0.0, sigma, size=points.shape)


def _filter_in_bounds(points: np.ndarray, camera: Camera) -> np.ndarray:
    """Keep only points that fall within the image bounds.

    Args:
        points: (N, 2) array of (x, y) coordinates.
        camera: Camera supplying x_resolution and y_resolution.

    Returns:
        (M, 2) array containing only rows with 0 ≤ x < width and 0 ≤ y < height.
    """
    mask = (
        (points[:, 0] >= 0)
        & (points[:, 0] < camera.x_resolution)
        & (points[:, 1] >= 0)
        & (points[:, 1] < camera.y_resolution)
    )
    return points[mask]


def _false_points(
    n: int,
    camera: Camera,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate n points drawn uniformly at random within the image.

    Args:
        n:      Number of false points to generate.
        camera: Camera supplying x_resolution and y_resolution.
        rng:    Random generator for reproducibility.

    Returns:
        (n, 2) array of (x, y) coordinates.
    """
    x = rng.uniform(0, camera.x_resolution, size=n)
    y = rng.uniform(0, camera.y_resolution, size=n)
    return np.column_stack([x, y])


def _truncate(points: np.ndarray, decimals: int) -> np.ndarray:
    """Round point coordinates to a fixed number of decimal places.

    Args:
        points:   (N, 2) float64 array of (x, y) coordinates.
        decimals: Number of decimal places to round to (e.g. 0 for integer
                  pixels, 2 for sub-pixel precision to two places).

    Returns:
        (N, 2) float64 array with coordinates rounded to decimals places.
    """
    return np.round(points, decimals=decimals).astype(np.float64)


def add_point_noise(
    points: np.ndarray,
    camera: Camera,
    gaussian_sigma: float | None = None,
    n_false_points: int = 0,
    max_points: int | None = None,
    truncate: int = 0,
    rng: np.random.Generator | None = None,
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
        max_points: If set, randomly downsample input points to at most this many
            before applying noise. Ignored if None or if input has fewer points.
        truncate: Number of decimal places for point coordinates (e.g. 0 for
            integer pixels, 2 for two decimals). Applied to all returned points.
        rng: Random generator for reproducibility. If None, uses default generator.

    Returns:
        (M, 2) array of (x, y) coordinates; all rows satisfy
        0 <= x < width and 0 <= y < height. Order: perturbed true points
        (that remain in bounds) first, then false points.
    """
    rng = rng if rng is not None else np.random.default_rng()
    out: list[np.ndarray] = []

    if points.size > 0:
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError("points must have shape (N, 2)")
        if gaussian_sigma is not None:
            pts = _add_gaussian_noise(pts, gaussian_sigma, rng)
        pts = _truncate(_filter_in_bounds(pts, camera), truncate)
        if pts.size > 0:
            out.append(pts)

    if n_false_points > 0:
        out.append(_truncate(_false_points(n_false_points, camera, rng), truncate))

    noisy_points = np.vstack(out).astype(np.float64)
    if max_points is not None and max_points >= 1:
        noisy_points = _downsample(noisy_points, max_points, rng)

    return noisy_points
