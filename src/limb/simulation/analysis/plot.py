"""Plot a window of the image around a point with edge points and optional conic.

Also provides OpNav validation plots: radius residuals vs range and centroid
residuals in the illumination frame.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.image import imread
from scipy import stats
from scipy.spatial import cKDTree

from limb.simulation.analysis.metrics import fill_pixel_metrics

# Columns used for camera identity (must be present in df)
_CAMERA_KEY_COLS = ("cam_focal_length", "cam_x_resolution", "cam_y_resolution")

# When point count exceeds these, show density underlay and/or subsample scatter for clarity
_HEXBIN_MIN_POINTS = 800
_MAX_SCATTER_POINTS = 3500


def _camera_id_from_row(row: pd.Series) -> tuple[float, int, int]:
    """Unique camera identifier from a metadata row."""
    return (
        float(row["cam_focal_length"]),
        int(row["cam_x_resolution"]),
        int(row["cam_y_resolution"]),
    )


def _camera_label(cam_id: tuple[float, int, int]) -> str:
    """Human-readable label for legend."""
    f, wx, wy = cam_id
    return f"{wx}×{wy} f={f:.4g}"


def _density_scaled_sizes(
    x: np.ndarray,
    y: Optional[np.ndarray] = None,
    base_size: float = 40.0,
    min_size: float = 1.0,
    max_size: float = 100.0,
    k: int = 50,
) -> np.ndarray:
    """Scale point sizes inversely with local density so 100k+ points remain interpretable.

    For each point, computes distance to k-th nearest neighbor (in x, or in (x,y) if y
    given). Size is proportional to that distance (sparse => larger, dense => smaller),
    normalized by the median distance and clamped to [min_size, max_size].
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    if n == 0:
        return np.array([])
    if y is not None:
        y = np.asarray(y, dtype=np.float64).ravel()
        if len(y) != n:
            y = None
    k = min(k, n - 1)
    if k < 1:
        return np.full(n, base_size)
    # Stack (x,) or (x, y) for distance
    if y is not None:
        xy = np.column_stack([x, y])
        # Scale y to similar range as x for distance
        xr = np.ptp(x)
        yr = np.ptp(y)
        if yr > 0 and xr > 0:
            xy = np.column_stack([x, y * (xr / yr)])
        pts = xy
    else:
        pts = x.reshape(-1, 1)
    # k+1 because the first neighbor is self (distance 0)
    tree = cKDTree(pts)
    d_k, _ = tree.query(pts, k=k + 1)
    dist_k = d_k[:, -1]
    d_median = np.median(dist_k)
    if d_median <= 0 or not np.isfinite(d_median):
        return np.full(n, base_size)
    # size proportional to distance; normalize so median point gets base_size
    raw = base_size * (dist_k / d_median)
    sizes = np.clip(raw, min_size, max_size)
    return np.asarray(sizes, dtype=np.float64)


def _clumped_prediction_interval(
    x: np.ndarray,
    y: np.ndarray,
    confidence: float = 0.997,
    n_bins: Optional[int] = None,
    fallback_sigma: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute prediction interval per x-clump (bin), for data clumped at specific x values.

    Bins x into n_bins; for each bin with data, computes mean(y) and a prediction
    interval for a new observation at that x: mean ± t * std * sqrt(1 + 1/n),
    where n is the count in the bin. Returns (x_centers, lower, upper) at bin centers
    for bins that have at least one point. Then call _smooth_prediction_bounds to
    fit a curve across the plot.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    if not np.any(ok):
        return np.array([]), np.array([]), np.array([])
    x, y = x[ok], y[ok]
    n = len(x)
    xmin, xmax = float(np.min(x)), float(np.max(x))
    if n_bins is None:
        n_bins = min(50, max(15, n // 3))
    n_bins = max(2, int(n_bins))
    global_std = float(np.nanstd(y)) if n > 1 else 0.0
    if not np.isfinite(global_std) or global_std == 0:
        global_std = 0.0
    fallback = fallback_sigma if fallback_sigma is not None else global_std
    if fallback is None:
        fallback = 0.0
    edges = np.linspace(xmin, xmax, n_bins + 1)
    x_centers_list: list[float] = []
    lower_list: list[float] = []
    upper_list: list[float] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        in_bin = (x >= lo) & (x < hi) if i < n_bins - 1 else (x >= lo) & (x <= hi)
        y_bin = y[in_bin]
        n_bin = len(y_bin)
        if n_bin == 0:
            continue
        mean_y = float(np.mean(y_bin))
        std_y = float(np.std(y_bin))
        if n_bin < 2 or std_y == 0 or not np.isfinite(std_y):
            std_y = fallback
        # Prediction interval for one new observation at this x: mean ± t * std * sqrt(1 + 1/n)
        df = max(n_bin - 1, 1)
        t_mult = stats.t.ppf(0.5 + confidence / 2.0, df)
        se_pred = std_y * np.sqrt(1.0 + 1.0 / n_bin)
        half = t_mult * se_pred
        x_centers_list.append(0.5 * (lo + hi))
        lower_list.append(mean_y - half)
        upper_list.append(mean_y + half)
    if not x_centers_list:
        return np.array([]), np.array([]), np.array([])
    return (
        np.array(x_centers_list, dtype=np.float64),
        np.array(lower_list, dtype=np.float64),
        np.array(upper_list, dtype=np.float64),
    )


def _ridge_prediction_interval(
    x: np.ndarray,
    y: np.ndarray,
    x_eval: np.ndarray,
    confidence: float = 0.997,
    degree: int = 5,
    alpha: float = 1000.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prediction interval for a new response using ridge regression (PSU STAT 501 style).

    Uses the formula from
    https://online.stat.psu.edu/stat501/lesson/3/3.3 :
    PI = ŷ_h ± t_{1-α/2, n-p} × sqrt(MSE × (1 + leverage_h))
    where MSE = SSE/(n-p), leverage_h = x_h'(X'X+αI)^{-1}x_h for ridge design at x_h.
    Returns (x_eval, lower, upper) with symmetric interval around the fitted curve.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    x_eval = np.asarray(x_eval, dtype=np.float64).ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    if not np.any(ok):
        y_mean = float(np.nanmean(y))
        return x_eval, np.full_like(x_eval, y_mean), np.full_like(x_eval, y_mean)
    x, y = x[ok], y[ok]
    n = len(x)
    xmin, xmax = float(np.min(x)), float(np.max(x))
    if xmax <= xmin:
        y_mean = float(np.mean(y))
        return x_eval, np.full_like(x_eval, y_mean), np.full_like(x_eval, y_mean)
    xn = (x - xmin) / (xmax - xmin)
    xn_eval = np.clip((x_eval - xmin) / (xmax - xmin), 0.0, 1.0)
    degree = min(degree, n - 1)
    if degree < 0:
        y_mean = float(np.mean(y))
        return x_eval, np.full_like(x_eval, y_mean), np.full_like(x_eval, y_mean)
    p = degree + 1
    X = np.column_stack([xn**i for i in range(p)])
    Xe = np.column_stack([xn_eval**i for i in range(p)])
    gram = X.T @ X + alpha * np.eye(p)
    try:
        gram_inv = np.linalg.inv(gram)
    except np.linalg.LinAlgError:
        y_mean = float(np.mean(y))
        return x_eval, np.full_like(x_eval, y_mean), np.full_like(x_eval, y_mean)
    coef = gram_inv @ (X.T @ y)
    y_fit = X @ coef
    residuals = y - y_fit
    sse = float(np.sum(residuals**2))
    mse = sse / max(n - p, 1)
    if mse <= 0 or not np.isfinite(mse):
        mse = 0.0
    # Leverage at evaluation points: row_h (X'X+αI)^{-1} row_h'
    levers = np.einsum("ij,jk,ik->i", Xe, gram_inv, Xe)
    levers = np.maximum(levers, 0.0)
    se_pred = np.sqrt(mse * (1.0 + levers))
    y_eval = Xe @ coef
    df = max(n - p, 1)
    t_mult = stats.t.ppf(0.5 + confidence / 2.0, df)
    half = t_mult * se_pred
    lower = y_eval - half
    upper = y_eval + half
    return x_eval, lower, upper


def _ridge_fit_curve(
    x: np.ndarray,
    y: np.ndarray,
    x_eval: np.ndarray,
    degree: int = 5,
    alpha: float = 1000.0,
) -> np.ndarray:
    """Fit y as a polynomial of x with ridge regression; return predicted values at x_eval."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    x_eval = np.asarray(x_eval, dtype=np.float64).ravel()
    # Drop NaNs so fit is well-defined
    ok = np.isfinite(x) & np.isfinite(y)
    if not np.any(ok):
        return np.full_like(x_eval, np.nanmean(y))
    x, y = x[ok], y[ok]
    xmin, xmax = float(np.min(x)), float(np.max(x))
    if xmax <= xmin:
        return np.full_like(x_eval, float(np.mean(y)))
    xn = (x - xmin) / (xmax - xmin)
    xn_eval = np.clip((x_eval - xmin) / (xmax - xmin), 0.0, 1.0)
    degree = min(degree, len(x) - 1)
    if degree < 0:
        return np.full_like(x_eval, float(np.mean(y)))
    X = np.column_stack([xn**i for i in range(degree + 1)])
    Xe = np.column_stack([xn_eval**i for i in range(degree + 1)])
    # Ridge: (X'X + alpha*I)^{-1} X' y
    n_features = X.shape[1]
    gram = X.T @ X + alpha * np.eye(n_features)
    rhs = X.T @ y
    try:
        coef = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        return np.full_like(x_eval, float(np.mean(y)))
    if not np.all(np.isfinite(coef)) or np.any(np.abs(coef) > 1e10):
        return np.full_like(x_eval, float(np.mean(y)))
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        out = Xe @ coef
    if not np.any(np.isfinite(out)):
        out = np.full_like(x_eval, float(np.mean(y)))
    return np.asarray(out, dtype=np.float64)


def _smooth_prediction_bounds(
    x: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    n_fine: int = 500,
    degree: int = 5,
    alpha: float = 1000.0,
    x_fine: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit prediction interval bounds with ridge regression; return symmetric band on a fine grid.

    Strategies for symmetry (we use center + half-width):
    - Center + half-width: From (lower, upper) compute center = (lower+upper)/2 and
      half_width = (upper-lower)/2; fit ridge to center and to half_width; then
      lower_smooth = center_smooth - half_width_smooth, upper_smooth = center_smooth + half_width_smooth.
      Guarantees a symmetric band.
    - Fit mean and std in k-NN, then smooth: Have _prediction_interval_bounds return (mean, std);
      fit center_smooth = ridge(mean), half_width_smooth = ridge(n_sigma*std); same result conceptually.
    - Fit lower and upper then symmetrize: Fit ridge to lower and upper; set center = (lower_smooth+upper_smooth)/2,
      half_width = abs(upper_smooth - lower_smooth)/2; then redefine lower/upper as center ± half_width.
      Equivalent to center + half-width with post-fit center/half_width.

    We use center + half-width so the band is symmetric by construction.
    If x_fine is provided, the curve is evaluated on it; otherwise linspace(x.min(), x.max(), n_fine).
    """
    if x_fine is not None:
        x_fine = np.asarray(x_fine, dtype=np.float64).ravel()
    else:
        x_fine = np.linspace(float(x.min()), float(x.max()), n_fine)
    center = (lower + upper) / 2.0
    half_width = np.maximum((upper - lower) / 2.0, 0.0)
    center_smooth = _ridge_fit_curve(x, center, x_fine, degree=degree, alpha=alpha)
    half_width_smooth = _ridge_fit_curve(x, half_width, x_fine, degree=degree, alpha=alpha)
    half_width_smooth = np.maximum(half_width_smooth, 0.0)
    lower_smooth = center_smooth - half_width_smooth
    upper_smooth = center_smooth + half_width_smooth
    return x_fine, lower_smooth, upper_smooth


def _prediction_interval_bounds(
    x: np.ndarray,
    y: np.ndarray,
    x_grid: np.ndarray,
    n_sigma: float,
    min_points: int = 5,
    fallback_sigma: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute prediction interval upper/lower curves vs x via k-nearest neighbors.

    For each x_grid value, take the k nearest data points (by x distance), compute
    mean and std of their y values, then lower = mean - n_sigma*std,
    upper = mean + n_sigma*std. This keeps the band varying across the full range
    instead of flattening where data is sparse. k = max(min_points, 10); if there
    are fewer than k points total, all points are used. fallback_sigma is used
    when local std is 0 or invalid.

    Returns:
        lower, upper: arrays same shape as x_grid.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x_grid = np.asarray(x_grid, dtype=np.float64)
    n_data = len(x)
    k = min(max(min_points, 10), n_data)
    if k < 1:
        global_std = float(fallback_sigma) if fallback_sigma is not None else 0.0
        if fallback_sigma is None and n_data:
            global_std = float(np.nanstd(y)) or 0.0
        return (
            np.full_like(x_grid, -n_sigma * global_std),
            np.full_like(x_grid, n_sigma * global_std),
        )
    lower = np.full_like(x_grid, np.nan)
    upper = np.full_like(x_grid, np.nan)
    global_std = None if fallback_sigma is not None else np.nanstd(y)
    if global_std is not None and (not np.isfinite(global_std) or global_std == 0):
        global_std = 0.0
    fallback = fallback_sigma if fallback_sigma is not None else global_std
    if fallback is None:
        fallback = 0.0
    for i, x0 in enumerate(x_grid):
        dist = np.abs(x - x0)
        idx = np.argpartition(dist, k - 1)[:k]
        y_sub = y[idx]
        mu = np.mean(y_sub)
        s = np.std(y_sub)
        if s == 0 or not np.isfinite(s):
            s = fallback
        lower[i] = mu - n_sigma * s
        upper[i] = mu + n_sigma * s
    return lower, upper


def edge_plot(
    image_path: str,
    points: np.ndarray,
    center_point: int,
    window_length: float,
    true_points: np.ndarray | None = None,
    save_path: str | None = None,
) -> None:
    """Plot a window of the image centered at points[center_point] with pixel grid and overlays.

    Draws the image crop, subpixel point locations (tiny 'x'), and optionally the
    pixel conic as a line. Pixel boundaries are shown clearly.

    Args:
        image_path: Path to the image file.
        points: (N, 2) array of (x, y) subpixel coordinates.
        center_point: Index into points for the window center.
        window_length: Side length of the square window (pixels).
        true_points: Optional (N, 2) array of true limb points; if given, drawn as red line.
        save_path: Optional path to save the plot image (e.g. .png or .pdf).
    """
    img = imread(image_path)
    if img.ndim == 3:
        height, width = img.shape[0], img.shape[1]
        # Keep black and white: show as grayscale so matplotlib does not change colors
        crop_img = np.mean(img, axis=2)
    else:
        height, width = img.shape[0], img.shape[1]
        crop_img = img

    center = np.asarray(points[center_point], dtype=np.float64)
    half = window_length / 2.0
    x0 = max(0, int(np.floor(center[0] - half)))
    x1 = min(width, int(np.ceil(center[0] + half)))
    y0 = max(0, int(np.floor(center[1] - half)))
    y1 = min(height, int(np.ceil(center[1] + half)))

    crop = crop_img[y0:y1, x0:x1]
    if crop.size == 0:
        raise ValueError("Window is empty (fully outside image).")

    fig, ax = plt.subplots()
    ax.imshow(
        crop,
        extent=[x0, x1, y1, y0],
        aspect="equal",
        interpolation="nearest",
        cmap="gray",
    )
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)

    # Pixel grid; no numbers on axes
    ax.set_xticks(np.arange(x0, x1 + 1))
    ax.set_yticks(np.arange(y0, y1 + 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, color="black", linewidth=0.5, alpha=0.7)

    # Edge points as blue dots
    in_window = (
        (points[:, 0] >= x0)
        & (points[:, 0] < x1)
        & (points[:, 1] >= y0)
        & (points[:, 1] < y1)
    )
    px = points[in_window, 0]
    py = points[in_window, 1]
    if px.size:
        ax.plot(px, py, ".", color="blue", markersize=3, label="Detected Limb Points")

    # Optional: true limb as red line
    if true_points is not None:
        in_window = (
            (true_points[:, 0] >= x0)
            & (true_points[:, 0] < x1)
            & (true_points[:, 1] >= y0)
            & (true_points[:, 1] < y1)
        )
        px_true = true_points[in_window, 0]
        py_true = true_points[in_window, 1]
        if px_true.size:
            ax.plot(px_true, py_true, "-", color="red", linewidth=1, label="True Limb")

    ax.legend()
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
    plt.show()


def plot_radius_residuals_vs_range(
    df: pd.DataFrame,
    radius_1sigma_px: Optional[float] = None,
    target_label: str = "Moon",
    save_path: Optional[Path | str] = None,
):
    """Plot radius residuals (pixels) vs true range (m) with 99.7% prediction interval.

    Accepts the simulation metadata DataFrame directly: if true_r_apparent / out_r_apparent
    (and camera columns) are missing, fill_pixel_metrics(df) is applied so a raw orchestrate
    df can be passed. Uses true_pos_*, true_r_apparent, out_r_apparent, and camera columns.
    Range = ‖true_pos‖; radius residual = out_r_apparent - true_r_apparent. Rows with
    missing out_r_apparent are dropped. The band is a range-dependent 3σ prediction interval.
    """
    need = ["true_pos_x", "true_pos_y", "true_pos_z", "true_r_apparent", "out_r_apparent"] + list(_CAMERA_KEY_COLS)
    work = df.copy()
    if not all(c in work.columns for c in need):
        work = fill_pixel_metrics(work)
    for c in need:
        if c not in work.columns:
            raise ValueError(f"DataFrame must contain column {c!r} (or a df that fill_pixel_metrics can fill)")
    valid = work["true_r_apparent"].notna() & work["out_r_apparent"].notna()
    work = work.loc[valid].copy()
    if len(work) == 0:
        raise ValueError("No rows with both true_r_apparent and out_r_apparent")
    work["_range_m"] = np.linalg.norm(
        work[["true_pos_x", "true_pos_y", "true_pos_z"]].values, axis=1
    )
    work["_radius_residual_px"] = work["out_r_apparent"] - work["true_r_apparent"]
    work["_camera_id"] = work.apply(_camera_id_from_row, axis=1)
    ranges_m = work["_range_m"].values
    radius_residuals_px = work["_radius_residual_px"].values
    sigma = radius_1sigma_px
    if sigma is None or not np.isfinite(sigma):
        sigma = float(np.nanstd(radius_residuals_px)) or 1.0

    # Scale point size by local density (in x = range) so dense stacks get small points
    work["_size"] = _density_scaled_sizes(
        ranges_m, None,
        base_size=40.0, min_size=1.0, max_size=100.0,
        k=min(50, len(work) - 1),
    )
    n_pts = len(work)
    show = np.ones(n_pts, dtype=bool)
    if n_pts > _MAX_SCATTER_POINTS:
        rng = np.random.default_rng(42)
        show = np.zeros(n_pts, dtype=bool)
        show[rng.choice(n_pts, size=_MAX_SCATTER_POINTS, replace=False)] = True
    unique_cameras = sorted(set(work["_camera_id"]))
    fig, ax = plt.subplots(figsize=(10, 6))
    if n_pts >= _HEXBIN_MIN_POINTS:
        hb = ax.hexbin(
            ranges_m, radius_residuals_px,
            gridsize=min(50, max(20, n_pts // 200)),
            mincnt=1, cmap="Blues", alpha=0.45, edgecolors="none",
        )
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_cameras)))
    for i, cam_id in enumerate(unique_cameras):
        mask = (work["_camera_id"] == cam_id) & show
        if not np.any(mask):
            continue
        ax.scatter(
            work.loc[mask, "_range_m"],
            work.loc[mask, "_radius_residual_px"],
            c=[colors[i]],
            label=_camera_label(cam_id),
            alpha=0.6,
            s=work.loc[mask, "_size"],
            marker="s",
        )
    x_fine = np.linspace(ranges_m.min(), ranges_m.max(), 500)
    x_centers, lower_b, upper_b = _clumped_prediction_interval(
        ranges_m, radius_residuals_px, confidence=0.997, fallback_sigma=sigma
    )
    if len(x_centers) > 0:
        x_fine, lower_smooth, upper_smooth = _smooth_prediction_bounds(
            x_centers, lower_b, upper_b, x_fine=x_fine
        )
    else:
        x_fine, lower_smooth, upper_smooth = _ridge_prediction_interval(
            ranges_m, radius_residuals_px, x_fine, confidence=0.997
        )
    ax.fill_between(x_fine, lower_smooth, upper_smooth, alpha=0.15, color="gray", label="99.7% prediction interval")
    ax.plot(x_fine, upper_smooth, "k--", linewidth=1.5)
    ax.plot(x_fine, lower_smooth, "k--", linewidth=1.5)
    ax.axhline(0, color="k", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("True Range (m)", fontsize=12)
    ax.set_ylabel("Radius Residual (pixels)", fontsize=12)
    ax.set_title(f"{target_label} Radius Residuals by Camera", fontsize=14)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax


def plot_centroid_residuals_in_illumination_frame(
    df: pd.DataFrame,
    centroid_1sigma_px: Optional[float] = None,
    target_label: str = "Moon",
    save_path: Optional[Path | str] = None,
):
    """Plot X and Y centroid residuals (illumination frame) vs true range (m).

    Accepts the simulation metadata DataFrame directly: if true/out centroid columns
    (and camera columns) are missing, fill_pixel_metrics(df) is applied so a raw
    orchestrate df can be passed. Uses true_pos_*, true/out x/y centroids, and camera
    columns. Range = ‖true_pos‖; residuals = out_*_centroid - true_*_centroid. Rows
    with missing out centroids are dropped. Each subplot shows a 99.7% prediction interval.
    """
    need = [
        "true_pos_x", "true_pos_y", "true_pos_z",
        "true_x_centroid", "true_y_centroid",
        "out_x_centroid", "out_y_centroid",
    ] + list(_CAMERA_KEY_COLS)
    work = df.copy()
    if not all(c in work.columns for c in need):
        work = fill_pixel_metrics(work)
    for c in need:
        if c not in work.columns:
            raise ValueError(f"DataFrame must contain column {c!r} (or a df that fill_pixel_metrics can fill)")
    valid = (
        work["true_x_centroid"].notna() & work["true_y_centroid"].notna()
        & work["out_x_centroid"].notna() & work["out_y_centroid"].notna()
    )
    work = work.loc[valid].copy()
    if len(work) == 0:
        raise ValueError("No rows with both true and out centroid columns")
    work["_range_m"] = np.linalg.norm(
        work[["true_pos_x", "true_pos_y", "true_pos_z"]].values, axis=1
    )
    work["_centroid_x_residual_px"] = work["out_x_centroid"] - work["true_x_centroid"]
    work["_centroid_y_residual_px"] = work["out_y_centroid"] - work["true_y_centroid"]
    work["_camera_id"] = work.apply(_camera_id_from_row, axis=1)
    ranges_m = work["_range_m"].values
    centroid_x_px = work["_centroid_x_residual_px"].values
    centroid_y_px = work["_centroid_y_residual_px"].values
    sigma = centroid_1sigma_px
    if sigma is None or not np.isfinite(sigma):
        sigma = float(np.nanstd(np.concatenate([centroid_x_px, centroid_y_px]))) or 1.0

    # Scale point size by local density (in x = range) so dense stacks get small points
    k_nn = min(50, len(work) - 1)
    size_arr = _density_scaled_sizes(
        ranges_m, None, base_size=40.0, min_size=1.0, max_size=100.0, k=k_nn
    )
    work["_size_x"] = size_arr
    work["_size_y"] = size_arr
    n_pts = len(work)
    show = np.ones(n_pts, dtype=bool)
    if n_pts > _MAX_SCATTER_POINTS:
        rng = np.random.default_rng(42)
        show = np.zeros(n_pts, dtype=bool)
        show[rng.choice(n_pts, size=_MAX_SCATTER_POINTS, replace=False)] = True
    unique_cameras = sorted(set(work["_camera_id"]))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    for ax, y_col, size_col in (
        (ax1, "_centroid_x_residual_px", "_size_x"),
        (ax2, "_centroid_y_residual_px", "_size_y"),
    ):
        if n_pts >= _HEXBIN_MIN_POINTS:
            y_vals = work[y_col].values
            ax.hexbin(
                ranges_m, y_vals,
                gridsize=min(50, max(20, n_pts // 200)),
                mincnt=1, cmap="Blues", alpha=0.45, edgecolors="none",
            )
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_cameras)))
    for i, cam_id in enumerate(unique_cameras):
        mask = (work["_camera_id"] == cam_id) & show
        if not np.any(mask):
            continue
        ax1.scatter(
            work.loc[mask, "_range_m"],
            work.loc[mask, "_centroid_x_residual_px"],
            c=[colors[i]],
            label=_camera_label(cam_id),
            alpha=0.6,
            s=work.loc[mask, "_size_x"],
            marker="s",
        )
    for i, cam_id in enumerate(unique_cameras):
        mask = (work["_camera_id"] == cam_id) & show
        if not np.any(mask):
            continue
        ax2.scatter(
            work.loc[mask, "_range_m"],
            work.loc[mask, "_centroid_y_residual_px"],
            c=[colors[i]],
            alpha=0.6,
            s=work.loc[mask, "_size_y"],
            marker="s",
        )
    x_fine = np.linspace(ranges_m.min(), ranges_m.max(), 500)
    for ax, y_vals, label in (
        (ax1, centroid_x_px, "99.7% prediction interval"),
        (ax2, centroid_y_px, "99.7% prediction interval"),
    ):
        x_centers, lower_b, upper_b = _clumped_prediction_interval(
            ranges_m, y_vals, confidence=0.997, fallback_sigma=sigma
        )
        if len(x_centers) > 0:
            x_plot, lower_smooth, upper_smooth = _smooth_prediction_bounds(
                x_centers, lower_b, upper_b, x_fine=x_fine
            )
        else:
            x_plot, lower_smooth, upper_smooth = _ridge_prediction_interval(
                ranges_m, y_vals, x_fine, confidence=0.997
            )
        ax.fill_between(x_plot, lower_smooth, upper_smooth, alpha=0.15, color="gray", label=label)
        ax.plot(x_fine, upper_smooth, "k--", linewidth=1.5)
        ax.plot(x_fine, lower_smooth, "k--", linewidth=1.5)
        ax.axhline(0, color="k", linestyle="-", linewidth=0.5, alpha=0.5)
        ax.grid(True, alpha=0.3)
    ax1.set_ylabel("X Centroid Residual (pixels)", fontsize=11)
    ax2.set_ylabel("Y Centroid Residual (pixels)", fontsize=11)
    ax2.set_xlabel("True Range (m)", fontsize=12)
    ax1.set_title(
        f"{target_label} Centroid Residuals in Camera Frame", fontsize=14
    )
    ax1.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, (ax1, ax2)
