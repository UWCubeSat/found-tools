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

# Columns used for camera identity (must be present in df)
_CAMERA_KEY_COLS = ("cam_focal_length", "cam_x_resolution", "cam_y_resolution")


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


def _ridge_fit_curve(
    x: np.ndarray,
    y: np.ndarray,
    x_eval: np.ndarray,
    degree: int = 6,
    alpha: float = 10.0,
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
    coef = np.linalg.solve(gram, rhs)
    out = Xe @ coef
    if not np.any(np.isfinite(out)):
        out = np.full_like(x_eval, float(np.mean(y)))
    return out


def _smooth_prediction_bounds(
    x: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    n_fine: int = 500,
    degree: int = 6,
    alpha: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit prediction interval bounds with ridge regression (polynomial) and evaluate on a fine grid."""
    x_fine = np.linspace(float(x.min()), float(x.max()), n_fine)
    lower_smooth = _ridge_fit_curve(x, lower, x_fine, degree=degree, alpha=alpha)
    upper_smooth = _ridge_fit_curve(x, upper, x_fine, degree=degree, alpha=alpha)
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

    Uses simulation metadata DataFrame with true_pos_*, true_r_apparent, out_r_apparent,
    and camera columns. Range = ‖true_pos‖; radius residual = out_r_apparent - true_r_apparent.
    Points with different cameras (focal length, resolution) are plotted in different colors.
    Rows with missing out_r_apparent are dropped. The band is a range-dependent 3σ prediction
    interval.
    """
    need = ["true_pos_x", "true_pos_y", "true_pos_z", "true_r_apparent", "out_r_apparent"] + list(_CAMERA_KEY_COLS)
    for c in need:
        if c not in df.columns:
            raise ValueError(f"DataFrame must contain column {c!r}")
    valid = df["true_r_apparent"].notna() & df["out_r_apparent"].notna()
    work = df.loc[valid].copy()
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

    unique_cameras = sorted(set(work["_camera_id"]))
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_cameras)))
    for i, cam_id in enumerate(unique_cameras):
        mask = work["_camera_id"] == cam_id
        ax.scatter(
            work.loc[mask, "_range_m"],
            work.loc[mask, "_radius_residual_px"],
            c=[colors[i]],
            label=_camera_label(cam_id),
            alpha=0.6,
            s=40,
        )
    range_array = np.linspace(ranges_m.min(), ranges_m.max(), 100)
    lower, upper = _prediction_interval_bounds(
        ranges_m,
        radius_residuals_px,
        range_array,
        n_sigma=3.0,
        fallback_sigma=sigma,
    )
    x_smooth, lower_smooth, upper_smooth = _smooth_prediction_bounds(
        range_array, lower, upper
    )
    ax.fill_between(x_smooth, lower_smooth, upper_smooth, alpha=0.15, color="gray", label="99.7% prediction interval")
    ax.plot(x_smooth, upper_smooth, "k--", linewidth=1.5)
    ax.plot(x_smooth, lower_smooth, "k--", linewidth=1.5)
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

    Uses simulation metadata DataFrame with true_pos_*, true/out x/y centroids,
    and camera columns. Range = ‖true_pos‖; residuals = out_*_centroid - true_*_centroid.
    Points with different cameras are plotted in different colors. Rows with
    missing out_x_centroid or out_y_centroid are dropped. Each subplot shows a
    range-dependent 99.7% prediction interval.
    """
    need = [
        "true_pos_x", "true_pos_y", "true_pos_z",
        "true_x_centroid", "true_y_centroid",
        "out_x_centroid", "out_y_centroid",
    ] + list(_CAMERA_KEY_COLS)
    for c in need:
        if c not in df.columns:
            raise ValueError(f"DataFrame must contain column {c!r}")
    valid = (
        df["true_x_centroid"].notna() & df["true_y_centroid"].notna()
        & df["out_x_centroid"].notna() & df["out_y_centroid"].notna()
    )
    work = df.loc[valid].copy()
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

    unique_cameras = sorted(set(work["_camera_id"]))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_cameras)))
    for i, cam_id in enumerate(unique_cameras):
        mask = work["_camera_id"] == cam_id
        ax1.scatter(
            work.loc[mask, "_range_m"],
            work.loc[mask, "_centroid_x_residual_px"],
            c=[colors[i]],
            label=_camera_label(cam_id),
            alpha=0.6,
            s=40,
        )
    for i, cam_id in enumerate(unique_cameras):
        mask = work["_camera_id"] == cam_id
        ax2.scatter(
            work.loc[mask, "_range_m"],
            work.loc[mask, "_centroid_y_residual_px"],
            c=[colors[i]],
            alpha=0.6,
            s=40,
        )
    range_array = np.linspace(ranges_m.min(), ranges_m.max(), 100)
    for ax, y_vals, label in (
        (ax1, centroid_x_px, "99.7% prediction interval"),
        (ax2, centroid_y_px, "99.7% prediction interval"),
    ):
        lower, upper = _prediction_interval_bounds(
            ranges_m,
            y_vals,
            range_array,
            n_sigma=3.0,
            fallback_sigma=sigma,
        )
        x_smooth, lower_smooth, upper_smooth = _smooth_prediction_bounds(
            range_array, lower, upper
        )
        ax.fill_between(x_smooth, lower_smooth, upper_smooth, alpha=0.15, color="gray", label=label)
        ax.plot(x_smooth, upper_smooth, "k--", linewidth=1.5)
        ax.plot(x_smooth, lower_smooth, "k--", linewidth=1.5)
        ax.axhline(0, color="k", linestyle="-", linewidth=0.5, alpha=0.5)
        ax.grid(True, alpha=0.3)
    ax1.set_ylabel("X Centroid Residual (pixels)\n(along sun direction)", fontsize=11)
    ax2.set_ylabel("Y Centroid Residual (pixels)\n(perpendicular to sun)", fontsize=11)
    ax2.set_xlabel("True Range (m)", fontsize=12)
    ax1.set_title(
        f"{target_label} Centroid Residuals in Illumination Frame by Camera", fontsize=14
    )
    ax1.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, (ax1, ax2)
