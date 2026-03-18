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

from limb.simulation.analysis.metrics import column_summary, fill_pixel_metrics


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


def plot_column_summary(
    df: pd.DataFrame,
    column: str,
    n_points: int = 500,
    square_size: float = 4.0,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    n_bins: int = 10,
    confidence: float = 0.95,
    distance_column: str | None = None,
    ax: Optional[plt.Axes] = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot column vs distance with downsampled squares and prediction intervals.

    Uses :func:`~limb.simulation.analysis.metrics.column_summary` to compute
    per-distance-clump stats, then plots: (1) raw data downsampled to ``n_points``
    as tiny squares, (2) prediction interval (errorbar) per clump.

    Parameters
    ----------
    df : pd.DataFrame
        Simulation DataFrame with the column and distance (or true_pos_*).
    column : str
        Numeric column to plot on y-axis.
    n_points : int, optional
        Max number of data points to draw (random sample). Default 500.
    square_size : float, optional
        Scatter marker size in points² (matplotlib ``s``). Default 4.0 (tiny squares).
    title : str or None, optional
        Axes title. If None, defaults to ``column``.
    xlabel : str or None, optional
        x-axis label. If None, defaults to "Distance (m)".
    ylabel : str or None, optional
        y-axis label. If None, defaults to ``column``.
    n_bins : int, optional
        Number of distance bins for prediction intervals. Default 10.
    confidence : float, optional
        Prediction interval confidence level. Default 0.95.
    distance_column : str or None, optional
        Column used as distance; if None, use ‖(true_pos_x, true_pos_y, true_pos_z)‖.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw on. If None, a new figure is created.
    save_path : path-like or None, optional
        If set, save the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
        The figure (existing or new).
    """
    # Same distance logic as column_summary: build work with valid column + distance
    if column not in df.columns:
        raise ValueError(f"Column {column!r} not in DataFrame")
    work = df[[column]].copy().dropna(subset=[column])
    if work.empty:
        raise ValueError(f"Column {column!r} has no valid (non-NaN) values")
    if distance_column is not None:
        if distance_column not in df.columns:
            raise ValueError(f"Distance column {distance_column!r} not in DataFrame")
        work["_distance"] = df.loc[work.index, distance_column].values
    else:
        for c in ("true_pos_x", "true_pos_y", "true_pos_z"):
            if c not in df.columns:
                raise ValueError(
                    "Either provide distance_column or ensure true_pos_x, true_pos_y, true_pos_z exist"
                )
        pos = df.loc[work.index, ["true_pos_x", "true_pos_y", "true_pos_z"]].values
        work["_distance"] = np.linalg.norm(pos, axis=1)
    work = work.dropna(subset=["_distance"])
    if work.empty:
        raise ValueError("No rows with valid distance")

    x_all = work["_distance"].values
    y_all = work[column].values
    n_pts = len(x_all)

    # Downsample to n_points
    rng = np.random.default_rng()
    if n_pts <= n_points:
        plot_idx = np.arange(n_pts)
    else:
        plot_idx = rng.choice(n_pts, size=n_points, replace=False)
    x_plot = x_all[plot_idx]
    y_plot = y_all[plot_idx]

    # Per-clump stats (no table print)
    clumps = column_summary(
        df,
        column,
        confidence=confidence,
        dropna=True,
        n_bins=n_bins,
        distance_column=distance_column,
        print_results=False,
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    ax.scatter(
        x_plot,
        y_plot,
        marker="s",
        s=square_size,
        color="tab:blue",
        alpha=0.6,
        edgecolors="none",
        label=f"data (n={len(plot_idx)} shown)" if n_pts > n_points else "data",
    )

    # Prediction intervals: second-order polynomial fit; center ± half-width so lines don't cross
    x_bin = np.array([(r["distance_lo"] + r["distance_hi"]) / 2 for r in clumps])
    pi_lo = np.array([r["pi_lower"] for r in clumps])
    pi_hi = np.array([r["pi_upper"] for r in clumps])
    order = np.argsort(x_bin)
    x_sorted = x_bin[order]
    center = (pi_lo[order] + pi_hi[order]) / 2
    half_width = (pi_hi[order] - pi_lo[order]) / 2
    coeff_center = np.polyfit(x_sorted, center, 2)
    coeff_hw = np.polyfit(x_sorted, half_width, 2)
    x_min, x_max = x_all.min(), x_all.max()
    pad = 0.05 * (x_max - x_min) if x_max > x_min else 1.0
    # Confidence interval curves from least to greatest distance (full data range)
    x_curve = np.linspace(x_min, x_max, 200)
    center_curve = np.polyval(coeff_center, x_curve)
    hw_curve = np.maximum(np.polyval(coeff_hw, x_curve), 0.0)
    ax.plot(
        x_curve,
        center_curve - hw_curve,
        linestyle="-",
        color="black",
        linewidth=1.5,
        label=f"{int(round(confidence * 100))}% prediction interval",
    )
    ax.plot(
        x_curve,
        center_curve + hw_curve,
        linestyle="-",
        color="black",
        linewidth=1.5,
    )

    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_xlabel(xlabel if xlabel is not None else "Distance (m)")
    ax.set_ylabel(ylabel if ylabel is not None else column)
    ax.set_title(title if title is not None else column)
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
    return fig

