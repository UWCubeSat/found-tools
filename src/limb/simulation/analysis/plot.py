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
from matplotlib.lines import Line2D
from scipy import stats
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree

from limb.simulation.analysis.metrics import (
    column_summary,
    fill_pixel_metrics,
    split_df_by_camera,
)


# Columns used for camera identity (must be present in df)
_CAMERA_KEY_COLS = ("cam_focal_length", "cam_x_resolution", "cam_y_resolution")


def _hyper3(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Three-parameter rectangular hyperbola: a + b / (x + c). Requires x + c > 0."""
    return a + b / (x + c)


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


def _horizontal_fov_deg_from_row(row: pd.Series) -> float:
    """Full horizontal field of view in degrees from pinhole geometry."""
    if "cam_x_pixel_pitch" not in row.index:
        raise ValueError("Row must contain cam_x_pixel_pitch to compute FOV")
    sensor_w = float(row["cam_x_resolution"]) * float(row["cam_x_pixel_pitch"])
    f = float(row["cam_focal_length"])
    if f <= 0:
        raise ValueError("cam_focal_length must be positive for FOV")
    return float(np.rad2deg(2.0 * np.arctan(sensor_w / (2.0 * f))))


def _resolution_key_from_row(row: pd.Series) -> tuple[int, int]:
    return (int(row["cam_x_resolution"]), int(row["cam_y_resolution"]))


def plot_column_summary_by_camera(
    df: pd.DataFrame,
    column: str,
    *,
    use_fit_line: bool = False,
    n_bins: int = 10,
    confidence: float = 0.95,
    distance_column: str | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    ax: Optional[plt.Axes] = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot per-distance-bin mean of *column* vs range, split by camera configuration.

    For each camera group from :func:`~limb.simulation.analysis.metrics.split_df_by_camera`,
    runs :func:`~limb.simulation.analysis.metrics.column_summary` (no printed tables).
    x-values are bin mid-distances ``(distance_lo + distance_hi) / 2``; y-values are
    clump means.

    **Encoding**

    - **Color** — horizontal FOV (degrees), derived from focal length and sensor width.
    - **Marker** (point mode) or **linestyle** (fit-line mode) — resolution ``(width × height)``.

    **Legend**

    Two legend boxes: FOV → color, and resolution → marker or linestyle depending on
    ``use_fit_line``.

    Parameters
    ----------
    df : pd.DataFrame
        Simulation DataFrame with camera columns, the value *column*, and distance
        (via ``distance_column`` or ``true_pos_*``); see :func:`column_summary`.
    column : str
        Numeric column whose per-bin means are plotted on the y-axis.
    use_fit_line : bool, optional
        If False (default), plot one scatter series per camera (color × marker).
        If True, plot a linear least-squares fit through each camera's bin means
        over the same x midpoints (color × linestyle).
    n_bins, confidence, distance_column
        Passed to :func:`column_summary`.
    title, xlabel, ylabel : str or None
        Axis labels; if None, defaults are derived from *column* and distance.
    ax : matplotlib.axes.Axes or None
        Axes to draw on; if None, a new figure is created.
    save_path : path-like or None
        If set, save the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
        The figure used for plotting.
    """
    by_cam = split_df_by_camera(df)
    if not by_cam:
        raise ValueError(
            "split_df_by_camera returned no groups (empty DataFrame or missing camera columns)."
        )

    # Collect FOV and resolution keys across cameras for stable color / marker maps
    fov_by_cam: dict[tuple[float, int, int], float] = {}
    res_by_cam: dict[tuple[float, int, int], tuple[int, int]] = {}
    for cam_id, sub in by_cam.items():
        row0 = sub.iloc[0]
        fov_by_cam[cam_id] = _horizontal_fov_deg_from_row(row0)
        res_by_cam[cam_id] = _resolution_key_from_row(row0)

    unique_fovs = sorted({round(v, 6) for v in fov_by_cam.values()})
    # Map rounded FOV back to a representative label
    fov_to_display: dict[float, str] = {}
    for f in unique_fovs:
        fov_to_display[f] = f"{f:g}°"

    unique_res = sorted({res_by_cam[k] for k in by_cam})

    cmap = plt.get_cmap("tab10")
    fov_color: dict[float, tuple] = {
        f: cmap(i % cmap.N) for i, f in enumerate(unique_fovs)
    }

    point_markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "p", "h", "8"]
    res_marker: dict[tuple[int, int], str] = {
        r: point_markers[i % len(point_markers)] for i, r in enumerate(unique_res)
    }

    linestyles = ["-", "--", "-.", ":"]
    res_linestyle: dict[tuple[int, int], str] = {
        r: linestyles[i % len(linestyles)] for i, r in enumerate(unique_res)
    }

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    # Plot each camera's column_summary series
    for cam_id, sub in by_cam.items():
        fov_r = round(fov_by_cam[cam_id], 6)
        color = fov_color[fov_r]
        res_k = res_by_cam[cam_id]
        marker = res_marker[res_k]
        ls = res_linestyle[res_k]

        clumps = column_summary(
            sub,
            column,
            confidence=confidence,
            dropna=True,
            n_bins=n_bins,
            distance_column=distance_column,
            print_results=False,
        )
        if not clumps:
            continue
        x_mid = np.array([(r["distance_lo"] + r["distance_hi"]) / 2.0 for r in clumps])
        y_mean = np.array([r["mean"] for r in clumps])
        order = np.argsort(x_mid)
        x_mid = x_mid[order]
        y_mean = y_mean[order]

        if use_fit_line:
            if x_mid.size >= 2:
                m, b = np.polyfit(x_mid, y_mean, 1)
                x_line = np.linspace(float(x_mid.min()), float(x_mid.max()), 200)
                y_line = m * x_line + b
                ax.plot(
                    x_line,
                    y_line,
                    linestyle=ls,
                    color=color,
                    linewidth=2.0,
                )
            elif x_mid.size == 1:
                ax.plot(
                    x_mid,
                    y_mean,
                    linestyle=ls,
                    color=color,
                    marker=marker,
                    markersize=6,
                )
        else:
            ax.scatter(
                x_mid,
                y_mean,
                marker=marker,
                s=36,
                color=color,
                edgecolors="black",
                linewidths=0.4,
                alpha=0.85,
                zorder=3,
            )

    # Compound legend: FOV (color) and resolution (marker or linestyle)
    fov_handles: list[Line2D] = []
    for f in unique_fovs:
        if use_fit_line:
            fov_handles.append(
                Line2D(
                    [0, 1],
                    [0, 0],
                    linestyle="-",
                    color=fov_color[f],
                    linewidth=2.5,
                    label=fov_to_display[f],
                )
            )
        else:
            fov_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="s",
                    linestyle="None",
                    markersize=10,
                    markerfacecolor=fov_color[f],
                    markeredgecolor="black",
                    label=fov_to_display[f],
                )
            )
    if use_fit_line:
        res_handles = [
            Line2D(
                [0],
                [0],
                color="0.35",
                linestyle=res_linestyle[r],
                linewidth=2.0,
                label=f"{r[0]}×{r[1]}",
            )
            for r in unique_res
        ]
    else:
        res_handles = [
            Line2D(
                [0],
                [0],
                marker=res_marker[r],
                linestyle="None",
                markersize=9,
                markerfacecolor="0.85",
                markeredgecolor="black",
                label=f"{r[0]}×{r[1]}",
            )
            for r in unique_res
        ]

    leg1 = ax.legend(
        handles=fov_handles,
        title="FOV",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        framealpha=0.9,
    )
    ax.add_artist(leg1)
    leg2 = ax.legend(
        handles=res_handles,
        title="Resolution" if not use_fit_line else "Resolution (line style)",
        loc="lower left",
        bbox_to_anchor=(1.02, 0.0),
        borderaxespad=0.0,
        framealpha=0.9,
    )

    ax.set_xlabel(
        xlabel
        if xlabel is not None
        else (
            f"Range: {distance_column} (m)"
            if distance_column is not None
            else "Range: ‖true position‖ (m)"
        )
    )
    ax.set_ylabel(ylabel if ylabel is not None else f"Mean {column} (per bin)")
    mode = "linear fit" if use_fit_line else "bin means"
    default_title = f"{column} vs range by camera ({mode})"
    ax.set_title(title if title is not None else default_title)
    ax.grid(True, alpha=0.3)
    # Leave room outside axes for the two anchored legends
    fig.tight_layout(rect=(0.0, 0.0, 0.74, 1.0))
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


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
    raw_intervals: bool = False,
    ax: Optional[plt.Axes] = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot column vs distance with downsampled squares and prediction intervals.

    Uses :func:`~limb.simulation.analysis.metrics.column_summary` to compute
    per-distance-clump stats, then plots: (1) raw data downsampled to ``n_points``
    as tiny squares, (2) prediction interval bounds (smooth fit or raw per-bin).

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
    raw_intervals : bool, optional
        If True, plot only per-bin prediction intervals as vertical error bars (no lines
        between bins, no smooth fit). If False (default), only the smooth hyperbolic/polynomial
        bounds (no per-bin error bars).
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

    # Prediction intervals: either raw per-bin bounds or smooth fit (hyperbolic / polynomial)
    x_bin = np.array([(r["distance_lo"] + r["distance_hi"]) / 2 for r in clumps])
    pi_lo = np.array([r["pi_lower"] for r in clumps])
    pi_hi = np.array([r["pi_upper"] for r in clumps])
    n_per_bin = np.array([r["n"] for r in clumps], dtype=np.float64)
    order = np.argsort(x_bin)
    x_sorted = x_bin[order]
    pi_lo_sorted = pi_lo[order]
    pi_hi_sorted = pi_hi[order]
    center = (pi_lo_sorted + pi_hi_sorted) / 2
    half_width = (pi_hi_sorted - pi_lo_sorted) / 2

    x_min, x_max = x_all.min(), x_all.max()
    pad = 0.05 * (x_max - x_min) if x_max > x_min else 1.0

    if raw_intervals:
        # No fit, no lines between bins: only per-bin error bars (caps show PI at each range)
        yerr_lo = center - pi_lo_sorted
        yerr_hi = pi_hi_sorted - center
        ax.errorbar(
            x_sorted,
            center,
            yerr=[yerr_lo, yerr_hi],
            fmt="none",
            color="black",
            ecolor="black",
            elinewidth=1.5,
            capsize=4,
            capthick=1.5,
            zorder=5,
            label=f"{int(round(confidence * 100))}% prediction interval",
        )
    else:
        # Smooth fit: weight by bin size, tie-break by range
        n_sorted = n_per_bin[order]
        x_span = x_sorted.max() - x_sorted.min()
        range_tie = (x_sorted - x_sorted.min()) / (x_span + 1e-10)
        w = np.maximum(n_sorted + range_tie, 1.0)
        sigma = 1.0 / np.sqrt(w)
        x_curve = np.linspace(x_min, x_max, 200)

        def _fit_hyper3(
            x: np.ndarray, y: np.ndarray, sig: np.ndarray, x_eval: np.ndarray
        ) -> tuple[bool, np.ndarray]:
            """Try y = a + b/(x+c); on failure use degree-2 polynomial. Returns (ok_hyper, params)."""
            w_poly = 1.0 / (sig.astype(np.float64) ** 2)
            if len(x) < 3:
                coeff = np.polyfit(x, y, min(2, len(x)), w=w_poly)
                return False, coeff
            x_min_all = float(np.minimum(np.min(x), np.min(x_eval)))
            x_max_all = float(np.maximum(np.max(x), np.max(x_eval)))
            x_rng = max(x_max_all - x_min_all, 1e-9)
            # Keep x + c > 0 everywhere we evaluate
            c_lo = -x_min_all + max(1e-9, 1e-6 * max(abs(x_min_all), 1.0))
            a0 = float(np.mean(y))
            b0 = float((y[0] - y[-1]) * (x_min_all + max(x_span, 1.0))) if len(y) > 1 else 0.0
            # Start inside feasible region: x_min + c0 = small positive fraction of range
            c0 = float(-x_min_all + 0.05 * x_rng)
            if c0 <= c_lo:
                c0 = c_lo + 0.05 * x_rng
            try:
                (a, b, c), _ = curve_fit(
                    _hyper3,
                    x,
                    y,
                    p0=[a0, b0, c0],
                    sigma=sig,
                    bounds=([-np.inf, -np.inf, c_lo], [np.inf, np.inf, np.inf]),
                )
                return True, np.array([float(a), float(b), float(c)])
            except (ValueError, RuntimeError):
                coeff = np.polyfit(x, y, 2, w=w_poly)
                return False, coeff

        def _eval_fit(ok_hyper: bool, params: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
            if ok_hyper:
                return _hyper3(x_eval, *params)
            return np.polyval(params, x_eval)

        ok_center, params_center = _fit_hyper3(x_sorted, center, sigma, x_curve)
        ok_hw, params_hw = _fit_hyper3(x_sorted, half_width, sigma, x_curve)
        center_curve = _eval_fit(ok_center, params_center, x_curve)
        hw_curve = np.maximum(_eval_fit(ok_hw, params_hw, x_curve), 0.0)
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

    # Y-axis: 1.3 × largest CI half-width, centered at 0
    max_half_width = float(np.max(half_width)) if len(half_width) > 0 else 1.0
    y_half = 1.3 * max_half_width
    ax.set_ylim(-y_half, y_half)

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

