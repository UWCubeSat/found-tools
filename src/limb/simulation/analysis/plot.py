"""Plot a window of the image around a point with edge points and optional conic.

Also provides OpNav validation plots: radius residuals vs range and centroid
residuals in the illumination frame.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.image import imread
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from scipy import stats
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree

from limb.simulation.analysis.metrics import (
    column_summary,
    fill_pixel_metrics,
    horizontal_fov_deg_from_row,
    split_df_by_camera,
    split_df_by_column_value,
)


# Columns used for camera identity (must be present in df)
_CAMERA_KEY_COLS = ("cam_focal_length", "cam_x_resolution", "cam_y_resolution")


def _hyper3(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Three-parameter rectangular hyperbola: a + b / (x + c). Requires x + c > 0."""
    return a + b / (x + c)


def _softplus(x: np.ndarray, beta: float, m: float, b: float, L: float) -> np.ndarray:
    """L + (1/β) log(1 + exp(β (m x + b))); stable for :func:`scipy.optimize.curve_fit`."""
    x = np.asarray(x, dtype=np.float64)
    u = np.clip(float(beta) * (float(m) * x + float(b)), -700.0, 700.0)
    return float(L) + (1.0 / float(beta)) * np.log1p(np.exp(u))


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


def _resolution_key_from_row(row: pd.Series) -> tuple[int, int]:
    return (int(row["cam_x_resolution"]), int(row["cam_y_resolution"]))


def _distance_series(df: pd.DataFrame, distance_column: str | None) -> pd.Series:
    """Per-row distance consistent with :func:`column_summary`."""
    if distance_column is not None:
        if distance_column not in df.columns:
            raise ValueError(f"Distance column {distance_column!r} not in DataFrame")
        return df[distance_column]
    for c in ("true_pos_x", "true_pos_y", "true_pos_z"):
        if c not in df.columns:
            raise ValueError(
                "Either provide distance_column or ensure true_pos_x, true_pos_y, true_pos_z exist"
            )
    pos = df[["true_pos_x", "true_pos_y", "true_pos_z"]].values
    return pd.Series(np.linalg.norm(pos, axis=1), index=df.index, dtype="float64")


def _filter_df_by_distance_range(
    df: pd.DataFrame,
    distance_column: str | None,
    distance_min: float | None,
    distance_max: float | None,
) -> pd.DataFrame:
    """Keep rows whose distance is in ``[distance_min, distance_max]`` (inclusive)."""
    if distance_min is None and distance_max is None:
        return df
    dist = _distance_series(df, distance_column)
    mask = dist.notna()
    if distance_min is not None:
        mask &= dist >= float(distance_min)
    if distance_max is not None:
        mask &= dist <= float(distance_max)
    return df.loc[mask].reset_index(drop=True)


def _normalize_resolution_filter(
    resolutions: Sequence[int | tuple[int, int]] | None,
) -> set[tuple[int, int]] | None:
    if resolutions is None:
        return None
    out: set[tuple[int, int]] = set()
    for r in resolutions:
        if isinstance(r, (int, np.integer)):
            ri = int(r)
            out.add((ri, ri))
        else:
            wx, wy = r
            out.add((int(wx), int(wy)))
    return out


def _fov_matches_any(fov_deg: float, fovs: Sequence[float], atol: float) -> bool:
    return any(np.isclose(fov_deg, float(t), rtol=0.0, atol=atol) for t in fovs)


def _resolve_include_fovs(
    fovs: Sequence[float] | None,
    include_fovs: Sequence[float] | None,
) -> Sequence[float] | None:
    """Return the FOV include-list, or None for no filtering."""
    if fovs is not None and include_fovs is not None:
        raise ValueError("Pass only one of fovs and include_fovs.")
    return include_fovs if include_fovs is not None else fovs


def _effective_poly_degree(n_points: int, requested: int) -> int:
    requested = max(0, int(requested))
    if n_points <= 0:
        return 0
    return min(requested, max(0, n_points - 1))


def _availability_fit_line(
    x_mid: np.ndarray,
    avail: np.ndarray,
    *,
    fit_poly_degree: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares polynomial through bin midpoints; result clipped to ``0–100`` %.

    Degree is ``min(2, fit_poly_degree)`` capped by :func:`_effective_poly_degree` (so
    availability plots use **at most** a quadratic, even when ``fit_poly_degree`` is
    larger elsewhere in the API).
    """
    x_mid = np.asarray(x_mid, dtype=np.float64)
    avail = np.asarray(avail, dtype=np.float64)
    x_lo, x_hi = float(x_mid.min()), float(x_mid.max())
    x_line = (
        np.linspace(x_lo, x_hi, 200)
        if x_hi > x_lo
        else np.full(200, x_lo)
    )
    deg_req = min(max(0, int(fit_poly_degree)), 2)
    deg_eff = _effective_poly_degree(int(x_mid.size), deg_req)
    coeffs = np.polyfit(x_mid, avail, deg_eff)
    y_line = np.clip(np.polyval(coeffs, x_line), 0.0, 100.0)
    return x_line, y_line


def _resolve_availability_ylim(
    y_min: float | None,
    y_max: float | None,
) -> tuple[float, float]:
    """Return (lo, hi) availability % for ``ax.set_ylim``; None uses 0 and/or 100."""
    lo = 0.0 if y_min is None else float(y_min)
    hi = 100.0 if y_max is None else float(y_max)
    if lo < 0.0 or hi > 100.0:
        raise ValueError(
            f"availability y-limits must be within [0, 100] percent; got [{lo}, {hi}]."
        )
    if hi <= lo:
        raise ValueError(
            f"availability_y_max ({hi}) must be greater than availability_y_min ({lo})."
        )
    return lo, hi


def _proportion_wilson_ci_pct(k: int, n: int, confidence: float) -> tuple[float, float]:
    """Wilson CI for binomial proportion *k*/*n*; return ``(low_pct, high_pct)``."""
    if n <= 0:
        return 0.0, 100.0
    ci = stats.binomtest(int(k), int(n)).proportion_ci(
        confidence_level=float(confidence),
        method="wilson",
    )
    return float(ci.low * 100.0), float(ci.high * 100.0)


def _filtered_camera_subsets(
    df: pd.DataFrame,
    *,
    distance_column: str | None,
    distance_min: float | None,
    distance_max: float | None,
    fovs: Sequence[float] | None,
    fov_match_atol: float,
    resolutions: Sequence[int | tuple[int, int]] | None,
) -> tuple[
    dict[tuple[float, int, int], pd.DataFrame],
    dict[tuple[float, int, int], float],
    dict[tuple[float, int, int], tuple[int, int]],
]:
    """Split *df* by camera and apply distance / FOV / resolution filters."""
    df_work = _filter_df_by_distance_range(
        df, distance_column, distance_min, distance_max
    )
    if df_work.empty:
        raise ValueError(
            "No rows left after applying distance_min / distance_max filters."
        )

    by_cam_all = split_df_by_camera(df_work)
    if not by_cam_all:
        raise ValueError(
            "split_df_by_camera returned no groups (empty DataFrame or missing camera columns)."
        )

    res_allow = _normalize_resolution_filter(resolutions)

    fov_by_cam: dict[tuple[float, int, int], float] = {}
    res_by_cam: dict[tuple[float, int, int], tuple[int, int]] = {}
    for cam_id, sub in by_cam_all.items():
        row0 = sub.iloc[0]
        fov_by_cam[cam_id] = horizontal_fov_deg_from_row(row0)
        res_by_cam[cam_id] = _resolution_key_from_row(row0)

    by_cam: dict[tuple[float, int, int], pd.DataFrame] = {}
    for cam_id, sub in by_cam_all.items():
        fov_deg = fov_by_cam[cam_id]
        res_k = res_by_cam[cam_id]
        if fovs is not None and not _fov_matches_any(fov_deg, fovs, fov_match_atol):
            continue
        if res_allow is not None and res_k not in res_allow:
            continue
        by_cam[cam_id] = sub

    if not by_cam:
        raise ValueError(
            "No camera groups left after fovs / resolutions filters "
            "(check values and fov_match_atol)."
        )

    return by_cam, fov_by_cam, res_by_cam


def plot_column_availability_by_camera(
    df: pd.DataFrame,
    column: str,
    *,
    availability_bound: float,
    fit_poly_degree: int = 1,
    n_bins: int = 10,
    confidence: float = 0.95,
    distance_column: str | None = None,
    distance_min: float | None = None,
    distance_max: float | None = None,
    fovs: Sequence[float] | None = None,
    include_fovs: Sequence[float] | None = None,
    fov_match_atol: float = 1e-3,
    resolutions: Sequence[int | tuple[int, int]] | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    zoom_x_to_data: bool = True,
    x_pad_fraction: float = 0.02,
    availability_y_min: float | None = None,
    availability_y_max: float | None = None,
    plot_bin_points: bool = False,
    bin_error_confidence: float = 0.95,
    ax: Optional[plt.Axes] = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot **availability** vs range per camera (quadratic fit by default, not bin connectors).

    For each camera, uses :func:`~limb.simulation.analysis.metrics.column_summary` with
    ``availability_below=availability_bound``. **Availability** in each distance bin is
    ``100 * (count of values with column < bound) / n`` (strict ``<``).
    **100%** means every point in that bin satisfies ``column < availability_bound``;
    **0%** means none do.

    By default the x-axis is **zoomed** to the span where fit lines are drawn (per-camera
    distance extent), with a small pad. The y-axis is **linear** in percent; use
    ``availability_y_min`` / ``availability_y_max`` to **crop** the vertical range
    (defaults: full ``0–100`` when both are None; a lone ``None`` uses ``0`` or ``100``
    for that bound).

    Bin midpoints are fit with a **quadratic** (degree at most 2; least squares through
    per-bin availability, then clipped to ``0–100``). The requested ``fit_poly_degree``
    is capped at 2 for this plot. The smooth curve uses the camera's resolution linestyle.
    Optionally ``plot_bin_points`` overlays per-bin **points** with **Wilson** confidence
    intervals on the availability proportion (see ``bin_error_confidence``).

    **Encoding:** color = FOV (degrees), linestyle = resolution ``(width × height)``.

    Parameters
    ----------
    df, column
        Same as :func:`plot_column_summary_by_camera`.
    availability_bound : float
        Strict upper bound: points with ``column < availability_bound`` count as
        "available" toward the percentage.
    fit_poly_degree : int, optional
        Maximum polynomial degree for the smooth curve (capped at **2** for this plot).
        Further capped per camera to ``n_bins - 1`` when there are fewer bins than the
        requested degree.
    n_bins, confidence, distance_column
        Passed to :func:`column_summary`.
    distance_min, distance_max, fovs, include_fovs, fov_match_atol, resolutions
        Same filtering as :func:`plot_column_summary_by_camera`. ``fovs`` and
        ``include_fovs`` are synonyms (**include-only**; other FOVs excluded); pass
        only one.
    zoom_x_to_data : bool, optional
        If True (default), set x-limits to the min/max range covered by the plotted
        fit lines (plus ``x_pad_fraction`` padding). If False, matplotlib autoscale.
    x_pad_fraction : float, optional
        Pad added to each side of the data x-span as a fraction of that span
        (default 0.02). Ignored if ``zoom_x_to_data`` is False.
    availability_y_min, availability_y_max : float or None, optional
        Crop the y-axis to this availability range (percent). ``None`` uses ``0`` for
        *min* and ``100`` for *max*. Values are plain linear percent (no warping).
    plot_bin_points : bool, optional
        If True, draw each bin's availability at the distance midpoint as a point with
        asymmetric vertical error bars (Wilson interval for ``n_below`` / ``n``).
    bin_error_confidence : float, optional
        Confidence level for Wilson intervals when ``plot_bin_points`` is True
        (default 0.95).
    title, xlabel, ylabel, ax, save_path
        Plot labels and I/O; defaults describe availability and the bound.

    Returns
    -------
    matplotlib.figure.Figure
    """
    y_lo, y_hi = _resolve_availability_ylim(availability_y_min, availability_y_max)

    fovs_f = _resolve_include_fovs(fovs, include_fovs)
    by_cam, fov_by_cam, res_by_cam = _filtered_camera_subsets(
        df,
        distance_column=distance_column,
        distance_min=distance_min,
        distance_max=distance_max,
        fovs=fovs_f,
        fov_match_atol=fov_match_atol,
        resolutions=resolutions,
    )

    unique_fovs = sorted({round(fov_by_cam[cid], 6) for cid in by_cam})
    fov_to_display: dict[float, str] = {f: f"{f:g}°" for f in unique_fovs}
    unique_res = sorted({res_by_cam[cid] for cid in by_cam})

    cmap = plt.get_cmap("tab10")
    fov_color: dict[float, tuple] = {
        f: cmap(i % cmap.N) for i, f in enumerate(unique_fovs)
    }

    linestyles = ["-", "--", "-.", ":"]
    res_linestyle: dict[tuple[int, int], str] = {
        r: linestyles[i % len(linestyles)] for i, r in enumerate(unique_res)
    }

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    x_min_data = np.inf
    x_max_data = -np.inf

    for cam_id, sub in by_cam.items():
        fov_r = round(fov_by_cam[cam_id], 6)
        color = fov_color[fov_r]
        res_k = res_by_cam[cam_id]
        ls = res_linestyle[res_k]

        clumps = column_summary(
            sub,
            column,
            confidence=confidence,
            dropna=True,
            n_bins=n_bins,
            distance_column=distance_column,
            print_results=False,
            availability_below=float(availability_bound),
        )
        if not clumps:
            continue
        x_mid = np.array([(r["distance_lo"] + r["distance_hi"]) / 2.0 for r in clumps])
        avail = np.array([float(r["pct_below"]) for r in clumps])
        order = np.argsort(x_mid)
        x_mid = x_mid[order]
        avail = avail[order]
        n_arr = np.array([int(r["n"]) for r in clumps], dtype=np.int64)[order]
        k_arr = np.array([int(r["n_below"]) for r in clumps], dtype=np.int64)[order]

        if x_mid.size >= 1:
            x_line, y_line = _availability_fit_line(
                x_mid, avail, fit_poly_degree=fit_poly_degree
            )
            x_min_data = min(x_min_data, float(np.min(x_line)))
            x_max_data = max(x_max_data, float(np.max(x_line)))
            ax.plot(
                x_line,
                y_line,
                linestyle=ls,
                color=color,
                linewidth=2.0,
                alpha=0.9,
                zorder=3,
            )
            x_min_data = min(x_min_data, float(np.min(x_mid)))
            x_max_data = max(x_max_data, float(np.max(x_mid)))

        if plot_bin_points and x_mid.size >= 1:
            ci_lo = np.empty_like(avail, dtype=np.float64)
            ci_hi = np.empty_like(avail, dtype=np.float64)
            for i in range(x_mid.size):
                ci_lo[i], ci_hi[i] = _proportion_wilson_ci_pct(
                    int(k_arr[i]), int(n_arr[i]), bin_error_confidence
                )
            yerr_lo = np.maximum(0.0, avail - ci_lo)
            yerr_hi = np.maximum(0.0, ci_hi - avail)
            ax.errorbar(
                x_mid,
                avail,
                yerr=[yerr_lo, yerr_hi],
                fmt="o",
                color=color,
                ecolor=color,
                elinewidth=1.2,
                capsize=3,
                capthick=1.0,
                markersize=5,
                markeredgecolor="black",
                markeredgewidth=0.5,
                linestyle="none",
                zorder=5,
                alpha=0.9,
            )

    if zoom_x_to_data and np.isfinite(x_min_data) and np.isfinite(x_max_data):
        span = x_max_data - x_min_data
        pad = float(x_pad_fraction) * (span if span > 0 else max(abs(x_max_data), 1.0) * 0.01)
        ax.set_xlim(x_min_data - pad, x_max_data + pad)

    fov_handles: list[Line2D] = [
        Line2D(
            [0, 1],
            [0, 0],
            linestyle="-",
            color=fov_color[f],
            linewidth=2.5,
            label=fov_to_display[f],
        )
        for f in unique_fovs
    ]
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
    if plot_bin_points:
        pct_e = int(round(float(bin_error_confidence) * 100))
        res_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markersize=6,
                markerfacecolor="0.75",
                markeredgecolor="black",
                label=f"Bin % ± {pct_e}% Wilson CI",
            )
        )

    leg1 = ax.legend(
        handles=fov_handles,
        title="FOV",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        framealpha=0.9,
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=res_handles,
        title="Resolution",
        loc="lower left",
        bbox_to_anchor=(1.02, 0.0),
        borderaxespad=0.0,
        framealpha=0.9,
    )

    range_bits = []
    if distance_min is not None:
        range_bits.append(f"≥{distance_min:g}")
    if distance_max is not None:
        range_bits.append(f"≤{distance_max:g}")
    range_suffix = f" [{', '.join(range_bits)}]" if range_bits else ""

    ax.set_xlabel(
        xlabel
        if xlabel is not None
        else (
            f"Range (m){range_suffix}"
        )
    )
    default_yl = (
        f"Availability (%)"
    )
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    else:
        crop_note = (
            f" (axis {y_lo:g}–{y_hi:g}%)"
            if (y_lo > 0.0 or y_hi < 100.0)
            else ""
        )
        ax.set_ylabel(default_yl + crop_note)
    default_title = (
        f"Availability vs Range (quadratic fit, {column} < "
        f"{availability_bound:g}){range_suffix}"
    )
    ax.set_title(title if title is not None else default_title)
    ax.set_ylim(y_lo, y_hi)
    ax.yaxis.set_major_locator(MaxNLocator(nbins="auto", steps=[1, 2, 2.5, 5, 10]))
    ax.grid(True, alpha=0.3)
    fig.tight_layout(rect=(0.0, 0.0, 0.74, 1.0))
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def _filtered_category_subsets(
    df: pd.DataFrame,
    *,
    category_column: str,
    distance_column: str | None,
    distance_min: float | None,
    distance_max: float | None,
) -> dict[Any, pd.DataFrame]:
    df_work = _filter_df_by_distance_range(
        df, distance_column, distance_min, distance_max
    )
    if category_column not in df_work.columns:
        raise ValueError(
            f"category_column {category_column!r} not in DataFrame columns"
        )
    return split_df_by_column_value(df_work, category_column)


def plot_column_summary_by_category(
    df: pd.DataFrame,
    column: str,
    *,
    category_column: str,
    category_legend_title: str = "Category",
    use_fit_line: bool = False,
    fit_poly_degree: int = 1,
    n_bins: int = 10,
    confidence: float = 0.95,
    distance_column: str | None = None,
    distance_min: float | None = None,
    distance_max: float | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    ax: Optional[plt.Axes] = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot per-distance-bin mean of *column* vs range, one series per *category_column* value.

    Same binning and optional polynomial fit as :func:`plot_column_summary_by_camera`,
    but **color** encodes category (e.g. regression algorithm); no FOV/resolution split.
    """
    by_cat = _filtered_category_subsets(
        df,
        category_column=category_column,
        distance_column=distance_column,
        distance_min=distance_min,
        distance_max=distance_max,
    )
    if not by_cat:
        raise ValueError(
            f"No rows after distance filter or no categories in {category_column!r}"
        )

    unique_cats = sorted(by_cat.keys(), key=lambda k: (pd.isna(k), str(k)))
    cmap = plt.get_cmap("tab10")
    cat_color: dict[Any, tuple] = {
        c: cmap(i % cmap.N) for i, c in enumerate(unique_cats)
    }
    point_markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "p", "h", "8"]
    cat_marker: dict[Any, str] = {
        c: point_markers[i % len(point_markers)] for i, c in enumerate(unique_cats)
    }

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    for cat_key in unique_cats:
        sub = by_cat[cat_key]
        color = cat_color[cat_key]
        marker = cat_marker[cat_key]

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
            if x_mid.size >= 1:
                deg_eff = _effective_poly_degree(int(x_mid.size), fit_poly_degree)
                coeffs = np.polyfit(x_mid, y_mean, deg_eff)
                x_lo, x_hi = float(x_mid.min()), float(x_mid.max())
                x_line = (
                    np.linspace(x_lo, x_hi, 200)
                    if x_hi > x_lo
                    else np.full(200, x_lo)
                )
                y_line = np.polyval(coeffs, x_line)
                ax.plot(
                    x_line,
                    y_line,
                    linestyle="-",
                    color=color,
                    linewidth=2.0,
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

    handles: list[Line2D] = []
    for c in unique_cats:
        label = "" if pd.isna(c) else str(c)
        if use_fit_line:
            handles.append(
                Line2D(
                    [0, 1],
                    [0, 0],
                    linestyle="-",
                    color=cat_color[c],
                    linewidth=2.5,
                    label=label,
                )
            )
        else:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker=cat_marker[c],
                    linestyle="None",
                    markersize=10,
                    markerfacecolor=cat_color[c],
                    markeredgecolor="black",
                    label=label,
                )
            )

    ax.legend(
        handles=handles,
        title=category_legend_title,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        framealpha=0.9,
    )

    range_bits: list[str] = []
    if distance_min is not None:
        range_bits.append(f"≥{distance_min:g}")
    if distance_max is not None:
        range_bits.append(f"≤{distance_max:g}")
    range_suffix = f" [{', '.join(range_bits)}]" if range_bits else ""

    ax.set_xlabel(
        xlabel
        if xlabel is not None
        else (
            f"Range (m){range_suffix}"
        )
    )
    ax.set_ylabel(ylabel if ylabel is not None else f"Mean {column} (per bin)")
    if use_fit_line:
        mode = f"deg-{fit_poly_degree} poly fit (capped by n)"
    else:
        mode = "bin means"
    default_title = f"{column} vs Range ({mode}){range_suffix}"
    ax.set_title(title if title is not None else default_title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout(rect=(0.0, 0.0, 0.78, 1.0))
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_column_availability_by_category(
    df: pd.DataFrame,
    column: str,
    *,
    category_column: str,
    availability_bound: float,
    category_legend_title: str = "Category",
    fit_poly_degree: int = 1,
    plot_fit_line: bool = True,
    n_bins: int = 10,
    confidence: float = 0.95,
    distance_column: str | None = None,
    distance_min: float | None = None,
    distance_max: float | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    zoom_x_to_data: bool = True,
    x_pad_fraction: float = 0.02,
    availability_y_min: float | None = None,
    availability_y_max: float | None = None,
    plot_bin_points: bool = False,
    bin_error_confidence: float = 0.95,
    ax: Optional[plt.Axes] = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Availability vs range with one series per category (fit curve and/or bin points).

    Same binning as :func:`plot_column_availability_by_camera`. **Color** encodes category.
    By default draws a clipped polynomial **fit**; set ``plot_fit_line=False`` for bin markers
    only (typically with ``plot_bin_points=True``).
    """
    if not plot_fit_line and not plot_bin_points:
        raise ValueError(
            "plot_column_availability_by_category requires plot_fit_line and/or plot_bin_points."
        )
    y_lo, y_hi = _resolve_availability_ylim(availability_y_min, availability_y_max)

    by_cat = _filtered_category_subsets(
        df,
        category_column=category_column,
        distance_column=distance_column,
        distance_min=distance_min,
        distance_max=distance_max,
    )
    if not by_cat:
        raise ValueError(
            f"No rows after distance filter or no categories in {category_column!r}"
        )

    unique_cats = sorted(by_cat.keys(), key=lambda k: (pd.isna(k), str(k)))
    cmap = plt.get_cmap("tab10")
    cat_color: dict[Any, tuple] = {
        c: cmap(i % cmap.N) for i, c in enumerate(unique_cats)
    }

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    x_min_data = np.inf
    x_max_data = -np.inf

    for cat_key in unique_cats:
        sub = by_cat[cat_key]
        color = cat_color[cat_key]

        clumps = column_summary(
            sub,
            column,
            confidence=confidence,
            dropna=True,
            n_bins=n_bins,
            distance_column=distance_column,
            print_results=False,
            availability_below=float(availability_bound),
        )
        if not clumps:
            continue
        x_mid = np.array([(r["distance_lo"] + r["distance_hi"]) / 2.0 for r in clumps])
        avail = np.array([float(r["pct_below"]) for r in clumps])
        order = np.argsort(x_mid)
        x_mid = x_mid[order]
        avail = avail[order]
        n_arr = np.array([int(r["n"]) for r in clumps], dtype=np.int64)[order]
        k_arr = np.array([int(r["n_below"]) for r in clumps], dtype=np.int64)[order]

        if x_mid.size >= 1:
            if plot_fit_line:
                x_line, y_line = _availability_fit_line(
                    x_mid, avail, fit_poly_degree=fit_poly_degree
                )
                x_min_data = min(x_min_data, float(np.min(x_line)))
                x_max_data = max(x_max_data, float(np.max(x_line)))
                ax.plot(
                    x_line,
                    y_line,
                    linestyle="-",
                    color=color,
                    linewidth=2.0,
                    alpha=0.9,
                    zorder=3,
                )
            x_min_data = min(x_min_data, float(np.min(x_mid)))
            x_max_data = max(x_max_data, float(np.max(x_mid)))

        if plot_bin_points and x_mid.size >= 1:
            ci_lo = np.empty_like(avail, dtype=np.float64)
            ci_hi = np.empty_like(avail, dtype=np.float64)
            for i in range(x_mid.size):
                ci_lo[i], ci_hi[i] = _proportion_wilson_ci_pct(
                    int(k_arr[i]), int(n_arr[i]), bin_error_confidence
                )
            yerr_lo = np.maximum(0.0, avail - ci_lo)
            yerr_hi = np.maximum(0.0, ci_hi - avail)
            ax.errorbar(
                x_mid,
                avail,
                yerr=[yerr_lo, yerr_hi],
                fmt="o",
                color=color,
                ecolor=color,
                elinewidth=1.2,
                capsize=3,
                capthick=1.0,
                markersize=5,
                markeredgecolor="black",
                markeredgewidth=0.5,
                linestyle="none",
                zorder=5,
                alpha=0.9,
            )

    if zoom_x_to_data and np.isfinite(x_min_data) and np.isfinite(x_max_data):
        span = x_max_data - x_min_data
        pad = float(x_pad_fraction) * (
            span if span > 0 else max(abs(x_max_data), 1.0) * 0.01
        )
        ax.set_xlim(x_min_data - pad, x_max_data + pad)

    handles: list[Line2D] = []
    for c in unique_cats:
        color = cat_color[c]
        lab = "" if pd.isna(c) else str(c)
        if plot_fit_line:
            handles.append(
                Line2D(
                    [0, 1],
                    [0, 0],
                    linestyle="-",
                    color=color,
                    linewidth=2.5,
                    label=lab,
                )
            )
        elif plot_bin_points:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="None",
                    color=color,
                    markersize=6,
                    markeredgecolor="black",
                    markeredgewidth=0.5,
                    label=lab,
                )
            )
    if plot_bin_points:
        pct_e = int(round(float(bin_error_confidence) * 100))
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markersize=6,
                markerfacecolor="0.75",
                markeredgecolor="black",
                label=f"Bin % ± {pct_e}% Wilson CI",
            )
        )

    ax.legend(
        handles=handles,
        title=category_legend_title,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        framealpha=0.9,
    )

    range_bits: list[str] = []
    if distance_min is not None:
        range_bits.append(f"≥{distance_min:g}")
    if distance_max is not None:
        range_bits.append(f"≤{distance_max:g}")
    range_suffix = f" [{', '.join(range_bits)}]" if range_bits else ""

    ax.set_xlabel(
        xlabel
        if xlabel is not None
        else (
            f"Range (m){range_suffix}"
        )
    )
    default_yl = (
        f"Availability (%)"
    )
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    else:
        crop_note = (
            f" (axis {y_lo:g}–{y_hi:g}%)"
            if (y_lo > 0.0 or y_hi < 100.0)
            else ""
        )
        ax.set_ylabel(default_yl + crop_note)
    if title is None:
        if plot_fit_line:
            mode = f"{column} < {availability_bound:g}"
        else:
            mode = f"{column} < {availability_bound:g}"
        default_title = f"Availability vs Range ({mode}){range_suffix}"
    else:
        default_title = title
    ax.set_title(default_title)
    ax.set_ylim(y_lo, y_hi)
    ax.yaxis.set_major_locator(MaxNLocator(nbins="auto", steps=[1, 2, 2.5, 5, 10]))
    ax.grid(True, alpha=0.3)
    fig.tight_layout(rect=(0.0, 0.0, 0.78, 1.0))
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_column_summary_by_camera(
    df: pd.DataFrame,
    column: str,
    *,
    use_fit_line: bool = False,
    fit_poly_degree: int = 1,
    n_bins: int = 10,
    confidence: float = 0.95,
    distance_column: str | None = None,
    distance_min: float | None = None,
    distance_max: float | None = None,
    fovs: Sequence[float] | None = None,
    include_fovs: Sequence[float] | None = None,
    fov_match_atol: float = 1e-3,
    resolutions: Sequence[int | tuple[int, int]] | None = None,
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

    For availability-only plots (0–100% with 100% = all points below a bound), use
    :func:`plot_column_availability_by_camera`.

    Parameters
    ----------
    df : pd.DataFrame
        Simulation DataFrame with camera columns, the value *column*, and distance
        (via ``distance_column`` or ``true_pos_*``); see :func:`column_summary`.
    column : str
        Numeric column whose per-bin means are plotted on the y-axis.
    use_fit_line : bool, optional
        If False (default), plot one scatter series per camera (color × marker).
        If True, plot a polynomial least-squares fit (see ``fit_poly_degree``) through
        each camera's bin means.
    fit_poly_degree : int, optional
        Polynomial degree for ``use_fit_line`` (default 1). Capped per series to
        ``n_points - 1`` when there are fewer bin means than the requested degree.
    n_bins, confidence, distance_column
        Passed to :func:`column_summary`.
    distance_min, distance_max : float or None, optional
        If set, only rows with distance in this inclusive range are used (same distance
        definition as ``column_summary``). Narrows data before binning and acts as an
        effective x-axis zoom.
    fovs : sequence of float or None, optional
        **Include-only** list: if set, only cameras whose horizontal FOV (degrees)
        matches one of these values (within ``fov_match_atol``) are plotted; all
        other FOVs are excluded.
    include_fovs : sequence of float or None, optional
        Same as ``fovs`` (clearer name). Pass only one of ``fovs`` and
        ``include_fovs``.
    fov_match_atol : float, optional
        Absolute tolerance in degrees for matching ``fovs`` (default 1e-3).
    resolutions : sequence or None, optional
        If set, only these resolutions are included. Each entry is ``(width, height)``
        in pixels, or a single ``int`` ``n`` meaning ``(n, n)``.
    title, xlabel, ylabel : str or None
        Primary axis title/labels; if None, defaults are derived from *column*,
        distance, and mode.
    ax : matplotlib.axes.Axes or None
        Axes to draw on; if None, a new figure is created.
    save_path : path-like or None
        If set, save the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
        The figure used for plotting.
    """
    fovs_f = _resolve_include_fovs(fovs, include_fovs)
    by_cam, fov_by_cam, res_by_cam = _filtered_camera_subsets(
        df,
        distance_column=distance_column,
        distance_min=distance_min,
        distance_max=distance_max,
        fovs=fovs_f,
        fov_match_atol=fov_match_atol,
        resolutions=resolutions,
    )

    # Maps keyed only by cameras we plot
    unique_fovs = sorted({round(fov_by_cam[cid], 6) for cid in by_cam})
    fov_to_display: dict[float, str] = {f: f"{f:g}°" for f in unique_fovs}
    unique_res = sorted({res_by_cam[cid] for cid in by_cam})

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
            if x_mid.size >= 1:
                deg_eff = _effective_poly_degree(int(x_mid.size), fit_poly_degree)
                coeffs = np.polyfit(x_mid, y_mean, deg_eff)
                x_lo, x_hi = float(x_mid.min()), float(x_mid.max())
                x_line = np.linspace(x_lo, x_hi, 200) if x_hi > x_lo else np.full(200, x_lo)
                y_line = np.polyval(coeffs, x_line)
                ax.plot(
                    x_line,
                    y_line,
                    linestyle=ls,
                    color=color,
                    linewidth=2.0,
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

    range_bits = []
    if distance_min is not None:
        range_bits.append(f"≥{distance_min:g}")
    if distance_max is not None:
        range_bits.append(f"≤{distance_max:g}")
    range_suffix = f" [{', '.join(range_bits)}]" if range_bits else ""

    ax.set_xlabel(
        xlabel
        if xlabel is not None
        else (
            f"Range (m){range_suffix}"
        )
    )
    ax.set_ylabel(ylabel if ylabel is not None else f"Mean {column} (per bin)")
    if use_fit_line:
        mode = f"deg-{fit_poly_degree} poly fit (capped by n)"
    else:
        mode = "bin means"
    default_title = f"{column} vs Range ({mode}){range_suffix}"
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

    # Prediction intervals: either raw per-bin bounds or smooth fit (softplus / polynomial)
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

        def _fit_softplus(
            x: np.ndarray, y: np.ndarray, sig: np.ndarray
        ) -> tuple[bool, np.ndarray]:
            """Try softplus; on failure use degree-2 polynomial. Returns (ok_softplus, params)."""
            w_poly = 1.0 / (sig.astype(np.float64) ** 2)
            if len(x) < 4:
                coeff = np.polyfit(x, y, min(2, len(x)), w=w_poly)
                return False, coeff
            L0 = float(np.min(y))
            if x_span > 0:
                m0 = float((y[-1] - y[0]) / x_span) if len(y) > 1 else 0.0
                b0 = float(y[0] - m0 * x[0])
            else:
                m0, b0 = 0.0, float(y[0])
            beta0 = 2.0
            try:
                (beta, m, b, L), _ = curve_fit(
                    _softplus,
                    x,
                    y,
                    p0=[beta0, m0, b0, L0],
                    sigma=sig,
                    bounds=([1e-6, -np.inf, -np.inf, -np.inf], [100.0, np.inf, np.inf, np.inf]),
                )
                return True, np.array([float(beta), float(m), float(b), float(L)])
            except (ValueError, RuntimeError):
                coeff = np.polyfit(x, y, 2, w=w_poly)
                return False, coeff

        def _eval_fit(ok_softplus: bool, params: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
            if ok_softplus:
                return _softplus(x_eval, *params)
            return np.polyval(params, x_eval)

        ok_center, params_center = _fit_softplus(x_sorted, center, sigma)
        ok_hw, params_hw = _fit_softplus(x_sorted, half_width, sigma)
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

