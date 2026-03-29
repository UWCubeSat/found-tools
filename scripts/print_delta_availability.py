#!/usr/bin/env python3
"""Print availability tables and save availability plots for delta columns.

Uses the same binning and filters as
:func:`~limb.simulation.analysis.plot.plot_column_availability_by_camera`:
strict ``column < bound``, distance quantile bins, distance / FOV filters.

For each column, prints per-camera tables (via :func:`column_summary`) and writes
``column_availability_by_camera_<column>.png`` under ``--output-dir``.

Default columns: ``delta_centroid`` (``max(|Δx|,|Δy|)`` px — both axes below
``--availability-bound``), ``delta_r_apparent``, ``delta_distance``. Run
``scripts/add_delta_distance_column.py`` or ``fill_pixel_metrics`` to populate
``delta_centroid`` when missing from CSV.

Examples::

    uv run python scripts/print_delta_availability.py --csv data/multi-cam-26.csv
    uv run python scripts/print_delta_availability.py --no-print --output-dir plots/out
    uv run python scripts/print_delta_availability.py --availability-bound 50 --distance-max 2.5e7
    uv run python scripts/print_delta_availability.py --availability-bound-delta-distance 50000
    uv run python scripts/print_delta_availability.py --availability-y-min 90 --availability-y-min-delta-distance 0
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from limb.simulation.analysis.metrics import (
    _print_column_summary_table,
    column_summary,
    fill_pixel_metrics,
    filter_df_by_fovs,
)
from limb.simulation.analysis.plot import (
    _filtered_camera_subsets,
    _resolve_availability_ylim,
    _resolve_include_fovs,
    plot_column_availability_by_camera,
)

DEFAULT_COLUMNS = (
    "delta_centroid",
    "delta_r_apparent",
    "delta_distance",
)
DEFAULT_INCLUDE_FOVS = (5.0, 20.0, 50.0, 85.0)


def _ensure_delta_centroid_column(df: pd.DataFrame) -> None:
    """Define ``delta_centroid`` from ``delta_x_centroid`` / ``delta_y_centroid`` if absent."""
    if "delta_centroid" in df.columns:
        return
    need = ("delta_x_centroid", "delta_y_centroid")
    if not all(c in df.columns for c in need):
        return
    dx = df["delta_x_centroid"].to_numpy(dtype=np.float64, copy=False)
    dy = df["delta_y_centroid"].to_numpy(dtype=np.float64, copy=False)
    ok = np.isfinite(dx) & np.isfinite(dy)
    mag = np.full(len(df), np.nan, dtype=np.float64)
    mag[ok] = np.maximum(np.abs(dx[ok]), np.abs(dy[ok]))
    df["delta_centroid"] = mag
    print("Computed delta_centroid = max(|Δx|,|Δy|) from delta_x/y_centroid")


def _availability_bound_for_column(column: str, args: argparse.Namespace) -> float:
    """Pixel deltas use --availability-bound; ``delta_distance`` (m) uses --availability-bound-delta-distance."""
    if column == "delta_distance":
        return float(args.availability_bound_delta_distance)
    return float(args.availability_bound)


def _availability_ylim_for_column(
    column: str, args: argparse.Namespace
) -> tuple[float | None, float | None]:
    """Plot y-limits (percent). ``delta_distance`` uses *-delta-distance* flags, else global *y-min/y-max*."""
    if column == "delta_distance":
        y_min = (
            args.availability_y_min_delta_distance
            if args.availability_y_min_delta_distance is not None
            else args.availability_y_min
        )
        y_max = (
            args.availability_y_max_delta_distance
            if args.availability_y_max_delta_distance is not None
            else args.availability_y_max
        )
        return y_min, y_max
    return args.availability_y_min, args.availability_y_max


def _format_bound_for_title(b: float) -> str:
    """Format bound for titles; drop a leading 0 before the decimal (e.g. ``0.05`` → ``.05``)."""
    s = f"{b:g}"
    if len(s) >= 2 and s[0] == "0" and s[1] == ".":
        return s[1:]
    return s


def _availability_bound_unit_word(column: str) -> str:
    return "meters" if column == "delta_distance" else "pixels"


def _availability_plot_title(column: str, b: float) -> str:
    bound_s = _format_bound_for_title(b)
    unit = _availability_bound_unit_word(column)
    if column == "delta_centroid":
        return f"Availability vs Range (both |Δx| & |Δy| < {bound_s} {unit})"
    return f"Availability vs Range ({column} < {bound_s} {unit})"


def _safe_filename_fragment(name: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", name).strip("_") or "column"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Path to simulation CSV (default: repo root huge-multi-cam-angle-noise… csv)",
    )
    p.add_argument(
        "--columns",
        nargs="*",
        default=list(DEFAULT_COLUMNS),
        help=f"Columns to summarize (default: {' '.join(DEFAULT_COLUMNS)})",
    )
    p.add_argument(
        "--skip-fill-pixel-metrics",
        action="store_true",
        help="Do not run fill_pixel_metrics (use if CSV already has centroid columns)",
    )
    p.add_argument(
        "--fovs",
        type=float,
        nargs="*",
        default=None,
        metavar="DEG",
        help="Filter entire dataframe to these horizontal FOVs (degrees) first",
    )
    p.add_argument(
        "--fov-match-atol",
        type=float,
        default=1e-3,
        help="Absolute tolerance in degrees for --fovs and include-FOV matching",
    )
    p.add_argument(
        "--include-fovs",
        type=float,
        nargs="*",
        default=list(DEFAULT_INCLUDE_FOVS),
        metavar="DEG",
        help="Per-camera plot filter: keep only these horizontal FOVs (default: 5 20 50 85)",
    )
    p.add_argument(
        "--availability-bound",
        type=float,
        default=1.0,
        metavar="BOUND",
        help=(
            "Threshold for delta_centroid, delta_r_apparent (pixels): rows with value < BOUND "
            "count as available; for delta_centroid, BOUND applies to both |Δx| and |Δy| via max(|Δx|,|Δy|)"
        ),
    )
    p.add_argument(
        "--availability-bound-delta-distance",
        type=float,
        default=1000.0,
        metavar="M",
        help=(
            "Threshold for ``delta_distance`` only (metres): rows with "
            "delta_distance < M count as available (default: 1e5)"
        ),
    )
    p.add_argument(
        "--distance-min",
        type=float,
        default=1.0e7,
        help="Min ‖true position‖ (m); same as test availability plot default",
    )
    p.add_argument(
        "--distance-max",
        type=float,
        default=20.0e7,
        help="Max ‖true position‖ (m); same as test availability plot default",
    )
    p.add_argument(
        "--distance-column",
        type=str,
        default=None,
        help="Optional distance column instead of ‖true_pos‖",
    )
    p.add_argument(
        "--n-bins",
        type=int,
        default=8,
        help="Number of distance quantile bins per camera (default: 8)",
    )
    p.add_argument(
        "--confidence",
        type=float,
        default=0.99,
        help="Prediction interval confidence printed in table header (default: 0.99)",
    )
    p.add_argument(
        "--no-print",
        action="store_true",
        help="Only write plots; skip console tables",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for PNGs (default: scripts/test_plot_outputs under repo root)",
    )
    p.add_argument(
        "--fit-poly-degree",
        type=int,
        default=1,
        help="Polynomial degree for availability fit curve (default: 1)",
    )
    p.add_argument(
        "--availability-y-min",
        type=float,
        default=None,
        metavar="PCT",
        help=(
            "Crop plot y-axis bottom (percent) for centroid / apparent-radius columns; "
            "also the fallback for ``delta_distance`` when --availability-y-min-delta-distance is omitted"
        ),
    )
    p.add_argument(
        "--availability-y-max",
        type=float,
        default=None,
        metavar="PCT",
        help=(
            "Crop plot y-axis top (percent) for centroid / apparent-radius columns; "
            "fallback for ``delta_distance`` when --availability-y-max-delta-distance is omitted"
        ),
    )
    p.add_argument(
        "--availability-y-min-delta-distance",
        type=float,
        default=None,
        metavar="PCT",
        help=(
            "Y-axis bottom (percent) for ``delta_distance`` plots only; "
            "overrides --availability-y-min for that column when set"
        ),
    )
    p.add_argument(
        "--availability-y-max-delta-distance",
        type=float,
        default=None,
        metavar="PCT",
        help=(
            "Y-axis top (percent) for ``delta_distance`` plots only; "
            "overrides --availability-y-max for that column when set"
        ),
    )
    p.add_argument(
        "--plot-bin-points",
        action="store_true",
        help="Overlay bin midpoints with Wilson CI error bars",
    )
    p.add_argument(
        "--bin-error-confidence",
        type=float,
        default=0.95,
        help="Wilson interval confidence when --plot-bin-points (default: 0.95)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    csv_path = args.csv
    if csv_path is None:
        for name in (
            "huge-multi-cam-angle-noise_with_delta_distance.csv",
            "huge-multi-cam-angle-noise.csv",
            "data/multi-cam-26.csv",
        ):
            cand = (repo_root / name).resolve()
            if cand.is_file():
                csv_path = cand
                break
        if csv_path is None:
            csv_path = repo_root / "data" / "multi-cam-26.csv"
    else:
        csv_path = csv_path.expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    print(f"Reading {csv_path} …")
    df = pd.read_csv(csv_path)
    print(f"Rows: {len(df)}")
    if "delta_distance" not in df.columns and "delta_range" in df.columns:
        df = df.rename(columns={"delta_range": "delta_distance"})

    if not args.skip_fill_pixel_metrics:
        print("fill_pixel_metrics …")
        df = fill_pixel_metrics(df)

    _ensure_delta_centroid_column(df)

    if args.fovs:
        df = filter_df_by_fovs(df, args.fovs, fov_match_atol=args.fov_match_atol)
        print(f"Filtered to FOVs {list(args.fovs)} — rows: {len(df)}")

    fovs_f = _resolve_include_fovs(None, args.include_fovs)
    by_cam, fov_by_cam, res_by_cam = _filtered_camera_subsets(
        df,
        distance_column=args.distance_column,
        distance_min=args.distance_min,
        distance_max=args.distance_max,
        fovs=fovs_f,
        fov_match_atol=args.fov_match_atol,
        resolutions=None,
    )

    n_bins = int(args.n_bins)
    conf = float(args.confidence)

    range_bits = []
    if args.distance_min is not None:
        range_bits.append(f"≥{args.distance_min:g} m")
    if args.distance_max is not None:
        range_bits.append(f"≤{args.distance_max:g} m")
    range_note = ", ".join(range_bits) if range_bits else "full range"
    fov_note = (
        f"include FOVs {list(args.include_fovs)}°"
        if args.include_fovs
        else "all FOVs"
    )

    cam_order = sorted(
        by_cam.keys(),
        key=lambda k: (round(fov_by_cam[k], 6), res_by_cam[k]),
    )

    out_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else repo_root / "scripts" / "test_plot_outputs"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    fov_tag = ""
    if args.fovs:
        fov_tag = "_fov-" + "-".join(
            str(int(f)) if f == int(f) else str(f) for f in args.fovs
        )

    include_list = list(args.include_fovs) if args.include_fovs else None

    for column in args.columns:
        if column not in df.columns:
            print(f"\nSkipping missing column {column!r}\n")
            continue

        b = _availability_bound_for_column(column, args)
        unit_w = _availability_bound_unit_word(column)

        if not args.no_print:
            print()
            print("=" * 72)
            print(f"Column: {column}")
            print(
                f"Availability: fraction of rows with {column!r} < {b:g} {unit_w} (strict); "
                f"n_bins={n_bins}; {range_note}; {fov_note}"
            )
            print(f"Prediction interval in table: {int(round(conf * 100))}%")
            print("=" * 72)

            for cam_id in cam_order:
                sub = by_cam[cam_id]
                if column not in sub.columns:
                    print(
                        f"\n--- skip camera {cam_id}: missing column {column!r} ---\n"
                    )
                    continue
                fov = fov_by_cam[cam_id]
                w, h = res_by_cam[cam_id]
                print(f"\n--- Camera: {fov:g}° horizontal FOV, {w}×{h} px ---")

                try:
                    clumps = column_summary(
                        sub,
                        column,
                        confidence=conf,
                        dropna=True,
                        n_bins=n_bins,
                        distance_column=args.distance_column,
                        print_results=False,
                        availability_below=b,
                    )
                except ValueError as exc:
                    print(f"(no table: {exc})")
                    continue
                if not clumps:
                    print("(no clumps)")
                    continue
                _print_column_summary_table(
                    column, clumps, conf, availability_below=b
                )

        col_slug = _safe_filename_fragment(column)
        bound_slug = _safe_filename_fragment(f"lt_{b:g}")
        y_min_p, y_max_p = _availability_ylim_for_column(column, args)
        y_slug = ""
        if y_min_p is not None or y_max_p is not None:
            y_lo, y_hi = _resolve_availability_ylim(y_min_p, y_max_p)
            y_slug = "_" + _safe_filename_fragment(f"y{y_lo:g}-{y_hi:g}")
        plot_path = (
            out_dir
            / f"column_availability_by_camera_{col_slug}_{bound_slug}{y_slug}{fov_tag}.png"
        )
        print(f"\nPlot availability → {plot_path}")
        fig = plot_column_availability_by_camera(
            df,
            column,
            availability_bound=b,
            fit_poly_degree=int(args.fit_poly_degree),
            distance_min=args.distance_min,
            distance_max=args.distance_max,
            distance_column=args.distance_column,
            n_bins=n_bins,
            confidence=conf,
            xlabel="Range (m)",
            ylabel="Availability (%)",
            title=_availability_plot_title(column, b),
            save_path=plot_path,
            include_fovs=include_list,
            fov_match_atol=args.fov_match_atol,
            availability_y_min=y_min_p,
            availability_y_max=y_max_p,
            plot_bin_points=args.plot_bin_points,
            bin_error_confidence=float(args.bin_error_confidence),
        )
        plt.close(fig)

    print("\nDone.")


if __name__ == "__main__":
    main()
