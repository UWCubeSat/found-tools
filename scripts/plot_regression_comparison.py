"""Compare regression algorithms on simulation CSV: availability vs range (pixel errors).

Uses :func:`~limb.simulation.analysis.metrics.fill_pixel_metrics` so centroids/radius
and signed deltas are consistent pixel-space quantities from camera pose; availability
plots use **absolute** pixel error ``|out − true|`` per axis and for apparent radius.
Per-metric availability curves are supplemented by one **combined** plot: fraction of
rows where **all three** |Δx|, |Δy|, and |Δr| are strictly below their bounds on the
same row.

Optional ``--with-mean`` adds per-bin mean of ``out_*`` vs range (same helpers as
multi-camera column summary). By default availability uses a **quadratic** (max degree 2)
fit; ``--availability-bins-only`` plots Wilson bin points only. Use
``--regression-methods ransac tls`` to restrict rows by ``--category-column``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from limb.simulation.analysis.metrics import fill_pixel_metrics
from limb.simulation.analysis.plot import (
    plot_column_availability_by_category,
    plot_column_summary_by_category,
)

_COMBINED_PIXEL_FAIL_COL = "_regression_combined_pixel_fail"
# Synthetic column is 0.0 when |Δx|,|Δy|,|Δr| are all strictly below their bounds, else 1.0
_COMBINED_AVAIL_SYNTHETIC_BOUND = 0.5


def _distance_range_suffix(
    distance_min: float | None, distance_max: float | None
) -> str:
    bits: list[str] = []
    if distance_min is not None:
        bits.append(f"≥{distance_min:g}")
    if distance_max is not None:
        bits.append(f"≤{distance_max:g}")
    return f" [{', '.join(bits)}]" if bits else ""


def _add_combined_pixel_fail_column(
    df: pd.DataFrame, bx: float, by: float, br: float
) -> None:
    vx = df["abs_delta_x_centroid"].to_numpy(dtype=np.float64, copy=False)
    vy = df["abs_delta_y_centroid"].to_numpy(dtype=np.float64, copy=False)
    vr = df["abs_delta_r_apparent"].to_numpy(dtype=np.float64, copy=False)
    ok = np.isfinite(vx) & np.isfinite(vy) & np.isfinite(vr)
    pass_all = ok & (vx < bx) & (vy < by) & (vr < br)
    fail = np.ones(len(df), dtype=np.float64)
    fail[pass_all] = 0.0
    df[_COMBINED_PIXEL_FAIL_COL] = fail


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Path to CSV (default: regression.csv in repo root)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for PNG outputs (default: scripts/test_plot_outputs)",
    )
    p.add_argument(
        "--category-column",
        default="distance_regression",
        help="Column whose distinct values define one curve color each (default: distance_regression)",
    )
    p.add_argument(
        "--legend-title",
        default="Regression",
        help="Legend title for algorithm names",
    )
    p.add_argument(
        "--distance-column",
        default=None,
        help="Distance / range column in meters; default ‖true position‖ from true_pos_*",
    )
    p.add_argument("--distance-min", type=float, default=None)
    p.add_argument("--distance-max", type=float, default=None)
    p.add_argument("--n-bins", type=int, default=10)
    p.add_argument("--fit-poly-degree", type=int, default=2)
    p.add_argument(
        "--with-mean",
        action="store_true",
        help="Also plot per-bin mean of out_x/y_centroid and out_r_apparent vs range",
    )
    p.add_argument(
        "--no-fit-line",
        action="store_true",
        help="With --with-mean: plot bin means as scatter instead of polynomial fit lines",
    )
    p.add_argument(
        "--skip-availability",
        action="store_true",
        help="Do not write availability plots",
    )
    p.add_argument(
        "--avail-delta-x-bound",
        type=float,
        default=.01,
        help="Availability: fraction with |Δx| < this (strict); pixels after fill_pixel_metrics",
    )
    p.add_argument(
        "--avail-delta-y-bound",
        type=float,
        default=.01,
        help="Availability: fraction with |Δy| < this (strict); pixels",
    )
    p.add_argument(
        "--avail-delta-r-bound",
        type=float,
        default=.01,
        help="Availability: fraction with |Δr| < this (strict); pixels",
    )
    p.add_argument(
        "--availability-y-min",
        type=float,
        default=None,
        help="Crop availability plot y-axis bottom (percent)",
    )
    p.add_argument(
        "--availability-y-max",
        type=float,
        default=None,
        help="Crop availability plot y-axis top (percent)",
    )
    p.add_argument(
        "--plot-bin-points",
        action="store_true",
        help="On availability plots: bin midpoints with Wilson CI error bars",
    )
    p.add_argument(
        "--availability-bins-only",
        action="store_true",
        help=(
            "Availability plots: draw bin points only (implies --plot-bin-points); "
            "omit polynomial fit curves"
        ),
    )
    p.add_argument(
        "--regression-methods",
        nargs="*",
        default=None,
        metavar="NAME",
        help=(
            "Only rows whose --category-column equals one of these names "
            "(e.g. ransac tls). Default: all categories in the CSV"
        ),
    )
    p.add_argument(
        "--bin-error-confidence",
        type=float,
        default=0.95,
        help="Wilson interval confidence when --plot-bin-points (default: 0.95)",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively after saving",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    csv_path = (
        args.csv.resolve()
        if args.csv is not None
        else (repo_root / "regression.csv").resolve()
    )
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    out_dir = (
        args.out_dir.resolve()
        if args.out_dir is not None
        else (repo_root / "scripts" / "test_plot_outputs").resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Rows: {len(df)}")
    df = fill_pixel_metrics(df)
    # Availability uses symmetric pixel error vs CSV-signed delta columns.
    df["abs_delta_x_centroid"] = df["delta_x_centroid"].abs()
    df["abs_delta_y_centroid"] = df["delta_y_centroid"].abs()
    df["abs_delta_r_apparent"] = df["delta_r_apparent"].abs()

    if args.category_column not in df.columns:
        raise ValueError(
            f"Missing category column {args.category_column!r}; "
            f"have: {list(df.columns)[:20]}..."
        )

    if args.regression_methods:
        methods = list(args.regression_methods)
        n_before = len(df)
        df = df.loc[df[args.category_column].isin(methods)].reset_index(drop=True)
        print(f"Filtered to {args.category_column} in {methods!r}: {len(df)} / {n_before} rows")
        if df.empty:
            raise ValueError("No rows left after --regression-methods filter")

    plot_bin_points = bool(args.plot_bin_points or args.availability_bins_only)
    plot_fit_line = not args.availability_bins_only

    avail_kw = dict(
        category_column=args.category_column,
        category_legend_title=args.legend_title,
        distance_column=args.distance_column,
        distance_min=args.distance_min,
        distance_max=args.distance_max,
        n_bins=args.n_bins,
        fit_poly_degree=args.fit_poly_degree,
        plot_fit_line=plot_fit_line,
        availability_y_min=args.availability_y_min,
        availability_y_max=args.availability_y_max,
        plot_bin_points=plot_bin_points,
        bin_error_confidence=args.bin_error_confidence,
    )

    if args.with_mean:
        common_plot_kw = dict(
            category_column=args.category_column,
            category_legend_title=args.legend_title,
            distance_column=args.distance_column,
            distance_min=args.distance_min,
            distance_max=args.distance_max,
            n_bins=args.n_bins,
            fit_poly_degree=args.fit_poly_degree,
            use_fit_line=not args.no_fit_line,
        )
        mean_columns = ("out_x_centroid", "out_y_centroid", "out_r_apparent")
        for col in mean_columns:
            if col not in df.columns:
                raise ValueError(f"CSV missing column {col!r} for mean vs range plot")
            safe = col.replace(" ", "_")
            path = out_dir / f"regression_{safe}_vs_range_by_{args.category_column}.png"
            print(f"Writing {path.name}...")
            plot_column_summary_by_category(
                df,
                col,
                save_path=path,
                **common_plot_kw,
            )
            plt.close("all")

    if not args.skip_availability:
        avail_specs = (
            ("abs_delta_x_centroid", args.avail_delta_x_bound),
            ("abs_delta_y_centroid", args.avail_delta_y_bound),
            ("abs_delta_r_apparent", args.avail_delta_r_bound),
        )
        for col, bound in avail_specs:
            if col not in df.columns:
                print(f"Skip availability: missing {col!r}")
                continue
            safe = col.replace(" ", "_")
            path = (
                out_dir
                / f"regression_availability_{safe}_lt_{bound:g}_by_{args.category_column}.png"
            )
            print(f"Writing {path.name}...")
            plot_column_availability_by_category(
                df,
                col,
                availability_bound=float(bound),
                save_path=path,
                **avail_kw,
            )
            plt.close("all")

        if all(c in df.columns for c, _ in avail_specs):
            bx, by, br = (
                float(args.avail_delta_x_bound),
                float(args.avail_delta_y_bound),
                float(args.avail_delta_r_bound),
            )
            _add_combined_pixel_fail_column(df, bx, by, br)
            rsfx = _distance_range_suffix(args.distance_min, args.distance_max)
            combined_path = (
                out_dir
                / f"regression_availability_all_three_lt_{bx:g}_{by:g}_{br:g}_by_{args.category_column}.png"
            )
            # if plot_fit_line:
            #     mode = f"quadratic fit, all |Δx|<{bx:g}, |Δy|<{by:g}, |Δr|<{br:g} px"
            # else:
            #     mode = (
            #         f"bin % ± Wilson CI, all |Δx|<{bx:g}, |Δy|<{by:g}, |Δr|<{br:g} px"
            #     )
            mode = f"all |Δx|<{bx:g}, |Δy|<{by:g}, |Δr|<{br:g} px"
            combined_title = (
                f"Availability vs Range ({mode}){rsfx}"
            )
            combined_ylabel = (
                "Availability (%)"
            )
            print(f"Writing {combined_path.name}...")
            plot_column_availability_by_category(
                df,
                _COMBINED_PIXEL_FAIL_COL,
                availability_bound=_COMBINED_AVAIL_SYNTHETIC_BOUND,
                title=combined_title,
                ylabel=combined_ylabel,
                save_path=combined_path,
                **avail_kw,
            )
            plt.close("all")
        else:
            print("Skip combined availability: missing one or more abs_delta_* columns")

    print("Done.")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
