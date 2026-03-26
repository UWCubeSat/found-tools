"""Test column_summary and plot helpers on a multi-camera simulation CSV.

Use --fovs DEG [DEG ...] to include only those horizontal FOVs (degrees); other
cameras are removed via limb.simulation.analysis.metrics.filter_df_by_fovs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from limb.simulation.analysis.metrics import (
    column_summary,
    fill_pixel_metrics,
    filter_df_by_fovs,
)
from limb.simulation.analysis.plot import (
    plot_column_availability_by_camera,
    plot_column_summary,
    plot_column_summary_by_camera,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Path to CSV (default: huge-multi-cam-angle-noise_with_delta_distance.csv in repo root)",
    )
    p.add_argument(
        "--column",
        default="delta_x_centroid",
        help="Column for summaries and multi-camera plots",
    )
    p.add_argument(
        "--fovs",
        type=float,
        nargs="*",
        default=None,
        metavar="DEG",
        help="Include only these horizontal FOVs (degrees); all other cameras excluded",
    )
    p.add_argument(
        "--fov-match-atol",
        type=float,
        default=1e-3,
        help="Absolute tolerance in degrees when matching --fovs (default: 1e-3)",
    )
    p.add_argument(
        "--availability-y-min",
        type=float,
        default=None,
        help="Crop availability plot y-axis bottom (percent); default 0",
    )
    p.add_argument(
        "--availability-y-max",
        type=float,
        default=None,
        help="Crop availability plot y-axis top (percent); default 100",
    )
    p.add_argument(
        "--plot-bin-points",
        action="store_true",
        help="On availability plot: draw bin midpoints with Wilson CI error bars",
    )
    p.add_argument(
        "--bin-error-confidence",
        type=float,
        default=0.95,
        help="Confidence level for Wilson intervals (default: 0.95)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    csv_path = args.csv if args.csv is not None else repo_root / "huge-multi-cam-angle-noise_with_delta_distance.csv"
    csv_path = csv_path.resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    print("Loading CSV...")
    df = pd.read_csv(csv_path)
    print(f"Rows: {len(df)}")

    print("Filling pixel metrics...")
    df = fill_pixel_metrics(df)
    print("Done.")

    fovs_arg = args.fovs if args.fovs else None
    if fovs_arg:
        df = filter_df_by_fovs(df, fovs_arg, fov_match_atol=args.fov_match_atol)
        print(f"Filtered to FOVs (deg) {list(fovs_arg)} — rows: {len(df)}")

    column = args.column
    print(f"\n--- column_summary for {column} (printed table) ---\n")
    column_summary(df, column, n_bins=10, print_results=True)

    out_dir = repo_root / "scripts" / "test_plot_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    fov_tag = "_fov-" + "-".join(str(int(f)) if f == int(f) else str(f) for f in fovs_arg) if fovs_arg else ""
    save_path = out_dir / f"column_summary_true_r_apparent{fov_tag}.png"

    print(f"\nPlotting {column} vs distance (saving to {save_path})...")
    plot_column_summary(
        df,
        column,
        n_points=800,
        square_size=3.0,
        title="True apparent radius vs range",
        xlabel="Range (m)",
        ylabel="True apparent radius (px)",
        n_bins=10,
        confidence=0.99,
        save_path=save_path,
        raw_intervals=False,
    )

    by_cam_path = out_dir / f"column_summary_by_camera_points{fov_tag}.png"
    print(f"\nMulti-camera bin means (points) → {by_cam_path}")
    fig2 = plot_column_summary_by_camera(
        df,
        column,
        use_fit_line=False,
        n_bins=10,
        confidence=0.99,
        title=None,
        xlabel="Range (m)",
        ylabel=None,
        save_path=by_cam_path,
    )
    plt.close(fig2)

    by_cam_fit_path = out_dir / f"column_summary_by_camera_fit{fov_tag}.png"
    print(f"Multi-camera linear fit → {by_cam_fit_path}")
    fig3 = plot_column_summary_by_camera(
        df,
        column,
        use_fit_line=True,
        fit_poly_degree=1,
        n_bins=10,
        confidence=0.99,
        title="Per-camera linear fit to bin means",
        xlabel="Range (m)",
        ylabel=None,
        save_path=by_cam_fit_path,
    )
    plt.close(fig3)

    zoom_poly_path = out_dir / f"column_summary_by_camera_zoom_poly2{fov_tag}.png"
    print(f"Zoom + deg-2 fit → {zoom_poly_path}")
    fig4 = plot_column_summary_by_camera(
        df,
        column,
        use_fit_line=True,
        fit_poly_degree=2,
        distance_min=1.0e7,
        distance_max=2.5e7,
        n_bins=8,
        confidence=0.99,
        xlabel="Range (m)",
        save_path=zoom_poly_path,
    )
    plt.close(fig4)

    avail_path = out_dir / f"column_availability_by_camera_{column}{fov_tag}.png"
    print(f"Availability only (100% = all points < bound) → {avail_path}")
    fig5 = plot_column_availability_by_camera(
        df,
        column,
        availability_bound=100,
        fit_poly_degree=1,
        distance_min=1.0e7,
        distance_max=20.0e7,
        n_bins=8,
        confidence=0.99,
        xlabel="Range (m)",
        ylabel="Availability (%)",
        title="Availability vs Range (Distance Error < 100m)",
        save_path=avail_path,
        include_fovs=[5, 20, 50, 85],
        availability_y_min=40,
        availability_y_max=90,
        plot_bin_points=args.plot_bin_points,
        bin_error_confidence=args.bin_error_confidence,
    )
    plt.close(fig5)
    print("Done.")


if __name__ == "__main__":
    main()
