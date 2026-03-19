"""Test column_summary and plot_column_summary on distance_multi-camera-leo-geo.csv.

Loads the CSV, fills pixel metrics, then prints per-distance stats and produces
a plot of a chosen column vs distance with prediction intervals.
"""

from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from limb.simulation.analysis.metrics import column_summary, fill_pixel_metrics
from limb.simulation.analysis.plot import (
    plot_column_summary,
    plot_column_summary_by_camera,
)


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    csv_path = repo_root / "distance_multi-camera-leo-geo.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    print("Loading CSV...")
    df = pd.read_csv(csv_path)
    print(f"Rows: {len(df)}")

    print("Filling pixel metrics...")
    df = fill_pixel_metrics(df)
    print("Done.")

    # Column to summarize and plot (e.g. apparent radius or residual)
    column = "delta_r_apparent"
    print(f"\n--- column_summary for {column} (printed table) ---\n")
    column_summary(df, column, n_bins=10, print_results=True)

    out_dir = repo_root / "scripts" / "test_plot_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / "column_summary_true_r_apparent.png"

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

    by_cam_path = out_dir / "column_summary_by_camera_points.png"
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

    by_cam_fit_path = out_dir / "column_summary_by_camera_fit.png"
    print(f"Multi-camera linear fit → {by_cam_fit_path}")
    fig3 = plot_column_summary_by_camera(
        df,
        column,
        use_fit_line=True,
        n_bins=10,
        confidence=0.99,
        title="Per-camera linear fit to bin means",
        xlabel="Range (m)",
        ylabel=None,
        save_path=by_cam_fit_path,
    )
    plt.close(fig3)

    print("Done.")


if __name__ == "__main__":
    main()
