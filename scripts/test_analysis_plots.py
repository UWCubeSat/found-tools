"""Test analysis plots (excluding edge_plot): radius residuals and centroid residuals.

Run from repo root with:
  uv run python scripts/test_analysis_plots.py
  uv run python scripts/test_analysis_plots.py --keep

Uses synthetic data and non-interactive backend. With --keep, writes outputs
to scripts/test_plot_outputs/ for inspection.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Non-interactive backend so script can run without display
import matplotlib
matplotlib.use("Agg")

from limb.simulation.analysis.plot import (
    plot_radius_residuals_vs_range,
    plot_centroid_residuals_in_illumination_frame,
)


def _synthetic_radius_df(n: int = 30, n_cameras: int = 3) -> pd.DataFrame:
    """Build a minimal df with range, radius residuals, and multiple cameras."""
    np.random.seed(42)
    n_per_cam = max(1, (n + n_cameras - 1) // n_cameras)
    rows = []
    for c in range(n_cameras):
        f = 0.003 + c * 0.001
        res = 512 * (c + 1)
        ranges = np.linspace(6e6, 7.5e6, n_per_cam) + np.random.randn(n_per_cam) * 5e4
        for i in range(n_per_cam):
            r = ranges[i]
            th, ph = np.random.uniform(0, 2 * np.pi), np.random.uniform(0, np.pi)
            x = r * np.sin(ph) * np.cos(th)
            y = r * np.sin(ph) * np.sin(th)
            z = r * np.cos(ph)
            true_r = 50.0 + np.random.randn() * 2
            out_r = (true_r + np.random.randn() * 2.0)*0
            rows.append({
                "true_pos_x": x, "true_pos_y": y, "true_pos_z": z,
                "true_r_apparent": true_r, "out_r_apparent": out_r,
                "cam_focal_length": f, "cam_x_resolution": res, "cam_y_resolution": res,
            })
    return pd.DataFrame(rows)


def _synthetic_centroid_df(n: int = 30, n_cameras: int = 3) -> pd.DataFrame:
    """Build a minimal df with range, centroid residuals, and multiple cameras."""
    np.random.seed(43)
    n_per_cam = max(1, (n + n_cameras - 1) // n_cameras)
    rows = []
    for c in range(n_cameras):
        f = 0.003 + c * 0.001
        res = 512 * (c + 1)
        ranges = np.linspace(6e6, 7.5e6, n_per_cam) + np.random.randn(n_per_cam) * 5e4
        for i in range(n_per_cam):
            r = ranges[i]
            th, ph = np.random.uniform(0, 2 * np.pi), np.random.uniform(0, np.pi)
            x = r * np.sin(ph) * np.cos(th)
            y = r * np.sin(ph) * np.sin(th)
            z = r * np.cos(ph)
            rows.append({
                "true_pos_x": x, "true_pos_y": y, "true_pos_z": z,
                "true_x_centroid": 512 + np.random.randn(),
                "true_y_centroid": 512 + np.random.randn(),
                "out_x_centroid": 512 + np.random.randn() * 1.5,
                "out_y_centroid": 512 + np.random.randn() * 1.2,
                "cam_focal_length": f, "cam_x_resolution": res, "cam_y_resolution": res,
            })
    return pd.DataFrame(rows)


def test_radius_residuals_plot(out_dir: Path | None) -> None:
    """Call plot_radius_residuals_vs_range with synthetic df (multiple cameras)."""
    df = _synthetic_radius_df(n=30, n_cameras=3)
    save_path = out_dir / "radius_residuals_vs_range.png" if out_dir else None
    fig, ax = plot_radius_residuals_vs_range(
        df,
        radius_1sigma_px=2.0,
        target_label="Earth",
        save_path=save_path,
    )
    assert fig is not None and ax is not None
    if save_path:
        assert save_path.exists(), f"Expected {save_path} to be created"


def test_centroid_residuals_plot(out_dir: Path | None) -> None:
    """Call plot_centroid_residuals_in_illumination_frame with synthetic df (multiple cameras)."""
    df = _synthetic_centroid_df(n=30, n_cameras=3)
    save_path = out_dir / "centroid_residuals_illumination_frame.png" if out_dir else None
    fig, (ax1, ax2) = plot_centroid_residuals_in_illumination_frame(
        df,
        centroid_1sigma_px=1.5,
        target_label="Earth",
        save_path=save_path,
    )
    assert fig is not None and ax1 is not None and ax2 is not None
    if save_path:
        assert save_path.exists(), f"Expected {save_path} to be created"


def main() -> int:
    parser = argparse.ArgumentParser(description="Test analysis plots (no edge_plot).")
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Write plot images to scripts/test_plot_outputs/",
    )
    args = parser.parse_args()
    out_dir = None
    if args.keep:
        out_dir = Path(__file__).resolve().parent / "test_plot_outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Outputs will be written to {out_dir}")

    try:
        test_radius_residuals_plot(out_dir)
        print("OK plot_radius_residuals_vs_range")
        test_centroid_residuals_plot(out_dir)
        print("OK plot_centroid_residuals_in_illumination_frame")
    except Exception as e:
        print(f"FAIL: {e}", file=sys.stderr)
        raise
    print("All analysis plot tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
