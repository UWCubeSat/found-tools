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

# Non-interactive backend so script can run without display
import matplotlib
matplotlib.use("Agg")

from limb.simulation.analysis.plot import (
    plot_radius_residuals_vs_range,
    plot_centroid_residuals_in_illumination_frame,
)


def _synthetic_radius_data(n: int = 30, n_passes: int = 3):
    """Synthetic ranges and radius residuals for plot_radius_residuals_vs_range."""
    np.random.seed(42)
    ranges_m = np.linspace(6000000, 7500000, n) + np.random.randn(n) * 50000
    radius_residuals_px = np.random.randn(n) * 2.0
    pass_ids = [f"pass_{i % n_passes}" for i in range(n)]
    return ranges_m, radius_residuals_px, pass_ids


def _synthetic_centroid_data(n: int = 30, n_passes: int = 3):
    """Synthetic ranges and centroid residuals for plot_centroid_residuals_in_illumination_frame."""
    np.random.seed(43)
    ranges_m = np.linspace(6000000, 7500000, n) + np.random.randn(n) * 50000
    centroid_x_px = np.random.randn(n) * 1.5
    centroid_y_px = np.random.randn(n) * 1.2
    pass_ids = [f"pass_{i % n_passes}" for i in range(n)]
    return ranges_m, centroid_x_px, centroid_y_px, pass_ids


def test_radius_residuals_plot(out_dir: Path | None) -> None:
    """Call plot_radius_residuals_vs_range with synthetic data."""
    ranges_m, radius_residuals_px, pass_ids = _synthetic_radius_data()
    save_path = out_dir / "radius_residuals_vs_range.png" if out_dir else None
    fig, ax = plot_radius_residuals_vs_range(
        ranges_m,
        radius_residuals_px,
        pass_ids,
        radius_1sigma_px=2.0,
        target_label="Earth",
        save_path=save_path,
    )
    assert fig is not None and ax is not None
    if save_path:
        assert save_path.exists(), f"Expected {save_path} to be created"


def test_centroid_residuals_plot(out_dir: Path | None) -> None:
    """Call plot_centroid_residuals_in_illumination_frame with synthetic data."""
    ranges_m, centroid_x_px, centroid_y_px, pass_ids = _synthetic_centroid_data()
    save_path = out_dir / "centroid_residuals_illumination_frame.png" if out_dir else None
    fig, (ax1, ax2) = plot_centroid_residuals_in_illumination_frame(
        ranges_m,
        centroid_x_px,
        centroid_y_px,
        pass_ids,
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
