"""Test analysis plots (excluding edge_plot): radius residuals and centroid residuals.

Run from repo root with:
  uv run python scripts/test_analysis_plots.py
  uv run python scripts/test_analysis_plots.py --keep

By default also runs the high-n test (5000 points) to exercise density-scaled sizes,
hexbin underlay, and scatter subsampling. Use --high-n 0 to skip that test.
With --keep, writes all plot images to scripts/test_plot_outputs/.
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
            out_r = (true_r + np.random.randn() * 2.0)
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


def _synthetic_radius_df_y_f_of_x(n: int, n_cameras: int = 3, seed: int = 44) -> pd.DataFrame:
    """Synthetic df with many points; y (residual) is a function of x (range) + noise.

    Samples range from tightly clustered x values (so many points share similar x), and
    radius residual = f(range) + noise with f(range) = a*(range - 6.75e6)/1e6 so
    structure depends on x. Used to test density-scaled point sizes with 1e3–1e5 points.
    Tight clusters (small cluster_std) ensure clear size variation: dense = small, sparse = large.
    """
    np.random.seed(seed)
    # Tight clusters so density varies strongly (dense at cluster centers, sparse between)
    n_clusters = max(5, n // 500)
    cluster_centers = np.linspace(6.1e6, 7.4e6, n_clusters)
    cluster_std = 5e3  # tight: points clump at similar x so k-NN distance varies a lot
    ranges = np.concatenate([
        np.random.randn(n // n_clusters) * cluster_std + c for c in cluster_centers
    ])
    if len(ranges) < n:
        ranges = np.r_[ranges, np.random.choice(cluster_centers, n - len(ranges)) + np.random.randn(n - len(ranges)) * 1e4]
    ranges = ranges[:n]
    # y = f(x) + noise: residual depends on range
    f = (ranges - 6.75e6) / 1e6 * 2.0
    radius_residuals = f + np.random.randn(n) * 0.5
    true_r = 50.0 + np.random.randn(n) * 0.1
    out_r = true_r + radius_residuals
    rows = []
    for i in range(n):
        r = ranges[i]
        th, ph = np.random.uniform(0, 2 * np.pi), np.random.uniform(0, np.pi)
        x = r * np.sin(ph) * np.cos(th)
        y = r * np.sin(ph) * np.sin(th)
        z = r * np.cos(ph)
        cam = i % n_cameras
        f_foc = 0.003 + cam * 0.001
        res = 512 * (cam + 1)
        rows.append({
            "true_pos_x": x, "true_pos_y": y, "true_pos_z": z,
            "true_r_apparent": true_r[i], "out_r_apparent": out_r[i],
            "cam_focal_length": f_foc, "cam_x_resolution": res, "cam_y_resolution": res,
        })
    return pd.DataFrame(rows)


def _synthetic_centroid_df_y_f_of_x(n: int, n_cameras: int = 3, seed: int = 45) -> pd.DataFrame:
    """Synthetic df with many points; centroid residuals depend on range (x) + noise. Tight clusters for size test."""
    np.random.seed(seed)
    n_clusters = max(5, n // 500)
    cluster_centers = np.linspace(6.1e6, 7.4e6, n_clusters)
    cluster_std = 5e3
    ranges = np.concatenate([
        np.random.randn(n // n_clusters) * cluster_std + c for c in cluster_centers
    ])
    if len(ranges) < n:
        ranges = np.r_[ranges, np.random.choice(cluster_centers, n - len(ranges)) + np.random.randn(n - len(ranges)) * 1e4]
    ranges = ranges[:n]
    f = (ranges - 6.75e6) / 1e6 * 1.0
    cx = 512 + f * 10 + np.random.randn(n) * 0.8
    cy = 512 - f * 5 + np.random.randn(n) * 0.8
    rows = []
    for i in range(n):
        r = ranges[i]
        th, ph = np.random.uniform(0, 2 * np.pi), np.random.uniform(0, np.pi)
        x = r * np.sin(ph) * np.cos(th)
        y = r * np.sin(ph) * np.sin(th)
        z = r * np.cos(ph)
        cam = i % n_cameras
        f_foc = 0.003 + cam * 0.001
        res = 512 * (cam + 1)
        rows.append({
            "true_pos_x": x, "true_pos_y": y, "true_pos_z": z,
            "true_x_centroid": cx[i], "true_y_centroid": cy[i],
            "out_x_centroid": cx[i] + np.random.randn() * 0.5,
            "out_y_centroid": cy[i] + np.random.randn() * 0.5,
            "cam_focal_length": f_foc, "cam_x_resolution": res, "cam_y_resolution": res,
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


def test_density_scaled_high_n(out_dir: Path | None, n: int = 5000) -> None:
    """Test density-scaled point sizes with many points from y=f(x) synthetic data.

    Asserts that computed sizes actually vary (min < max, std > 0) so the capability is tested.
    """
    from limb.simulation.analysis.plot import _density_scaled_sizes

    df_radius = _synthetic_radius_df_y_f_of_x(n=n, n_cameras=3)
    df_centroid = _synthetic_centroid_df_y_f_of_x(n=n, n_cameras=3)
    ranges = np.linalg.norm(
        df_radius[["true_pos_x", "true_pos_y", "true_pos_z"]].values, axis=1
    )
    # Assert density scaling produces varying sizes (dense vs sparse)
    sizes = _density_scaled_sizes(ranges, None, base_size=40.0, min_size=1.0, max_size=100.0, k=min(50, n - 1))
    assert sizes.min() < sizes.max(), "Density-scaled sizes should vary (min < max)"
    assert np.std(sizes) > 1.0, "Density-scaled sizes should have meaningful spread (std > 1)"

    save_radius = out_dir / "radius_residuals_vs_range_high_n.png" if out_dir else None
    save_centroid = out_dir / "centroid_residuals_illumination_frame_high_n.png" if out_dir else None
    fig1, ax1 = plot_radius_residuals_vs_range(
        df_radius, radius_1sigma_px=1.0, target_label="Earth", save_path=save_radius
    )
    assert fig1 is not None and ax1 is not None
    if save_radius:
        assert save_radius.exists(), f"Expected {save_radius} to be created"
    fig2, (ax2a, ax2b) = plot_centroid_residuals_in_illumination_frame(
        df_centroid, centroid_1sigma_px=1.0, target_label="Earth", save_path=save_centroid
    )
    assert fig2 is not None and ax2a is not None and ax2b is not None
    if save_centroid:
        assert save_centroid.exists(), f"Expected {save_centroid} to be created"


def main() -> int:
    parser = argparse.ArgumentParser(description="Test analysis plots (no edge_plot).")
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Write plot images to scripts/test_plot_outputs/",
    )
    parser.add_argument(
        "--high-n",
        type=int,
        metavar="N",
        default=5000,
        help="Run density/hexbin/subsample test with N points (default 5000). Use 0 to skip. With --keep, saves high-n plots.",
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
        if args.high_n > 0:
            test_density_scaled_high_n(out_dir, n=args.high_n)
            print(f"OK density/hexbin/subsample plots (n={args.high_n})")
    except Exception as e:
        print(f"FAIL: {e}", file=sys.stderr)
        raise
    print("All analysis plot tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
