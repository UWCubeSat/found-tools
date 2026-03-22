#!/usr/bin/env python3
"""Plot range (distance) vs a metric for each distinct camera in a simulation CSV.

Uses ``split_df_by_camera`` so each panel is one camera configuration
``(cam_focal_length, cam_x_resolution, cam_y_resolution)``.

Writes two PNGs under ``scripts/test_plot_outputs/`` (or ``--output-dir``):

1. Facets — downsampled scatter of *column* vs ||true position|| (m) per camera.
2. Coverage — horizontal segments showing min–max distance per camera.

Examples:

  uv run python scripts/test_plot_camera_ranges.py --csv huge-multi-cam-angle-noise.csv
  uv run python scripts/test_plot_camera_ranges.py --column delta_r_apparent --max-points 1200
  uv run python scripts/test_plot_camera_ranges.py --fovs 70 80
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from limb.simulation.analysis.metrics import (
    fill_pixel_metrics,
    filter_df_by_fovs,
    split_df_by_camera,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _default_csv_path() -> Path:
    root = _repo_root()
    for name in (
        "huge-multi-cam-angle-noise.csv",
        "distance_multi-camera-leo-geo.csv",
        "sim_metadata.csv",
    ):
        p = root / name
        if p.is_file():
            return p
    raise FileNotFoundError(
        "No default CSV found in repo root. Pass --csv explicitly. "
        f"Tried: huge-multi-cam-angle-noise.csv, distance_multi-camera-leo-geo.csv, "
        f"sim_metadata.csv under {root}"
    )


def _distance_m(sub: pd.DataFrame) -> np.ndarray:
    for c in ("true_pos_x", "true_pos_y", "true_pos_z"):
        if c not in sub.columns:
            raise ValueError(f"CSV needs {c} for range (distance) axis")
    pos = sub[["true_pos_x", "true_pos_y", "true_pos_z"]].to_numpy(dtype=np.float64)
    return np.linalg.norm(pos, axis=1)


def _camera_panel_title(cam_id: tuple[float, int, int]) -> str:
    f_len, wx, wy = cam_id
    return f"{wx}×{wy} px\nf = {f_len:.4g} m"


def _downsample_idx(n: int, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if n <= max_points:
        return np.arange(n)
    return rng.choice(n, size=max_points, replace=False)


def plot_facets_per_camera(
    df: pd.DataFrame,
    column: str,
    *,
    max_points: int,
    ncols: int,
    out_path: Path,
) -> None:
    by_cam = split_df_by_camera(df)
    if not by_cam:
        raise ValueError("No camera groups in CSV (missing cam_* columns?)")

    cams = sorted(by_cam.keys(), key=lambda k: (k[0], k[1], k[2]))
    n = len(cams)
    ncols = max(1, min(ncols, n))
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.2 * ncols, 3.4 * nrows),
        squeeze=False,
        sharex=False,
        sharey=False,
    )
    rng = np.random.default_rng(0)

    for i, cam_id in enumerate(cams):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        sub = by_cam[cam_id]
        dist = _distance_m(sub)
        y = sub[column].to_numpy(dtype=np.float64)
        valid = np.isfinite(dist) & np.isfinite(y)
        dist = dist[valid]
        y = y[valid]
        if dist.size == 0:
            ax.set_title(_camera_panel_title(cam_id))
            ax.text(0.5, 0.5, "no valid rows", ha="center", va="center", transform=ax.transAxes)
            continue

        idx = _downsample_idx(dist.size, max_points, rng)
        ax.scatter(dist[idx], y[idx], s=4, alpha=0.45, c="tab:blue", edgecolors="none")
        ax.set_title(_camera_panel_title(cam_id), fontsize=9)
        ax.set_xlabel("Range ‖r‖ (m)")
        ax.set_ylabel(column)
        ax.grid(True, alpha=0.3)
        d0, d1 = float(np.min(dist)), float(np.max(dist))
        ax.set_xlim(d0 - 0.02 * (d1 - d0 or 1.0), d1 + 0.02 * (d1 - d0 or 1.0))

    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].set_visible(False)

    fig.suptitle(f"{column} vs range — one panel per camera configuration", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved facet plot: {out_path}")


def plot_distance_coverage_per_camera(
    df: pd.DataFrame,
    *,
    out_path: Path,
) -> None:
    by_cam = split_df_by_camera(df)
    cams = sorted(by_cam.keys(), key=lambda k: (k[0], k[1], k[2]))
    lows: list[float] = []
    highs: list[float] = []
    for cam_id in cams:
        sub = by_cam[cam_id]
        dist = _distance_m(sub)
        dist = dist[np.isfinite(dist)]
        if dist.size == 0:
            lows.append(float("nan"))
            highs.append(float("nan"))
        else:
            lows.append(float(np.min(dist)))
            highs.append(float(np.max(dist)))

    fig, ax = plt.subplots(figsize=(10, max(3.0, 0.35 * len(cams))))
    y = np.arange(len(cams))
    cmap = plt.get_cmap("tab10")
    for yi, lo, hi in zip(y, lows, highs, strict=True):
        if not (np.isfinite(lo) and np.isfinite(hi)):
            continue
        ax.plot(
            [lo, hi],
            [yi, yi],
            color=cmap(int(yi) % cmap.N),
            linewidth=5,
            solid_capstyle="butt",
        )

    ax.set_yticks(y)
    ax.set_yticklabels([f"{cid[1]}×{cid[2]}, f={cid[0]:.4g}" for cid in cams], fontsize=8)
    ax.set_xlabel("Distance ‖true position‖ (m)")
    ax.set_title("Distance range covered in CSV (per camera configuration)")
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_ylim(-0.5, len(cams) - 0.5)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved coverage plot: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Path to simulation metadata CSV (default: first match in repo root)",
    )
    parser.add_argument(
        "--column",
        default="true_r_apparent",
        help="Y-axis column (default: true_r_apparent; use delta_r_apparent after metrics)",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=800,
        help="Max scatter points per camera panel (downsampled)",
    )
    parser.add_argument(
        "--ncols",
        type=int,
        default=3,
        help="Number of columns in facet grid",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: scripts/test_plot_outputs under repo)",
    )
    parser.add_argument(
        "--no-fill-metrics",
        action="store_true",
        help="Skip fill_pixel_metrics (use if column already exists and no pose fill needed)",
    )
    parser.add_argument(
        "--fovs",
        type=float,
        nargs="*",
        default=None,
        metavar="DEG",
        help="Include only these horizontal FOVs (degrees); exclude all other cameras",
    )
    parser.add_argument(
        "--fov-match-atol",
        type=float,
        default=1e-3,
        help="Degrees tolerance for matching --fovs (default: 1e-3)",
    )
    args = parser.parse_args()

    csv_path = args.csv.resolve() if args.csv is not None else _default_csv_path()
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    out_dir = args.output_dir
    if out_dir is None:
        out_dir = _repo_root() / "scripts" / "test_plot_outputs"
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = csv_path.stem.replace(" ", "_")
    fov_tag = ""
    if args.fovs:
        fov_tag = "_fov-" + "-".join(
            str(int(f)) if f == int(f) else str(f) for f in args.fovs
        )
    facet_path = out_dir / f"camera_ranges_facets_{stem}{fov_tag}.png"
    coverage_path = out_dir / f"camera_ranges_coverage_{stem}{fov_tag}.png"

    print(f"Loading {csv_path} …")
    df = pd.read_csv(csv_path)
    print(f"Rows: {len(df)}")

    if not args.no_fill_metrics:
        print("fill_pixel_metrics …")
        df = fill_pixel_metrics(df)

    if args.fovs:
        df = filter_df_by_fovs(df, args.fovs, fov_match_atol=args.fov_match_atol)
        print(f"Filtered to FOVs (deg) {list(args.fovs)} — rows: {len(df)}")

    if args.column not in df.columns:
        raise ValueError(
            f"Column {args.column!r} not in DataFrame. "
            f"Try --column true_r_apparent or run without --no-fill-metrics."
        )

    n_cam = len(split_df_by_camera(df))
    print(f"Distinct camera configurations: {n_cam}")

    plot_facets_per_camera(
        df,
        args.column,
        max_points=args.max_points,
        ncols=args.ncols,
        out_path=facet_path,
    )
    plot_distance_coverage_per_camera(df, out_path=coverage_path)
    print("Done.")


if __name__ == "__main__":
    main()
