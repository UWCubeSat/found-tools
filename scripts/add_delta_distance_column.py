#!/usr/bin/env python3
"""Add ``delta_distance`` to a simulation metadata CSV.

Computes **3-D position error** in metres (same definition as
``limb.simulation.analysis.metrics``)::

    delta_distance = ‖out_pos − true_pos‖

using ``out_pos_*`` and ``true_pos_*``. Rows with any missing ``out_pos_*`` or
``true_pos_*`` get ``NaN`` for ``delta_distance``.

If ``delta_x_centroid`` and ``delta_y_centroid`` are present, **``delta_centroid``**
is set to ``max(|Δx|,|Δy|)`` (px); ``column < bound`` then means both in-plane
centroid errors are strictly below that bound.

Any existing ``delta_distance`` or legacy ``delta_range`` column is replaced;
``delta_centroid`` is overwritten when centroid columns exist.

With ``--split-by camera``, writes one CSV per camera configuration (same keys as
:func:`~limb.simulation.analysis.metrics.split_df_by_camera`). With
``--split-by regression``, writes one CSV per distinct value of
``--regression-column`` (default ``distance_regression``).

Examples::

    uv run python scripts/add_delta_distance_column.py data/regression-26.csv
    uv run python scripts/add_delta_distance_column.py sim.csv -o copy.csv
    uv run python scripts/add_delta_distance_column.py sim.csv --split-by camera -o out_parts/
    uv run python scripts/add_delta_distance_column.py sim.csv --split-by regression
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from limb.simulation.analysis.metrics import split_df_by_camera, split_df_by_column_value


def add_delta_centroid_column(out_df: pd.DataFrame) -> None:
    """Set ``delta_centroid`` = ``max(|Δx|,|Δy|)`` when centroid delta columns exist."""
    need = ("delta_x_centroid", "delta_y_centroid")
    if not all(c in out_df.columns for c in need):
        return
    dx = out_df["delta_x_centroid"].to_numpy(dtype=np.float64, copy=False)
    dy = out_df["delta_y_centroid"].to_numpy(dtype=np.float64, copy=False)
    ok = np.isfinite(dx) & np.isfinite(dy)
    mag = np.full(len(out_df), np.nan, dtype=np.float64)
    mag[ok] = np.maximum(np.abs(dx[ok]), np.abs(dy[ok]))
    out_df["delta_centroid"] = mag


def add_delta_distance_column(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with ``delta_distance`` (m) and optional ``delta_centroid``."""
    need_true = ("true_pos_x", "true_pos_y", "true_pos_z")
    need_out = ("out_pos_x", "out_pos_y", "out_pos_z")
    for col in need_true + need_out:
        if col not in df.columns:
            raise ValueError(f"CSV must contain column {col!r}")

    true = df.loc[:, list(need_true)].to_numpy(dtype=np.float64, copy=True)
    out = df.loc[:, list(need_out)].to_numpy(dtype=np.float64, copy=True)
    valid = np.isfinite(true).all(axis=1) & np.isfinite(out).all(axis=1)
    delta = np.full(len(df), np.nan, dtype=np.float64)
    delta[valid] = np.linalg.norm(out[valid] - true[valid], axis=1)

    out_df = df.copy()
    for legacy in ("delta_range", "delta_distance"):
        if legacy in out_df.columns:
            out_df = out_df.drop(columns=[legacy])
    out_df["delta_distance"] = delta
    add_delta_centroid_column(out_df)
    return out_df


def _slug_segment(s: str) -> str:
    t = re.sub(r"[^\w.\-]+", "_", s).strip("_")
    return t or "key"


def _slug_camera_key(key: tuple[float, int, int]) -> str:
    fl, rx, ry = key
    return f"cam_f{_slug_segment(f'{fl:g}')}_r{rx}x{ry}"


def _slug_regression_key(key: Any) -> str:
    return f"reg_{_slug_segment(str(key))}"


def _resolve_split_output_dir(path: Path, output: Path | None) -> Path:
    root = output.expanduser().resolve() if output is not None else path.parent
    if root.exists() and root.is_file():
        raise ValueError(
            "With --split-by camera or regression, --output must be a directory "
            f"(got file: {root})"
        )
    if not root.exists() and root.suffix.lower() == ".csv":
        raise ValueError(
            "With --split-by camera or regression, pass a directory for --output "
            f"(got {root}); use --split-by none for a single .csv path."
        )
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv",
        type=Path,
        help="Input CSV path",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Without --split-by: output CSV file (default: overwrite input). "
            "With --split-by: output directory (default: same dir as input)."
        ),
    )
    parser.add_argument(
        "--split-by",
        choices=("none", "camera", "regression"),
        default="none",
        help="Write separate CSVs per camera configuration or per regression value",
    )
    parser.add_argument(
        "--regression-column",
        default="distance_regression",
        help="Column for --split-by regression (default: distance_regression)",
    )
    args = parser.parse_args()

    path = args.csv.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Not a file: {path}")

    print(f"Reading {path} …")
    df = pd.read_csv(path)
    n = len(df)
    print(f"Rows: {n}")

    if args.split_by == "none":
        out_path = args.output.expanduser().resolve() if args.output is not None else path
        df_out = add_delta_distance_column(df)
        non_nan = int(df_out["delta_distance"].notna().sum())
        print(f"delta_distance: {non_nan} finite values, {n - non_nan} NaN")
        if "delta_centroid" in df_out.columns:
            nc = int(df_out["delta_centroid"].notna().sum())
            print(f"delta_centroid (max|Δx|,|Δy|): {nc} finite values, {n - nc} NaN")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(out_path, index=False)
        print(f"Wrote {out_path}")
        return

    out_dir = _resolve_split_output_dir(path, args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = path.stem

    if args.split_by == "camera":
        groups = split_df_by_camera(df)
        print(f"Split into {len(groups)} camera group(s)")
        for key, sub in sorted(groups.items(), key=lambda kv: kv[0]):
            slug = _slug_camera_key(key)
            df_out = add_delta_distance_column(sub)
            nn = int(df_out["delta_distance"].notna().sum())
            out_path = out_dir / f"{stem}_{slug}.csv"
            print(f"  {slug}: rows {len(sub)}, delta_distance finite {nn} → {out_path.name}")
            df_out.to_csv(out_path, index=False)
        print(f"Wrote under {out_dir}")
        return

    groups = split_df_by_column_value(df, args.regression_column)
    print(
        f"Split into {len(groups)} group(s) on {args.regression_column!r}"
    )
    for key, sub in groups.items():
        slug = _slug_regression_key(key)
        df_out = add_delta_distance_column(sub)
        nn = int(df_out["delta_distance"].notna().sum())
        out_path = out_dir / f"{stem}_{slug}.csv"
        print(f"  {slug}: rows {len(sub)}, delta_distance finite {nn} → {out_path.name}")
        df_out.to_csv(out_path, index=False)
    print(f"Wrote under {out_dir}")


if __name__ == "__main__":
    main()
