#!/usr/bin/env python3
"""Add ``delta_distance`` to a simulation metadata CSV.

Computes **signed** difference in metres between estimated and true slant range::

    delta_distance = ‖out_pos‖ − ‖true_pos‖

using ``out_pos_*`` and ``true_pos_*``. Rows with any missing ``out_pos_*`` or
``true_pos_*`` get ``NaN`` for ``delta_distance``.

Examples::

    uv run python scripts/add_delta_distance_column.py huge-multi-cam-angle-noise.csv -o with_delta.csv
    uv run python scripts/add_delta_distance_column.py sim_metadata.csv --in-place
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def add_delta_distance_column(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with ``delta_distance`` (m)."""
    need_true = ("true_pos_x", "true_pos_y", "true_pos_z")
    need_out = ("out_pos_x", "out_pos_y", "out_pos_z")
    for col in need_true + need_out:
        if col not in df.columns:
            raise ValueError(f"CSV must contain column {col!r}")

    true = df.loc[:, list(need_true)].to_numpy(dtype=np.float64, copy=True)
    out = df.loc[:, list(need_out)].to_numpy(dtype=np.float64, copy=True)
    true_norm = np.linalg.norm(true, axis=1)
    out_norm = np.linalg.norm(out, axis=1)
    valid = (
        np.isfinite(true).all(axis=1)
        & np.isfinite(out).all(axis=1)
    )
    delta = np.full(len(df), np.nan, dtype=np.float64)
    delta[valid] = out_norm[valid] - true_norm[valid]

    out_df = df.copy()
    if "delta_distance" in out_df.columns:
        raise ValueError("Column delta_distance already exists; drop it first or use a different output.")
    out_df["delta_distance"] = delta
    return out_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv",
        type=Path,
        help="Input CSV path",
    )
    g = parser.add_mutually_exclusive_group(required=False)
    g.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: <input_stem>_with_delta_distance.csv next to input)",
    )
    g.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input file",
    )
    args = parser.parse_args()

    path = args.csv.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Not a file: {path}")

    if args.in_place:
        out_path = path
    elif args.output is not None:
        out_path = args.output.expanduser().resolve()
    else:
        out_path = path.with_name(f"{path.stem}_with_delta_distance{path.suffix}")

    print(f"Reading {path} …")
    df = pd.read_csv(path)
    n = len(df)
    print(f"Rows: {n}")

    df_out = add_delta_distance_column(df)
    non_nan = int(df_out["delta_distance"].notna().sum())
    print(f"delta_distance: {non_nan} finite values, {n - non_nan} NaN")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
