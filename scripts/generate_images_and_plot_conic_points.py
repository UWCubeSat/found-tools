#!/usr/bin/env python3
"""Generate simulated limb images with setup_simulation, then overlay conic edge points.

1. Runs setup_simulation to write metadata CSV.
2. Renders conic images (same pipeline as limb.simulation.main).
3. For each image, computes edge points via the conic module and plots them on the image.
4. Saves the overlaid images to an output folder.

Usage:
    python scripts/generate_images_and_plot_conic_points.py
    python scripts/generate_images_and_plot_conic_points.py --max-images 4 --output-points-dir sim_overlay
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from limb.simulation.metadata.orchestrate import (
    _calculate_conic_coeffs,
    points_from_row,
    setup_simulation,
)
from limb.simulation.render import conic as render_conic


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate sim images with setup_simulation and plot conic edge points on them."
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("sim_metadata.csv"),
        help="Path for simulation metadata CSV (default: sim_metadata.csv).",
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        default=Path("sim_images"),
        help="Folder for rendered conic images (default: sim_images).",
    )
    parser.add_argument(
        "--output-points-dir",
        type=Path,
        default=Path("sim_images_with_points"),
        help="Folder for images with conic points overlaid (default: sim_images_with_points).",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Max number of images to generate and plot (default: all).",
    )
    parser.add_argument(
        "--semi-axes",
        nargs=3,
        type=float,
        default=(6378137.0, 6378137.0, 6356752.31424518),
        metavar=("A", "B", "C"),
        help="Ellipsoid semi-axes a, b, c (m).",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=2.0,
        help="Gaussian blur sigma for rendered edge (default: 2.0).",
    )
    parser.add_argument(
        "--point-sigma",
        type=float,
        default=None,
        help="Gaussian sigma for edge point noise in pixels (default: none).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Batch size for rendering (default: 500).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def _small_sim_grid():
    """Small grid so the script runs quickly by default."""
    return {
        "fovs": [10.0],
        "resolutions": [256],
        "distances": [7.5e6],
        "num_earth_points": 1,
        "num_positions_per_point": 2,
        "num_spins_per_position": 2,
        "num_radials_per_spin": 2,
    }


def run_setup_and_render(args):
    """Run setup_simulation (writes CSV) and render conic images."""
    grid = _small_sim_grid()
    setup_simulation(
        semi_axes=list(args.semi_axes),
        fovs=grid["fovs"],
        resolutions=grid["resolutions"],
        distances=grid["distances"],
        num_earth_points=grid["num_earth_points"],
        num_positions_per_point=grid["num_positions_per_point"],
        num_spins_per_position=grid["num_spins_per_position"],
        num_radials_per_spin=grid["num_radials_per_spin"],
        output_path=str(args.output_csv),
    )

    coeffs_by_res = _calculate_conic_coeffs(args.output_csv)
    args.output_folder.mkdir(parents=True, exist_ok=True)
    for (width, height), (row_indices, coeffs, K, rc) in coeffs_by_res.items():
        render_conic.process_simulation(
            coeffs_nx6=torch.from_numpy(coeffs),
            width=int(width),
            height=int(height),
            output_folder=str(args.output_folder),
            K=torch.from_numpy(K),
            rc=torch.from_numpy(rc),
            batch_size=args.batch_size,
            sigma=args.sigma,
            row_indices=row_indices,
            noise_config=None,
        )


def plot_points_on_image(image_path: Path, points: np.ndarray, out_path: Path) -> None:
    """Load image, draw points (x, y) as red circles, save to out_path."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    # Red in BGR
    color = (0, 0, 255)
    radius = 2
    for x, y in points.astype(np.float32):
        cx, cy = int(round(x)), int(round(y))
        cv2.circle(img, (cx, cy), radius, color, -1)
    cv2.imwrite(str(out_path), img)


def main():
    args = _parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)

    print("Running setup_simulation and rendering conic images...")
    run_setup_and_render(args)

    df = pd.read_csv(args.output_csv, index_col=0)
    n = len(df)
    if args.max_images is not None:
        n = min(n, args.max_images)
    indices = df.index[:n].tolist()

    args.output_points_dir.mkdir(parents=True, exist_ok=True)
    kwargs = {} if args.point_sigma is None else {"gaussian_sigma": args.point_sigma}
    for idx in indices:
        row = df.loc[idx]
        points = points_from_row(row, **kwargs)
        image_path = args.output_folder / f"img_{idx:06d}.png"
        if not image_path.exists():
            print(f"Skip {idx}: image not found {image_path}")
            continue
        out_path = args.output_points_dir / f"img_{idx:06d}.png"
        if points.size == 0:
            print(f"Skip {idx}: no edge points in image")
            continue
        plot_points_on_image(image_path, points, out_path)
        print(f"Wrote {out_path} ({len(points)} points)")

    print(f"Done. Overlaid images in {args.output_points_dir}")


if __name__ == "__main__":
    main()
