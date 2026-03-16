import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from limb.simulation.metadata.orchestrate import (
    _calculate_conic_coeffs,
    setup_simulation,
)
from limb.simulation.render import conic as render_conic


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate simulated limb images by running setup_simulation (metadata + conics) "
            "and rendering conics to images."
        )
    )

    parser.add_argument(
        "--semi-axes",
        nargs=3,
        type=float,
        default=(6378137, 6378137, 6356752.31424518),
        metavar=("A", "B", "C"),
        help="Ellipsoid semi-axes a, b, c (m).",
    )
    parser.add_argument(
        "--fovs",
        nargs="+",
        type=float,
        required=True,
        metavar="FOV",
        help="Field-of-view values (degrees) to sweep.",
    )
    parser.add_argument(
        "--resolutions",
        nargs="+",
        type=int,
        required=True,
        metavar="N",
        help="Sensor resolutions (pixels, square) to sweep.",
    )
    parser.add_argument(
        "--distances",
        nargs="+",
        type=float,
        required=True,
        metavar="D",
        help="Distances from ellipsoid center to satellite (m) to sweep.",
    )
    parser.add_argument(
        "--num-earth-points",
        type=int,
        default=1,
        help="Number of uniform earth-point directions.",
    )
    parser.add_argument(
        "--num-positions-per-point",
        type=int,
        required=True,
        help="Number of satellite positions per earth point.",
    )
    parser.add_argument(
        "--num-spins-per-position",
        type=int,
        required=True,
        help="Number of image spins per position.",
    )
    parser.add_argument(
        "--num-radials-per-spin",
        type=int,
        required=True,
        help="Number of image radials per spin.",
    )

    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("sim_metadata.csv"),
        help="Path to save the simulation metadata CSV (relative to cwd). Default: sim_metadata.csv",
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        default=Path("sim_images"),
        help="Folder to save the generated images (relative to cwd). Default: sim_images",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Number of images to generate per batch.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=2.0,
        help="Standard deviation of Gaussian blur applied to the edge of rendered images.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if any(axis <= 0 for axis in args.semi_axes):
        raise ValueError("All semi-axes must be positive.")
    if not args.fovs or any(f <= 0 for f in args.fovs):
        raise ValueError("--fovs must be non-empty and all positive.")
    if not args.resolutions or any(r < 1 for r in args.resolutions):
        raise ValueError("--resolutions must be non-empty and all >= 1.")
    if not args.distances or any(d <= 0 for d in args.distances):
        raise ValueError("--distances must be non-empty and all positive.")
    if args.num_earth_points < 1:
        raise ValueError("--num-earth-points must be >= 1.")
    if args.num_positions_per_point < 1:
        raise ValueError("--num-positions-per-point must be >= 1.")
    if args.num_spins_per_position < 1:
        raise ValueError("--num-spins-per-position must be >= 1.")
    if args.num_radials_per_spin < 1:
        raise ValueError("--num-radials-per-spin must be >= 1.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")
    if args.sigma <= 0:
        raise ValueError("--sigma must be > 0.")


def main() -> None:  # pragma: no cover
    args = _parse_args()
    _validate_args(args)

    if args.seed is not None:
        np.random.seed(args.seed)

    setup_simulation(
        semi_axes=list(args.semi_axes),
        fovs=args.fovs,
        resolutions=args.resolutions,
        distances=args.distances,
        num_earth_points=args.num_earth_points,
        num_positions_per_point=args.num_positions_per_point,
        num_spins_per_position=args.num_spins_per_position,
        num_radials_per_spin=args.num_radials_per_spin,
        output_path=str(args.output_csv),
    )

    coeffs_and_centroid_by_resolution = _calculate_conic_coeffs(args.output_csv)
    args.output_folder.mkdir(parents=True, exist_ok=True)
    for (width, height), (row_indices, coeffs, K, rc) in coeffs_and_centroid_by_resolution.items():
        render_conic.process_simulation(
            coeffs_nx6=render_conic.torch.from_numpy(coeffs),
            width=width,
            height=height,
            output_folder=str(args.output_folder),
            K=render_conic.torch.from_numpy(K),
            rc=render_conic.torch.from_numpy(rc),
            batch_size=args.batch_size,
            sigma=args.sigma,
            row_indices=row_indices,
        )


if __name__ == "__main__":
    main()
