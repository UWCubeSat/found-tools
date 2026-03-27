import argparse
from pathlib import Path
import numpy as np

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
        help="Gaussian blur sigma for the limb edge (must be > 0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    # Optional noise pipeline (applied during render)
    parser.add_argument(
        "--noise-gaussian",
        nargs=2,
        type=float,
        metavar=("MEAN", "SIGMA"),
        default=None,
        help=(
            "Add Gaussian noise: mean and sigma (pixel units); sigma 0 disables. "
            "e.g. 0 10 or 0 0 for none."
        ),
    )
    parser.add_argument(
        "--noise-stars",
        type=float,
        metavar="PROB",
        default=None,
        help="Add salt (white) noise for stars; PROB = fraction of pixels. e.g. 0.005.",
    )
    parser.add_argument(
        "--noise-dead-pixels",
        nargs=2,
        type=float,
        metavar=("SALT", "PEPPER"),
        default=None,
        help="Add salt-and-pepper as last step (dead pixels): salt_prob pepper_prob. e.g. 0.01 0.01.",
    )
    parser.add_argument(
        "--noise-salt-pepper",
        nargs=2,
        type=float,
        metavar=("SALT", "PEPPER"),
        dest="noise_dead_pixels",
        help="Alias for --noise-dead-pixels (deprecated).",
    )
    parser.add_argument(
        "--noise-discretization",
        type=int,
        metavar="LEVELS",
        default=None,
        help="Discretize intensity to LEVELS per channel; 0 skips. e.g. 8.",
    )
    parser.add_argument(
        "--noise-motion-blur",
        type=int,
        metavar="KERNEL",
        default=None,
        help="Apply motion blur with kernel size KERNEL (odd); 0 skips. e.g. 5.",
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
        raise ValueError("--sigma must be positive.")
    if args.noise_gaussian is not None and args.noise_gaussian[1] < 0:
        raise ValueError("--noise-gaussian sigma must be >= 0 (use 0 for no sensor noise).")
    if args.noise_stars is not None and (args.noise_stars < 0 or args.noise_stars > 1):
        raise ValueError("--noise-stars must be in [0, 1].")
    if args.noise_discretization is not None and args.noise_discretization < 0:
        raise ValueError("--noise-discretization must be >= 0 (use 0 to skip).")
    if args.noise_motion_blur is not None and args.noise_motion_blur != 0:
        if args.noise_motion_blur < 1 or args.noise_motion_blur % 2 == 0:
            raise ValueError(
                "--noise-motion-blur must be 0 (skip) or an odd integer >= 1."
            )


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

    noise_config: dict = {}
    if args.noise_gaussian is not None:
        g_mean, g_sigma = float(args.noise_gaussian[0]), float(args.noise_gaussian[1])
        if g_mean != 0.0 or g_sigma != 0.0:
            noise_config["gaussian"] = {"mean": g_mean, "sigma": g_sigma}
    if args.noise_stars is not None and args.noise_stars > 0.0:
        noise_config["stars"] = {"prob": float(args.noise_stars)}
    if args.noise_dead_pixels is not None and (
        args.noise_dead_pixels[0] > 0.0 or args.noise_dead_pixels[1] > 0.0
    ):
        noise_config["dead_pixels"] = {
            "salt_prob": float(args.noise_dead_pixels[0]),
            "pepper_prob": float(args.noise_dead_pixels[1]),
        }
    if args.noise_discretization is not None and args.noise_discretization > 0:
        noise_config["discretization"] = {
            "levels": int(args.noise_discretization),
        }
    if args.noise_motion_blur is not None and args.noise_motion_blur > 0:
        noise_config["motion_blur"] = {
            "kernel_size": int(args.noise_motion_blur),
        }
    noise_config = noise_config if noise_config else None

    coeffs_and_centroid_by_resolution = _calculate_conic_coeffs(args.output_csv)
    args.output_folder.mkdir(parents=True, exist_ok=True)
    for (width, height), (
        row_indices,
        coeffs,
        K,
        rc,
    ) in coeffs_and_centroid_by_resolution.items():
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
            noise_config=noise_config,
        )


if __name__ == "__main__":
    main()
