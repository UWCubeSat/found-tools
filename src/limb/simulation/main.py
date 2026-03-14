import argparse
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import numpy as np

from limb.simulation.metadata.orchestrate import initialize_sim_df, setup_expirement
from limb.simulation.metadata.state import (
    generate_uniform_directions,
)
from limb.simulation.render import conic as render_conic
from limb.utils._camera import Camera


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate simulated limb images by creating satellite states from metadata, "
            "projecting conics into pixel space, and rendering the result."
        )
    )

    parser.add_argument(
        "--semi-axes",
        nargs=3,
        type=float,
        default=(6378137, 6378137, 6356752.31424518),
        metavar=("A", "B", "C"),
    )
    parser.add_argument(
        "--earth-point-direction",
        nargs=3,
        type=float,
        required=False,
        metavar=("X", "Y", "Z"),
        help="Single earth-point direction vector. If omitted, directions are generated with --num-directions.",
    )
    parser.add_argument(
        "--num-directions",
        type=int,
        default=1,
        help="Number of uniform directions to generate when --earth-point-direction is omitted.",
    )
    parser.add_argument(
        "--distance",
        type=float,
        required=True,
        help="Distance from the center of the ellipsoid to the satellite (meters).",
    )
    parser.add_argument(
        "--num-satellite-positions",
        type=int,
        required=True,
        help="Number of satellite positions to generate.",
    )
    parser.add_argument(
        "--num-image-spins",
        type=int,
        required=True,
        help="Number of image spins to generate.",
    )
    parser.add_argument(
        "--num-image-radials",
        type=int,
        required=True,
        help="Number of image radials to generate.",
    )

    parser.add_argument(
        "--focal-length",
        type=float,
        required=True,
        help="Focal length of the camera (meters).",
    )
    parser.add_argument(
        "--x-pixel-pitch",
        type=float,
        required=True,
        help="Pixel pitch in the x-direction (meters).",
    )
    parser.add_argument(
        "--y-pixel-pitch",
        type=float,
        default=None,
        help="Pixel pitch in the y-direction (meters).",
    )
    parser.add_argument(
        "--x-resolution",
        type=int,
        required=True,
        help="Number of pixels in the x-direction.",
    )
    parser.add_argument(
        "--y-resolution",
        type=int,
        required=True,
        help="Number of pixels in the y-direction.",
    )
    parser.add_argument(
        "--x-center",
        type=float,
        default=None,
        help="X-coordinate of the principal point (pixels).",
    )
    parser.add_argument(
        "--y-center",
        type=float,
        default=None,
        help="Y-coordinate of the principal point (pixels).",
    )

    parser.add_argument(
        "--output-folder",
        type=Path,
        required=True,
        help="Folder to save the generated images.",
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
    parser.add_argument(
        "--save-coeffs",
        type=Path,
        default=None,
        help="Path to save the conic coefficients.",
    )

    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if any(axis <= 0 for axis in args.semi_axes):
        raise ValueError("All semi-axes must be positive.")
    if args.distance <= 0:
        raise ValueError("--distance must be > 0.")
    if args.num_satellite_positions < 1:
        raise ValueError("--num-satellite-positions must be >= 1.")
    if args.num_image_spins < 1:
        raise ValueError("--num-image-spins must be >= 1.")
    if args.num_image_radials < 1:
        raise ValueError("--num-image-radials must be >= 1.")
    if args.num_directions < 1:
        raise ValueError("--num-directions must be >= 1.")
    if args.x_resolution < 1 or args.y_resolution < 1:
        raise ValueError("Camera resolution must be >= 1 in each axis.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")
    if args.sigma <= 0:
        raise ValueError("--sigma must be > 0.")


def main() -> None:  # pragma: no cover
	args = _parse_args()
	_validate_args(args)

	if args.seed is not None:
		np.random.seed(args.seed)

	if args.earth_point_direction is not None:
		earth_directions = np.array([args.earth_point_direction], dtype=np.float64)
	else:
		earth_directions = generate_uniform_directions(args.num_directions)

	camera = Camera(
		focal_length=args.focal_length,
		x_pixel_pitch=args.x_pixel_pitch,
		y_pixel_pitch=args.y_pixel_pitch,
		x_resolution=args.x_resolution,
		y_resolution=args.y_resolution,
		x_center=args.x_center,
		y_center=args.y_center,
	)

	df = initialize_sim_df()

	df, conic_coeffs = setup_expirement(
							df,
							args.semi_axes,
							earth_directions,
							args.distance,
							camera,
							args.num_satellite_positions,
							args.num_image_spins,
							args.num_image_radials)

	render_conic.process_simulation(
		coeffs_nx6=render_conic.torch.from_numpy(
			np.array(conic_coeffs, dtype=np.float32)
		),
		width=args.x_resolution,
		height=args.y_resolution,
		output_folder=str(args.output_folder),
		batch_size=args.batch_size,
		sigma=args.sigma,
	)

if __name__ == "__main__":
    main()
