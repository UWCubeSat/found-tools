import argparse
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import numpy as np

from limb.simulation.edge.conic import generate_camera_conic, generatePixelConic
from limb.simulation.metadata.df import _fill_cam_columns, _initialize_sim_df
from limb.simulation.metadata.state import (
    generate_satellite_state,
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


def _shape_matrix_from_axes(semi_axes: list[float]) -> np.ndarray:
    a, b, c = semi_axes
    return np.diag([1.0 / (a * a), 1.0 / (b * b), 1.0 / (c * c)])


def _conic_matrix_to_coeffs(conic: np.ndarray) -> np.ndarray:
    # Conic matrix C corresponds to Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0.
    return np.array(
        [
            conic[0, 0],
            2.0 * conic[0, 1],
            conic[1, 1],
            2.0 * conic[0, 2],
            2.0 * conic[1, 2],
            conic[2, 2],
        ],
        dtype=np.float32,
    )


def main() -> None:  # pragma: no cover
    args = _parse_args()
    _validate_args(args)

    if args.seed is not None:
        np.random.seed(args.seed)

    shape_matrix = _shape_matrix_from_axes(args.semi_axes)

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

    # sat_positions, r_tcps = _sat_position_boresight(
    # 	shape_matrix,
    # 	earth_directions,
    # 	args.num_satellite_positions,
    # 	args.distance,
    # )

    sat_positions, tcps = generate_satellite_state(
        shape_matrix,
        earth_directions,
        args.distance,
        camera,
        args.num_satellite_positions,
        args.num_image_spins,
        args.num_image_radials,
    )
    df = _initialize_sim_df(sat_positions.shape[0])
    df = _fill_cam_columns(df, camera)

    # tcps = r_tcps.as_matrix()

    conic_coeffs = []
    for tcp, sat_pos in zip(tcps, sat_positions):
        tpc = tcp.T  # Convert to world-to-camera rotation matrices.
        sph_pos = tpc @ sat_pos.T

        image_conic = generate_camera_conic(sph_pos, shape_matrix, tpc)
        pixel_conic = generatePixelConic(image_conic, camera)
        conic_coeffs.append(_conic_matrix_to_coeffs(pixel_conic))

        df[["true_pos_x", "true_pos_y", "true_pos_z"]] = sat_positions
        df[["shape_axis_a", "shape_axis_b", "shape_axis_c"]] = np.array(args.semi_axes)
        df[["true_attitude_ra", "true_attitude_dec", "true_attitude_roll"]] = (
            R.from_matrix(tcp).as_euler("zyx", degrees=True)
        )

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


# 	conic_coeffs = []
# 	state_records = []
# 	for earth_point_direction in earth_directions:
# 		satellite_positions, satellite_orientations = generateSatelliteState(
# 			shape_matrix,
# 			earth_point_direction,
# 			args.distance,
# 			camera,
# 			args.num_satellite_positions,
# 			args.num_satellite_orientations,
# 		)

# 		for orient_idx in range(args.num_satellite_orientations):
# 			for pos_idx in range(args.num_satellite_positions):
# 				orientation = satellite_orientations[orient_idx, pos_idx]
# 				# Transform satellite position from world frame to camera frame (rc = TPC @ rp).
# 				TPC = np.linalg.inv(orientation)
# 				rc = TPC @ satellite_positions[pos_idx]
# 				camera_conic = generateCameraConic(
# 					rc,
# 					shape_matrix,
# 					TPC,
# 				)
# 				pixel_conic = generatePixelConic(camera_conic, camera.inverseCalibrationMatrix_)
# 				coeff = _conic_matrix_to_coeffs(pixel_conic)
# 				conic_coeffs.append(coeff)
# 				state_records.append(
# 					np.concatenate([
# 						satellite_positions[pos_idx],
# 						satellite_orientations[orient_idx, pos_idx].ravel(),
# 						coeff,
# 					])
# 				)

# 	coeffs_nx6 = np.stack(conic_coeffs, axis=0)
# 	if args.save_coeffs is not None:
# 		args.save_coeffs.parent.mkdir(parents=True, exist_ok=True)
# 		records_array = np.stack(state_records, axis=0)
# 		records_array[np.abs(records_array) < 0.01] = 0.0
# 		np.savetxt(
# 			args.save_coeffs.with_suffix(".csv"),
# 			records_array,
# 			delimiter=",",
# 			header="pos_x,pos_y,pos_z,R00,R01,R02,R10,R11,R12,R20,R21,R22,A,B,C,D,E,F",
# 			comments="",
# 			fmt="%.1f",
# 		)

# 	render_conic.process_simulation(
# 		coeffs_nx6=render_conic.torch.from_numpy(coeffs_nx6),
# 		width=args.x_resolution,
# 		height=args.y_resolution,
# 		output_folder=str(args.output_folder),
# 		batch_size=args.batch_size,
# 		sigma=args.sigma,
# 	)


if __name__ == "__main__":
    main()
