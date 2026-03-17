import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

from limb.simulation.edge.conic import (
    generate_camera_conic,
    generate_edge_points,
    generate_pixel_conic,
    _conic_matrix_to_coeffs,
    _shape_matrix_from_axes,
)
from limb.simulation.metadata.state import (
    generate_satellite_state,
    generate_uniform_directions,
)
from limb.utils._camera import Camera, focal_length_from_fov


def initialize_sim_df() -> pd.DataFrame:
    """Initialize an empty observation DataFrame with all simulation columns.

    Column groups
    -------------
    Inputs (independent variables):
        true_pos_x/y/z          Satellite position in world frame (rectangular, m).
        qx, qy, qz, qw             Optical-axis attitude as quaternion (scipy order: x, y, z, w).
        shape_axis_a/b/c         Ellipsoid semi-axes (m); diagonal entries of shape matrix.

    Camera parameters (one scalar column rper intrinsic):
        cam_focal_length         Focal length (m).
        cam_x_pixel_pitch        Pixel pitch in x (m).
        cam_y_pixel_pitch        Pixel pitch in y (m).
        cam_x_resolution         Sensor width (pixels).
        cam_y_resolution         Sensor height (pixels).
        cam_x_center             Principal point x (pixels).
        cam_y_center             Principal point y (pixels).

    Outputs (dependent variables):
        out_pos_x/y/z            Estimated satellite position in world frame (rectangular, m).

    Runtime metrics (fill when measuring the FOUND binary):
        runtime_sec             Wall-clock seconds for the run (e.g. time.perf_counter()).
        instructions            CPU instruction count (e.g. perf stat -e instructions).
        bytes_allocated         Total bytes allocated (e.g. Valgrind memcheck), optional.
        allocations             Number of alloc/free calls, optional.

    Generate (generated helper variables):
        true_x_centroid          True limb centroid x in pixel coordinates.
        true_y_centroid          True limb centroid y in pixel coordinates.
        true_r_apparent          True apparent radius (pixels).
        out_x_centroid           Estimated limb centroid x (pixels).
        out_y_centroid           Estimated limb centroid y (pixels).
        out_r_apparent           Estimated apparent radius (pixels).

    Returns
    -------
    pd.DataFrame
        Empty DataFrame with all columns. float64/int64/bool as noted; optional
        metric columns may be left NaN/empty.
    """
    columns = {
        # --- true state ---
        "true_pos_x": pd.Series(dtype="float64"),
        "true_pos_y": pd.Series(dtype="float64"),
        "true_pos_z": pd.Series(dtype="float64"),
        "qx": pd.Series(dtype="float64"),
        "qy": pd.Series(dtype="float64"),
        "qz": pd.Series(dtype="float64"),
        "qw": pd.Series(dtype="float64"),
        # --- planet model ---
        "shape_axis_a": pd.Series(dtype="float64"),
        "shape_axis_b": pd.Series(dtype="float64"),
        "shape_axis_c": pd.Series(dtype="float64"),
        # --- camera intrinsics ---
        "cam_focal_length": pd.Series(dtype="float64"),
        "cam_x_pixel_pitch": pd.Series(dtype="float64"),
        "cam_y_pixel_pitch": pd.Series(dtype="float64"),
        "cam_x_resolution": pd.Series(dtype="int64"),
        "cam_y_resolution": pd.Series(dtype="int64"),
        "cam_x_center": pd.Series(dtype="float64"),
        "cam_y_center": pd.Series(dtype="float64"),
        # --- estimated position ---
        "out_pos_x": pd.Series(dtype="float64"),
        "out_pos_y": pd.Series(dtype="float64"),
        "out_pos_z": pd.Series(dtype="float64"),
        # --- runtime metrics ---
        "runtime_sec": pd.Series(dtype="float64"),
        "instructions": pd.Series(dtype="Int64"),
        "bytes_allocated": pd.Series(dtype="Int64"),
        "allocations": pd.Series(dtype="Int64"),
        # --- true measurements ---
        "true_x_centroid": pd.Series(dtype="float64"),
        "true_y_centroid": pd.Series(dtype="float64"),
        "true_r_apparent": pd.Series(dtype="float64"),
        # --- estimated measurements ---
        "out_x_centroid": pd.Series(dtype="float64"),
        "out_y_centroid": pd.Series(dtype="float64"),
        "out_r_apparent": pd.Series(dtype="float64"),
    }
    return pd.DataFrame(columns=columns.keys())


def _fill_setup(
    df: pd.DataFrame,
    camera: Camera,
    sat_position: np.ndarray,
    semi_axes: np.ndarray,
    quat: np.ndarray,
) -> pd.DataFrame:
    """Add a new row to the simulation DataFrame with all input columns populated.

    Args:
        df: DataFrame initialised by ``initialize_sim_df``.
        camera: Camera object supplying intrinsic parameters.
        sat_position: 1-D array ``[x, y, z]`` – satellite position in world frame (m).
        semi_axes: 1-D array ``[a, b, c]`` – ellipsoid semi-axes (m).
        quat: 1-D array ``[x, y, z, w]`` – optical-axis attitude quaternion (scipy order).

    Returns:
        A new DataFrame with the additional row appended.
    """
    new_row = {
        "cam_focal_length": camera.focal_length,
        "cam_x_pixel_pitch": camera.x_pixel_pitch,
        "cam_y_pixel_pitch": camera.y_pixel_pitch,
        "cam_x_resolution": camera.x_resolution,
        "cam_y_resolution": camera.y_resolution,
        "cam_x_center": camera.x_center,
        "cam_y_center": camera.y_center,
        "true_pos_x": sat_position[0],
        "true_pos_y": sat_position[1],
        "true_pos_z": sat_position[2],
        "shape_axis_a": semi_axes[0],
        "shape_axis_b": semi_axes[1],
        "shape_axis_c": semi_axes[2],
        "qx": quat[0],
        "qy": quat[1],
        "qz": quat[2],
        "qw": quat[3],
    }
    return pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)


def _row_to_pose(row: pd.Series) -> tuple[Camera, np.ndarray, np.ndarray, np.ndarray]:
    """Parse a simulation metadata row into camera, shape matrix, TPC, and position in camera frame.

    Row must contain: shape_axis_a/b/c, cam_*, qx,qy,qz,qw, true_pos_x/y/z.

    Returns
    -------
    camera : Camera
        Camera built from row intrinsics.
    shape_matrix : np.ndarray
        (3, 3) ellipsoid shape matrix.
    tpc : np.ndarray
        (3, 3) rotation from world to camera frame.
    rc : np.ndarray
        (3,) satellite position in camera coordinates.
    """
    camera = Camera.from_row(row)
    semi_axes = [row["shape_axis_a"], row["shape_axis_b"], row["shape_axis_c"]]
    shape_matrix = _shape_matrix_from_axes(semi_axes)
    quat = np.array(
        [row["qx"], row["qy"], row["qz"], row["qw"]],
        dtype=np.float64,
    )
    tpc = R.from_quat(quat).as_matrix().T
    sat_pos = np.array(
        [row["true_pos_x"], row["true_pos_y"], row["true_pos_z"]],
        dtype=np.float64,
    )
    rc = tpc @ sat_pos
    return camera, shape_matrix, tpc, rc


def _conic_from_row(row: pd.Series) -> np.ndarray:
    """Compute conic coefficients (a,b,c,d,e,f) from a single simulation metadata row.

    Row must contain: shape_axis_a/b/c, cam_*, qx,qy,qz,qw, true_pos_x/y/z.
    Returns shape (6,) float64.
    """
    camera, shape_matrix, tpc, rc = _row_to_pose(row)
    image_conic = generate_camera_conic(rc, shape_matrix, tpc)
    return generate_pixel_conic(image_conic, camera)


def points_from_row(
    row: pd.Series,
    *,
    gaussian_sigma: float | tuple[float, float] | None = None,
    n_false_points: int = 0,
    truncate: int = 1,
) -> np.ndarray:
    """Compute edge points in pixel coordinates from a simulation metadata row.

    Args:
        row: One row from the simulation DataFrame (shape, pose, camera, etc.).
        gaussian_sigma: Optional point noise sigma in pixels. Float for isotropic,
            or (sigma_x, sigma_y). Passed to generate_edge_points.

    Returns:
        (N, 2) array of (x, y) pixel coordinates on the limb edge.
    """
    camera, shape_matrix, tpc, rc = _row_to_pose(row)
    return generate_edge_points(
        rc,
        shape_matrix,
        tpc,
        camera,
        gaussian_sigma=gaussian_sigma,
        n_false_points=n_false_points,
        truncate=truncate,
    )


def _calculate_conic_coeffs(
    df_or_path: pd.DataFrame | str | Path,
) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Compute conic coefficients from simulation metadata (DataFrame or CSV path).

    Returns one (row_indices, coeffs, centroid) pair per (width, height) so the caller can
    render with filenames that match the DataFrame index.

    Returns
    -------
    dict[tuple[int, int], tuple[np.ndarray, np.ndarray, tuple[float32, float32]]]
        Keys are (cam_x_resolution, cam_y_resolution). Values are
        (row_indices, coeffs, centroid): row_indices 1d int (df row index per image),
        coeffs shape (M, 6) float32. cal_mat
        Keys are sorted for deterministic iteration.
    """
    if isinstance(df_or_path, (str, Path)):
        df = pd.read_csv(df_or_path, index_col=0)
    else:
        df = df_or_path

    out: dict[
        tuple[int, int],
        tuple[list[int], list[np.ndarray], list[np.ndarray], list[np.ndarray]],
    ] = {}

    for idx, row in df.iterrows():
        camera, _, _, rc = _row_to_pose(row)
        conic = _conic_from_row(row)
        coeffs = _conic_matrix_to_coeffs(conic)

        key = (camera.x_resolution, camera.y_resolution)
        if key not in out:
            out[key] = ([], [], [], [])
        out[key][0].append(idx)
        out[key][1].append(coeffs)
        out[key][2].append(camera.inverse_calibration_matrix)
        out[key][3].append(rc)

    return {
        k: (
            np.array(idxs, dtype=np.int64),
            np.array(v, dtype=np.float32),
            np.array(c, dtype=np.float32),
            np.array(r, dtype=np.float32),
        )
        for k, (idxs, v, c, r) in sorted(out.items())
    }


def setup_expirement(
    df: pd.DataFrame,
    semi_axes,
    earth_point_directions,
    distance,
    camera,
    num_satellite_positions,
    num_image_spins,
    num_image_radials,
) -> pd.DataFrame:
    """Append rows to the simulation DataFrame for one experiment (one camera, one distance)."""
    max_axis = max(semi_axes)
    assert distance > max_axis, (
        f"distance {distance} must be greater than all semi_axes (max {max_axis})"
    )
    shape_matrix = _shape_matrix_from_axes(semi_axes)

    sat_positions, tcps = generate_satellite_state(
        shape_matrix,
        earth_point_directions,
        distance,
        camera,
        num_satellite_positions,
        num_image_spins,
        num_image_radials,
    )

    for tcp, sat_pos in zip(tcps, sat_positions):
        quat = R.from_matrix(tcp).as_quat()  # scipy order: x, y, z, w
        df = _fill_setup(
            df=df,
            camera=camera,
            sat_position=sat_pos,
            semi_axes=semi_axes,
            quat=quat,
        )

    return df


def _setup_simulation(
    semi_axes: list,
    fovs: list,
    resolutions: list,
    distances: list,
    num_earth_points: int,
    num_positions_per_point: int,
    num_spins_per_position: int,
    num_radials_per_spin: int,
) -> pd.DataFrame:
    """Build the simulation DataFrame (no I/O). Caller may write CSV or use in memory."""
    PIXEL_PITCH = 5e-6  # doesn't affect simulation as long as consistent

    expirements = itertools.product(fovs, resolutions, distances)
    df = initialize_sim_df()
    earth_directions = generate_uniform_directions(num_earth_points)

    for exp in expirements:
        focal_length = focal_length_from_fov(
            fov=exp[0], resolution=exp[1], pixel_pitch=PIXEL_PITCH
        )
        camera = Camera(
            focal_length=focal_length,
            x_pixel_pitch=PIXEL_PITCH,
            x_resolution=exp[1],
            y_resolution=exp[1],
        )
        df = setup_expirement(
            df,
            semi_axes,
            earth_directions,
            exp[2],
            camera,
            num_positions_per_point,
            num_spins_per_position,
            num_radials_per_spin,
        )

    return df


def setup_simulation(
    semi_axes: list,
    fovs: list,
    resolutions: list,
    distances: list,
    num_earth_points: int,
    num_positions_per_point: int,
    num_spins_per_position: int,
    num_radials_per_spin: int,
    output_path: str,
) -> None:
    """Run simulation grid, write metadata to CSV. Returns nothing."""
    df = _setup_simulation(
        semi_axes,
        fovs,
        resolutions,
        distances,
        num_earth_points,
        num_positions_per_point,
        num_spins_per_position,
        num_radials_per_spin,
    )
    df.to_csv(output_path, index=True)
