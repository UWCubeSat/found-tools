import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

from limb.simulation.edge.conic import generate_camera_conic, generate_pixel_conic
from limb.simulation.metadata.state import generate_satellite_state
from limb.utils._camera import Camera


def initialize_sim_df() -> pd.DataFrame:
    """Initialize an empty observation DataFrame with all simulation columns.

    Column groups
    -------------
    Inputs (independent variables):
        true_pos_x/y/z          Satellite position in world frame (rectangular, m).
        true_attitude_ra/dec/roll  Optical-axis direction and roll (degrees).
        shape_axis_a/b/c         Ellipsoid semi-axes (m); diagonal entries of shape matrix.
        atmosphere_blur          Gaussian blur sigma applied to the limb edge (pixels).

    Camera parameters (one scalar column per intrinsic):
        cam_focal_length         Focal length (m).
        cam_x_pixel_pitch        Pixel pitch in x (m).
        cam_y_pixel_pitch        Pixel pitch in y (m).
        cam_x_resolution         Sensor width (pixels).
        cam_y_resolution         Sensor height (pixels).
        cam_x_center             Principal point x (pixels).
        cam_y_center             Principal point y (pixels).

    Outputs (dependent variables):
        out_pos_x/y/z            Estimated satellite position in world frame (rectangular, m).

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
        Empty DataFrame with all columns typed as float64 except
        ``cam_x_resolution``, ``cam_y_resolution`` (int64) and
        ``cam_edge_angle_mode`` (object/str).
    """
    columns = {
        # --- true state ---
        "true_pos_x": pd.Series(dtype="float64"),
        "true_pos_y": pd.Series(dtype="float64"),
        "true_pos_z": pd.Series(dtype="float64"),
        "true_attitude_ra": pd.Series(dtype="float64"),
        "true_attitude_dec": pd.Series(dtype="float64"),
        "true_attitude_roll": pd.Series(dtype="float64"),
        # --- planet model ---
        "shape_axis_a": pd.Series(dtype="float64"),
        "shape_axis_b": pd.Series(dtype="float64"),
        "shape_axis_c": pd.Series(dtype="float64"),
        # --- atmosphere ---
        "atmosphere_blur": pd.Series(dtype="float64"),
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
        euler_angles: np.ndarray) -> pd.DataFrame:
    """Add a new row to the simulation DataFrame with all input columns populated.

    Args:
        df: DataFrame initialised by ``_initialize_sim_df``.
        camera: Camera object supplying intrinsic parameters.
        sat_position: 1-D array ``[x, y, z]`` – satellite position in world frame (m).
        semi_axes: 1-D array ``[a, b, c]`` – ellipsoid semi-axes (m).
        tcp: 1-D array ``[ra, dec, roll]`` – optical-axis attitude (degrees).

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
        "true_attitude_ra": euler_angles[0],
        "true_attitude_dec": euler_angles[1],
        "true_attitude_roll": euler_angles[2],
    }
    return pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

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

def setup_expirement(
        df: pd.DataFrame,
        semi_axes,
        earth_point_directions,
        distance,
        camera,
        num_satellite_positions,
        num_image_spins,
        num_image_radials) -> pd.DataFrame:

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

    conic_coeffs = []
    for tcp, sat_pos in zip(tcps, sat_positions):
        df = _fill_setup(
            df=df,
            camera=camera,
            sat_position=sat_pos,
            semi_axes=semi_axes,
            euler_angles = R.from_matrix(tcp).as_euler("zyx", degrees=True)
        )

        tpc = tcp.T  # Convert to world-to-camera rotation matrices.
        sph_pos = tpc @ sat_pos.T

        image_conic = generate_camera_conic(sph_pos, shape_matrix, tpc)
        pixel_conic = generate_pixel_conic(image_conic, camera)
        conic_coeffs.append(_conic_matrix_to_coeffs(pixel_conic))

    return df, conic_coeffs

