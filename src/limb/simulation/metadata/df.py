
import numpy as np
import pandas as pd

from limb.utils._camera import Camera


def _initialize_sim_df(size: int) -> pd.DataFrame:
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
    return pd.DataFrame(index = np.arange(size), columns=columns.keys())

def _fill_cam_columns(df: pd.DataFrame, camera: Camera) -> pd.DataFrame:
    """Fill camera parameter columns in the simulation DataFrame.

    Args:
        df: DataFrame with camera parameter columns.
        camera: Camera object with focal length, resolution, and edge-offset properties.

    Returns:
        The same DataFrame with camera columns filled in-place.
    """
    df["cam_focal_length"]    = camera.focal_length
    df["cam_x_pixel_pitch"]   = camera.x_pixel_pitch
    df["cam_y_pixel_pitch"]   = camera.y_pixel_pitch
    df["cam_x_resolution"]    = camera.x_resolution
    df["cam_y_resolution"]    = camera.y_resolution
    df["cam_x_center"]        = camera.x_center
    df["cam_y_center"]        = camera.y_center

    return df