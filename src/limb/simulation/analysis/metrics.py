"""True metrics from simulation metadata: centroid and apparent radius in pixel space."""

import numpy as np
import pandas as pd

from limb.simulation.metadata.orchestrate import _row_to_pose
from limb.utils._camera import Camera


def apparent_radius_pixels(
    position_camera: np.ndarray,
    radius: float,
    camera: Camera,
) -> float:
    """Compute apparent limb radius in pixels (sphere on optical axis).

    For a sphere: apparent angular semi-diameter is arcsin(R/d); in the image
    plane (focal length f) the semi-diameter is f*tan(arcsin(R/d)) = f*R/sqrt(d²-R²);
    in pixels that is (f/p)*R/sqrt(d²-R²) with p the pixel pitch. When f=p=1 this
    equals R/sqrt(d²-R²), which approximates R/d for small R/d.

    Parameters
    ----------
    position_camera : np.ndarray
        Shape (3,) – satellite/body center in camera frame (m). Optical axis is x.
    radius : float
        Body semi-axis (m) used for apparent size; e.g. shape_axis_a for a sphere.
    calibration_matrix : np.ndarray
        Shape (3, 3) – camera calibration matrix; fx = -cal[0,1] gives image→pixel scale.

    Returns
    -------
    float
        Apparent radius in pixels.
    """
    # Ensure scalars so comparisons and division are unambiguous for downstream
    position_camera = np.asarray(position_camera, dtype=np.float64).ravel()
    if position_camera.size != 3:
        raise ValueError("position_camera must have exactly 3 elements")
    distance = float(np.linalg.norm(position_camera))
    radius = float(radius)
    if distance <= radius:
        return np.nan
    # Image-plane semi-diameter = R/sqrt(d²-R²); pixel scale from calibration
    fx = camera.focal_length / camera.x_pixel_pitch
    return float(radius / distance * fx)


def fill_pixel_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Fill true_* and out_* centroid/apparent-radius columns from metadata.

    For each row: transform satellite position (true and optionally out) to
    camera frame, project to pixel coordinates for the centroid, and compute
    apparent radius. out_x_centroid, out_y_centroid, out_r_apparent, and
    position_distance_error_m are filled only when out_pos_x, out_pos_y, out_pos_z
    are all non-NaN. position_distance_error_m = |‖true_pos‖ - ‖out_pos‖| (m).

    Parameters
    ----------
    df : pd.DataFrame
        Simulation DataFrame from orchestrate (initialize_sim_df / setup_expirement).
        Must contain pose, camera intrinsics, and shape columns.

    Returns
    -------
    pd.DataFrame
        Same DataFrame with true_x_centroid, true_y_centroid, true_r_apparent
        and (when out_pos_* present) out_x_centroid, out_y_centroid, out_r_apparent,
        position_distance_error_m filled.
    """

    if "position_distance_error_m" not in df.columns:
        df["position_distance_error_m"] = np.nan

    for idx, row in df.iterrows():
            camera, _, tpc, rc = _row_to_pose(row)
            radius = float(row["shape_axis_a"])
            camera_to_earth_origing = np.array([abs(rc[0]), -rc[1], -rc[2]], dtype=np.float64)
            # Calculate true metrics
            px, py = camera.camera_to_pixel(camera_to_earth_origing)
            df.at[idx, "true_x_centroid"] = px
            df.at[idx, "true_y_centroid"] = py
            df.at[idx, "true_r_apparent"] = apparent_radius_pixels(rc, radius, camera)

            # Check for 'out' position columns (use float() so values are plain scalars)
            out_x, out_y, out_z = row.get("out_pos_x"), row.get("out_pos_y"), row.get("out_pos_z")
            if pd.notna(out_x) and pd.notna(out_y) and pd.notna(out_z):
                out_vec = np.array(
                    [float(out_x), float(out_y), float(out_z)],
                    dtype=np.float64,
                )
                rc_out = np.ravel(np.asarray(tpc @ out_vec, dtype=np.float64))
                camera_to_earth_origing_out = np.array(
                    [abs(rc_out[0]), -rc_out[1], -rc_out[2]],
                    dtype=np.float64,
                )
                ox, oy = camera.camera_to_pixel(camera_to_earth_origing_out)
                df.at[idx, "out_x_centroid"] = float(ox)
                df.at[idx, "out_y_centroid"] = float(oy)
                df.at[idx, "out_r_apparent"] = apparent_radius_pixels(
                    rc_out, radius, camera
                )
                # Absolute distance error (m): |‖true_pos‖ - ‖out_pos‖|
                true_pos = np.array(
                    [float(row["true_pos_x"]), float(row["true_pos_y"]), float(row["true_pos_z"])],
                    dtype=np.float64,
                )
                df.at[idx, "position_distance_error_m"] = float(
                    np.abs(np.linalg.norm(true_pos) - np.linalg.norm(out_vec))
                )

    return df
