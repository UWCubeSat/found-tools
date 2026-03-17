"""True metrics from simulation metadata: centroid and apparent radius in pixel space."""

import numpy as np
import pandas as pd

from limb.simulation.metadata.orchestrate import _row_to_pose


def apparent_radius_pixels(
    position_camera: np.ndarray,
    radius: float,
    calibration_matrix: np.ndarray,
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
    shape_matrix : np.ndarray
        Shape (3, 3) – ellipsoid shape matrix (diag 1/a², 1/b², 1/c²). For sphere, R=1/√S[0,0].
    calibration_matrix : np.ndarray
        Shape (3, 3) – camera calibration matrix; fx = -cal[0,1] gives image→pixel scale.

    Returns
    -------
    float
        Apparent radius in pixels.
    """
    d = np.linalg.norm(position_camera)
    if d <= radius:
        return np.nan
    # Image-plane semi-diameter = R/sqrt(d²-R²); pixel scale from calibration
    fx = -calibration_matrix[0, 1]
    image_plane_radius = radius / np.sqrt(d * d - radius * radius)
    return float(fx * image_plane_radius)


def fill_pixel_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Fill true_* and out_* centroid/apparent-radius columns from metadata.

    For each row: transform satellite position (true and optionally out) to
    camera frame, project to pixel coordinates for the centroid, and compute
    apparent radius (placeholder). out_x_centroid, out_y_centroid, out_r_apparent
    are filled only when out_pos_x, out_pos_y, out_pos_z are all non-NaN.

    Parameters
    ----------
    df : pd.DataFrame
        Simulation DataFrame from orchestrate (initialize_sim_df / setup_expirement).
        Must contain pose, camera intrinsics, and shape columns.

    Returns
    -------
    pd.DataFrame
        Same DataFrame with true_x_centroid, true_y_centroid, true_r_apparent
        and (when out_pos_* present) out_x_centroid, out_y_centroid, out_r_apparent filled.
    """
    out = df.copy()
    n = len(out)
    true_x = np.full(n, np.nan, dtype=np.float64)
    true_y = np.full(n, np.nan, dtype=np.float64)
    true_r = np.full(n, np.nan, dtype=np.float64)
    out_x = np.full(n, np.nan, dtype=np.float64)
    out_y = np.full(n, np.nan, dtype=np.float64)
    out_r = np.full(n, np.nan, dtype=np.float64)
    radius = df["shape_axis_a"]

    for i, (_, row) in enumerate(out.iterrows()):
        camera, shape_matrix, tpc, rc = _row_to_pose(row)
        if rc[0] > 0:
            px, py = camera.camera_to_pixel(rc)
            true_x[i] = px
            true_y[i] = py
            true_r[i] = apparent_radius_pixels(
                rc, shape_matrix, camera.calibration_matrix
            )

        out_pos_x = row.get("out_pos_x")
        out_pos_y = row.get("out_pos_y")
        out_pos_z = row.get("out_pos_z")
        if pd.notna(out_pos_x) and pd.notna(out_pos_y) and pd.notna(out_pos_z):
            rc_out = tpc @ np.array([out_pos_x, out_pos_y, out_pos_z], dtype=np.float64)
            if rc_out[0] > 0:
                ox, oy = camera.camera_to_pixel(rc_out)
                out_x[i] = ox
                out_y[i] = oy
                out_r[i] = apparent_radius_pixels(
                    rc_out, radius, camera.calibration_matrix
                )

    out["true_x_centroid"] = true_x
    out["true_y_centroid"] = true_y
    out["true_r_apparent"] = true_r
    out["out_x_centroid"] = out_x
    out["out_y_centroid"] = out_y
    out["out_r_apparent"] = out_r
    return out
