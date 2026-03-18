"""True metrics from simulation metadata: centroid and apparent radius in pixel space."""

import numpy as np
import pandas as pd
from scipy import stats

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
    apparent radius. out_x_centroid, out_y_centroid, out_r_apparent,
    position_distance_error_m, and delta columns are filled only when
    out_pos_x, out_pos_y, out_pos_z are all non-NaN.
    position_distance_error_m = |‖true_pos‖ - ‖out_pos‖| (m).
    Deltas are signed (out - true): delta_x_centroid, delta_y_centroid,
    delta_r_apparent (not absolute values).

    Parameters
    ----------
    df : pd.DataFrame
        Simulation DataFrame from orchestrate (initialize_sim_df / setup_expirement).
        Must contain pose, camera intrinsics, and shape columns.

    Returns
    -------
    pd.DataFrame
        Same DataFrame with true_x_centroid, true_y_centroid, true_r_apparent;
        (when out_pos_* present) out_x_centroid, out_y_centroid, out_r_apparent,
        position_distance_error_m, delta_x_centroid, delta_y_centroid,
        delta_r_apparent filled.
    """

    if "position_distance_error_m" not in df.columns:
        df["position_distance_error_m"] = np.nan
    for col in ("delta_x_centroid", "delta_y_centroid", "delta_r_apparent"):
        if col not in df.columns:
            df[col] = np.nan

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
                out_r = apparent_radius_pixels(rc_out, radius, camera)
                df.at[idx, "out_r_apparent"] = out_r
                # Signed deltas (out - true), not absolute
                df.at[idx, "delta_x_centroid"] = float(ox) - px
                df.at[idx, "delta_y_centroid"] = float(oy) - py
                df.at[idx, "delta_r_apparent"] = float(out_r) - df.at[idx, "true_r_apparent"]
                # Absolute distance error (m): |‖true_pos‖ - ‖out_pos‖|
                true_pos = np.array(
                    [float(row["true_pos_x"]), float(row["true_pos_y"]), float(row["true_pos_z"])],
                    dtype=np.float64,
                )
                df.at[idx, "position_distance_error_m"] = float(
                    np.abs(np.linalg.norm(true_pos) - np.linalg.norm(out_vec))
                )

    return df


def _single_clump_stats(
    data: np.ndarray,
    confidence: float,
) -> dict[str, float | int]:
    """Mean, std, and prediction interval for one array. Used by column_summary."""
    n = data.size
    if n == 0:
        return {"mean": np.nan, "std": np.nan, "pi_lower": np.nan, "pi_upper": np.nan, "n": 0}
    mean = float(np.mean(data))
    std = float(np.std(data, ddof=1))
    if n == 1 or std == 0:
        pi_lower = pi_upper = mean
    else:
        alpha = 1 - confidence
        t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
        half_width = t_crit * std * np.sqrt(1 + 1 / n)
        pi_lower = float(mean - half_width)
        pi_upper = float(mean + half_width)
    return {"mean": mean, "std": std, "pi_lower": pi_lower, "pi_upper": pi_upper, "n": n}


def column_summary(
    df: pd.DataFrame,
    column: str,
    confidence: float = 0.95,
    dropna: bool = True,
    n_bins: int = 10,
    distance_column: str | None = None,
    print_results: bool = True,
) -> list[dict[str, float | int]]:
    """Compute mean, std, and prediction interval per distance clump, then print.

    Groups rows into distance bins with edges from the minimum to maximum distance
    (equal-width), so there is always a bin at the least and greatest distance.
    For each clump, computes mean, standard deviation, and
    prediction interval for the given column, then prints a table and returns the
    per-clump stats.

    Distance is taken from ``distance_column`` if provided; otherwise from
    ‖(true_pos_x, true_pos_y, true_pos_z)‖ (requires those columns).

    Uses a prediction interval (for a single new observation), not a confidence
    interval (for the mean). Formula: mean ± t * s * sqrt(1 + 1/n); t-based for
    small n.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the column and distance (or true_pos_*).
    column : str
        Name of the numeric column to summarize.
    confidence : float, optional
        Confidence level for the prediction interval (e.g. 0.95 for 95%).
        Default is 0.95.
    dropna : bool, optional
        If True (default), drop NaN in the target column before grouping/stats.
    n_bins : int, optional
        Number of distance bins (equal-width from min to max distance). Default is 10.
    distance_column : str or None, optional
        Column to use as distance for binning. If None, distance is computed as
        norm of (true_pos_x, true_pos_y, true_pos_z).
    print_results : bool, optional
        If True (default), print a table of stats per clump.

    Returns
    -------
    list[dict[str, float | int]]
        One dict per clump with keys "distance_lo", "distance_hi", "mean", "std",
        "pi_lower", "pi_upper", "n".
    """
    if column not in df.columns:
        raise ValueError(f"Column {column!r} not in DataFrame")

    work = df[[column]].copy()
    if dropna:
        work = work.dropna(subset=[column])
    if work.empty:
        raise ValueError(f"Column {column!r} has no valid (non-NaN) values")

    # Resolve distance: explicit column or norm of true_pos_*
    if distance_column is not None:
        if distance_column not in df.columns:
            raise ValueError(f"Distance column {distance_column!r} not in DataFrame")
        work["_distance"] = df.loc[work.index, distance_column].values
    else:
        for c in ("true_pos_x", "true_pos_y", "true_pos_z"):
            if c not in df.columns:
                raise ValueError(
                    "Either provide distance_column or ensure true_pos_x, true_pos_y, true_pos_z exist"
                )
        pos = df.loc[work.index, ["true_pos_x", "true_pos_y", "true_pos_z"]].values
        work["_distance"] = np.linalg.norm(pos, axis=1)

    work = work.dropna(subset=["_distance"])
    if work.empty:
        raise ValueError("No rows with valid distance for binning")

    # Bin by distance so first bin includes min and last bin includes max (no edge drop)
    n_bins = max(1, int(n_bins))
    d_min = float(work["_distance"].min())
    d_max = float(work["_distance"].max())
    step = (d_max - d_min) / n_bins if d_max > d_min else 1.0
    # Assign bin 0..n_bins-1; clip so max value (d_max) lands in last bin
    work["_bin"] = np.clip(
        np.floor((work["_distance"].values - d_min) / step).astype(int),
        0,
        n_bins - 1,
    )
    bin_ids = sorted(work["_bin"].unique())

    results: list[dict[str, float | int]] = []
    for b in bin_ids:
        subset = work[work["_bin"] == b]
        dist = subset["_distance"]
        distance_lo = float(dist.min())
        distance_hi = float(dist.max())
        data = np.asarray(subset[column], dtype=np.float64)
        stats_dict = _single_clump_stats(data, confidence)
        stats_dict["distance_lo"] = distance_lo
        stats_dict["distance_hi"] = distance_hi
        results.append(stats_dict)

    # Ensure first bin starts at min and last bin ends at max (for plotting/coverage)
    if results:
        results[0]["distance_lo"] = d_min
        results[-1]["distance_hi"] = d_max

    if print_results:
        _print_column_summary_table(column, results, confidence)

    return results


def _print_column_summary_table(
    column: str,
    results: list[dict[str, float | int]],
    confidence: float,
) -> None:
    """Print a simple table of per-clump stats."""
    pct = int(round(confidence * 100))
    print(f"Column: {column}  ({pct}% prediction interval)")
    print("-" * 72)
    print(f"{'distance range (m)':<28} {'mean':>10} {'std':>10} {'n':>6}")
    print("-" * 72)
    for r in results:
        lo, hi = r["distance_lo"], r["distance_hi"]
        label = f"[{lo:.2g}, {hi:.2g}]"
        print(f"{label:<28} {r['mean']:>10.4g} {r['std']:>10.4g} {r['n']:>6}")
    print("-" * 72)
    print(f"{'PI bounds':<28} {'pi_lower':>10} {'pi_upper':>10}")
    for r in results:
        lo, hi = r["distance_lo"], r["distance_hi"]
        label = f"[{lo:.2g}, {hi:.2g}]"
        print(f"{label:<28} {r['pi_lower']:>10.4g} {r['pi_upper']:>10.4g}")


# Columns that uniquely identify camera configuration (focal → FOV; resolution)
_CAMERA_KEY_COLS = ("cam_focal_length", "cam_x_resolution", "cam_y_resolution")


def split_df_by_camera(
    df: pd.DataFrame,
) -> dict[tuple[float, int, int], pd.DataFrame]:
    """Split a simulation DataFrame into one DataFrame per camera configuration.

    Groups by camera type (focal length), resolution (x, y pixels), and effective
    FOV (determined by focal length and resolution). Each unique combination
    yields one sub-DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Simulation DataFrame with columns cam_focal_length, cam_x_resolution,
        cam_y_resolution.

    Returns
    -------
    dict[tuple[float, int, int], pd.DataFrame]
        Keys are (cam_focal_length, cam_x_resolution, cam_y_resolution).
        Values are DataFrames containing only rows with that camera configuration.
    """
    for col in _CAMERA_KEY_COLS:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain column {col!r}")
    out: dict[tuple[float, int, int], pd.DataFrame] = {}
    for key, group in df.groupby(
        by=[*_CAMERA_KEY_COLS],
        sort=False,
    ):
        # key is (focal_length, x_res, y_res) from the grouped columns
        out[(float(key[0]), int(key[1]), int(key[2]))] = group.reset_index(drop=True)
    return out
