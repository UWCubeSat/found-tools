"""Test limb.utils.plot.edge_plot on the first sim image using sim_metadata.csv row 0."""
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

from limb.simulation.edge.conic import (
    _shape_matrix_from_axes,
    generate_camera_conic,
    generate_edge_points,
    generate_pixel_conic,
)
from limb.utils._camera import Camera
from limb.utils.plot import edge_plot

def main():
    df = pd.read_csv("sim_metadata.csv", index_col=0)
    row = df.iloc[0]

    semi_axes = [row["shape_axis_a"], row["shape_axis_b"], row["shape_axis_c"]]
    shape_matrix = _shape_matrix_from_axes(semi_axes)
    w, h = int(row["cam_x_resolution"]), int(row["cam_y_resolution"])
    camera = Camera(
        focal_length=row["cam_focal_length"],
        x_pixel_pitch=row["cam_x_pixel_pitch"],
        y_pixel_pitch=row["cam_y_pixel_pitch"],
        x_resolution=w,
        y_resolution=h,
        x_center=row["cam_x_center"],
        y_center=row["cam_y_center"],
    )
    quat = np.array([row["qx"], row["qy"], row["qz"], row["qw"]], dtype=np.float64)
    tcp = R.from_quat(quat).as_matrix()
    tpc = tcp.T  # world to camera
    sat_pos = np.array([row["true_pos_x"], row["true_pos_y"], row["true_pos_z"]], dtype=np.float64)
    rc = tpc @ sat_pos

    image_conic = generate_camera_conic(rc, shape_matrix, tpc)
    pixel_conic = generate_pixel_conic(image_conic, camera)
    points = generate_edge_points(rc, shape_matrix, tpc, camera)

    if points.size == 0:
        print("No edge points in image; cannot plot.")
        return

    # Center window on point nearest image center
    center_xy = np.array([camera.x_center, camera.y_center])
    dist = np.linalg.norm(points - center_xy, axis=1)
    center_point = int(np.argmin(dist))
    window_length = 50.0

    edge_plot(
        "sim_images/img_000000.png",
        points,
        center_point=center_point,
        window_length=window_length,
        pixel_conic=pixel_conic,
    )

if __name__ == "__main__":
    main()
