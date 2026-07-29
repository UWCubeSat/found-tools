"""Camera geometry utilities.

This module provides a pure-Python :class:`Camera` implementation along with
helpers for building camera rotation and intrinsics matrices. Not usning FOUND camera class
to avoid nonlinear dependencies.
"""

import numpy as np
import pandas as pd


class Camera:
    """Simple pinhole camera model with public camera parameters."""

    @classmethod
    def from_row(cls, row: pd.Series) -> Camera:  # pragma: no cover
        """Build a Camera from a single row of the simulation/orchestrate DataFrame.

        Row must contain: cam_focal_length, cam_x_pixel_pitch, cam_y_pixel_pitch,
        cam_x_resolution, cam_y_resolution, cam_x_center, cam_y_center.
        """
        return cls(
            focal_length=float(row["cam_focal_length"]),
            x_pixel_pitch=float(row["cam_x_pixel_pitch"]),
            y_pixel_pitch=float(row["cam_y_pixel_pitch"]),
            x_resolution=int(row["cam_x_resolution"]),
            y_resolution=int(row["cam_y_resolution"]),
            x_center=float(row["cam_x_center"]),
            y_center=float(row["cam_y_center"]),
        )

    def __init__(
        self,
        focal_length: float,
        x_pixel_pitch: float,
        x_resolution: int,
        y_resolution: int | None = None,
        x_center: float | None = None,
        y_center: float | None = None,
        y_pixel_pitch: float | None = None,
    ) -> None:
        self.focal_length = float(focal_length)
        self.x_resolution = int(x_resolution)
        self.y_resolution = (
            int(y_resolution) if y_resolution is not None else int(x_resolution)
        )
        self.x_center = self.x_resolution / 2.0 if x_center is None else float(x_center)
        self.y_center = self.y_resolution / 2.0 if y_center is None else float(y_center)
        self.x_pixel_pitch = float(x_pixel_pitch)
        self.y_pixel_pitch = (
            self.x_pixel_pitch if y_pixel_pitch is None else float(y_pixel_pitch)
        )
        self.calibration_matrix = self._calibration_matrix()
        self.inverse_calibration_matrix = np.linalg.inv(self.calibration_matrix)

    def _calibration_matrix(self):
        """Compute the calibration matrix from the public parameters."""
        fx = self.focal_length / self.x_pixel_pitch
        fy = self.focal_length / self.y_pixel_pitch
        return np.array(
            [
                [self.x_center, -fx, 0.0],
                [self.y_center, 0.0, -fy],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

    def min_image_dimension(self) -> float:
        """Compute the minimum image dimension in metres."""
        return min(
            self.x_resolution * self.x_pixel_pitch,
            self.y_resolution * self.y_pixel_pitch,
        )

    def max_edge_angle(self):  # pragma: no cover
        """Compute the maximum angle and object can be offset from the optical axis
        and still be garaunteed to be captured."""
        return float(np.arctan(self.min_image_dimension() / 2 / self.focal_length))

    def camera_to_pixel(self, vector_camera: np.ndarray) -> np.ndarray:
        """Project a 3D vector in camera frame to pixel coordinates.

        Camera frame: x is optical axis (depth), y and z are image plane.
        Uses similar triangles for camera→image, then calibration matrix for image→pixel.
        Cannot handle points behind the camera (vector_camera[0] must be > 0).

        Parameters
        ----------
        vector_camera : np.ndarray
            Shape (3,) – position or direction in camera frame (m or arbitrary scale).

        Returns
        -------
        np.ndarray
            Shape (2,) – (pixel_x, pixel_y).

        Raises
        ------
        AssertionError
            If vector_camera[0] <= 0 (point behind or on camera plane).
        """
        assert vector_camera[0] > 0, "Cannot project points behind the camera"
        x, y, z = vector_camera[0], vector_camera[1], vector_camera[2]
        homogenous_image = np.array([1.0, y / x, z / x], dtype=np.float64)
        homogenous_pixel = self.calibration_matrix @ homogenous_image
        return homogenous_pixel[:2]


def focal_length_from_fov(
    fov: float, resolution: int, pixel_pitch: float
) -> float:  # pragma: no cover
    """Compute focal length from field of view.

    Args:
        fov: Full field of view in degrees.
        pixel_pitch: Pixel pitch (m).
        resolution: Sensor resolution in pixels along the same axis as *fov*.

    Returns:
        Focal length in metres.
    """
    sensor_size = resolution * pixel_pitch
    return sensor_size / (2.0 * np.tan(np.deg2rad(fov) / 2.0))
