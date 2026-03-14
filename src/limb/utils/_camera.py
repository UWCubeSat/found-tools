"""Camera geometry utilities.

This module provides a pure-Python :class:`Camera` implementation along with
helpers for building camera rotation and intrinsics matrices. Not usning FOUND camera class
to avoid nonlinear dependencies.
"""

import numpy as np
import random


class Camera:
    """Simple pinhole camera model with public camera parameters."""

    def __init__(
        self,
        focal_length: float,
        x_pixel_pitch: float,
        x_resolution: int,
        y_resolution: int,
        x_center: float | None = None,
        y_center: float | None = None,
        y_pixel_pitch: float | None = None,
        edge_offset: float | None = None,
        edge_angle_mode: str = "randomized",
    ) -> None:
        self.focal_length = float(focal_length)
        self.x_resolution = int(x_resolution)
        self.y_resolution = int(y_resolution)
        self.x_center = self.x_resolution / 2.0 if x_center is None else float(x_center)
        self.y_center = self.y_resolution / 2.0 if y_center is None else float(y_center)
        self.x_pixel_pitch = float(x_pixel_pitch)
        self.y_pixel_pitch = (
            self.x_pixel_pitch if y_pixel_pitch is None else float(y_pixel_pitch)
        )
        self.min_image_dimension = min(
            self.x_resolution * self.x_pixel_pitch,
            self.y_resolution * self.y_pixel_pitch
        )
        self.image_max_edge_angle = self._max_edge_angle()
        self.calibration_matrix = self._calibration_matrix()
        self.inverse_calibration_matrix = np.linalg.inv(self.calibration_matrix)

    def _max_edge_angle(self):
        """Compute the maximum edge angle from the public parameters."""
        # v = np.array([self.focal_length, self.min_image_dimension, 0.0], dtype=np.float64)
        # min_image_direction = v / np.linalg.norm(v)
        return float(np.arctan(self.min_image_dimension / 2 / self.focal_length))

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

    def edge_angle(self) -> float:
        """Return an edge angle using the constructor-selected strategy."""
        if self.edge_angle_mode == "randomized":
            return random.uniform(0.0, self.image_max_edge_angle)
        return 0.0
