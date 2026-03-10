"""Camera geometry utilities.

This module provides a pure-Python :class:`Camera` implementation along with
helpers for building camera rotation and intrinsics matrices.
"""

import numpy as np


class Camera:
    """Simple pinhole camera model with public camera parameters."""

    def __init__(
        self,
        focalLength: float,
        xPixelPitch: float,
        xResolution: int,
        yResolution: int,
        xCenter: float | None = None,
        yCenter: float | None = None,
        yPixelPitch: float | None = None,
    ) -> None:
        self.focalLength_ = float(focalLength)
        self.xResolution_ = int(xResolution)
        self.yResolution_ = int(yResolution)
        self.xCenter_ = self.xResolution_ / 2.0 if xCenter is None else float(xCenter)
        self.yCenter_ = self.yResolution_ / 2.0 if yCenter is None else float(yCenter)
        self.xPixelPitch_ = float(xPixelPitch)
        self.yPixelPitch_ = (
            self.xPixelPitch_ if yPixelPitch is None else float(yPixelPitch)
        )
        self.calibrationMatrix_ = self.initCalibrationMatrix()
        self.inverseCalibrationMatrix_ = np.linalg.inv(self.calibrationMatrix_)

    def initCalibrationMatrix(self) -> np.ndarray:
        """Compute and store the calibration matrix from the public parameters."""
        fx = self.focalLength_ / self.xPixelPitch_
        fy = self.focalLength_ / self.yPixelPitch_
        self.calibrationMatrix_ = np.array(
            [
                [self.xCenter_, -fx, 0.0],
                [self.yCenter_, 0.0, -fy],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        return self.calibrationMatrix_
