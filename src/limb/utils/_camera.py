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
        focalLength: float,
        xPixelPitch: float,
        xResolution: int,
        yResolution: int,
        xCenter: float | None = None,
        yCenter: float | None = None,
        yPixelPitch: float | None = None,
        edgeOffset: float | None = None,
        edgeAngleMode: str = "randomized",
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
        self.minImageDimension_ = min(
            self.xResolution_ * self.xPixelPitch_,
            self.yResolution_ * self.yPixelPitch_,
        )
        # Convert edge offset from pixels to physical units using minimum pixel pitch.
        min_pixel_pitch = min(self.xPixelPitch_, self.yPixelPitch_)
        if edgeOffset is None:
            self.edgeOffset_ = 0.1 * self.minImageDimension_
        else:
            self.edgeOffset_ = float(edgeOffset) * min_pixel_pitch
        if self.edgeOffset_ < 0.0 or self.edgeOffset_ >= self.minImageDimension_:
            raise ValueError(
                "edgeOffset must satisfy 0 <= edgeOffset < min image dimension"
            )
        self.imagePoint_ = np.array(
            [self.focalLength_, (self.minImageDimension_ - self.edgeOffset_) / 2.0, 0.0],
            dtype=np.float64,
        )
        self.imageMaxEdgeAngle_ = float(
            np.arccos(
                np.dot(np.array([1.0, 0.0, 0.0]), self.imagePoint_)
                / np.linalg.norm(self.imagePoint_)
            )
        )
        self.edgeAngleMode_ = edgeAngleMode
        if self.edgeAngleMode_ == "randomized":
            self.edgeAngleFunction_ = self.randomizedEdgeAngle
        elif self.edgeAngleMode_ == "zero":
            self.edgeAngleFunction_ = self.zeroEdgeAngle
        else:
            raise ValueError("edgeAngleMode must be one of: randomized, zero")
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
    
    def edgeAngle(self) -> float:
        """Return an edge angle using the constructor-selected strategy."""
        return self.edgeAngleFunction_()

    def randomizedEdgeAngle(self) -> float:
        """Sample a random edge angle constrained by the camera geometry."""
        return random.uniform(0.0, self.imageMaxEdgeAngle_)
    
    def zeroEdgeAngle(self) -> float:
        """Return the edge angle corresponding to the optical axis."""
        return 0.0
