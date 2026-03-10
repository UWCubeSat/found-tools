"""Camera geometry utilities backed by the found C++ library.

The ``_cameraGeometry`` extension module exposes Camera, Vec2/Vec3/Mat3,
EulerAngles, Quaternion, Attitude, and the free functions SphericalToQuaternion,
FovToFocalLength, FocalLengthToFov.

This module re-exports all of those and additionally provides:

* :func:`sphericalToRotation` – z-y'-x'' Euler angles (rad) → 3×3 rotation matrix
* :func:`sphericalToRotationDegrees` – same, angles in degrees
* :func:`calibrationMatrix` – intrinsics K matrix from a Camera object
"""

import math

import numpy as np

from ._cameraGeometry import (  # noqa: F401  (re-exported for callers)
    Attitude,
    Camera,
    EulerAngles,
    FocalLengthToFov,
    FovToFocalLength,
    Mat3,
    Quaternion,
    SphericalToQuaternion,
    Vec2,
    Vec3,
)


def _mat3ToNumpy(m: Mat3) -> np.ndarray:
    return np.array([[m.At(i, j) for j in range(3)] for i in range(3)])


def sphericalToRotation(ra: float, dec: float, roll: float) -> np.ndarray:
    """Return the 3×3 equatorial-to-camera rotation matrix.

    Converts a z-y'-x'' Euler attitude (the same convention used by found's
    ``SphericalToQuaternion``) into the direction-cosine matrix that rotates
    vectors *from* equatorial (world) coordinates *into* camera-frame
    coordinates.

    Args:
        ra:   Right ascension in radians; yaw about Z applied first.
        dec:  Declination in radians; pitch about Y' applied second.
        roll: Roll in radians; rotation about X'' applied last.

    Returns:
        R: 3×3 NumPy float64 array, equatorial → camera.
    """
    q = SphericalToQuaternion(ra, dec, roll)
    return _mat3ToNumpy(Attitude(q).GetDCM())


def sphericalToRotationDegrees(
    ra_deg: float, dec_deg: float, roll_deg: float
) -> np.ndarray:
    """Return the 3×3 equatorial-to-camera rotation matrix (angles in degrees).

    Converts each angle from degrees to radians and delegates to
    :func:`sphericalToRotation`.

    Args:
        ra_deg:   Right ascension in degrees, in [0, 360).
        dec_deg:  Declination in degrees, in [-90, 90].
        roll_deg: Roll in degrees, in [0, 360). Positive roll is CCW about
                  the camera boresight.

    Returns:
        R: 3×3 NumPy float64 array, equatorial → camera.
    """
    return sphericalToRotation(
        math.radians(ra_deg),
        math.radians(dec_deg),
        math.radians(roll_deg),
    )


def calibrationMatrix(cam: Camera) -> np.ndarray:
    """Build the 3×3 intrinsics matrix K for *cam*.

    Uses the convention from Fundamentals of Space OpNav (Christian, 2026)
    where the camera X-axis is the optical axis (depth) and the virtual
    image plane is placed at x = 1::

        K = [[ cx,  -f/p,   0  ],
             [ cy,    0,  -f/p ],
             [  1,    0,    0  ]]

    This is consistent with found's ``SpatialToCamera`` / ``CameraToSpatial``
    implementations (pixel = K @ [1, y/x, z/x]).

    Args:
        cam: A :class:`Camera` instance (square pixels assumed).

    Note:
        When *cam* was constructed with explicit x/y centers those values
        are not accessible via the current C++ API; half-resolution is used
        as the principal point.

    Returns:
        K: 3×3 NumPy float64 array.
    """
    d = cam.FocalLength() / cam.PixelSize()
    cx = cam.XResolution() / 2.0
    cy = cam.YResolution() / 2.0
    return np.array(
        [[cx, -d, 0.0], [cy, 0.0, -d], [1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
