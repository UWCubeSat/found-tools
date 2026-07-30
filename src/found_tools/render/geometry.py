"""Ephemeris and frame-conversion helpers for the render tool.

Provides the Sun direction in an Earth-centered inertial (ECI, mean-of-date)
frame and a rotation from ECI to Earth-centered Earth-fixed (ECEF), which
together are enough to pose the Sun lamp and orient the Earth mesh in
Blender for a given UTC datetime.

We deliberately use closed-form, low-precision analytical formulas (Vallado,
*Fundamentals of Astrodynamics and Applications*, Alg. 29 for the Sun vector;
the IAU 1982 GMST polynomial for the ECI->ECEF rotation) rather than an
ephemeris library such as spiceypy or astropy. For synthetic test imagery,
sub-arcminute precision is more than sufficient, and dropping the SPICE
kernel / IERS data dependency keeps this tool's setup to "pip install" and
keeps it fully unit-testable offline. If a project later needs
higher-precision ephemerides (e.g. planetary parallax), spiceypy or
astropy.coordinates can be swapped in behind the same function signatures.
"""

from datetime import UTC, datetime

import numpy as np

from found_tools.render.constants import (
    DAYS_PER_JULIAN_CENTURY,
    EARTH_OBLIQUITY_DEG,
    EARTH_OBLIQUITY_RATE_DEG_PER_DAY,
    J2000_JD,
)


def to_julian_date(when: datetime) -> float:
    """Converts a UTC datetime to a Julian Date.

    Args:
        when: The UTC datetime to convert. Naive datetimes are assumed UTC.

    Returns:
        float: The Julian Date.
    """
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    when = when.astimezone(UTC)

    unix_epoch_jd = 2440587.5
    return unix_epoch_jd + when.timestamp() / 86400.0


def days_since_j2000(when: datetime) -> float:
    """Computes the number of days elapsed since the J2000.0 epoch.

    Args:
        when: The UTC datetime.

    Returns:
        float: Days since J2000.0 (may be negative for earlier dates).
    """
    return to_julian_date(when) - J2000_JD


def sun_vector_eci(when: datetime) -> np.ndarray:
    """Computes the unit vector from Earth to the Sun in the ECI frame.

    Uses the low-precision solar coordinates formula (Vallado, Alg. 29),
    accurate to about 0.01 degrees, which is sufficient for driving a
    Sun lamp direction and terminator position in a rendered scene.

    Args:
        when: The UTC datetime at which to evaluate the Sun's position.

    Returns:
        np.ndarray: Shape (3,) unit vector pointing from Earth to the Sun,
            expressed in the ECI (mean equator/equinox of date) frame.
    """
    n = days_since_j2000(when)

    mean_longitude = np.deg2rad((280.460 + 0.9856474 * n) % 360.0)
    mean_anomaly = np.deg2rad((357.528 + 0.9856003 * n) % 360.0)

    ecliptic_longitude = mean_longitude + np.deg2rad(
        1.915 * np.sin(mean_anomaly) + 0.020 * np.sin(2 * mean_anomaly)
    )

    obliquity = np.deg2rad(EARTH_OBLIQUITY_DEG + EARTH_OBLIQUITY_RATE_DEG_PER_DAY * n)

    x = np.cos(ecliptic_longitude)
    y = np.cos(obliquity) * np.sin(ecliptic_longitude)
    z = np.sin(obliquity) * np.sin(ecliptic_longitude)

    return np.array([x, y, z], dtype=np.float64)


def gmst_radians(when: datetime) -> float:
    """Computes the Greenwich Mean Sidereal Time.

    Uses the IAU 1982 GMST polynomial. Treats the input UTC time as UT1,
    which is within ~1 second and negligible for rendering purposes.

    Args:
        when: The UTC datetime.

    Returns:
        float: GMST in radians, wrapped to [0, 2*pi).
    """
    jd = to_julian_date(when)
    t = (jd - J2000_JD) / DAYS_PER_JULIAN_CENTURY

    gmst_deg = (
        280.46061837
        + 360.98564736629 * (jd - J2000_JD)
        + 0.000387933 * t**2
        - (t**3) / 38710000.0
    )

    return np.deg2rad(gmst_deg % 360.0)


def eci_to_ecef_matrix(when: datetime) -> np.ndarray:
    """Computes the rotation matrix from ECI to ECEF at a given time.

    Ignores precession, nutation, and polar motion (sub-arcsecond effects),
    modeling ECEF as a pure Earth-rotation-angle rotation of ECI about the Z
    axis. This is appropriate for posing the Earth mesh and Sun lamp for a
    rendered image, not for high-precision navigation.

    Args:
        when: The UTC datetime.

    Returns:
        np.ndarray: Shape (3, 3) rotation matrix R such that
            v_ecef = R @ v_eci.
    """
    theta = gmst_radians(when)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [cos_t, sin_t, 0.0],
            [-sin_t, cos_t, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def eci_to_ecef(vector_eci: np.ndarray, when: datetime) -> np.ndarray:
    """Rotates a vector from the ECI frame into the ECEF frame.

    Args:
        vector_eci: Shape (3,) vector expressed in the ECI frame.
        when: The UTC datetime at which to evaluate the rotation.

    Returns:
        np.ndarray: Shape (3,) vector expressed in the ECEF frame.
    """
    return eci_to_ecef_matrix(when) @ np.asarray(vector_eci, dtype=np.float64)


def sun_vector_ecef(when: datetime) -> np.ndarray:
    """Computes the unit vector from Earth to the Sun in the ECEF frame.

    Args:
        when: The UTC datetime at which to evaluate the Sun's position.

    Returns:
        np.ndarray: Shape (3,) unit vector pointing from Earth to the Sun,
            expressed in the ECEF frame.
    """
    return eci_to_ecef(sun_vector_eci(when), when)
