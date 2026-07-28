from datetime import datetime, timezone

import numpy as np
import pytest

from found_tools.render.geometry import (
    days_since_j2000,
    eci_to_ecef,
    eci_to_ecef_matrix,
    gmst_radians,
    sun_vector_ecef,
    sun_vector_eci,
    to_julian_date,
)

J2000_EPOCH = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
J2000_EPOCH_NAIVE = datetime(2000, 1, 1, 12, 0, 0)


def test_to_julian_date_at_j2000():
    assert to_julian_date(J2000_EPOCH) == pytest.approx(2451545.0, abs=1e-9)


def test_to_julian_date_treats_naive_datetime_as_utc():
    assert to_julian_date(J2000_EPOCH_NAIVE) == pytest.approx(
        to_julian_date(J2000_EPOCH), abs=1e-9
    )


def test_to_julian_date_one_day_later():
    later = datetime(2000, 1, 2, 12, 0, 0, tzinfo=timezone.utc)
    assert to_julian_date(later) == pytest.approx(2451546.0, abs=1e-9)


def test_days_since_j2000_at_epoch():
    assert days_since_j2000(J2000_EPOCH) == pytest.approx(0.0, abs=1e-9)


def test_days_since_j2000_before_epoch():
    earlier = datetime(1999, 12, 31, 12, 0, 0, tzinfo=timezone.utc)
    assert days_since_j2000(earlier) == pytest.approx(-1.0, abs=1e-9)


def test_sun_vector_eci_is_unit_length():
    for when in [
        J2000_EPOCH,
        datetime(2026, 3, 20, 9, 0, 0, tzinfo=timezone.utc),
        datetime(2026, 6, 21, 4, 0, 0, tzinfo=timezone.utc),
        datetime(2026, 12, 21, 21, 0, 0, tzinfo=timezone.utc),
    ]:
        vector = sun_vector_eci(when)
        assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-6)


def test_sun_vector_eci_near_summer_solstice_points_toward_positive_z():
    # Near the June solstice, the Sun's declination is close to its maximum
    # (~+23.4 deg), so the ECI z-component should be strongly positive.
    when = datetime(2026, 6, 21, 4, 0, 0, tzinfo=timezone.utc)
    vector = sun_vector_eci(when)
    assert vector[2] == pytest.approx(np.sin(np.deg2rad(23.4)), abs=0.02)


def test_sun_vector_eci_near_winter_solstice_points_toward_negative_z():
    when = datetime(2026, 12, 21, 21, 0, 0, tzinfo=timezone.utc)
    vector = sun_vector_eci(when)
    assert vector[2] == pytest.approx(-np.sin(np.deg2rad(23.4)), abs=0.02)


def test_sun_vector_eci_near_equinox_has_small_z_component():
    when = datetime(2026, 3, 20, 9, 0, 0, tzinfo=timezone.utc)
    vector = sun_vector_eci(when)
    assert abs(vector[2]) < 0.02


def test_gmst_radians_at_j2000_matches_known_value():
    # At T=0 (J2000.0 exactly), the IAU 1982 GMST polynomial reduces to its
    # constant term, 280.46061837 degrees.
    expected = np.deg2rad(280.46061837 % 360.0)
    assert gmst_radians(J2000_EPOCH) == pytest.approx(expected, abs=1e-9)


def test_gmst_radians_is_wrapped_to_2pi():
    for when in [J2000_EPOCH, datetime(2030, 6, 1, tzinfo=timezone.utc)]:
        theta = gmst_radians(when)
        assert 0.0 <= theta < 2 * np.pi


def test_eci_to_ecef_matrix_is_orthonormal():
    matrix = eci_to_ecef_matrix(J2000_EPOCH)
    assert matrix @ matrix.T == pytest.approx(np.eye(3), abs=1e-9)
    assert np.linalg.det(matrix) == pytest.approx(1.0, abs=1e-9)


def test_eci_to_ecef_preserves_vector_length():
    vector = np.array([1.0, 2.0, 3.0])
    rotated = eci_to_ecef(vector, J2000_EPOCH)
    assert np.linalg.norm(rotated) == pytest.approx(np.linalg.norm(vector), abs=1e-9)


def test_eci_to_ecef_z_component_is_unchanged():
    # The ECI->ECEF rotation here is purely about the Z axis.
    vector = np.array([1.0, 2.0, 3.0])
    rotated = eci_to_ecef(vector, J2000_EPOCH)
    assert rotated[2] == pytest.approx(vector[2], abs=1e-9)


def test_sun_vector_ecef_is_unit_length():
    vector = sun_vector_ecef(J2000_EPOCH)
    assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-6)


def test_sun_vector_ecef_z_component_matches_eci():
    # Rotation about Z leaves the Z component (declination-driven) unchanged.
    when = datetime(2026, 6, 21, 4, 0, 0, tzinfo=timezone.utc)
    eci = sun_vector_eci(when)
    ecef = sun_vector_ecef(when)
    assert ecef[2] == pytest.approx(eci[2], abs=1e-9)
