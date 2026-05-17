import numpy as np
import pytest

from limb.utils._camera import Camera

MATRIX_INVERSE_TOL = 1e-9


@pytest.fixture
def ideal_camera():
    # focalLength=0.01, pixelPitch=1e-5, 1000x1000 → fx=fy=1000, center=(500,500)
    return Camera(0.01, 1e-5, 1000, 1000)


@pytest.fixture
def full_camera():
    # focalLength=0.05, 1280x720, center=(640.5,360.5), xPitch=5e-6, yPitch=4e-6
    # → fx=10000, fy=12500
    return Camera(0.05, 5e-6, 1280, 720, 640.5, 360.5, 4e-6)


################################
### CALIBRATION MATRIX TESTS ###
################################


def test_ideal_camera_calibration_matrix_values(ideal_camera):
    K = ideal_camera.calibration_matrix
    # Row 0: [x_center, -fx, 0]  fx = 0.01/1e-5 = 1000
    assert K[0, 0] == pytest.approx(500.0, abs=1e-9)
    assert K[0, 1] == pytest.approx(-1000.0, abs=1e-9)
    assert K[0, 2] == pytest.approx(0.0, abs=1e-9)
    # Row 1: [y_center, 0, -fy]  fy = 0.01/1e-5 = 1000
    assert K[1, 0] == pytest.approx(500.0, abs=1e-9)
    assert K[1, 1] == pytest.approx(0.0, abs=1e-9)
    assert K[1, 2] == pytest.approx(-1000.0, abs=1e-9)
    # Row 2: [1, 0, 0]
    assert K[2, 0] == pytest.approx(1.0, abs=1e-9)
    assert K[2, 1] == pytest.approx(0.0, abs=1e-9)
    assert K[2, 2] == pytest.approx(0.0, abs=1e-9)


def test_full_camera_calibration_matrix_values(full_camera):
    K = full_camera.calibration_matrix
    # fx = 0.05/5e-6 = 10000, fy = 0.05/4e-6 = 12500
    # Row 0: [x_center, -fx, 0]
    assert K[0, 0] == pytest.approx(640.5, abs=1e-6)
    assert K[0, 1] == pytest.approx(-10000.0, abs=1e-6)
    assert K[0, 2] == pytest.approx(0.0, abs=1e-6)
    # Row 1: [y_center, 0, -fy]
    assert K[1, 0] == pytest.approx(360.5, abs=1e-6)
    assert K[1, 1] == pytest.approx(0.0, abs=1e-6)
    assert K[1, 2] == pytest.approx(-12500.0, abs=1e-6)
    # Row 2: [1, 0, 0]
    assert K[2, 0] == pytest.approx(1.0, abs=1e-6)
    assert K[2, 1] == pytest.approx(0.0, abs=1e-6)
    assert K[2, 2] == pytest.approx(0.0, abs=1e-6)


def test_ideal_inverse_calibration_matrix(ideal_camera):
    product = ideal_camera.calibration_matrix @ ideal_camera.inverse_calibration_matrix
    assert product == pytest.approx(np.eye(3), abs=MATRIX_INVERSE_TOL)


def test_full_inverse_calibration_matrix(full_camera):
    product = full_camera.calibration_matrix @ full_camera.inverse_calibration_matrix
    assert product == pytest.approx(np.eye(3), abs=MATRIX_INVERSE_TOL)


####################################
### CAMERA TO PIXEL COORD TESTS  ###
####################################


def test_camera_to_pixel_on_axis(ideal_camera):
    # On-axis projection is independent of depth; all map to center (500, 500)
    for v in [np.array([1, 0, 0]), np.array([5, 0, 0]), np.array([100, 0, 0])]:
        p = ideal_camera.camera_to_pixel(v)
        assert p[0] == pytest.approx(500.0, abs=1e-9)
        assert p[1] == pytest.approx(500.0, abs=1e-9)


def test_camera_to_pixel_simple_right(ideal_camera):
    # (2, -1, 0): pixel_x = 500 - 1000*(-0.5) = 1000, pixel_y = 500
    p = ideal_camera.camera_to_pixel(np.array([2, -1, 0]))
    assert p[0] == pytest.approx(1000.0, abs=1e-9)
    assert p[1] == pytest.approx(500.0, abs=1e-9)


def test_camera_to_pixel_simple_down(ideal_camera):
    # (2, 0, -1): pixel_x = 500, pixel_y = 500 - 1000*(-0.5) = 1000
    p = ideal_camera.camera_to_pixel(np.array([2, 0, -1]))
    assert p[0] == pytest.approx(500.0, abs=1e-9)
    assert p[1] == pytest.approx(1000.0, abs=1e-9)


def test_camera_to_pixel_perspective(ideal_camera):
    # (2, -1, -1): pixel_x = 1000, pixel_y = 1000
    p = ideal_camera.camera_to_pixel(np.array([2, -1, -1]))
    assert p[0] == pytest.approx(1000.0, abs=1e-9)
    assert p[1] == pytest.approx(1000.0, abs=1e-9)


#################################
### PIXEL TO IMAGE COORD TESTS ###
#################################


def test_pixel_to_image_center(ideal_camera):
    img = ideal_camera.inverse_calibration_matrix @ np.array([500.0, 500.0, 1.0])
    assert img[0] == pytest.approx(1.0, abs=1e-9)
    assert img[1] == pytest.approx(0.0, abs=1e-9)
    assert img[2] == pytest.approx(0.0, abs=1e-9)


def test_pixel_to_image_off_center_positive(ideal_camera):
    img = ideal_camera.inverse_calibration_matrix @ np.array([1000.0, 1000.0, 1.0])
    assert img[0] == pytest.approx(1.0, abs=1e-9)
    assert img[1] == pytest.approx(-0.5, abs=1e-9)
    assert img[2] == pytest.approx(-0.5, abs=1e-9)


def test_pixel_to_image_off_center_negative(ideal_camera):
    img = ideal_camera.inverse_calibration_matrix @ np.array([0.0, 0.0, 1.0])
    assert img[0] == pytest.approx(1.0, abs=1e-9)
    assert img[1] == pytest.approx(0.5, abs=1e-9)
    assert img[2] == pytest.approx(0.5, abs=1e-9)


############################
### IN-SENSOR BOUND TESTS ###
############################


def _in_sensor(cam: Camera, px: float, py: float) -> bool:
    return 0 <= px < cam.x_resolution and 0 <= py < cam.y_resolution


def test_in_sensor_out_of_bounds_negative_x(ideal_camera):
    assert not _in_sensor(ideal_camera, -1, 500)


def test_in_sensor_out_of_bounds_large_x(ideal_camera):
    assert not _in_sensor(ideal_camera, 1001, 500)


def test_in_sensor_out_of_bounds_negative_y(ideal_camera):
    assert not _in_sensor(ideal_camera, 500, -1)


def test_in_sensor_out_of_bounds_large_y(ideal_camera):
    assert not _in_sensor(ideal_camera, 500, 1001)


##############################
### MIN IMAGE DIMENSION TESTS ###
##############################


def test_min_image_dimension_ideal_camera(ideal_camera):
    # min(1000 * 1e-5, 1000 * 1e-5) = 0.01 m
    assert ideal_camera.min_image_dimension() == pytest.approx(0.01, abs=1e-12)


def test_min_image_dimension_full_camera(full_camera):
    # min(1280 * 5e-6, 720 * 4e-6) = min(0.0064, 0.00288) = 0.00288 m
    assert full_camera.min_image_dimension() == pytest.approx(0.00288, abs=1e-12)


def test_min_image_dimension_picks_smaller_axis():
    # x-axis: 100 * 1e-5 = 0.001, y-axis: 2000 * 1e-5 = 0.02 → should return 0.001
    cam = Camera(0.01, 1e-5, 100, 2000)
    assert cam.min_image_dimension() == pytest.approx(0.001, abs=1e-12)
