import numpy as np

def normalize(v):
    """Safely normalize a 2D vector."""
    norm = np.linalg.norm(v)
    if norm == 0:
        raise ValueError("Zero-length vector cannot be normalized")
    return v / norm


def illumination_frame_residual(
    centroid_est,
    centroid_true,
    sun_direction_image
):
    """
    Compute centroid residuals in the illumination frame.

    Parameters
    ----------
    centroid_est : array-like, shape (2,)
        Estimated centroid [u, v] in pixels
    centroid_true : array-like, shape (2,)
        Truth centroid [u, v] in pixels
    sun_direction_image : array-like, shape (2,)
        Sun direction projected into image plane (pixels or unitless)

    Returns
    -------
    residual_illum : ndarray, shape (2,)
        [x_sun, y_cross_sun] residuals in pixels
    """

    centroid_est = np.asarray(centroid_est, dtype=float)
    centroid_true = np.asarray(centroid_true, dtype=float)
    sun_direction_image = np.asarray(sun_direction_image, dtype=float)

    residual_img = centroid_est - centroid_true

    x_hat = normalize(sun_direction_image)
    y_hat = np.array([-x_hat[1], x_hat[0]])

    x_sun = np.dot(residual_img, x_hat)
    y_cross_sun = np.dot(residual_img, y_hat)

    return np.array([x_sun, y_cross_sun])
