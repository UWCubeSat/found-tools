"""CLI entry point for the parametric covariance model."""

import argparse

import numpy as np

from .constants import DEFAULT_NUM_POINTS
from .parametric_model import compute_parametric_covariance


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute the Christian OpNav parametric covariance matrix"
    )
    
    # Body radius parameter: usually the radius of the earth or moon
    # A larger body radius reduces the covariance because it increases the curvature of the horizon, making it easier to fit accurately
    parser.add_argument("--body-radius", type=float, required=True)

    # Position vector of the camera
    # The output of FOUND, what the covariance is meant to be a model of the error of
    parser.add_argument("--position-vector", nargs=3, type=float, required=True)

    # Sun vector: unit vector pointing from the celestial body to the sun
    # Only used to define the camera frame basis
    parser.add_argument("--sun-vector", nargs=3, type=float, required=True)

    # Maximum angle of the visible surface from the camera boresight, in degrees
    # Basically the length of the horizon arc seen by the camera
    # A larger angle reduces the covariance but increases the risk of outliers in the horizon fitting step of FOUND
    parser.add_argument("--theta-max-deg", type=float, required=True)

    # Measurement noise standard deviation in the camera frame, in meters
    # More noise = higher covariance
    parser.add_argument("--sigma-x", type=float, required=True)

    # Number of points used in the horizon fitting step of FOUND
    # A larger number of points reduces the covariance but increases the risk of outliers
    parser.add_argument("--num-points", type=int, default=DEFAULT_NUM_POINTS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    covariance, diagnostics = compute_parametric_covariance(
        args.position_vector,
        args.sun_vector,
        args.body_radius,
        args.theta_max_deg,
        args.sigma_x,
        args.num_points,
    )

    np.set_printoptions(precision=6, suppress=True)
    print("Parametric covariance matrix:")
    print(covariance)
    print()
    print("Diagnostics:")
    print(f"slant_range = {diagnostics.slant_range:.6f}")
    print(f"body_radius = {diagnostics.body_radius:.6f}")
    print(f"theta_max_deg = {diagnostics.theta_max_deg:.6f}")
    print(f"theta_max_rad = {diagnostics.theta_max_rad:.6f}")
    print(f"denominator_d = {diagnostics.denominator_d:.6f}")
    print(f"scale_factor = {diagnostics.scale_factor:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
