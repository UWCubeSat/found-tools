# Navigation Analysis: Parametric Covariance Model

This folder contains a Python implementation of a parametric covariance model for optical navigation (OpNav), based on a Christian textbook excerpt (Eq. 9.168 and Eq. 9.169).

The model estimates a 3x3 covariance matrix for a position estimate derived from horizon-based measurements. It maps image-domain noise into inertial-frame position uncertainty.

## What the model computes

Given observation geometry and image noise assumptions, the model computes:

1. A camera-frame geometry matrix.
2. A scalar scale factor based on noise, slant range, body radius, and number of points.
3. A camera-frame covariance.
4. An inertial-frame covariance by rotating with a camera basis matrix.

In code, this is performed by:

- `build_camera_frame_basis(...)`
- `build_geometry_matrix(...)`
- `compute_parametric_covariance(...)`

All of these are implemented in `model.py`.

## Inputs

The CLI in `main.py` accepts:

- `--body-radius` (`float`, required): Radius of the observed body.
- `--position-vector` (`3 floats`, required): Camera position vector relative to body center.
- `--sun-vector` (`3 floats`, required): Sun direction vector used to define the camera basis.
- `--theta-max-deg` (`float`, required): Horizon-angle parameter in degrees.
- `--sigma-x` (`float`, required): Measurement noise standard deviation.
- `--num-points` (`int`, optional): Number of points used in the fit (default from constants).

## Outputs

The CLI prints:

- `Parametric covariance matrix`: Final 3x3 covariance in inertial coordinates.
- `Diagnostics`: Intermediate values used to build the covariance.

Diagnostics include:

- `slant_range`
- `body_radius`
- `theta_max_deg`
- `theta_max_rad`
- `denominator_d`
- `scale_factor`

The model function also returns matrix diagnostics (basis and intermediate covariance matrices) through `ParametricCovarianceDiagnostics`.

## Practical interpretation

- Larger `sigma_x` increases covariance.
- Larger `num_points` decreases covariance (approximately inverse scaling).
- Larger slant range generally increases uncertainty.
- Off-diagonal covariance terms indicate coupling between axis errors.

## Run from repository root

Example:

```bash
python .\found-tools\src\found_CLI_tools\navigation_analysis\main.py \
  --body-radius 1.0 \
  --position-vector 0 0 2 \
  --sun-vector 0 1 0 \
  --theta-max-deg 70 \
  --sigma-x 0.5 \
  --num-points 100
```

## Notes

- This implementation assumes valid geometric configuration (`slant_range > body_radius`).
- The sun vector and position vector must be non-zero 3-vectors.
- The matrix structure in the camera frame follows the textbook model and is symmetric.
