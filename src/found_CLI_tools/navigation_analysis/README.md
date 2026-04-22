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

## Generic Covariance Model

The generic case uses a different formulation from the parametric model. Instead of a closed-form geometry matrix, it builds the covariance from residuals of horizon points against a conic constraint.

This is the right path when the horizon is not well represented by the parametric circular model, or when you already have a conic matrix model for the observed limb.

### How the generic model works

The generic implementation follows this flow:

1. Evaluate residuals for each horizon point: `e_i = x_i^T M_c(sc) x_i`.
2. Compute the residual variance for each point: `sigma_i^2 = 4 x_i^T M_c R_xbar M_c x_i`.
3. Build the Jacobian `H` with respect to the 3-vector position estimate `s_c`.
4. Accumulate Fisher information: `I = sum_i H_i^T sigma_i^-2 H_i`.
5. Invert the Fisher matrix to get the covariance: `P_sc = I^-1`.

### Inputs

The generic model needs a different set of inputs than the parametric model:

- `sc_hat`: Estimated position vector in camera coordinates.
- `x_list`: Homogeneous horizon points, each a 3-vector.
- `recompute_Mc(sc)`: Callback that returns the 3x3 conic matrix for a given `s_c`.
- `sigma_x`: Image noise standard deviation.
- `eps`: Finite-difference step for the Jacobian.

### Diagnostics

The generic path should return or expose diagnostics so the computation is auditable:

- residuals per horizon point
- residual variances `sigma_i^2`
- Jacobian matrix `H`
- Fisher information matrix `I`
- Fisher condition number
- the conic matrix used at the solution

### Implementation plan for the generic case

The current `generic.py` file is the starting point for this work. The production version should:

1. Validate vector and matrix shapes before computing anything.
2. Keep the callback-based `recompute_Mc(sc)` interface so different conic models can be plugged in.
3. Return diagnostics alongside the covariance, like the parametric implementation does.
4. Detect singular or ill-conditioned Fisher matrices before inversion.
5. Add tests for residuals, Jacobians, Fisher assembly, and the failure path when the Fisher matrix is not invertible.

### Practical notes

- This model is more flexible than the parametric one, but it is also more numerical and more sensitive to noise in the Jacobian step.
- The finite-difference step `eps` may need tuning for different conic models.
- If you later have an analytical derivative for `M_c(sc)`, it can replace the finite-difference Jacobian for better accuracy and speed.

### Errors
- There should not be a supplied epsilon term. The "epsilon" that was generated
  here was a misinterpretation of the fact that we are interpreting the residuals
  as perturbations of the conic incidence equation.

1. We need a class that computes the covariance matrix (generic case)
2. This class should assume it's given all parameters needed, as we are not
   making the pipeline, just the component. These parameters include:
   - The shape matrix for the planetary body
   - The position vector estimate
   - The horizon points established in the image
   - The noise in each of the horizon points
3. For the class, we should take in one argument which is a `std::unique_ptr` to
   a class representing the CRA. Then we can calculate A_C from the principal
   axes and the Camera object in the CRA class.
