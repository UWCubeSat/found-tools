"""Manual test / debug script for add_point_noise.

Run from repo root:
  uv run python scripts/test_add_point_noise.py

Or run under the debugger using the "Debug add_point_noise script" launch config.
Set breakpoints in this file or in limb.simulation.edge.conic.add_point_noise.
"""

from __future__ import annotations

import numpy as np

from limb.simulation.edge.conic import add_point_noise
from limb.utils._camera import Camera


def main() -> None:
    camera = Camera(
        focal_length=0.035,
        x_pixel_pitch=5e-6,
        x_resolution=64,
        y_resolution=64,
    )

    # 1) Empty points
    out_empty = add_point_noise(np.empty((0, 2)), camera)
    assert out_empty.shape == (0, 2), out_empty.shape

    # 2) Single point, Gaussian noise (scalar sigma)
    # Use dtype=np.float64 explicitly so input is double; otherwise display/caller can look like float32.
    pts = np.array(
        [[32.59765865087698758764587654, 32.59765865087698758764587654]],
        dtype=np.float64,
    )
    assert pts.dtype == np.float64, f"Expected float64, got {pts.dtype}"
    rng = np.random.default_rng(42)
    out_gaussian = add_point_noise(pts, camera, gaussian_sigma=0.5, truncate=22, rng=rng)
    assert out_gaussian.shape[1] == 2
    print("Gaussian (scalar sigma):", out_gaussian.shape[0], "points")

    # 3) Gaussian noise with (sigma_x, sigma_y)
    out_xy = add_point_noise(
        pts, camera, gaussian_sigma=(0.1, 0.2), rng=np.random.default_rng(42)
    )
    print("Gaussian (tuple sigma):", out_xy.shape[0], "points")

    # 4) False points only
    out_false = add_point_noise(
        np.empty((0, 2)), camera, n_false_points=5, rng=np.random.default_rng(42)
    )
    assert out_false.shape == (5, 2)
    print("False points:", out_false.shape[0])

    # 5) Combined: noisy edge + false points
    edge_pts = np.array([[10.0, 20.0], [50.0, 30.0], [32.0, 32.0]])
    out_combined = add_point_noise(
        edge_pts,
        camera,
        gaussian_sigma=0.3,
        n_false_points=3,
        truncate=2,
        rng=np.random.default_rng(123),
    )
    print("Combined (noise + false, truncate=2):", out_combined.shape[0], "points")
    print("Sample output:\n", out_combined[:5])

    print("All add_point_noise checks passed.")


if __name__ == "__main__":
    main()
