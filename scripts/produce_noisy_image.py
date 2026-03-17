"""Produce a noisy image using limb.noise_generator_image (non-interactive)."""
import sys
from pathlib import Path

import cv2

from limb.noise_generator_image.noise import (
    add_gaussian_noise,
    add_salt_pepper_noise,
    apply_discretization,
    apply_radial_distortion,
)

def main():
    input_path = Path("sim_images/img_000000.png")
    output_path = Path("noisy_limb.png")
    if not input_path.exists():
        print(f"Missing {input_path}; run limb_simulation first.")
        return 1
    img = cv2.imread(str(input_path))
    if img is None:
        print(f"Could not load {input_path}")
        return 1
    # Apply a small pipeline: light Gaussian, light salt/pepper, discretization, slight distortion
    out = add_gaussian_noise(img, mean=0, sigma=8)
    out = add_salt_pepper_noise(out, salt_prob=0.005, pepper_prob=0.005)
    out = apply_discretization(out, levels=12)
    out = apply_radial_distortion(out, k1=0.02, k2=0.0, p1=0.0, p2=0.0)
    cv2.imwrite(str(output_path), out)
    print(f"Saved {output_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
