#!/usr/bin/env python3
"""
This module provides an interactive GUI-based tool for adjusting noise and
distortion parameters on pre-generated images. Users can load an image,
adjust various noise parameters via trackbars in real-time, and save the
result when satisfied.
"""
import argparse
import sys
import cv2

from found_CLI_tools.noise_generator_image.noise import (
    add_gaussian_noise,
    add_salt_pepper_noise,
    apply_discretization,
    apply_motion_blur,
    apply_radial_distortion,
)

def interactive_noise_adjustment(base_image, output_path):
    """
    Open an interactive GUI window for real-time noise adjustment.
    
    Args:
        base_image (np.ndarray): Input image to apply noise to (H x W x C).
                                This image is never modified.
        output_path (str): File path where the final image should be saved
                          when user presses 'S'.
    
    Returns:
        bool: True if image was saved successfully, False if user exited
              without saving.
    """
    
    def nothing(x):
        """
        Dummy callback function for OpenCV trackbars.
        
        Args:
            x: Trackbar value (unused)
        
        Returns:
            None
        """
        pass
    
    window_name = 'Noise & Distortion - Press S to Save, ESC to Quit'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 800)
    
    # Create trackbars for noise parameters
    cv2.createTrackbar('Gaussian Sigma', window_name, 0, 100, nothing)
    cv2.createTrackbar('Salt Prob x1000', window_name, 0, 100, nothing)
    cv2.createTrackbar('Pepper Prob x1000', window_name, 0, 100, nothing)
    cv2.createTrackbar('Discretization', window_name, 3, 32, nothing)
    cv2.createTrackbar('k1 x100', window_name, 0, 100, nothing)
    cv2.createTrackbar('k2 x100', window_name, 0, 100, nothing)
    cv2.createTrackbar('p1 x100', window_name, 0, 100, nothing)
    cv2.createTrackbar('p2 x100', window_name, 0, 100, nothing)
    cv2.createTrackbar('Motion Kernel', window_name, 0, 100, nothing)
    
    print("\n=== Interactive Noise Adjustment ===")
    print("Adjust the sliders to modify noise and distortion")
    print("Press 'S' to save the current image")
    print("Press 'ESC' to quit without saving")
    print("====================================\n")
    
    saved = False
    
    while True:
        # Get trackbar values
        sigma = cv2.getTrackbarPos('Gaussian Sigma', window_name)
        salt = cv2.getTrackbarPos('Salt Prob x1000', window_name) / 1000.0
        pepper = cv2.getTrackbarPos('Pepper Prob x1000', window_name) / 1000.0
        discretization_levels = max(1, cv2.getTrackbarPos('Discretization', window_name))
        k1 = cv2.getTrackbarPos('k1 x100', window_name) / 100.0
        k2 = cv2.getTrackbarPos('k2 x100', window_name) / 100.0
        p1 = cv2.getTrackbarPos('p1 x100', window_name) / 100.0
        p2 = cv2.getTrackbarPos('p2 x100', window_name) / 100.0
        motion_kernel = max(1, cv2.getTrackbarPos('Motion Kernel', window_name))
        
        # Apply effects to base image
        result = apply_radial_distortion(base_image.copy(), k1, k2, p1, p2)
        result = add_gaussian_noise(result, sigma=sigma)
        result = add_salt_pepper_noise(result, salt, pepper)
        result = apply_discretization(result, levels=discretization_levels)
        result = apply_motion_blur(result, motion_kernel)
        
        # Display
        cv2.imshow(window_name, result)
        
        # Handle key presses
        key = cv2.waitKey(50) & 0xFF
        
        if key == 27:  # ESC
            print("Exiting without saving...")
            break
        elif key == ord('s') or key == ord('S'):  # Save
            cv2.imwrite(output_path, result)
            print(f"\nImage saved to: {output_path}")
            print(f"  Gaussian Sigma: {sigma}")
            print(f"  Salt Probability: {salt:.3f}")
            print(f"  Pepper Probability: {pepper:.3f}")
            print(f"  Discretization Levels: {discretization_levels}")
            print(f"  Radial Distortion: k1={k1:.2f}, k2={k2:.2f}, p1={p1:.2f}, p2={p2:.2f}")
            print(f"  Motion Kernel: {motion_kernel}")
            saved = True
            break
    
    cv2.destroyAllWindows()
    return saved


def main():
    """
    Main entry point for the interactive noise adjuster.
    
    Returns:
        int: Exit code (0 if image saved, 1 if error or user cancelled)
    """
    parser = argparse.ArgumentParser(
        description='Load an image and interactively adjust noise/distortion',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Load an image and adjust noise interactively
            python -m found_CLI_tools.noise_generator_image example_earth.png noisy_output.png
            
            # Same with different images
            python -m found_CLI_tools.noise_generator_image my_image.png result.png

            Once the GUI opens:
            - Adjust sliders to modify noise and distortion in real-time
            - Press 'S' to save the image
            - Press 'ESC' to quit without saving
        """
    )
    
    # Input and output files (required)
    parser.add_argument('input', help='Input image file path')
    parser.add_argument('output', help='Output image file path')
    
    args = parser.parse_args()
    
    # Load the input image
    print(f"Loading image: {args.input}")
    base_image = cv2.imread(args.input)
    
    if base_image is None:
        print(f"ERROR: Could not load image from {args.input}")
        print("Please check that the file exists and is a valid image.")
        return 1
    
    print(f"Image loaded: {base_image.shape[1]}x{base_image.shape[0]} pixels")
    print("\nOpening interactive adjustment window...")
    
    # Open interactive GUI
    saved = interactive_noise_adjustment(base_image, args.output)
    
    if not saved:
        print("\nNo image was saved.")
    
    return 0 if saved else 1

if __name__ == '__main__':
    sys.exit(main())