import numpy as np
import cv2

"""
This module provides functions for adding various types of noise and distortion
to images, simulating real-world camera artifacts and sensor imperfections.
These effects include Gaussian noise, salt-and-pepper noise, radial distortion,
motion blur, and color discretization.
"""


def add_gaussian_noise(image, mean=0, sigma=10):
    """Adds Gaussian noise to an image
    Args:
        image (np.ndarray): Input image as a NumPy array (H x W x C)
        mean (int, optional): Mean of the Gaussian distribution. Defaults to 0.
        sigma (int, optional): Standard deviation of the Gaussian distribution.
                               Higher values produce more noise. Defaults to 10.
    Returns:
        np.ndarray: Image with Gaussian noise added, same shape as input
    """
    gauss = np.random.normal(mean, sigma, image.shape).astype("uint8")
    noisy = cv2.add(image, gauss)
    return noisy


def add_salt_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
    """Adds salt-and-pepper noise to an image
    Args:
        image (np.ndarray): Input image as a NumPy array (H x W x C)
        salt_prob (float, optional): Probability of a pixel becoming white (salt).
                                     Range [0, 1]. Defaults to 0.01 (1%).
        pepper_prob (float, optional): Probability of a pixel becoming black (pepper).
                                       Range [0, 1]. Defaults to 0.01 (1%).
    Returns:
        np.ndarray: Image with salt-and-pepper noise added, same shape as input
    """
    noisy = image.copy()
    total_pixels = image.size

    num_salt = np.ceil(salt_prob * total_pixels)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy[tuple(coords)] = 255

    num_pepper = np.ceil(pepper_prob * total_pixels)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy[tuple(coords)] = 0
    return noisy


def apply_radial_distortion(image, k1=0.0, k2=0.0, p1=0.0, p2=0.0):
    """Applies radial and tangential distortion to an image following the Brown-Conrady model
    Args:
        image (np.ndarray): Input image as a NumPy array (H x W x C)
        k1 (float, optional): Radial distortion coefficient (2nd order).
                             Positive = pincushion, Negative = barrel. Defaults to 0.0.
        k2 (float, optional): Radial distortion coefficient (4th order). Defaults to 0.0.
        p1 (float, optional): Tangential distortion coefficient. Defaults to 0.0.
        p2 (float, optional): Tangential distortion coefficient. Defaults to 0.0.
    Returns:
        np.ndarray: Distorted image, same shape as input
    """
    h, w = image.shape[:2]
    fx = fy = 1.0
    cx, cy = w / 2, h / 2
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    D = np.array([k1, k2, p1, p2, 0])
    map1, map2 = cv2.initUndistortRectifyMap(K, D, None, K, (w, h), cv2.CV_32FC1)
    distorted = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)
    return distorted


def apply_motion_blur(image, kernel_size=5):
    """Applies horizontal motion blur to an image
    Args:
        image (np.ndarray): Input image as a NumPy array (H x W x C)
        kernel_size (int, optional): Size of the motion blur kernel in pixels.
                                    Larger values create stronger blur.
                                    Must be odd and >= 1. Defaults to 5.
    Returns:
        np.ndarray: Motion-blurred image, same shape as input
    """
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    kernel = kernel / kernel_size
    blurred = cv2.filter2D(image, -1, kernel)
    return blurred


def apply_discretization(image, levels=8):
    """Applies color discretization to an image
    Args:
        image (np.ndarray): Input image as a NumPy array (H x W x C)
        levels (int, optional): Number of discrete intensity levels per channel.
                               Range [1, 256]. Defaults to 8.
    Returns:
        np.ndarray: Discretized image with reduced color depth, same shape as input
    """
    if levels < 1:
        levels = 1
    factor = max(1, 256 // levels)
    temp = (image.astype(np.int32) // factor) * factor
    discretized = np.clip(temp, 0, 255).astype(np.uint8)
    return discretized
