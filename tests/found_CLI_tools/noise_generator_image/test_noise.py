import unittest
import numpy as np

from src.found_CLI_tools.noise_generator_image.noise import (
    add_gaussian_noise,
    add_salt_pepper_noise,
    apply_radial_distortion,
    apply_motion_blur,
    apply_discretization,
)

class NoiseTest(unittest.TestCase):
    """Gaussian Noise Tests"""

    def test_gaussian_noise_zero_sigma(self):
        """Zero sigma should produce identical output"""
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        result = add_gaussian_noise(img, sigma=0)
        
        diff = np.abs(result.astype(np.int16) - img.astype(np.int16))
        self.assertTrue(np.all(diff <= 1))

    def test_gaussian_noise_properties(self):
        """Test Gaussian noise preserves basic image properties"""
        TESTS = 100
        
        for _ in range(TESTS):
            h, w = np.random.randint(50, 200, 2)
            img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
            sigma = np.random.rand() * 50
            
            result = add_gaussian_noise(img, sigma=sigma)
            
            self.assertEqual(result.shape, img.shape)
            self.assertEqual(result.dtype, np.uint8)
            self.assertTrue(np.all(result >= 0))
            self.assertTrue(np.all(result <= 255))

    def test_gaussian_noise_increases_variance(self):
        """Gaussian noise should increase image variance"""
        TESTS = 50
        
        for _ in range(TESTS):
            img = np.full((100, 100, 3), 128, dtype=np.uint8)
            sigma = np.random.rand() * 30 + 10
            
            result = add_gaussian_noise(img, sigma=sigma)
            
            self.assertGreater(np.var(result), np.var(img))

    """ Salt and Pepper Noise Tests """

    def test_salt_pepper_zero_probability(self):
        """Zero probability should produce identical output"""
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        result = add_salt_pepper_noise(img, salt_prob=0.0, pepper_prob=0.0)
        
        np.testing.assert_array_equal(result, img)

    def test_salt_pepper_properties(self):
        """Test salt and pepper noise basic properties"""
        TESTS = 100
        
        for _ in range(TESTS):
            h, w = np.random.randint(50, 200, 2)
            img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
            salt_prob = np.random.rand() * 0.05
            pepper_prob = np.random.rand() * 0.05
            
            result = add_salt_pepper_noise(img, salt_prob, pepper_prob)
            
            self.assertEqual(result.shape, img.shape)
            self.assertEqual(result.dtype, np.uint8)
            self.assertTrue(np.all(result >= 0))
            self.assertTrue(np.all(result <= 255))

    def test_salt_pepper_probability_distribution(self):
        """Test salt and pepper noise follows specified probability"""
        TESTS = 20
        
        for _ in range(TESTS):
            img = np.full((500, 500, 3), 128, dtype=np.uint8)
            prob = np.random.rand() * 0.02 + 0.01
            
            result = add_salt_pepper_noise(img, salt_prob=prob, pepper_prob=0)
            
            salt_count = np.sum(result == 255)
            total_pixels = img.size
            actual_prob = salt_count / total_pixels
            
            # Within 50% of expected
            np.testing.assert_allclose(actual_prob, prob, rtol=0.5)

    """ Radial Distortion Tests """

    def test_radial_distortion_zero_coefficients(self):
        """Zero distortion should produce nearly identical output"""
        TESTS = 20
        
        for _ in range(TESTS):
            h, w = np.random.randint(100, 300, 2)
            img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
            
            result = apply_radial_distortion(img, k1=0, k2=0, p1=0, p2=0)
            
            self.assertEqual(result.shape, img.shape)

    def test_radial_distortion_properties(self):
        """Test radial distortion preserves image properties"""
        TESTS = 100
        
        for _ in range(TESTS):
            h, w = np.random.randint(50, 200, 2)
            img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
            k1 = np.random.rand() * 0.4 - 0.2
            k2 = np.random.rand() * 0.2 - 0.1
            p1 = np.random.rand() * 0.1
            p2 = np.random.rand() * 0.1
            
            result = apply_radial_distortion(img, k1, k2, p1, p2)
            
            self.assertEqual(result.shape, img.shape)
            self.assertEqual(result.dtype, np.uint8)
            self.assertTrue(np.all(result >= 0))
            self.assertTrue(np.all(result <= 255))

    def test_radial_distortion_deterministic(self):
        """Same parameters should produce same output"""
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        k1, k2, p1, p2 = 0.1, 0.05, 0.01, 0.01
        
        result1 = apply_radial_distortion(img, k1, k2, p1, p2)
        result2 = apply_radial_distortion(img, k1, k2, p1, p2)
        
        np.testing.assert_array_equal(result1, result2)

    """ Motion Blur Tests """

    def test_motion_blur_kernel_size_one(self):
        """Kernel size 1 should produce identical output"""
        TESTS = 20
        
        for _ in range(TESTS):
            img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
            result = apply_motion_blur(img, kernel_size=1)
            
            np.testing.assert_array_equal(result, img)

    def test_motion_blur_properties(self):
        """Test motion blur preserves image properties"""
        TESTS = 100
        
        for _ in range(TESTS):
            h, w = np.random.randint(50, 200, 2)
            img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
            kernel_size = int(np.random.randint(1, 16))
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            result = apply_motion_blur(img, kernel_size)
            
            self.assertEqual(result.shape, img.shape)
            self.assertEqual(result.dtype, np.uint8)
            self.assertTrue(np.all(result >= 0))
            self.assertTrue(np.all(result <= 255))

    def test_motion_blur_reduces_variance(self):
        """Motion blur should reduce image variance"""
        TESTS = 50
        
        for _ in range(TESTS):
            img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
            kernel_size = int(np.random.randint(3, 16, 1)[0])
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            result = apply_motion_blur(img, kernel_size)
            
            # Blur should reduce variance (smoothing)
            if kernel_size > 1:
                self.assertLessEqual(np.var(result), np.var(img) * 1.1)

    def test_motion_blur_preserves_mean(self):
        """Motion blur should preserve mean intensity"""
        TESTS = 50
        
        for _ in range(TESTS):
            img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
            kernel_size = int(np.random.randint(3, 16))
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            result = apply_motion_blur(img, kernel_size)
            
            np.testing.assert_allclose(np.mean(result), np.mean(img), rtol=0.1)

    """ Discretization Tests """

    def test_discretization_max_levels(self):
        """256 levels should produce nearly identical output"""
        TESTS = 20
        
        for _ in range(TESTS):
            img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
            result = apply_discretization(img, levels=256)
            
            self.assertEqual(result.shape, img.shape)

    def test_discretization_properties(self):
        """Test discretization preserves image properties"""
        TESTS = 100
        
        for _ in range(TESTS):
            h, w = np.random.randint(50, 200, 2)
            img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
            levels = int(np.random.randint(1, 33))
            
            result = apply_discretization(img, levels)
            
            self.assertEqual(result.shape, img.shape)
            self.assertEqual(result.dtype, np.uint8)
            self.assertTrue(np.all(result >= 0))
            self.assertTrue(np.all(result <= 255))

    def test_discretization_reduces_unique_values(self):
        """Discretization should reduce number of unique values"""
        TESTS = 50
        
        for _ in range(TESTS):
            img = np.arange(256, dtype=np.uint8).reshape(1, 256, 1)
            img = np.repeat(img, 3, axis=2)
            levels = int(np.random.randint(2, 17))
            
            result = apply_discretization(img, levels)
            
            unique_before = len(np.unique(img[:, :, 0]))
            unique_after = len(np.unique(result[:, :, 0]))
            
            self.assertLessEqual(unique_after, unique_before)
            self.assertLessEqual(unique_after, levels + 5)

    def test_discretization_values_are_multiples(self):
        """Discretized values should be multiples of step size"""
        TESTS = 50
        
        for _ in range(TESTS):
            img = np.arange(256, dtype=np.uint8).reshape(1, 256, 1)
            img = np.repeat(img, 3, axis=2)
            levels = int(np.random.randint(2, 17))
            
            result = apply_discretization(img, levels)
            
            step = max(1, 256 // levels)
            unique_vals = np.unique(result[:, :, 0])
            
            for val in unique_vals:
                self.assertEqual(val % step, 0)

    def test_discretization_invalid_levels(self):
        """Invalid levels should default to minimum"""
        img = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        
        result_zero = apply_discretization(img, levels=0)
        result_neg = apply_discretization(img, levels=-5)
        
        self.assertEqual(result_zero.shape, img.shape)
        self.assertEqual(result_neg.shape, img.shape)

    """ Cross-function Property Tests """

    def test_all_functions_preserve_shape(self):
        """All noise functions should preserve image shape"""
        TESTS = 100
        
        for _ in range(TESTS):
            h, w = np.random.randint(50, 200, 2)
            img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
            
            results = [
                add_gaussian_noise(img, sigma=np.random.rand() * 20),
                add_salt_pepper_noise(img, np.random.rand() * 0.02, np.random.rand() * 0.02),
                apply_radial_distortion(img, np.random.rand() * 0.1, np.random.rand() * 0.05, 0, 0),
                apply_motion_blur(img, int(np.random.randint(1, 10)) * 2 + 1),
                apply_discretization(img, int(np.random.randint(1, 17)))
            ]
            
            for result in results:
                self.assertEqual(result.shape, img.shape)
                self.assertEqual(result.dtype, np.uint8)

    def test_all_functions_maintain_valid_range(self):
        """All noise functions should maintain valid uint8 range"""
        TESTS = 100
        
        for _ in range(TESTS):
            img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
            
            results = [
                add_gaussian_noise(img, sigma=np.random.rand() * 50),
                add_salt_pepper_noise(img, np.random.rand() * 0.1, np.random.rand() * 0.1),
                apply_radial_distortion(img, np.random.rand() * 0.2, np.random.rand() * 0.1, 0, 0),
                apply_motion_blur(img, int(np.random.randint(1, 10)) * 2 + 1),
                apply_discretization(img, int(np.random.randint(1, 33)))
            ]
            
            for result in results:
                self.assertTrue(np.all(result >= 0))
                self.assertTrue(np.all(result <= 255))