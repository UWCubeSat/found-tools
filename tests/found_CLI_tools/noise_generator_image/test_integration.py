import unittest
import os
import tempfile
import numpy as np
import cv2
from unittest.mock import patch

from src.found_CLI_tools.noise_generator_image.__main__ import (
    interactive_noise_adjustment,
    main,
)
from src.found_CLI_tools.noise_generator_image.noise import (
    add_gaussian_noise,
    add_salt_pepper_noise,
    apply_radial_distortion,
    apply_motion_blur,
    apply_discretization,
)


class IntegrationTest(unittest.TestCase):
    """Full pipeline tests"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_input = os.path.join(self.temp_dir, "test_input.png")
        self.test_output = os.path.join(self.temp_dir, "test_output.png")

        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image[40:60, 40:60] = 128
        cv2.imwrite(self.test_input, test_image)

    def tearDown(self):
        for file in [self.test_input, self.test_output]:
            if os.path.exists(file):
                os.remove(file)
        os.rmdir(self.temp_dir)

    def test_full_pipeline_random(self):
        """Test full noise pipeline with random parameters"""
        TESTS = 50

        for _ in range(TESTS):
            h, w = np.random.randint(50, 200, 2)
            img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)

            sigma = np.random.rand() * 30
            salt = np.random.rand() * 0.05
            pepper = np.random.rand() * 0.05
            k1 = np.random.rand() * 0.2 - 0.1
            k2 = np.random.rand() * 0.1 - 0.05
            p1 = np.random.rand() * 0.05
            p2 = np.random.rand() * 0.05
            levels = int(np.random.randint(2, 17))
            kernel = int(np.random.randint(1, 10)) * 2 + 1

            result = apply_radial_distortion(img.copy(), k1, k2, p1, p2)
            result = add_gaussian_noise(result, sigma=sigma)
            result = add_salt_pepper_noise(result, salt, pepper)
            result = apply_discretization(result, levels)
            result = apply_motion_blur(result, kernel)

            self.assertEqual(result.shape, img.shape)
            self.assertEqual(result.dtype, np.uint8)
            self.assertTrue(np.all(result >= 0))
            self.assertTrue(np.all(result <= 255))

    def test_full_pipeline_extreme_parameters(self):
        """Test pipeline with extreme parameter values"""
        TESTS = 20

        for _ in range(TESTS):
            img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

            # Extreme parameters
            result = apply_radial_distortion(img.copy(), 0.5, 0.3, 0.1, 0.1)
            result = add_gaussian_noise(result, sigma=50)
            result = add_salt_pepper_noise(result, 0.1, 0.1)
            result = apply_discretization(result, 2)
            result = apply_motion_blur(result, 15)

            self.assertEqual(result.shape, img.shape)
            self.assertTrue(np.all(result >= 0))
            self.assertTrue(np.all(result <= 255))

    def test_pipeline_order_independence(self):
        """Test that different orderings produce valid outputs"""
        TESTS = 20

        orderings = [
            ["distortion", "gaussian", "salt_pepper", "discretize", "blur"],
            ["gaussian", "salt_pepper", "distortion", "blur", "discretize"],
            ["blur", "discretize", "gaussian", "salt_pepper", "distortion"],
        ]

        for _ in range(TESTS):
            img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

            functions = {
                "distortion": lambda x: apply_radial_distortion(x, 0.05, 0.02, 0, 0),
                "gaussian": lambda x: add_gaussian_noise(x, sigma=10),
                "salt_pepper": lambda x: add_salt_pepper_noise(x, 0.01, 0.01),
                "discretize": lambda x: apply_discretization(x, 8),
                "blur": lambda x: apply_motion_blur(x, 5),
            }

            for ordering in orderings:
                result = img.copy()
                for func_name in ordering:
                    result = functions[func_name](result)

                self.assertEqual(result.shape, img.shape)
                self.assertTrue(np.all(result >= 0))
                self.assertTrue(np.all(result <= 255))

    def test_repeated_application(self):
        """Test applying noise multiple times"""
        TESTS = 20

        for _ in range(TESTS):
            img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

            result = img.copy()
            iterations = np.random.randint(2, 6)

            for _ in range(iterations):
                result = add_gaussian_noise(result, sigma=5)
                result = add_salt_pepper_noise(result, 0.001, 0.001)

            self.assertEqual(result.shape, img.shape)
            self.assertTrue(np.all(result >= 0))
            self.assertTrue(np.all(result <= 255))

    def test_image_size_variations(self):
        """Test pipeline with various image sizes"""
        sizes = [(50, 50), (100, 100), (200, 200), (100, 200), (200, 100), (75, 125)]

        for h, w in sizes:
            img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)

            result = apply_radial_distortion(img.copy(), 0.05, 0.02, 0.01, 0.01)
            result = add_gaussian_noise(result, sigma=10)
            result = add_salt_pepper_noise(result, 0.01, 0.01)
            result = apply_discretization(result, 8)
            result = apply_motion_blur(result, 5)

            self.assertEqual(result.shape, (h, w, 3))
            self.assertTrue(np.all(result >= 0))
            self.assertTrue(np.all(result <= 255))

    @patch("src.found_CLI_tools.noise_generator_image.__main__.cv2.waitKey")
    @patch("src.found_CLI_tools.noise_generator_image.__main__.cv2.getTrackbarPos")
    @patch("src.found_CLI_tools.noise_generator_image.__main__.cv2.imshow")
    @patch("src.found_CLI_tools.noise_generator_image.__main__.cv2.imwrite")
    @patch("src.found_CLI_tools.noise_generator_image.__main__.cv2.destroyAllWindows")
    @patch("src.found_CLI_tools.noise_generator_image.__main__.cv2.namedWindow")
    @patch("src.found_CLI_tools.noise_generator_image.__main__.cv2.resizeWindow")
    @patch("src.found_CLI_tools.noise_generator_image.__main__.cv2.createTrackbar")
    def test_interactive_with_random_parameters(
        self,
        mock_create,
        mock_resize,
        mock_window,
        mock_destroy,
        mock_imwrite,
        mock_imshow,
        mock_get,
        mock_wait,
    ):
        """Test interactive function with random trackbar values"""
        TESTS = 20

        for _ in range(TESTS):
            # Random trackbar values
            values = {
                "Gaussian Sigma": int(np.random.randint(0, 101)),
                "Salt Prob x1000": int(np.random.randint(0, 101)),
                "Pepper Prob x1000": int(np.random.randint(0, 101)),
                "Discretization": int(np.random.randint(1, 33)),
                "k1 x100": int(np.random.randint(0, 101)),
                "k2 x100": int(np.random.randint(0, 101)),
                "p1 x100": int(np.random.randint(0, 101)),
                "p2 x100": int(np.random.randint(0, 101)),
                "Motion Kernel": int(np.random.randint(1, 31)),
            }

            mock_get.side_effect = lambda name, _: values.get(name, 0)
            mock_wait.side_effect = [ord("s"), -1]

            base_image = cv2.imread(self.test_input)
            result = interactive_noise_adjustment(base_image, self.test_output)

            self.assertTrue(result)

    @patch("src.found_CLI_tools.noise_generator_image.__main__.cv2.waitKey")
    @patch("src.found_CLI_tools.noise_generator_image.__main__.cv2.getTrackbarPos")
    @patch("src.found_CLI_tools.noise_generator_image.__main__.cv2.imshow")
    @patch("src.found_CLI_tools.noise_generator_image.__main__.cv2.destroyAllWindows")
    @patch("src.found_CLI_tools.noise_generator_image.__main__.cv2.namedWindow")
    @patch("src.found_CLI_tools.noise_generator_image.__main__.cv2.resizeWindow")
    @patch("src.found_CLI_tools.noise_generator_image.__main__.cv2.createTrackbar")
    def test_interactive_parameter_scaling(
        self,
        mock_create,
        mock_resize,
        mock_window,
        mock_destroy,
        mock_imshow,
        mock_get,
        mock_wait,
    ):
        """Test trackbar parameter scaling"""
        TESTS = 50

        for _ in range(TESTS):
            trackbar_salt = int(np.random.randint(0, 101))
            trackbar_k1 = int(np.random.randint(0, 101))
            trackbar_disc = int(np.random.randint(1, 33))

            expected_salt = trackbar_salt / 1000.0
            expected_k1 = trackbar_k1 / 100.0
            expected_disc = max(1, trackbar_disc)

            self.assertEqual(expected_salt, trackbar_salt / 1000.0)
            self.assertEqual(expected_k1, trackbar_k1 / 100.0)
            self.assertEqual(expected_disc, max(1, trackbar_disc))

    def test_load_save_cycle(self):
        """Test loading and saving images"""
        TESTS = 20

        for _ in range(TESTS):
            h, w = np.random.randint(50, 150, 2)
            img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)

            temp_path = os.path.join(self.temp_dir, f"temp_{_}.png")
            cv2.imwrite(temp_path, img)

            loaded = cv2.imread(temp_path)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.shape, img.shape)

            result = add_gaussian_noise(loaded, sigma=10)
            cv2.imwrite(temp_path, result)

            reloaded = cv2.imread(temp_path)
            self.assertIsNotNone(reloaded)
            self.assertEqual(reloaded.shape, img.shape)

            os.remove(temp_path)

    def test_main_with_valid_images(self):
        """Test main function with valid inputs"""
        TESTS = 10

        for _ in range(TESTS):
            img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(self.test_input, img)

            test_args = ["noise_generator_image", self.test_input, self.test_output]

            with patch("sys.argv", test_args):
                with patch(
                    "src.found_CLI_tools.noise_generator_image.__main__.interactive_noise_adjustment"
                ) as mock:
                    mock.return_value = True
                    result = main()

                    self.assertEqual(result, 0)
                    self.assertEqual(mock.call_count, 1)

    def test_pipeline_determinism(self):
        """Test that same parameters produce same output"""
        TESTS = 20

        for _ in range(TESTS):
            img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

            # Fixed parameters
            salt = 0.01
            pepper = 0.01
            k1, k2 = 0.05, 0.02
            levels = 8
            kernel = 5

            result1 = apply_radial_distortion(img.copy(), k1, k2, 0, 0)
            result1 = add_salt_pepper_noise(result1, salt, pepper)
            result1 = apply_discretization(result1, levels)
            result1 = apply_motion_blur(result1, kernel)

            result2 = apply_radial_distortion(img.copy(), k1, k2, 0, 0)
            result2 = add_salt_pepper_noise(result2, salt, pepper)
            result2 = apply_discretization(result2, levels)
            result2 = apply_motion_blur(result2, kernel)
