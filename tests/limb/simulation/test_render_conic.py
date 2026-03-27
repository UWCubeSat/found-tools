"""Tests for limb.simulation.render.conic."""

import os
import tempfile
import unittest

import numpy as np
import torch

from limb.simulation.render import conic as render_conic


class TestAddSaltPepper(unittest.TestCase):
    def test_adds_noise(self):
        img = np.zeros((10, 10), dtype=np.uint8)
        out = render_conic._add_salt_pepper(img, salt_prob=0.1, pepper_prob=0.1)
        self.assertEqual(out.shape, img.shape)
        self.assertEqual(out.dtype, np.uint8)
        self.assertNotEqual(out.sum(), 0)

    def test_zero_probs_unchanged(self):
        img = np.ones((10, 10), dtype=np.uint8) * 128
        out = render_conic._add_salt_pepper(img, salt_prob=0, pepper_prob=0)
        np.testing.assert_array_equal(out, img)


class TestApplyDiscretization(unittest.TestCase):
    def test_reduces_levels(self):
        img = np.arange(256, dtype=np.uint8).reshape(16, 16)
        out = render_conic._apply_discretization(img, levels=8)
        self.assertEqual(out.dtype, np.uint8)
        unique = np.unique(out)
        self.assertLessEqual(len(unique), 8)

    def test_levels_one(self):
        img = np.array([[100]], dtype=np.uint8)
        out = render_conic._apply_discretization(img, levels=1)
        self.assertEqual(out[0, 0], 0)

    def test_levels_zero_capped_to_one(self):
        """Cover levels < 1 branch (line 32-33)."""
        img = np.array([[100]], dtype=np.uint8)
        out = render_conic._apply_discretization(img, levels=0)
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(out.shape, img.shape)


class TestApplyMotionBlur(unittest.TestCase):
    def test_returns_same_shape(self):
        img = np.random.randint(0, 256, (20, 20), dtype=np.uint8)
        out = render_conic._apply_motion_blur(img, kernel_size=5)
        self.assertEqual(out.shape, img.shape)

    def test_even_kernel_corrected(self):
        img = np.ones((10, 10), dtype=np.uint8)
        out = render_conic._apply_motion_blur(img, kernel_size=4)
        self.assertEqual(out.shape, img.shape)


class TestApplyGaussian(unittest.TestCase):
    def test_adds_noise(self):
        img = np.ones((10, 10), dtype=np.uint8) * 128
        out = render_conic._apply_gaussian(img, mean=0, sigma=5)
        self.assertEqual(out.shape, img.shape)
        self.assertEqual(out.dtype, np.uint8)
        self.assertTrue(np.any(out != 128))


class TestApplyNoisePipeline(unittest.TestCase):
    def test_empty_config_unchanged(self):
        img = np.ones((10, 10), dtype=np.uint8) * 100
        out = render_conic._apply_noise_pipeline(img, None)
        np.testing.assert_array_equal(out, img)

    def test_empty_dict_unchanged(self):
        img = np.ones((10, 10), dtype=np.uint8) * 100
        out = render_conic._apply_noise_pipeline(img, {})
        np.testing.assert_array_equal(out, img)

    def test_gaussian_key(self):
        img = np.ones((10, 10), dtype=np.uint8) * 100
        out = render_conic._apply_noise_pipeline(
            img, {"gaussian": {"mean": 0, "sigma": 1}}
        )
        self.assertEqual(out.shape, img.shape)

    def test_stars_key(self):
        img = np.ones((10, 10), dtype=np.uint8) * 100
        out = render_conic._apply_noise_pipeline(img, {"stars": {"prob": 0.01}})
        self.assertEqual(out.shape, img.shape)

    def test_discretization_key(self):
        img = np.ones((10, 10), dtype=np.uint8) * 100
        out = render_conic._apply_noise_pipeline(img, {"discretization": {"levels": 4}})
        self.assertEqual(out.shape, img.shape)

    def test_motion_blur_key(self):
        img = np.ones((10, 10), dtype=np.uint8) * 100
        out = render_conic._apply_noise_pipeline(
            img, {"motion_blur": {"kernel_size": 3}}
        )
        self.assertEqual(out.shape, img.shape)

    def test_dead_pixels_key(self):
        img = np.ones((10, 10), dtype=np.uint8) * 100
        out = render_conic._apply_noise_pipeline(
            img, {"dead_pixels": {"salt_prob": 0.01, "pepper_prob": 0.01}}
        )
        self.assertEqual(out.shape, img.shape)


class TestSaveImageWorker(unittest.TestCase):
    def test_save_without_noise(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.png")
            render_conic.save_image_worker((np.ones((4, 4)) * 0.5, path))
            self.assertTrue(os.path.isfile(path))

    def test_save_with_noise_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.png")
            render_conic.save_image_worker(
                (np.ones((4, 4)) * 0.5, path, {"gaussian": {"mean": 0, "sigma": 1}})
            )
            self.assertTrue(os.path.isfile(path))


class TestSideOfHyperbola(unittest.TestCase):
    def test_returns_tensor(self):
        A = torch.tensor([1.0])
        B = torch.tensor([0.0])
        C = torch.tensor([-1.0])
        D = torch.tensor([0.0])
        E = torch.tensor([0.0])
        out = render_conic.side_of_hyperbola(
            np.array(0.0), np.array(0.0), A, B, C, D, E
        )
        self.assertIsInstance(out, torch.Tensor)

    def test_numpy_input_converted(self):
        A = torch.tensor([1.0])
        B = torch.tensor([0.0])
        C = torch.tensor([-1.0])
        D = torch.tensor([0.0])
        E = torch.tensor([0.0])
        out = render_conic.side_of_hyperbola(
            np.array(0.0), np.array(0.0), A, B, C, D, E
        )
        self.assertIsInstance(out, torch.Tensor)


class TestQAnchorBodyInterior(unittest.TestCase):
    def test_behind_camera_uses_conic_center(self):
        """Earth at -x (behind) uses Q at ellipse center, not bogus projection."""
        device = torch.device("cpu")
        dtype = torch.float64
        rc = torch.tensor([[1.0, 0.0, 0.0]], device=device, dtype=dtype)
        K = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
        A1 = torch.tensor([1.0], dtype=dtype)
        B1 = torch.zeros(1, dtype=dtype)
        C1 = torch.tensor([1.0], dtype=dtype)
        D1 = torch.zeros(1, dtype=dtype)
        E1 = torch.zeros(1, dtype=dtype)
        F1 = torch.tensor([-4.0], dtype=dtype)
        qa = render_conic._q_anchor_body_interior(rc, K, A1, B1, C1, D1, E1, F1)
        self.assertAlmostEqual(float(qa.item()), -4.0, places=5)


class TestProcessSimulation(unittest.TestCase):
    def test_process_simulation_sigma_non_positive_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            coeffs = torch.randn(1, 6)
            K = torch.eye(3).unsqueeze(0)
            rc = torch.randn(1, 3)
            with self.assertRaises(ValueError) as ctx:
                render_conic.process_simulation(
                    coeffs,
                    width=8,
                    height=8,
                    output_folder=tmpdir,
                    K=K,
                    rc=rc,
                    sigma=0.0,
                )
            self.assertIn("sigma", str(ctx.exception).lower())

    def test_process_simulation_writes_images(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            n = 4
            coeffs = torch.randn(n, 6)
            coeffs[:, 0] = torch.abs(coeffs[:, 0]) + 0.001
            coeffs[:, 2] = torch.abs(coeffs[:, 2]) + 0.001
            coeffs[:, 5] = -0.5
            K = torch.eye(3).unsqueeze(0).expand(n, 3, 3).clone()
            K[:, 0, 1] = -100.0
            K[:, 1, 2] = -100.0
            rc = torch.randn(n, 3)
            render_conic.process_simulation(
                coeffs,
                width=32,
                height=32,
                output_folder=tmpdir,
                K=K,
                rc=rc,
                batch_size=2,
                sigma=1.0,
                row_indices=np.arange(n),
                noise_config=None,
            )
            for i in range(n):
                self.assertTrue(
                    os.path.isfile(os.path.join(tmpdir, f"img_{i:06d}.png"))
                )

    def test_process_simulation_without_row_indices(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            n = 2
            coeffs = torch.randn(n, 6)
            coeffs[:, 0] = 1.0
            coeffs[:, 2] = 1.0
            coeffs[:, 5] = -1.0
            K = torch.eye(3).unsqueeze(0).expand(n, 3, 3).clone()
            rc = torch.randn(n, 3)
            render_conic.process_simulation(
                coeffs,
                width=16,
                height=16,
                output_folder=tmpdir,
                K=K,
                rc=rc,
                batch_size=10,
                sigma=1.0,
                row_indices=None,
            )
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "img_000000.png")))
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "img_000001.png")))

    def test_process_simulation_row_indices_length_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            coeffs = torch.randn(3, 6)
            K = torch.eye(3).unsqueeze(0).expand(3, 3, 3)
            rc = torch.randn(3, 3)
            with self.assertRaises(ValueError) as ctx:
                render_conic.process_simulation(
                    coeffs,
                    width=16,
                    height=16,
                    output_folder=tmpdir,
                    K=K,
                    rc=rc,
                    row_indices=np.array([0, 1]),  # length 2 != 3
                )
            self.assertIn("row_indices", str(ctx.exception))

    def test_process_simulation_with_noise_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            n = 2
            coeffs = torch.randn(n, 6)
            coeffs[:, 0] = 1.0
            coeffs[:, 2] = 1.0
            coeffs[:, 5] = -1.0
            K = torch.eye(3).unsqueeze(0).expand(n, 3, 3).clone()
            rc = torch.randn(n, 3)
            render_conic.process_simulation(
                coeffs,
                width=16,
                height=16,
                output_folder=tmpdir,
                K=K,
                rc=rc,
                row_indices=np.arange(n),
                noise_config={"gaussian": {"mean": 0, "sigma": 2}},
            )
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "img_000000.png")))


if __name__ == "__main__":
    unittest.main()
