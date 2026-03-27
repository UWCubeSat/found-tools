"""Tests for limb.simulation.main (CLI and validation)."""

import tempfile
import unittest
from unittest.mock import patch


def _minimal_argv():
    return [
        "limb_simulation",
        "--fovs",
        "70",
        "--resolutions",
        "64",
        "--distances",
        "6800000",
        "--num-positions-per-point",
        "2",
        "--num-spins-per-position",
        "2",
        "--num-radials-per-spin",
        "2",
        "--output-csv",
        "sim_metadata.csv",
    ]


class SimulationMainTest(unittest.TestCase):
    def _run_main(self, argv, mock_render=True):
        from limb.simulation.main import main

        with tempfile.TemporaryDirectory() as tmpdir:
            full_argv = argv + ["--output-folder", tmpdir]
            with patch("sys.argv", full_argv):
                if mock_render:
                    with patch("limb.simulation.main.render_conic.process_simulation"):
                        main()
                else:
                    main()

    def test_main_minimal(self):
        """Main runs with minimal required args and mocked render."""
        self._run_main(_minimal_argv())

    def test_main_with_semi_axes(self):
        self._run_main(
            _minimal_argv() + ["--semi-axes", "6378137", "6378137", "6356752.31424518"]
        )

    def test_main_with_seed(self):
        self._run_main(_minimal_argv() + ["--seed", "42"])

    def test_main_with_noise_options(self):
        self._run_main(
            _minimal_argv()
            + [
                "--noise-gaussian",
                "0",
                "10",
                "--noise-stars",
                "0.005",
                "--noise-discretization",
                "8",
                "--noise-motion-blur",
                "5",
                "--noise-dead-pixels",
                "0.01",
                "0.01",
            ]
        )

    def test_main_noise_all_zero_no_noise_config_passed(self):
        """0/0 gaussian and disabled flags yield no sensor pipeline."""
        from limb.simulation.main import main

        with tempfile.TemporaryDirectory() as tmpdir:
            full_argv = _minimal_argv() + [
                "--output-folder",
                tmpdir,
                "--noise-gaussian",
                "0",
                "0",
                "--noise-stars",
                "0",
                "--noise-discretization",
                "0",
                "--noise-motion-blur",
                "0",
                "--noise-dead-pixels",
                "0",
                "0",
                "--sigma",
                "2",
            ]
            with patch("sys.argv", full_argv):
                with patch("limb.simulation.main.render_conic.process_simulation") as m:
                    main()
        for call in m.call_args_list:
            self.assertIsNone(call.kwargs.get("noise_config"))

    def test_main_with_batch_size_and_sigma(self):
        self._run_main(_minimal_argv() + ["--batch-size", "10", "--sigma", "1.5"])


class SimulationValidationTest(unittest.TestCase):
    """Test _parse_args and _validate_args for error paths."""

    def _parse(self, argv):
        from limb.simulation.main import _parse_args

        with patch("sys.argv", argv):
            return _parse_args()

    def _validate(self, args):
        from limb.simulation.main import _validate_args

        return _validate_args(args)

    def test_valid_args_pass(self):
        args = self._parse(_minimal_argv())
        self._validate(args)  # no raise

    def test_semi_axes_non_positive(self):
        argv = _minimal_argv() + ["--semi-axes", "1", "1", "0"]
        args = self._parse(argv)
        with self.assertRaises(ValueError) as ctx:
            self._validate(args)
        self.assertIn("semi-axes", str(ctx.exception))

    def test_fovs_empty_or_non_positive(self):
        argv = _minimal_argv().copy()
        argv[argv.index("70")] = "0"
        args = self._parse(argv)
        with self.assertRaises(ValueError) as ctx:
            self._validate(args)
        self.assertIn("fovs", str(ctx.exception))

    def test_resolutions_invalid(self):
        argv = _minimal_argv().copy()
        argv[argv.index("64")] = "0"
        args = self._parse(argv)
        with self.assertRaises(ValueError) as ctx:
            self._validate(args)
        self.assertIn("resolutions", str(ctx.exception))

    def test_distances_non_positive(self):
        argv = _minimal_argv().copy()
        argv[argv.index("6800000")] = "-1"
        args = self._parse(argv)
        with self.assertRaises(ValueError) as ctx:
            self._validate(args)
        self.assertIn("distances", str(ctx.exception))

    def test_num_earth_points_invalid(self):
        argv = _minimal_argv() + ["--num-earth-points", "0"]
        args = self._parse(argv)
        with self.assertRaises(ValueError) as ctx:
            self._validate(args)
        self.assertIn("num-earth-points", str(ctx.exception))

    def test_num_positions_per_point_invalid(self):
        argv = [
            "limb_simulation",
            "--fovs",
            "70",
            "--resolutions",
            "64",
            "--distances",
            "6800000",
            "--num-positions-per-point",
            "0",
            "--num-spins-per-position",
            "2",
            "--num-radials-per-spin",
            "2",
            "--output-csv",
            "out.csv",
        ]
        args = self._parse(argv)
        with self.assertRaises(ValueError) as ctx:
            self._validate(args)
        self.assertIn("num-positions-per-point", str(ctx.exception))

    def test_batch_size_invalid(self):
        argv = _minimal_argv() + ["--batch-size", "0"]
        args = self._parse(argv)
        with self.assertRaises(ValueError) as ctx:
            self._validate(args)
        self.assertIn("batch-size", str(ctx.exception))

    def test_sigma_non_positive(self):
        for bad in ("-1", "0"):
            with self.subTest(sigma=bad):
                argv = _minimal_argv() + ["--sigma", bad]
                args = self._parse(argv)
                with self.assertRaises(ValueError) as ctx:
                    self._validate(args)
                self.assertIn("sigma", str(ctx.exception).lower())

    def test_noise_stars_out_of_range(self):
        argv = _minimal_argv() + ["--noise-stars", "1.5"]
        args = self._parse(argv)
        with self.assertRaises(ValueError) as ctx:
            self._validate(args)
        self.assertIn("noise-stars", str(ctx.exception))

    def test_noise_discretization_zero_passes(self):
        argv = _minimal_argv() + ["--noise-discretization", "0"]
        args = self._parse(argv)
        self._validate(args)

    def test_noise_discretization_negative(self):
        argv = _minimal_argv() + ["--noise-discretization", "-1"]
        args = self._parse(argv)
        with self.assertRaises(ValueError) as ctx:
            self._validate(args)
        self.assertIn("noise-discretization", str(ctx.exception))

    def test_noise_motion_blur_even(self):
        argv = _minimal_argv() + ["--noise-motion-blur", "4"]
        args = self._parse(argv)
        with self.assertRaises(ValueError) as ctx:
            self._validate(args)
        self.assertIn("noise-motion-blur", str(ctx.exception))

    def test_noise_motion_blur_zero_passes(self):
        argv = _minimal_argv() + ["--noise-motion-blur", "0"]
        args = self._parse(argv)
        self._validate(args)

    def test_noise_gaussian_negative_sigma(self):
        argv = _minimal_argv() + ["--noise-gaussian", "0", "-1"]
        args = self._parse(argv)
        with self.assertRaises(ValueError) as ctx:
            self._validate(args)
        self.assertIn("noise-gaussian", str(ctx.exception).lower())

    def test_num_spins_per_position_invalid(self):
        argv = _minimal_argv().copy()
        idx = argv.index("--num-spins-per-position")
        argv[idx + 1] = "0"
        args = self._parse(argv)
        with self.assertRaises(ValueError) as ctx:
            self._validate(args)
        self.assertIn("num-spins-per-position", str(ctx.exception))

    def test_num_radials_per_spin_invalid(self):
        argv = _minimal_argv().copy()
        idx = argv.index("--num-radials-per-spin")
        argv[idx + 1] = "0"
        args = self._parse(argv)
        with self.assertRaises(ValueError) as ctx:
            self._validate(args)
        self.assertIn("num-radials-per-spin", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
