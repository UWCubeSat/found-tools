import tempfile
import unittest
from unittest.mock import patch


class SimulationMainIntegrationTest(unittest.TestCase):
    def _run_main(self, argv):
        from limb.simulation.main import main

        with tempfile.TemporaryDirectory() as tmpdir:
            full_argv = argv + ["--output-folder", tmpdir]
            with patch("sys.argv", full_argv):
                with patch("limb.simulation.render.conic.process_simulation"):
                    main()

    def test_main_with_zero_edge_angle(self):
        self._run_main(
            [
                "limb_simulation",
                "--semi-axes",
                "10",
                "10",
                "10",
                "--earth-point-direction",
                "1",
                "0",
                "0",
                "--distance",
                "15",
                "--num-satellite-positions",
                "4",
                "--num-image-spins",
                "3",
                "--num-image-radials",
                "2",
                "--focal-length",
                "0.035",
                "--x-pixel-pitch",
                "5e-6",
                "--x-resolution",
                "1024",
                "--y-resolution",
                "1024",
            ]
        )

    def test_main_with_wgs84_earth_geometry(self):
        self._run_main(
            [
                "limb_simulation",
                "--earth-point-direction",
                "1",
                "0",
                "0",
                "--distance",
                "6800000",
                "--num-satellite-positions",
                "4",
                "--num-image-spins",
                "2",
                "--num-image-radials",
                "2",
                "--focal-length",
                "0.035",
                "--x-pixel-pitch",
                "5e-6",
                "--x-resolution",
                "1024",
                "--y-resolution",
                "1024",
            ]
        )


if __name__ == "__main__":
    unittest.main()
