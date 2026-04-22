import io
import unittest
from contextlib import redirect_stdout

from found_CLI_tools.navigation_analysis import parametric


class ParametricCovarianceMainTest(unittest.TestCase):
    def test_main_prints_covariance_and_diagnostics(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            exit_code = parametric.main(
                [
                    "--body-radius",
                    "1.0",
                    "--position-vector",
                    "0.0",
                    "0.0",
                    "2.0",
                    "--sun-vector",
                    "0.0",
                    "1.0",
                    "0.0",
                    "--theta-max-deg",
                    "70.0",
                    "--sigma-x",
                    "0.5",
                    "--num-points",
                    "4",
                ]
            )

        output = buffer.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("Parametric covariance matrix:", output)
        self.assertIn("Diagnostics:", output)
        self.assertIn("slant_range = 2.000000", output)
