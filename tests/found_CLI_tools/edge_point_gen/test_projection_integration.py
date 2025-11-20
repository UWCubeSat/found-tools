import io
import unittest
from contextlib import redirect_stdout

from found_CLI_tools.edge_point_gen import main as edge_cli


class ProjectionIntegrationTest(unittest.TestCase):
    def test_parse_args_handles_basic_values(self):
        args = edge_cli.parse_args(
            [
                "--point",
                "0",
                "1",
                "2",
                "--rotation",
                "0",
                "0",
                "0",
                "1",
                "--focal-length",
                "10",
            ]
        )

        self.assertEqual([0.0, 1.0, 2.0], args.point)
        self.assertEqual([0.0, 0.0, 0.0, 1.0], args.rotation)
        self.assertEqual(10.0, args.focal_length)

    def test_main_outputs_precise_edge_point(self):
        argv = [
            "--point",
            "0",
            "0",
            "5",
            "--rotation",
            "0",
            "0",
            "0",
            "1",
            "--focal-length",
            "25",
        ]
        buffer = io.StringIO()

        with redirect_stdout(buffer):
            exit_code = edge_cli.main(argv)

        self.assertEqual(0, exit_code)
        output = buffer.getvalue().strip()
        self.assertIn("Film edge point", output)
        self.assertIn("x: 0", output)
        self.assertIn("y: 0", output)
