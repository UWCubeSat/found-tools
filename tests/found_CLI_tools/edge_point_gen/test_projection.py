import unittest

from scipy.spatial.transform import Rotation

from found_CLI_tools.edge_point_gen.projection import FilmEdgePoint, generate_edge_point


class ProjectionTest(unittest.TestCase):
    def test_identity_projection(self):
        result = generate_edge_point((0.0, 0.0, 10.0), (0.0, 0.0, 0.0, 1.0), 35.0)

        self.assertIsInstance(result, FilmEdgePoint)
        self.assertAlmostEqual(0.0, result.x)
        self.assertAlmostEqual(0.0, result.y)

    def test_projection_scales_with_depth(self):
        result = generate_edge_point((2.0, -1.0, 20.0), (0.0, 0.0, 0.0, 1.0), 50.0)

        self.assertAlmostEqual(5.0, result.x)
        self.assertAlmostEqual(-2.5, result.y)

    def test_projection_respects_rotation(self):
        yaw_90 = Rotation.from_euler("y", 90, degrees=True).as_quat()
        result = generate_edge_point((10.0, 0.0, 0.0), yaw_90, 20.0)

        self.assertAlmostEqual(0.0, result.x, places=7)
        self.assertAlmostEqual(0.0, result.y, places=7)

    def test_point_behind_camera_rejected(self):
        with self.assertRaises(ValueError):
            generate_edge_point((0.0, 0.0, -5.0), (0.0, 0.0, 0.0, 1.0), 35.0)

    def test_invalid_focal_length_rejected(self):
        with self.assertRaises(ValueError):
            generate_edge_point((0.0, 0.0, 5.0), (0.0, 0.0, 0.0, 1.0), 0.0)

    def test_invalid_vector_length_rejected(self):
        with self.assertRaises(ValueError):
            generate_edge_point((0.0, 1.0), (0.0, 0.0, 0.0, 1.0), 10.0)

        with self.assertRaises(ValueError):
            generate_edge_point((0.0, 1.0, 5.0), (0.0, 0.0, 1.0), 10.0)

    def test_invalid_vector_type_rejected(self):
        with self.assertRaises(TypeError):
            generate_edge_point(0.5, (0.0, 0.0, 0.0, 1.0), 10.0)  # type: ignore[arg-type]

        with self.assertRaises(TypeError):
            generate_edge_point((0.0, 1.0, 5.0), None, 10.0)  # type: ignore[arg-type]
