"""Command-line entry point for the edge-point generator."""

from __future__ import annotations

import argparse
from typing import Sequence

from .projection import FilmEdgePoint, generate_edge_point


def _tuple(values: Sequence[float]) -> tuple[float, ...]:
    return tuple(float(component) for component in values)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Project a world-space coordinate into the film plane using a camera "
            "orientation expressed as a quaternion."
        )
    )
    parser.add_argument(
        "--point",
        nargs=3,
        metavar=("X", "Y", "Z"),
        type=float,
        required=True,
        help="World/cartesian coordinate of the target point.",
    )
    parser.add_argument(
        "--rotation",
        nargs=4,
        metavar=("QX", "QY", "QZ", "QW"),
        type=float,
        required=True,
        help="Camera orientation quaternion (x, y, z, w).",
    )
    parser.add_argument(
        "--focal-length",
        type=float,
        default=1.0,
        help="Film-plane distance from the camera origin (default: 1).",
    )
    return parser.parse_args(argv)


def format_edge_point(edge_point: FilmEdgePoint) -> str:
    """Render the FilmEdgePoint with full precision."""

    return f"Film edge point -> x: {edge_point.x:.15g}, y: {edge_point.y:.15g}"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    edge_point = generate_edge_point(
        point=_tuple(args.point),
        quaternion=_tuple(args.rotation),
        focal_length=float(args.focal_length),
    )
    print(format_edge_point(edge_point))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
