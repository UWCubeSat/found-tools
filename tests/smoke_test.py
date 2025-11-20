#!/usr/bin/env python3
"""
Smoke test to verify the package is installed correctly.
Tests basic imports and CLI availability.
"""

import sys


def main():
    print("Running smoke test...")

    # Test 1: Import the package
    try:
        import found_CLI_tools  # noqa: F401

        print("Package imported successfully")
    except ImportError as e:
        print(f"Failed to import package: {e}")
        sys.exit(1)

    # Test 2: Import the attitude module
    try:
        from found_CLI_tools.attitude import main, transform  # noqa: F401

        print("Attitude module imported successfully")
    except ImportError as e:
        print(f"Failed to import attitude module: {e}")
        sys.exit(1)

    # Test 2b: Import the edge point generator
    try:
        from found_CLI_tools import edge_point_gen  # noqa: F401

        print("Edge point generator imported successfully")
    except ImportError as e:
        print(f"Failed to import edge point generator: {e}")
        sys.exit(1)

    # Test 3: Check that the main function exists
    try:
        from found_CLI_tools.attitude.main import main as attitude_main

        assert callable(attitude_main)
        print("CLI entry point exists")
    except (ImportError, AssertionError) as e:
        print(f"CLI entry point not found: {e}")
        sys.exit(1)

    # Test 4: Import key classes
    try:
        from found_CLI_tools.attitude.transform import Attitude, DCM  # noqa: F401
        from found_CLI_tools.edge_point_gen.projection import (  # noqa: F401
            FilmEdgePoint,
            generate_edge_point,
        )

        print("Core classes imported successfully")
    except ImportError as e:
        print(f"Failed to import core classes: {e}")
        sys.exit(1)

    print("\nAll smoke tests passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
