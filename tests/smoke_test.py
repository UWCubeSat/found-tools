#!/usr/bin/env python3
"""
Smoke test to verify the package is installed correctly.
Tests basic imports and CLI availability.
"""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


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

    # Test 3: Import the navigation analysis module
    try:
        from found_CLI_tools.navigation_analysis import generic, parametric, parametric_model  # noqa: F401

        print("Navigation analysis module imported successfully")
    except ImportError as e:
        print(f"Failed to import navigation analysis module: {e}")
        sys.exit(1)

    # Test 4: Check that the main function exists
    try:
        from found_CLI_tools.attitude.main import main as attitude_main

        assert callable(attitude_main)
        print("CLI entry point exists")
    except (ImportError, AssertionError) as e:
        print(f"CLI entry point not found: {e}")
        sys.exit(1)

    # Test 5: Import key classes
    try:
        from found_CLI_tools.attitude.transform import Attitude, DCM  # noqa: F401

        print("Core classes imported successfully")
    except ImportError as e:
        print(f"Failed to import core classes: {e}")
        sys.exit(1)

    # Test 6: Check navigation analysis entry point exists
    try:
        from found_CLI_tools.navigation_analysis.parametric import main as covariance_main

        assert callable(covariance_main)
        print("Covariance CLI entry point exists")
    except (ImportError, AssertionError) as e:
        print(f"Covariance CLI entry point not found: {e}")
        sys.exit(1)

    print("\nAll smoke tests passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
