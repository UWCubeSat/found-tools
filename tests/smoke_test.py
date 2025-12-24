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

    # Test 3: Check that the attitude CLI entry point exists
    try:
        from found_CLI_tools.attitude.main import main as attitude_main

        assert callable(attitude_main)
        print("Attitude CLI entry point exists")
    except (ImportError, AssertionError) as e:
        print(f"Attitude CLI entry point not found: {e}")
        sys.exit(1)

    # Test 4: Import key attitude classes
    try:
        from found_CLI_tools.attitude.transform import Attitude, DCM  # noqa: F401

        print("Attitude core classes imported successfully")
    except ImportError as e:
        print(f"Failed to import attitude core classes: {e}")
        sys.exit(1)

    # Test 5: Import generator module and CLI
    try:
        from found_CLI_tools import generator  # noqa: F401
        from found_CLI_tools.generator.__main__ import (  # type: ignore[attr-defined]
            parse_args,
        )
        from found_CLI_tools.generator.spatial.coordinate import (  # noqa: F401
            Vector,
            Attitude,
        )
        from found_CLI_tools.generator.spatial.camera import Camera  # noqa: F401
        from found_CLI_tools.generator.curve.spherical import (  # noqa: F401
            SphericalCurveProvider,
        )
        from found_CLI_tools.generator.image.printer import Printer  # noqa: F401

        assert callable(parse_args)
        print("Generator module, CLI, and core classes imported successfully")
    except (ImportError, AssertionError) as e:
        print(f"Failed to import generator CLI or core classes: {e}")
        sys.exit(1)

    # Test 6: Import noise generator image module and helpers
    try:
        from found_CLI_tools import noise_generator_image  # noqa: F401
        from found_CLI_tools.noise_generator_image import noise  # noqa: F401
        from found_CLI_tools.noise_generator_image.__main__ import (  # type: ignore[attr-defined]
            interactive_noise_adjustment,
            main as noise_cli_main,
        )

        assert callable(interactive_noise_adjustment)
        assert callable(noise_cli_main)
        print("Noise generator image module and CLI entry point imported successfully")
    except (ImportError, AssertionError) as e:
        print(f"Failed to import noise generator image CLI: {e}")
        sys.exit(1)

    print("\nAll smoke tests passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
