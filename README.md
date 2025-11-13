<div align="center">
  <h1>found-tools</h1>
</div>

<div align="center">
  <a href="https://github.com/UWCubeSat/found-tools/actions/workflows/ci.yml">
    <img src="https://github.com/UWCubeSat/found-tools/actions/workflows/ci.yml/badge.svg" alt="CI Status">
  </a>
  <a href="https://pypi.org/project/found-tools/">
    <img src="https://img.shields.io/pypi/v/found-tools" alt="PyPI Version">
  </a>
</div>

A collection of useful simulation and analyses tools to test [found]("https://github.com/UWCubeSat/found").

## Installation

You can install `found-tools` directly from PyPI:

```bash
pip install found-tools
```

## Usage

Once installed, you can use the command-line tools provided by the package. For example:

```bash
found-attitude --help
```

## Contributing

Interested in contributing to `found-tools`? Great! Follow these steps to set up your development environment.

### Developer Setup

The development setup uses an install script to ensure all necessary tools like `uv` and `just` are available.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/jlando/found-tools.git
    cd found-tools
    ```

2.  **Run the setup script:**
    For Windows users, please use [Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/install).

    The script supports Linux (with `apt-get`) and macOS (with `brew`).
    ```bash
    chmod +x install.sh
    ./install.sh
    ```
    This will install all required development dependencies.

### Testing

We use `just` to streamline common development tasks.

-   **Run all checks (Lint, Types, Tests):**
    ```bash
    just check-all
    ```

-   **Run tests:**
    ```bash
    just test
    ```

-   **Lint and format code:**
    ```bash
    just lint
    ```

### Pull Request & Deploying to PyPI

All work should be done on a feature branch. Before opening a pull request, you must increment the `version` in `pyproject.toml` on your feature branch. Follow semantic versioning:
- **PATCH** for backward-compatible bug fixes (e.g., `0.1.0` → `0.1.1`).
- **MINOR** for adding functionality in a backward-compatible manner (e.g., `0.1.0` → `0.2.0`).
- **MAJOR** for incompatible API changes.

Once your work is complete and the version is updated, open a pull request to merge into the `main` branch.

Deployment is automated via GitHub Actions using the `publish.yml` workflow. The workflow is triggered when a new tag starting with `v` (e.g., `v0.1.0`) is pushed to the `main` branch.

To release a new version after your pull request has been merged:
1.  Check out the `main` branch locally and pull the latest changes.
2.  Tag the release on the `main` branch:
    ```bash
    git tag vX.Y.Z
    ```
3.  Push the tag to GitHub:
    ```bash
    git push origin vX.Y.Z
    ```
This only works if you properly incremented the version in the `pyproject.toml` file before making the `pull-request`.
