# Run tests
[group('qa')]
test *args:
    uv run -m pytest {{ args }}

_cov *args:
    uv run -m coverage {{ args }}


# Run tests and measure coverage
[group('qa')]
@cov: 
    just _cov erase
    just _cov run -m pytest tests
    just _cov html
    just _cov report --fail-under=100

# Run linters
[group('qa')]
lint:
    uvx ruff check
    uvx ruff format

# Check types
[group('qa')]
typing:
    uvx ty check --python .venv src

# Perform all checks
[group('qa')]
check-all: lint cov typing

# Remove all __pycache__ folders
[group('maintenance')]
clean-pycache:
    find . -type d -name "__pycache__" -prune -exec rm -rf {} +