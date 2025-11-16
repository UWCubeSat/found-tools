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
    # Ensure ASGI entrypoint is importable.
    # You can also use coverage to run your CLI entrypoints.
    # just _cov run -m hello_svc.asgi
    # just _cov combine
    just _cov report
    just _cov html

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