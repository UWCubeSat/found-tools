# found-tools

Python tools (`uv` + `just`) for testing [found](https://github.com/UWCubeSat/found).

## Before pushing or opening a PR

**Run `just check-all` locally and make sure it is fully green before pushing
or opening a PR.** CI enforces a subset of the same checks (see below) and
will fail the same way `just check-all` does locally -- there is no reason to
push and wait for CI to tell you what you could have caught locally.

```bash
just check-all
```

If it doesn't pass locally, don't push.

## What `just check-all` actually runs

`check-all` is `lint`, `cov`, then `typing` (see `justfile`), in that order:

- **`just lint`** -- `uvx ruff check` then `uvx ruff format` (the local
  formatter, no `--check`, so it will silently reformat files in place; see
  the CI mismatch note below).
- **`just cov`** -- `uv run coverage erase`, then
  `uv run coverage run -m pytest tests`, `coverage html`, then
  `coverage report --fail-under=100`. **Coverage must be 100%** (see
  `[tool.coverage.report] fail_under = 100` in `pyproject.toml`). Lines that
  genuinely can't be covered (CLI entry points, `bpy`-only code) are marked
  `# pragma: no cover` or excluded via `[tool.coverage.run] omit`, not left
  uncovered.
- **`just typing`** -- `uvx ty check --python .venv src` (Astral's `ty` type
  checker). **This step does not run in CI** (see below) -- it's local-only,
  so don't skip it just because CI is green.

Run any single piece directly when iterating, e.g. `just test`,
`just lint`, `just typing`.

## What GitHub Actions actually runs

Two workflows under `.github/workflows/`:

- **`ci.yml`** (`CI`) -- runs on every push to `main` and every PR into
  `main`. On `ubuntu-latest`, it:
  1. `uv tool install ruff@latest`
  2. `uv python install` (reads `.python-version`)
  3. `uv sync --locked --all-extras --dev` -- **`--locked` means CI fails
     outright if `uv.lock` is stale relative to `pyproject.toml`.** Run
     `uv lock` (or just `uv sync`) locally after touching dependencies and
     commit the updated `uv.lock`.
  4. `uv run ruff check` and `uv run ruff format --check`
  5. `uv run coverage erase`, `coverage run -m pytest tests`,
     `coverage html`, `coverage report --fail-under=100`
  6. Uploads the `htmlcov/` coverage report as a build artifact
  - **CI does not run `ty check`.** A branch can be green on GitHub Actions
    with type errors that only `just typing` / `just check-all` would catch.
  - **CI does not install Blender or run a real render smoke test** beyond
    whatever `pytest` covers -- the `bpy`-based tests in
    `tests/found_tools/render/test_blender_scene.py` run for real (no
    Blender is required separately; `bpy` is a project dependency, pinned
    in `uv.lock`), but nothing in CI actually opens the rendered PNG.
- **`publish.yml`** (`Publish`) -- runs on a successful `CI` run whose
  triggering push was a `v*` tag, or directly on `push` of a `v*` tag.
  Builds with `uv build`, smoke-tests the wheel and sdist, then
  `uv publish`s to PyPI. Not relevant to day-to-day PRs; only matters when
  cutting a release per the README's tagging instructions.

## Known CI/local mismatches (things that pass locally but fail in CI, or vice versa)

- **`ruff` version drift.** CI always installs `ruff@latest` fresh
  (`uv tool install ruff@latest`), so its rule set can be newer than
  whatever `uvx ruff` resolves to locally out of a stale `uvx` cache. This
  has already caused a real CI failure (25 lint errors -- `DTZ001`, `UP017`,
  `C408`, import sorting -- that a locally-cached older `ruff` didn't catch).
  Before pushing, force the same "always fetch latest" behavior CI uses:
  ```bash
  uvx ruff@latest check
  uvx ruff@latest format --check
  ```
  (`just lint` uses a bare `uvx ruff`, which can silently use a cached
  version -- don't rely on it alone to match CI.)
- **`just lint` reformats; CI only checks.** `just lint`'s `uvx ruff format`
  (no `--check`) will happily rewrite files to fix formatting. CI's
  `ruff format --check` just fails if anything *would* be reformatted. Run
  `just lint` (or `ruff format`) and let it fix things, then re-run and
  commit before pushing -- don't assume a clean CI run just because
  `just check-all` didn't error; it may have silently reformatted files out
  from under you that still need to be `git add`ed.
- **`ty check` is local-only.** Don't treat a green GitHub Actions run as
  proof the codebase type-checks; run `just typing` explicitly.
- **`--locked` sync.** If you add/change a dependency in `pyproject.toml`
  and forget `uv lock`, tests can still pass locally (using your unlocked,
  freshly-resolved environment) while CI's `uv sync --locked` fails
  immediately, before any lint or test runs.

## Project structure

- `src/found_tools/<tool_name>/` -- one directory per CLI tool
  (`calibrate/`, `edge/`, `render/`, `utils/`), each with its own
  `main.py`/entry point registered under `[project.scripts]` in
  `pyproject.toml`, and often a tool-specific `README.md`.
- `tests/found_tools/<tool_name>/` -- mirrors the `src/` layout.
- The `render` tool (`found_tools.render`) is the only place `bpy` (Blender
  as a Python module) is imported; see `src/found_tools/render/README.md`
  for how it's structured (`geometry.py`/`scene.py` are pure and fully
  tested, `blender_scene.py` does the actual Blender scene construction).
