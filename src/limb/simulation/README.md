# Limb simulation

Generate simulated limb (horizon) images: run a parameter sweep over FOV, resolution, and distance; write metadata to CSV; compute conic coefficients from the metadata; render conics to PNGs. Image filenames match the DataFrame row index (`img_000000.png` = row 0).

## CLI

Entry point: **`limb_simulation`** (after `pip install -e .` from the repo root).

Outputs (defaults are relative to the current working directory):

- **`--output-csv`** — Path for the simulation metadata CSV. Default: `sim_metadata.csv`.
- **`--output-folder`** — Directory for rendered images (no subfolders). Default: `sim_images`.

All list options use **space-separated** values (e.g. `--fovs 80 70`, not `80, 70`).

## Example (copyable)

```bash
limb_simulation \
  --fovs 80 70 \
  --resolutions 1024 2048 \
  --distances 7000000 8000000 9000000 \
  --num-positions-per-point 2 \
  --num-spins-per-position 2 \
  --num-radials-per-spin 2 \
  --output-csv sim_metadata.csv \
  --output-folder sim_images \
  --batch-size 100
```

Omit `--output-csv` and `--output-folder` to write to `sim_metadata.csv` and `sim_images/` in the current directory.

## Options

| Option | Required | Default | Description |
|--------|----------|---------|-------------|
| `--semi-axes` | No | WGS84 ellipsoid | Ellipsoid semi-axes a b c (m). |
| `--fovs` | Yes | — | Field-of-view values (degrees), space-separated. |
| `--resolutions` | Yes | — | Sensor resolutions (pixels, square), space-separated. |
| `--distances` | Yes | — | Distances from ellipsoid center to satellite (m), space-separated. |
| `--num-earth-points` | No | 1 | Number of uniform earth-point directions. |
| `--num-positions-per-point` | Yes | — | Satellite positions per earth point. |
| `--num-spins-per-position` | Yes | — | Image spins per position. |
| `--num-radials-per-spin` | Yes | — | Image radials per spin. |
| `--output-csv` | No | `sim_metadata.csv` | Metadata CSV path. |
| `--output-folder` | No | `sim_images` | Output directory for images. |
| `--batch-size` | No | 500 | Images per render batch. |
| `--sigma` | No | 2.0 | Gaussian blur sigma for limb edge. |
| `--seed` | No | None | Random seed for reproducibility. |
