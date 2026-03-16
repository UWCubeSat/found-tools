"""Plot a window of the image around a point with edge points and optional conic."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.image import imread

from limb.simulation.edge.conic import _conic_matrix_to_coeffs, solve_general_conic


def edge_plot(
    image_path: str,
    points: np.ndarray,
    center_point: int,
    window_length: float,
    pixel_conic: np.ndarray | None = None,
) -> None:
    """Plot a window of the image centered at points[center_point] with pixel grid and overlays.

    Draws the image crop, subpixel point locations (tiny 'x'), and optionally the
    pixel conic as a line. Pixel boundaries are shown clearly.

    Args:
        image_path: Path to the image file.
        points: (N, 2) array of (x, y) subpixel coordinates.
        center_point: Index into points for the window center.
        window_length: Side length of the square window (pixels).
        pixel_conic: Optional 3×3 conic matrix in pixel coordinates; if given, the
            conic is drawn as a line in the window.
    """
    img = imread(image_path)
    if img.ndim == 3:
        # RGB/RGBA: leave as-is for imshow
        height, width = img.shape[0], img.shape[1]
    else:
        height, width = img.shape[0], img.shape[1]

    center = np.asarray(points[center_point], dtype=np.float64)
    half = window_length / 2.0
    x0 = max(0, int(np.floor(center[0] - half)))
    x1 = min(width, int(np.ceil(center[0] + half)))
    y0 = max(0, int(np.floor(center[1] - half)))
    y1 = min(height, int(np.ceil(center[1] + half)))

    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        raise ValueError("Window is empty (fully outside image).")

    fig, ax = plt.subplots()
    ax.imshow(crop, extent=[x0, x1, y1, y0], aspect="equal", interpolation="nearest")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)

    # Clear pixel delineations: grid at integer coordinates
    ax.set_xticks(np.arange(x0, x1 + 1))
    ax.set_yticks(np.arange(y0, y1 + 1))
    ax.grid(True, color="w", linewidth=0.5, alpha=0.7)
    ax.tick_params(axis="both", labelsize=8)
    ax.set_xlabel("x (pixel)")
    ax.set_ylabel("y (pixel)")

    # Subpixel locations of points inside the window (tiny x)
    in_window = (
        (points[:, 0] >= x0)
        & (points[:, 0] < x1)
        & (points[:, 1] >= y0)
        & (points[:, 1] < y1)
    )
    px = points[in_window, 0]
    py = points[in_window, 1]
    if px.size:
        ax.plot(px, py, "x", color="cyan", markersize=3, markeredgewidth=0.8)

    # Optional: pixel conic as a line in the window
    if pixel_conic is not None:
        a, b, c, d, e, f = _conic_matrix_to_coeffs(pixel_conic)
        conic_pts: list[tuple[float, float]] = []
        # Sample over x in window
        for x in np.linspace(x0, x1, max(2, int(window_length) * 4)):
            sols = solve_general_conic(a, b, c, d, e, f, float(x), "solve_y")
            if sols is not None:
                for y in sols:
                    if y0 <= y <= y1:
                        conic_pts.append((float(x), float(y)))
        # Sample over y in window
        for y in np.linspace(y0, y1, max(2, int(window_length) * 4)):
            sols = solve_general_conic(a, b, c, d, e, f, float(y), "solve_x")
            if sols is not None:
                for x in sols:
                    if x0 <= x <= x1:
                        conic_pts.append((float(x), float(y)))
        if conic_pts:
            arr = np.array(conic_pts)
            # Order by angle around center for a continuous curve
            angles = np.arctan2(arr[:, 1] - center[1], arr[:, 0] - center[0])
            order = np.argsort(angles)
            ax.plot(
                arr[order, 0],
                arr[order, 1],
                "-",
                color="lime",
                linewidth=1.0,
                alpha=0.9,
            )

    plt.tight_layout()
    plt.show()
