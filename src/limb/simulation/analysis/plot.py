"""Plot a window of the image around a point with edge points and optional conic.

Also provides OpNav validation plots: radius residuals vs range and centroid
residuals in the illumination frame.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.image import imread


def edge_plot(
    image_path: str,
    points: np.ndarray,
    center_point: int,
    window_length: float,
    true_points: np.ndarray | None = None,
    save_path: str | None = None,
) -> None:
    """Plot a window of the image centered at points[center_point] with pixel grid and overlays.

    Draws the image crop, subpixel point locations (tiny 'x'), and optionally the
    pixel conic as a line. Pixel boundaries are shown clearly.

    Args:
        image_path: Path to the image file.
        points: (N, 2) array of (x, y) subpixel coordinates.
        center_point: Index into points for the window center.
        window_length: Side length of the square window (pixels).
        true_points: Optional (N, 2) array of true limb points; if given, drawn as red line.
        save_path: Optional path to save the plot image (e.g. .png or .pdf).
    """
    img = imread(image_path)
    if img.ndim == 3:
        height, width = img.shape[0], img.shape[1]
        # Keep black and white: show as grayscale so matplotlib does not change colors
        crop_img = np.mean(img, axis=2)
    else:
        height, width = img.shape[0], img.shape[1]
        crop_img = img

    center = np.asarray(points[center_point], dtype=np.float64)
    half = window_length / 2.0
    x0 = max(0, int(np.floor(center[0] - half)))
    x1 = min(width, int(np.ceil(center[0] + half)))
    y0 = max(0, int(np.floor(center[1] - half)))
    y1 = min(height, int(np.ceil(center[1] + half)))

    crop = crop_img[y0:y1, x0:x1]
    if crop.size == 0:
        raise ValueError("Window is empty (fully outside image).")

    fig, ax = plt.subplots()
    ax.imshow(
        crop,
        extent=[x0, x1, y1, y0],
        aspect="equal",
        interpolation="nearest",
        cmap="gray",
    )
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)

    # Pixel grid; no numbers on axes
    ax.set_xticks(np.arange(x0, x1 + 1))
    ax.set_yticks(np.arange(y0, y1 + 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, color="black", linewidth=0.5, alpha=0.7)

    # Edge points as blue dots
    in_window = (
        (points[:, 0] >= x0)
        & (points[:, 0] < x1)
        & (points[:, 1] >= y0)
        & (points[:, 1] < y1)
    )
    px = points[in_window, 0]
    py = points[in_window, 1]
    if px.size:
        ax.plot(px, py, ".", color="blue", markersize=3, label="Detected Limb Points")

    # Optional: true limb as red line
    if true_points is not None:
        in_window = (
            (true_points[:, 0] >= x0)
            & (true_points[:, 0] < x1)
            & (true_points[:, 1] >= y0)
            & (true_points[:, 1] < y1)
        )
        px_true = true_points[in_window, 0]
        py_true = true_points[in_window, 1]
        if px_true.size:
            ax.plot(px_true, py_true, "-", color="red", linewidth=1, label="True Limb")

    ax.legend()
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
    plt.show()


def plot_radius_residuals_vs_range(
    ranges_m: np.ndarray,
    radius_residuals_px: np.ndarray,
    pass_ids: list[str],
    radius_1sigma_px: float,
    target_label: str = "Moon",
    save_path: Optional[Path | str] = None,
):
    """Plot radius residuals (pixels) vs true range (m) for all passes with 3σ bounds.

    X-axis: true range (meters). Y-axis: radius residual (pixels). Multiple passes
    are plotted with distinct colors; 3σ error bounds are drawn from the given
    sigma.
    """
    unique_passes = sorted(set(pass_ids))
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_passes)))
    for i, pass_id in enumerate(unique_passes):
        mask = np.array([pid == pass_id for pid in pass_ids])
        ax.scatter(
            ranges_m[mask],
            radius_residuals_px[mask],
            c=[colors[i]],
            label=pass_id,
            alpha=0.6,
            s=40,
        )
    range_array = np.linspace(ranges_m.min(), ranges_m.max(), 100)
    sigma = radius_1sigma_px
    ax.plot(
        range_array,
        3 * sigma * np.ones_like(range_array),
        "k--",
        linewidth=2,
        label="3σ bounds",
    )
    ax.plot(range_array, -3 * sigma * np.ones_like(range_array), "k--", linewidth=2)
    ax.fill_between(range_array, -3 * sigma, 3 * sigma, alpha=0.1, color="gray")
    ax.axhline(0, color="k", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("True Range (m)", fontsize=12)
    ax.set_ylabel("Radius Residual (pixels)", fontsize=12)
    ax.set_title(f"{target_label} Radius Residuals for All Passes", fontsize=14)
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax


def plot_centroid_residuals_in_illumination_frame(
    ranges_m: np.ndarray,
    centroid_x_px: np.ndarray,
    centroid_y_px: np.ndarray,
    pass_ids: list[str],
    centroid_1sigma_px: float,
    target_label: str = "Moon",
    save_path: Optional[Path | str] = None,
):
    """Plot X and Y centroid residuals (illumination frame) vs true range (m).

    Two subplots: X along sun direction, Y perpendicular to sun. 3σ bounds are
    drawn from the given sigma.
    """
    unique_passes = sorted(set(pass_ids))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_passes)))
    for i, pass_id in enumerate(unique_passes):
        mask = np.array([pid == pass_id for pid in pass_ids])
        ax1.scatter(
            ranges_m[mask],
            centroid_x_px[mask],
            c=[colors[i]],
            label=pass_id,
            alpha=0.6,
            s=40,
        )
    for i, pass_id in enumerate(unique_passes):
        mask = np.array([pid == pass_id for pid in pass_ids])
        ax2.scatter(
            ranges_m[mask],
            centroid_y_px[mask],
            c=[colors[i]],
            alpha=0.6,
            s=40,
        )
    sigma = centroid_1sigma_px
    for ax in (ax1, ax2):
        ax.axhline(3 * sigma, color="k", linestyle="--", linewidth=2)
        ax.axhline(-3 * sigma, color="k", linestyle="--", linewidth=2)
        ax.fill_between(
            [ranges_m.min(), ranges_m.max()],
            -3 * sigma,
            3 * sigma,
            alpha=0.1,
            color="gray",
        )
        ax.axhline(0, color="k", linestyle="-", linewidth=0.5, alpha=0.5)
        ax.grid(True, alpha=0.3)
    ax1.set_ylabel("X Centroid Residual (pixels)\n(along sun direction)", fontsize=11)
    ax2.set_ylabel("Y Centroid Residual (pixels)\n(perpendicular to sun)", fontsize=11)
    ax2.set_xlabel("True Range (m)", fontsize=12)
    ax1.set_title(
        f"{target_label} Centroid Residuals in Illumination Frame", fontsize=14
    )
    ax1.legend(loc="best", framealpha=0.9)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, (ax1, ax2)
