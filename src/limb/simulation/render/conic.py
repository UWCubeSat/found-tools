import torch
import cv2
import os
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


def _legendre_gauss_torch(n: int, device: torch.device, dtype: torch.dtype):
    """Gauss–Legendre nodes and weights on [-1, 1] (numpy poly, torch tensors)."""
    x_np, w_np = np.polynomial.legendre.leggauss(int(n))
    return (
        torch.tensor(x_np, device=device, dtype=dtype),
        torch.tensor(w_np, device=device, dtype=dtype),
    )


def _pixel_xy_from_rc_camera(batch_rc: torch.Tensor, k_inv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pixel (x, y) where ``batch_rc`` projects; matches ``Camera.camera_to_pixel``.

    Camera frame: depth along axis 0; ``k_inv`` is ``inverse_calibration_matrix`` per image,
    shape (B, 3, 3).
    """
    x_c = batch_rc[:, 0].clamp(min=torch.finfo(batch_rc.dtype).eps)
    y_c, z_c = batch_rc[:, 1], batch_rc[:, 2]
    homog_im = torch.stack(
        (torch.ones_like(x_c), y_c / x_c, z_c / x_c),
        dim=1,
    )
    k_fwd = torch.linalg.inv(k_inv)
    p = torch.einsum("bij,bj->bi", k_fwd, homog_im)
    return p[:, 0], p[:, 1]


def _disk_pixel_coverage_fraction_batched(
    cx: torch.Tensor,
    cy: torch.Tensor,
    r: torch.Tensor,
    height: int,
    width: int,
    n_gauss: int = 24,
) -> torch.Tensor:
    """Batched horizontal-chord quadrature: disk ∩ unit pixel / pixel area.

    Pixels are [j−½,j+½]×[i−½,i+½] in (x,y), matching integer centers from
    ``linspace(0, n−1)`` with ``indexing='ij'`` (row i, col j).

    Parameters
    ----------
    cx, cy, r : (B,)
    Returns
    -------
    (B, H, W) in [0, 1].
    """
    device, dtype = cx.device, cx.dtype
    Bn = cx.shape[0]
    t, wt = _legendre_gauss_torch(n_gauss, device, dtype)
    t_g = t.view(1, 1, 1, n_gauss)
    w_g = wt.view(1, 1, 1, n_gauss)

    jx = torch.arange(width, device=device, dtype=dtype).view(1, 1, width)
    iy = torch.arange(height, device=device, dtype=dtype).view(1, height, 1)
    cx_b = cx.view(Bn, 1, 1)
    cy_b = cy.view(Bn, 1, 1)
    r_b = r.view(Bn, 1, 1)

    x0 = (jx - 0.5 - cx_b).expand(Bn, height, width)
    x1 = (jx + 0.5 - cx_b).expand(Bn, height, width)
    y0 = (iy - 0.5 - cy_b).expand(Bn, height, width)
    y1 = (iy + 0.5 - cy_b).expand(Bn, height, width)

    half_h = 0.5 * (y1 - y0)
    mid_y = 0.5 * (y1 + y0)
    v = mid_y.unsqueeze(-1) + half_h.unsqueeze(-1) * t_g
    rsq = (r_b * r_b).unsqueeze(-1)
    s = torch.sqrt(torch.clamp(rsq - v * v, min=0.0))
    left = torch.maximum(x0.unsqueeze(-1), -s)
    right = torch.minimum(x1.unsqueeze(-1), s)
    chord = torch.clamp(right - left, min=0.0)
    # ∫ L(v) dv ≈ ((y1-y0)/2) * Σ w_k L(v_k) with half_h = (y1-y0)/2
    area = half_h * (chord * w_g).sum(dim=-1)
    return area.clamp(0.0, 1.0)


# -----------------------------------------------------------------------------
# Custom noise implementations (used only within this render pipeline).
# Pipeline order: gaussian -> stars -> discretization -> motion_blur -> dead_pixels (last).
# -----------------------------------------------------------------------------


def _add_salt_pepper(img, salt_prob=0.01, pepper_prob=0.01):
    """Add salt-and-pepper noise; used as last step for dead pixels. img: uint8 (H,W) or (H,W,C)."""
    out = img.copy()
    n = img.size
    num_salt = int(np.ceil(salt_prob * n))
    num_pepper = int(np.ceil(pepper_prob * n))
    if num_salt > 0:
        coords = [np.random.randint(0, s, num_salt) for s in img.shape]
        out[tuple(coords)] = 255
    if num_pepper > 0:
        coords = [np.random.randint(0, s, num_pepper) for s in img.shape]
        out[tuple(coords)] = 0
    return out


def _apply_discretization(img, levels=8):
    """Reduce intensity to levels per channel. img: uint8."""
    if levels < 1:
        levels = 1
    factor = max(1, 256 // levels)
    out = (img.astype(np.int32) // factor) * factor
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_motion_blur(img, kernel_size=5):
    """Horizontal motion blur. img: uint8 (H,W) or (H,W,C)."""
    if kernel_size < 1 or kernel_size % 2 == 0:
        kernel_size = max(1, kernel_size | 1)
    k = np.zeros((kernel_size, kernel_size), dtype=np.float64)
    k[kernel_size // 2, :] = 1.0 / kernel_size
    return cv2.filter2D(img, -1, k)


def _apply_gaussian(img, mean=0, sigma=10):
    """Add Gaussian noise. img: uint8 (H,W) or (H,W,C)."""
    g = np.random.normal(mean, sigma, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int32) + g, 0, 255).astype(np.uint8)


def _apply_noise_pipeline(img_uint8, noise_config):
    """Apply configured noise effects in order: gaussian -> stars -> discretization ->
    motion_blur -> dead_pixels (last)."""
    if not noise_config:
        return img_uint8
    out = img_uint8
    if "gaussian" in noise_config:
        o = noise_config["gaussian"]
        out = _apply_gaussian(out, mean=o.get("mean", 0), sigma=o.get("sigma", 10))
    if "stars" in noise_config:
        o = noise_config["stars"]
        out = _add_salt_pepper(out, salt_prob=o.get("prob", 0.005), pepper_prob=0.0)
    if "discretization" in noise_config:
        o = noise_config["discretization"]
        out = _apply_discretization(out, levels=o.get("levels", 8))
    if "motion_blur" in noise_config:
        o = noise_config["motion_blur"]
        out = _apply_motion_blur(out, kernel_size=o.get("kernel_size", 5))
    if "dead_pixels" in noise_config:
        o = noise_config["dead_pixels"]
        out = _add_salt_pepper(
            out,
            salt_prob=o.get("salt_prob", 0.01),
            pepper_prob=o.get("pepper_prob", 0.01),
        )
    return out


def save_image_worker(args):
    """Save a single image; optionally apply noise pipeline before saving."""
    if len(args) == 3:
        img_data, path, noise_config = args
    else:
        img_data, path = args
        noise_config = None
    img_uint8 = (np.clip(img_data, 0.0, 1.0) * 255).astype(np.uint8)
    if noise_config:
        img_uint8 = _apply_noise_pipeline(img_uint8, noise_config)
    cv2.imwrite(path, img_uint8, [cv2.IMWRITE_PNG_COMPRESSION, 0])


def side_of_hyperbola(ptx, pty, A, B, C, D, E):
    if not isinstance(ptx, torch.Tensor):
        ptx = torch.from_numpy(ptx).to(A.device)
    if not isinstance(pty, torch.Tensor):
        pty = torch.from_numpy(pty).to(A.device)

    # (h,k) is the hyperbola center
    h = (B * E - 2 * C * D) / (4 * A * C - B * B + 0.0001)
    k = (B * D - 2 * A * E) / (4 * A * C - B * B + 0.0001)
    # this is the angle of the transverse axis
    theta = torch.atan2(B, A - C) / 2
    # we find the vector from the center to the centroid
    x = ptx - h
    y = pty - k
    return torch.sign(y * torch.cos(-theta) - x * torch.sin(-theta))


def process_simulation(
    coeffs_nx6,
    width,
    height,
    output_folder,
    K,
    rc,
    batch_size=200,
    sigma=0.0,
    row_indices=None,
    noise_config=None,
    circle_tol: float = 1e-4,
    disk_quadrature_points: int = 24,
):
    """Render conic coefficients to images.

    Parameters
    ----------
    coeffs_nx6 : array-like
        Shape (N, 6) conic coefficients.
    width, height : int
        Image dimensions (same for all in batch).
    output_folder : str
        Directory to write images (no subfolders).
    K : array-like
        Shape (N, 3, 3), inverse intrinsics per image.
    rc : array-like
        Shape (N, 3), satellite position in camera frame (for visibility and fill anchor).
    batch_size : int
        Number of images per batch.
    sigma : float
        Gaussian blur sigma for edge (Taubin path only; see Notes). If ``<= 0``, hard edge
        on the ``q_fill`` half-plane, or geometric disk fill when applicable.
    circle_tol : float
        Relative tolerance: treat conic as a circle when ``|B|/A`` and ``|A-C|/A`` are
        below this (after ``|A|`` scaling).
    disk_quadrature_points : int
        Gauss–Legendre order for disk–pixel area (horizontal chord integral).

    Notes
    -----
    **Which side is white:** ``Q`` at the pixel projection of ``rc`` defines the anchor
    sign; intensity uses the region where ``Q`` matches that sign (via ``q_fill``).

    **Circular conics** (limb near a circle): when ``sigma <= 0`` and coefficients pass
    the circle test, intensity is the **exact** fraction of each pixel covered by the disk
    ``(x-cx)²+(y-cy)² ≤ r²``, or its complement, according to the same anchor rule
    (Gauss–Legendre quadrature of the horizontal slice integral—machine‑accurate for
    axis‑aligned pixels). Non-circular conics use the Taubin distance on ``q_fill``.
    If ``sigma > 0``, Taubin is used for all images (including circles).
    row_indices : array-like, optional
        Shape (N,) integer indices for filenames. If provided, images are saved
        as img_{row_indices[j]:06d}.png so they match a DataFrame row index.
        If None, uses 0..N-1.
    noise_config : dict, optional
        If provided, apply camera/sensor noise in the pipeline. Order: gaussian, stars,
        discretization, motion_blur, dead_pixels (last). Keys: "gaussian", "stars",
        "discretization", "motion_blur", "dead_pixels". Each value is a dict of kwargs. Example: {"stars": {"prob": 0.005},
        "dead_pixels": {"salt_prob": 0.01, "pepper_prob": 0.01}}.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_folder, exist_ok=True)
    n_total = coeffs_nx6.shape[0]
    if row_indices is not None:
        row_indices = np.asarray(row_indices, dtype=np.int64)
        if row_indices.shape[0] != n_total:
            raise ValueError("row_indices length must match coeffs_nx6 row count")

    x = torch.linspace(0, width - 1, steps=width, device=device)
    y = torch.linspace(0, height - 1, steps=height, device=device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    grid_x, grid_y = grid_x.unsqueeze(0), grid_y.unsqueeze(0)

    with ThreadPoolExecutor(max_workers=8) as executor:
        for i in tqdm(range(0, n_total, batch_size), desc="Batches"):
            upper = min(i + batch_size, n_total)
            batch_coeffs = coeffs_nx6[i:upper].to(device)
            batch_K = K[i:upper].to(device)
            # Slice the original rc parameter and keep a separate batched view variable
            batch_rc = rc[i:upper].to(device)

            A = batch_coeffs[:, 0].view(-1, 1, 1)
            B = batch_coeffs[:, 1].view(-1, 1, 1)
            C = batch_coeffs[:, 2].view(-1, 1, 1)
            D = batch_coeffs[:, 3].view(-1, 1, 1)
            E = batch_coeffs[:, 4].view(-1, 1, 1)
            F = batch_coeffs[:, 5].view(-1, 1, 1)
            calibrationx = batch_K[:, :, 0].view(-1, 3, 1, 1)

            calibrationy = batch_K[:, :, 1].view(-1, 3, 1, 1)
            calibrationz = batch_K[:, :, 2].view(-1, 3, 1, 1)
            # rc_batched: camera position in camera frame (batch, 3, 1, 1), same role as rc in edge/conic
            rc_batched = batch_rc.view(-1, 3, 1, 1)

            with torch.no_grad():
                Q = (
                    A * grid_x**2
                    + B * grid_x * grid_y
                    + C * grid_y**2
                    + D * grid_x
                    + E * grid_y
                    + F
                )

                px_s, py_s = _pixel_xy_from_rc_camera(batch_rc, batch_K)
                A1 = batch_coeffs[:, 0]
                B1 = batch_coeffs[:, 1]
                C1 = batch_coeffs[:, 2]
                D1 = batch_coeffs[:, 3]
                E1 = batch_coeffs[:, 4]
                F1 = batch_coeffs[:, 5]
                q_sat = (
                    A1 * px_s**2
                    + B1 * px_s * py_s
                    + C1 * py_s**2
                    + D1 * px_s
                    + E1 * py_s
                    + F1
                )
                sign_sat = torch.sign(q_sat)
                sign_sat = torch.where(
                    sign_sat == 0, torch.ones_like(sign_sat), sign_sat
                )
                sign_sat_map = sign_sat.view(-1, 1, 1)
                q_fill = -Q * sign_sat_map

                # Visible arc mask: dot(ray, rc) > 0 means wrong side (consistent with edge logic)
                pixel_vec = calibrationx * grid_x + calibrationy * grid_y + calibrationz
                wrong_side_mask = (pixel_vec * rc_batched).sum(dim=1) > 0

                # --- Taubin path (q_fill half-space distance) ---
                gx = 2 * A * grid_x + B * grid_y + D
                gy = B * grid_x + 2 * C * grid_y + E
                grad_mag = torch.sqrt(gx**2 + gy**2 + 1e-8)
                dist = torch.clamp(q_fill, min=0) / grad_mag
                sigma_f = float(sigma)
                if sigma_f > 0.0:
                    intensity = torch.exp(-(dist**2) / (2.0 * sigma_f**2))
                else:
                    intensity = torch.where(
                        dist <= 0,
                        torch.ones_like(dist),
                        torch.zeros_like(dist),
                    )

                # --- Geometric disk coverage for near-circle conics (sigma == 0) ---
                scale = torch.maximum(torch.abs(A1), torch.abs(C1)).clamp(min=1e-12)
                is_circle = (torch.abs(B1) / scale < circle_tol) & (
                    torch.abs(A1 - C1) / scale < circle_tol
                )
                denom = 4.0 * A1 * C1 - B1 * B1
                denom = denom + torch.where(
                    denom.abs() < 1e-18, torch.full_like(denom, 1e-18), torch.zeros_like(denom)
                )
                cx = (B1 * E1 - 2.0 * C1 * D1) / denom
                cy = (B1 * D1 - 2.0 * A1 * E1) / denom
                r_sq_phys = cx * cx + cy * cy - F1 / A1
                r = torch.sqrt(torch.clamp(r_sq_phys, min=0.0))
                geom_ok = (
                    is_circle
                    & (A1.abs() > 1e-12)
                    & (r_sq_phys > 1e-14)
                    & (r > 1e-8)
                    & (sigma_f <= 0.0)
                )
                if geom_ok.any():
                    idx = torch.where(geom_ok)[0]
                    frac = _disk_pixel_coverage_fraction_batched(
                        cx[idx],
                        cy[idx],
                        r[idx],
                        height,
                        width,
                        n_gauss=int(disk_quadrature_points),
                    )
                    fill_disk = (q_sat[idx] <= 0).view(-1, 1, 1)
                    int_geom = torch.where(fill_disk, frac, 1.0 - frac)
                    intensity[geom_ok] = int_geom

                intensity = torch.where(
                    wrong_side_mask, torch.zeros_like(intensity), intensity
                )

            batch_cpu = intensity.cpu().numpy()
            if row_indices is not None:
                batch_indices = row_indices[i:upper]
                save_tasks = [
                    (
                        batch_cpu[j],
                        os.path.join(output_folder, f"img_{batch_indices[j]:06d}.png"),
                        noise_config,
                    )
                    for j in range(batch_cpu.shape[0])
                ]
            else:
                save_tasks = [
                    (
                        batch_cpu[j],
                        os.path.join(output_folder, f"img_{i + j:06d}.png"),
                        noise_config,
                    )
                    for j in range(batch_cpu.shape[0])
                ]

            list(executor.map(save_image_worker, save_tasks))


if __name__ == "__main__":
    # Test with 1000 conics
    N_total = 1000
    all_coeffs = torch.randn(N_total, 6)
    # Ensure we get some closed ellipses for testing
    all_coeffs[:, 0] = torch.abs(all_coeffs[:, 0]) + 0.001
    all_coeffs[:, 2] = torch.abs(all_coeffs[:, 2]) + 0.001
    all_coeffs[:, 5] = -0.5

    eye = torch.eye(3, dtype=all_coeffs.dtype).unsqueeze(0).expand(N_total, 3, 3).clone()
    rc0 = torch.tensor([[1.0, 0.0, 0.0]], dtype=all_coeffs.dtype).expand(N_total, 3)
    process_simulation(
        all_coeffs,
        1280,
        720,
        "sim_out",
        K=eye,
        rc=rc0,
        batch_size=100,
    )
