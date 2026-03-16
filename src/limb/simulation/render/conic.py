import torch
import cv2
import os
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


def save_image_worker(args):
    img_data, path = args
    img_uint8 = (img_data * 255).astype(np.uint8)
    cv2.imwrite(path, img_uint8, [cv2.IMWRITE_PNG_COMPRESSION, 0])

def side_of_hyperbola(ptx, pty, A, B, C, D, E):
    if not isinstance(ptx, torch.Tensor):
        ptx = torch.from_numpy(ptx).to(A.device)
    if not isinstance(pty, torch.Tensor):
        pty = torch.from_numpy(pty).to(A.device)

    # (h,k) is the hyperbola center
    h = (B*E-2*C*D)/(4*A*C-B*B + 0.0001)
    k = (B*D-2*A*E)/(4*A*C-B*B + 0.0001)
    # this is the angle of the transverse axis
    theta = torch.atan2(B, A - C) / 2
    # we find the vector from the center to the centroid
    x = ptx - h
    y = pty - k
    print(pty)
    print(y)
    return torch.sign(y*torch.cos(-theta)-x*torch.sin(-theta))


def process_simulation(
    coeffs_nx6,
    width,
    height,
    output_folder,
    K,
    rc,
    batch_size=200,
    sigma=2.0,
    row_indices=None,
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
    batch_size : int
        Number of images per batch.
    sigma : float
        Gaussian blur sigma for edge.
    row_indices : array-like, optional
        Shape (N,) integer indices for filenames. If provided, images are saved
        as img_{row_indices[j]:06d}.png so they match a DataFrame row index.
        If None, uses 0..N-1.
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
            batch_rc = rc[i:upper].to(device)

            A = batch_coeffs[:, 0].view(-1, 1, 1)
            B = batch_coeffs[:, 1].view(-1, 1, 1)
            C = batch_coeffs[:, 2].view(-1, 1, 1)
            D = batch_coeffs[:, 3].view(-1, 1, 1)
            E = batch_coeffs[:, 4].view(-1, 1, 1)
            F = batch_coeffs[:, 5].view(-1, 1, 1)
            calibrationx = batch_K[:,:,0].view(-1, 3, 1, 1)
            
            calibrationy = batch_K[:,:,1].view(-1, 3, 1, 1)
            calibrationz = batch_K[:,:,2].view(-1, 3, 1, 1)
            dirToEarthx = batch_rc[:, 0].view(-1, 1, 1)
            dirToEarthy = batch_rc[:, 1].view(-1, 1, 1)
            dirToEarthz = batch_rc[:, 2].view(-1, 1, 1)

            with torch.no_grad():
                Q = (
                    A * grid_x**2
                    + B * grid_x * grid_y
                    + C * grid_y**2
                    + D * grid_x
                    + E * grid_y
                    + F
                )
                # check if we're dealing with a hyperbola (yuck)
                pixelVec = (calibrationx*grid_x+calibrationy*grid_y+calibrationz)
                print(pixelVec.size())
                print(pixelVec[:,0].size())
                print(dirToEarthx.size())
                print((pixelVec[:,0]*dirToEarthx).size())
                wrong_side_mask = (pixelVec[:,0]*dirToEarthx + pixelVec[:,1]*dirToEarthy+ pixelVec[:,2]*dirToEarthz) > 0
                # 1. Determine Fill Polarity
                # Trace of Hessian = 2A + 2C.
                # If sign is positive, 'inside' is Q < 0. If negative, 'inside' is Q > 0.
                fill_sign = -torch.sign(A + C)
                # Normalize Q so that 'inside' is always negative
                # DOESN'T WORK?? Q WORKS FINE
                Q_normalized = Q * fill_sign

                

                # 2. Gradient for Taubin Distance
                gx = 2 * A * grid_x + B * grid_y + D
                gy = B * grid_x + 2 * C * grid_y + E
                grad_mag = torch.sqrt(gx**2 + gy**2 + 1e-8)

                # 3. Distance Calculation
                # clamp(Q_normalized, min=0) targets the 'outside' for blurring
                dist = torch.clamp(Q, min=0) / grad_mag
                intensity = torch.exp(-(dist**2) / (2 * sigma**2))
                intensity = torch.where(wrong_side_mask, torch.zeros_like(intensity), intensity)

            batch_cpu = intensity.cpu().numpy()
            if row_indices is not None:
                batch_indices = row_indices[i:upper]
                save_tasks = [
                    (
                        batch_cpu[j],
                        os.path.join(output_folder, f"img_{batch_indices[j]:06d}.png"),
                    )
                    for j in range(batch_cpu.shape[0])
                ]
            else:
                save_tasks = [
                    (batch_cpu[j], os.path.join(output_folder, f"img_{i + j:06d}.png"))
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

    process_simulation(all_coeffs, 1280, 720, "sim_out", batch_size=100)
