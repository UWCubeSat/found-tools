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


def process_simulation(
    coeffs_nx6, width, height, output_folder, batch_size=200, sigma=2.0
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_folder, exist_ok=True)
    n_total = coeffs_nx6.shape[0]

    x = torch.linspace(0, width - 1, steps=width, device=device)
    y = torch.linspace(0, height - 1, steps=height, device=device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    grid_x, grid_y = grid_x.unsqueeze(0), grid_y.unsqueeze(0)

    with ThreadPoolExecutor(max_workers=8) as executor:
        for i in tqdm(range(0, n_total, batch_size), desc="Batches"):
            upper = min(i + batch_size, n_total)
            batch_coeffs = coeffs_nx6[i:upper].to(device)

            A = batch_coeffs[:, 0].view(-1, 1, 1)
            B = batch_coeffs[:, 1].view(-1, 1, 1)
            C = batch_coeffs[:, 2].view(-1, 1, 1)
            D = batch_coeffs[:, 3].view(-1, 1, 1)
            E = batch_coeffs[:, 4].view(-1, 1, 1)
            F = batch_coeffs[:, 5].view(-1, 1, 1)

            with torch.no_grad():
                Q = (
                    A * grid_x**2
                    + B * grid_x * grid_y
                    + C * grid_y**2
                    + D * grid_x
                    + E * grid_y
                    + F
                )

                # 1. Determine Fill Polarity
                # Trace of Hessian = 2A + 2C.
                # If sign is positive, 'inside' is Q < 0. If negative, 'inside' is Q > 0.
                fill_sign = -torch.sign(A + C)
                # Normalize Q so that 'inside' is always negative
                Q_normalized = Q * fill_sign

                # 2. Gradient for Taubin Distance
                gx = 2 * A * grid_x + B * grid_y + D
                gy = B * grid_x + 2 * C * grid_y + E
                grad_mag = torch.sqrt(gx**2 + gy**2 + 1e-8)

                # 3. Distance Calculation
                # clamp(Q_normalized, min=0) targets the 'outside' for blurring
                dist = torch.clamp(Q_normalized, min=0) / grad_mag
                intensity = torch.exp(-(dist**2) / (2 * sigma**2))

            batch_cpu = intensity.cpu().numpy()
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
