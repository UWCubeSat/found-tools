import torch
import cv2
import os
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def save_image_worker(args):
    """Fastest possible disk write: No compression, multi-threaded."""
    img_data, path = args
    # Scaling float32 [0, 1] to uint8 [0, 255]
    img_uint8 = (img_data * 255).astype(np.uint8)
    # Using compression 0 for maximum write speed
    cv2.imwrite(path, img_uint8, [cv2.IMWRITE_PNG_COMPRESSION, 0])

def process_simulation(coeffs_nx6, width, height, output_folder, batch_size=500, sigma=2.0):
    """
    Handles N > Memory and CPU/GPU fallback.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running simulation on: {device}")
    
    os.makedirs(output_folder, exist_ok=True)
    n_total = coeffs_nx6.shape[0]
    
    # 1. Pre-calculate the Coordinate Grid once
    x = torch.linspace(0, width - 1, steps=width, device=device)
    y = torch.linspace(0, height - 1, steps=height, device=device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    grid_x, grid_y = grid_x.unsqueeze(0), grid_y.unsqueeze(0) # (1, H, W)

    # 2. Process in Batches
    with ThreadPoolExecutor(max_workers=8) as executor:
        for i in tqdm(range(0, n_total, batch_size), desc="Batches"):
            upper = min(i + batch_size, n_total)
            batch_coeffs = coeffs_nx6[i:upper].to(device)
            
            # Extract and Reshape to (Batch, 1, 1)
            A = batch_coeffs[:, 0].view(-1, 1, 1)
            B = batch_coeffs[:, 1].view(-1, 1, 1)
            C = batch_coeffs[:, 2].view(-1, 1, 1)
            D = batch_coeffs[:, 3].view(-1, 1, 1)
            E = batch_coeffs[:, 4].view(-1, 1, 1)
            F = batch_coeffs[:, 5].view(-1, 1, 1)

            # 3. Vectorized Math (Implicit Equation + Analytic Blur)
            with torch.no_grad():
                Q = A*grid_x**2 + B*grid_x*grid_y + C*grid_y**2 + D*grid_x + E*grid_y + F
                
                # Gradient Magnitude for distance approximation
                gx = 2*A*grid_x + B*grid_y + D
                gy = B*grid_x + 2*C*grid_y + E
                grad_mag = torch.sqrt(gx**2 + gy**2 + 1e-8)
                
                # Distance and Intensity
                dist = torch.clamp(Q, min=0) / grad_mag
                intensity = torch.exp(-(dist**2) / (2 * sigma**2))

            # 4. Offload to CPU and Queue for Disk I/O
            batch_cpu = intensity.cpu().numpy()
            save_tasks = []
            for j in range(batch_cpu.shape[0]):
                file_path = os.path.join(output_folder, f"img_{i + j:06d}.png")
                save_tasks.append((batch_cpu[j], file_path))
            
            # Map the saving tasks to the thread pool
            executor.map(save_image_worker, save_tasks)

# --- Execution ---
if __name__ == "__main__":
    # Simulate 10,000 conics
    N_total = 10000
    all_coeffs = torch.randn(N_total, 6) # Start on CPU
    all_coeffs[:, 5] = -0.5 # Bias toward showing shapes
    
    # Recommended batch_size for 8GB VRAM at 1080p is ~100-200. 
    # For lower resolutions, you can go much higher.
    process_simulation(all_coeffs, 1280, 720, "sim_out", batch_size=200)