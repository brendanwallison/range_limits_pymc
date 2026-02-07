import sys
import os
import time
import argparse
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from jax.numpy.fft import fft2, ifft2
from tqdm import tqdm

# --- Setup Paths ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Only import the kernel builder, we define the splits locally
from src.model.build_kernels import make_radial_directional_kernels

# =========================================================
# PART 1: The "Scaling Trick" (Path Integration)
# =========================================================

def resize_kernel_stack(kernel_stack, scale):
    """
    Resizes a stack of kernels by a scale factor 's'.
    
    CRITICAL FIX: 
    Kernels are in FFT layout (mass at corners).
    We must fftshift (mass to center) BEFORE resizing/padding, 
    then ifftshift (mass back to corners) AFTER.
    Otherwise, padding moves the mass from index 0 to N/2 (massive displacement).
    """
    K, Ly, Lx = kernel_stack.shape
    new_H = int(Ly * scale)
    new_W = int(Lx * scale)
    
    if new_H < 1 or new_W < 1:
        return jnp.zeros_like(kernel_stack)

    # 1. Shift Mass to Center (Spatial Layout)
    stack_centered = jnp.fft.fftshift(kernel_stack, axes=(-2, -1))

    # 2. Resize (Shrink the centered blob)
    shrunk = jax.image.resize(stack_centered, (K, new_H, new_W), method='bilinear')
    
    # 3. Pad (Restores original frame size, keeping blob in center)
    pad_y = (Ly - new_H) // 2
    pad_x = (Lx - new_W) // 2
    pad_y_end = Ly - new_H - pad_y
    pad_x_end = Lx - new_W - pad_x
    padded = jnp.pad(shrunk, ((0,0), (pad_y, pad_y_end), (pad_x, pad_x_end)))
    
    # 4. Unshift Mass to Corners (FFT Layout)
    result = jnp.fft.ifftshift(padded, axes=(-2, -1))
    
    # 5. Restore Sum (Mass Conservation)
    current_sum = jnp.sum(result, axis=(1,2), keepdims=True)
    target_sum = jnp.sum(kernel_stack, axis=(1,2), keepdims=True)
    scale_factor = jnp.where(current_sum > 1e-9, target_sum / current_sum, 0.0)
    
    return result * scale_factor

@jax.jit
def convolve_step(Z_t, kernel_stack_fft):
    """Efficiently convolves a batch of Z features."""
    Ny, Nx, _ = Z_t.shape
    K, Ly, Lx = kernel_stack_fft.shape
    Z_padded = jnp.pad(Z_t, ((0, Ly-Ny), (0, Lx-Nx), (0, 0)))
    Z_fft = jnp.fft.fft2(Z_padded, axes=(0, 1)).transpose(2, 0, 1)
    
    # (Batch, 1, Ly, Lx) * (1, K, Ly, Lx)
    # Standard convolution here correctly "looks back" at the history
    conv_fft = Z_fft[:, None, :, :] * kernel_stack_fft[None, :, :, :]
    conv_spatial = jnp.real(jnp.fft.ifft2(conv_fft, axes=(-2, -1)))
    return conv_spatial[:, :, :Ny, :Nx]

@jax.jit
def convolve_mask_step(mask, kernel_stack_fft):
    """Convolves binary land mask to find normalization weights."""
    Ny, Nx = mask.shape
    K, Ly, Lx = kernel_stack_fft.shape
    mask_padded = jnp.pad(mask, ((0, Ly-Ny), (0, Lx-Nx)))
    mask_fft = fft2(mask_padded)
    conv_fft = mask_fft[None, :, :] * kernel_stack_fft
    return jnp.real(ifft2(conv_fft, axes=(-2, -1)))[:, :Ny, :Nx]

def integrate_paths(Z, kernel_stack, land_mask, steps=10, feature_batch_size=4):
    """Computes path-dependent features using Normalized Convolution."""
    print(f"Path Integration: Z={Z.shape}, Kernels={kernel_stack.shape}, Steps={steps}")
    
    Time, Ny, Nx, M = Z.shape
    K_kernels = kernel_stack.shape[0]
    
    # 1. Sanitize Z (NaN -> 0.0)
    Z_safe = jnp.nan_to_num(Z, nan=0.0)
    
    # 2. Mask Z (Water -> 0.0)
    # This ensures water pixels don't contribute to the numerator
    mask_bc = land_mask[None, :, :, None] 
    Z_masked = jnp.where(mask_bc > 0.5, Z_safe, 0.0)
    
    Z_disp_acc = jnp.zeros((Time, M, K_kernels, Ny, Nx))
    s_vals = np.linspace(1.0 / steps, 1.0, steps)
    
    for s in tqdm(s_vals, desc="Integrating Paths"):
        # A. Resize Kernels
        scaled_kernels = resize_kernel_stack(kernel_stack, s)
        scaled_kernels_fft = fft2(scaled_kernels)
        
        # B. Compute Normalizer (Denominator: How much land was traversed?)
        land_weight = convolve_mask_step(land_mask, scaled_kernels_fft)
        land_weight = jnp.maximum(land_weight, 1e-6)
        
        for t in range(Time):
            for i in range(0, M, feature_batch_size):
                z_slice = Z_masked[t, :, :, i : i + feature_batch_size]
                
                # Convolution (Numerator: Sum of Z on Land)
                num_slice = convolve_step(z_slice, scaled_kernels_fft)
                
                # Normalization (Numerator / Denominator)
                avg_feat_slice = num_slice / land_weight[None, :, :, :]
                
                Z_disp_acc = Z_disp_acc.at[t, i : i + feature_batch_size].add(avg_feat_slice)
            
    Z_disp_final = jnp.transpose(Z_disp_acc / steps, (0, 3, 4, 1, 2))
    return Z_disp_final

# =========================================================
# PART 2: Helper Functions
# =========================================================

def get_log_spaced_splits(min_dist, max_dist, n_bins):
    """Generates geometric splits."""
    start = np.log10(max(min_dist, 1.0))
    end = np.log10(max_dist)
    log_points = np.logspace(start, end, n_bins + 1)
    splits = [0.0] + list(log_points[1:])
    splits[-1] = 1e9 
    return splits

def load_land_mask_and_meta(tif_path):
    with rasterio.open(tif_path) as src:
        ocean_data = src.read(1)
        res_x = src.res[0]
        units = src.crs.linear_units if src.crs else None
        
        if (units and 'metre' in units.lower()) or (res_x > 100):
            cell_size_km = res_x / 1000.0
        elif res_x < 10:
            print("Warning: Units appear to be degrees. Approximating 111km/deg.")
            cell_size_km = res_x * 111.0
        else:
            cell_size_km = res_x
            
        if src.nodata is not None:
             ocean_data = np.where(ocean_data == src.nodata, 0, ocean_data)
        
        # Invert: 1=Land, 0=Ocean
        land_mask = 1.0 - (ocean_data > 0).astype(np.float32)
        
    return land_mask, cell_size_km

def visualize_results(Z_disp, labels, land_mask, output_dir, year):
    """
    Generates figures for the top 3 Z-features.
    MASKS OCEAN PIXELS for clean visualization.
    """
    data = Z_disp[0] # (Ny, Nx, M, K)
    Ny, Nx, M, K = data.shape
    
    # Masking for visualization: Set Ocean to NaN so it plots transparently
    land_mask_bc = land_mask[:, :, None, None] # Broadcast to data shape
    data_vis = np.where(land_mask_bc > 0.5, data, np.nan)
    
    directions = ['NORTH', 'SOUTH', 'EAST', 'WEST']
    
    # Infer number of bins from K
    n_bins = K // 4
    
    num_feats_to_plot = min(M, 3)
    
    print(f"\n--- Generating Visualizations for Year {year} ---")
    
    for f in range(num_feats_to_plot):
        fig, axes = plt.subplots(4, n_bins, figsize=(5*n_bins, 16), constrained_layout=True)
        fig.suptitle(f"Path Integrals: Year {year} | Feature Z_{f}", fontsize=20)
        
        # Calculate robust global limits for this feature (ignoring NaNs)
        feat_data = data_vis[:, :, f, :]
        vmin = np.nanpercentile(feat_data, 2)
        vmax = np.nanpercentile(feat_data, 98)
        
        for i, direction in enumerate(directions):
            for j in range(n_bins):
                k_idx = i * n_bins + j
                if k_idx >= len(labels): continue
                
                label_text = labels[k_idx]
                ax = axes[i, j]
                map_data = data_vis[:, :, f, k_idx]
                
                im = ax.imshow(map_data, origin='upper', cmap='viridis', vmin=vmin, vmax=vmax)
                ax.set_title(f"{direction} - Bin {j+1}\n({label_text})", fontsize=10)
                ax.axis('off')
                
        fig.colorbar(im, ax=axes, shrink=0.5, label=f"Integrated Z_{f} Value")
        
        save_path = os.path.join(output_dir, f"vis_year_{year}_feature_{f}.png")
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Saved {save_path}")

def main(args):
    print(f"--- Processing Single Year: {args.year} ---")
    
    # 1. FILE PATHS
    z_filename = f"Z_latent_{args.year}.npy"
    z_path = os.path.join(args.input_dir, z_filename)
    tif_path = os.path.join(args.input_dir, "ocean_mask_4km.tif")
    
    # 2. LOAD LAND MASK & METADATA
    if not os.path.exists(tif_path):
        print(f"Error: Could not find mask file: {tif_path}")
        return

    land_mask_np, cell_size_km = load_land_mask_and_meta(tif_path)
    print(f"Grid: {land_mask_np.shape} | Cell: {cell_size_km:.2f} km")

    # 3. LOAD Z DATA
    if not os.path.exists(z_path):
        print(f"Error: Could not find {z_path}")
        return

    print(f"Loading {z_filename}...")
    Z_year = jnp.load(z_path)
    if Z_year.ndim == 3: Z_year = Z_year[None, ...]
        
    _, Ny, Nx, M = Z_year.shape
    Ly, Lx = 2*Ny-1, 2*Nx-1
    land_mask = jnp.array(land_mask_np, dtype=jnp.float32)
    
    # 4. BUILD KERNELS (UPDATED: Geometric Splits)
    # Using 3 robust bins: ~150km, ~600km, 600km+
    splits = get_log_spaced_splits(min_dist=30.0, max_dist=3000.0, n_bins=3)
    
    print(f"Using Geometric Splits (km): {[f'{x:.1f}' for x in splits]}")
    
    kernel_stack, labels = make_radial_directional_kernels(
        Lx, Ly, 
        cell_size=cell_size_km, 
        radii_splits=splits
    )
    
    # 5. INTEGRATE (GPU)
    print(f"Integrating on GPU (Steps={args.steps})...")
    start = time.time()
    
    Z_disp = integrate_paths(Z_year, kernel_stack, land_mask, steps=args.steps)
    Z_disp.block_until_ready()
    
    print(f"Done in {time.time() - start:.2f}s")
    
    # 6. SAVE DATA
    os.makedirs(args.output_dir, exist_ok=True)
    out_name = f"Z_disp_{args.year}.npz"
    save_path = os.path.join(args.output_dir, out_name)
    
    np.savez_compressed(
        save_path, 
        Z_disp=Z_disp, 
        cell_size_km=cell_size_km,
        labels=labels,
        land_mask=land_mask_np
    )
    
    print(f"Saved processed data to {save_path}")
    
    # 7. VISUALIZE
    visualize_results(Z_disp, labels, land_mask_np, args.output_dir, args.year)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=str, default="1990")
    parser.add_argument("--input_dir", type=str, default="/home/breallis/datasets/latent_avian_community_similarities")
    parser.add_argument("--output_dir", type=str, default="/home/breallis/processed_data/datasets/latent_avian_path_diagnostics")
    parser.add_argument("--steps", type=int, default=20) 
    
    args = parser.parse_args()
    main(args)