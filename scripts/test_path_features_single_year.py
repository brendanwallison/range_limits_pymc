import sys
import os
import time
import argparse
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import rasterio
from jax.numpy.fft import fft2, ifft2
from tqdm import tqdm

# --- Setup Paths ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# 1. Import Kernels
from src.model.build_kernels import make_radial_directional_kernels

# 2. Import Path Integration
from src.model.build_path_features import integrate_paths

# =========================================================
# Helper Functions
# =========================================================

def get_log_spaced_splits(min_dist, max_dist, n_bins):
    """
    Generates geometric splits to ensure distinct spatial scales.
    Bin 1: Local | Bin 2: Regional | Bin 3: Continental
    """
    start = np.log10(max(min_dist, 1.0))
    end = np.log10(max_dist)
    log_points = np.logspace(start, end, n_bins + 1)
    splits = [0.0] + list(log_points[1:])
    splits[-1] = 1e9 
    return splits

def load_land_mask_and_meta(tif_path):
    """Loads TIF, returns Land Mask (1=Land, 0=Water) and Cell Size (km)."""
    with rasterio.open(tif_path) as src:
        ocean_data = src.read(1)
        res_x = src.res[0]
        units = src.crs.linear_units if src.crs else None
        
        # Heuristic unit conversion
        if (units and 'metre' in units.lower()) or (res_x > 100):
            print(f"  [Metadata] Detected units in Meters (Resolution: {res_x:.2f}m)")
            cell_size_km = res_x / 1000.0
        elif res_x < 10:
            print(f"  [Warning] Resolution {res_x} appears to be in Degrees.")
            print("  Approximating 1 deg ~ 111 km. PROJECTION RECOMMENDED.")
            cell_size_km = res_x * 111.0
        else:
            cell_size_km = res_x
            
        if src.nodata is not None:
             ocean_data = np.where(ocean_data == src.nodata, 0, ocean_data)
        
        # Invert: Input 1=Ocean -> Output 1=Land
        land_mask = 1.0 - (ocean_data > 0).astype(np.float32)
        
    return land_mask, cell_size_km

def visualize_results(Z_disp, Z_raw, labels, land_mask, output_dir, year):
    """
    Generates figures comparing Base Map vs Path Integrals.
    Layout: 
      Top: Base Feature (Local)
      Grid: Directional/Radial Integrals (Upstream)
    """
    # Z_disp: (1, Ny, Nx, M, K) -> take index 0 -> (Ny, Nx, M, K)
    # Z_raw:  (1, Ny, Nx, M)    -> take index 0 -> (Ny, Nx, M)
    
    data_disp = Z_disp[0]
    data_base = Z_raw[0]
    
    Ny, Nx, M, K = data_disp.shape
    
    # Masking: Set Ocean to NaN so it plots transparently
    land_mask_bc_disp = land_mask[:, :, None, None] 
    land_mask_bc_base = land_mask[:, :, None]
    
    data_vis_disp = np.where(land_mask_bc_disp > 0.5, data_disp, np.nan)
    data_vis_base = np.where(land_mask_bc_base > 0.5, data_base, np.nan)
    
    directions = ['NORTH', 'SOUTH', 'EAST', 'WEST']
    
    # Infer number of bins based on K (kernels)
    n_bins = K // 4
    
    num_feats_to_plot = min(M, 3)
    
    print(f"\n--- Generating Visualizations for Year {year} ---")
    
    for f in range(num_feats_to_plot):
        # Create GridSpec: 
        # Row 0: Base Map (Large)
        # Row 1-4: Directions
        fig = plt.figure(figsize=(5*n_bins, 20), constrained_layout=True)
        gs = gridspec.GridSpec(5, n_bins, figure=fig, height_ratios=[1.5, 1, 1, 1, 1])
        
        fig.suptitle(f"Path Integrals: Year {year} | Feature Z_{f}", fontsize=20)
        
        # --- 1. Plot Base Map (Top Row) ---
        ax_base = fig.add_subplot(gs[0, :])
        base_img = data_vis_base[:, :, f]
        
        # Calculate shared color limits for THIS feature
        # We want to compare Base vs Paths on the SAME SCALE
        # Combine data to find robust min/max
        all_vals = np.concatenate([base_img.flatten(), data_vis_disp[:, :, f, :].flatten()])
        vmin = np.nanpercentile(all_vals, 2)
        vmax = np.nanpercentile(all_vals, 98)
        
        im_base = ax_base.imshow(base_img, origin='upper', cmap='viridis', vmin=vmin, vmax=vmax)
        ax_base.set_title(f"BASE MAP (Local Z_{f})\nWhat is the environment HERE?", fontsize=14, fontweight='bold')
        ax_base.axis('off')
        plt.colorbar(im_base, ax=ax_base, orientation='vertical', shrink=0.8, label=f"Z_{f} Value")
        
        # --- 2. Plot Path Integrals (Rows 1-4) ---
        for i, direction in enumerate(directions):
            for j in range(n_bins):
                k_idx = i * n_bins + j
                if k_idx >= len(labels): continue
                
                label_text = labels[k_idx]
                
                # Grid rows start at index 1
                ax = fig.add_subplot(gs[i+1, j])
                
                map_data = data_vis_disp[:, :, f, k_idx]
                
                im = ax.imshow(map_data, origin='upper', cmap='viridis', vmin=vmin, vmax=vmax)
                ax.set_title(f"{direction} - Bin {j+1}\n({label_text})", fontsize=10)
                ax.axis('off')
                
        # Save
        save_path = os.path.join(output_dir, f"vis_year_{year}_feature_{f}_comparison.png")
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
    
    # 4. BUILD KERNELS (Geometric Splits)
    splits = get_log_spaced_splits(min_dist=50.0, max_dist=1500.0, n_bins=3)
    
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
        Z_raw=Z_year, # Optional: Save raw Z alongside for convenience
        cell_size_km=cell_size_km,
        labels=labels,
        land_mask=land_mask_np
    )
    
    print(f"Saved processed data to {save_path}")
    
    # 7. VISUALIZE (Passing Z_year explicitly for comparison)
    visualize_results(Z_disp, Z_year, labels, land_mask_np, args.output_dir, args.year)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=str, default="1990")
    parser.add_argument("--input_dir", type=str, default="/home/breallis/datasets/latent_avian_community_similarities")
    parser.add_argument("--output_dir", type=str, default="/home/breallis/processed_data/datasets/latent_avian_path_diagnostics")
    parser.add_argument("--steps", type=int, default=20) 
    
    args = parser.parse_args()
    main(args)