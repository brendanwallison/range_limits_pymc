import sys
import os
import time
import argparse
import glob
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import rasterio
from tqdm import tqdm

# --- Setup Paths ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# 1. Import Kernels
from src.model.build_kernels import make_radial_directional_kernels

# 2. Import Path Integration (Ensure resize_kernel_stack is fixed in here!)
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
    
    # print(f"    Generating plots for {num_feats_to_plot} features...")
    
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
                ax = fig.add_subplot(gs[i+1, j])
                
                map_data = data_vis_disp[:, :, f, k_idx]
                im = ax.imshow(map_data, origin='upper', cmap='viridis', vmin=vmin, vmax=vmax)
                ax.set_title(f"{direction} - Bin {j+1}\n({label_text})", fontsize=10)
                ax.axis('off')
                
        save_path = os.path.join(output_dir, f"vis_year_{year}_feature_{f}_comparison.png")
        plt.savefig(save_path, dpi=150)
        plt.close(fig)

def main(args):
    print(f"--- Starting Full Path Integration Pipeline ---")
    
    # 1. SETUP & FIND FILES
    tif_path = os.path.join(args.input_dir, "ocean_mask_4km.tif")
    if not os.path.exists(tif_path):
        print(f"Error: Mask file not found: {tif_path}")
        return

    # Find years
    if args.year == 'all':
        pattern = os.path.join(args.input_dir, "Z_latent_*.npy")
        files = glob.glob(pattern)
        years = []
        for f in files:
            # Filename format: Z_latent_1990.npy
            try:
                y = os.path.basename(f).split('_')[-1].split('.')[0]
                if y.isdigit():
                    years.append(y)
            except:
                continue
        years = sorted(years)
        print(f"Found {len(years)} years to process: {years}")
    else:
        years = [args.year]

    if not years:
        print("No files found matching Z_latent_*.npy")
        return

    # 2. LOAD STATIC DATA (Geometry & Kernels) - Run Once!
    print("\n--- Initializing Geometry & Kernels (Shared) ---")
    land_mask_np, cell_size_km = load_land_mask_and_meta(tif_path)
    print(f"Grid: {land_mask_np.shape} | Cell: {cell_size_km:.2f} km")
    
    Ny, Nx = land_mask_np.shape
    Ly, Lx = 2*Ny-1, 2*Nx-1
    land_mask = jnp.array(land_mask_np, dtype=jnp.float32)

    # Define Splits (Geometric / Log-Spaced)
    # Bin 1: 0-50km (Local)
    # Bin 2: 50-275km (Regional)
    # Bin 3: 275-1500km (Continental)
    splits = get_log_spaced_splits(min_dist=50.0, max_dist=1500.0, n_bins=3)
    print(f"Using Geometric Splits (km): {[f'{x:.1f}' for x in splits]}")
    
    kernel_stack, labels = make_radial_directional_kernels(
        Lx, Ly, 
        cell_size=cell_size_km, 
        radii_splits=splits
    )
    # Force compilation on a dummy input to avoid recompiling inside the loop
    print("Pre-compiling JAX kernels...")
    dummy_Z = jnp.zeros((1, Ny, Nx, 1)) # Small dummy
    _ = integrate_paths(dummy_Z, kernel_stack, land_mask, steps=2) 
    print("Compilation complete.\n")

    # 3. PROCESSING LOOP
    os.makedirs(args.output_dir, exist_ok=True)
    
    total_start = time.time()
    
    for year in tqdm(years, desc="Processing Years"):
        z_filename = f"Z_latent_{year}.npy"
        z_path = os.path.join(args.input_dir, z_filename)
        out_name = f"Z_disp_{year}.npz"
        save_path = os.path.join(args.output_dir, out_name)
        
        # Check if already exists (optional skip)
        # if os.path.exists(save_path): continue 

        # A. Load Z
        Z_year = jnp.load(z_path)
        if Z_year.ndim == 3: Z_year = Z_year[None, ...]
        
        # B. Integrate
        # Note: kernel_stack and land_mask are reused from memory!
        Z_disp = integrate_paths(Z_year, kernel_stack, land_mask, steps=args.steps)
        Z_disp.block_until_ready()
        
        # C. Save
        np.savez_compressed(
            save_path, 
            Z_disp=Z_disp, 
            Z_raw=Z_year, 
            cell_size_km=cell_size_km,
            labels=labels,
            land_mask=land_mask_np
        )
        
        # D. Visualize (First 3 features)
        if args.viz:
            visualize_results(Z_disp, Z_year, labels, land_mask_np, args.output_dir, year)

    print(f"\nAll done! Total time: {(time.time() - total_start)/60:.2f} minutes.")
    print(f"Output directory: {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=str, default="all", help="Year to process (e.g. '1990' or 'all')")
    parser.add_argument("--input_dir", type=str, default="/home/breallis/datasets/latent_avian_community_similarities")
    parser.add_argument("--output_dir", type=str, default="/home/breallis/processed_data/datasets/latent_avian_paths")
    parser.add_argument("--steps", type=int, default=20, help="Number of integration steps")
    parser.add_argument("--viz", action='store_true', default=True, help="Generate PNG visualizations")
    
    args = parser.parse_args()
    main(args)