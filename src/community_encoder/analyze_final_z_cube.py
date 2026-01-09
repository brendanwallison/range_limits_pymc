import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def analyze_raw_z_evolution():
    # --- CONFIG ---
    Z_DIR = "/home/breallis/dev/range_limits_pymc/misc_outputs/ruzicka_sweep_global/sigma_1.5/results/spacetime_cube"
    OUT_DIR = os.path.join(Z_DIR, "analysis_plots_raw")
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # 1. SETUP
    years = range(1900, 2025)
    
    # We will track the first 3 latent dimensions (assuming they are the most dominant)
    # If your latent_dim is small (e.g., 4 or 8), you might want to plot all of them.
    NUM_DIMS_TO_PLOT = 3 
    
    # Storage
    # A. Global Trajectory: Mean Z values per year (T, D)
    global_means = []
    
    # B. Hovmöller: Mean Z values per Latitude row per year (T, H, D)
    # We need to know H first, so we'll init this on the first loop
    hovmoller_grids = None
    
    print(f"Interrogating raw Z-space evolution for dimensions 0-{NUM_DIMS_TO_PLOT-1}...")
    
    for i, year in enumerate(tqdm(years)):
        fpath = os.path.join(Z_DIR, f"Z_latent_{year}.npy")
        if not os.path.exists(fpath):
            print(f"Skipping {year} (file not found)")
            global_means.append([np.nan]*NUM_DIMS_TO_PLOT)
            continue
            
        # Load Z Cube: (H, W, D_total)
        z_cube = np.load(fpath)
        H, W, D_total = z_cube.shape
        
        # Initialize Hovmoller container once we know H
        if hovmoller_grids is None:
            hovmoller_grids = np.full((len(years), H, NUM_DIMS_TO_PLOT), np.nan)
        
        # A. Global Means (Centroid of the "community cloud")
        # Collapse H and W dimensions -> (D_total,)
        # nanmean handles the ocean/gaps automatically
        g_mean_all = np.nanmean(z_cube.reshape(-1, D_total), axis=0)
        global_means.append(g_mean_all[:NUM_DIMS_TO_PLOT])
        
        # B. Hovmöller Prep (Latitudinal Means)
        # Collapse Longitude (axis 1) -> (H, D_total)
        lat_means_all = np.nanmean(z_cube, axis=1)
        hovmoller_grids[i] = lat_means_all[:, :NUM_DIMS_TO_PLOT]

    global_means = np.array(global_means) # (T, 3)
    
    # =========================================================
    # 2. PLOT: PHASE SPACE TRAJECTORY (Z0 vs Z1)
    # =========================================================
    # "Is the community state drifting directionally?"
    print("Generating Phase Space Trajectory (Z0 vs Z1)...")
    plt.figure(figsize=(10, 8))
    
    # Scatter with color for time
    sc = plt.scatter(global_means[:, 0], global_means[:, 1], 
                     c=years, cmap='plasma', s=60, edgecolors='k', alpha=0.9)
    
    # Line to show continuity
    plt.plot(global_means[:, 0], global_means[:, 1], 'k-', alpha=0.3, lw=1)
    
    # Annotate decades for context
    for i, y in enumerate(years):
        if y % 20 == 0 or y == 1900 or y == 2024:
            plt.text(global_means[i, 0], global_means[i, 1], str(y), 
                     fontsize=9, fontweight='bold', ha='right', va='bottom')
    
    plt.xlabel("Latent Dimension 0 ($Z_0$)")
    plt.ylabel("Latent Dimension 1 ($Z_1$)")
    plt.title("Drift of Continental Community State (1900-2024)\n(Raw Latent Space Trajectory)")
    plt.colorbar(sc, label="Year")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUT_DIR, "trajectory_z0_z1.png"), dpi=300)
    plt.close()

    # =========================================================
    # 3. PLOT: HOVMOLLER DIAGRAMS (Z0, Z1, Z2)
    # =========================================================
    # "Are communities migrating North?"
    print("Generating Latitudinal Hovmöller Diagrams...")
    
    fig, axes = plt.subplots(1, NUM_DIMS_TO_PLOT, figsize=(6 * NUM_DIMS_TO_PLOT, 8))
    if NUM_DIMS_TO_PLOT == 1: axes = [axes] # Handle single plot case
    
    for d in range(NUM_DIMS_TO_PLOT):
        # Data shape is (Time, Lat). Transpose to (Lat, Time) for plotting
        # Time on X-axis, Latitude on Y-axis
        data = hovmoller_grids[:, :, d].T 
        
        # Center colorbar around the mean of the data for better contrast
        vmax = np.nanmax(np.abs(data))
        
        im = axes[d].imshow(data, aspect='auto', cmap='RdBu_r', 
                            extent=[years[0], years[-1], 0, H],
                            interpolation='nearest') 
                            # 'nearest' keeps pixels crisp; use 'bilinear' for smooth look
        
        axes[d].set_title(f"Hovmöller: Latent Dim $Z_{d}$")
        axes[d].set_xlabel("Year")
        if d == 0: axes[d].set_ylabel("Latitude Index (South -> North)")
        
        plt.colorbar(im, ax=axes[d], orientation='horizontal', pad=0.08, label="Mean Activation")
    
    plt.suptitle("Latitudinal Migration of Community States (1900-2024)", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "hovmoller_latitudinal_raw.png"), dpi=300)
    plt.close()
    
    # =========================================================
    # 4. PLOT: ANOMALY (DELTA) MAPS
    # =========================================================
    # "How different is 2024 from 1900?"
    # We just plot the Delta for Z0 to see where the biggest changes happened.
    print("Generating Total Change Map (2024 - 1900)...")
    
    fpath_1900 = os.path.join(Z_DIR, "Z_latent_1900.npy")
    fpath_2024 = os.path.join(Z_DIR, "Z_latent_2024.npy")
    
    if os.path.exists(fpath_1900) and os.path.exists(fpath_2024):
        z_1900 = np.load(fpath_1900)
        z_2024 = np.load(fpath_2024)
        
        # Calculate Delta Vector Magnitude: ||Z_2024 - Z_1900||
        # This collapses all dims into a single scalar "Magnitude of Change"
        delta_vec = z_2024 - z_1900
        change_mag = np.linalg.norm(delta_vec, axis=2) # (H, W)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(change_mag, cmap='magma', vmin=0)
        plt.colorbar(label="Euclidean Distance in Z-Space")
        plt.title("Total Community Shift Magnitude (1900 vs 2024)")
        plt.axis('off')
        plt.savefig(os.path.join(OUT_DIR, "map_total_magnitude_change.png"), dpi=300)
        plt.close()

    print(f"Analysis complete. See {OUT_DIR}")

if __name__ == "__main__":
    analyze_raw_z_evolution()