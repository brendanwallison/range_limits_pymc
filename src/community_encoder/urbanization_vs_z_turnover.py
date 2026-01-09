import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

def overlay_urban_vs_turnover():
    # --- CONFIGURATION ---
    DATA_DIR = "/home/breallis/datasets"
    Z_DIR = "/home/breallis/dev/range_limits_pymc/misc_outputs/ruzicka_sweep_global/sigma_1.5/results/spacetime_cube"
    OUT_DIR = os.path.join(Z_DIR, "analysis_plots_raw")
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # Locate the BUI State Files
    # Try the subdirectory first (where they are usually generated)
    hist_candidates = [
        f"{DATA_DIR}/smoothed_prism_bui/yearly_states",
        f"{DATA_DIR}/smoothed_prism_bui"
    ]
    HIST_DIR = next((d for d in hist_candidates if os.path.exists(d)), hist_candidates[0])

    print(f"Loading BUI states from: {HIST_DIR}")
    print(f"Loading Z-Cubes from:  {Z_DIR}")

    # 1. LOAD DATA (1900 vs 2024)
    try:
        # Load Latent Community State (Z)
        z_1900 = np.load(os.path.join(Z_DIR, "Z_latent_1900.npy"))
        
        # Try 2024, fallback to 2023 if the simulation stopped early
        f_z24 = os.path.join(Z_DIR, "Z_latent_2024.npy")
        if not os.path.exists(f_z24):
            print("Z_2024 not found, falling back to 2023...")
            f_z24 = os.path.join(Z_DIR, "Z_latent_2023.npy")
        z_2024 = np.load(f_z24)

        # Load Urban State (BUI)
        # These .npz files contain 'bui' (H, W, 7) and 'prism'
        s_1900 = np.load(os.path.join(HIST_DIR, "state_1900_bio_ema10.npz"))
        bui_1900 = s_1900['bui']

        f_s24 = os.path.join(HIST_DIR, "state_2024_bio_ema10.npz")
        if not os.path.exists(f_s24):
            f_s24 = os.path.join(HIST_DIR, "state_2023_bio_ema10.npz")
        s_2024 = np.load(f_s24)
        bui_2024 = s_2024['bui']
        
    except FileNotFoundError as e:
        print(f"CRITICAL ERROR: Missing files. {e}")
        return

    # 2. CALCULATE MAGNITUDE OF CHANGE
    print("Computing change vectors...")
    
    # A. Community Turnover (Euclidean distance in Z-space)
    # || Z_2024 - Z_1900 ||
    delta_z_vec = z_2024 - z_1900
    delta_z_mag = np.linalg.norm(delta_z_vec, axis=2)

    # B. Urban Intensification (Euclidean distance in Transformed BUI-space)
    # We use the power transform (x**0.1) because that is the input manifold the model learned.
    # This highlights the "First Impact" of settlement (0->1) more than densification (50->51).
    bui_1900_t = bui_1900**0.1
    bui_2024_t = bui_2024**0.1
    
    delta_b_vec = bui_2024_t - bui_1900_t
    delta_b_mag = np.linalg.norm(delta_b_vec, axis=2)

    # 3. STATISTICAL CORRELATION
    # Flatten and mask NaNs (Ocean)
    mask = ~np.isnan(delta_z_mag)
    
    dz_flat = delta_z_mag[mask]
    db_flat = delta_b_mag[mask]
    
    corr, p_val = pearsonr(db_flat, dz_flat)
    print(f"\n>>> STATS: Pixel-wise Correlation (Turnover vs. Urbanization): r = {corr:.4f}")
    
    # 4. PLOTTING
    print("Generating Overlay Plots...")
    
    # Plot 1: Side-by-Side Comparison Maps
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    
    # Community Map
    im1 = axes[0].imshow(delta_z_mag, cmap='magma', vmin=0)
    axes[0].set_title("Avian Community Turnover Magnitude (1900 - 2024)\n(Euclidean Shift in Latent Space)", fontsize=14)
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], orientation='horizontal', pad=0.03, fraction=0.04)
    
    # Urban Map
    # Use 'inferno' to make the cities "burn" visually
    im2 = axes[1].imshow(delta_b_mag, cmap='inferno', vmin=0)
    axes[1].set_title("Urban Intensification Magnitude (1900 - 2024)\n(Shift in BUI Space)", fontsize=14)
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], orientation='horizontal', pad=0.03, fraction=0.04)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "compare_turnover_vs_urbanization.png"), dpi=300)
    
    # Plot 2: 2D Scatter/Density Plot
    # Shows the functional relationship
    plt.figure(figsize=(10, 8))
    plt.hist2d(db_flat, dz_flat, bins=100, cmap='viridis', norm=plt.Normalize(vmin=0))
    plt.colorbar(label="Count of Pixels")
    
    # Simple linear fit for visual guide
    if len(db_flat) > 0:
        m, b = np.polyfit(db_flat, dz_flat, 1)
        x_fit = np.linspace(db_flat.min(), db_flat.max(), 100)
        plt.plot(x_fit, m*x_fit + b, 'r--', lw=2, label=f"Trend (r={corr:.2f})")
    
    plt.xlabel("Magnitude of Urban Change (Delta BUI)")
    plt.ylabel("Magnitude of Community Turnover (Delta Z)")
    plt.title("Correlation: Does Building Cities Drive Community Shift?")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(OUT_DIR, "scatter_turnover_vs_urban.png"), dpi=300)
    
    print(f"Analysis Complete. Plots saved to {OUT_DIR}")

if __name__ == "__main__":
    overlay_urban_vs_turnover()