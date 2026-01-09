import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

def correlate_climate_drivers_robust():
    # --- CONFIG ---
    DATA_DIR = "/home/breallis/datasets"
    Z_DIR = "/home/breallis/dev/range_limits_pymc/misc_outputs/ruzicka_sweep_global/sigma_1.5/results/spacetime_cube"
    OUT_DIR = os.path.join(Z_DIR, "analysis_plots_raw")
    
    # Locate State Files
    hist_candidates = [
        f"{DATA_DIR}/smoothed_prism_bui/yearly_states",
        f"{DATA_DIR}/smoothed_prism_bui"
    ]
    HIST_DIR = next((d for d in hist_candidates if os.path.exists(d)), None)
    
    if HIST_DIR is None:
        print("CRITICAL: Could not find yearly states directory.")
        return

    # 1. LOAD DATA
    print("Loading 1900 vs 2024 Data...")
    try:
        # Load Z (Turnover)
        z_1900 = np.load(os.path.join(Z_DIR, "Z_latent_1900.npy"))
        z_2024 = np.load(os.path.join(Z_DIR, "Z_latent_2024.npy"))
        
        # Load PRISM (Environment)
        s_1900 = np.load(os.path.join(HIST_DIR, "state_1900_bio_ema10.npz"))
        p_1900 = s_1900['prism']
        
        f_s24 = os.path.join(HIST_DIR, "state_2024_bio_ema10.npz")
        if not os.path.exists(f_s24): f_s24 = os.path.join(HIST_DIR, "state_2023_bio_ema10.npz")
        p_2024 = np.load(f_s24)['prism']
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # 2. CALCULATE CLIMATE DELTAS
    print("Computing Climate Deltas...")
    
    # A. Temperature Delta (Annual Mean)
    # Tmean is channels 36-47
    t_1900_annual = np.nanmean(p_1900[:, :, 36:48], axis=2)
    t_2024_annual = np.nanmean(p_2024[:, :, 36:48], axis=2)
    delta_temp = t_2024_annual - t_1900_annual 
    
    # B. Precipitation Delta (Total Annual)
    # PPT is channels 0-11
    ppt_1900_total = np.nansum(p_1900[:, :, 0:12], axis=2)
    ppt_2024_total = np.nansum(p_2024[:, :, 0:12], axis=2)
    delta_ppt = ppt_2024_total - ppt_1900_total
    
    # C. Community Turnover (Magnitude)
    delta_z = np.linalg.norm(z_2024 - z_1900, axis=2)
    
    # 3. STRICT MASKING (The Fix)
    # We must mask ANY pixel that is NaN in ANY of the 3 maps
    mask_z = ~np.isnan(delta_z)
    mask_t = ~np.isnan(delta_temp)
    mask_p = ~np.isnan(delta_ppt)
    
    # Combined intersection
    strict_mask = mask_z & mask_t & mask_p
    
    # Additional check: Remove Infs if any division error occurred
    strict_mask = strict_mask & ~np.isinf(delta_temp) & ~np.isinf(delta_ppt)

    print(f"Valid pixels for correlation: {np.sum(strict_mask)}")
    
    if np.sum(strict_mask) < 100:
        print("Error: Too few valid pixels to correlate.")
        return

    dz_flat = delta_z[strict_mask]
    dt_flat = np.abs(delta_temp[strict_mask]) # Magnitude of warming/cooling
    dp_flat = np.abs(delta_ppt[strict_mask])  # Magnitude of wetting/drying
    
    # 4. CORRELATIONS
    r_temp, p_temp = pearsonr(dt_flat, dz_flat)
    r_ppt, p_ppt = pearsonr(dp_flat, dz_flat)
    
    print(f"\n>>> STATS: Drivers of Community Turnover")
    print(f"Correlation with Delta-Temperature (Abs): r = {r_temp:.4f}")
    print(f"Correlation with Delta-Precipitation (Abs): r = {r_ppt:.4f}")
    
    # 5. PLOTTING
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    # Map 1: Turnover
    im1 = axes[0].imshow(delta_z, cmap='magma', vmin=0)
    axes[0].set_title("Community Turnover (Delta Z)")
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    axes[0].axis('off')
    
    # Map 2: Temp Change
    # Use 'RdBu_r' centered on 0 
    val_max_t = np.nanmax(np.abs(delta_temp))
    im2 = axes[1].imshow(delta_temp, cmap='RdBu_r', vmin=-val_max_t, vmax=val_max_t)
    axes[1].set_title(f"Temperature Change (C)\n(Corr with Z: {r_temp:.2f})")
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    axes[1].axis('off')
    
    # Map 3: Precip Change
    # Use 'BrBG' (Brown=Dry, Green=Wet)
    val_max_p = np.nanmax(np.abs(delta_ppt))
    im3 = axes[2].imshow(delta_ppt, cmap='BrBG', vmin=-val_max_p, vmax=val_max_p)
    axes[2].set_title(f"Precipitation Change (mm)\n(Corr with Z: {r_ppt:.2f})")
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "drivers_climate_vs_turnover_robust.png"), dpi=300)
    print(f"Plots saved to {OUT_DIR}")

if __name__ == "__main__":
    correlate_climate_drivers_robust()