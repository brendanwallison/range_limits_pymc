import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# --- CONFIGURATION ---
# Path to the OUTPUT of ingest_model_data.py
INPUT_DIR = "/home/breallis/processed_data/model_inputs/numpyro_input"
METADATA_PATH = os.path.join(INPUT_DIR, "metadata.pkl")
OUTPUT_PLOT_PATH = "check_ingested_data.png"

def plot_ingested_data():
    print(f"Loading metadata from {METADATA_PATH}...")
    
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(f"Metadata not found at {METADATA_PATH}. Run ingest_model_data.py first.")
        
    with open(METADATA_PATH, 'rb') as f:
        meta = pickle.load(f)
        
    # --- EXTRACT DATA ---
    init_pop = meta['initpop_latent']
    years = meta['years']
    land_mask = meta['land_mask']
    
    # Reconstruct Observation Arrays
    # obs_time_indices is 0-indexed relative to 'years'
    obs_time_idx = meta['obs_time_indices']
    obs_rows = meta['obs_rows']
    obs_cols = meta['obs_cols']
    obs_counts = meta['observed_results']
    
    # Define Years to Check
    start_year_idx = 0
    start_year = years[start_year_idx]
    
    # Try to find a year with real data (e.g., 1975) for alignment check
    try:
        # Find index for 1975 or last year
        check_year = 1975
        if check_year in years:
            later_year_idx = np.where(years == check_year)[0][0]
        else:
            later_year_idx = len(years) - 1
            check_year = years[later_year_idx]
    except:
        later_year_idx = 0
        check_year = start_year

    print(f"Checking Start Year: {start_year}")
    print(f"Checking Data Year: {check_year}")

    # --- PLOTTING ---
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    # --- PLOT 1: INITIALIZATION MAP ---
    ax1 = axes[0]
    # Mask zeros to see the difference between Land(0) and Ocean
    # But init_pop is density. We want to see 0.5 vs 0.05.
    init_masked = np.ma.masked_where(init_pop == 0, init_pop)
    
    # Background: Land Mask
    ax1.imshow(land_mask, cmap='gray_r', vmin=0, vmax=2, alpha=0.3, origin='upper')
    
    # Overlay: Initialization
    im1 = ax1.imshow(init_masked, cmap='OrRd', vmin=0, vmax=0.6, origin='upper')
    ax1.set_title("1. Model Initialization (t=0)\n(Red=Core 0.5, Orange=Margin 0.05)")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # --- PLOT 2: PSEUDO-ZEROS (Start Year) ---
    ax2 = axes[1]
    ax2.imshow(land_mask, cmap='gray_r', vmin=0, vmax=2, alpha=0.3, origin='upper')
    
    mask_start = (obs_time_idx == start_year_idx)
    rows_start = obs_rows[mask_start]
    cols_start = obs_cols[mask_start]
    counts_start = obs_counts[mask_start]
    
    # Plot Zeros
    zeros_mask = (counts_start == 0)
    ax2.scatter(
        cols_start[zeros_mask], rows_start[zeros_mask], 
        c='blue', marker='x', s=20, alpha=0.6, label='Pseudo-Zero (Uninvaded)'
    )
    
    # Plot Real (if any exist in start year)
    real_mask = (counts_start > 0)
    if np.any(real_mask):
        ax2.scatter(
            cols_start[real_mask], rows_start[real_mask], 
            c='red', s=20, label='Real Obs'
        )
        
    ax2.set_title(f"2. Observations in {start_year}\n(Blue 'X' should cover East)")
    ax2.legend(loc='upper right')

    # --- PLOT 3: REAL DATA ALIGNMENT (Later Year) ---
    ax3 = axes[2]
    ax3.imshow(land_mask, cmap='gray_r', vmin=0, vmax=2, alpha=0.3, origin='upper')
    
    mask_later = (obs_time_idx == later_year_idx)
    rows_later = obs_rows[mask_later]
    cols_later = obs_cols[mask_later]
    counts_later = obs_counts[mask_later]
    
    # Plot Real Counts
    # We filter for counts > 0 to verify alignment of detections
    pos_mask = (counts_later > 0)
    
    if np.any(pos_mask):
        sc = ax3.scatter(
            cols_later[pos_mask], rows_later[pos_mask], 
            c=counts_later[pos_mask], cmap='magma', s=15, alpha=0.8, 
            norm=mcolors.LogNorm() # Log scale for counts
        )
        plt.colorbar(sc, ax=ax3, fraction=0.046, pad=0.04, label="Count (Log)")
    else:
        ax3.text(0.5, 0.5, "No positive counts this year", ha='center', transform=ax3.transAxes)
        
    ax3.set_title(f"3. Real Data Check ({check_year})\n(Verifies Land Mask Alignment)")

    plt.tight_layout()
    print(f"Saving diagnostic plot to {OUTPUT_PLOT_PATH}...")
    plt.savefig(OUTPUT_PLOT_PATH)
    print("Done.")

if __name__ == "__main__":
    plot_ingested_data()