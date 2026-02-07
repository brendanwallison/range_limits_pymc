import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# --- CONFIGURATION ---
# Path to the output of ingest_bbs_data.py
BBS_NPZ_PATH = "/home/breallis/datasets/bbs_2024_release/bbs_data_for_python.npz"
OUTPUT_PLOT_PATH = "check_bbs_npz.png"

def check_bbs_npz():
    print(f"Loading {BBS_NPZ_PATH}...")
    
    if not os.path.exists(BBS_NPZ_PATH):
        raise FileNotFoundError(f"NPZ not found. Run ingest_bbs_data.py first.")
        
    data = np.load(BBS_NPZ_PATH)
    
    # 1. Extract Grid info
    # Ingest script saves 1=Land, 0=Ocean (or vice versa depending on logic). 
    # Usually: land mask is boolean where True=Land.
    land_mask = data['land']
    Ny, Nx = land_mask.shape
    print(f"Grid Dimensions: {Ny}x{Nx}")
    
    # 2. Extract Initialization Hull (Indices)
    init_rows = data['initpop_rows']
    init_cols = data['initpop_cols']
    print(f"Initialization Pixels (Naive Hull): {len(init_rows)}")
    
    # 3. Extract Observations
    obs_rows = data['obs_rows']
    obs_cols = data['obs_cols']
    obs_year = data['obs_year']
    obs_counts = data['observed_results']
    N_pseudo = int(data['N_pseudo'])
    N_obs = int(data['N_obs']) # This is usually N_real
    
    print(f"Total Observation Records: {len(obs_rows)}")
    print(f"  Pseudo-Zeros (Uninvaded East): {N_pseudo}")
    print(f"  Real BBS Data: {len(obs_rows) - N_pseudo}")

    # --- PLOTTING ---
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    # PLOT A: Initialization Hull (Native Range)
    ax1 = axes[0]
    ax1.imshow(land_mask, cmap='gray_r', vmin=0, vmax=2, alpha=0.3, origin='upper')
    
    # Reconstruct init mask for visualization
    init_map = np.zeros_like(land_mask, dtype=float)
    init_map[init_rows, init_cols] = 1.0
    
    ax1.imshow(np.ma.masked_equal(init_map, 0), cmap='Reds', alpha=0.6, origin='upper')
    ax1.set_title(f"A. Naive Initialization Hull\n(Should cover West/SW US)")
    
    # PLOT B: Pseudo-Zeros (First N_pseudo records)
    ax2 = axes[1]
    ax2.imshow(land_mask, cmap='gray_r', vmin=0, vmax=2, alpha=0.3, origin='upper')
    
    p_rows = obs_rows[:N_pseudo]
    p_cols = obs_cols[:N_pseudo]
    
    # Scatter a sample if too dense
    if len(p_rows) > 10000:
        idx = np.random.choice(len(p_rows), 10000, replace=False)
        ax2.scatter(p_cols[idx], p_rows[idx], c='blue', s=1, alpha=0.3, label='Pseudo-Zero (Sample)')
    else:
        ax2.scatter(p_cols, p_rows, c='blue', s=1, alpha=0.3, label='Pseudo-Zero')
        
    ax2.set_title(f"B. Pseudo-Zeros (Uninvaded Range)\n(Should cover East US)")
    ax2.legend()

    # PLOT C: Real Observations (Last Year Check)
    ax3 = axes[2]
    ax3.imshow(land_mask, cmap='gray_r', vmin=0, vmax=2, alpha=0.3, origin='upper')
    
    # Filter for Real Data (after N_pseudo)
    real_rows = obs_rows[N_pseudo:]
    real_cols = obs_cols[N_pseudo:]
    real_years = obs_year[N_pseudo:]
    real_counts = obs_counts[N_pseudo:]
    
    # Find a year with good data (e.g., 2000)
    target_year = 2000
    if target_year not in real_years:
        target_year = real_years.max()
        
    mask_yr = (real_years == target_year)
    yr_rows = real_rows[mask_yr]
    yr_cols = real_cols[mask_yr]
    yr_counts = real_counts[mask_yr]
    
    # Plot positive counts
    pos_mask = yr_counts > 0
    if np.any(pos_mask):
        sc = ax3.scatter(
            yr_cols[pos_mask], yr_rows[pos_mask], 
            c=yr_counts[pos_mask], cmap='magma', s=5, alpha=0.8
        )
        plt.colorbar(sc, ax=ax3, fraction=0.046, label='Count')
        
    ax3.set_title(f"C. Real Observations ({target_year})\n(Sanity Check Alignment)")

    plt.tight_layout()
    print(f"Saving diagnostic to {OUTPUT_PLOT_PATH}...")
    plt.savefig(OUTPUT_PLOT_PATH)
    print("Done.")

if __name__ == "__main__":
    check_bbs_npz()