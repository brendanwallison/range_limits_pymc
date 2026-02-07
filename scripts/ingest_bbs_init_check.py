import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# --- CONFIGURATION ---
BBS_NPZ_PATH = "/home/breallis/datasets/bbs_2024_release/bbs_data_for_python.npz"
OUTPUT_PLOT_PATH = "check_bbs_npz.png"

def check_bbs_npz():
    print(f"Loading {BBS_NPZ_PATH}...")
    
    if not os.path.exists(BBS_NPZ_PATH):
        raise FileNotFoundError(f"NPZ not found.")
        
    data = np.load(BBS_NPZ_PATH)
    
    land_mask = data['land']
    init_density = data['initpop_density'] # Load the new float map
    
    # Stats
    print(f"Initialization Map Stats:")
    print(f"  Core Pixels (0.5): {np.sum(init_density == 0.5)}")
    print(f"  Margin Pixels (0.01): {np.sum(init_density == 0.01)}")
    
    # Extract Obs
    obs_rows = data['obs_rows']
    obs_cols = data['obs_cols']
    N_pseudo = int(data['N_pseudo'])

    # --- PLOTTING ---
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # PLOT A: Core/Margin Map
    ax1 = axes[0]
    ax1.imshow(land_mask, cmap='gray_r', vmin=0, vmax=2, alpha=0.3, origin='upper')
    
    # Mask zeros
    masked_init = np.ma.masked_where(init_density == 0, init_density)
    
    im = ax1.imshow(masked_init, cmap='OrRd', vmin=0, vmax=0.6, origin='upper')
    ax1.set_title("A. Core (Red) vs Margin (Orange) Initialization")
    plt.colorbar(im, ax=ax1, fraction=0.046)
    
    # PLOT B: Pseudo-Zeros (Buffer Check)
    ax2 = axes[1]
    ax2.imshow(land_mask, cmap='gray_r', vmin=0, vmax=2, alpha=0.3, origin='upper')
    
    # Plot pseudo zeros
    p_rows = obs_rows[:N_pseudo]
    p_cols = obs_cols[:N_pseudo]
    
    # Sample if dense
    if len(p_rows) > 10000:
        idx = np.random.choice(len(p_rows), 10000, replace=False)
        ax2.scatter(p_cols[idx], p_rows[idx], c='blue', s=1, alpha=0.3, label='Pseudo-Zero (Sample)')
    else:
        ax2.scatter(p_cols, p_rows, c='blue', s=1, alpha=0.3, label='Pseudo-Zero')
        
    ax2.set_title("B. Uninvaded East (1000km Buffer)")
    ax2.legend()

    plt.tight_layout()
    print(f"Saving diagnostic to {OUTPUT_PLOT_PATH}...")
    plt.savefig(OUTPUT_PLOT_PATH)
    print("Done.")

if __name__ == "__main__":
    check_bbs_npz()