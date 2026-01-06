import os
import glob
import re
import numpy as np
import pandas as pd
import torch
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import rasterio
import json

# ============================================================
# 1. Robust Data Loader (Species-Major Order)
# ============================================================
def load_tifs_structured(folder, pattern="*_abundance_median_*.tif"):
    """
    Parses filenames to enforce strict (Species, Time) ordering.
    Returns: stack (H, W, S*T), meta dict
    """
    files = sorted(glob.glob(os.path.join(folder, pattern)))
    if not files:
        raise ValueError(f"No files found in {folder} matching {pattern}")

    # Regex captures <species> and <date>
    regex = re.compile(r"([a-z0-9]+)_abundance_median_(\d{4}-\d{2}-\d{2})")
    records = []
    for fpath in files:
        fname = os.path.basename(fpath)
        match = regex.match(fname)
        if match:
            records.append({
                "species": match.group(1),
                "date": datetime.strptime(match.group(2), "%Y-%m-%d"),
                "path": fpath
            })
        else:
            print(f"Skipping non-matching file: {fname}")

    df = pd.DataFrame(records)
    if df.empty: raise ValueError("No filenames matched regex.")
    
    # Validation: Ensure grid is complete
    n_species = df['species'].nunique()
    n_weeks = df['date'].nunique()
    if len(df) != n_species * n_weeks:
        raise ValueError(f"Grid incomplete. Expected {n_species*n_weeks} files, found {len(df)}.")

    # Sort -> Species Major: Species 1 [Week 1..52], Species 2 [Week 1..52]...
    df_sorted = df.sort_values(by=['species', 'date'])
    ordered_paths = df_sorted['path'].tolist()
    
    # Load first file to get dimensions
    with rasterio.open(ordered_paths[0]) as src:
        H, W = src.shape
        
    print(f"Loading {len(ordered_paths)} rasters ({n_species} sp x {n_weeks} wks)...")
    
    # Pre-allocate stack
    full_stack = np.zeros((H, W, len(ordered_paths)), dtype=np.float32)
    
    for i, p in enumerate(ordered_paths):
        with rasterio.open(p) as src:
            full_stack[:, :, i] = src.read(1)

    return full_stack, {"n_species": n_species, "n_weeks": n_weeks}

# ============================================================
# 2. Smoothing Transform (No Hellinger)
# ============================================================
def smooth_abundances(ebird_flat, n_weeks, sigma):
    """
    1. Reshapes to (N, Species, Weeks)
    2. Applies Gaussian blur along Time axis (if sigma > 0)
    3. Returns flattened array (N, S*T) preserving absolute abundance
    """
    N, D = ebird_flat.shape
    n_species = D // n_weeks
    
    # Reshape: N x Species x Weeks
    data_3d = ebird_flat.reshape(N, n_species, n_weeks)
    
    if sigma > 1e-5:
        # mode='wrap' assumes the yearly cycle connects Week 52 -> Week 1
        data_smoothed = gaussian_filter1d(data_3d, sigma=sigma, axis=-1, mode='wrap')
    else:
        data_smoothed = data_3d
        
    # Ensure non-negative (smoothing might introduce negligible negatives near 0)
    data_smoothed = np.maximum(data_smoothed, 0.0)
    
    # Flatten back to N x D
    return data_smoothed.reshape(N, -1)

# ============================================================
# 3. Global Ruzicka Kernel & Nyström
# ============================================================
def compute_optimal_latent_z_ruzicka(ebird_flat, n_species, n_weeks, latent_dim, n_landmarks=10000, device='cuda'):
    """
    Computes Mercer features using the GLOBAL Ruzicka Kernel (Generalized Jaccard).
    
    K_global(x,y) = Sum_all( min(x_i, y_i) ) / Sum_all( max(x_i, y_i) )
    
    This captures cross-species interactions and usually results in a higher rank 
    embedding than the additive version.
    """
    import torch
    
    N, D = ebird_flat.shape
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = 'cpu'
    
    # --- A. Setup ---
    # We no longer need to reshape to (N, Species, Weeks) for the kernel logic.
    # We treat the entire S*T vector as the feature descriptor.
    
    # Select Landmarks
    idx_lm = np.random.choice(N, min(N, n_landmarks), replace=False)
    X_lm_np = ebird_flat[idx_lm] # (M, D)
    M = X_lm_np.shape[0]

    print(f"Computing Exact Global Kernel on {M} landmarks (Dim={D})...")

    # --- B. Compute K_MM (Landmark vs Landmark) Exactly ---
    try:
        T_lm = torch.tensor(X_lm_np, device=device, dtype=torch.float32)
    except RuntimeError:
        print("VRAM limit warning: Landmarks too large. Reduce n_landmarks.")
        return None

    # Global Ruzicka Logic:
    # 2*min(a,b) = a + b - |a-b|
    # 2*max(a,b) = a + b + |a-b|
    
    # Sum over ALL dimensions (Species * Time)
    sum_lm = T_lm.sum(dim=1, keepdim=True) # (M, 1)
    
    # L1 Distance on full vectors: (M, M)
    # This captures the total absolute difference across the entire community vector
    l1_dist = torch.cdist(T_lm, T_lm, p=1)
    
    # Terms for Global Ruzicka
    sum_plus_sum = sum_lm + sum_lm.T
    
    numerator = 0.5 * (sum_plus_sum - l1_dist)
    denominator = 0.5 * (sum_plus_sum + l1_dist)
    
    # Handle Orthogonal Absence (Double Zero)
    mask = denominator > 1e-6
    K_mm_total = torch.zeros_like(numerator)
    K_mm_total[mask] = numerator[mask] / denominator[mask]

    # --- C. Eigendecomposition of Master Kernel ---
    L, U = torch.linalg.eigh(K_mm_total.cpu())
    
    # Sort descending
    idx_sort = torch.argsort(L, descending=True)[:latent_dim]
    L = L[idx_sort]
    U = U[:, idx_sort].to(device) # (M, latent_dim)
    
    # Filter small eigenvalues
    L = torch.clamp(L, min=1e-10).to(device)
    
    # Standard Nystrom Projection Matrix: P = U * L^(-0.5)
    proj_mat = U * torch.rsqrt(L) 

    # --- D. Batch Projection of All Points (N) ---
    Z_opt = np.zeros((N, latent_dim), dtype=np.float32)
    batch_size = 5000 # Adjusted slightly for potentially larger D processing
    
    print(f"Projecting {N} points in batches...")
    
    # We can use the original array for batches to save GPU memory
    X_all_np = ebird_flat 

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            
            batch_np = X_all_np[start:end]
            T_batch = torch.tensor(batch_np, device=device, dtype=torch.float32)
            
            # --- Global Ruzicka Kernel (Batch vs Landmarks) ---
            sum_b = T_batch.sum(dim=1, keepdim=True) # (B, 1)
            # sum_lm is already (M, 1)
            
            # Rectangular L1 dist (B, M)
            l1_rect = torch.cdist(T_batch, T_lm, p=1)
            
            sum_plus_sum_rect = sum_b + sum_lm.T # Broadcast (B, M)
            
            num = 0.5 * (sum_plus_sum_rect - l1_rect)
            den = 0.5 * (sum_plus_sum_rect + l1_rect)
            
            mask = den > 1e-6
            K_batch_lm = torch.zeros_like(num)
            K_batch_lm[mask] = num[mask] / den[mask]
            
            # Project: Z = K_NM * P
            z_batch = K_batch_lm @ proj_mat
            
            # NO RESCALING by sqrt(S) needed for Global Ruzicka
            # It is already naturally normalized in [0, 1]
            
            Z_opt[start:end] = z_batch.cpu().numpy()

    return Z_opt

def compute_kernel_diagnostics_ruzicka(z, ebird_flat, n_species, n_weeks, max_samples=500):
    """
    Computes RMSE between the Nyström approximation (ZZ^T) and the
    Exact GLOBAL Ruzicka Kernel (K_true).
    """
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Sample indices
    N = z.shape[0]
    idx = np.random.choice(N, min(N, max_samples), replace=False)
    
    # 2. Subset data
    z_s = torch.tensor(z[idx], device=device, dtype=torch.float32)  # (M, d)
    
    # Load raw features for subset
    X_s = torch.tensor(ebird_flat[idx], device=device, dtype=torch.float32) # (M, D)
    M = len(idx)

    # 3. Approximate kernel from embedding
    K_approx = z_s @ z_s.T 

    # 4. Exact Global Ruzicka Kernel
    sum_s = X_s.sum(dim=1, keepdim=True)
    l1_dist = torch.cdist(X_s, X_s, p=1)
    
    sum_plus_sum = sum_s + sum_s.T
    num = 0.5 * (sum_plus_sum - l1_dist)
    den = 0.5 * (sum_plus_sum + l1_dist)
    
    mask = den > 1e-6
    K_true = torch.zeros_like(num)
    K_true[mask] = num[mask] / den[mask]
    
    # 5. RMSE
    diff = K_approx - K_true
    rmse = torch.sqrt(torch.mean(diff**2)).item()
    k_scale = torch.sqrt(torch.mean(K_true**2)).item()

    # 6. Effective rank of Z
    svals = torch.linalg.svd(z_s, full_matrices=False)[1]
    svals_sq = svals**2
    eff_rank_Z = (svals_sq.sum()**2) / torch.sum(svals_sq**2)

    return {
        "rmse": rmse,
        "rmse_norm": rmse / (k_scale + 1e-8),
        "effective_rank": eff_rank_Z.item()
    }

def calculate_utility(row):
    # Simple utility favoring higher rank 
    return row['rank']


def visualize_nystrom_component(Z_k, cmap='viridis', fade_continuous=True, iqr_factor=0.5,
                                title=None, show_colorbar=True):
    """
    Visualize a single Nyström eigenfeature.
    """
    
    valid = ~np.isnan(Z_k)
    vals = Z_k[valid]
    
    # --- 1. Signed percentiles for color ---
    order = np.argsort(vals)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.linspace(0, 1, len(vals))
    
    pct_map = np.full_like(Z_k, np.nan, dtype=float)
    pct_map[valid] = ranks
    
    # --- 2. Adaptive support ---
    abs_vals = np.abs(vals)
    med = np.median(abs_vals)
    iqr = np.percentile(abs_vals, 75) - np.percentile(abs_vals, 25)
    threshold = med + iqr_factor * iqr
    support_mask = np.full_like(Z_k, False, dtype=bool)
    support_mask[valid] = abs_vals >= threshold
    
    # --- 3. Compute fading (alpha channel) ---
    alpha = np.ones_like(Z_k, dtype=float)
    if fade_continuous:
        sorted_idx = np.argsort(abs_vals)
        abs_ranks = np.empty_like(sorted_idx, dtype=float)
        abs_ranks[sorted_idx] = np.linspace(0, 1, len(abs_vals))
        alpha[valid] = abs_ranks
    else:
        alpha[valid] = support_mask[valid].astype(float)
    
    # --- 4. Plot ---
    plt.figure(figsize=(5, 4))
    plt.imshow(pct_map, cmap=cmap, alpha=alpha)
    if show_colorbar:
        plt.colorbar(label="Signed percentile along component")
    if title is not None:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    
    return pct_map, support_mask


# ============================================================
# Main Execution
# ============================================================
if __name__ == "__main__":
    # --- 1. Load Data ---
    DATA_DIR = "/home/breallis/datasets/ebird_weekly_2023_albers"
    ebird_stack, meta = load_tifs_structured(DATA_DIR, "*_abundance_median_2023-*.tif")
    
    H, W, D = ebird_stack.shape
    
    # Create mask (Land vs Ocean)
    valid_mask = np.any(~np.isnan(ebird_stack), axis=-1)
    valid_flat = valid_mask.flatten()
    
    # Flatten and remove NaNs (Zero-fill for safe smoothing)
    ebird_flat_raw = np.nan_to_num(ebird_stack).reshape(-1, D)[valid_flat]
    
    OUT_DIR = "/home/breallis/dev/range_limits_pymc/misc_outputs/ruzicka_sweep_global"
    os.makedirs(OUT_DIR, exist_ok=True)
    
    print(f"Processing {len(ebird_flat_raw)} valid pixels. Shape: {ebird_flat_raw.shape}")

    # --- 2. Parameter Sweep ---
    # Ruzicka is non-parametric. We sweep Sigma (Smoothing) and Dim.
    sigmas = [0.0, 0.5, 1.0, 1.5]
    latent_dims = [8, 16, 32] 
    n_landmarks = 30000

    total_runs = len(sigmas) * len(latent_dims)
    run_count = 0
    results = []

    print(f"\n{'Run':<4} | {'Dim':<4} | {'Sig':<4} | {'Eff Rank':<10} | {'RMSE (N)':<10} | {'RMSE (U)':<10}")    
    print("-" * 55)

    for sig in sigmas:
        # A. Apply smoothing (or skip if sig=0)
        X_smooth = smooth_abundances(ebird_flat_raw, meta['n_weeks'], sigma=sig)
        
        # B. Compute Max Dimension once per sigma
        max_dim = max(latent_dims)
        
        try:
            # Compute Global Ruzicka features
            Z_max = compute_optimal_latent_z_ruzicka(
                X_smooth, 
                meta['n_species'], 
                meta['n_weeks'], 
                max_dim, 
                n_landmarks=n_landmarks
            )
            
            # C. Slice for specific dimensions
            for dim in sorted(latent_dims):
                run_count += 1
                
                Z_slice = Z_max[:, :dim]
                
                # Diagnostics (Using Global Kernel logic)
                diag = compute_kernel_diagnostics_ruzicka(
                    Z_slice, 
                    X_smooth, # Pass the smoothed raw data
                    meta['n_species'], 
                    meta['n_weeks']
                )

                print(f"{run_count}/{total_runs:<3} | {dim:<4} | {sig:<4.1f} | {diag['effective_rank']:<10.2f} | {diag['rmse_norm']:<10.4f} | {diag['rmse']:<10.4f}")
                results.append({
                    "dim": dim, "sigma": sig, 
                    "rank": diag['effective_rank'],
                    "rmse_norm": diag['rmse_norm'],
                    "rmse_unnorm": diag['rmse']
                })

                # D. Save Maps (Only for max dim to save space)
                if dim == max_dim:
                    # Save latent representation and mask
                    sigma_dir = os.path.join(OUT_DIR, f"sigma_{sig}")
                    os.makedirs(sigma_dir, exist_ok=True)

                    np.save(os.path.join(sigma_dir, "Z.npy"), Z_slice)
                    np.save(os.path.join(sigma_dir, "valid_mask.npy"), valid_mask)

                    # Save minimal metadata
                    meta_out = {
                        "sigma": sig,
                        "latent_dim": dim,
                        "n_species": meta['n_species'],
                        "n_weeks": meta['n_weeks'],
                        "kernel": "global_ruzicka",
                    }
                    with open(os.path.join(sigma_dir, "meta.json"), "w") as f:
                        json.dump(meta_out, f, indent=2)

                    Z_full = np.full((H*W, dim), np.nan, dtype=np.float32)
                    Z_full[valid_flat] = Z_slice
                    
                    base_name = f"d{dim}_s{sig}_ruzicka"
                    
                    # Replacement plotting block using the helper function
                    for k in range(min(dim, 10)):
                        latent_map = Z_full[:, k].reshape(H, W)
                        base_title = f"Z{k+1} (Sig={sig})\nR:{diag['effective_rank']:.1f}"
                        out_path = os.path.join(OUT_DIR, f"map_{base_name}_c{k+1}.png")

                        # Call the helper function
                        pct_map, support_mask = visualize_nystrom_component(
                            latent_map,
                            cmap="viridis",
                            fade_continuous=True,
                            iqr_factor=0.5,
                            title=base_title,
                            show_colorbar=True
                        )
                        
                        # Save the figure
                        plt.savefig(out_path, dpi=100)
                        plt.close()

                        
        except Exception as e:
            print(f"Error for sigma {sig}: {e}")
            import traceback
            traceback.print_exc()

    # --- 3. Summary ---
    df = pd.DataFrame(results)    
    df.to_csv(os.path.join(OUT_DIR, "sweep_summary.csv"), index=False)

    if not df.empty:
        plt.figure(figsize=(10, 6))
        for sig in sigmas:
            subset = df[df['sigma'] == sig]
            plt.plot(subset['dim'], subset['rank'], marker='o', label=f"Sigma={sig}")
            
        plt.xlabel('Latent Dimension')
        plt.ylabel('Effective Rank')
        plt.title('Global Ruzicka Kernel: Rank vs Dimension')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(OUT_DIR, "summary_plot.png"))
        plt.close()
        print("\nSummary plot saved.")