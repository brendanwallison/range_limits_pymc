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
# 2. Smoothed Hellinger Transform
# ============================================================
def smoothed_hellinger_transform(ebird_flat, n_weeks, sigma):
    """
    1. Reshapes to (N, Species, Weeks)
    2. Applies Gaussian blur along Time axis (if sigma > 0)
    3. Reflattens and applies Hellinger Transform
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
        
    # Flatten back to N x D
    data_flat = data_smoothed.reshape(N, -1)
    
    # Hellinger: sqrt( x / sum(x) )
    row_sums = data_flat.sum(axis=1, keepdims=True)
    row_sums[row_sums < 1e-9] = 1.0  # Avoid div/0
    return np.sqrt(data_flat / row_sums)

# ============================================================
# 3. RQ Kernel Nyström
# ============================================================
# def compute_optimal_latent_z_rq(ebird_flat, latent_dim, lengthscale, alpha, n_landmarks=5000, seed=42):
#     np.random.seed(seed)
#     N = ebird_flat.shape[0]
    
#     # Select Landmarks
#     idx_lm = np.random.choice(N, min(N, n_landmarks), replace=False)
#     X_lm = ebird_flat[idx_lm]

#     # Helper: pairwise squared distances
#     def get_sq_dists(A, B):
#         # (A-B)^2 = A^2 + B^2 - 2AB
#         return np.maximum(
#             np.sum(A**2, axis=1)[:,None] + np.sum(B**2, axis=1)[None,:] - 2*(A@B.T), 
#             0.0
#         )

#     # Landmark Kernel K_mm
#     sq_dists_mm = get_sq_dists(X_lm, X_lm)
#     base_mm = 1.0 + sq_dists_mm / (2.0 * alpha * lengthscale**2)
#     K_mm = np.power(base_mm, -alpha)

#     # Eigendecomposition
#     eigvals, eigvecs = np.linalg.eigh(K_mm)
#     idx = np.argsort(eigvals)[::-1]
#     eigvals = eigvals[idx][:latent_dim]
#     eigvecs = eigvecs[:, idx][:, :latent_dim]
    
#     # Nyström Projection Matrix
#     U_k = eigvecs
#     S_inv_sqrt = np.diag(1.0 / np.sqrt(np.clip(eigvals, 1e-12, None)))
    
#     # Blocked Projection of all points
#     block_size = 50000
#     Z_opt = np.zeros((N, latent_dim), dtype=np.float32)
    
#     for start in range(0, N, block_size):
#         end = min(start + block_size, N)
#         X_blk = ebird_flat[start:end]
        
#         sq_dists_blk = get_sq_dists(X_blk, X_lm)
#         base_blk = 1.0 + sq_dists_blk / (2.0 * alpha * lengthscale**2)
#         K_blk = np.power(base_blk, -alpha)
        
#         Z_opt[start:end] = (K_blk @ U_k @ S_inv_sqrt).astype(np.float32)
        
#     return Z_opt

def compute_optimal_latent_z_rq(ebird_flat, latent_dim, lengthscale, alpha, n_landmarks=20000, device='cuda'):
    """
    Scaled Nystrom for large landmark sets (M=20,000+).
    Uses Torch for accelerated distance calcs and batching.
    """
    import torch
    N, D = ebird_flat.shape
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Select Landmarks
    idx_lm = np.random.choice(N, min(N, n_landmarks), replace=False)
    X_lm = torch.tensor(ebird_flat[idx_lm], device=device, dtype=torch.float32)

    # 2. Compute K_mm (Landmark-to-Landmark)
    # Using (A-B)^2 = A^2 + B^2 - 2AB in torch
    with torch.no_grad():
        dist_sq = torch.cdist(X_lm, X_lm, p=2)**2
        K_mm = (1.0 + dist_sq / (2.0 * alpha * lengthscale**2)).pow(-alpha)
        
        # Eigen-decomposition (Small enough for CPU/GPU)
        # Using .cpu() for the solver often more stable for large M
        L, U = torch.linalg.eigh(K_mm.cpu()) 
        
        # Sort descending
        idx = torch.argsort(L, descending=True)[:latent_dim]
        L = L[idx]
        U = U[:, idx].to(device)
        
        # SVD Projection Matrix: U * S^-0.5
        proj_mat = U / torch.sqrt(torch.clamp(L.to(device), min=1e-10))

    # 3. Batch Projection (The "Memory Guard")
    # Project all N points in chunks to avoid OOM
    Z_opt = np.zeros((N, latent_dim), dtype=np.float32)
    batch_size = 10000 
    
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            X_batch = torch.tensor(ebird_flat[start:end], device=device, dtype=torch.float32)
            
            # Distance from batch to landmarks
            d_batch_lm = torch.cdist(X_batch, X_lm, p=2)**2
            K_batch_lm = (1.0 + d_batch_lm / (2.0 * alpha * lengthscale**2)).pow(-alpha)
            
            # Project: (Batch x M) @ (M x Latent_dim)
            Z_opt[start:end] = (K_batch_lm @ proj_mat).cpu().numpy()
            
    return Z_opt

def compute_kernel_diagnostics(z, eBird, max_samples=1000):
    idx = np.random.choice(z.shape[0], min(z.shape[0], max_samples), replace=False)
    z_s, e_s = torch.tensor(z[idx]), torch.tensor(eBird[idx])
    
    # RMSE
    Kz = z_s @ z_s.T
    Ke = e_s @ e_s.T
    rmse = torch.sqrt(torch.mean((Kz - Ke)**2)).item()
    ke_scale = torch.sqrt(torch.mean(Ke**2)).item()
    
    # Effective Rank
    svals_sq = np.linalg.svd(z, compute_uv=False)**2
    eff_rank = (svals_sq.sum()**2) / np.sum(svals_sq**2)
    
    return {"rmse_norm": rmse / (ke_scale + 1e-8), "effective_rank": eff_rank}

def calculate_utility(row):
    # Penalize the "White Noise" snap (RMSE near 1.0)
    if row['rmse'] > 0.95:
        return 0
    
    # Penalize the "Blob" snap (Rank near 1.0)
    if row['rank'] < 2.0:
        return 0

    # Reward Rank, but only if RMSE is holding steady.
    # This acts like an "Information Density" score.
    # We want high rank, but we weigh it against (1 - RMSE)
    return row['rank'] * (1.0 - row['rmse'])

# ============================================================
# Main Execution
# ============================================================
if __name__ == "__main__":
    # --- 1. Load Data ---
    DATA_DIR = "/home/breallis/datasets/ebird_weekly_2023_albers"
    ebird_stack, meta = load_tifs_structured(DATA_DIR, "*_abundance_median_2023-*.tif")
    
    H, W, D = ebird_stack.shape
    
    # Create mask (Land vs Ocean)
    # Assume any pixel with NaN in any band is invalid
    valid_mask = np.any(~np.isnan(ebird_stack), axis=-1)
    valid_flat = valid_mask.flatten()
    
    # Flatten and remove NaNs (Zero-fill for safe smoothing)
    ebird_flat_raw = np.nan_to_num(ebird_stack).reshape(-1, D)[valid_flat]
    
    OUT_DIR = "/home/breallis/dev/range_limits_pymc/misc_outputs/wide_sweep_v1"
    os.makedirs(OUT_DIR, exist_ok=True)
    
    print(f"Processing {len(ebird_flat_raw)} valid pixels. Shape: {ebird_flat_raw.shape}")

    # --- 2. Broad Parameter Sweep ---
    # Dims: Extended to 32 as requested
    latent_dims = [16, 32] 
    
    # Parameters focused on the 0.2-0.5 Ell "Transition Zone"
    sigmas = [0.5, 1.0, 1.5]
    ells = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    alphas = [4.0, 8.0, 16.0]
    latent_dims = [32, 64]
    n_landmarks = 10000

    total_runs = len(sigmas) * len(latent_dims) * len(ells) * len(alphas)
    run_count = 0
    results = []

    print(f"\n{'Run':<4} | {'Dim':<4} | {'Sig':<4} | {'Ell':<5} | {'Alph':<4} | {'Eff Rank':<10} | {'RMSE':<10}")
    print("-" * 65)

    # Outer loop: Smoothing is the expensive transform
    for sig in sigmas:
        X_smooth = smoothed_hellinger_transform(ebird_flat_raw, meta['n_weeks'], sigma=sig)
        
        # Sort dims to ensure logic is deterministic
        sorted_dims = sorted(latent_dims)
        max_dim = sorted_dims[-1]
        
        for ell in ells:
            for alpha in alphas:
                try:
                    # A. Compute the nested solution once for the largest dimension
                    Z_max = compute_optimal_latent_z_rq(
                        X_smooth, max_dim, ell, alpha, n_landmarks=n_landmarks
                    )
                    
                    # B. Slice and diagnose for each dimension in the sorted list
                    for dim in sorted_dims:
                        run_count += 1
                        
                        # Slice the pre-computed Z_max
                        # Because dimensions are orthogonal and ordered by eigenvalue,
                        # Z_max[:, :8] is identical to computing Z with latent_dim=8.
                        Z_slice = Z_max[:, :dim]
                        
                        # Calculate Diagnostics
                        diag = compute_kernel_diagnostics(Z_slice, X_smooth)
                        
                        # Update progress
                        print(f"{run_count}/{total_runs:<3} | {dim:<4} | {sig:<4.1f} | {ell:<4.2f} | {alpha:<4.1f} | {diag['effective_rank']:<10.2f} | {diag['rmse_norm']:<10.4f}")
                        
                        results.append({
                            "dim": dim, "sigma": sig, "ell": ell, "alpha": alpha,
                            "rank": diag['effective_rank'], "rmse": diag['rmse_norm']
                        })
                        
                        # C. Save Maps
                        # Only save maps for the max_dim to avoid redundant images
                        # (Z1, Z2, Z3 are identical regardless of whether you calculated 8 or 32 dims)
                        if dim == max_dim:
                            Z_full = np.full((H*W, dim), np.nan, dtype=np.float32)
                            Z_full[valid_flat] = Z_slice
                            
                            base_name = f"d{dim}_s{sig}_l{ell}_a{alpha}"
                            
                            for k in range(min(dim, 3)):
                                latent_map = Z_full[:, k].reshape(H, W)
                                plt.figure(figsize=(5, 4))
                                plt.imshow(latent_map, cmap="viridis")
                                plt.title(f"Z{k+1} (Sig={sig}, L={ell}, a={alpha})\nR:{diag['effective_rank']:.1f}")
                                plt.axis("off")
                                plt.tight_layout()
                                plt.savefig(os.path.join(OUT_DIR, f"map_{base_name}_c{k+1}.png"), dpi=100)
                                plt.close()
                            
                except Exception as e:
                    # If the max_dim calculation fails, increment run_count for all dims in the group
                    print(f"Error for nested config at s{sig} l{ell} a{alpha}: {e}")
                    run_count += len(sorted_dims)

    # --- 3. Summary Visualization ---
    df = pd.DataFrame(results)    
    # Calculate the Utility Score
    df['utility'] = df.apply(calculate_utility, axis=1)
    df.to_csv(os.path.join(OUT_DIR, "sweep_summary.csv"), index=False)

   
    # Identify the Pareto Front (Non-dominated solutions)
    # A point is on the front if no other point has (Higher Rank AND Lower RMSE)
    pareto_front = []
    for idx, row in df.iterrows():
        is_dominated = ((df['rank'] >= row['rank']) & (df['rmse'] <= row['rmse']) & 
                        ((df['rank'] > row['rank']) | (df['rmse'] < row['rmse'])))
        if not is_dominated.any():
            pareto_front.append(row)
            
    df_pareto = pd.DataFrame(pareto_front).sort_values("rank")
    df_pareto.to_csv(os.path.join(OUT_DIR, "sweep_pareto.csv"), index=False)

    print("\n" + "="*80)
    print(f"{'THE PARETO FRONT (Best Trade-offs)':^80}")
    print("="*80)
    print(df_pareto[['dim', 'sigma', 'ell', 'alpha', 'rank', 'rmse', 'utility']].to_string(index=False))
    
    if not df.empty:
        plt.figure(figsize=(12, 10))
        # X=Rank, Y=RMSE, Color=Sigma, Size=Ell, Marker=Alpha
        # Since we have 4 vars, we need to be creative. 
        # Let's facet or use marker styles, but for now, color=Alpha, Size=Ell, Shape by Sigma?
        # Simple approach: Plot '0.0' and '1.5' sigma as separate series.
        
        subset_0 = df[df['sigma'] == 0.0]
        subset_1 = df[df['sigma'] == 1.5]
        
        plt.scatter(
            subset_0['rank'], subset_0['rmse'], 
            c=subset_0['alpha'], cmap='viridis', 
            s=subset_0['ell']*200 + 50, alpha=0.6, edgecolors='k', marker='o', label='Sigma=0.0'
        )
        
        plt.scatter(
            subset_1['rank'], subset_1['rmse'], 
            c=subset_1['alpha'], cmap='viridis', 
            s=subset_1['ell']*200 + 50, alpha=0.6, edgecolors='r', marker='s', label='Sigma=1.5'
        )

        plt.colorbar(label='Alpha (Tail Weight)')
        plt.legend(loc='upper right')
        plt.xlabel('Effective Rank (Higher is Better)')
        plt.ylabel('RMSE Norm (Lower is Better)')
        plt.title('Broad Parameter Sweep: Rank vs RMSE\n(Size=Lengthscale, Color=Alpha, Shape=Smoothing)')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(OUT_DIR, "summary_plot.png"))
        plt.close()
        print("\nSummary plot saved.")