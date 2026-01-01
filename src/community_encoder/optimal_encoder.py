import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import gaussian_filter1d
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ============================================================
# Spatial smoothing utilities
# ============================================================

def nan_aware_gaussian(X, sigma):
    mask = np.isfinite(X).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0)

    num = gaussian_filter1d(X * mask, sigma, axis=0, mode="constant")
    num = gaussian_filter1d(num, sigma, axis=1, mode="constant")
    den = gaussian_filter1d(mask, sigma, axis=0, mode="constant")
    den = gaussian_filter1d(den, sigma=1, mode="constant")

    out = num / (den + 1e-6)
    out[den < 1e-6] = np.nan
    return out

# ============================================================
# Dataset loader
# ============================================================

def load_tifs(folder, pattern):
    paths = sorted(glob.glob(os.path.join(folder, pattern)))
    arrays = []
    for p in paths:
        import rasterio
        with rasterio.open(p) as src:
            data = src.read()  # (bands, H, W)
            data = np.moveaxis(data, 0, -1)  # (H, W, bands)
            arrays.append(data.astype(np.float32))
    if not arrays:
        raise ValueError(f"No files found: {folder}/{pattern}")
    return np.concatenate(arrays, axis=-1)


def mask_ebird(ebird_array):
    valid = np.any(~np.isnan(ebird_array), axis=-1)
    return valid, np.nan_to_num(ebird_array, nan=0.0)


# ============================================================
# Modified Nyström Function for Rational Quadratic Kernel
# ============================================================

def compute_optimal_latent_z_rq(
    ebird_flat: np.ndarray,
    latent_dim: int = 6,
    n_landmarks: int = 5000,
    block_size: int = 250_000,
    lengthscale: float = 1.0,
    alpha: float = 0.1, 
    seed: int = 42,
):
    """
    Nyström approximation using the Rational Quadratic (RQ) Kernel.
    
    k(x, y) = (1 + ||x-y||^2 / (2 * alpha * lengthscale^2))^(-alpha)
    """
    np.random.seed(seed)
    N, D = ebird_flat.shape

    # --- Landmark selection ---
    if N <= n_landmarks:
        idx_lm = np.arange(N)
    else:
        # Check if we accidentally pass the whole array instead of valid pixels
        if N > 1_000_000 and n_landmarks > 20_000: 
             print("Warning: Large N and large landmarks. Ensure memory is sufficient.")
        idx_lm = np.random.choice(N, n_landmarks, replace=False)

    X_lm = ebird_flat[idx_lm]  # (M x D)
    M = X_lm.shape[0]

    # --- Landmark RQ kernel ---
    # Squared Euclidean distances
    sq_dists_mm = (
        np.sum(X_lm**2, axis=1)[:, None]
        + np.sum(X_lm**2, axis=1)[None, :]
        - 2 * (X_lm @ X_lm.T)
    )
    # Ensure non-negative distances due to float errors
    sq_dists_mm = np.maximum(sq_dists_mm, 0.0) 

    # RQ Formula: (1 + r^2 / (2*alpha*l^2))^(-alpha)
    base_mm = 1.0 + sq_dists_mm / (2.0 * alpha * lengthscale**2)
    K_mm = np.power(base_mm, -alpha)

    # --- Eigendecomposition (PSD-safe) ---
    eigvals, eigvecs = np.linalg.eigh(K_mm)
    idx = np.argsort(eigvals)[::-1]
    
    # Filter eigenvalues < 1e-10 to avoid exploding S_inv
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    # Selecting top k
    eigvals_k = eigvals[:latent_dim]
    eigvecs_k = eigvecs[:, :latent_dim]
    
    # Numerical guard: if top eigenvalues are tiny, RQ might be too flat
    if eigvals_k[-1] < 1e-12:
        print(f"Warning: Eigenvalue collapse. Min top-{latent_dim} eigval: {eigvals_k[-1]}")

    U_k = eigvecs_k
    S_inv_sqrt = np.diag(1.0 / np.sqrt(np.clip(eigvals_k, 1e-12, None)))

    # --- Allocate output ---
    Z_opt = np.zeros((N, latent_dim), dtype=np.float32)

    # --- Blocked projection ---
    for start in range(0, N, block_size):
        end = min(start + block_size, N)
        X_blk = ebird_flat[start:end]

        # Squared distances between block and landmarks
        sq_dists_blk = (
            np.sum(X_blk**2, axis=1)[:, None]
            + np.sum(X_lm**2, axis=1)[None, :]
            - 2 * (X_blk @ X_lm.T)
        )
        sq_dists_blk = np.maximum(sq_dists_blk, 0.0)

        # RQ Kernel calculation
        base_blk = 1.0 + sq_dists_blk / (2.0 * alpha * lengthscale**2)
        K_blk = np.power(base_blk, -alpha)

        # Project
        Z_blk = K_blk @ U_k @ S_inv_sqrt
        Z_opt[start:end] = Z_blk.astype(np.float32)

        del X_blk, sq_dists_blk, base_blk, K_blk, Z_blk

    return Z_opt

def compute_optimal_latent_z_rbf(
    ebird_flat: np.ndarray,
    latent_dim: int = 6,
    n_landmarks: int = 5000,
    block_size: int = 250_000,
    lengthscale: float = 1.0,
    seed: int = 42,
):
    """
    Memory-efficient Nyström approximation of RBF kernel latent embedding.

    Args:
        ebird_flat: (N x S) Hellinger-transformed pseudo-species matrix
        latent_dim: desired latent dimensionality
        n_landmarks: number of Nyström landmarks
        block_size: number of pixels per block
        lengthscale: RBF lengthscale (ell)
        seed: RNG seed

    Returns:
        Z_opt: (N x latent_dim) optimal latent embedding approximating RBF similarity
    """
    np.random.seed(seed)
    N, D = ebird_flat.shape

    # --- Landmark selection ---
    if N <= n_landmarks:
        idx_lm = np.arange(N)
    else:
        idx_lm = np.random.choice(N, n_landmarks, replace=False)

    X_lm = ebird_flat[idx_lm]  # (M x D)
    M = X_lm.shape[0]

    # --- Landmark RBF kernel ---
    # Efficient pairwise squared distances
    sq_dists = (
        np.sum(X_lm**2, axis=1)[:, None]
        + np.sum(X_lm**2, axis=1)[None, :]
        - 2 * (X_lm @ X_lm.T)
    )
    K_mm = np.exp(-sq_dists / (2 * lengthscale**2))

    # --- Eigendecomposition (PSD-safe) ---
    eigvals, eigvecs = np.linalg.eigh(K_mm)
    idx = np.argsort(eigvals)[::-1]
    eigvals = np.clip(eigvals[idx], 1e-10, None)  # numerical stability
    eigvecs = eigvecs[:, idx]

    U_k = eigvecs[:, :latent_dim]                      # (M x k)
    S_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals[:latent_dim]))  # (k x k)

    # --- Allocate output ---
    Z_opt = np.zeros((N, latent_dim), dtype=np.float32)

    # --- Blocked projection ---
    for start in range(0, N, block_size):
        end = min(start + block_size, N)
        X_blk = ebird_flat[start:end]  # (B x D)

        # Compute RBF between block and landmarks
        sq_dists_blk = (
            np.sum(X_blk**2, axis=1)[:, None]
            + np.sum(X_lm**2, axis=1)[None, :]
            - 2 * (X_blk @ X_lm.T)
        )
        K_blk = np.exp(-sq_dists_blk / (2 * lengthscale**2))  # (B x M)

        # Project
        Z_blk = K_blk @ U_k @ S_inv_sqrt
        Z_opt[start:end] = Z_blk.astype(np.float32)

        # Cleanup
        del X_blk, sq_dists_blk, K_blk, Z_blk

    return Z_opt

def compute_optimal_latent_z(
    ebird_flat: np.ndarray,
    latent_dim: int = 6,
    n_landmarks: int = 5000,
    block_size: int = 250_000,
    seed: int = 42,
):
    """
    Memory-safe SVD–Nyström optimal latent embedding.

    Computes Z = K_nm U Λ^{-1/2} in blocks, never materializing K_nm.

    Args:
        ebird_flat: (N x S) Hellinger-transformed pseudo-species matrix
        latent_dim: desired latent dimensionality
        n_landmarks: Nyström landmarks
        block_size: number of pixels per projection block
        seed: RNG seed

    Returns:
        Z_opt: (N x latent_dim) optimal latent embedding
    """
    np.random.seed(seed)
    N, D = ebird_flat.shape

    # --- Landmark selection ---
    if N <= n_landmarks:
        idx_lm = np.arange(N)
    else:
        idx_lm = np.random.choice(N, n_landmarks, replace=False)

    X_lm = ebird_flat[idx_lm]  # (M x D)
    M = X_lm.shape[0]

    # --- Landmark kernel ---
    K_mm = X_lm @ X_lm.T  # (M x M)

    # --- Eigendecomposition (PSD-safe) ---
    eigvals, eigvecs = np.linalg.eigh(K_mm)

    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Clip for numerical stability
    eigvals = np.clip(eigvals, 1e-10, None)

    # Truncate to latent_dim
    U_k = eigvecs[:, :latent_dim]                      # (M x k)
    S_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals[:latent_dim]))  # (k x k)

    # --- Allocate output ---
    Z_opt = np.zeros((N, latent_dim), dtype=np.float32)

    # --- Blocked projection ---
    for start in range(0, N, block_size):
        end = min(start + block_size, N)
        X_blk = ebird_flat[start:end]          # (B x D)

        # K_nm block = X_blk @ X_lm.T
        K_blk = X_blk @ X_lm.T                  # (B x M)

        # Project
        Z_blk = K_blk @ U_k @ S_inv_sqrt         # (B x k)

        Z_opt[start:end] = Z_blk.astype(np.float32)

        # Explicit cleanup (WSL-critical)
        del X_blk, K_blk, Z_blk

    return Z_opt

def hellinger_transform(ebird_flat):
    """
    Hellinger transform of abundance matrix.
    
    Args:
        ebird_flat: (N x S) abundance matrix, one row per pixel, one column per species/week

    Returns:
        Hellinger-transformed array of same shape
    """
    row_sums = ebird_flat.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1.0
    return np.sqrt(ebird_flat / row_sums)


def estimate_rbf_lengthscale(ebird_flat: np.ndarray, n_samples: int = 5000, quantile: float = 0.1, seed: int = 42) -> float:
    """
    Estimate a sensible RBF lengthscale for Nyström kernel projection.

    Args:
        ebird_flat: (N x S) Hellinger-transformed pseudo-species matrix
        n_samples: number of pixels to sample for distance estimation
        quantile: which quantile of pairwise distances to use (0.5 = median)
        seed: RNG seed

    Returns:
        float: recommended RBF lengthscale
    """
    np.random.seed(seed)
    N = ebird_flat.shape[0]

    # Sample pixels
    if N <= n_samples:
        X_sample = ebird_flat
    else:
        idx = np.random.choice(N, n_samples, replace=False)
        X_sample = ebird_flat[idx]

    # Compute squared pairwise distances
    sq_dists = (
        np.sum(X_sample**2, axis=1)[:, None]
        + np.sum(X_sample**2, axis=1)[None, :]
        - 2 * (X_sample @ X_sample.T)
    )

    # Take only upper triangle to avoid duplicates and diagonal
    triu_idx = np.triu_indices_from(sq_dists, k=1)
    sq_dists_vec = sq_dists[triu_idx]

    # Lengthscale as sqrt of chosen quantile distance
    lengthscale = np.sqrt(np.quantile(sq_dists_vec, quantile))

    return float(lengthscale)


# ============================================================
# Kernel diagnostics
# ============================================================

def compute_kernel_diagnostics(z: np.ndarray, eBird: np.ndarray, max_pairs: int = 512):
    B = z.shape[0]

    if B > max_pairs:
        idx = np.random.choice(B, max_pairs, replace=False)
        z_s = torch.tensor(z[idx], dtype=torch.float32)
        e_s = torch.tensor(eBird[idx], dtype=torch.float32)
    else:
        z_s = torch.tensor(z, dtype=torch.float32)
        e_s = torch.tensor(eBird, dtype=torch.float32)

    # Dot-product kernels
    Kz = z_s @ z_s.T
    Ke = e_s @ e_s.T

    kernel_mse = ((Kz - Ke) ** 2).mean().item()
    kernel_rmse = np.sqrt(kernel_mse)
    ke_scale = np.sqrt((Ke ** 2).mean().item())
    kernel_rmse_norm = kernel_rmse / (ke_scale + 1e-8)

    z_normed = F.normalize(z_s, dim=1)
    e_normed = F.normalize(e_s, dim=1)
    cos_sim_mean = ((z_normed @ z_normed.T) * (e_normed @ e_normed.T)).mean().item()

    z_norm_mean = z_s.norm(dim=1).mean().item()
    z_norm_std  = z_s.norm(dim=1).std().item()

    # --- Effective rank ---
    # Compute singular values squared
    svals_sq = np.linalg.svd(z, compute_uv=False)**2
    effective_rank = (svals_sq.sum()**2) / np.sum(svals_sq**2)

    return {
        "kernel_rmse": kernel_rmse,
        "kernel_rmse_norm": kernel_rmse_norm,
        "cos_sim_mean": cos_sim_mean,
        "z_norm_mean": z_norm_mean,
        "z_norm_std": z_norm_std,
        "effective_rank": effective_rank
    }

# ============================================================
# Main evaluation loop
# ============================================================

if __name__ == "__main__":
    # --- Load data ---
    ebird = load_tifs("/home/breallis/datasets/ebird_weekly_2023_albers", 
                      "whcspa_abundance_median_2023-*.tif")
    valid_mask, ebird = mask_ebird(ebird)  # valid_mask: H x W

    H, W, S_weeks = ebird.shape
    ebird_flat = ebird.reshape(-1, S_weeks)          # (H*W x S_weeks)
    valid_flat = valid_mask.flatten()                # (H*W,)
    ebird_flat_valid = ebird_flat[valid_flat]       # only valid pixels

    # Hellinger transform
    ebird_flat_valid = hellinger_transform(ebird_flat_valid)

    # Assume ebird_flat is Hellinger-transformed and masked
    ell_recommended = estimate_rbf_lengthscale(ebird_flat_valid, quantile=0.05, n_samples=5000)
    print("Recommended RBF lengthscale:", ell_recommended)

    # --- Output folder ---
    out_dir = "/home/breallis/dev/range_limits_pymc/misc_outputs/optimal_latent"
    os.makedirs(out_dir, exist_ok=True)

    # ============================================================
    # Latent dimension & Hyperparameter Sweep (RQ Kernel)
    # ============================================================
    
    # 1. Parameter Grid
    # Added 16 to test if rank can go higher than 8 given sufficient flexibility
    latent_dims = [4, 8, 16] 
    
    # Hellinger space is small (max dist 2.0). 
    # Smaller ells focus on local jaggedness; larger ells smooth it out.
    ells = [0.1, 0.2, 0.4, 0.8]
    
    # Alpha controls the tail weight. 
    # 0.1 = Very heavy tails (continental signal); 1.6 = Moderate tails.
    alphas = [0.1, 0.2, 0.4, 0.8, 1.6]
    
    # Store all results for final summary plotting
    # Structure: summary_data[(dim, ell, alpha)] = diag_dict
    summary_data = {}

    print(f"\n{'Dim':<4} | {'Ell':<6} | {'Alpha':<6} | {'Eff Rank':<10} | {'RMSE Norm':<10} | {'Cos Sim':<10}")
    print("-" * 60)

    for dim in latent_dims:
        for ell in ells:
            for alpha in alphas:
                
                # --- A. Compute Latent Embedding (Valid Pixels) ---
                try:
                    Z_opt_valid = compute_optimal_latent_z_rq(
                        ebird_flat_valid, 
                        latent_dim=dim, 
                        lengthscale=ell, 
                        alpha=alpha, 
                        n_landmarks=5000 # Keep modest for speed in sweep
                    )
                except Exception as e:
                    print(f"{dim:<4} | {ell:<6.1f} | {alpha:<6.1f} | FAILED: {e}")
                    continue

                # --- B. Compute Diagnostics ---
                diag = compute_kernel_diagnostics(Z_opt_valid, ebird_flat_valid)
                
                # Print status line
                print(f"{dim:<4} | {ell:<6.1f} | {alpha:<6.1f} | {diag['effective_rank']:<10.2f} | {diag['kernel_rmse_norm']:<10.4f} | {diag['cos_sim_mean']:<10.4f}")
                
                # Store for summary
                key = (dim, ell, alpha)
                summary_data[key] = diag

                # --- C. Save Full Raster (Mapping back to HxW) ---
                Z_opt_full = np.full((H*W, dim), np.nan, dtype=np.float32)
                Z_opt_full[valid_flat] = Z_opt_valid
                
                # Construct unique filename
                base_name = f"dim{dim}_ell{ell:.1f}_alpha{alpha:.1f}"
                
                # Save the latent array
                np.save(os.path.join(out_dir, f"Z_{base_name}.npy"), Z_opt_full)
                
                # Save diagnostics
                np.savez(os.path.join(out_dir, f"diag_{base_name}.npz"), **diag)

                # --- D. Save Visualization Maps (First 3 components only) ---
                # Limiting to 3 components prevents generating excessive images
                plot_range = range(min(dim, 3)) 
                
                for k in plot_range:
                    latent_map = Z_opt_full[:, k].reshape(H, W)
                    
                    plt.figure(figsize=(8, 6))
                    plt.imshow(latent_map, cmap="viridis")
                    plt.colorbar(label=f"Latent Comp {k+1}")
                    plt.title(f"Latent {k+1} (D={dim}, l={ell}, a={alpha})\nRank:{diag['effective_rank']:.2f}, RMSE:{diag['kernel_rmse_norm']:.2f}")
                    plt.axis("off")
                    plt.tight_layout()
                    plt.savefig(os.path.join(out_dir, f"map_{base_name}_comp{k+1}.png"))
                    plt.close()

    # ============================================================
    # Summary Visualization
    # ============================================================
    
    # Save aggregate results dictionary
    np.save(os.path.join(out_dir, "summary_data.npy"), summary_data)
    
    # Create a scatter plot of Effective Rank vs RMSE for all runs
    # Color coded by Alpha, Size by Lengthscale
    
    ranks = []
    rmses = []
    alpha_vals = []
    ell_vals = []
    dims = []
    
    for (d, l, a), res in summary_data.items():
        ranks.append(res['effective_rank'])
        rmses.append(res['kernel_rmse_norm'])
        alpha_vals.append(a)
        ell_vals.append(l)
        dims.append(d)
        
    plt.figure(figsize=(10, 8))
    
    # Scatter plot
    # s argument scales circle size by lengthscale for visual grouping
    sc = plt.scatter(ranks, rmses, c=alpha_vals, s=np.array(ell_vals)*150 + 30, 
                     cmap='viridis', alpha=0.8, edgecolors='k')
    
    cbar = plt.colorbar(sc)
    cbar.set_label('Alpha (Tail Weight: Dark=Heavy, Yellow=Light)')
    
    plt.xlabel('Effective Rank (Higher is better)')
    plt.ylabel('Kernel RMSE Norm (Lower is better)')
    plt.title('Hyperparameter Sweep: Rank vs RMSE\n(Point Size = Lengthscale)')
    plt.grid(True, alpha=0.3)
    
    # Annotate points that are "interesting" (High Rank + Decent RMSE)
    # This helps identify the sweet spot quickly
    for i in range(len(ranks)):
        if ranks[i] > 4.0 and rmses[i] < 0.85:
            plt.text(ranks[i], rmses[i], f" d{dims[i]}_l{ell_vals[i]}_a{alpha_vals[i]}", fontsize=8)

    plt.savefig(os.path.join(out_dir, "parameter_sweep_summary.png"))
    plt.close()

    print(f"\nSweep complete. Results and summary plots saved to {out_dir}")