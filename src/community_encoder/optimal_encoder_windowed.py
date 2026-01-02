#!/usr/bin/env python3
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
from sklearn.kernel_approximation import RBFSampler

# ============================================================
# 1. Robust Data Loader (Species-Major Order)
# ============================================================
def load_tifs_structured(folder, pattern="*_abundance_median_*.tif"):
    files = sorted(glob.glob(os.path.join(folder, pattern)))
    if not files:
        raise ValueError(f"No files found in {folder} matching {pattern}")

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

    n_species = df['species'].nunique()
    n_weeks = df['date'].nunique()
    if len(df) != n_species * n_weeks:
        raise ValueError(f"Grid incomplete. Expected {n_species*n_weeks} files, found {len(df)}.")

    df_sorted = df.sort_values(by=['species', 'date'])
    ordered_paths = df_sorted['path'].tolist()

    with rasterio.open(ordered_paths[0]) as src:
        H, W = src.shape

    print(f"Loading {len(ordered_paths)} rasters ({n_species} sp x {n_weeks} wks)...")

    full_stack = np.zeros((H, W, len(ordered_paths)), dtype=np.float32)
    for i, p in enumerate(ordered_paths):
        with rasterio.open(p) as src:
            full_stack[:, :, i] = src.read(1)

    return full_stack, {"n_species": n_species, "n_weeks": n_weeks}


# ============================================================
# 2. Smoothed Hellinger Transform
# ============================================================
def smoothed_hellinger_transform(ebird_flat, n_weeks, sigma):
    N, D = ebird_flat.shape
    n_species = D // n_weeks
    data_3d = ebird_flat.reshape(N, n_species, n_weeks)
    if sigma > 1e-5:
        data_smoothed = gaussian_filter1d(data_3d, sigma=sigma, axis=-1, mode='wrap')
    else:
        data_smoothed = data_3d
    data_flat = data_smoothed.reshape(N, -1)
    row_sums = data_flat.sum(axis=1, keepdims=True)
    row_sums[row_sums < 1e-9] = 1.0
    return np.sqrt(data_flat / row_sums)


# ============================================================
# 3. RBF Kernel Nyström
# ============================================================
def compute_optimal_latent_z_rbf(ebird_flat, latent_dim, lengthscale, n_landmarks=20000, device='cuda'):
    N, D = ebird_flat.shape
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    idx_lm = np.random.choice(N, min(N, n_landmarks), replace=False)
    X_lm = torch.tensor(ebird_flat[idx_lm], device=device, dtype=torch.float32)

    with torch.no_grad():
        dist_sq = torch.cdist(X_lm, X_lm, p=2)**2
        K_mm = torch.exp(-dist_sq / (2 * lengthscale**2))
        L, U = torch.linalg.eigh(K_mm.cpu())
        idx = torch.argsort(L, descending=True)[:latent_dim]
        L = L[idx]
        U = U[:, idx].to(device)
        proj_mat = U / torch.sqrt(torch.clamp(L.to(device), min=1e-10))

    Z_opt = np.zeros((N, latent_dim), dtype=np.float32)
    batch_size = 10000
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            X_batch = torch.tensor(ebird_flat[start:end], device=device, dtype=torch.float32)
            d_batch_lm = torch.cdist(X_batch, X_lm, p=2)**2
            K_batch_lm = torch.exp(-d_batch_lm / (2 * lengthscale**2))
            Z_opt[start:end] = (K_batch_lm @ proj_mat).cpu().numpy()
    return Z_opt


# ============================================================
# 4. Diagnostics
# ============================================================
def compute_kernel_diagnostics(z, eBird, max_samples=1000):
    idx = np.random.choice(z.shape[0], min(z.shape[0], max_samples), replace=False)
    z_s, e_s = torch.tensor(z[idx]), torch.tensor(eBird[idx])
    Kz = z_s @ z_s.T
    Ke = e_s @ e_s.T
    rmse = torch.sqrt(torch.mean((Kz - Ke)**2)).item()
    ke_scale = torch.sqrt(torch.mean(Ke**2)).item()
    svals_sq = np.linalg.svd(z, compute_uv=False)**2
    eff_rank = (svals_sq.sum()**2) / np.sum(svals_sq**2)
    return {"rmse_norm": rmse / (ke_scale + 1e-8), "effective_rank": eff_rank}


# ============================================================
# 5. Main Execution
# ============================================================
if __name__ == "__main__":
    DATA_DIR = "/home/breallis/datasets/ebird_weekly_2023_albers"
    OUT_DIR = "/home/breallis/dev/range_limits_pymc/misc_outputs/rbf_vs_rff"
    os.makedirs(OUT_DIR, exist_ok=True)

    ebird_stack, meta = load_tifs_structured(DATA_DIR, "*_abundance_median_2023-*.tif")
    H, W, D = ebird_stack.shape

    valid_mask = np.any(~np.isnan(ebird_stack), axis=-1)
    valid_flat = valid_mask.flatten()
    ebird_flat_raw = np.nan_to_num(ebird_stack).reshape(-1, D)[valid_flat]

    print(f"Processing {len(ebird_flat_raw)} valid pixels. Shape: {ebird_flat_raw.shape}")

    # --- Preprocess: Smoothed Hellinger ---
    sigma = 0.5
    X_smooth = smoothed_hellinger_transform(ebird_flat_raw, meta['n_weeks'], sigma=sigma)

    latent_dim = 32
    lengthscale = 0.55

    # --- 1) RBF Nyström ---
    Z_rbf = compute_optimal_latent_z_rbf(X_smooth, latent_dim, lengthscale)
    diag_rbf = compute_kernel_diagnostics(Z_rbf, X_smooth)
    print(f"RBF Nyström | Rank: {diag_rbf['effective_rank']:.2f}, RMSE Norm: {diag_rbf['rmse_norm']:.4f}")

    # --- 2) Random Fourier Features via scikit-learn ---
    rff = RBFSampler(gamma=1/(2*lengthscale**2), n_components=latent_dim, random_state=42)
    Z_rff = rff.fit_transform(X_smooth)
    diag_rff = compute_kernel_diagnostics(Z_rff, X_smooth)
    print(f"Random Fourier | Rank: {diag_rff['effective_rank']:.2f}, RMSE Norm: {diag_rff['rmse_norm']:.4f}")

    # --- Optional: Save comparison ---
    df_compare = pd.DataFrame({
        "method": ["RBF Nyström", "Random Fourier"],
        "rank": [diag_rbf['effective_rank'], diag_rff['effective_rank']],
        "rmse_norm": [diag_rbf['rmse_norm'], diag_rff['rmse_norm']]
    })
    df_compare.to_csv(os.path.join(OUT_DIR, "rbf_vs_rff_summary.csv"), index=False)
    print(f"Summary saved to {OUT_DIR}")
