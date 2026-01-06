import os
import glob
import re
import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from scipy.ndimage import gaussian_filter1d
import rasterio

# ============================================================
# 1. Robust Data Loader
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
            records.append({"species": match.group(1), "date": match.group(2), "path": fpath})
    df = pd.DataFrame(records)
    df_sorted = df.sort_values(by=['species','date'])
    ordered_paths = df_sorted['path'].tolist()
    species_list = sorted(df['species'].unique().tolist())
    n_species = len(species_list)
    n_weeks = df['date'].nunique()
    with rasterio.open(ordered_paths[0]) as src:
        H, W = src.shape
    full_stack = np.zeros((H, W, len(ordered_paths)), dtype=np.float32)
    for i, p in enumerate(ordered_paths):
        with rasterio.open(p) as src:
            full_stack[:, :, i] = src.read(1)
    return full_stack, {"n_species": n_species, "n_weeks": n_weeks}

# ============================================================
# 2. Mask Loader
# ============================================================
def load_mask(folder, pattern="*_abundance_median_*.tif"):
    files = sorted(glob.glob(os.path.join(folder, pattern)))
    if not files:
        raise ValueError(f"No files found in {folder} matching {pattern}")
    with rasterio.open(files[0]) as src:
        H, W = src.shape
    mask = np.zeros((H, W), dtype=bool)
    for f in files:
        with rasterio.open(f) as src:
            mask |= ~np.isnan(src.read(1))
    return mask

# ============================================================
# 3. Temporal smoothing
# ============================================================
def smooth_ebird(ebird_flat, n_weeks, sigma=0.0):
    N, D = ebird_flat.shape
    n_species = D // n_weeks
    data_3d = ebird_flat.reshape(N, n_species, n_weeks)
    if sigma > 1e-5:
        data_3d = gaussian_filter1d(data_3d, sigma=sigma, axis=-1, mode='wrap')
    return data_3d.reshape(N, -1)

# ============================================================
# 4. Batch Ruzicka Kernel
# ============================================================
def batch_ruzicka_kernel(X_batch, X_lm=None, eps=1e-6):
    if X_lm is None:
        X_lm = X_batch
    B, S, T = X_batch.shape
    M = X_lm.shape[0]
    K_batch = torch.zeros((B, M), device=X_batch.device, dtype=torch.float32)
    for s in range(S):
        b_s = X_batch[:, s, :]
        lm_s = X_lm[:, s, :]
        l1 = torch.cdist(b_s, lm_s, p=1)
        sum_b = b_s.sum(dim=1, keepdim=True)
        sum_lm = lm_s.sum(dim=1, keepdim=True)
        numerator = 0.5 * (sum_b + sum_lm.T - l1)
        denominator = 0.5 * (sum_b + sum_lm.T + l1)
        mask = denominator > eps
        k_s = torch.zeros_like(numerator)
        k_s[mask] = numerator[mask] / denominator[mask]
        K_batch += k_s
    return K_batch

# ============================================================
# 5. Nyström Initialization
# ============================================================
def nystrom_init(ebird_flat, n_species, n_weeks, latent_dim=32, n_landmarks=10000, device='cuda'):
    N, D = ebird_flat.shape
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    X_all = torch.tensor(ebird_flat.reshape(N, n_species, n_weeks), dtype=torch.float32, device=device)
    idx_lm = np.random.choice(N, min(N, n_landmarks), replace=False)
    X_lm = X_all[idx_lm]
    M = X_lm.shape[0]

    # Landmark kernel
    K_mm = batch_ruzicka_kernel(X_lm)

    # Eigendecomposition
    L, U = torch.linalg.eigh(K_mm.cpu())
    idx_sort = torch.argsort(L, descending=True)[:latent_dim]
    L = torch.clamp(L[idx_sort], min=1e-10).to(device)
    U = U[:, idx_sort].to(device)
    proj = U * torch.rsqrt(L)

    # Batch projection
    batch_size = 5000
    Z_all = np.zeros((N, latent_dim), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            X_batch = X_all[start:end]
            K_NM = batch_ruzicka_kernel(X_batch, X_lm)
            Z_batch = K_NM @ proj
            Z_batch /= np.sqrt(n_species)
            Z_all[start:end] = Z_batch.cpu().numpy()
    return Z_all

def stochastic_ruzicka_fit_batch(
    ebird_flat,
    Z_init,
    n_species,
    n_weeks,
    batch_size=20000,
    n_iters=200,
    lr=1e-3,
    ortho_weight=1.0,
    rmse_sample=1024,
    ortho_sample=8192,
    device='cuda'
):
    """
    Stochastic Ruzicka fit with sample-based RMSE and soft orthogonality loss.
    Prints full diagnostics: MSE, Ortho, Total Loss, RMSE (U/N), Effective Rank.
    """
    N, D = ebird_flat.shape
    latent_dim = Z_init.shape[1]

    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    X_all = torch.tensor(ebird_flat.reshape(N, n_species, n_weeks),
                         dtype=torch.float32, device=device)
    Z = torch.tensor(Z_init, dtype=torch.float32, device=device, requires_grad=True)

    optimizer = torch.optim.Adam([Z], lr=lr)

    for it in range(n_iters):
        optimizer.zero_grad()

        # --- 1. Batch MSE loss ---
        idx = np.random.choice(N, min(batch_size, N), replace=False)
        X_batch = X_all[idx]  # (B, S, T)
        Z_batch = Z[idx]      # (B, d)

        K_batch = batch_ruzicka_kernel(X_batch)
        K_pred = Z_batch @ Z_batch.T
        loss_mse = torch.mean((K_pred - K_batch) ** 2)

        # --- 2. Soft orthogonality ---
        if N > ortho_sample:
            idx_o = np.random.choice(N, ortho_sample, replace=False)
            Z_ortho = Z[idx_o]
        else:
            Z_ortho = Z

        ZtZ = Z_ortho.T @ Z_ortho / Z_ortho.shape[0]
        I = torch.eye(latent_dim, device=device)
        loss_ortho = torch.mean((ZtZ - I)**2)

        # --- 3. Total loss ---
        total_loss = loss_mse + ortho_weight * loss_ortho
        total_loss.backward()
        optimizer.step()

        # --- 4. Sample-based RMSE & effective rank ---
        with torch.no_grad():
            idx_rmse = np.random.choice(N, min(rmse_sample, N), replace=False)
            Z_s = Z[idx_rmse]
            X_s = X_all[idx_rmse]

            K_approx = Z_s @ Z_s.T

            K_true = torch.zeros_like(K_approx)
            for s in range(n_species):
                b_s = X_s[:, s, :]
                sum_s = b_s.sum(dim=1, keepdim=True)
                l1 = torch.cdist(b_s, b_s, p=1)
                num = 0.5 * (sum_s + sum_s.T - l1)
                den = 0.5 * (sum_s + sum_s.T + l1)
                mask = den > 1e-6
                k_s = torch.zeros_like(num)
                k_s[mask] = num[mask] / den[mask]
                K_true += k_s / n_species

            rmse_unnorm = torch.sqrt(torch.mean((K_approx - K_true)**2)).item()
            rmse_norm = rmse_unnorm / (torch.sqrt(torch.mean(K_true**2)).item() + 1e-12)

            # Effective rank: corrected formula
            svals = torch.linalg.svd(Z_s, full_matrices=False)[1]
            svals_sq = svals**2
            eff_rank = (svals_sq.sum()**2 / (svals_sq**2).sum()).item()
            eff_rank = min(eff_rank, latent_dim)  # cannot exceed latent_dim

            if it % 10 == 0 or it == n_iters - 1:
                print(f"[{it}/{n_iters}] "
                      f"Loss MSE: {loss_mse.item():.6e} | "
                      f"Loss Ortho: {loss_ortho.item():.6e} | "
                      f"Total Loss: {total_loss.item():.6e} | "
                      f"RMSE (U): {rmse_unnorm:.6f} | "
                      f"RMSE (N): {rmse_norm:.6f} | "
                      f"Eff Rank: {eff_rank:.2f}")

    return Z.detach().cpu().numpy()


# ============================================================
# 7. Main Execution
# ============================================================
if __name__ == "__main__":
    DATA_DIR = "/home/breallis/datasets/ebird_weekly_2023_albers"
    OUT_DIR = "/home/breallis/dev/range_limits_pymc/misc_outputs/ruzicka_stochastic"
    os.makedirs(OUT_DIR, exist_ok=True)

    stack, meta = load_tifs_structured(DATA_DIR, "*_abundance_median_2023-*.tif")
    valid_mask = load_mask(DATA_DIR, "*_abundance_median_2023-*.tif")
    ebird_flat = stack[valid_mask]
    ebird_smooth = smooth_ebird(ebird_flat, meta['n_weeks'], sigma=0.5)

    latent_dim = 32
    Z_init = nystrom_init(ebird_smooth, meta['n_species'], meta['n_weeks'], latent_dim=latent_dim, n_landmarks=1000, device='cuda')

    Z_stoch = stochastic_ruzicka_fit_batch(
        ebird_flat=ebird_smooth,
        Z_init=Z_init,
        n_species=meta['n_species'],
        n_weeks=meta['n_weeks'],
        batch_size=20000,
        n_iters=200,
        lr=1e-4,
        device='cuda',
        ortho_weight=1.0
    )

    np.save(os.path.join(OUT_DIR, "Z_stochastic.npy"), Z_stoch)
    np.save(os.path.join(OUT_DIR, "valid_mask.npy"), valid_mask)
    print(f"Saved Z_stochastic.npy and valid_mask.npy to {OUT_DIR}")
