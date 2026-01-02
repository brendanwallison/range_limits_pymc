import os
import glob
import re
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from scipy.ndimage import gaussian_filter1d
from torch import nn
from torch.optim import Adam
import rasterio

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
    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("No filenames matched regex.")

    n_species = df['species'].nunique()
    n_weeks = df['date'].nunique()
    if len(df) != n_species * n_weeks:
        raise ValueError(f"Grid incomplete. Expected {n_species*n_weeks} files, found {len(df)}.")

    df_sorted = df.sort_values(by=['species', 'date'])
    ordered_paths = df_sorted['path'].tolist()

    with rasterio.open(ordered_paths[0]) as src:
        H, W = src.shape

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
# 3. RBF Nyström Initialization
# ============================================================
def rbf_nystrom_init(X, latent_dim=64, lengthscale=0.55, alpha=1.0, n_landmarks=10000, device='cuda'):
    N = X.shape[0]
    idx_lm = np.random.choice(N, min(n_landmarks, N), replace=False)
    X_lm = torch.tensor(X[idx_lm], dtype=torch.float32, device=device)

    with torch.no_grad():
        dist_sq = torch.cdist(X_lm, X_lm, p=2)**2
        K_mm = torch.exp(-dist_sq / (2.0 * lengthscale**2)) ** alpha

        L, U = torch.linalg.eigh(K_mm.cpu())
        idx_sort = torch.argsort(L, descending=True)[:latent_dim]
        L = L[idx_sort]
        U = U[:, idx_sort].to(device)

        proj_mat = U / torch.sqrt(torch.clamp(L.to(device), min=1e-10))

    Z_init = torch.zeros((N, latent_dim), dtype=torch.float32, device=device)
    batch_size = 32000
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            X_batch = X_t[start:end]
            dist_sq_blk = torch.cdist(X_batch, X_lm, p=2)**2
            K_blk = torch.exp(-dist_sq_blk / (2.0 * lengthscale**2)) ** alpha
            Z_init[start:end] = K_blk @ proj_mat
    return Z_init.cpu().numpy()

# ============================================================
# 4. Weighted Kernel Loss
# ============================================================
class WeightedKernelLoss(torch.nn.Module):
    """
    Weighted kernel MSE loss.
    Inputs:
        - z: (B, D) latent features
        - K_batch: (B, B) target kernel
        - weight_mode: 'magnitude' | 'uniform' | 'squared'
    """
    def __init__(self, weight_mode="magnitude"):
        super().__init__()
        assert weight_mode in ["magnitude", "uniform", "squared"]
        self.weight_mode = weight_mode

    def forward(self, z, K_batch):
        Kz = z @ z.T
        if self.weight_mode == "magnitude":
            W = K_batch.abs()
        elif self.weight_mode == "squared":
            W = K_batch ** 2
        else:
            W = torch.ones_like(K_batch)
        diff = Kz - K_batch
        loss = (W * diff ** 2).mean()
        return loss

# ============================================================
# 5. Normalized RMSE Diagnostic
# ============================================================
def batch_rmse(K_pred, K_target):
    mse = torch.mean((K_pred - K_target) ** 2)
    scale = torch.sqrt(torch.mean(K_target ** 2))
    return torch.sqrt(mse) / (scale + 1e-8)

# ============================================================
# 6. Stochastic RBF Solver
# ============================================================
def stochastic_rbf_fit(
    X,
    Z_init,
    lengthscale=0.55,
    alpha=1.0,
    batch_size=4096,
    n_iters=2000,
    lr=0.01,
    weight_mode="magnitude",
    device="cuda"
):
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    Z = torch.tensor(Z_init, dtype=torch.float32, device=device, requires_grad=True)
    optimizer = Adam([Z], lr=lr)
    N = X.shape[0]

    loss_fn = WeightedKernelLoss(weight_mode=weight_mode)

    for it in range(n_iters):
        optimizer.zero_grad()
        idx = np.random.choice(N, min(batch_size, N), replace=False)
        X_batch = X_t[idx]
        Z_batch = Z[idx]

        dist_sq = torch.cdist(X_batch, X_batch, p=2)**2
        Kx = torch.exp(-dist_sq / (2.0 * lengthscale**2)) ** alpha

        Z_norm = nn.functional.normalize(Z_batch, dim=1)
        loss = loss_fn(Z_norm, Kx)
        loss.backward()
        optimizer.step()
        Z.data[idx] = nn.functional.normalize(Z.data[idx], dim=1)

        if it % 100 == 0 or it == n_iters - 1:
            Kz_batch = Z_norm @ Z_norm.T
            rmse_norm = batch_rmse(Kz_batch, Kx).item()
            print(f"[{it}/{n_iters}] Loss: {loss.item():.6f} | RMSE Norm: {rmse_norm:.4f}")

    return Z.detach().cpu().numpy()

# ============================================================
# 7. Kernel Diagnostics
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
# 8. Main Execution
# ============================================================
if __name__ == "__main__":
    DATA_DIR = "/home/breallis/datasets/ebird_weekly_2023_albers"
    OUT_DIR = "/home/breallis/dev/range_limits_pymc/misc_outputs/rbf_stochastic"
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load and flatten
    ebird_stack, meta = load_tifs_structured(DATA_DIR, "*_abundance_median_2023-*.tif")
    H, W, D = ebird_stack.shape
    valid_mask = np.any(~np.isnan(ebird_stack), axis=-1)
    valid_flat = valid_mask.flatten()
    ebird_flat_raw = np.nan_to_num(ebird_stack).reshape(-1, D)[valid_flat]

    # Smooth + Hellinger
    ebird_flat_smooth = smoothed_hellinger_transform(ebird_flat_raw, meta['n_weeks'], sigma=0.5)

    # RBF Nyström Init
    Z_init = rbf_nystrom_init(ebird_flat_smooth, latent_dim=32, lengthscale=0.55, alpha=1, n_landmarks=20000)

    # Stochastic RBF fit
    Z_stoch = stochastic_rbf_fit(
        X=ebird_flat_smooth,
        Z_init=Z_init,
        lengthscale=0.55,
        batch_size=8192,
        n_iters=2000,
        lr=0.001,
        device='cuda'
    )

    # Diagnostics
    diag = compute_kernel_diagnostics(Z_stoch, ebird_flat_smooth)
    print(f"Stochastic RBF | Rank: {diag['effective_rank']:.2f}, RMSE Norm: {diag['rmse_norm']:.4f}")

    # Save
    # We save BOTH the Z matrix and the mask used to create it.
    np.save(os.path.join(OUT_DIR, "Z_stochastic.npy"), Z_stoch)
    np.save(os.path.join(OUT_DIR, "valid_mask.npy"), valid_mask)
    pd.DataFrame([diag]).to_csv(os.path.join(OUT_DIR, "diagnostics.csv"), index=False)
    print(f"Saved Z_stochastic.npy, valid_mask.npy, and diagnostics.csv to {OUT_DIR}")