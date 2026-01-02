import os
import glob
import re
import numpy as np
import pandas as pd
import rasterio
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import gaussian_filter1d
from datetime import datetime

# ============================================================
# 1. Data Utilities (Adapted for eBird Input)
# ============================================================

def load_ebird_stack(folder, pattern):
    """
    Loads the raw abundance stack to be used as INPUT features.
    """
    files = sorted(glob.glob(os.path.join(folder, pattern)))
    if not files:
        raise ValueError(f"No files found: {folder}/{pattern}")
    
    # Load dimensions
    with rasterio.open(files[0]) as src:
        H, W = src.shape
        
    print(f"Loading {len(files)} abundance layers...")
    
    # We load into a pre-allocated array for speed
    # (H, W, Channels)
    stack = np.zeros((H, W, len(files)), dtype=np.float32)
    
    for i, p in enumerate(files):
        with rasterio.open(p) as src:
            stack[:, :, i] = src.read(1)
            
    return stack

def smoothed_hellinger_transform(ebird_flat, n_weeks, sigma=0.5):
    """
    Applies the exact transform used to generate Z.
    We must feed the network the same 'language' Z was trained on.
    """
    N, D = ebird_flat.shape
    n_species = D // n_weeks
    data_3d = ebird_flat.reshape(N, n_species, n_weeks)
    
    # 1. Temporal Smoothing
    if sigma > 1e-5:
        data_smoothed = gaussian_filter1d(data_3d, sigma=sigma, axis=-1, mode='wrap')
    else:
        data_smoothed = data_3d
        
    # 2. Hellinger Normalization
    data_flat = data_smoothed.reshape(N, -1)
    row_sums = data_flat.sum(axis=1, keepdims=True)
    row_sums[row_sums < 1e-9] = 1.0 
    return np.sqrt(data_flat / row_sums)

def load_z_matrix(path, H, W, valid_mask):
    """Loads and aligns Z, same as before."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Z matrix not found at {path}")
    
    Z_flat = np.load(path) 
    N_valid = np.sum(valid_mask)
    N_Z = Z_flat.shape[0]
    
    print(f"Aligning Z ({N_Z} rows) to Map ({N_valid} valid pixels)...")
    
    if N_Z > N_valid:
        print(f"WARNING: Z has {N_Z - N_valid} extra rows. Truncating.")
        Z_flat = Z_flat[:N_valid]
    elif N_Z < N_valid:
        print(f"WARNING: Mask has {N_valid - N_Z} extra pixels. Truncating mask.")
        flat_indices = np.where(valid_mask.flatten())[0]
        valid_indices = flat_indices[:N_Z]
        new_mask = np.zeros(H*W, dtype=bool)
        new_mask[valid_indices] = True
        valid_mask = new_mask.reshape(H, W)

    Z_full = np.zeros((H, W, Z_flat.shape[1]), dtype=np.float32)
    Z_full[valid_mask] = Z_flat
    return Z_full

# ============================================================
# 2. Dataset
# ============================================================

class AbundanceDataset(Dataset):
    def __init__(self, abundance_features, z_target, block_rows=8, block_cols=8, split="train"):
        H, W, C = abundance_features.shape
        blocks = []

        row_blocks = np.array_split(np.arange(H), block_rows)
        col_blocks = np.array_split(np.arange(W), block_cols)

        for rb in row_blocks:
            for cb in col_blocks:
                # Features: The transformed abundance vectors
                feat = abundance_features[rb[:, None], cb] 
                # Target: The Latent Z
                z = z_target[rb[:, None], cb]

                feat = feat.reshape(-1, C)
                z = z.reshape(-1, z_target.shape[-1])

                # Mask: Valid Z (non-zero) AND Valid Features (not all zero)
                mask = (
                    (np.abs(z).sum(axis=1) > 1e-9) &
                    (feat.sum(axis=1) > 1e-9)
                )

                if mask.sum() > 0:
                    blocks.append((
                        torch.tensor(feat[mask]),
                        torch.tensor(z[mask]),
                    ))

        split_idx = int(0.8 * len(blocks))
        blocks = blocks[:split_idx] if split == "train" else blocks[split_idx:]

        self.features = torch.cat([x[0] for x in blocks])
        self.z_target = torch.cat([x[1] for x in blocks])

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return self.features[idx], self.z_target[idx]

# ============================================================
# 3. Model Architecture (Simple Encoder)
# ============================================================

class BMLPBlock(nn.Module):
    def __init__(self, m, k=4, dropout=0.5):
        super().__init__()
        self.ln = nn.LayerNorm(m)
        self.fc1 = nn.Linear(m, m*k)
        self.fc2 = nn.Linear(m*k, m)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        z = self.ln(x)
        z = F.gelu(self.fc1(z))
        z = self.drop(z)
        z = self.fc2(z)
        return x + z

class AbundanceAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        # 1300 inputs -> compress to Latent
        h = 512 # Robust hidden size

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h),
            nn.GELU(),
            BMLPBlock(h),
            BMLPBlock(h),
            nn.Linear(h, latent_dim) # Output raw Z
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h),
            nn.GELU(),
            BMLPBlock(h),
            nn.Linear(h, input_dim) # Reconstruct abundance
        )

    def forward(self, x):
        z_raw = self.encoder(x)
        # Enforce hypersphere to match Z target
        z_norm = F.normalize(z_raw, dim=1)
        recon = self.decoder(z_norm)
        return z_norm, recon

# ============================================================
# 4. Losses & Diagnostics
# ============================================================

def reconstruction_loss(recon, target):
    return F.mse_loss(recon, target)

def metric_proxy_loss(z_pred, z_target):
    return F.mse_loss(z_pred, z_target)

# ... [Previous imports and classes remain the same] ...

# ============================================================
# 4. Diagnostics (Restored)
# ============================================================

# ============================================================
# 4. Diagnostics (Updated for End-to-End Check)
# ============================================================

@torch.no_grad()
def compute_kernel_diagnostics(z_pred, z_target, raw_features, max_pairs=1024):
    """
    Computes geometric fidelity against BOTH the Z-Target and the Raw Biology.
    """
    B = z_pred.shape[0]
    device = z_pred.device

    # Subsample
    if B > max_pairs:
        idx = torch.randperm(B, device=device)[:max_pairs]
        z_p = z_pred[idx]
        z_t = z_target[idx]
        raw = raw_features[idx]
    else:
        z_p, z_t, raw = z_pred, z_target, raw_features

    # 1. Compute Kernels
    K_pred = z_p @ z_p.T
    K_targ = z_t @ z_t.T
    K_bio  = raw @ raw.T # The original Hellinger similarity
    
    # 2. Fidelity to Z-Target (Approximation Error)
    mse_approx = ((K_pred - K_targ) ** 2).mean()
    scale_targ = torch.sqrt((K_targ ** 2).mean())
    rmse_approx_norm = torch.sqrt(mse_approx) / (scale_targ + 1e-8)

    # 3. Fidelity to Raw Biology (Total System Error)
    mse_total = ((K_pred - K_bio) ** 2).mean()
    scale_bio = torch.sqrt((K_bio ** 2).mean())
    rmse_total_norm = torch.sqrt(mse_total) / (scale_bio + 1e-8)

    # 4. Alignment & Rank
    cos_sim = (z_p * z_t).sum(dim=1).mean()
    
    svals = torch.linalg.svdvals(z_p)
    eff_rank = (svals / svals.sum()).pow(2).sum().reciprocal()

    return {
        "rmse_vs_Z": rmse_approx_norm.item(),    # Neural Net Error
        "rmse_vs_Bio": rmse_total_norm.item(),   # End-to-End Error
        "cos_sim": cos_sim.item(),
        "rank": eff_rank.item(),
    }

# ============================================================
# 5. Training Loop (Updated)
# ============================================================

def train_sanity_check(train, val, input_dim, latent_dim,
                       batch_size=4096, epochs=50, lr=1e-3):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Sanity Check: Mapping {input_dim} features -> {latent_dim} Z-dims")
    
    model = AbundanceAutoencoder(input_dim, latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
    loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    
    for ep in range(1, epochs+1):
        model.train()
        for x, z_t in loader:
            x, z_t = x.to(device), z_t.to(device)
            opt.zero_grad()
            z_pred, recon = model(x)
            
            # Loss focuses on Z fidelity
            loss = metric_proxy_loss(z_pred, z_t) + 0.1 * reconstruction_loss(recon, x)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            x_v = val.features.to(device)
            z_v = val.z_target.to(device)
            z_pred, recon = model(x_v)
            
            z_loss = metric_proxy_loss(z_pred, z_v).item()
            
            # Pass inputs (x_v) to diagnostics now
            diag = compute_kernel_diagnostics(z_pred, z_v, x_v)
            
        scheduler.step(z_loss)
        
        print(
            f"Ep {ep:02d} | "
            f"Z_MSE: {z_loss:.5f} | "
            f"CosSim: {diag['cos_sim']:.4f} | "
            f"Err_vs_Z: {diag['rmse_vs_Z']:.4f} | "
            f"Err_vs_Bio: {diag['rmse_vs_Bio']:.4f} | " # <--- NEW METRIC
            f"Rank: {diag['rank']:.2f}"
        )

# ============================================================
# 6. Main
# ============================================================
if __name__ == "__main__":
    DATA_DIR = "/home/breallis/datasets"
    Z_PATH = "/home/breallis/dev/range_limits_pymc/misc_outputs/rbf_stochastic/Z_stochastic.npy"
    
    # 1. Load Raw Abundance Stack
    print("--- 1. Loading Abundance Data ---")
    ebird_stack = load_ebird_stack(f"{DATA_DIR}/ebird_weekly_2023_albers", "*_abundance_median_2023-*.tif")
    H, W, D = ebird_stack.shape
    
    # 2. Recreate Mask (Using the stack itself)
    print("--- 2. Generating Valid Mask ---")
    valid_mask = np.any(~np.isnan(ebird_stack), axis=-1)
    
    # 3. Load Z
    print(f"--- 3. Hydrating Z ({Z_PATH}) ---")
    z_full = load_z_matrix(Z_PATH, H, W, valid_mask)
    latent_dim = z_full.shape[-1]
    
    # 4. Transform Features (Critical Step: Match Z training data)
    print("--- 4. Applying Hellinger Transform ---")
    # We flatten only the valid pixels to save memory/compute
    valid_flat = valid_mask.flatten()
    features_raw = np.nan_to_num(ebird_stack).reshape(-1, D)[valid_flat]
    
    # 52 weeks is hardcoded based on filename knowledge, or derive from D/25
    features_transformed = smoothed_hellinger_transform(features_raw, n_weeks=52, sigma=0.5)
    
    # Hydrate features back to (H, W, D) for block-splitting
    features_full = np.zeros((H, W, D), dtype=np.float32)
    features_full[valid_mask] = features_transformed
    
    # 5. Create Datasets
    print("--- 5. Creating Datasets ---")
    train_ds = AbundanceDataset(features_full, z_full, split="train")
    val_ds   = AbundanceDataset(features_full, z_full, split="val")
    
    # 6. Run Sanity Check
    train_sanity_check(train_ds, val_ds, input_dim=D, latent_dim=latent_dim)