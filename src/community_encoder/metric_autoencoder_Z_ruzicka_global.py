import os
import glob
import numpy as np
import pandas as pd
import rasterio
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 1. LOADERS & UTILITIES
# ============================================================
def load_env_tifs(folder, pattern):
    paths = sorted(glob.glob(os.path.join(folder, pattern)))
    arrays = []
    if not paths:
        raise ValueError(f"No environmental files found: {folder}/{pattern}")
    for p in paths:
        with rasterio.open(p) as src:
            data = src.read()
            data = np.moveaxis(data, 0, -1)
            arrays.append(data.astype(np.float32))
    return np.concatenate(arrays, axis=-1)

def load_or_recover_mask(z_dir, ebird_dir, ebird_pattern="*_abundance_median_*.tif"):
    """
    Tries to load 'valid_mask.npy' from the Z directory. 
    If missing, falls back to scanning the eBird stack.
    """
    mask_path = os.path.join(z_dir, "valid_mask.npy")
    
    if os.path.exists(mask_path):
        print(f"Loading existing mask from: {mask_path}")
        return np.load(mask_path)
    
    print(f"Mask not found at {mask_path}. Recovering from TIFs...")
    return recover_union_mask(ebird_dir, ebird_pattern)

def recover_union_mask(folder, pattern):
    files = sorted(glob.glob(os.path.join(folder, pattern)))
    if not files:
        raise ValueError(f"No eBird files found in {folder}")
    print(f"Scanning {len(files)} files to build Union Mask...")
    with rasterio.open(files[0]) as src:
        H, W = src.shape
    union_mask = np.zeros((H, W), dtype=bool)
    for i, p in enumerate(files):
        with rasterio.open(p) as src:
            data = src.read(1)
            union_mask |= ~np.isnan(data)
    total_pixels = np.sum(union_mask)
    print(f"Union Mask Complete. Total valid pixels: {total_pixels}")
    return union_mask

def load_z_matrix(path, H, W, valid_mask):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Z matrix not found at {path}")
    
    Z_flat = np.load(path) 
    N_valid = np.sum(valid_mask)
    N_Z = Z_flat.shape[0]
    
    print(f"Aligning Z ({N_Z} rows, {Z_flat.shape[1]} dims) to Map ({N_valid} valid pixels)...")
    
    if N_Z == N_valid:
        print(" -> Perfect alignment.")
    elif N_Z > N_valid:
        print(f"WARNING: Z has {N_Z - N_valid} more rows than mask. Truncating Z.")
        Z_flat = Z_flat[:N_valid]
    elif N_Z < N_valid:
        print(f"WARNING: Mask has {N_valid - N_Z} more pixels than Z. Truncating mask.")
        flat_indices = np.where(valid_mask.flatten())[0]
        valid_indices = flat_indices[:N_Z]
        new_mask_flat = np.zeros(H*W, dtype=bool)
        new_mask_flat[valid_indices] = True
        valid_mask = new_mask_flat.reshape(H, W)

    Z_full = np.zeros((H, W, Z_flat.shape[1]), dtype=np.float32)
    Z_full[valid_mask] = Z_flat
    return Z_full

# ============================================================
# 2. DATASET & NORMALIZATION
# ============================================================
class PixelDataset(Dataset):
    def __init__(self, prism, bui, z_target, block_rows=8, block_cols=8, split="train"):
        H, W, _ = prism.shape
        blocks = []
        row_blocks = np.array_split(np.arange(H), block_rows)
        col_blocks = np.array_split(np.arange(W), block_cols)

        for rb in row_blocks:
            for cb in col_blocks:
                p = prism[rb[:, None], cb]
                b = bui[rb[:, None], cb]
                z = z_target[rb[:, None], cb]

                p = p.reshape(-1, prism.shape[-1])
                b = b.reshape(-1, bui.shape[-1])
                z = z.reshape(-1, z_target.shape[-1])

                mask = (
                    ~np.any(np.isnan(p), axis=1) &
                    ~np.any(np.isnan(b), axis=1) &
                    (np.abs(z).sum(axis=1) > 1e-9) 
                )

                if mask.sum() > 0:
                    blocks.append((
                        torch.tensor(p[mask]),
                        torch.tensor(b[mask]),
                        torch.tensor(z[mask]),
                    ))

        split_idx = int(0.8 * len(blocks))
        blocks = blocks[:split_idx] if split == "train" else blocks[split_idx:]
        
        self.prism = torch.cat([x[0] for x in blocks])
        self.bui   = torch.cat([x[1] for x in blocks])
        self.z_target = torch.cat([x[2] for x in blocks])

    def __len__(self):
        return self.prism.shape[0]

    def __getitem__(self, idx):
        return self.prism[idx], self.bui[idx], self.z_target[idx]

def normalize_dataset(train, val):
    stats = {}
    
    mu = train.prism.mean(0, keepdim=True)
    sd = train.prism.std(0, keepdim=True)
    train.prism = (train.prism - mu) / (sd + 1e-6)
    val.prism   = (val.prism   - mu) / (sd + 1e-6)
    stats['prism_mu'] = mu
    stats['prism_sd'] = sd

    bui_train = train.bui ** 0.1
    mu = bui_train.mean(0, keepdim=True)
    sd = bui_train.std(0, keepdim=True)
    train.bui = (bui_train - mu) / (sd + 1e-6)
    val.bui   = ((val.bui ** 0.1) - mu) / (sd + 1e-6)
    stats['bui_mu'] = mu
    stats['bui_sd'] = sd
    return train, val, stats

# ============================================================
# 3. GEOGRAPHIC COVARIANCE PROBE
# ============================================================
class GeoCovarianceProbe:
    def __init__(self, Z, valid_mask, H, W, out_dir):
        self.Z = Z
        self.mask = valid_mask
        self.H, self.W = H, W
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        
        y_grid, x_grid = np.indices((H, W))
        self.coords = np.stack([y_grid[valid_mask], x_grid[valid_mask]], axis=1)
        self.Z_full = np.full((H, W, Z.shape[1]), np.nan)
        self.Z_full[valid_mask] = Z

    def plot_iso_similarity(self, focal_points_idx=None):
        print(" -> Generating Iso-Similarity Maps...")
        N = self.Z.shape[0]
        if focal_points_idx is None:
            focal_points_idx = np.random.choice(N, 4, replace=False)
            
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        axes = axes.flatten()
        
        for i, idx in enumerate(focal_points_idx):
            z_ref = self.Z[idx]
            # --- UPDATED PROBE TO USE RUZICKA LOGIC ---
            # Manually computing Ruzicka similarity for visualization
            # 2*min = a+b-|a-b|, 2*max = a+b+|a-b|
            sum_ab = self.Z + z_ref
            abs_diff = np.abs(self.Z - z_ref)
            num = 0.5 * np.sum(sum_ab - abs_diff, axis=1)
            den = 0.5 * np.sum(sum_ab + abs_diff, axis=1)
            sims = num / (den + 1e-8)
            
            sim_map = np.full((self.H, self.W), np.nan)
            sim_map[self.mask] = sims
            
            ax = axes[i]
            im = ax.imshow(sim_map, cmap='RdYlBu_r', vmin=0, vmax=1)
            
            fy, fx = self.coords[idx]
            ax.scatter(fx, fy, c='black', marker='*', s=200, edgecolors='white', label='Reference')
            ax.set_title(f"Predicted Similarity to Point ({fy}, {fx})")
            ax.axis('off')
            
        plt.tight_layout()
        plt.colorbar(im, ax=axes, fraction=0.046, pad=0.04, label="Global Ruzicka Similarity")
        plt.savefig(os.path.join(self.out_dir, "pred_geo_01_iso_similarity.png"), bbox_inches='tight')
        plt.close()

    def plot_variogram(self, max_dist=200, n_samples=5000):
        print(" -> Generating Community Variogram...")
        N = self.Z.shape[0]
        idx_a = np.random.choice(N, n_samples)
        idx_b = np.random.choice(N, n_samples)
        
        d_phys = np.linalg.norm(self.coords[idx_a] - self.coords[idx_b], axis=1)
        
        # --- UPDATED PROBE TO USE RUZICKA LOGIC ---
        za = self.Z[idx_a]
        zb = self.Z[idx_b]
        sum_ab = za + zb
        abs_diff = np.abs(za - zb)
        num = 0.5 * np.sum(sum_ab - abs_diff, axis=1)
        den = 0.5 * np.sum(sum_ab + abs_diff, axis=1)
        sims = num / (den + 1e-8)
        
        d_eco = 1.0 - sims
        
        mask = d_phys < max_dist
        d_phys = d_phys[mask]
        d_eco = d_eco[mask]
        
        plt.figure(figsize=(10, 6))
        plt.hexbin(d_phys, d_eco, gridsize=50, cmap='inferno', mincnt=1)
        try:
            sns.regplot(x=d_phys, y=d_eco, scatter=False, color='cyan', line_kws={'linestyle':'--'})
        except:
            pass
        plt.title("Predicted Speed of Ecology: Ruzicka Turnover vs. Distance")
        plt.xlabel("Physical Distance (pixels)")
        plt.ylabel("Predicted Ruzicka Distance")
        plt.savefig(os.path.join(self.out_dir, "pred_geo_02_variogram.png"), bbox_inches='tight')
        plt.close()

    def run_all(self):
        self.plot_iso_similarity()
        self.plot_variogram()

# ============================================================
# 4. MODEL & LOSSES
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

class MultiInputAutoencoder(nn.Module):
    def __init__(self, prism_dim, bui_dim, latent_dim):
        super().__init__()
        h = max(128, latent_dim * 4) 

        self.prism_enc = nn.Sequential(
            nn.Linear(prism_dim, h),
            nn.GELU(),
            BMLPBlock(h),
            BMLPBlock(h),
        )
        self.bui_enc = nn.Sequential(
            nn.Linear(bui_dim, h),
            nn.GELU(),
            BMLPBlock(h),
            BMLPBlock(h),
        )
        self.mixer = nn.Sequential(
            nn.Linear(2*h, 2*h),
            nn.GELU(),
            nn.Linear(2*h, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 2*h),
            nn.GELU(),
            nn.Linear(2*h, prism_dim + bui_dim),
        )

    def forward(self, prism, bui):
        h = torch.cat([self.prism_enc(prism), self.bui_enc(bui)], dim=1)
        z_raw = self.mixer(h)
        z_norm = F.normalize(z_raw, dim=1) 
        return z_norm, self.decoder(z_norm)

def reconstruction_loss(recon, prism, bui):
    target = torch.cat([prism, bui], dim=1)
    return F.mse_loss(recon, target)

def metric_proxy_loss(z_pred, z_target):
    return F.mse_loss(z_pred, z_target)

def global_ruzicka_similarity(x, y, eps=1e-8):
    """
    Computes Global Ruzicka (Generalized Jaccard) Similarity.
    Uses robust identity: 2*min(a,b) = a+b-|a-b|
    """
    # 2 * Min
    sum_plus = x + y
    diff_abs = torch.abs(x - y)
    
    # Sum over feature dim (dim=1)
    numerator = 0.5 * torch.sum(sum_plus - diff_abs, dim=1)
    denominator = 0.5 * torch.sum(sum_plus + diff_abs, dim=1)
    
    return numerator / (denominator + eps)

def metric_loss_preserve_kernel(z, z_target, num_pairs=4096):
    B = z.shape[0]
    idx = torch.randint(0, B, (2, num_pairs), device=z.device)
    i, j = idx[0], idx[1]
    
    # --- UPDATED LOSS: RUZICKA SIMILARITY ---
    sim_pred = global_ruzicka_similarity(z[i], z[j])
    
    with torch.no_grad():
        sim_targ = global_ruzicka_similarity(z_target[i], z_target[j])
        
    kernel_mse = ((sim_pred - sim_targ) ** 2).mean()
    return kernel_mse

# ============================================================
# 5. TRAINING LOOP
# ============================================================
def train_model(train, val, prism_dim, bui_dim, latent_dim, batch_size=16384, epochs=100, lr=1e-3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing model with Latent Dim: {latent_dim}")
    
    model = MultiInputAutoencoder(prism_dim, bui_dim, latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
    loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    
    # LOSS WEIGHTS (Dual Loss)
    lambda_z = 1.0         
    lambda_kernel = 0.5    
    lambda_recon = 0.05    

    for ep in range(1, epochs+1):
        model.train()
        for p, b, z_t in loader:
            p, b, z_t = p.to(device), b.to(device), z_t.to(device)
            opt.zero_grad()
            z_pred, recon = model(p, b)
            
            loss = (lambda_z * metric_proxy_loss(z_pred, z_t) + 
                    lambda_kernel * metric_loss_preserve_kernel(z_pred, z_t) + 
                    lambda_recon * reconstruction_loss(recon, p, b))
            
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            p_v = val.prism.to(device)
            b_v = val.bui.to(device)
            z_v = val.z_target.to(device)
            z_pred, recon = model(p_v, b_v)

            val_recon = reconstruction_loss(recon, p_v, b_v).item()
            val_z_mse = metric_proxy_loss(z_pred, z_v).item()
            
            # Diagnostics using Ruzicka
            B = z_pred.shape[0]
            idx = torch.randperm(B)[:1024]
            z_p, z_t_sub = z_pred[idx], z_v[idx]
            
            # Diagnostic: Average Pairwise Ruzicka Similarity
            ruz_sim = global_ruzicka_similarity(z_p, z_t_sub).mean().item()
            
            # Compute Full Pairwise Matrices for RMSE
            # Note: Doing fully broadcasted Ruzicka on matrix is heavy, 
            # so we approximate kernel RMSE by just checking the diagonal/sampled pairs in logic above,
            # but for logging let's do a smaller subset full check
            M_sub = min(256, B)
            sub_p = z_p[:M_sub]
            sub_t = z_t_sub[:M_sub]
            
            # Helper for matrix broadcasting ruzicka
            def matrix_ruzicka(A):
                # A: (N, D)
                # Returns (N, N)
                sum_a = A.unsqueeze(1) + A.unsqueeze(0) # (N, N, D)
                abs_a = torch.abs(A.unsqueeze(1) - A.unsqueeze(0))
                num = 0.5 * (sum_a - abs_a).sum(dim=-1)
                den = 0.5 * (sum_a + abs_a).sum(dim=-1)
                return num / (den + 1e-8)

            Kp = matrix_ruzicka(sub_p)
            Kt = matrix_ruzicka(sub_t)
            
            k_mse = ((Kp - Kt)**2).mean()
            k_scale = torch.sqrt((Kt**2).mean())
            k_rmse_norm = torch.sqrt(k_mse) / (k_scale + 1e-8)
            
            svals = torch.linalg.svdvals(z_p)
            rank = (svals / svals.sum()).pow(2).sum().reciprocal().item()

            val_total = lambda_z * val_z_mse + lambda_recon * val_recon

        scheduler.step(val_total)
        print(f"Ep {ep:03d} | Z_MSE: {val_z_mse:.5f} | RuzSim: {ruz_sim:.4f} | "
              f"K_RMSE: {k_rmse_norm:.4f} | Rank: {rank:.2f}")

    return model

@torch.no_grad()
def generate_prediction_map(model, prism_full, bui_full, valid_mask, stats):
    model.eval()
    device = next(model.parameters()).device
    p_flat = prism_full[valid_mask]
    b_flat = bui_full[valid_mask]
    
    p_norm = (torch.tensor(p_flat) - stats['prism_mu']) / (stats['prism_sd'] + 1e-6)
    b_norm = (torch.tensor(b_flat**0.1) - stats['bui_mu']) / (stats['bui_sd'] + 1e-6)
    
    batch_size = 32000
    z_preds = []
    print(f"Generating full map prediction for {len(p_flat)} pixels...")
    for i in range(0, len(p_flat), batch_size):
        p_batch = p_norm[i:i+batch_size].to(device)
        b_batch = b_norm[i:i+batch_size].to(device)
        z_batch, _ = model(p_batch, b_batch)
        z_preds.append(z_batch.cpu().numpy())
    return np.concatenate(z_preds, axis=0)

# ============================================================
# 6. MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    DATA_DIR = "/home/breallis/datasets"
    Z_DIR = "/home/breallis/dev/range_limits_pymc/misc_outputs/rbf_stochastic"
    Z_PATH = os.path.join(Z_DIR, "Z_stochastic.npy")
    OUT_DIR = "/home/breallis/dev/range_limits_pymc/misc_outputs/env_model_results"
    os.makedirs(OUT_DIR, exist_ok=True)
    
    print("--- 1. Loading Env Data ---")
    prism = load_env_tifs(f"{DATA_DIR}/prism_monthly_4km_albers", "prism_*_2023*.tif")
    bui   = load_env_tifs(f"{DATA_DIR}/HBUI/BUI_4km_interp", "2020_BUI_4km_interp.tif")
    H, W, _ = prism.shape
    
    print("--- 2. Loading or Recovering Mask ---")
    # This logic now prioritizes the saved mask to ensure alignment
    valid_mask = load_or_recover_mask(
        Z_DIR, 
        f"{DATA_DIR}/ebird_weekly_2023_albers", 
        "*.tif" # Adjusted pattern slightly to be safer if specific file naming varies
    )
    
    print(f"--- 3. Loading Z ({Z_PATH}) ---")
    z_full = load_z_matrix(Z_PATH, H, W, valid_mask)
    actual_latent_dim = z_full.shape[-1]
    
    print("--- 4. Creating Datasets ---")
    train_ds = PixelDataset(prism, bui, z_full, split="train")
    val_ds   = PixelDataset(prism, bui, z_full, split="val")
    train_ds, val_ds, stats = normalize_dataset(train_ds, val_ds)

    print("--- 5. Training Environmental Model (Dual Loss) ---")
    model = train_model(
        train_ds, val_ds,
        prism_dim=train_ds.prism.shape[1],
        bui_dim=train_ds.bui.shape[1],
        latent_dim=actual_latent_dim,
        epochs=50 
    )
    
    print("--- 6. Saving Final Predictions ---")
    torch.save(model.state_dict(), os.path.join(OUT_DIR, "env_model.pth"))
    z_pred_flat = generate_prediction_map(model, prism, bui, valid_mask, stats)
    
    # Save predictions AND the mask they correspond to
    np.save(os.path.join(OUT_DIR, "Z_pred_env.npy"), z_pred_flat)
    np.save(os.path.join(OUT_DIR, "valid_mask.npy"), valid_mask)
    print(f"Saved Z_pred ({z_pred_flat.shape}) and Mask to {OUT_DIR}")

    print("--- 7. Running Geographic Probe on Predictions ---")
    probe = GeoCovarianceProbe(z_pred_flat, valid_mask, H, W, OUT_DIR)
    probe.run_all()