import os
import glob
import re
import numpy as np
import rasterio
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import pandas as pd

# ============================================================
# 1. LOADERS 
# ============================================================
def load_tifs_structured(folder, pattern="*_abundance_median_*.tif"):
    """
    Returns: stack (H, W, S*T) with NaNs preserved.
    Used for loading the eBird abundance targets.
    """
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
    if df.empty: raise ValueError("No filenames matched regex.")
    
    df_sorted = df.sort_values(by=['species', 'date'])
    ordered_paths = df_sorted['path'].tolist()
    
    with rasterio.open(ordered_paths[0]) as src:
        H, W = src.shape
        
    print(f"Loading {len(ordered_paths)} raw abundance rasters (preserving NaNs)...")
    full_stack = np.zeros((H, W, len(ordered_paths)), dtype=np.float32)
    
    for i, p in enumerate(ordered_paths):
        with rasterio.open(p) as src:
            full_stack[:, :, i] = src.read(1) # Keep NaNs!

    return full_stack

def compute_strict_mask(ebird_stack, prism_stack, bui_stack):
    """
    Implements the strict union/intersection logic.
    """
    print("Computing strict validity mask...")
    
    # 1. eBird: Valid if AT LEAST ONE layer is valid (Union)
    mask_ebird = np.any(~np.isnan(ebird_stack), axis=-1)
    print(f" -> eBird valid pixels: {np.sum(mask_ebird)}")
    
    # 2. Env: Pixel is valid only if ALL features are present
    mask_prism = np.all(~np.isnan(prism_stack), axis=-1)
    print(f" -> PRISM valid pixels: {np.sum(mask_prism)}")
    
    mask_bui = np.all(~np.isnan(bui_stack), axis=-1)
    print(f" -> BUI valid pixels:   {np.sum(mask_bui)}")
    
    # 3. Intersection
    final_mask = mask_ebird & mask_prism & mask_bui
    print(f" -> FINAL INTERSECTION: {np.sum(final_mask)}")
    
    return final_mask

# ============================================================
# 2. DATASETS (Supervised & Historical)
# ============================================================
class PixelDataset(Dataset):
    """
    Standard Supervised Dataset (2023 Data).
    Contains: Env (P, B), Reference Z (z_ref), and Raw Species (x_raw).
    """
    def __init__(self, p_flat, b_flat, z_flat, x_flat, split="train"):
        self.p_flat = p_flat
        self.b_flat = b_flat
        self.z_flat = z_flat
        self.x_flat = x_flat

        # Train/Val Split (80/20 sequential)
        N = self.p_flat.shape[0]
        indices = torch.randperm(N) 
        split_idx = int(0.8 * N)
        
        train_idx = indices[:split_idx]
        val_idx = indices[split_idx:]
        
        if split == "train":
            self.idx = train_idx
        else:
            self.idx = val_idx
            
        print(f"[Supervised-{split}] Dataset ready. N={len(self.idx)}")

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        idx = self.idx[i]
        return self.p_flat[idx], self.b_flat[idx], self.z_flat[idx], self.x_flat[idx]

class HistoricalDataset(Dataset):
    """
    Unsupervised Dataset (1900-2023 Bag of Vectors).
    Contains: Env (P, B) only.
    Handles slicing and normalization internally.
    """
    def __init__(self, path, stats, p_dim=84):
        print(f"Loading historical bag from {path}...")
        # Load unstructured bag (N, 91)
        raw_data = np.load(path)
        self.data = torch.tensor(raw_data, dtype=torch.float32)
        
        # Slice PRISM (First p_dim cols) and BUI (Remaining cols)
        self.p_data = self.data[:, :p_dim]
        self.b_data = self.data[:, p_dim:]
        
        # APPLY 2023 NORMALIZATION (CRITICAL FOR MANIFOLD ALIGNMENT)
        print("Normalizing history with 2023 stats...")
        self.p_data = (self.p_data - stats['p_mu']) / (stats['p_sd'] + 1e-6)
        
        # Apply BUI Power Transform AND Normalization
        self.b_data = (self.b_data**0.1 - stats['b_mu']) / (stats['b_sd'] + 1e-6)
        
        print(f"[Historical] Dataset ready. N={len(self.p_data)}")

    def __len__(self):
        return len(self.p_data)

    def __getitem__(self, idx):
        # Return only environmental vars
        return self.p_data[idx], self.b_data[idx]

# ============================================================
# 3. KERNEL LOGIC
# ============================================================
def true_kernel_loss(z_pred, x_raw, num_pairs=4096):
    """
    Computes loss between the dot product in Z and the Ruzicka similarity in X.
    Only applicable to the Supervised batch.
    """
    B = z_pred.shape[0]
    idx = torch.randint(0, B, (2, num_pairs), device=z_pred.device)
    i, j = idx[0], idx[1]
    
    xi, xj = x_raw[i], x_raw[j]
    
    # Ruzicka
    sum_plus = xi + xj
    diff_abs = torch.abs(xi - xj)
    numerator = 0.5 * torch.sum(sum_plus - diff_abs, dim=1)
    denominator = 0.5 * torch.sum(sum_plus + diff_abs, dim=1)
    
    # Mask out "Double Zero" pairs to prevent gradient explosion
    valid_pairs = denominator > 1e-3
    if valid_pairs.sum() == 0:
         return torch.tensor(0.0, device=z_pred.device, requires_grad=True)

    sim_true = numerator[valid_pairs] / (denominator[valid_pairs] + 1e-8)
    
    # Dot Product in Z
    zi, zj = z_pred[i][valid_pairs], z_pred[j][valid_pairs]
    sim_pred = (zi * zj).sum(dim=1)
    
    return F.mse_loss(sim_pred, sim_true)

# ============================================================
# 4. MODEL
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
            nn.Linear(prism_dim, h), nn.GELU(),
            BMLPBlock(h), BMLPBlock(h),
        )
        self.bui_enc = nn.Sequential(
            nn.Linear(bui_dim, h), nn.GELU(),
            BMLPBlock(h), BMLPBlock(h),
        )
        self.mixer = nn.Sequential(
            nn.Linear(2*h, 2*h), nn.GELU(),
            nn.Linear(2*h, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 2*h), nn.GELU(),
            nn.Linear(2*h, prism_dim + bui_dim),
        )

    def forward(self, prism, bui):
        h = torch.cat([self.prism_enc(prism), self.bui_enc(bui)], dim=1)
        z_pred = self.mixer(h)
        return z_pred, self.decoder(z_pred)

# ============================================================
# 5. EXECUTION (Semi-Supervised)
# ============================================================
def train_model_semisup(train_ds, val_ds, hist_ds, dims, epochs=50, lr=1e-3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = MultiInputAutoencoder(dims['p'], dims['b'], dims['z']).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
    
    # 1. Dual DataLoaders
    # Supervised: Standard epoch-based iteration
    loader_sup = DataLoader(train_ds, batch_size=4096, shuffle=True, drop_last=True)
    
    # Unsupervised: Infinite cycling
    loader_unsup = DataLoader(hist_ds, batch_size=4096, shuffle=True, drop_last=True)
    iter_unsup = iter(loader_unsup)
    
    # 2. Weights
    # Stabilizing Loss (MSE to Z_ref)
    W_STAB = 1.0
    # Metric Geometry Loss (Ruzicka)
    W_TRUE = 5.0
    # Reconstruction (Applied to BOTH 2023 and History)
    W_RECON = 0.1 

    print("--- Starting Semi-Supervised Training ---")

    for ep in range(1, epochs+1):
        model.train()
        total_recon_hist = 0.0
        steps = 0
        
        for p, b, z_ref, x_raw in loader_sup:
            steps += 1
            # A. Load Supervised Batch (2023)
            p, b, z_ref, x_raw = p.to(device), b.to(device), z_ref.to(device), x_raw.to(device)
            
            # B. Load Unsupervised Batch (History)
            try:
                p_hist, b_hist = next(iter_unsup)
            except StopIteration:
                iter_unsup = iter(loader_unsup)
                p_hist, b_hist = next(iter_unsup)
            p_hist, b_hist = p_hist.to(device), b_hist.to(device)
            
            opt.zero_grad()
            
            # --- Forward Pass A (Supervised) ---
            # Calculates Gradient for: Encoder, Mixer, Decoder
            z_pred, recon = model(p, b)
            
            loss_stab = torch.mean(torch.sum((z_pred - z_ref)**2, dim=1))
            loss_true = true_kernel_loss(z_pred, x_raw)
            target_sup = torch.cat([p, b], dim=1)
            loss_recon_sup = F.mse_loss(recon, target_sup)
            
            # --- Forward Pass B (History Regularization) ---
            # Calculates Gradient for: Encoder, Mixer, Decoder (But NO Metric Loss)
            _, recon_hist = model(p_hist, b_hist)
            target_hist = torch.cat([p_hist, b_hist], dim=1)
            loss_recon_hist = F.mse_loss(recon_hist, target_hist)
            
            # --- Combine ---
            # We reconstruct both present and past to ensure the manifold handles both
            loss_final = (W_STAB * loss_stab) + \
                         (W_TRUE * loss_true) + \
                         (W_RECON * (loss_recon_sup + loss_recon_hist))
            
            loss_final.backward()
            opt.step()
            
            total_recon_hist += loss_recon_hist.item()

        # Validation (Supervised Only)
        model.eval()
        with torch.no_grad():
            val_iter = iter(DataLoader(val_ds, batch_size=4096))
            p_v, b_v, z_v, x_v = next(val_iter)
            p_v, b_v, z_v, x_v = p_v.to(device), b_v.to(device), z_v.to(device), x_v.to(device)

            z_pred, recon = model(p_v, b_v)
            
            target = torch.cat([p_v, b_v], dim=1)
            loss_recon = F.mse_loss(recon, target).item()
            loss_stab = torch.mean(torch.sum((z_pred - z_v)**2, dim=1)).item()
            loss_true = true_kernel_loss(z_pred, x_v).item()
            cos_stab = F.cosine_similarity(z_pred, z_v).mean().item()
            mag_p = z_pred.norm(dim=1).mean().item()
            
            # Avg Hist Loss for logging
            avg_hist = total_recon_hist / steps

            # Step scheduler on stability (or could use loss_true)
            scheduler.step(loss_stab)
            
            print(f"Ep {ep:03d} | Stab: {loss_stab:.4f} | True: {loss_true:.4f} | "
                  f"Rec(Sup/His): {loss_recon:.4f}/{avg_hist:.4f} | Cos: {cos_stab:.3f}")

    return model

if __name__ == "__main__":
    # 
    
    DATA_DIR = "/home/breallis/datasets"
    HIST_DIR = f"{DATA_DIR}/smoothed_prism_bui"
    
    # Output/Z Directory
    Z_DIR = "/home/breallis/dev/range_limits_pymc/misc_outputs/ruzicka_sweep_global/sigma_1.5/" 
    OUT_DIR = os.path.join(Z_DIR, "results")
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. LOAD DATA
    # A. Load Species Targets (Raw eBird)
    ebird_stack = load_tifs_structured(f"{DATA_DIR}/ebird_weekly_2023_albers", "*_abundance_median_2023-*.tif")
    
    # B. Load 2023 Environmental EMA State (Supervised Inputs)
    # Replaces raw TIF loading to ensure consistency with history
    print("Loading 2023 EMA State...")
    state_2023 = np.load(os.path.join(HIST_DIR, "state_2023_bio_ema10.npz"))
    prism_stack = state_2023['prism'] # (H, W, 84)
    bui_stack = state_2023['bui']     # (H, W, 7)
    
    H, W, _ = prism_stack.shape

    # 2. LOAD PREVIOUS Z DATA
    try:
        old_mask = np.load(os.path.join(Z_DIR, "valid_mask.npy"))
        z_old_flat = np.load(os.path.join(Z_DIR, "Z.npy"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find Z.npy or valid_mask.npy in {Z_DIR}")

    # 3. COMPUTE STRICT MASK
    # Valid = (Strict Env Data) AND (Pixel exists in Z_ref)
    strict_data_mask = compute_strict_mask(ebird_stack, prism_stack, bui_stack)
    final_valid_mask = strict_data_mask & old_mask
    
    print(f"Final Intersection Mask: {final_valid_mask.sum()} pixels")
    np.save(os.path.join(OUT_DIR, "training_mask.npy"), final_valid_mask)

    # 4. PREPARE SUPERVISED TENSORS
    # Prism/BUI (Flattened with mask)
    p_flat = torch.tensor(prism_stack[final_valid_mask], dtype=torch.float32)
    b_flat = torch.tensor(bui_stack[final_valid_mask], dtype=torch.float32)
    
    # eBird (Flattened, then fill NaNs with 0)
    x_flat_np = ebird_stack[final_valid_mask]
    x_flat_np = np.nan_to_num(x_flat_np, nan=0.0)
    x_flat = torch.tensor(x_flat_np, dtype=torch.float32)

    # Z Ref (Map back to grid, then re-extract)
    Z_grid = np.zeros((H, W, z_old_flat.shape[1]), dtype=np.float32)
    Z_grid[old_mask] = z_old_flat
    z_flat = torch.tensor(Z_grid[final_valid_mask], dtype=torch.float32)

    # 5. NORMALIZATION
    # Calculate stats on 2023 data
    p_mu, p_sd = p_flat.mean(0), p_flat.std(0)
    b_mu, b_sd = (b_flat**0.1).mean(0), (b_flat**0.1).std(0)
    
    # Apply to 2023 tensors
    p_flat = (p_flat - p_mu) / (p_sd + 1e-6)
    b_flat = (b_flat**0.1 - b_mu) / (b_sd + 1e-6)

    # Package stats for History Dataset
    stats_dict = {
        'p_mu': p_mu, 'p_sd': p_sd,
        'b_mu': b_mu, 'b_sd': b_sd
    }

    # 6. DATASETS
    # Supervised (2023)
    train_ds = PixelDataset(p_flat, b_flat, z_flat, x_flat, split="train")
    val_ds = PixelDataset(p_flat, b_flat, z_flat, x_flat, split="val")
    
    # Unsupervised (History 1900-2023)
    hist_ds = HistoricalDataset(
        os.path.join(HIST_DIR, "history_vectors_bio_ema10.npy"),
        stats=stats_dict,
        p_dim=p_flat.shape[1] # Pass 84 explicitly
    )

    dims = {'p': p_flat.shape[1], 'b': b_flat.shape[1], 'z': z_flat.shape[1]}
    
    # 7. TRAIN
    model = train_model_semisup(train_ds, val_ds, hist_ds, dims, epochs=100)
    
    torch.save(model.state_dict(), os.path.join(OUT_DIR, "env_model_semisup.pth"))