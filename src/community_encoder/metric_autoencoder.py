import os
import glob
import numpy as np
import rasterio
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import gaussian_filter1d

# ============================================================
# Spatial kernels (4 km grid)
# ============================================================

def nan_aware_gaussian(X, sigma):
    mask = np.isfinite(X).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0)

    num = gaussian_filter1d(X * mask, sigma, axis=0, mode="constant")
    num = gaussian_filter1d(num,        sigma, axis=1, mode="constant")

    den = gaussian_filter1d(mask, sigma, axis=0, mode="constant")
    den = gaussian_filter1d(den,  sigma, axis=1, mode="constant")

    out = num / (den + 1e-6)
    out[den < 1e-6] = np.nan
    return out

def add_spatial_context(X):
    feats = [X]
    feats.append(nan_aware_gaussian(X, sigma=5.0))
    # feats.append(nan_aware_gaussian(X, sigma=20.0))
    return np.concatenate(feats, axis=-1)



# ============================================================
# Data loading
# ============================================================

def load_tifs(folder, pattern):
    paths = sorted(glob.glob(os.path.join(folder, pattern)))
    arrays = []
    for p in paths:
        with rasterio.open(p) as src:
            data = src.read()                  # (bands, H, W)
            data = np.moveaxis(data, 0, -1)    # (H, W, bands)
            arrays.append(data.astype(np.float32))
    if not arrays:
        raise ValueError(f"No files found: {folder}/{pattern}")
    return np.concatenate(arrays, axis=-1)


def mask_ebird(ebird_array):
    valid = np.any(~np.isnan(ebird_array), axis=-1)
    return valid, np.nan_to_num(ebird_array, nan=0.0)


# ============================================================
# Dataset
# ============================================================

class PixelDataset(Dataset):
    def __init__(self, prism, bui, ebird, block_rows=8, block_cols=8, split="train"):
        H, W, _ = prism.shape
        blocks = []

        row_blocks = np.array_split(np.arange(H), block_rows)
        col_blocks = np.array_split(np.arange(W), block_cols)

        for rb in row_blocks:
            for cb in col_blocks:
                p = prism[rb[:, None], cb]
                b = bui[rb[:, None], cb]
                e = ebird[rb[:, None], cb]

                p = p.reshape(-1, prism.shape[-1])
                b = b.reshape(-1, bui.shape[-1])
                e = e.reshape(-1, ebird.shape[-1])

                mask = (
                    ~np.any(np.isnan(p), axis=1) &
                    ~np.any(np.isnan(b), axis=1) &
                    np.any(e != 0, axis=1)
                )

                if mask.sum() > 0:
                    blocks.append((
                        torch.tensor(p[mask]),
                        torch.tensor(b[mask]),
                        torch.tensor(e[mask]),
                    ))

        split_idx = int(0.8 * len(blocks))
        blocks = blocks[:split_idx] if split == "train" else blocks[split_idx:]

        self.prism = torch.cat([x[0] for x in blocks])
        self.bui   = torch.cat([x[1] for x in blocks])
        self.ebird = torch.cat([x[2] for x in blocks])

    def __len__(self):
        return self.prism.shape[0]

    def __getitem__(self, idx):
        return self.prism[idx], self.bui[idx], self.ebird[idx]


# ============================================================
# Normalization
# ============================================================

def normalize_dataset(train, val):
    # --- PRISM ---
    mu = train.prism.mean(0, keepdim=True)
    sd = train.prism.std(0, keepdim=True)
    train.prism = (train.prism - mu) / (sd + 1e-6)
    val.prism   = (val.prism   - mu) / (sd + 1e-6)

    # --- BUI ---
    bui_train = train.bui ** 0.1
    mu = bui_train.mean(0, keepdim=True)
    sd = bui_train.std(0, keepdim=True)
    train.bui = (bui_train - mu) / (sd + 1e-6)
    val.bui   = ((val.bui ** 0.1) - mu) / (sd + 1e-6)

    # --- eBird ---
    e_train = train.ebird ** 0.5
    scale = e_train.norm(dim=1).mean()
    train.ebird = e_train / scale
    val.ebird   = (val.ebird ** 0.5) / scale

    return train, val


# ============================================================
# Model
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
    def __init__(self, prism_dim, bui_dim, latent_dim=6):
        super().__init__()
        h = latent_dim * 4

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
        z = self.mixer(h)
        return z, self.decoder(z)


# ============================================================
# Losses
# ============================================================

# def reconstruction_loss(recon, prism, bui):
#     return F.mse_loss(recon, torch.cat([prism, bui], dim=1))

def reconstruction_loss(recon: torch.Tensor, prism: torch.Tensor, bui: torch.Tensor, num_pairs: int = 16384):
    """
    Dot-product–based reconstruction loss with optional random pairs.
    Produces RMSE roughly O(1) by normalizing each vector pair.

    Args:
        recon: model reconstruction (B x (prism_dim + bui_dim))
        prism: original prism input (B x prism_dim)
        bui: original bui input (B x bui_dim)
        num_pairs: number of random pairs to sample

    Returns:
        loss: scalar tensor
    """
    device = recon.device
    target = torch.cat([prism, bui], dim=1)
    B, D = target.shape

    # Random pair indices
    idx = torch.randint(0, B, (2, num_pairs), device=device)
    i, j = idx[0], idx[1]

    # Gather pairs
    recon_i, recon_j = recon[i], recon[j]
    target_i, target_j = target[i], target[j]

    # Normalize each vector to unit norm to bound dot products
    recon_i_norm = F.normalize(recon_i, p=2, dim=1)
    recon_j_norm = F.normalize(recon_j, p=2, dim=1)
    target_i_norm = F.normalize(target_i, p=2, dim=1)
    target_j_norm = F.normalize(target_j, p=2, dim=1)

    # Pairwise dot products
    dot_recon = torch.sum(recon_i_norm * recon_j_norm, dim=1)
    dot_target = torch.sum(target_i_norm * target_j_norm, dim=1)

    # MSE over sampled pairs
    loss = torch.mean((dot_recon - dot_target) ** 2)
    return loss



def metric_loss_preserve_kernel(z, eBird, num_pairs=16384, lambda_norm=1e-3):
    B = z.shape[0]
    idx = torch.randint(0, B, (2, num_pairs), device=z.device)
    i, j = idx[0], idx[1]

    dot_z = (z[i] * z[j]).sum(dim=1)
    dot_e = (eBird[i] * eBird[j]).sum(dim=1)

    kernel_mse = ((dot_z - dot_e) ** 2).mean()

    # per-sample norm alignment (BLR-critical)
    norm_loss = ((z.norm(dim=1) - eBird.norm(dim=1)) ** 2).mean()

    return kernel_mse + lambda_norm * norm_loss


# ============================================================
# Validation metrics
# ============================================================

@torch.no_grad()
def compute_kernel_diagnostics(
    z: torch.Tensor,
    eBird: torch.Tensor,
    max_pairs: int = 512,
):
    """
    Comprehensive kernel diagnostics for validation / logging.
    Does NOT affect training. Uses subsampling to control cost.

    Args:
        z: latent embeddings (B x latent_dim)
        eBird: abundance vectors (B x species_dim)
        max_pairs: number of samples used for kernel diagnostics

    Returns:
        dict of diagnostic scalars
    """
    B = z.shape[0]
    device = z.device

    # --- Subsample ---
    if B > max_pairs:
        idx = torch.randperm(B, device=device)[:max_pairs]
        z_s = z[idx]
        e_s = eBird[idx]
    else:
        z_s = z
        e_s = eBird

    # --- Dot-product kernels ---
    Kz = z_s @ z_s.T
    Ke = e_s @ e_s.T

    # --- Kernel RMSE ---
    kernel_mse = ((Kz - Ke) ** 2).mean()
    kernel_rmse = torch.sqrt(kernel_mse)

    # --- Normalized kernel RMSE ---
    ke_scale = torch.sqrt((Ke ** 2).mean())
    kernel_rmse_norm = kernel_rmse / (ke_scale + 1e-8)

    # --- Directional alignment ---
    z_normed = F.normalize(z_s, dim=1)
    e_normed = F.normalize(e_s, dim=1)
    cos_sim_matrix = (z_normed @ z_normed.T) * (e_normed @ e_normed.T)
    cos_sim_mean = cos_sim_matrix.mean()

    # --- Latent norm statistics ---
    z_norm = z_s.norm(dim=1)
    z_norm_mean = z_norm.mean()
    z_norm_std = z_norm.std()

    # --- Kernel magnitude diagnostics ---
    Kz_diag = torch.diag(Kz)
    Ke_diag = torch.diag(Ke)

    Kz_off = Kz[~torch.eye(Kz.size(0), dtype=bool, device=device)]
    Ke_off = Ke[~torch.eye(Ke.size(0), dtype=bool, device=device)]

    # --- Effective rank (geometry collapse check) ---
    # small batch SVD is cheap at max_pairs ~ 512
    svals = torch.linalg.svdvals(z_s)
    eff_rank = (svals / svals.sum()).pow(2).sum().reciprocal()

    return {
        # Core kernel metrics
        "kernel_rmse": kernel_rmse.item(),
        "kernel_rmse_norm": kernel_rmse_norm.item(),
        "cos_sim_mean": cos_sim_mean.item(),

        # Latent scale
        "z_norm_mean": z_norm_mean.item(),
        "z_norm_std": z_norm_std.item(),

        # Kernel magnitude
        "Kz_diag_mean": Kz_diag.mean().item(),
        "Ke_diag_mean": Ke_diag.mean().item(),
        "Kz_off_mean": Kz_off.mean().item(),
        "Ke_off_mean": Ke_off.mean().item(),

        # Geometry
        "latent_eff_rank": eff_rank.item(),
    }


# ============================================================
# Training
# ============================================================

def train_model(train, val, prism_dim, bui_dim, latent_dim=6,
                batch_size=16384, epochs=100, lr=1e-2):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultiInputAutoencoder(prism_dim, bui_dim, latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    loader = DataLoader(train, batch_size=batch_size, shuffle=True)

    for ep in range(1, epochs+1):
        model.train()
        for p, b, e in loader:
            p, b, e = p.to(device), b.to(device), e.to(device)
            opt.zero_grad()
            z, recon = model(p, b)
            loss = reconstruction_loss(recon, p, b) + metric_loss_preserve_kernel(z, e)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            # Move validation data to device
            prism_v = val.prism.to(device)
            bui_v   = val.bui.to(device)
            ebird_v = val.ebird.to(device)

            # Forward pass
            z, recon = model(prism_v, bui_v)

            # Core validation losses
            val_recon_loss  = reconstruction_loss(recon, prism_v, bui_v).item()
            val_metric_loss = metric_loss_preserve_kernel(z, ebird_v, num_pairs=8192, lambda_norm=0.0).item()

            # --- Kernel diagnostics ---
            diagnostics = compute_kernel_diagnostics(z, ebird_v, max_pairs=512)

        # Print everything in one clean line
        print(
            f"Epoch {ep}: "
            f"ValRecon={val_recon_loss:.6f}, "
            f"ValMetric={val_metric_loss:.6f}, "
            f"KernelRMSE={diagnostics['kernel_rmse']:.6f}, "
            f"KernelRMSENorm={diagnostics['kernel_rmse_norm']:.6f}, "
            f"CosSimMean={diagnostics['cos_sim_mean']:.6f}, "
            f"ZNormMean={diagnostics['z_norm_mean']:.4f}, "
            f"ZNormStd={diagnostics['z_norm_std']:.4f}, "
            f"KzDiagMean={diagnostics['Kz_diag_mean']:.4f}, "
            f"KeDiagMean={diagnostics['Ke_diag_mean']:.4f}, "
            f"KzOffMean={diagnostics['Kz_off_mean']:.4f}, "
            f"KeOffMean={diagnostics['Ke_off_mean']:.4f}, "
            f"LatentEffRank={diagnostics['latent_eff_rank']:.4f}"
        )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    prism = load_tifs("/home/breallis/datasets/prism_monthly_4km_albers", "prism_*_2023*.tif")
    bui   = load_tifs("/home/breallis/datasets/HBUI/BUI_4km_interp", "2020_BUI_4km_interp.tif")
    ebird = load_tifs("/home/breallis/datasets/ebird_weekly_2023_albers", "whcspa_abundance_median_2023-*.tif")

    ebird_mask, ebird = mask_ebird(ebird)

    # prism = add_spatial_context(prism)
    # bui   = add_spatial_context(bui)

    train = PixelDataset(prism, bui, ebird, split="train")
    val   = PixelDataset(prism, bui, ebird, split="val")

    train, val = normalize_dataset(train, val)

    train_model(
        train, val,
        prism_dim=train.prism.shape[1],
        bui_dim=train.bui.shape[1],
        latent_dim=6,
    )
