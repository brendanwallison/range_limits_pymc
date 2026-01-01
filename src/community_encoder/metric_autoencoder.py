import os
import glob
import numpy as np
import rasterio
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.stats import spearmanr


# ----------------------
# Data Handling Utils
# ----------------------

def load_tifs(folder, pattern):
    paths = sorted(glob.glob(os.path.join(folder, pattern)))
    arrays = []
    for p in paths:
        with rasterio.open(p) as src:
            data = src.read()  # shape: (bands, H, W)
            data = np.moveaxis(data, 0, -1)  # (H, W, bands)
            data = data.astype(np.float32)
            arrays.append(data)
    if len(arrays) == 0:
        raise ValueError(f"No files found in {folder} matching {pattern}")
    # concatenate along last axis to get flat channel/features axis
    data_all = np.concatenate(arrays, axis=-1)  # shape: (H, W, sum(bands over files))
    return data_all


def mask_prism(prism_array):
    valid = ~np.any(np.isnan(prism_array), axis=-1)
    return valid


def mask_bui(bui_array):
    valid = ~np.any(np.isnan(bui_array), axis=-1)
    return valid


def mask_ebird(ebird_array):
    valid = np.any(~np.isnan(ebird_array), axis=-1)
    ebird_array = np.nan_to_num(ebird_array, nan=0.0)
    return valid, ebird_array

# ----------------------
# Dataset with block splitting
# ----------------------

class PixelDataset(Dataset):
    def __init__(self, prism_array, bui_array, ebird_array, block_rows=8, block_cols=8, split='train'):
        H, W, _ = prism_array.shape
        self.blocks = []
        row_blocks = np.array_split(np.arange(H), block_rows)
        col_blocks = np.array_split(np.arange(W), block_cols)

        for r_block in row_blocks:
            for c_block in col_blocks:
                prism_block = prism_array[r_block[:, None], c_block, :]
                bui_block = bui_array[r_block[:, None], c_block, :]
                ebird_block = ebird_array[r_block[:, None], c_block, :]

                prism_flat = prism_block.reshape(-1, prism_array.shape[-1])
                bui_flat = bui_block.reshape(-1, bui_array.shape[-1])
                ebird_flat = ebird_block.reshape(-1, ebird_array.shape[-1])

                prism_mask = ~np.any(np.isnan(prism_flat), axis=1)
                bui_mask = ~np.any(np.isnan(bui_flat), axis=1)
                ebird_mask = np.any(ebird_flat != 0, axis=1)
                valid_mask = prism_mask & bui_mask & ebird_mask

                if np.sum(valid_mask) == 0:
                    continue

                self.blocks.append({
                    'prism': torch.tensor(prism_flat[valid_mask], dtype=torch.float32),
                    'bui': torch.tensor(bui_flat[valid_mask], dtype=torch.float32),
                    'ebird': torch.tensor(ebird_flat[valid_mask], dtype=torch.float32)
                })

        n_blocks = len(self.blocks)
        split_idx = int(0.8 * n_blocks)
        if split == 'train':
            self.blocks = self.blocks[:split_idx]
        else:
            self.blocks = self.blocks[split_idx:]

        self.data = {'prism': [], 'bui': [], 'ebird': []}
        for b in self.blocks:
            self.data['prism'].append(b['prism'])
            self.data['bui'].append(b['bui'])
            self.data['ebird'].append(b['ebird'])
        self.data['prism'] = torch.cat(self.data['prism'], dim=0)
        self.data['bui'] = torch.cat(self.data['bui'], dim=0)
        self.data['ebird'] = torch.cat(self.data['ebird'], dim=0)

    def __len__(self):
        return self.data['prism'].shape[0]

    def __getitem__(self, idx):
        return (self.data['prism'][idx], self.data['bui'][idx], self.data['ebird'][idx])

# ----------------------
# Dataset Normalization Module
# ----------------------

def normalize_dataset(train_dataset, val_dataset):
    # --- BUI: power transform then standardize ---
    bui_train = train_dataset.data['bui'] ** 0.1
    mean_bui = bui_train.mean(0, keepdims=True)
    std_bui = bui_train.std(0, keepdims=True)
    train_dataset.data['bui'] = (bui_train - mean_bui) / (std_bui + 1e-6)
    val_dataset.data['bui'] = ((val_dataset.data['bui'] ** 0.1) - mean_bui) / (std_bui + 1e-6)


    # --- eBird: sqrt transform then divide by number of species ---
    # ebird_train = train_dataset.data['ebird'] ** 0.5
    # n_species = ebird_train.shape[1]
    # train_dataset.data['ebird'] = ebird_train / n_species
    # val_dataset.data['ebird'] = (val_dataset.data['ebird'] ** 0.5) / n_species
    ebird_train = train_dataset.data['ebird'] ** 0.5
    scale = ebird_train.norm(dim=1).mean()
    train_dataset.data['ebird'] = ebird_train / scale
    val_dataset.data['ebird'] = (val_dataset.data['ebird'] ** 0.5) / scale

    # --- PRISM: standardize per band ---
    prism_train = train_dataset.data['prism']
    mean_prism = prism_train.mean(0, keepdims=True)
    std_prism = prism_train.std(0, keepdims=True)
    train_dataset.data['prism'] = (prism_train - mean_prism) / (std_prism + 1e-6)
    val_dataset.data['prism'] = (val_dataset.data['prism'] - mean_prism) / (std_prism + 1e-6)
    return train_dataset, val_dataset

# ----------------------
# B-MLP Block
# ----------------------

class BMLPBlock(nn.Module):
    def __init__(self, m, k=4, dropout=0.5):
        super().__init__()
        self.layernorm = nn.LayerNorm(m)
        self.expand = nn.Linear(m, m*k)
        self.collapse = nn.Linear(m*k, m)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        z = self.layernorm(x)
        z = self.expand(z)
        z = F.gelu(z)
        z = self.dropout(z)  # <-- dropout
        z = self.collapse(z)
        return x + z
    
def add_input_noise(prism, bui, sigma_prism=0.01, sigma_bui=0.01):
    prism_noisy = prism + sigma_prism * torch.randn_like(prism)
    bui_noisy = bui + sigma_bui * torch.randn_like(bui)
    return prism_noisy, bui_noisy


# ----------------------
# Autoencoder Model
# ----------------------

class MultiInputAutoencoder(nn.Module):
    def __init__(self, prism_dim, bui_dim, latent_dim=16):
        super().__init__()
        self.prism_encoder = nn.Sequential(
            nn.Linear(prism_dim, latent_dim*4),   # reduce to intermediate dim
            nn.GELU(),
            BMLPBlock(latent_dim*4),
            BMLPBlock(latent_dim*4),
            nn.GELU(),
        )

        # -------------------
        # BUI encoder
        # -------------------
        self.bui_encoder = nn.Sequential(
            nn.Linear(bui_dim, latent_dim*4),   # reduce to intermediate dim
            nn.GELU(),
            BMLPBlock(latent_dim*4),
            BMLPBlock(latent_dim*4),
            nn.GELU(),
        )
        
        # mix_dim = prism_dim + bui_dim
        mix_dim = latent_dim*8
        self.mixer = nn.Sequential(
            nn.Linear(mix_dim, mix_dim),
            nn.GELU(),
            nn.Linear(mix_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, mix_dim),
            nn.GELU(),
            nn.Linear(mix_dim, prism_dim + bui_dim)
        )

    def forward(self, prism, bui):
        h_prism = self.prism_encoder(prism)
        h_bui = self.bui_encoder(bui)
        h = torch.cat([h_prism, h_bui], dim=1)
        z = self.mixer(h)
        recon = self.decoder(z)
        return z, recon

# ----------------------
# Loss Functions
# ----------------------

def reconstruction_loss(recon, prism, bui):
    target = torch.cat([prism, bui], dim=1)
    return nn.MSELoss()(recon, target)


def metric_loss(z, ebird):
    dot_z = torch.sum(z[:-1] * z[1:], dim=1)
    dot_ebird = torch.sum(ebird[:-1] * ebird[1:], dim=1)
    return nn.MSELoss()(dot_z, dot_ebird)

# ----------------------
# Modified Metric Loss Preserving Latent Magnitude
# ----------------------

# def metric_loss_preserve_kernel(z: torch.Tensor, eBird: torch.Tensor, lambda_norm: float = 1e-3):
#     """
#     Computes MSE between latent dot products and eBird dot products (all pairs),
#     while regularizing latent norms to match the average eBird norm.
    
#     Args:
#         z: latent embeddings (B x latent_dim)
#         eBird: target abundance vectors (B x species_dim)
#         lambda_norm: weight for latent norm regularization
    
#     Returns:
#         loss: scalar, combined kernel MSE + latent norm regularization
#     """
#     # --- Dot-product kernel over all pairs ---
#     z_dot = z @ z.T          # (B x B)
#     e_dot = eBird @ eBird.T  # (B x B)

#     # Kernel MSE
#     kernel_mse = ((z_dot - e_dot) ** 2).mean()

#     # --- Latent norm regularization ---
#     target_norm = eBird.norm(dim=1).mean()
#     latent_norm = z.norm(dim=1).mean()
#     norm_loss = lambda_norm * (latent_norm - target_norm) ** 2

#     # Combined loss
#     loss = kernel_mse + norm_loss
#     return loss

def metric_loss_preserve_kernel(z: torch.Tensor, eBird: torch.Tensor, num_pairs: int = 16384, lambda_norm: float = 1e-3):
    """
    Vectorized MSE between latent and target dot products using random pairs,
    with latent norm regularization. Fully GPU-friendly.

    Args:
        z: latent embeddings (B x latent_dim)
        eBird: target abundance vectors (B x species_dim)
        num_pairs: number of random pairs to sample
        lambda_norm: weight for latent norm regularization

    Returns:
        loss: scalar, combined kernel MSE + latent norm regularization
    """
    B = z.shape[0]

    # Generate random pair indices
    idx = torch.randint(0, B, (2, num_pairs), device=z.device)
    i, j = idx[0], idx[1]

    # Compute dot products using gathered indices (vectorized)
    dot_z = torch.sum(z[i] * z[j], dim=1)
    dot_e = torch.sum(eBird[i] * eBird[j], dim=1)

    # Kernel MSE
    kernel_mse = torch.mean((dot_z - dot_e) ** 2)

    # Latent norm regularization
    target_norm = eBird.norm(dim=1).mean()
    latent_norm = z.norm(dim=1).mean()
    norm_loss = lambda_norm * (latent_norm - target_norm) ** 2

    return kernel_mse + norm_loss


# ----------------------
# Training Loop
# ----------------------

# ----------------------
# Interpret Metric Loss and Cosine Similarity
# ----------------------

def interpret_metric_loss(z, ebird, sample_size=1000):
    N = z.shape[0]
    if N > sample_size:
        idx = torch.randperm(N)[:sample_size]
        z_sample = z[idx]
        ebird_sample = ebird[idx]
    else:
        z_sample = z
        ebird_sample = ebird

    dot_z = torch.sum(z_sample[:-1] * z_sample[1:], dim=1)
    dot_ebird = torch.sum(ebird_sample[:-1] * ebird_sample[1:], dim=1)
    mse = torch.mean((dot_z - dot_ebird) ** 2).item()
    avg_dot_z = dot_z.mean().item()
    avg_dot_ebird = dot_ebird.mean().item()

    # Cosine similarity
    z_norm = z_sample / (z_sample.norm(dim=1, keepdim=True) + 1e-8)
    ebird_norm = ebird_sample / (ebird_sample.norm(dim=1, keepdim=True) + 1e-8)
    cos_z = torch.sum(z_norm[:-1] * z_norm[1:], dim=1).mean().item()
    cos_ebird = torch.sum(ebird_norm[:-1] * ebird_norm[1:], dim=1).mean().item()

    return avg_dot_z, avg_dot_ebird, mse, cos_z, cos_ebird


def compute_composite_metric(z: torch.Tensor, eBird: torch.Tensor, alpha: float = 0.5, beta: float = 0.5):
    """
    Computes a composite validation metric preserving the dot-product kernel,
    using RMSE for interpretability, and also logs directional alignment.

    Args:
        z (torch.Tensor): Latent embeddings (B x latent_dim)
        eBird (torch.Tensor): Target abundance vectors (B x species_dim)
        alpha (float): Weight for kernel RMSE component
        beta (float): Weight for directional alignment (cosine similarity)
    
    Returns:
        composite: float, weighted combination for logging (not used for training)
        kernel_rmse_norm: float, normalized kernel RMSE
        cos_sim_mean: float, mean pairwise cosine similarity of directions
    """
    B = z.shape[0]

    # --- Raw dot-product kernels ---
    z_dot = z @ z.T              # (B x B)
    e_dot = eBird @ eBird.T      # (B x B)

    # Kernel RMSE over all pairs
    kernel_mse = ((z_dot - e_dot) ** 2).mean()
    kernel_rmse = torch.sqrt(kernel_mse)

    # Normalized RMSE for logging
    e_dot_rmse = torch.sqrt((e_dot ** 2).mean())
    kernel_rmse_norm = kernel_rmse / e_dot_rmse  # dimensionless, interpretable

    # --- Directional alignment (cosine similarity) ---
    z_norm = F.normalize(z, p=2, dim=1)
    e_norm = F.normalize(eBird, p=2, dim=1)
    cos_sim_matrix = (z_norm @ z_norm.T) * (e_norm @ e_norm.T)
    cos_sim_mean = cos_sim_matrix.mean()

    # cos_sim_mean = F.cosine_similarity(z, eBird, dim=1).mean()

    # --- Composite metric for logging ---
    composite = alpha * kernel_rmse_norm + beta * (1 - cos_sim_mean)

    return composite.item(), kernel_rmse_norm.item(), cos_sim_mean.item()

# ----------------------
# Updated Training Loop with Interpretable Metric
# ----------------------

# ----------------------
# Updated Training Loop with Training and Validation Losses + Interpretable Metrics
# ----------------------

def train_model(train_dataset, val_dataset, prism_dim, bui_dim, latent_dim=16, batch_size=16384, epochs=50, lr=1e-2):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MultiInputAutoencoder(prism_dim, bui_dim, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(1, epochs+1):
        model.train()
        train_recon_loss = 0.0
        train_metric_loss = 0.0
        n_batches = 0

        for prism, bui, ebird in train_loader:
            prism = prism.to(device)
            bui = bui.to(device)
            ebird = ebird.to(device)

            # Optional input augmentation
            prism, bui = add_input_noise(prism, bui)

            optimizer.zero_grad()
            z, recon = model(prism, bui)
            loss_recon = reconstruction_loss(recon, prism, bui)
            loss_metric = metric_loss_preserve_kernel(z, ebird)
            loss = loss_recon + loss_metric
            loss.backward()
            optimizer.step()

            train_recon_loss += loss_recon.item()
            train_metric_loss += loss_metric.item()
            n_batches += 1

        train_recon_loss /= n_batches
        train_metric_loss /= n_batches

        # ----------------------
        # Validation evaluation
        # ----------------------
        model.eval()
        with torch.no_grad():
            val_prism = val_dataset.data['prism'].to(device)
            val_bui = val_dataset.data['bui'].to(device)
            val_ebird = val_dataset.data['ebird'].to(device)

            z_val, recon_val = model(val_prism, val_bui)
            recon_loss_val = reconstruction_loss(recon_val, val_prism, val_bui).item()
            metric_loss_val = metric_loss_preserve_kernel(z_val, val_ebird).item()

            # Compute combined latent alignment metrics
            composite, kernel_rmse, cos_sim_mean = compute_composite_metric(z_val, val_ebird)


        print(f"Epoch {epoch}: Train Recon={train_recon_loss:.6f}, Train Metric={train_metric_loss:.6f}, "
            f"Val Recon={recon_loss_val:.6f}, Val Metric={metric_loss_val:.6f}, "
            f"KernelRMSE={kernel_rmse:.6f}, CosSimMean={cos_sim_mean:.6f}, "
            f"CompositeMetric={composite:.6f}")


# ----------------------
# Example Usage
# ----------------------

if __name__ == '__main__':
    prism_array = load_tifs('/home/breallis/datasets/prism_monthly_4km_albers', 'prism_*_2023*.tif')
    bui_array = load_tifs('/home/breallis/datasets/HBUI/BUI_4km_interp', '2020_BUI_4km_interp.tif')
    ebird_array = load_tifs('/home/breallis/datasets/ebird_weekly_2023_albers', 'whcspa_abundance_median_2023-*.tif')

    prism_mask = mask_prism(prism_array)
    bui_mask = mask_bui(bui_array)
    ebird_mask, ebird_array = mask_ebird(ebird_array)

    valid_mask = prism_mask & bui_mask & ebird_mask

    train_dataset = PixelDataset(prism_array, bui_array, ebird_array, block_rows=8, block_cols=8, split='train')
    val_dataset = PixelDataset(prism_array, bui_array, ebird_array, block_rows=8, block_cols=8, split='val')

    # --- Apply normalization ---
    train_dataset, val_dataset = normalize_dataset(train_dataset, val_dataset)

    train_model(train_dataset, val_dataset, prism_dim=prism_array.shape[-1], bui_dim=bui_array.shape[-1], latent_dim=6)