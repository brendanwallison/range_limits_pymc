#!/usr/bin/env python3
"""
Bayesian linear regression on latent Z
Predicting House Finch mean relative abundance

Outputs:
- posterior_mean.png
- posterior_sd.png
- residuals.png
- posterior_mean_log1p.png
- observed_log1p.png
"""

import numpy as np
import rasterio
import matplotlib.pyplot as plt

# ============================================================
# 1. Paths
# ============================================================

import os

# ============================================================
# 1. Directories
# ============================================================

result_dir = "/home/breallis/dev/range_limits_pymc/misc_outputs/ruzicka_sweep_global/sigma_0.5"
out_dir = "/home/breallis/dev/range_limits_pymc/misc_outputs/ruzicka_sweep_global/sigma_0.5/blr"

os.makedirs(out_dir, exist_ok=True)

# ============================================================
# 2. Input paths
# ============================================================

Z_PATH = os.path.join(result_dir, "Z.npy")                   # (N_valid, D)
MASK_PATH = os.path.join(result_dir, "valid_mask.npy")       # (H, W) boolean
EBIRD_TIF = os.path.join(result_dir, "houfin_abundance_seasonal_year_round_mean_2023_bui4km.tif") # (H, W)

# ============================================================
# 3. Output paths
# ============================================================

OUT_MEAN = os.path.join(out_dir, "posterior_mean.png")
OUT_SD = os.path.join(out_dir, "posterior_sd.png")
OUT_RESID = os.path.join(out_dir, "residuals.png")
OUT_MEAN_LOG = os.path.join(out_dir, "posterior_mean_log1p.png")
OUT_OBS_LOG = os.path.join(out_dir, "observed_log1p.png")


# ============================================================
# 2. Load data
# ============================================================

Z = np.load(Z_PATH)                          # (N_comm, D)
valid_mask = np.load(MASK_PATH).astype(bool)

with rasterio.open(EBIRD_TIF) as src:
    y_map = src.read(1).astype(np.float32)

H, W = y_map.shape
N_comm, D = Z.shape

# ============================================================
# 3. Extract observed y at community-valid pixels
#    and intersect with observation validity
# ============================================================

# Flatten spatial layers
mask_flat = valid_mask.ravel()
y_flat = y_map.ravel()

# Community-valid pixels
y_comm = y_flat[mask_flat]                  # (N_comm,)

# Observation validity (House Finch)
obs_ok = np.isfinite(y_comm)

n_dropped = np.sum(~obs_ok)
if n_dropped > 0:
    print(f"Dropping {n_dropped} / {N_comm} pixels with non-finite observations")

# Final aligned data
y = y_comm[obs_ok]                          # (N_final,)
Z = Z[obs_ok]                               # (N_final, D)

# Store spatial indices for reconstruction
valid_idx = np.where(valid_mask)
valid_idx = (valid_idx[0][obs_ok], valid_idx[1][obs_ok])

N_final = y.shape[0]

# ============================================================
# 4. Validity checks (strict but correct)
# ============================================================

assert Z.shape[0] == y.shape[0], \
    f"Z rows ({Z.shape[0]}) != y rows ({y.shape[0]})"

assert np.isfinite(Z).all(), "Z contains non-finite values"
assert np.isfinite(y).all(), "y contains non-finite values"

print(f"Loaded {N_final} aligned pixels (from {N_comm}), latent dim = {D}")


# ============================================================
# 5. Bayesian linear regression (closed form)
# ============================================================

# Hyperparameters
alpha = 1.0      # prior precision on weights
sigma2 = np.var(y) * 0.1 + 1e-6  # noise variance (reasonable default)

# Posterior covariance
A = alpha * np.eye(D) + (Z.T @ Z) / sigma2
A_inv = np.linalg.inv(A)

# Posterior mean
w_mean = A_inv @ (Z.T @ y) / sigma2

# ============================================================
# 6. Posterior predictive moments
# ============================================================

# Predictive mean
y_pred_mean = Z @ w_mean

# Predictive variance (diagonal only)
# Var[y*] = z^T Cov[w] z + sigma^2
quad = np.sum(Z @ A_inv * Z, axis=1)
y_pred_var = quad + sigma2
y_pred_sd = np.sqrt(y_pred_var)

# Residuals
residuals = y - y_pred_mean

# ============================================================
# 7. Reconstruct full rasters
# ============================================================

def reconstruct(values, fill=np.nan):
    """
    Reconstruct a spatial map from values aligned to valid_idx.
    """
    full = np.full((H, W), fill, dtype=np.float32)
    full[valid_idx] = values
    return full


mean_map = reconstruct(y_pred_mean)
sd_map = reconstruct(y_pred_sd)
resid_map = reconstruct(residuals)

# ============================================================
# 8. Plot helper
# ============================================================

def save_png(data, fname, title, cmap="viridis", vmin=None, vmax=None):
    plt.figure(figsize=(8, 6))
    im = plt.imshow(data, cmap=cmap, origin="upper", vmin=vmin, vmax=vmax)
    plt.colorbar(im)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()
    print(f"Saved {fname}")

# ============================================================
# 9. Save PNGs
# ============================================================

save_png(mean_map, OUT_MEAN,
         "Posterior Mean – House Finch Abundance")

save_png(sd_map, OUT_SD,
         "Posterior SD – House Finch Abundance")

save_png(resid_map, OUT_RESID,
         "Residuals (Observed − Predicted)",
         cmap="RdBu_r")

# log1p maps
save_png(np.log1p(mean_map), OUT_MEAN_LOG,
         "Posterior Mean (log1p scale)")

save_png(np.log1p(y_map), OUT_OBS_LOG,
         "Observed Abundance (log1p scale)")

# ============================================================
# 10. Final diagnostics
# ============================================================

rmse = np.sqrt(np.mean(residuals ** 2))
r2 = 1.0 - np.var(residuals) / np.var(y)

print(f"RMSE: {rmse:.6f}")
print(f"R^2:  {r2:.4f}")
