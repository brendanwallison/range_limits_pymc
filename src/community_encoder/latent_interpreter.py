import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio
from datetime import datetime
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import cdist

# ============================================================
# 1. CORE LOGIC (Data Loading & Transform)
# ============================================================
def load_tifs_structured(folder, pattern="*_abundance_median_*.tif"):
    """
    Loads all TIFs to reconstruct the exact 3D stack used for training.
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
    
    n_species = df['species'].nunique()
    n_weeks = df['date'].nunique()
    df_sorted = df.sort_values(by=['species', 'date'])
    ordered_paths = df_sorted['path'].tolist()
    
    species_list = sorted(df['species'].unique().tolist())

    with rasterio.open(ordered_paths[0]) as src:
        H, W = src.shape

    print(f"Loading {len(ordered_paths)} files into ({H}, {W}) stack...")
    full_stack = np.zeros((H, W, len(ordered_paths)), dtype=np.float32)
    for i, p in enumerate(ordered_paths):
        with rasterio.open(p) as src:
            full_stack[:, :, i] = src.read(1)

    return full_stack, {"n_species": n_species, "n_weeks": n_weeks, "species_list": species_list}

def load_or_recover_mask(z_dir, ebird_dir, ebird_pattern="*_abundance_median_*.tif"):
    """
    Robustly loads the mask. Checks for 'valid_mask.npy' first.
    """
    mask_path = os.path.join(z_dir, "valid_mask.npy")
    
    if os.path.exists(mask_path):
        print(f"Loading existing mask from: {mask_path}")
        return np.load(mask_path)
    
    print(f"Mask not found at {mask_path}. Recovering from TIF stack...")
    # We recover it by scanning files if necessary
    files = sorted(glob.glob(os.path.join(ebird_dir, ebird_pattern)))
    if not files:
        raise ValueError("No eBird files found to recover mask.")
        
    with rasterio.open(files[0]) as src:
        H, W = src.shape
        
    union_mask = np.zeros((H, W), dtype=bool)
    for i, p in enumerate(files):
        with rasterio.open(p) as src:
            data = src.read(1)
            union_mask |= ~np.isnan(data)
            
    return union_mask

def smoothed_hellinger_transform(ebird_flat, n_weeks, sigma):
    """
    Applies the exact same transformation used in training.
    """
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
# 2. THE LATENT AUDITOR (Biological Interpretation)
# ============================================================
class LatentAuditor:
    def __init__(self, Z, ebird_smooth, meta, valid_mask, H, W, out_dir):
        self.Z = Z
        self.ebird = ebird_smooth
        self.species_names = meta['species_list']
        self.n_sp = meta['n_species']
        self.n_w = meta['n_weeks']
        self.H, self.W = H, W
        self.mask = valid_mask
        self.K = Z.shape[1]
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        
        if Z.shape[0] != ebird_smooth.shape[0]:
            raise ValueError(f"CRITICAL MISMATCH: Z has {Z.shape[0]} pixels, but recreated mask has {ebird_smooth.shape[0]}.")

        print(f"Auditing {self.K} Latent Dimensions against {self.n_sp} Species x {self.n_w} Weeks...")
        
        # Subsample for correlation speed
        idx = np.random.choice(Z.shape[0], min(Z.shape[0], 25000), replace=False)
        zs = (Z[idx] - Z[idx].mean(0)) / (Z[idx].std(0) + 1e-8)
        es = (ebird_smooth[idx] - ebird_smooth[idx].mean(0)) / (ebird_smooth[idx].std(0) + 1e-8)
        
        self.loadings = (zs.T @ es) / len(idx)
        self.loadings_3d = self.loadings.reshape(self.K, self.n_sp, self.n_w)

    def audit_phenology(self):
        print(" -> Generating Phenology Heatmap...")
        temporal_pulse = np.abs(self.loadings_3d).mean(axis=1)
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(temporal_pulse, cmap="magma", cbar_kws={'label': 'Mean Correlation Magnitude'})
        plt.title("Phenological Pulse: Seasonal Activity of Latent Dimensions")
        plt.xlabel("Week of Year")
        plt.ylabel("Latent Dimension (Z)")
        plt.savefig(os.path.join(self.out_dir, "01_phenology_pulse.png"), bbox_inches='tight')
        plt.close()

    def audit_taxonomy(self):
        print(" -> Generating Taxonomic Heatmap...")
        taxa_importance = np.abs(self.loadings_3d).mean(axis=2)
        df = pd.DataFrame(taxa_importance.T, columns=[f"Z{i}" for i in range(self.K)], index=self.species_names)
        
        plt.figure(figsize=(16, 12))
        sns.heatmap(df, cmap="viridis", cbar_kws={'label': 'Mean Correlation Magnitude'})
        plt.title("Taxonomic Drivers: Species Importance per Latent Dimension")
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, "02_taxonomic_weights.png"), bbox_inches='tight')
        plt.close()

    def audit_spatial(self):
        print(" -> Generating Spatial Maps (Top Dimensions)...")
        n_map = min(self.K, 12)
        rows = 4
        cols = 3
        fig, axes = plt.subplots(rows, cols, figsize=(18, 20))
        axes = axes.flatten()
        
        for k in range(n_map):
            full_map = np.full((self.H, self.W), np.nan)
            full_map[self.mask] = self.Z[:, k]
            
            im = axes[k].imshow(full_map, cmap='Spectral_r')
            axes[k].set_title(f"Z{k} Spatial Distribution")
            axes[k].axis('off')
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, "03_spatial_distribution.png"), bbox_inches='tight')
        plt.close()

    def run_all(self):
        self.audit_phenology()
        self.audit_taxonomy()
        self.audit_spatial()
        
        summary = []
        for k in range(self.K):
            sp_scores = np.abs(self.loadings_3d[k]).mean(axis=1)
            top_sp_idx = np.argmax(sp_scores)
            wk_scores = np.abs(self.loadings_3d[k]).mean(axis=0)
            top_wk_idx = np.argmax(wk_scores)
            
            summary.append({
                "Latent_Dim": k,
                "Top_Species_Driver": self.species_names[top_sp_idx],
                "Peak_Activity_Week": top_wk_idx,
                "Max_Correlation": sp_scores[top_sp_idx]
            })
            
        pd.DataFrame(summary).to_csv(os.path.join(self.out_dir, "latent_interpretation_summary.csv"), index=False)
        print(f"Latent Audit Complete. Results saved to: {self.out_dir}")

# ============================================================
# 3. GEOGRAPHIC COVARIANCE PROBE (Spatial Structure)
# ============================================================
class GeoCovarianceProbe:
    def __init__(self, Z, valid_mask, H, W, out_dir):
        self.Z = Z
        self.mask = valid_mask
        self.H, self.W = H, W
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        
        y_grid, x_grid = np.indices((H, W))
        self.coords = np.stack([y_grid[valid_mask], x_grid[valid_mask]], axis=1) # (N, 2)
        self.Z_full = np.full((H, W, Z.shape[1]), np.nan)
        self.Z_full[valid_mask] = Z

        # FIX: Identify valid rows (Finite Z)
        # This handles cases where PRISM/BUI was NaN even inside the mask
        self.valid_indices = np.where(np.all(np.isfinite(Z), axis=1))[0]
        print(f"Probe Init: {len(self.valid_indices)} finite Z-vectors out of {Z.shape[0]} total masked pixels.")

    def plot_iso_similarity(self, focal_points_idx=None):
        print(" -> Generating Iso-Similarity Maps...")
        if len(self.valid_indices) == 0:
            print("WARNING: No valid Z vectors found. Skipping plot.")
            return

        N = self.Z.shape[0]
        if focal_points_idx is None:
            # Pick 4 random distinct points from VALID INDICES
            focal_points_idx = np.random.choice(self.valid_indices, 4, replace=False)
            
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        axes = axes.flatten()
        
        for i, idx in enumerate(focal_points_idx):
            z_ref = self.Z[idx]
            sims = self.Z @ z_ref
            
            sim_map = np.full((self.H, self.W), np.nan)
            sim_map[self.mask] = sims
            
            ax = axes[i]
            im = ax.imshow(sim_map, cmap='RdYlBu_r', vmin=0, vmax=1)
            
            fy, fx = self.coords[idx]
            ax.scatter(fx, fy, c='black', marker='*', s=200, edgecolors='white', label='Reference')
            ax.set_title(f"Community Similarity to Point ({fy}, {fx})")
            ax.axis('off')
            
        plt.tight_layout()
        plt.colorbar(im, ax=axes, fraction=0.046, pad=0.04, label="Kernel Similarity")
        plt.savefig(os.path.join(self.out_dir, "geo_01_iso_similarity.png"), bbox_inches='tight')
        plt.close()

    def plot_variogram(self, max_dist=200, n_samples=5000):
        print(" -> Generating Community Variogram...")
        if len(self.valid_indices) < 2:
            print("WARNING: Not enough valid Z vectors for variogram. Skipping.")
            return

        # Sample only from valid indices
        idx_a = np.random.choice(self.valid_indices, n_samples)
        idx_b = np.random.choice(self.valid_indices, n_samples)
        
        d_phys = np.linalg.norm(self.coords[idx_a] - self.coords[idx_b], axis=1)
        d_eco = 1.0 - np.sum(self.Z[idx_a] * self.Z[idx_b], axis=1)
        
        mask = d_phys < max_dist
        d_phys = d_phys[mask]
        d_eco = d_eco[mask]
        
        plt.figure(figsize=(10, 6))
        plt.hexbin(d_phys, d_eco, gridsize=50, cmap='inferno', mincnt=1)
        try:
            sns.regplot(x=d_phys, y=d_eco, scatter=False, color='cyan', line_kws={'linestyle':'--'})
        except:
            pass

        plt.title("The 'Speed' of Ecology: Community Turnover vs. Distance")
        plt.xlabel("Physical Distance (pixels)")
        plt.ylabel("Ecological Distance (1 - CosSim)")
        plt.savefig(os.path.join(self.out_dir, "geo_02_variogram.png"), bbox_inches='tight')
        plt.close()

    def plot_local_volatility(self):
        print(" -> Generating Ecological Volatility Map...")
        volatility_map = np.zeros((self.H, self.W))
        K = self.Z.shape[1]
        
        for k in range(K):
            layer = self.Z_full[:, :, k]
            grads = np.gradient(layer) 
            mag_sq = grads[0]**2 + grads[1]**2
            volatility_map += np.nan_to_num(mag_sq)
            
        volatility_map = np.sqrt(volatility_map / K)
        volatility_map[~self.mask] = np.nan
        
        plt.figure(figsize=(12, 10))
        im = plt.imshow(volatility_map, cmap='magma', vmax=np.nanpercentile(volatility_map, 98))
        plt.colorbar(im, label="Local Rate of Change (Gradient)")
        plt.title("Ecological Volatility: Where do communities change fastest?")
        plt.axis('off')
        plt.savefig(os.path.join(self.out_dir, "geo_03_volatility.png"), bbox_inches='tight')
        plt.close()

    def run_all(self):
        self.plot_iso_similarity()
        self.plot_variogram()
        self.plot_local_volatility()
        print(f"GeoCovariance Audit Complete. Results saved to: {self.out_dir}")

# ============================================================
# 4. EXECUTION BLOCK
# ============================================================
if __name__ == "__main__":
    # --- CONFIGURATION ---
    DATA_DIR = "/home/breallis/datasets/ebird_weekly_2023_albers"
    Z_DIR = "/home/breallis/dev/range_limits_pymc/misc_outputs/env_model_results"
    # Or wherever your Z and mask are located
    Z_PATH = os.path.join(Z_DIR, "Z_pred_env.npy")
    OUT_DIR = "/home/breallis/dev/range_limits_pymc/misc_outputs/env_model_results"
    SIGMA = 0.5

    # 1. Load Z
    if not os.path.exists(Z_PATH):
        raise FileNotFoundError(f"Z matrix not found at {Z_PATH}")
    Z = np.load(Z_PATH)
    print(f"Loaded Z matrix with shape: {Z.shape}")

    # 2. Load Mask (Use saved if available)
    valid_mask = load_or_recover_mask(Z_DIR, DATA_DIR, "*_abundance_median_2023-*.tif")
    
    # 3. Check Alignment
    valid_flat = valid_mask.flatten()
    N_pixels = np.sum(valid_flat)
    
    if N_pixels != Z.shape[0]:
        print(f"WARNING: Pixel mismatch (Mask: {N_pixels} vs Z: {Z.shape[0]}).")
        if abs(N_pixels - Z.shape[0]) > 0:
             raise ValueError("Pixel count mismatch is too large. Cannot align Z to Map.")

    # 4. Load Stack (Required for Latent Auditor Correlations)
    # Even though we have the mask, Auditor needs 'ebird_smooth' features
    print("Loading eBird stack for correlation analysis...")
    ebird_stack, meta = load_tifs_structured(DATA_DIR, "*_abundance_median_2023-*.tif")
    H, W, D = ebird_stack.shape

    print("Creating smoothed feature matrix for Biological Audit...")
    # Note: We must use the mask we just loaded to flatten the stack
    ebird_flat_raw = np.nan_to_num(ebird_stack).reshape(-1, D)[valid_flat]
    ebird_smooth = smoothed_hellinger_transform(ebird_flat_raw, meta['n_weeks'], sigma=SIGMA)
    
    # 5. Run Latent Auditor (Biology)
    latent_auditor = LatentAuditor(Z, ebird_smooth, meta, valid_mask, H, W, OUT_DIR)
    latent_auditor.run_all()

    # 6. Run Geographic Probe (Spatial)
    print("\n--- Starting Geographic Covariance Probe ---")
    geo_probe = GeoCovarianceProbe(Z, valid_mask, H, W, OUT_DIR)
    geo_probe.run_all()