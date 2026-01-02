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

# ============================================================
# 1. DATA LOADING
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

def load_mask_for_z(z_dir, ebird_dir, ebird_pattern="*_abundance_median_*.tif"):
    """
    Loads the mask corresponding to Z.
    Prioritizes the saved mask file to ensure correct spatial alignment.
    """
    candidates = ["valid_mask_env.npy", "valid_mask.npy"]
    for fname in candidates:
        p = os.path.join(z_dir, fname)
        if os.path.exists(p):
            print(f"Loading alignment mask from: {p}")
            return np.load(p)
            
    print(f"No saved mask found in {z_dir}. Recovering from TIFs...")
    files = sorted(glob.glob(os.path.join(ebird_dir, ebird_pattern)))
    if not files:
        raise ValueError("No eBird files found to recover mask.")
    with rasterio.open(files[0]) as src:
        H, W = src.shape
    union_mask = np.zeros((H, W), dtype=bool)
    for i, p in enumerate(files):
        with rasterio.open(p) as src:
            union_mask |= ~np.isnan(src.read(1))
    return union_mask

def smoothed_hellinger_transform(ebird_flat, n_weeks, sigma):
    N, D = ebird_flat.shape
    n_species = D // n_weeks
    data_3d = ebird_flat.reshape(N, n_species, n_weeks)
    
    # Fill NaNs with 0 before smoothing
    data_3d = np.nan_to_num(data_3d, nan=0.0)
    
    if sigma > 1e-5:
        data_smoothed = gaussian_filter1d(data_3d, sigma=sigma, axis=-1, mode='wrap')
    else:
        data_smoothed = data_3d
        
    data_flat = data_smoothed.reshape(N, -1)
    row_sums = data_flat.sum(axis=1, keepdims=True)
    row_sums[row_sums < 1e-9] = 1.0 
    return np.sqrt(data_flat / row_sums)

# ============================================================
# 2. LATENT AUDITOR (With Detailed CSV Summary)
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
        
        # 1. Alignment Check
        if Z.shape[0] != ebird_smooth.shape[0]:
            raise ValueError(f"ALIGNMENT ERROR: Z has {Z.shape[0]} rows, but eBird has {ebird_smooth.shape[0]}.")

        # 2. Robust Intersection Filter
        valid_z = np.all(np.isfinite(Z), axis=1)
        e_std = ebird_smooth.std(axis=1)
        valid_e = (e_std > 1e-9) & np.all(np.isfinite(ebird_smooth), axis=1)
        
        self.valid_indices = np.where(valid_z & valid_e)[0]
        
        print(f"Auditing {self.K} Latent Dimensions.")
        print(f" -> Valid for Correlation: {len(self.valid_indices)}")

        if len(self.valid_indices) == 0:
            raise ValueError("FATAL: No valid pixels found.")
        
        # 3. Subsample
        sample_size = min(len(self.valid_indices), 25000)
        idx = np.random.choice(self.valid_indices, sample_size, replace=False)
        
        z_sample = Z[idx]
        e_sample = ebird_smooth[idx]
        
        print(" -> Computing correlations...")
        # Safe Standardization
        zs = (z_sample - z_sample.mean(0)) / (z_sample.std(0) + 1e-9)
        es = (e_sample - e_sample.mean(0)) / (e_sample.std(0) + 1e-9)
        
        # Correlation
        self.loadings = (zs.T @ es) / len(idx)
        self.loadings = np.nan_to_num(self.loadings)
        self.loadings_3d = self.loadings.reshape(self.K, self.n_sp, self.n_w)

    def audit_phenology(self):
        print(" -> Generating Phenology Heatmap...")
        temporal_pulse = np.abs(self.loadings_3d).mean(axis=1)
        vmax = np.percentile(temporal_pulse, 99) if temporal_pulse.max() > 0 else 1.0
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(temporal_pulse, cmap="magma", vmax=vmax, cbar_kws={'label': 'Mean Correlation'})
        plt.title("Phenological Pulse: Seasonal Activity")
        plt.xlabel("Week of Year")
        plt.ylabel("Latent Dimension")
        plt.savefig(os.path.join(self.out_dir, "01_phenology_pulse.png"), bbox_inches='tight')
        plt.close()

    def audit_taxonomy(self):
        print(" -> Generating Taxonomic Heatmap...")
        taxa_importance = np.abs(self.loadings_3d).mean(axis=2)
        vmax = np.percentile(taxa_importance, 99) if taxa_importance.max() > 0 else 1.0
        
        df = pd.DataFrame(taxa_importance.T, columns=[f"Z{i}" for i in range(self.K)], index=self.species_names)
        
        plt.figure(figsize=(16, 12))
        sns.heatmap(df, cmap="viridis", vmax=vmax, cbar_kws={'label': 'Mean Correlation'})
        plt.title("Taxonomic Drivers: Species Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, "02_taxonomic_weights.png"), bbox_inches='tight')
        plt.close()

    def audit_guild_structure(self):
        print(" -> Generating Guild Structure...")
        habitat_affinity = self.loadings_3d.mean(axis=2).T 
        species_corr = np.corrcoef(habitat_affinity)
        species_corr = np.nan_to_num(species_corr)
        
        try:
            g = sns.clustermap(
                species_corr, 
                xticklabels=self.species_names, 
                yticklabels=self.species_names,
                cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                figsize=(20, 20),
                dendrogram_ratio=0.15
            )
            g.fig.suptitle("Community Guilds: Species-Species Latent Correlation", y=1.02)
            plt.savefig(os.path.join(self.out_dir, "04_guild_structure.png"), bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"WARNING: Clustering failed. {e}")

    def audit_ecological_seasons(self):
        print(" -> Generating Ecological Seasons...")
        temporal_state = self.loadings_3d.mean(axis=1).T
        week_corr = np.corrcoef(temporal_state)
        week_corr = np.nan_to_num(week_corr)
        
        try:
            plt.figure(figsize=(10, 8))
            sns.heatmap(week_corr, cmap="viridis", cbar_kws={'label': 'Correlation'})
            plt.title("Ecological Seasons: Week-to-Week Similarity")
            plt.xlabel("Week")
            plt.ylabel("Week")
            plt.savefig(os.path.join(self.out_dir, "05_ecological_seasons.png"), bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"WARNING: Time plot failed. {e}")

    def audit_spatial(self):
        print(" -> Generating Spatial Maps...")
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
        self.audit_guild_structure()
        self.audit_ecological_seasons()
        self.audit_spatial()
        
        # --- NEW SUMMARY GENERATION ---
        print(" -> Generating Detailed CSV Summary...")
        summary = []
        for k in range(self.K):
            # 1. Get Loading Matrix for this Dimension (Species x Weeks)
            layer = self.loadings_3d[k]
            
            # 2. Species Analysis (Averaged over time)
            sp_scores_signed = layer.mean(axis=1) # (n_species,)
            
            # Sort by signed value
            sorted_indices = np.argsort(sp_scores_signed)
            
            # Avoiders (Most Negative) -> First 3
            avoiders = sorted_indices[:3]
            # Drivers (Most Positive) -> Last 3 (reversed)
            drivers = sorted_indices[-3:][::-1]
            
            # 3. Temporal Analysis (Average Absolute Magnitude over species)
            # This detects "activity" regardless of sign
            wk_scores_abs = np.abs(layer).mean(axis=0) # (n_weeks,)
            
            # Get Top 5 Weeks
            top_weeks = np.argsort(wk_scores_abs)[-5:][::-1] # Descending order
            
            # Build Row
            row = {"Latent_Dim": k}
            
            # Add Drivers
            for i, idx in enumerate(drivers):
                row[f"Driver_{i+1}_Name"] = self.species_names[idx]
                row[f"Driver_{i+1}_Corr"] = f"{sp_scores_signed[idx]:.3f}"
                
            # Add Avoiders
            for i, idx in enumerate(avoiders):
                row[f"Avoider_{i+1}_Name"] = self.species_names[idx]
                row[f"Avoider_{i+1}_Corr"] = f"{sp_scores_signed[idx]:.3f}"
                
            # Add Peak Weeks (Joined string for readability)
            row["Top_5_Activity_Weeks"] = ", ".join(map(str, top_weeks))
            
            summary.append(row)
            
        # Save
        df_summary = pd.DataFrame(summary)
        out_path = os.path.join(self.out_dir, "latent_interpretation_detailed.csv")
        df_summary.to_csv(out_path, index=False)
        print(f"Detailed Audit Complete. Results saved to: {out_path}")

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

        self.valid_indices = np.where(np.all(np.isfinite(Z), axis=1))[0]
        print(f"Probe Init: {len(self.valid_indices)} finite Z-vectors.")

    def plot_iso_similarity(self, focal_points_idx=None):
        print(" -> Generating Iso-Similarity Maps...")
        if len(self.valid_indices) == 0: return

        if focal_points_idx is None:
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
            ax.scatter(fx, fy, c='black', marker='*', s=200, edgecolors='white')
            ax.set_title(f"Similarity to ({fy}, {fx})")
            ax.axis('off')
            
        plt.tight_layout()
        plt.colorbar(im, ax=axes, fraction=0.046, pad=0.04, label="Kernel Similarity")
        plt.savefig(os.path.join(self.out_dir, "geo_01_iso_similarity.png"), bbox_inches='tight')
        plt.close()

    def plot_variogram(self, max_dist=200, n_samples=5000):
        print(" -> Generating Community Variogram...")
        if len(self.valid_indices) < 2: return

        idx_a = np.random.choice(self.valid_indices, n_samples)
        idx_b = np.random.choice(self.valid_indices, n_samples)
        
        d_phys = np.linalg.norm(self.coords[idx_a] - self.coords[idx_b], axis=1)
        d_eco = 1.0 - np.sum(self.Z[idx_a] * self.Z[idx_b], axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.hexbin(d_phys, d_eco, gridsize=50, cmap='inferno', mincnt=1)
        try:
            sns.regplot(x=d_phys, y=d_eco, scatter=False, color='cyan', line_kws={'linestyle':'--'})
        except:
            pass
        plt.title("Community Turnover vs. Distance")
        plt.xlabel("Physical Distance (pixels)")
        plt.ylabel("Ecological Distance")
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
        plt.colorbar(im, label="Local Rate of Change")
        plt.axis('off')
        plt.savefig(os.path.join(self.out_dir, "geo_03_volatility.png"), bbox_inches='tight')
        plt.close()

    def run_all(self):
        self.plot_iso_similarity()
        self.plot_variogram()
        self.plot_local_volatility()
        print(f"GeoCovariance Audit Complete.")

# ============================================================
# 4. EXECUTION BLOCK
# ============================================================
if __name__ == "__main__":
    # --- CONFIGURATION ---
    DATA_DIR = "/home/breallis/datasets/ebird_weekly_2023_albers"
    
    # POINT THIS TO WHERE YOUR PREDICTION IS SAVED
    Z_DIR = "/home/breallis/dev/range_limits_pymc/misc_outputs/rbf_stochastic"
    Z_PATH = os.path.join(Z_DIR, "Z_stochastic.npy")
    OUT_DIR = "/home/breallis/dev/range_limits_pymc/misc_outputs/rbf_stochastic"
    SIGMA = 0.5

    if not os.path.exists(Z_PATH):
        raise FileNotFoundError(f"Z matrix not found at {Z_PATH}")
    Z = np.load(Z_PATH)
    print(f"Loaded Z matrix with shape: {Z.shape}")

    # 1. Load the Strict Mask
    valid_mask = load_mask_for_z(Z_DIR, DATA_DIR, "*_abundance_median_2023-*.tif")
    
    # 2. Check Alignment
    N_pixels = np.sum(valid_mask)
    if N_pixels != Z.shape[0]:
        raise ValueError(f"CRITICAL MISMATCH: Z has {Z.shape[0]} rows, but loaded mask has {N_pixels}.")

    # 3. Load Stack
    print("Loading eBird stack...")
    ebird_stack, meta = load_tifs_structured(DATA_DIR, "*_abundance_median_2023-*.tif")
    H, W, D = ebird_stack.shape

    print("Creating aligned feature matrix...")
    # CRITICAL: We flatten the stack using the Z-mask
    ebird_flat_raw = ebird_stack[valid_mask] 
    ebird_smooth = smoothed_hellinger_transform(ebird_flat_raw, meta['n_weeks'], sigma=SIGMA)
    
    # 4. Run Interpretation
    latent_auditor = LatentAuditor(Z, ebird_smooth, meta, valid_mask, H, W, OUT_DIR)
    latent_auditor.run_all() # Generates detailed CSV

    # 5. Run Probe
    print("\n--- Starting Geographic Covariance Probe ---")
    geo_probe = GeoCovarianceProbe(Z, valid_mask, H, W, OUT_DIR)
    geo_probe.run_all()