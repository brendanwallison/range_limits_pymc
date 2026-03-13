import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import glob
import re
import pandas as pd
from datetime import datetime
import rasterio

import jax
import jax.numpy as jnp
from numpyro.infer import Predictive
from numpyro.infer.autoguide import AutoDelta

# --- Setup Paths ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.model.age_priors import build_model_2d 
from src.model.run_map import load_data_to_gpu

# --- CONFIGURATION ---
PRECISION = 'float32' 
jax.config.update("jax_enable_x64", True if PRECISION == 'float64' else False)

INPUT_DIR = "/home/breallis/processed_data/model_inputs/numpyro_input"
RESULT_DIR = f"/home/breallis/processed_data/model_results/age_map_{PRECISION}_run_14"
EBIRD_DIR = "/home/breallis/datasets/ebird_weekly_2023_albers"

OUTPUT_PLOT_DIR = os.path.join(RESULT_DIR, "plots_community_mimicry")
os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)

# ============================================================
# 1. Model Reconstruction
# ============================================================
# ============================================================
# 1. Model Reconstruction
# ============================================================
def reconstruct_simulation(data, params):
    print("Reconstructing latent fields from MAP estimates...")
    from src.model.age_priors import build_model_2d 
    from numpyro.infer import Predictive
    from numpyro.infer.autoguide import AutoDelta
    import jax

    model = build_model_2d
    guide = AutoDelta(model)
    
    # We now need the demographic parameters to calculate R0
    return_sites = [
        "w_env", "alpha_a", "alpha_j", "alpha_f", "alpha_k",
        "gamma_a_raw", "gamma_j_diff", "gamma_f_raw", "gamma_k_raw",
        "allee_gamma", "n50_raw"
    ]
    
    predictive = Predictive(model, guide=guide, params=params, num_samples=1, return_sites=return_sites)
    rng_key = jax.random.PRNGKey(0)
    samples = predictive(rng_key, data=data)
    
    return {k: v[0] for k, v in samples.items()}


# ============================================================
# 2. Robust Data Loader (With Taxonomy Mapping)
# ============================================================
def load_tifs_structured(folder, taxonomy_csv=None, pattern="*_abundance_median_*.tif"):
    files = sorted(glob.glob(os.path.join(folder, pattern)))
    if not files:
        raise ValueError(f"No files found in {folder} matching {pattern}")

    regex = re.compile(r"^([a-zA-Z0-9]+)_abundance_median_(\d{4}-\d{2}-\d{2}).*\.tif$")
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
    
    # --- Apply Taxonomy Mapping ---
    if taxonomy_csv and os.path.exists(taxonomy_csv):
        tax_df = pd.read_csv(taxonomy_csv)
        # Build dictionary: {'abetow': "Abert's Towhee (Melozone aberti)"}
        tax_map = {}
        for _, row in tax_df.iterrows():
            code = str(row['SPECIES_CODE'])
            com_name = str(row['PRIMARY_COM_NAME'])
            sci_name = str(row['SCI_NAME'])
            tax_map[code] = f"{com_name} ({sci_name})"
        
        # Map the column, falling back to the raw code if missing
        df['species_label'] = df['species'].map(lambda x: tax_map.get(x, x))
    else:
        df['species_label'] = df['species']
    
    # Sort -> Species Major, chronological within species
    df_sorted = df.sort_values(by=['species', 'date']).reset_index(drop=True)
    ordered_paths = df_sorted['path'].tolist()
    
    # Generate the beautifully formatted labels
    pseudo_names = [
        f"{row.species_label} | {row.date.strftime('%b %d')}" 
        for row in df_sorted.itertuples()
    ]
    
    with rasterio.open(ordered_paths[0]) as src:
        H, W = src.shape
        
    total_files = len(ordered_paths)
    print(f"Loading {total_files} distinct pseudo-species rasters...")
    
    full_stack = np.zeros((H, W, total_files), dtype=np.float32)
    
    for i, p in enumerate(ordered_paths):
        with rasterio.open(p) as src:
            full_stack[:, :, i] = src.read(1)

    return full_stack, {
        "total_pseudo_species": total_files, 
        "unique_species": df_sorted['species'].nunique(),
        "pseudo_names": pseudo_names 
    }

# ============================================================
# 3. Visualization
# ============================================================
def plot_prior_vs_learned_mechanisms(sim_s_annual, sim_r_annual, R0_annual, unique_species, avonet_csv_path, output_dir):
    import pandas as pd
    import seaborn as sns
    from scipy.stats import spearmanr
    import matplotlib.pyplot as plt
    import os
    import re
    
    print("Integrating AVONET Priors with PyMC Learned Demographics...")
    
    # 1. Extract pure scientific names from "Common Name (Scientific Name)"
    def get_sci_name(label):
        match = re.search(r'\((.*?)\)', label)
        return match.group(1).strip().lower() if match else label.strip().lower()

    # Build a DataFrame of the PyMC Learned results
    df_pymc = pd.DataFrame({
        'sci_norm': [get_sci_name(sp) for sp in unique_species],
        'display_name': unique_species, # Keep the full string for pretty plotting later
        'Learned_Sim_Survival': sim_s_annual,
        'Learned_Sim_Reproduction': sim_r_annual,
        'Learned_R0_Impact': R0_annual
    })
    
    # 2. Load the AVONET Priors
    df_avonet = pd.read_csv(avonet_csv_path)
    df_avonet['sci_norm'] = df_avonet['sci_norm'].str.lower()
    
    # 3. Merge the datasets
    df_merged = pd.merge(df_pymc, df_avonet, on='sci_norm', how='inner')
    print(f"Successfully matched {len(df_merged)} species between PyMC and AVONET.")
    
    if len(df_merged) == 0:
        print("Error: No species matched. Halting AVONET integration.")
        return

    # --- PLOT 1: The Correlation Matrix ---
    prior_cols = ['Trait.Distance', 'Urban.Distance', 'Phylo.Distance', 'Mean.Rank']
    learned_cols = ['Learned_Sim_Survival', 'Learned_Sim_Reproduction', 'Learned_R0_Impact']
    
    # Calculate Spearman rank correlation
    corr_matrix = pd.DataFrame(index=prior_cols, columns=learned_cols)
    for p_col in prior_cols:
        for l_col in learned_cols:
            r, _ = spearmanr(df_merged[p_col], df_merged[l_col], nan_policy='omit')
            corr_matrix.loc[p_col, l_col] = r
            
    corr_matrix = corr_matrix.astype(float)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu', center=0, vmin=-1, vmax=1, fmt=".2f")
    plt.title("Spearman Correlation:\nAVONET Prior Distances vs. Learned Demographic Niche")
    plt.ylabel("Prior Divergence (AVONET)")
    plt.xlabel("Learned Similarity (PyMC)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "6_avonet_correlation_matrix.png"), dpi=300)
    plt.close()

    # --- PLOT 2: The Ecological Release Scatters ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    scatters = [
        ('Phylo.Distance', 'Learned_Sim_Survival', 'Phylogeny vs. Survival Mimicry'),
        ('Urban.Distance', 'Learned_Sim_Reproduction', 'Urban Tolerance vs. Reproduction Mimicry'),
        ('Trait.Distance', 'Learned_R0_Impact', 'Morphology vs. Continental R0 Impact')
    ]
    
    for ax, (x_col, y_col, title) in zip(axes, scatters):
        sns.regplot(data=df_merged, x=x_col, y=y_col, ax=ax, scatter_kws={'alpha':0.5, 'color':'gray'}, line_kws={'color':'red'})
        
        # Annotate top outliers (highest learned similarity, regardless of prior distance)
        top_deviants = df_merged.sort_values(by=y_col, ascending=False).head(3)
        for _, row in top_deviants.iterrows():
            # Cleanly extract just the common name for the plot label
            common_name = row['display_name'].split(" (")[0]
            ax.text(row[x_col], row[y_col], common_name, fontsize=9, color='blue', fontweight='bold')
            
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel(f"{x_col} (Lower = More Similar to HF)")
        ax.set_ylabel(y_col.replace("_", " "))
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "7_avonet_ecological_release_scatters.png"), dpi=300)
    plt.close()

def plot_community_cosine_similarities(sim, Z_native, pseudo_abundance_matrix, pseudo_names, output_dir, top_k=15, avonet_csv_path=None):
    import jax.nn as jnn
    import jax.numpy as jnp
    print("Calculating Community Cosine Similarities and R0 Impacts...")
        
    # --- 1. Extract MAP Parameters ---
    bs = sim['w_env'][:, 0]
    br = sim['w_env'][:, 1]
    
    alpha_a, alpha_j, alpha_f = sim['alpha_a'], sim['alpha_j'], sim['alpha_f']
    gamma_a = float(jnn.softplus(sim['gamma_a_raw']))
    gamma_j = gamma_a + sim['gamma_j_diff']
    gamma_f = float(jnn.softplus(sim['gamma_f_raw']))
    
    bs_norm = bs / (np.linalg.norm(bs) + 1e-9)
    br_norm = br / (np.linalg.norm(br) + 1e-9)
    
    # --- 2. Calculate Weekly Centroids ---
    # Raw summed projection
    W_ps = pseudo_abundance_matrix @ Z_native
    
    # L2 Normalized (for Cosine Similarity)
    W_ps_normalized = W_ps / (np.linalg.norm(W_ps, axis=1, keepdims=True) + 1e-9)
    
    # True Spatial Means (for R0 calculation)
    ps_sums = pseudo_abundance_matrix.sum(axis=1, keepdims=True) + 1e-9
    W_mean_ps = W_ps / ps_sums

    # --- 3. Aggregate to Annual Species Level ---
    base_species_names = [name.split(" | ")[0] for name in pseudo_names]
    unique_species = sorted(list(set(base_species_names)))
    
    W_annual = np.zeros((len(unique_species), Z_native.shape[1]))
    W_mean_annual = np.zeros((len(unique_species), Z_native.shape[1]))
    
    for i, sp in enumerate(unique_species):
        idx = [j for j, name in enumerate(base_species_names) if name == sp]
        W_annual[i] = np.mean(W_ps[idx], axis=0)
        W_mean_annual[i] = np.mean(W_mean_ps[idx], axis=0)

    W_annual_normalized = W_annual / (np.linalg.norm(W_annual, axis=1, keepdims=True) + 1e-9)

    # --- 4. Aggregate to Seasonal Community Level ---
    dates = [name.split(" | ")[1] for name in pseudo_names]
    unique_dates = list(dict.fromkeys(dates)) # Preserves chronological order
    
    W_seasonal = np.zeros((len(unique_dates), Z_native.shape[1]))
    for i, date in enumerate(unique_dates):
        idx = [j for j, name in enumerate(dates) if name == date]
        W_seasonal[i] = np.mean(W_ps[idx], axis=0)

    W_seasonal_normalized = W_seasonal / (np.linalg.norm(W_seasonal, axis=1, keepdims=True) + 1e-9)

    # --- 5. Calculate Metrics ---
    # Weekly similarities
    sim_s_weekly = W_ps_normalized @ bs_norm  
    sim_r_weekly = W_ps_normalized @ br_norm  
    
    # Annual similarities
    sim_s_annual = W_annual_normalized @ bs_norm
    sim_r_annual = W_annual_normalized @ br_norm
    
    # Seasonal similarities
    sim_s_seasonal = W_seasonal_normalized @ bs_norm
    sim_r_seasonal = W_seasonal_normalized @ br_norm

    # --- Calculate True Average Predictive Power (Absolute First) ---
    abs_power_s_seasonal = np.zeros(len(unique_dates))
    abs_power_r_seasonal = np.zeros(len(unique_dates))
    
    abs_s_weekly = np.abs(sim_s_weekly)
    abs_r_weekly = np.abs(sim_r_weekly)
    
    for i, date in enumerate(unique_dates):
        idx = [j for j, name in enumerate(dates) if name == date]
        # Average the absolute similarities of the individual birds present that week
        abs_power_s_seasonal[i] = np.mean(abs_s_weekly[idx])
        abs_power_r_seasonal[i] = np.mean(abs_r_weekly[idx])

    def _plot_chronological_absolute_power(power_s_scores, power_r_scores, date_labels, title, filename):
        fig, ax = plt.subplots(figsize=(12, 6))
        x_pos = np.arange(len(date_labels))
        
        ax.plot(x_pos, power_s_scores, marker='o', color='dodgerblue', linewidth=2.5, label="Average Survival Relevance ($|S_a, S_j|$)")
        ax.plot(x_pos, power_r_scores, marker='s', color='darkorange', linewidth=2.5, label="Average Reproduction Relevance ($|F_{max}|$)")
        
        tick_spacing = max(1, len(date_labels) // 12)
        ax.set_xticks(x_pos[::tick_spacing])
        ax.set_xticklabels([date_labels[i] for i in range(0, len(date_labels), tick_spacing)], rotation=45, ha='right')
        
        ax.set_ylabel("Mean Absolute Predictive Power")
        ax.set_title(title, fontsize=15, fontweight='bold')
        ax.set_ylim(0, 1.05) 
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename), dpi=300)
        plt.close()

    # Demographic R0 Projection (using true spatial means)
    def calc_R0_from_centroid(W_matrix):
        H_s = W_matrix @ bs
        H_r = W_matrix @ br
        S_a = jnp.array(jnn.sigmoid(alpha_a + gamma_a * H_s))
        S_j = jnp.array(jnn.sigmoid(alpha_j + gamma_j * H_s))
        F_max = jnp.array(jnp.exp(alpha_f + gamma_f * H_r))
        return np.array((F_max * S_j) / (1.0 - S_a + 1e-6))
        
    R0_annual = calc_R0_from_centroid(W_mean_annual)

    # --- 6. Plotting Helpers ---
    def _render_bar_chart(scores, labels, title, filename, c_pos, c_neg, xlabel, hline_idx=None):
        colors = [c_pos if s > 0 else c_neg for s in scores]
        fig, ax = plt.subplots(figsize=(10, 8 if hline_idx is None else 12))
        y_pos = np.arange(len(scores))
        ax.barh(y_pos, scores, color=colors, edgecolor='black', alpha=0.8)
        if hline_idx is not None: ax.axhline(hline_idx, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(0, color='black', linewidth=1)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlim(-1.1, 1.1)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename), dpi=300)
        plt.close()

    def _plot_diverging_ranking(sim_scores, labels, title, filename, color_pos, color_neg):
        sorted_idx = np.argsort(sim_scores)
        plot_idx = np.concatenate([sorted_idx[:top_k], sorted_idx[-top_k:]])
        _render_bar_chart(sim_scores[plot_idx], [labels[i] for i in plot_idx], title, filename, color_pos, color_neg, "Cosine Similarity", hline_idx=top_k - 0.5)

    def _plot_absolute_predictive_ranking(sim_scores, labels, title, filename, color_pos, color_neg):
        abs_scores = np.abs(sim_scores)
        sorted_idx = np.argsort(abs_scores)[-top_k:] 
        _render_bar_chart(sim_scores[sorted_idx], [labels[i] for i in sorted_idx], title, filename, color_pos, color_neg, "Cosine Similarity (Ranked by Absolute Power)")

    def _plot_R0_impact_ranking(R0_scores, labels, title, filename):
        sorted_idx = np.argsort(R0_scores)
        plot_idx = np.concatenate([sorted_idx[:top_k], sorted_idx[-top_k:]])
        plot_scores = R0_scores[plot_idx]
        plot_labels = [labels[i] for i in plot_idx]
        
        colors = ['#4575b4' if score >= 1.0 else '#d73027' for score in plot_scores]
        
        fig, ax = plt.subplots(figsize=(10, 12))
        y_pos = np.arange(len(plot_scores))
        ax.barh(y_pos, plot_scores, color=colors, edgecolor='black', alpha=0.8)
        ax.axhline(top_k - 0.5, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(1.0, color='black', linewidth=1.5, linestyle=':', label="Replacement Threshold (R0=1)")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(plot_labels, fontsize=10)
        ax.set_xlabel("Expected Continental $R_0$ of Focal Species")
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename), dpi=300)
        plt.close()

    def _plot_chronological_similarity(sim_s_scores, sim_r_scores, date_labels, title, filename):
        fig, ax = plt.subplots(figsize=(12, 6))
        x_pos = np.arange(len(date_labels))
        
        ax.plot(x_pos, sim_s_scores, marker='o', color='dodgerblue', linewidth=2.5, label="Survival Mimicry ($S_a, S_j$)")
        ax.plot(x_pos, sim_r_scores, marker='s', color='darkorange', linewidth=2.5, label="Reproduction Mimicry ($F_{max}$)")
        ax.axhline(0, color='black', linestyle='-', alpha=0.3)
        
        tick_spacing = max(1, len(date_labels) // 12)
        ax.set_xticks(x_pos[::tick_spacing])
        ax.set_xticklabels([date_labels[i] for i in range(0, len(date_labels), tick_spacing)], rotation=45, ha='right')
        
        ax.set_ylabel("Cosine Similarity to Community Centroid")
        ax.set_title(title, fontsize=15, fontweight='bold')
        ax.set_ylim(-1.1, 1.1)
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename), dpi=300)
        plt.close()

    # --- 7. Generate All Plots ---
    
    # Phenological (Weekly Diverging)
    _plot_diverging_ranking(sim_s_weekly, pseudo_names, f"Phenological Mimicry: Survival\nTop {top_k} Similar/Dissimilar Bird-Weeks", "1a_diverging_survival_pheno.png", 'dodgerblue', 'firebrick')
    _plot_diverging_ranking(sim_r_weekly, pseudo_names, f"Phenological Mimicry: Reproduction\nTop {top_k} Similar/Dissimilar Bird-Weeks", "1b_diverging_reproduction_pheno.png", 'darkorange', 'purple')

    # Annual (Species Diverging)
    _plot_diverging_ranking(sim_s_annual, unique_species, f"Annual Niche Mimicry: Survival\nTop {top_k} Similar/Dissimilar Species Overall", "2a_diverging_survival_annual.png", 'dodgerblue', 'firebrick')
    _plot_diverging_ranking(sim_r_annual, unique_species, f"Annual Niche Mimicry: Reproduction\nTop {top_k} Similar/Dissimilar Species Overall", "2b_diverging_reproduction_annual.png", 'darkorange', 'purple')

    # Annual (Species Absolute Predictive Power)
    _plot_absolute_predictive_ranking(sim_s_annual, unique_species, "Highest Predictive Power: Survival\n(Top 15 Magnitude, Preserving Direction)", "3a_predictive_power_survival.png", 'dodgerblue', 'firebrick')
    _plot_absolute_predictive_ranking(sim_r_annual, unique_species, "Highest Predictive Power: Reproduction\n(Top 15 Magnitude, Preserving Direction)", "3b_predictive_power_reproduction.png", 'darkorange', 'purple')

    # R0 Impact
    _plot_R0_impact_ranking(R0_annual, unique_species, "Aggregate Fitness Translation ($R_0$)\nExpected Focal Success within Community Niches", "4_R0_niche_impact.png")

    # Seasonal (Chronological)
    _plot_chronological_similarity(sim_s_seasonal, sim_r_seasonal, unique_dates, "Seasonal Community Mimicry: Tracking Life-History Across the Year", "5_cosine_similarity_community_seasonal_timeline.png")

    # 5. Seasonal (Chronological Direction)
    _plot_chronological_similarity(
        sim_s_seasonal, sim_r_seasonal, unique_dates, 
        "Seasonal Community Mimicry: Tracking Niche Direction Across the Year", 
        "5a_cosine_similarity_community_seasonal_timeline.png"
    )

    # 6. Seasonal (Chronological Absolute Power)
    _plot_chronological_absolute_power(
        sim_s_seasonal, sim_r_seasonal, unique_dates, 
        "Seasonal Niche Relevance: Absolute Predictive Power Across the Year", 
        "5b_predictive_power_community_seasonal_timeline.png"
    )

    # --- 8. AVONET Integration ---
    if avonet_csv_path and os.path.exists(avonet_csv_path):
        plot_prior_vs_learned_mechanisms(
            sim_s_annual=sim_s_annual, 
            sim_r_annual=sim_r_annual, 
            R0_annual=R0_annual, 
            unique_species=unique_species, 
            avonet_csv_path=avonet_csv_path, 
            output_dir=output_dir
        )
    else:
        print("Skipping AVONET integration: CSV path not provided or file not found.")

# ============================================================
# Main Execution
# ============================================================
if __name__ == "__main__":
    RUZICKA_DIR = "/home/breallis/dev/range_limits_pymc/misc_outputs/ruzicka_sweep_global/sigma_1.0" 
    TAXONOMY_FILE = os.path.join(EBIRD_DIR, "eBird_taxonomy_v2025.csv")
    AVONET_CSV_PATH = "/home/breallis/dev/range_limits_pymc/misc_outputs/AVONET_Comparison_WithPhylogeny_Urban.csv" # Update to full absolute path if needed
    
    print("1. Loading Model Data and MAP Parameters...")
    data = load_data_to_gpu(INPUT_DIR, precision=PRECISION)
    with open(os.path.join(RESULT_DIR, "map_params.pkl"), 'rb') as f: 
        params = pickle.load(f)

    # Reconstruct just what we need to get the MAP w_env
    sim = reconstruct_simulation(data, params)

    print("2. Loading eBird Weekly Abundance Maps...")
    ebird_stack, meta = load_tifs_structured(EBIRD_DIR, taxonomy_csv=TAXONOMY_FILE)
    
    print("3. Prepping Data Matrices on Native High-Res Grid...")
    pseudo_names = meta['pseudo_names']
    
    # Load the native high-res mask and Z matrix 
    valid_mask_high = np.load(os.path.join(RUZICKA_DIR, "valid_mask.npy"))
    Z_native_full = np.load(os.path.join(RUZICKA_DIR, "Z.npy"))
    
    # Get the actual number of latent dimensions used by the PyMC model
    M_model = sim['w_env'].shape[0]
    Z_native = Z_native_full[:, :M_model]
    
    valid_flat_high = valid_mask_high.flatten()
    
    # Zero-fill the NaNs before reshaping and slicing
    ebird_clean = np.nan_to_num(ebird_stack)
    pseudo_abundance_matrix = ebird_clean.reshape(-1, ebird_stack.shape[-1])[valid_flat_high].T

    print("4. Generating Cosine Similarity Rankings and AVONET Comparisons...")
    plot_community_cosine_similarities(
        sim=sim, 
        Z_native=Z_native, 
        pseudo_abundance_matrix=pseudo_abundance_matrix, 
        pseudo_names=pseudo_names, 
        output_dir=OUTPUT_PLOT_DIR,
        top_k=15,
        avonet_csv_path=AVONET_CSV_PATH # --- ADD THIS ARGUMENT ---
    )
    
    print(f"Done! Plots saved to {OUTPUT_PLOT_DIR}")