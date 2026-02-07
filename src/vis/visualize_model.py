import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import jax
import jax.numpy as jnp
import jax.nn as jnn
from numpyro.infer import Predictive
from numpyro.infer.autoguide import AutoDelta
import warnings
import glob

# --- Setup Paths ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.model.priors import build_model_2d 

# --- CONFIGURATION ---
INPUT_DIR = "/home/breallis/processed_data/model_inputs/numpyro_input"
RESULT_DIR = "/home/breallis/processed_data/model_results/map_float32_run_01"
OUTPUT_PLOT_DIR = os.path.join(RESULT_DIR, "plots_analysis")
os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)

from src.model.run_map import load_data_to_gpu

# --- DATA LOADERS ---

# --- CONFIGURATION ---
PRECISION = 'float32' 
jax.config.update("jax_enable_x64", True if PRECISION == 'float64' else False)

# ... inside load_data ...
# Use the same hybrid loader we built for run_map.py 
# to ensure 'st_basis' stays on CPU to avoid OOM


# # Change this to match your run config
# jax.config.update("jax_enable_x64", False)

# def load_data(input_dir):
#     print("Loading data for visualization...")
#     meta_path = os.path.join(input_dir, "metadata.pkl")
#     with open(meta_path, 'rb') as f:
#         meta = pickle.load(f)
        
#     # Keep the big stuff as NumPy arrays (Streaming)
#     z_mem = np.memmap(os.path.join(input_dir, meta['z_gathered_path']), 
#                       dtype='float32', mode='r', 
#                       shape=(meta['time'], meta['N_land'], meta['M']))
    
#     # NEW: Ensure st_basis is loaded too
#     meta['Z_gathered'] = np.array(z_mem).astype('float32')
#     meta['st_basis'] = np.array(meta['st_basis']).astype('float32')
    
#     # Small stuff goes to GPU
#     meta['land_mask'] = jnp.array(meta['land_mask'])
#     # ... (cast other small arrays to jnp as needed)
    
#     print("Data loaded (CPU-Stream ready).")
#     return meta

def plot_temporal_anomalies(params, data, output_dir):
    print("Visualizing Temporal Trends from Spectral Nudge...")
    
    # 1. Safe extraction of weights
    # AutoDelta always uses the _auto_loc suffix for the MAP estimate
    weights = params.get('st_weights_auto_loc')
    if weights is None:
        print("Error: st_weights_auto_loc not found. Check your params keys.")
        return

    # 2. Project weights onto the temporal basis
    # st_basis shape: (N_basis, Time, N_land)
    # Average across space to see the continental-scale yearly pulse
    # We use data['st_basis'] which is your CPU-backed NumPy array
    avg_basis = np.mean(data['st_basis'], axis=2) # Result: (N_basis, Time)
    
    # weights[:, 0] corresponds to the 'r' (growth rate) factor
    # (Time, N_basis) @ (N_basis,) -> (Time,)
    temporal_trend = np.dot(avg_basis.T, weights[:, 0])

    plt.figure(figsize=(12, 5))
    years = data['years']
    plt.plot(years, temporal_trend, marker='o', color='#2b83ba', linewidth=2)
    plt.axhline(0, color='black', linestyle='--', alpha=0.3)
    
    plt.fill_between(years, temporal_trend, 0, where=(temporal_trend > 0), 
                     color='green', alpha=0.1, label='Favorable Year')
    plt.fill_between(years, temporal_trend, 0, where=(temporal_trend < 0), 
                     color='red', alpha=0.1, label='Stressful Year')

    plt.title("Learned Temporal Nudge (Continental Average from Spectral Weights)")
    plt.xlabel("Year")
    plt.ylabel("Growth Deviation")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.savefig(os.path.join(output_dir, "diagnostic_nudge_temporal.png"))
    plt.close()

def reconstruct_simulation(data, params):
    print("Reconstructing simulation...")
    model = build_model_2d
    guide = AutoDelta(model)
    
    # We keep everything related to dispersal, plus the new spectral weights
    return_sites = [
        "simulated_density", "r_flat", "K_flat", "Q_flat", "s_flat", "expected_obs",
        "st_weights", "beta", "r_mean", "dispersal_random",
        "dispersal_intercept", "dispersal_logit_intercept", "dispersal_logit_slope",
        "dispersal_survival_threshold", "allee_intercept", "allee_slope_raw",
        "L_r", "L_K", "l_surv" # These are the coregionalization vectors
    ]
    
    predictive = Predictive(model, guide=guide, params=params, num_samples=1, return_sites=return_sites)
    rng_key = jax.random.PRNGKey(0)
    
    # This now uses the CPU-streaming/jnp.take logic from your updated fields.py
    samples = predictive(rng_key, data=data)
    
    return {k: v[0] for k, v in samples.items()}

def scatter_to_grid_robust(flat_array, rows, cols, shape):
    Ny, Nx = shape
    if flat_array is None: return None
    if flat_array.ndim == 1:
        grid = np.zeros((Ny, Nx)); grid[rows, cols] = flat_array; return grid
    elif flat_array.ndim == 2:
        if flat_array.shape[0] == len(rows):
            K = flat_array.shape[1]; grid = np.zeros((Ny, Nx, K)); grid[rows, cols, :] = flat_array
        else:
            T = flat_array.shape[0]; grid = np.zeros((T, Ny, Nx)); grid[:, rows, cols] = flat_array
        return grid
    elif flat_array.ndim == 3:
        T, _, K = flat_array.shape; grid = np.zeros((T, Ny, Nx, K)); grid[:, rows, cols, :] = flat_array
        return grid
    return None

def scatter_observations_to_grid(obs, t_idx, rows, cols, shape, time_steps):
    grid = np.full((time_steps, shape[0], shape[1]), np.nan)
    grid[t_idx, rows, cols] = obs
    return grid

# --- ORIGINAL VISUALIZATIONS (RESTORED) ---

def plot_spatial_residuals(obs_grid, density, output_dir, land_mask):
    print("Generating Residual Map...")
    log_obs = np.log1p(obs_grid)
    log_pred = np.log1p(density)
    diff = log_obs - log_pred
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mean_resid = np.nanmean(diff, axis=0)
    mean_resid = np.ma.masked_where(land_mask == 0, mean_resid)
    max_abs = np.nanmax(np.abs(mean_resid))
    
    plt.figure(figsize=(10, 8))
    plt.imshow(mean_resid, cmap='RdBu_r', origin='upper', vmin=-max_abs, vmax=max_abs)
    plt.colorbar(label="Mean Log-Residual (Log(Obs) - Log(Pred))")
    plt.title("Spatial Residuals\nRed = Under-prediction | Blue = Over-prediction")
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, "diagnostics_spatial_residuals.png"))
    plt.close()

def plot_global_abundance(data, sim, output_dir):
    """
    Correctly compares observations to simulation by sampling 
    only at observed locations.
    """
    print("Generating Global Abundance Time-Series...")
    
    # 1. Use Expected Observations (sampled at same points as real data)
    # expected_obs: (Time, N_observations)
    # observed_results: (N_observations,)
    # obs_time_indices: (N_observations,)
    
    years_full = data['years']
    exp_obs = sim['expected_obs'] # Get this from your 'reconstruct_simulation'
    obs_counts = data['observed_results']
    obs_t_idx = data['obs_time_indices']
    
    # Sum simulated expected counts per year
    # We need to map the sparse expected_obs back to a yearly sum
    sim_yearly_sum = np.zeros(len(years_full))
    obs_yearly_sum = np.zeros(len(years_full))

    for t_idx in np.unique(obs_t_idx):
        mask = (obs_t_idx == t_idx)
        # Sum only the birds we 'expected' to see at those specific sites
        sim_yearly_sum[t_idx] = np.sum(exp_obs[mask]) 
        obs_yearly_sum[t_idx] = np.sum(obs_counts[mask])

    # Filter for years that actually have data to avoid plotting zeros
    data_mask = obs_yearly_sum > 0
    
    plt.figure(figsize=(10, 6))
    
    # Plot the full simulation trajectory (Global Intensity)
    # We scale this by a factor if you want to see 'Global Biomass' 
    # vs 'Site-specific counts' on the same axes.
    global_intensity = np.sum(sim['simulated_density'], axis=(1, 2))
    
    ax1 = plt.gca()
    ax1.plot(years_full, global_intensity, 'b-', alpha=0.3, label='Total Continental Intensity')
    ax1.set_ylabel("Global Intensity", color='b')
    
    # Plot Site-Specific Comparison
    ax2 = ax1.twinx()
    ax2.plot(years_full[data_mask], sim_yearly_sum[data_mask], 'k--', label='Expected at Sites')
    ax2.scatter(years_full[data_mask], obs_yearly_sum[data_mask], c='r', s=30, label='Actual Observations')
    ax2.set_ylabel("Site-Specific Sums", color='k')
    
    plt.yscale('symlog', linthresh=1.0)
    plt.title("Population Trajectory: Model Fit vs. Observations")
    ax1.legend(loc='upper left'); ax2.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(output_dir, "diagnostics_abundance_fit.png"), dpi=200)
    plt.close()

def plot_parameter_distributions(r_flat, s_flat, Q_flat, output_dir):
    print("Generating Parameter Histograms...")
    
    # Fully flatten everything to view the total population of values
    # No averaging, no masking - just raw samples
    r_vals = np.array(r_flat).flatten()
    s_vals = np.array(s_flat).flatten()
    Q_vals = np.array(Q_flat).flatten()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Growth Rate (r)
    axes[0].hist(r_vals, bins=100, color='steelblue', alpha=0.7)
    axes[0].set_title("Growth Rate (r) - All Values")
    axes[0].set_yscale('log') # Log scale helpful to see the 'speckle' outliers
    
    # Settlement Survival (s)
    axes[1].hist(s_vals, bins=100, color='forestgreen', alpha=0.7)
    axes[1].set_title("Settlement Survival (s) - All Values")
    
    # Path Survival (Q)
    axes[2].hist(Q_vals, bins=100, color='firebrick', alpha=0.7)
    axes[2].set_title("Path Survival (Q) - All Values")

    for ax in axes:
        ax.set_xlabel("Parameter Value")
        ax.set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "diagnostics_param_histograms_raw.png"))
    plt.close()

# --- NEW ANALYTICAL DIAGNOSTICS ---

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import os
import numpy as np


def analyze_dispersal_kernel(data, scalars, output_dir):
    print("Analyzing Impulse Response Kernels...")
    p_juv = jnn.sigmoid(scalars['dispersal_intercept'])
    
    # Get kernel dimensions directly from the data
    adult_k = data['adult_fft_kernel']
    kh, kw = adult_k.shape
    
    # 1. Generate spatial kernels (FFT -> Spatial)
    impulse = jnp.zeros((kh, kw)).at[kh//2, kw//2].set(1.0)
    def conv(pad, kfft): 
        return jnp.real(jnp.fft.ifft2(jnp.fft.fft2(pad) * kfft))

    adult_spatial = jnp.fft.fftshift(conv(impulse, adult_k))
    
    juv_stack = data['juvenile_fft_kernel_stack']
    # Vectorized convolution across the kernel stack
    juv_spatial_all = jax.vmap(lambda k: conv(impulse, k))(juv_stack)
    juv_spatial = jnp.fft.fftshift(jnp.sum(juv_spatial_all, axis=0))

    # 2. Dynamic Radial Profiling (The Fix)
    bins = jnp.arange(0, 50, 1)

    def prof(img):
        h, w = img.shape
        # Center coordinates
        cy, cx = h // 2, w // 2
        # Create coordinate grids that match the image shape exactly
        y, x = jnp.ogrid[:h, :w]
        dist = jnp.sqrt((y - cy)**2 + (x - cx)**2)
        
        # Calculate mean for each radial bin
        radial_means = []
        for i in bins[:-1]:
            mask = (dist >= i) & (dist < i + 1)
            # Use jnp.where to handle empty bins and avoid NaNs
            val = jnp.where(mask.any(), jnp.mean(img[mask]), 0.0)
            radial_means.append(val)
        return jnp.array(radial_means)

    p_a = prof(adult_spatial)
    p_j = prof(juv_spatial)
    
    # Normalization
    p_a_norm = p_a / (jnp.max(p_a) + 1e-12)
    p_j_norm = p_j / (jnp.max(p_j) + 1e-12)
    
    # 3. Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(bins[:-1], p_a_norm, 'b-', label='Adults (Short-Range)')
    plt.plot(bins[:-1], p_j_norm, 'r--', label=f'Juveniles (Long-Range, {p_juv*100:.1f}%)')
    
    plt.xlabel("Distance (Pixels)")
    plt.ylabel("Relative Probability Density")
    plt.title("Learned Dispersal Kernels (Radial Profile)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "analysis_dispersal_reach.png"))
    plt.close()

def analyze_mass_balance(data, output_dir):
    print("Analyzing Dispersal Mass Balance...")
    juv_stack = data['juvenile_fft_kernel_stack']
    kernel_sum_fft = jnp.sum(juv_stack, axis=0)
    kernel_spatial = jnp.real(jnp.fft.ifft2(kernel_sum_fft))
    total_mass = jnp.sum(kernel_spatial)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(np.log1p(jnp.fft.fftshift(kernel_spatial)), cmap='viridis')
    plt.title(f"Kernel Mass Balance\nIntegrated Mass: {total_mass:.6f}")
    plt.savefig(os.path.join(output_dir, "diagnostics_kernel_mass.png")); plt.close()

# --- THE FIXES ---

def plot_growth_stress_test(sim_results, output_dir):
    print("Running Prior Stress Test on Smooth Nudge...")
    # Calculate the 'local nudge' by removing the mean
    r_flat = np.array(sim_results['r_flat'])
    r_mean_fixed = np.mean(r_flat, axis=1, keepdims=True)
    nudge_effect = r_flat - r_mean_fixed
    
    plt.figure(figsize=(10, 6))
    plt.hist(nudge_effect.flatten(), bins=100, color='crimson', alpha=0.7)
    # Since we used dist.Laplace(0, 0.005), most values should be within +/- 0.015
    plt.axvline(0.015, color='black', ls='--') 
    plt.axvline(-0.015, color='black', ls='--')
    plt.title("Magnitude of Smooth Nudge (r_flat deviation)")
    plt.yscale('log')
    plt.savefig(os.path.join(output_dir, "diagnostics_nudge_magnitude.png"))
    plt.close()

def map_outgoing_mortality_fixed(data, Q_grid, output_dir, land_mask):
    print("Mapping Mortality (FFT Fixed)...")
    
    # 1. Reduce Q_grid to 2D Spatial (Ny, Nx)
    # If (T, Ny, Nx, K), average over T (axis 0) and K (axis 3)
    if Q_grid.ndim == 4:
        Q_spatial = jnp.mean(Q_grid, axis=(0, 3))
    elif Q_grid.ndim == 3:
        Q_spatial = jnp.mean(Q_grid, axis=2)
    else:
        Q_spatial = Q_grid
        
    # Mortality is the inverse of path survival
    Mort = 1.0 - Q_spatial
    
    # 2. Setup FFT padding logic
    k_stack = data['juvenile_fft_kernel_stack']
    k_sum_fft = jnp.sum(k_stack, axis=0) # Shape (Kh, Kw)
    
    # Target shape comes from the FFT kernel (which is padded for the FFT)
    target_h, target_w = k_sum_fft.shape 
    ny, nx = Mort.shape
    
    pad_y = target_h - ny
    pad_x = target_w - nx
    
    # Now Mort is definitely 2D, so ((top, bottom), (left, right)) works
    padded_mort = jnp.pad(Mort, ((0, pad_y), (0, pad_x)), mode='edge')
    
    # 3. Perform the 'Sink' convolution
    # This spreads the mortality risk into the pixels that 'feed' into it
    risk_fft = jnp.fft.fft2(padded_mort)
    risk_spatial = jnp.real(jnp.fft.ifft2(risk_fft * k_sum_fft))
    
    # Crop back to original dimensions
    final_risk = risk_spatial[:ny, :nx]
    
    # 4. Visualization
    plt.figure(figsize=(10, 8))
    # Mask ocean for clarity
    masked_risk = np.ma.masked_where(land_mask == 0, final_risk)
    
    plt.imshow(masked_risk, cmap='Reds', origin='upper', vmin=0, vmax=1)
    plt.title("Outgoing Migrant Risk (Ecological Trap Analysis)")
    plt.colorbar(label="Average Probability of Dispersal Mortality")
    plt.savefig(os.path.join(output_dir, "analysis_mortality_risk.png"))
    plt.close()

# --- ANIMATION ---

def create_animation(density, obs_grid, years, output_dir, land_mask, filename="evolution_history.mp4"):
    print("Generating Evolution Animation...")
    fig, (ax_sim, ax_obs) = plt.subplots(1, 2, figsize=(14, 7))
    plt.subplots_adjust(top=0.85)
    vmax = np.log1p(np.nanpercentile(density, 99))
    im_sim = ax_sim.imshow(np.log1p(density[0]), cmap='magma', origin='upper', vmin=0, vmax=vmax)
    land_bg = np.zeros_like(land_mask, dtype=float); land_bg[land_mask == 1] = 0.2
    ax_obs.imshow(land_bg, cmap='gray_r', vmin=0, vmax=1, origin='upper', alpha=0.3)
    im_obs = ax_obs.imshow(np.log1p(obs_grid[0]), cmap='magma', origin='upper', vmin=0, vmax=vmax, interpolation='none')
    title = fig.suptitle(f"Year: {years[0]}", fontsize=16, fontweight='bold')

    def update(frame):
        title.set_text(f"Year: {years[frame]}")
        im_sim.set_data(np.log1p(density[frame])); im_obs.set_data(np.log1p(obs_grid[frame]))
        return im_sim, im_obs, title

    ani = animation.FuncAnimation(fig, update, frames=len(years), interval=100, blit=False)
    save_path = os.path.join(output_dir, filename)
    try:
        writer = animation.FFMpegWriter(fps=10, bitrate=1800)
        ani.save(save_path, writer=writer)
    except Exception:
        ani.save(save_path.replace(".mp4", ".gif"), writer='pillow', fps=10)
    plt.close()


def _render_niche_plot(data, grid, title, filename, output_dir, mode='continuous'):
    """Internal helper to handle the actual matplotlib boilerplate."""
    plt.figure(figsize=(10, 8))
    masked_grid = np.ma.masked_where(data['land_mask'] == 0, grid)
    
    if mode == 'binary':
        # Suitable (R > 0) vs Unsuitable (R <= 0)
        binary_data = (masked_grid > 0).astype(float)
        cmap = mcolors.ListedColormap(['#f7fbff', '#08306b']) # Pale blue to dark blue
        plt.imshow(binary_data, cmap=cmap, origin='upper')
        
        # Explicit Key/Legend
        suitable_patch = mpatches.Patch(color='#08306b', label='Suitable (R > 0)')
        unsuitable_patch = mpatches.Patch(color='#f7fbff', label='Unsuitable (R ≤ 0)')
        plt.legend(handles=[suitable_patch, unsuitable_patch], loc='lower right', frameon=True)
    else:
        # Source/Sink map (Red/Blue)
        v = np.nanpercentile(np.abs(grid), 99)
        im = plt.imshow(masked_grid, cmap='RdBu_r', vmin=-v, vmax=v, origin='upper')
        plt.colorbar(im, label="Net Growth Potential (R)")
        
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=200)
    plt.close()

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

def _render_comparison_plot(data, grid_a, grid_b, title, filename, output_dir, labels_ab=("A", "B")):
    """
    Renders a 4-color comparison map:
    - Blue: Both Valid
    - Green: Only A Valid
    - Red: Only B Valid
    - Grey: Both Invalid
    """
    plt.figure(figsize=(10, 8))
    
    # Binary masks (R > 0)
    bin_a = (grid_a > 0).astype(int)
    bin_b = (grid_b > 0).astype(int)
    
    # Combined logic:
    # 3: Both (A & B)
    # 2: Only A
    # 1: Only B
    # 0: Neither
    comp = np.zeros_like(bin_a)
    comp[(bin_a == 1) & (bin_b == 1)] = 3
    comp[(bin_a == 1) & (bin_b == 0)] = 2
    comp[(bin_a == 0) & (bin_b == 1)] = 1
    
    masked_comp = np.ma.masked_where(data['land_mask'] == 0, comp)
    
    # Color Map: [Grey, Red, Green, Blue]
    cmap = mcolors.ListedColormap(['#e0e0e0', '#d7191c', '#2ca02c', '#2b83ba'])
    plt.imshow(masked_comp, cmap=cmap, origin='upper', vmin=0, vmax=3)
    
    # Legend
    legend_elements = [
        mpatches.Patch(color='#2b83ba', label=f'Both Suitable'),
        mpatches.Patch(color='#2ca02c', label=f'Only {labels_ab[0]} Suitable'),
        mpatches.Patch(color='#d7191c', label=f'Only {labels_ab[1]} Suitable'),
        mpatches.Patch(color='#e0e0e0', label='Neither Suitable')
    ]
    plt.legend(handles=legend_elements, loc='lower right', frameon=True)
    plt.title(title)
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, filename), dpi=200)
    plt.close()

def plot_theoretical_niche_suite(data, r_grid, s_grid, Q_grid, scalars, labels, output_dir):
    print("Generating Comprehensive Niche & Comparison Suite...")
    
    # --- 1. Extraction & Scalar Context ---
    # p_juv: Fraction of population moving.
    p_juv = jnn.sigmoid(scalars['dispersal_intercept'])
    r_mean = np.mean(r_grid, axis=0) if r_grid.ndim == 3 else r_grid
    s_mean = np.mean(s_grid, axis=0) if s_grid.ndim == 3 else s_grid
    Q_spatial = np.mean(Q_grid, axis=0) if Q_grid.ndim == 4 else Q_grid
    
    # --- 2. The Baseline: Adult Static Niche ---
    # Established population persistence (Stayers + Local Settlement)
    phi_adult = (1 - p_juv) + (p_juv * s_mean)
    R_adult = r_mean + np.log(np.maximum(phi_adult, 1e-6))
    _render_niche_plot(data, R_adult, "Adult Static Niche (Established Persistence)", 
                       "niche_adult_static.png", output_dir, mode='binary')

    # --- 3. The Pure Juvenile Kinetic Niches ---
    # Colonization Potential: R = r + log(Q_k)
    # No "stayer" buffer. Does a pioneer survive arriving via kernel K?
    juv_dir = os.path.join(output_dir, "juvenile_kinetic")
    os.makedirs(juv_dir, exist_ok=True)
    
    # Store results for comparisons
    k_results = {}
    for k, lab in enumerate(labels):
        R_k = r_mean + np.log(np.maximum(Q_spatial[..., k], 1e-6))
        k_results[lab] = R_k
        _render_niche_plot(data, R_k, f"Juvenile Kinetic Niche: {lab}", 
                           f"juv_niche_{lab}.png", juv_dir, mode='binary')

    # --- 4. Cardinal & Combined Niches ---
    comp_dir = os.path.join(output_dir, "comparisons")
    os.makedirs(comp_dir, exist_ok=True)
    
    bins = sorted(list(set([l.split('_', 2)[-1] for l in labels])))
    for dist_bin in bins:
        # Define Groupings
        east = f"to_EAST_{dist_bin}"
        west = f"to_WEST_{dist_bin}"
        north = f"to_NORTH_{dist_bin}"
        south = f"to_SOUTH_{dist_bin}"

        # A. Binary Comparisons: Directional Advantage
        if east in k_results and west in k_results:
            _render_comparison_plot(data, k_results[east], k_results[west],
                                    f"Directional Advantage: East vs West ({dist_bin})",
                                    f"comp_E_vs_W_{dist_bin}.png", comp_dir, 
                                    labels_ab=("East", "West"))
        
        if north in k_results and south in k_results:
            _render_comparison_plot(data, k_results[north], k_results[south],
                                    f"Directional Advantage: North vs South ({dist_bin})",
                                    f"comp_N_vs_S_{dist_bin}.png", comp_dir, 
                                    labels_ab=("North", "South"))

        # B. Cost Comparison (Non-Binary Delta)
        # Where is East cheaper/safer than West?
        if east in k_results and west in k_results:
            delta = k_results[east] - k_results[west]
            plt.figure(figsize=(10, 8))
            v = np.nanpercentile(np.abs(delta), 98)
            plt.imshow(np.ma.masked_where(data['land_mask']==0, delta), 
                       cmap='PiYG', vmin=-v, vmax=v, origin='upper')
            plt.colorbar(label="Fitness Advantage (East - West)")
            plt.title(f"Relative Migration Cost: East vs West ({dist_bin})")
            plt.savefig(os.path.join(comp_dir, f"cost_delta_E_W_{dist_bin}.png"))
            plt.close()

    print(f"--- All Diagnostic Maps Generated in {output_dir} ---")

def plot_overall_static_niche(data, r_grid, s_grid, scalars, output_dir):
    """
    Visualizes the Stationary Niche (No Dispersal).
    R = r + log(Stayers + Local Settlers)
    """
    print("Generating Stationary Niche (Full population, zero dispersal)...")
    
    # Extraction
    p_juv = jnn.sigmoid(scalars['dispersal_intercept'])
    r_mean = np.mean(r_grid, axis=0) if r_grid.ndim == 3 else r_grid
    s_mean = np.mean(s_grid, axis=0) if s_grid.ndim == 3 else s_grid
    
    # phi is the probability that a member of the population survives 
    # the year GIVEN they do not use a long-distance kernel.
    phi_stationary = (1 - p_juv) + (p_juv * s_mean)
    R_static = r_mean + np.log(np.maximum(phi_stationary, 1e-6))
    
    _render_niche_plot(data, R_static, "Stationary Niche (Established Persistence)", 
                       "niche_static_stationary.png", output_dir, mode='binary')
    

def plot_global_k_trend(sim, years, output_dir):
    """Plots the sum of K across the landscape for every year."""
    # K_flat is (Time, N_land). Sum across N_land to get (Time,)
    k_total = np.sum(sim['K_flat'], axis=1)
    
    plt.figure(figsize=(10, 5))
    plt.plot(years, k_total, color='#08306b', linewidth=2, marker='s', markersize=4)
    plt.title("Total Continental Carrying Capacity ($K_{total}$)")
    plt.xlabel("Year")
    plt.ylabel("Sum of K across Land")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "trend_global_K.png"))
    plt.close()

def plot_parameter_shifts(data, sim, years, output_dir):
    """Generates maps showing where r, s, and K have increased or decreased."""
    Ny, Nx = data['Ny'], data['Nx']
    rows, cols = data['land_rows'], data['land_cols']
    
    # Indices for comparison
    t0, t1 = 0, -1
    label0, label1 = years[t0], years[t1]
    
    metrics = {
        'Growth (r)': sim['r_flat'],
        'Settlement (s)': sim['s_flat'],
        'Carrying Capacity (K)': sim['K_flat']
    }
    
    shift_dir = os.path.join(output_dir, "parameter_shifts")
    os.makedirs(shift_dir, exist_ok=True)

    for name, grid_flat in metrics.items():
        # Convert flat land arrays to 2D grids for t0 and t1
        m0 = scatter_to_grid_robust(grid_flat[t0], rows, cols, (Ny, Nx))
        m1 = scatter_to_grid_robust(grid_flat[t1], rows, cols, (Ny, Nx))
        
        # Calculate Difference
        delta = m1 - m0
        
        plt.figure(figsize=(10, 8))
        # Use a divergent colormap: Brown/Red (Loss) to Green/Blue (Gain)
        v = np.nanpercentile(np.abs(delta), 98)
        masked_delta = np.ma.masked_where(data['land_mask'] == 0, delta)
        
        plt.imshow(masked_delta, cmap='RdYlGn', vmin=-v, vmax=v, origin='upper')
        plt.colorbar(label=f"Change in {name}")
        plt.title(f"$\Delta$ {name}: {label0} to {label1}")
        plt.axis('off')
        
        filename = f"shift_{name.split('(')[0].strip().lower()}.png"
        plt.savefig(os.path.join(shift_dir, filename), dpi=200)
        plt.close()

# --- MAIN ---

def plot_results():
    data = load_data_to_gpu(INPUT_DIR, precision=PRECISION)
    with open(os.path.join(RESULT_DIR, "map_params.pkl"), 'rb') as f: params = pickle.load(f)

    PATH_INTEGRATION_DIR = "/home/breallis/processed_data/datasets/latent_avian_paths"
    disp_files = glob.glob(os.path.join(PATH_INTEGRATION_DIR, "Z_disp_*.npz"))
    if not disp_files:
        print("Warning: No Z_disp files found. Labels will be generic.")
        labels = [f"Kernel_{i}" for i in range(20)] # Fallback
    else:
        with np.load(disp_files[0]) as loader:
            labels = loader['labels']
    # ----------------------------------------------------------

    # 1. Standard Run
    sim = reconstruct_simulation(data, params)

    # 2. Counter-Factual Run
    params_no_inv = params.copy()

    # Look for the AutoDelta location parameter
    inv_key = 'inv_eta_auto_loc' 

    if inv_key in params_no_inv:
        # Set to -100 so softplus(-100) is effectively 0.0
        params_no_inv[inv_key] = jnp.full_like(params_no_inv[inv_key], -100.0)
    else:
        # Fallback/Safety: Print keys if it still fails to help us find the right one
        print(f"Key {inv_key} not found. Available keys: {list(params_no_inv.keys())}")

    sim_cf = reconstruct_simulation(data, params_no_inv)
    
    Ny, Nx = data['Ny'], data['Nx']; rows, cols = data['land_rows'], data['land_cols']
    years = data['years']
    
    density = np.array(sim['simulated_density']) * data['pop_scalar']
    density_cf = np.array(sim_cf['simulated_density']) * data['pop_scalar']
    r_flat = sim['r_flat']
    s_flat = sim['s_flat']
    Q_flat = sim['Q_flat']
    r_grid = scatter_to_grid_robust(np.array(sim['r_flat']), rows, cols, (Ny, Nx))
    s_grid = scatter_to_grid_robust(np.array(sim['s_flat']), rows, cols, (Ny, Nx))
    Q_grid = scatter_to_grid_robust(np.array(sim['Q_flat']), rows, cols, (Ny, Nx))
    obs_grid = scatter_observations_to_grid(data['observed_results'], data['obs_time_indices'], data['obs_rows'], data['obs_cols'], (Ny, Nx), data['time'])
    
    scalars = {k: sim[k] for k in ['dispersal_intercept', 'dispersal_logit_intercept', 'dispersal_logit_slope', 'allee_intercept']}

    # Execute Full Diagnostic Suite
    create_animation(density, obs_grid, years, OUTPUT_PLOT_DIR, data['land_mask'])
    create_animation(density_cf, obs_grid, years, OUTPUT_PLOT_DIR, data['land_mask'], "evolution_counterfactual.mp4")
    plot_global_k_trend(sim, data['years'], OUTPUT_PLOT_DIR)
    plot_parameter_shifts(data, sim, data['years'], OUTPUT_PLOT_DIR)
    plot_overall_static_niche(data, r_grid, s_grid, scalars, OUTPUT_PLOT_DIR)
    plot_temporal_anomalies(params, data, OUTPUT_PLOT_DIR)
    plot_theoretical_niche_suite(data, r_grid, s_grid, Q_grid, scalars, labels, OUTPUT_PLOT_DIR)
    plot_global_abundance(data, sim, OUTPUT_PLOT_DIR)
    #plot_automated_cardinal_pairs(data, r_grid, s_grid, Q_grid, scalars, labels, OUTPUT_PLOT_DIR)

    plot_spatial_residuals(obs_grid, density, OUTPUT_PLOT_DIR, data['land_mask'])

    plot_parameter_distributions(r_flat, s_flat, Q_flat, OUTPUT_PLOT_DIR)
    analyze_dispersal_kernel(data, scalars, OUTPUT_PLOT_DIR)
    analyze_mass_balance(data, OUTPUT_PLOT_DIR)
    plot_growth_stress_test(sim, OUTPUT_PLOT_DIR)
    map_outgoing_mortality_fixed(data, Q_grid, OUTPUT_PLOT_DIR, data['land_mask'])

    print(f"All diagnostics complete. Results at: {OUTPUT_PLOT_DIR}")

if __name__ == "__main__":
    plot_results()