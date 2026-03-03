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
from jax import lax
from numpyro.infer import Predictive
from numpyro.infer.autoguide import AutoDelta
import warnings
import glob
    
# --- Setup Paths ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.model.age_priors import build_model_2d 
from src.model.age_forward import dispersal_step_age_structured, reproduction_age_structured
from src.model.run_map import load_data_to_gpu

# --- CONFIGURATION ---
PRECISION = 'float32' 
jax.config.update("jax_enable_x64", True if PRECISION == 'float64' else False)

INPUT_DIR = "/home/breallis/processed_data/model_inputs/numpyro_input"
RESULT_DIR = f"/home/breallis/processed_data/model_results/age_map_{PRECISION}_run_01"
OUTPUT_PLOT_DIR = os.path.join(RESULT_DIR, "plots_analysis")
os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)

# --- DATA & RECONSTRUCTION ---

import matplotlib.patches as mpatches

def _render_comparison_plot(data, grid_a, grid_b, title, filename, output_dir, labels_ab=("A", "B")):
    """
    Renders a 4-color comparison map:
    - Blue: Both Valid (Fitness > 1)
    - Green: Only A Valid
    - Red: Only B Valid
    - Grey: Both Invalid
    """
    plt.figure(figsize=(10, 8))
    
    # Binary masks (Fitness > 1 indicates a successful pioneer)
    bin_a = (grid_a > 1.0).astype(int)
    bin_b = (grid_b > 1.0).astype(int)
    
    comp = np.zeros_like(bin_a)
    comp[(bin_a == 1) & (bin_b == 1)] = 3
    comp[(bin_a == 1) & (bin_b == 0)] = 2
    comp[(bin_a == 0) & (bin_b == 1)] = 1
    
    masked_comp = np.ma.masked_where(data['land_mask'] == 0, comp)
    
    cmap = mcolors.ListedColormap(['#e0e0e0', '#d7191c', '#2ca02c', '#2b83ba'])
    plt.imshow(masked_comp, cmap=cmap, origin='upper', vmin=0, vmax=3)
    
    legend_elements = [
        mpatches.Patch(color='#2b83ba', label=f'Both Successful'),
        mpatches.Patch(color='#2ca02c', label=f'Only {labels_ab[0]} Successful'),
        mpatches.Patch(color='#d7191c', label=f'Only {labels_ab[1]} Successful'),
        mpatches.Patch(color='#e0e0e0', label='Neither Successful')
    ]
    plt.legend(handles=legend_elements, loc='lower right', frameon=True)
    plt.title(title)
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, filename), dpi=200)
    plt.close()

def plot_directional_asymmetry(data, Sa_grid, Sj_grid, Fmax_grid, Q_grid, labels, output_dir):
    print("Analyzing Asymmetric Cost of Dispersal (Origins: West vs East, South vs North)...")
    
    comp_dir = os.path.join(output_dir, "comparisons")
    os.makedirs(comp_dir, exist_ok=True)
    
    # Average modern biology
    Sa_avg = np.mean(Sa_grid, axis=0)
    Sj_avg = np.mean(Sj_grid, axis=0)
    Fmax_avg = np.mean(Fmax_grid, axis=0)
    Q_spatial = np.mean(Q_grid, axis=0) if Q_grid.ndim == 4 else Q_grid
    
    # Stationary R0 (The destination's baseline quality)
    R0 = (Fmax_avg * Sj_avg) / (1.0 - Sa_avg + 1e-6)
    
    # Extract unique distance bins from labels (e.g., "30-150")
    bins = sorted(list(set([l.split('_', 2)[-1] for l in labels])))
    
    for dist_bin in bins:
        # ---------------------------------------------------------
        # 1. EAST vs WEST (Originating from WEST vs Originating from EAST)
        # ---------------------------------------------------------
        kernel_to_east = f"to_EAST_{dist_bin}"
        kernel_to_west = f"to_WEST_{dist_bin}"
        
        if kernel_to_east in labels and kernel_to_west in labels:
            idx_to_east = labels.index(kernel_to_east)
            idx_to_west = labels.index(kernel_to_west)
            
            # Pioneer Fitness = Destination R0 * Journey Survival
            # Traveling EAST means originating from the WEST
            fit_from_west = R0 * Q_spatial[..., idx_to_east]
            # Traveling WEST means originating from the EAST
            fit_from_east = R0 * Q_spatial[..., idx_to_west]
            
            # Binary Categorical Comparison
            _render_comparison_plot(
                data, fit_from_west, fit_from_east,
                f"Pioneer Success: Arriving from West vs East ({dist_bin}km)",
                f"comp_from_W_vs_E_{dist_bin}.png", comp_dir, 
                labels_ab=("From West", "From East")
            )
            
            # Continuous Fitness Delta Map
            delta_we = fit_from_west - fit_from_east
            
            plt.figure(figsize=(10, 8))
            v = np.nanpercentile(np.abs(delta_we), 98)
            masked_delta_we = np.ma.masked_where(data['land_mask']==0, delta_we)
            
            # Green (Positive) = Better to arrive from West
            # Pink (Negative) = Better to arrive from East
            plt.imshow(masked_delta_we, cmap='PiYG', vmin=-v, vmax=v, origin='upper')
            plt.colorbar(label="Advantage: Arriving from West > Arriving from East")
            plt.title(f"Migration Fitness Delta: Origin West vs Origin East ({dist_bin}km)")
            plt.axis('off')
            plt.savefig(os.path.join(comp_dir, f"cost_delta_from_W_E_{dist_bin}.png"), dpi=200)
            plt.close()

        # ---------------------------------------------------------
        # 2. NORTH vs SOUTH (Originating from SOUTH vs Originating from NORTH)
        # ---------------------------------------------------------
        kernel_to_north = f"to_NORTH_{dist_bin}"
        kernel_to_south = f"to_SOUTH_{dist_bin}"
        
        if kernel_to_north in labels and kernel_to_south in labels:
            idx_to_north = labels.index(kernel_to_north)
            idx_to_south = labels.index(kernel_to_south)
            
            # Traveling NORTH means originating from the SOUTH
            fit_from_south = R0 * Q_spatial[..., idx_to_north]
            # Traveling SOUTH means originating from the NORTH
            fit_from_north = R0 * Q_spatial[..., idx_to_south]
            
            # Binary Categorical Comparison
            _render_comparison_plot(
                data, fit_from_south, fit_from_north,
                f"Pioneer Success: Arriving from South vs North ({dist_bin}km)",
                f"comp_from_S_vs_N_{dist_bin}.png", comp_dir, 
                labels_ab=("From South", "From North")
            )
            
            # Continuous Fitness Delta Map
            delta_sn = fit_from_south - fit_from_north
            
            plt.figure(figsize=(10, 8))
            v = np.nanpercentile(np.abs(delta_sn), 98)
            masked_delta_sn = np.ma.masked_where(data['land_mask']==0, delta_sn)
            
            # Purple (Positive) = Better to arrive from South
            # Orange (Negative) = Better to arrive from North
            plt.imshow(masked_delta_sn, cmap='PuOr', vmin=-v, vmax=v, origin='upper')
            plt.colorbar(label="Advantage: Arriving from South > Arriving from North")
            plt.title(f"Migration Fitness Delta: Origin South vs Origin North ({dist_bin}km)")
            plt.axis('off')
            plt.savefig(os.path.join(comp_dir, f"cost_delta_from_S_N_{dist_bin}.png"), dpi=200)
            plt.close()

def scatter_observations_to_grid(obs, t_idx, rows, cols, shape, time_steps):
    grid = np.full((time_steps, shape[0], shape[1]), np.nan)
    grid[t_idx, rows, cols] = obs
    return grid

def create_animation(density, obs_grid, years, output_dir, land_mask, filename="evolution_history.mp4"):
    print(f"Generating Animation: {filename}...")
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
        im_sim.set_data(np.log1p(density[frame]))
        im_obs.set_data(np.log1p(obs_grid[frame]))
        return im_sim, im_obs, title

    ani = animation.FuncAnimation(fig, update, frames=len(years), interval=100, blit=False)
    save_path = os.path.join(output_dir, filename)
    try:
        writer = animation.FFMpegWriter(fps=10, bitrate=1800)
        ani.save(save_path, writer=writer)
    except Exception:
        print(f"FFMpegWriter failed. Saving as GIF instead: {save_path.replace('.mp4', '.gif')}")
        ani.save(save_path.replace(".mp4", ".gif"), writer='pillow', fps=10)
    plt.close()

def reconstruct_simulation(data, params):
    print("Reconstructing latent fields...")
    model = build_model_2d
    guide = AutoDelta(model)
    
    return_sites = [
        "simulated_density", "Sa_flat", "Sj_flat", "Fmax_flat", "K_flat", "Q_flat", "expected_obs",
        "st_weights", "beta_h", "dispersal_random",
        "dispersal_logit_intercept", "dispersal_logit_slope",
        "allee_intercept", "allee_slope_raw",
        "alpha_a", "alpha_j", "alpha_f", "alpha_k"
    ]
    
    predictive = Predictive(model, guide=guide, params=params, num_samples=1, return_sites=return_sites)
    rng_key = jax.random.PRNGKey(0)
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

def rebuild_age_pools(sim, data, params):
    """
    Reruns the forward simulation using the MAP parameters to explicitly 
    extract the N_adult and N_juvenile spatial grids over time.
    """
    print("Rebuilding explicit Age-Structured Pools (Na, Nj)...")
    time = data['time']
    Ny, Nx = data['Ny'], data['Nx']
    row, col = data['inv_location']
    inv_pop = jnn.softplus(params['inv_eta_auto_loc'])
    
    allee_scalar = data['pop_scalar'] * jnn.softplus(params['allee_slope_raw_auto_loc'])
    allee_int = params['allee_intercept_auto_loc']
    disp_log_int = params['dispersal_logit_intercept_auto_loc']
    disp_log_slope = params['dispersal_logit_slope_auto_loc']
    disp_random = sim['dispersal_random']

    def step(pools, t):
        N_a, N_j = pools
        
        # Invasion
        k = t - data['inv_timestep']
        is_invading = (k >= 0) & (k < inv_pop.shape[0])
        val = jnp.where(is_invading, inv_pop[jnp.minimum(jnp.maximum(0, k), inv_pop.shape[0]-1)], 0.0)
        N_a = N_a.at[row, col].add(val)

        # Scatter params
        Sa_g = jnp.zeros((Ny, Nx)).at[data['land_rows'], data['land_cols']].set(sim['Sa_flat'][t])
        Sj_g = jnp.zeros((Ny, Nx)).at[data['land_rows'], data['land_cols']].set(sim['Sj_flat'][t])
        Fmax_g = jnp.zeros((Ny, Nx)).at[data['land_rows'], data['land_cols']].set(sim['Fmax_flat'][t])
        K_g = jnp.zeros((Ny, Nx)).at[data['land_rows'], data['land_cols']].set(sim['K_flat'][t])
        
        Q_t = sim['Q_flat'][t]
        Q_temp = jnp.zeros((Ny, Nx, Q_t.shape[-1])).at[data['land_rows'], data['land_cols'], :].set(Q_t)
        Q_g = Q_temp.transpose(2, 0, 1)

        # Disperse
        N_a_post, N_j_post = dispersal_step_age_structured(
            N_a, N_j, K_g, 
            disp_log_int, disp_log_slope, 0.8,
            data['adult_edge_correction'], data['juvenile_edge_correction_stack'],
            data['adult_fft_kernel'], data['juvenile_fft_kernel_stack'],
            Q_grid=Q_g, eps=1e-6
        )
        
        # Reproduce & Survive
        N_a_new, N_j_new = reproduction_age_structured(
            N_a_post, N_j_post, Sa_g, Sj_g, Fmax_g, K_g, 
            allee_scalar, allee_int, eps=1e-12
        )
        
        N_a_new = jnp.maximum(N_a_new * data['land_mask'], 0.0)
        N_j_new = jnp.maximum(N_j_new * data['land_mask'], 0.0)
        
        return (N_a_new, N_j_new), (N_a_new, N_j_new)

    init_N_a = data['initpop_latent'] * 0.5
    init_N_j = data['initpop_latent'] * 0.5
    
    _, (Na_grid, Nj_grid) = lax.scan(step, (init_N_a, init_N_j), jnp.arange(time))
    return np.array(Na_grid), np.array(Nj_grid)

# --- VISUALIZATIONS ---

def plot_demographic_baselines(data, Sa_grid, Sj_grid, Fmax_grid, K_grid, output_dir):
    print("Generating Age-Structured Demographic Baselines...")
    land_mask = data['land_mask']
    
    Sa_avg = np.mean(Sa_grid, axis=0)
    Sj_avg = np.mean(Sj_grid, axis=0)
    Fmax_avg = np.mean(Fmax_grid, axis=0)
    K_avg = np.mean(K_grid, axis=0)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plots = [
        (Sa_avg, "Adult Survival ($S_a$)", "plasma", axes[0, 0], (0, 1)),
        (Sj_avg, "Juvenile Survival ($S_j$)", "plasma", axes[0, 1], (0, 1)),
        (Fmax_avg, "Maximum Fecundity ($F_{\max}$)", "viridis", axes[1, 0], None),
        (K_avg, "Carrying Capacity ($K$)", "cividis", axes[1, 1], 'log')
    ]

    for grid, title, cmap, ax, scale in plots:
        masked_grid = np.ma.masked_where(land_mask == 0, grid)
        if scale == 'log':
            im = ax.imshow(np.log1p(masked_grid), cmap=cmap, origin='upper')
            cbar_label = "Log(1+K)"
        elif isinstance(scale, tuple):
            im = ax.imshow(masked_grid, cmap=cmap, origin='upper', vmin=scale[0], vmax=scale[1])
            cbar_label = "Probability"
        else:
            im = ax.imshow(masked_grid, cmap=cmap, origin='upper')
            cbar_label = "Offspring per Adult"

        ax.set_title(title, fontsize=14)
        ax.axis('off')
        plt.colorbar(im, ax=ax, label=cbar_label, fraction=0.046, pad=0.04)

    plt.suptitle("Age-Structured Demographic Manifold (Mean 1960-2023)", fontsize=18, y=0.95)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "analysis_demographic_baselines.png"), dpi=200)
    plt.close()

def plot_age_structure_dynamics(Na_grid, Nj_grid, years, output_dir, land_mask):
    print("Visualizing Shifting Age Structure...")
    
    # 1. Global Time Series
    total_Na = np.sum(Na_grid, axis=(1, 2))
    total_Nj = np.sum(Nj_grid, axis=(1, 2))
    total_pop = total_Na + total_Nj + 1e-9
    global_rho = total_Nj / total_pop

    plt.figure(figsize=(10, 5))
    plt.plot(years, global_rho, color='forestgreen', linewidth=2)
    plt.axhline(0.5, color='black', linestyle='--', label="50/50 Theoretical Target")
    plt.title("Continental Juvenile Fraction ($\\rho = N_j / N_{total}$)")
    plt.xlabel("Year")
    plt.ylabel("Proportion of Population")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "trend_age_structure.png"))
    plt.close()

    # 2. Spatial Maps (Decadal Snapshots)
    snapshots = [1970, 1990, 2010, 2020]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    for ax, year in zip(axes.flatten(), snapshots):
        t_idx = np.where(years == year)[0][0]
        na = Na_grid[t_idx]
        nj = Nj_grid[t_idx]
        
        # Only plot age structure where birds actually exist (Density > 0.01)
        valid_pop = (na + nj) > 0.01
        rho_spatial = np.where(valid_pop, nj / (na + nj + 1e-9), np.nan)
        rho_masked = np.ma.masked_where((land_mask == 0) | ~valid_pop, rho_spatial)
        
        # Diverging colormap: Brown (Adult Dominated) to Green (Juvenile Dominated)
        cmap = plt.cm.get_cmap('BrBG', 11)
        im = ax.imshow(rho_masked, cmap=cmap, origin='upper', vmin=0.2, vmax=0.8)
        ax.set_title(f"Juvenile Fraction - {year}")
        ax.axis('off')

    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])
    fig.colorbar(im, cax=cbar_ax, orientation='horizontal', label="Juvenile Fraction (Brown = Adult heavy, Green = Juvenile heavy)")
    plt.suptitle("Spatiotemporal Age Structure Dynamics", fontsize=18, y=0.95)
    plt.savefig(os.path.join(output_dir, "map_age_structure_snapshots.png"), dpi=200)
    plt.close()

def analyze_source_sink_mortality(data, Sa_grid, Sj_grid, Fmax_grid, Q_grid, output_dir):
    print("Analyzing Source-Sink and Dispersal Mortality...")
    land_mask = data['land_mask']
    
    Sa_avg = np.mean(Sa_grid, axis=0)
    Sj_avg = np.mean(Sj_grid, axis=0)
    Fmax_avg = np.mean(Fmax_grid, axis=0)
    
    # 1. Calculate Expected Replacement Rate (R0)
    R0 = (Fmax_avg * Sj_avg) / (1.0 - Sa_avg + 1e-6)
    R0_masked = np.ma.masked_where(land_mask == 0, R0)
    
    # --- PLOT A: The Strict Binary Map ---
    plt.figure(figsize=(10, 8))
    # 1 = Source (R0 > 1), 0 = Sink (R0 <= 1)
    binary_map = (R0_masked > 1.0).astype(float)
    cmap_binary = mcolors.ListedColormap(['#d73027', '#4575b4']) # Red, Blue
    
    plt.imshow(binary_map, cmap=cmap_binary, origin='upper')
    
    # Custom Legend
    import matplotlib.patches as mpatches
    legend_elements = [
        mpatches.Patch(color='#4575b4', label='Source ($R_0 > 1$)'),
        mpatches.Patch(color='#d73027', label='Sink ($R_0 \leq 1$)')
    ]
    plt.legend(handles=legend_elements, loc='lower right', frameon=True)
    plt.title("Binary Source-Sink Landscape")
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, "analysis_source_sink_binary.png"), dpi=200)
    plt.close()

    # --- PLOT B: The Principled Core vs. Marginal Map ---
    plt.figure(figsize=(10, 8))
    
    # Define thresholds (e.g., +/- 10% deviation from replacement)
    upper_thresh = 1.1
    lower_thresh = 0.9
    
    # 0 = Core Sink, 1 = Marginal, 2 = Core Source
    ternary_map = np.zeros_like(R0_masked)
    ternary_map = np.where(R0_masked > upper_thresh, 2, ternary_map)
    ternary_map = np.where((R0_masked >= lower_thresh) & (R0_masked <= upper_thresh), 1, ternary_map)
    ternary_map = np.where(R0_masked < lower_thresh, 0, ternary_map)
    
    ternary_masked = np.ma.masked_where(land_mask == 0, ternary_map)
    
    # Red (Sink), Light Grey (Marginal), Blue (Source)
    cmap_ternary = mcolors.ListedColormap(['#d73027', '#e0e0e0', '#4575b4'])
    
    plt.imshow(ternary_masked, cmap=cmap_ternary, origin='upper')
    
    legend_elements_ternary = [
        mpatches.Patch(color='#4575b4', label=f'Core Source ($R_0 > {upper_thresh}$)'),
        mpatches.Patch(color='#e0e0e0', label=f'Marginal Habitat'),
        mpatches.Patch(color='#d73027', label=f'Core Sink ($R_0 < {lower_thresh}$)')
    ]
    plt.legend(handles=legend_elements_ternary, loc='lower right', frameon=True)
    plt.title(f"Core Source-Sink Landscape ($\pm 10\%$ Threshold)")
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, "analysis_source_sink_core.png"), dpi=200)
    plt.close()

def plot_spatial_residuals(obs_grid, density, output_dir, land_mask):
    print("Generating Spatial Residual Map...")
    
    # Use log1p to handle zeros and massive population variances smoothly
    log_obs = np.log1p(obs_grid)
    log_pred = np.log1p(density)
    
    # Residual = Observation - Prediction
    # Positive (Red) = We under-predicted (Birds are there, model says no)
    # Negative (Blue) = We over-predicted (Model says birds are there, they aren't)
    diff = log_obs - log_pred
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Average the residuals across all time steps
        mean_resid = np.nanmean(diff, axis=0)
        
    mean_resid = np.ma.masked_where(land_mask == 0, mean_resid)
    max_abs = np.nanmax(np.abs(mean_resid))
    
    plt.figure(figsize=(10, 8))
    plt.imshow(mean_resid, cmap='RdBu_r', origin='upper', vmin=-max_abs, vmax=max_abs)
    plt.colorbar(label="Mean Log-Residual (Log(Obs) - Log(Pred))")
    plt.title("Spatial Residuals (Average across Time)\nRed = Under-predicted | Blue = Over-predicted")
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, "diagnostics_spatial_residuals.png"), dpi=200)
    plt.close()

# --- MAIN ---

def plot_results():
    # --- Load Kernel Labels ---
    # Update this path to wherever your path integration .npz files live
    PATH_INTEGRATION_DIR = "/home/breallis/processed_data/datasets/latent_avian_path_diagnostics"
    disp_files = glob.glob(os.path.join(PATH_INTEGRATION_DIR, "Z_disp_*.npz"))
    
    if not disp_files:
        print("Warning: No Z_disp files found. Labels will be generic.")
        # Fallback assuming you have K kernels (find K from Q_grid shape)
        K_kernels = data['juvenile_fft_kernel_stack'].shape[0]
        labels = [f"Kernel_{i}" for i in range(K_kernels)] 
    else:
        with np.load(disp_files[0]) as loader:
            # np.load returns numpy arrays; convert to a standard Python list of strings
            labels = [str(lbl) for lbl in loader['labels']]
    # --------------------------

    data = load_data_to_gpu(INPUT_DIR, precision=PRECISION)
    with open(os.path.join(RESULT_DIR, "map_params.pkl"), 'rb') as f: params = pickle.load(f)

    # 1. Standard Run
    sim = reconstruct_simulation(data, params)
    
    # 2. Counter-Factual Run (No Invasion)
    params_no_inv = params.copy()
    inv_key = 'inv_eta_auto_loc' 
    if inv_key in params_no_inv:
        # Set to -100 so softplus(-100) is effectively 0.0
        params_no_inv[inv_key] = jnp.full_like(params_no_inv[inv_key], -100.0)
    else:
        print(f"Key {inv_key} not found. Available keys: {list(params_no_inv.keys())}")

    sim_cf = reconstruct_simulation(data, params_no_inv)
    
    Ny, Nx = data['Ny'], data['Nx']
    rows, cols = data['land_rows'], data['land_cols']
    years = data['years']
    
    # Extract Total Densities
    density = np.array(sim['simulated_density']) * data['pop_scalar']
    density_cf = np.array(sim_cf['simulated_density']) * data['pop_scalar']
    
    # Map 1D arrays back to 2D Spatial Grids
    Sa_grid = scatter_to_grid_robust(np.array(sim['Sa_flat']), rows, cols, (Ny, Nx))
    Sj_grid = scatter_to_grid_robust(np.array(sim['Sj_flat']), rows, cols, (Ny, Nx))
    Fmax_grid = scatter_to_grid_robust(np.array(sim['Fmax_flat']), rows, cols, (Ny, Nx))
    K_grid = scatter_to_grid_robust(np.array(sim['K_flat']), rows, cols, (Ny, Nx))
    Q_grid = scatter_to_grid_robust(np.array(sim['Q_flat']), rows, cols, (Ny, Nx))
    
    obs_grid = scatter_observations_to_grid(data['observed_results'], data['obs_time_indices'], data['obs_rows'], data['obs_cols'], (Ny, Nx), data['time'])

    start_idx = np.where(years >= 1960)[0][0]
    
    # Rebuild explicit N_a and N_j pools via forward pass
    Na_grid, Nj_grid = rebuild_age_pools(sim, data, params)

    # --- Execute Diagnostics ---
    plot_spatial_residuals(obs_grid, density, OUTPUT_PLOT_DIR, data['land_mask'])
    plot_directional_asymmetry(data, Sa_grid[start_idx:], Sj_grid[start_idx:], Fmax_grid[start_idx:], Q_grid[start_idx:], labels, OUTPUT_PLOT_DIR)

    # Age-Structured Analysis
    plot_demographic_baselines(data, Sa_grid[start_idx:], Sj_grid[start_idx:], Fmax_grid[start_idx:], K_grid[start_idx:], OUTPUT_PLOT_DIR)
    plot_age_structure_dynamics(Na_grid, Nj_grid, years, OUTPUT_PLOT_DIR, data['land_mask'])
    analyze_source_sink_mortality(data, Sa_grid[start_idx:], Sj_grid[start_idx:], Fmax_grid[start_idx:], Q_grid[start_idx:], OUTPUT_PLOT_DIR)

    # Original Animations
    create_animation(density, obs_grid, years, OUTPUT_PLOT_DIR, data['land_mask'])
    create_animation(density_cf, obs_grid, years, OUTPUT_PLOT_DIR, data['land_mask'], "evolution_counterfactual.mp4")
    
    print(f"All diagnostics complete. Results at: {OUTPUT_PLOT_DIR}")

if __name__ == "__main__":
    plot_results()