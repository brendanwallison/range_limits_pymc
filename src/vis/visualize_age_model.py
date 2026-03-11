import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
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
RESULT_DIR = f"/home/breallis/processed_data/model_results/age_map_{PRECISION}_run_10"
OUTPUT_PLOT_DIR = os.path.join(RESULT_DIR, "plots_analysis")
os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)

# --- 1. DIRECT POSTERIORS (MAP ESTIMATES) ---
def plot_posterior_weights(sim, M, z_names, output_dir):
    """Plots the MAP point estimates for beta_s and beta_r."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Extract exact MAP point estimates (no averaging needed)
    bs_map = sim['beta_s']
    br_map = sim['beta_r']
    
    y_pos = np.arange(M)
    
    # 1. Draw the connecting dotted lines FIRST so they stay in the background (zorder=1)
    for i in range(M):
        ax.plot([bs_map[i], br_map[i]], [y_pos[i], y_pos[i]], 
                color='gray', linestyle=':', alpha=0.7, zorder=1)
    
    # 2. Plot Survival Weights (no error bars for MAP)
    ax.scatter(bs_map, y_pos, color='dodgerblue', label='Survival (beta_s)', 
               marker='o', s=64, zorder=3)
    
    # 3. Plot Reproductive Weights
    ax.scatter(br_map, y_pos, color='darkorange', label='Reproduction (beta_r)', 
               marker='s', s=64, zorder=2)
    
    ax.axvline(0, color='black', linestyle='-', alpha=0.3, zorder=0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(z_names)
    ax.set_xlabel("MAP Weight Estimate")
    ax.set_title("Environmental Profile: Survival vs. Reproduction (MAP)")
    
    # Add a little padding to the top and bottom
    ax.set_ylim(-0.5, M - 0.5)
    
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "1_posterior_weights_map.png"), dpi=300)
    plt.close()

# --- 2. DEMOGRAPHIC RESPONSE CURVES ---
def plot_demographic_response_curves(sim, M, z_names, target_z_idx, output_dir):
    """Sweeps a target Z axis to show the non-linear biological response."""
    z_sweep = np.linspace(-3, 3, 100)
    
    Z_matrix = np.zeros((100, M))
    Z_matrix[:, target_z_idx] = z_sweep
    
    # Extract direct MAP estimates
    alpha_a = sim['alpha_a']
    alpha_j = sim['alpha_j']
    alpha_f = sim['alpha_f']
    
    gamma_a = float(jnn.softplus(sim['gamma_a_raw']))
    gamma_j = gamma_a + sim['gamma_j_diff']
    gamma_f = float(jnn.softplus(sim['gamma_f_raw']))
    
    bs = sim['beta_s']
    br = sim['beta_r']
    
    H_s = Z_matrix @ bs
    H_r = Z_matrix @ br
    
    S_a = jnn.sigmoid(alpha_a + gamma_a * H_s)
    S_j = jnn.sigmoid(alpha_j + gamma_j * H_s)
    F_max = jnp.exp(alpha_f + gamma_f * H_r)
    
    fig, ax1 = plt.subplots(figsize=(8, 6))
    
    ax1.plot(z_sweep, S_a, color='navy', linewidth=2, label='Adult Survival (S_a)')
    ax1.plot(z_sweep, S_j, color='royalblue', linewidth=2, linestyle='--', label='Juvenile Survival (S_j)')
    ax1.set_xlabel(f"{z_names[target_z_idx]} Gradient (Standardized)")
    ax1.set_ylabel("Annual Survival Probability", color='navy')
    ax1.set_ylim(0, 1.0)
    ax1.tick_params(axis='y', labelcolor='navy')
    
    ax2 = ax1.twinx()
    ax2.plot(z_sweep, F_max, color='darkorange', linewidth=2, label='Max Fecundity (F_max)')
    ax2.set_ylabel("Maximum Fecundity", color='darkorange')
    ax2.tick_params(axis='y', labelcolor='darkorange')
    
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
    
    plt.title(f"Demographic Response to {z_names[target_z_idx]}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"2_response_curve_{z_names[target_z_idx]}.png"), dpi=300)
    plt.close()

# --- 3. ENVIRONMENTAL DRIVERS AND LIMITS (TEMPORAL EPOCHS) ---
def plot_environmental_drivers_and_limits(sim, data, M, z_names, output_dir):
    """Maps the Z features acting as the primary drivers (+) and limits (-) across two eras."""
    print("   -> Calculating Temporal Epochs (1970-1975 vs 2018-2023)...")
    bs = sim['beta_s']
    br = sim['beta_r']
    
    Z_gathered = data['Z_gathered']
    years = data['years']
    
    idx_early = np.where((years >= 1970) & (years <= 1975))[0]
    idx_late = np.where((years >= 2018) & (years <= 2023))[0]
    
    if Z_gathered.ndim >= 3:
        Z_early = np.mean(Z_gathered[idx_early], axis=0) # Time averaging remains
        Z_late = np.mean(Z_gathered[idx_late], axis=0)
    else:
        Z_early = Z_late = Z_gathered
        
    cont_s_early, cont_r_early = Z_early * bs, Z_early * br
    cont_s_late, cont_r_late = Z_late * bs, Z_late * br
    
    limits = {
        'S_early': np.argmin(cont_s_early, axis=-1), 'S_late': np.argmin(cont_s_late, axis=-1),
        'R_early': np.argmin(cont_r_early, axis=-1), 'R_late': np.argmin(cont_r_late, axis=-1)
    }
    drivers = {
        'S_early': np.argmax(cont_s_early, axis=-1), 'S_late': np.argmax(cont_s_late, axis=-1),
        'R_early': np.argmax(cont_r_early, axis=-1), 'R_late': np.argmax(cont_r_late, axis=-1)
    }
    
    def _to_grid(flat_idx):
        if flat_idx.ndim == 1:
            grid = np.full((data['Ny'], data['Nx']), np.nan)
            grid[data['land_rows'], data['land_cols']] = flat_idx
            return grid
        return np.where(data['land_mask'] == 1, flat_idx, np.nan)

    cmap = plt.get_cmap('tab20', M)
    patches = [mpatches.Patch(color=cmap(i), label=z_names[i]) for i in range(M)]

    def _render_2x2_grid(data_dict, title_prefix, filename):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        axes[0, 0].imshow(_to_grid(data_dict['S_early']), cmap=cmap, origin='upper', vmin=-0.5, vmax=M-0.5)
        axes[0, 0].set_title("Survival ($S_a, S_j$) | 1970-1975", fontsize=14)
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(_to_grid(data_dict['R_early']), cmap=cmap, origin='upper', vmin=-0.5, vmax=M-0.5)
        axes[0, 1].set_title("Reproduction ($F_{max}$) | 1970-1975", fontsize=14)
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(_to_grid(data_dict['S_late']), cmap=cmap, origin='upper', vmin=-0.5, vmax=M-0.5)
        axes[1, 0].set_title("Survival ($S_a, S_j$) | 2018-2023", fontsize=14)
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(_to_grid(data_dict['R_late']), cmap=cmap, origin='upper', vmin=-0.5, vmax=M-0.5)
        axes[1, 1].set_title("Reproduction ($F_{max}$) | 2018-2023", fontsize=14)
        axes[1, 1].axis('off')
        
        fig.legend(handles=patches, bbox_to_anchor=(0.5, 0.02), loc='lower center', 
                   ncol=min(M, 6), title=f"Primary Environmental {title_prefix}", fontsize=12, title_fontsize=14)
        
        plt.suptitle(f"Spatial {title_prefix} Factors: Historical vs. Modern Era", fontsize=18, y=0.96)
        plt.tight_layout(rect=[0, 0.08, 1, 0.95])
        plt.savefig(os.path.join(output_dir, filename), dpi=300)
        plt.close()

    _render_2x2_grid(limits, "Limiting", "3_limiting_factors_temporal.png")
    _render_2x2_grid(drivers, "Driving", "4_driving_factors_temporal.png")

# --- 4. CONTINENTAL AGGREGATE IMPACTS (VIOLINS ONLY) ---
def plot_continental_aggregates(sim, data, M, z_names, output_dir):
    print("   -> Calculating Continental Impact Distributions (Violins)...")
    
    Z_gathered = data['Z_gathered']
    if Z_gathered.ndim >= 3:
        Z_modern = np.mean(Z_gathered[-10:], axis=0) 
    else:
        Z_modern = Z_gathered
        
    bs = sim['beta_s']
    br = sim['beta_r']
    
    cont_s = Z_modern * bs
    cont_r = Z_modern * br
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    y_pos = np.arange(M)
    
    def plot_violins(ax, data_matrix, title, color):
        parts = ax.violinplot(
            dataset=[data_matrix[:, i] for i in range(M)],
            positions=y_pos,
            vert=False,
            widths=0.7,
            showmeans=True,
            showextrema=False
        )
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_alpha(0.5)
        parts['cmeans'].set_color('black')
        ax.axvline(0, color='red', linestyle='--', alpha=0.5)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Linear Contribution ($Z_k \\times \\beta_k$)")

    plot_violins(axes[0], cont_s, "Spatial Impact on Survival", "dodgerblue")
    plot_violins(axes[1], cont_r, "Spatial Impact on Reproduction", "darkorange")
    
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(z_names, fontsize=12)
    plt.suptitle("Continental Environmental Impacts (Modern Era Average)", fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "4_continental_impacts_violins.png"), dpi=300)
    plt.close()


# --- 5. SPATIAL R0 ABLATION (THE KEYSTONE MAP) ---
def plot_R0_ablation_keystone(sim, data, M, z_names, output_dir):
    print("   -> Calculating Spatial R0 Ablation and Keystone Features...")
    
    Z_gathered = data['Z_gathered']
    if Z_gathered.ndim >= 3:
        Z_modern = np.mean(Z_gathered[-10:], axis=0) 
    else:
        Z_modern = Z_gathered
        
    bs = sim['beta_s']
    br = sim['beta_r']
    
    alpha_a = sim['alpha_a']
    alpha_j = sim['alpha_j']
    alpha_f = sim['alpha_f']
    
    gamma_a = float(jnn.softplus(sim['gamma_a_raw']))
    gamma_j = gamma_a + sim['gamma_j_diff']
    gamma_f = float(jnn.softplus(sim['gamma_f_raw']))
    
    # In plot_R0_ablation_keystone:
    def calc_R0(Z_matrix):
        H_s = Z_matrix @ bs
        H_r = Z_matrix @ br
        S_a = jnn.sigmoid(alpha_a + gamma_a * H_s)
        S_j = jnn.sigmoid(alpha_j + gamma_j * H_s)
        F_max = jnp.exp(alpha_f + gamma_f * H_r)
        
        # This represents the 'Biotic Potential' without the Allee bottleneck
        return (F_max * S_j) / (1.0 - S_a + 1e-6)

    baseline_R0 = calc_R0(Z_modern)
    
    delta_R0_spatial = np.zeros((M, len(Z_modern)))
    
    for i in range(M):
        Z_ablated = Z_modern.copy()
        Z_ablated[:, i] = 0.0  
        delta_R0_spatial[i] = baseline_R0 - calc_R0(Z_ablated)
        
    net_impact = np.mean(delta_R0_spatial, axis=1)          
    abs_impact = np.mean(np.abs(delta_R0_spatial), axis=1)  
    
    keystone_idx = np.argmax(np.abs(delta_R0_spatial), axis=0)
    
    if keystone_idx.ndim == 1:
        grid_keystone = np.full((data['Ny'], data['Nx']), np.nan)
        grid_keystone[data['land_rows'], data['land_cols']] = keystone_idx
    else:
        grid_keystone = np.where(data['land_mask'] == 1, keystone_idx, np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8), gridspec_kw={'width_ratios': [1, 1.2]})
    
    y_pos = np.arange(M)
    
    axes[0].barh(y_pos, abs_impact, color='lightgray', edgecolor='black', 
                 label='Mean Absolute Impact (Local Volatility)')
    
    colors = ['forestgreen' if val > 0 else 'firebrick' for val in net_impact]
    axes[0].barh(y_pos, net_impact, color=colors, height=0.5, alpha=0.9, 
                 label='Net Continental Impact (Direction)')
    
    axes[0].axvline(0, color='black', linestyle='-')
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(z_names, fontsize=12)
    axes[0].set_xlabel("Change in $R_0$")
    axes[0].set_title("Biological Leverage:\nLocal Volatility vs. Net Continental Impact", fontsize=15)
    axes[0].legend(loc='lower right')

    cmap = plt.get_cmap('tab20', M)
    im = axes[1].imshow(grid_keystone, cmap=cmap, origin='upper', vmin=-0.5, vmax=M-0.5)
    axes[1].axis('off')
    axes[1].set_title("The Keystone Feature Map\n(Feature causing largest absolute $\Delta R_0$ if removed)", fontsize=15)
    
    patches = [mpatches.Patch(color=cmap(i), label=z_names[i]) for i in range(M)]
    axes[1].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', title="Keystone Feature")

    plt.suptitle("Total Environmental Impact on Population Replacement ($R_0$)", fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "5_R0_ablation_keystone.png"), dpi=300)
    plt.close()

# --- MASTER SUITE CONTROLLER ---
def characterize_Z_dependence_suite(sim, data, output_dir, z_names=None):
    print("Initializing Z-Dependence Characterization Suite...")
    os.makedirs(output_dir, exist_ok=True)
    
    Z_gathered = data['Z_gathered'] 
    M = Z_gathered.shape[-1]
    
    if z_names is None:
        z_names = [f"Z_{i}" for i in range(M)]
        
    print("1. Generating Posterior Profiles...")
    plot_posterior_weights(sim, M, z_names, output_dir)
    
    print("2. Generating Demographic Response Curves...")
    for target_idx in range(min(3, M)):
        plot_demographic_response_curves(sim, M, z_names, target_idx, output_dir)
        
    print("3. Generating Environmental Drivers and Limits Maps...")
    plot_environmental_drivers_and_limits(sim, data, M, z_names, output_dir)

    print("4. Generating Continental Impact Distributions (Violins)...")
    plot_continental_aggregates(sim, data, M, z_names, output_dir)
    
    print("5. Generating Spatial R0 Keystone Map & Ablation...")
    plot_R0_ablation_keystone(sim, data, M, z_names, output_dir)
    
    print(f"Suite complete. Visualizations saved to {output_dir}")

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
            
            fit_from_south = R0 * Q_spatial[..., idx_to_north]
            fit_from_north = R0 * Q_spatial[..., idx_to_south]
            
            _render_comparison_plot(
                data, fit_from_south, fit_from_north,
                f"Pioneer Success: Arriving from South vs North ({dist_bin}km)",
                f"comp_from_S_vs_N_{dist_bin}.png", comp_dir, 
                labels_ab=("From South", "From North")
            )
            
            delta_sn = fit_from_south - fit_from_north
            
            plt.figure(figsize=(10, 8))
            v = np.nanpercentile(np.abs(delta_sn), 98)
            masked_delta_sn = np.ma.masked_where(data['land_mask']==0, delta_sn)
            
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

def create_animation(density, obs_grid, years, output_dir, land_mask, filename="evolution_history.mp4", logscale=False):
    print(f"Generating Animation: {filename}...")
    
    fig, (ax_sim, ax_obs) = plt.subplots(1, 2, figsize=(14, 7))
    plt.subplots_adjust(top=0.85)
    
    # 1. Define the dynamic normalization object
    vmax_val = np.nanpercentile(density, 99)
    if logscale:
        # vmin=1e-6 forces the colormap to stretch all the way down to microscopic densities
        norm = mcolors.LogNorm(vmin=1e-6, vmax=vmax_val)
    else:
        norm = mcolors.Normalize(vmin=0, vmax=vmax_val)
    
    # 2. Pass the 'norm' object directly to imshow instead of transforming the data
    im_sim = ax_sim.imshow(density[0], cmap='magma', origin='upper', norm=norm)
    
    land_bg = np.zeros_like(land_mask, dtype=float)
    land_bg[land_mask == 1] = 0.2
    ax_obs.imshow(land_bg, cmap='gray_r', vmin=0, vmax=1, origin='upper', alpha=0.3)
    
    im_obs = ax_obs.imshow(obs_grid[0], cmap='magma', origin='upper', norm=norm, interpolation='none')
    
    title = fig.suptitle(f"Year: {years[0]}", fontsize=16, fontweight='bold')

    def update(frame):
        title.set_text(f"Year: {years[frame]}")
        
        # 3. If logscale, clip the bottom at 1e-8 to prevent log(0) crashes during rendering
        sim_data = np.clip(density[frame], 1e-8, None) if logscale else density[frame]
        obs_data = np.clip(obs_grid[frame], 1e-8, None) if logscale else obs_grid[frame]
        
        im_sim.set_data(sim_data)
        im_obs.set_data(obs_data)
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
    
    # Change these lines in reconstruct_simulation:
    return_sites = [
        "simulated_density", "Sa_flat", "Sj_flat", "Fmax_flat", "K_flat", "Q_flat", "expected_obs",
        "st_weights", "w_env", "L_corr", "dispersal_random",
        "dispersal_logit_intercept", "dispersal_logit_slope",
        "n50_raw", "allee_gamma",
        "alpha_a", "alpha_j", "alpha_f", "alpha_k",
        "gamma_a_raw", "gamma_j_diff", "gamma_f_raw"
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
    print("Rebuilding explicit Age-Structured Pools (Na, Nj)...")
    time = data['time']
    Ny, Nx = data['Ny'], data['Nx']
    row, col = data['inv_location']
    
    # Use the _auto_loc suffix for parameters sampled directly in the model
    inv_pop = jnn.softplus(params['inv_eta_auto_loc'])
    
    # Pull the pre-scaled, pre-transformed gamma directly from the sim dict
    # This was saved via numpyro.deterministic("allee_gamma", ...)
    allee_gamma = sim['allee_gamma'] 
    
    # Pull dispersal parameters (standardizing the suffix usage)
    disp_log_int = params['dispersal_logit_intercept_auto_loc']
    disp_log_slope = params['dispersal_logit_slope_auto_loc']

    def step(pools, t):
        N_a, N_j = pools
        
        # 1. Invasion logic
        k = t - data['inv_timestep']
        is_invading = (k >= 0) & (k < inv_pop.shape[0])
        val = jnp.where(is_invading, inv_pop[jnp.clip(k, 0, inv_pop.shape[0]-1)], 0.0)
        N_a = N_a.at[row, col].add(val)

        # 2. Scatter params (Pulling from the reconstructed sim fields)
        Sa_g = jnp.zeros((Ny, Nx)).at[data['land_rows'], data['land_cols']].set(sim['Sa_flat'][t])
        Sj_g = jnp.zeros((Ny, Nx)).at[data['land_rows'], data['land_cols']].set(sim['Sj_flat'][t])
        Fmax_g = jnp.zeros((Ny, Nx)).at[data['land_rows'], data['land_cols']].set(sim['Fmax_flat'][t])
        K_g = jnp.zeros((Ny, Nx)).at[data['land_rows'], data['land_cols']].set(sim['K_flat'][t])
        
        Q_t = sim['Q_flat'][t]
        Q_temp = jnp.zeros((Ny, Nx, Q_t.shape[-1])).at[data['land_rows'], data['land_cols'], :].set(Q_t)
        Q_g = Q_temp.transpose(2, 0, 1)

        # 3. Dispersal
        N_a_post, N_j_post = dispersal_step_age_structured(
            N_a, N_j, K_g, 
            disp_log_int, disp_log_slope, 0.8,
            data['adult_edge_correction'], data['juvenile_edge_correction_stack'],
            data['adult_fft_kernel'], data['juvenile_fft_kernel_stack'],
            Q_grid=Q_g, eps=1e-6
        )
        
        # 4. Survival & Reproduction (The Allee-corrected step)
        N_a_new, N_j_new = reproduction_age_structured(
            N_a_post, N_j_post, Sa_g, Sj_g, Fmax_g, K_g, 
            allee_gamma, # Now pre-scaled and pre-transformed
            eps=1e-12
        )
        
        N_a_new = jnp.maximum(N_a_new * data['land_mask'], 0.0)
        N_j_new = jnp.maximum(N_j_new * data['land_mask'], 0.0)
        
        return (N_a_new, N_j_new), (N_a_new, N_j_new)

    init_N_a = data['initpop_latent'] * 0.5
    init_N_j = data['initpop_latent'] * 0.5
    
    _, (Na_grid, Nj_grid) = lax.scan(step, (init_N_a, init_N_j), jnp.arange(time))
    return np.array(Na_grid), np.array(Nj_grid)

# --- VISUALIZATIONS ---

def plot_demographic_timeseries(data, years, Sa_grid, Sj_grid, Fmax_grid, K_grid, output_dir):
    print("Generating Demographic Trajectories Over Time...")
    land_mask = data['land_mask']
    scalar = data.get('pop_scalar', 1.0)
    
    valid_pixels = land_mask == 1
    
    Sa_trend = np.mean(Sa_grid[:, valid_pixels], axis=1)
    Sj_trend = np.mean(Sj_grid[:, valid_pixels], axis=1)
    Fmax_trend = np.mean(Fmax_grid[:, valid_pixels], axis=1)
    K_trend = np.mean(K_grid[:, valid_pixels], axis=1) * scalar
    
    R0_trend = (Fmax_trend * Sj_trend) / (1.0 - Sa_trend + 1e-6)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    axes[0].plot(years, Sa_trend, label='Adult Survival ($S_a$)', color='darkblue', lw=2)
    axes[0].plot(years, Sj_trend, label='Juvenile Survival ($S_j$)', color='dodgerblue', lw=2)
    axes[0].set_ylabel("Mean Probability")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(years, Fmax_trend, label='Max Fecundity ($F_{max}$)', color='forestgreen', lw=2)
    axes[1].plot(years, R0_trend, label='Expected Replacement ($R_0$)', color='darkorange', lw=2)
    axes[1].axhline(1.0, color='red', linestyle='--', alpha=0.5, label='Stable Threshold')
    axes[1].set_ylabel("Mean Rate")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(years, K_trend, color='purple', lw=2)
    axes[2].set_ylabel("Mean Carrying Capacity ($K$)")
    axes[2].set_xlabel("Year")
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle("Continental Demographic Trends (1960-2023)", fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "trend_demographics_timeseries.png"), dpi=200)
    plt.close()

def plot_habitat_shift_map(data, Sa_grid, Sj_grid, Fmax_grid, output_dir):
    print("Mapping Historical Habitat Shifts (First vs. Last Decade)...")
    land_mask = data['land_mask']
    
    R0_grid = (Fmax_grid * Sj_grid) / (1.0 - Sa_grid + 1e-6)
    
    early_R0 = np.mean(R0_grid[:10], axis=0)
    late_R0 = np.mean(R0_grid[-10:], axis=0)
    
    delta_R0 = late_R0 - early_R0
    masked_delta = np.ma.masked_where(land_mask == 0, delta_R0)
    
    v = np.nanpercentile(np.abs(masked_delta), 98) 
    
    plt.figure(figsize=(10, 8))
    cmap = plt.cm.get_cmap('PuOr', 11) 
    im = plt.imshow(masked_delta, cmap=cmap, vmin=-v, vmax=v, origin='upper')
    
    plt.colorbar(im, label="Change in Replacement Rate ($\Delta R_0$)")
    plt.title("Ecological Shift: Habitat Degradation vs Improvement\n(Last Decade vs First Decade)")
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, "map_habitat_shift_R0.png"), dpi=200)
    plt.close()

def explore_manifold_correlation(sim, Sj_grid, Fmax_grid, output_dir, land_mask):
    print("Exploring 2D Manifold Correlations...")
    
    L_corr = sim['L_corr']
    corr_matrix = L_corr @ L_corr.T
    learned_rho = corr_matrix[0, 1]
        
    w_env = sim['w_env']
    beta_s = w_env[:, 0]
    beta_r = w_env[:, 1]
    weight_corr = np.corrcoef(beta_s, beta_r)[0, 1]
    
    Sj_avg = np.mean(Sj_grid, axis=0)
    Fmax_avg = np.mean(Fmax_grid, axis=0)
    
    valid_pixels = land_mask == 1
    Sj_vals = Sj_avg[valid_pixels]
    Fmax_vals = Fmax_avg[valid_pixels]
    
    fig = plt.figure(figsize=(12, 6))
    
    ax_text = fig.add_subplot(121)
    ax_text.axis('off')
    info_text = (
        f"Manifold Diagnostics\n\n"
        f"Learned LKJ Correlation (ρ): {learned_rho:.3f}\n"
        f"Empirical Weight Corr:     {weight_corr:.3f}\n\n"
    )
    
    if learned_rho < 0:
        info_text += "Conclusion: The model explicitly learned a\nNEGATIVE correlation. It believes environments\nthat are good for survival are structurally bad\nfor reproduction."
    elif learned_rho < 0.3:
        info_text += "Conclusion: The model learned near-zero\ncorrelation. Survival and Reproduction operate\non totally independent environmental axes."
    else:
        info_text += "Conclusion: The model learned a positive\ncorrelation, but spatial deviations are likely\ndriving local trade-offs."
        
    ax_text.text(0.05, 0.5, info_text, fontsize=14, va='center', ha='left', family='monospace')
    
    ax_scatter = fig.add_subplot(122)
    hb = ax_scatter.hexbin(Sj_vals, Fmax_vals, gridsize=40, cmap='inferno', mincnt=1)
    ax_scatter.set_xlabel("Average Juvenile Survival ($S_j$)")
    ax_scatter.set_ylabel("Average Max Fecundity ($F_{max}$)")
    ax_scatter.set_title("Spatial Trade-off: Productivity vs. Survival")
    plt.colorbar(hb, ax=ax_scatter, label="Number of Pixels")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "diagnostics_manifold_correlation.png"), dpi=200)
    plt.close()

def plot_demographic_baselines(data, Sa_grid, Sj_grid, Fmax_grid, K_grid, output_dir):
    print("Generating Age-Structured Demographic Baselines...")
    land_mask = data['land_mask']
    
    scalar = data.get('pop_scalar', 1.0)
    
    Sa_avg = np.mean(Sa_grid, axis=0)
    Sj_avg = np.mean(Sj_grid, axis=0)
    Fmax_avg = np.mean(Fmax_grid, axis=0)
    
    K_avg = np.mean(K_grid, axis=0) * scalar
    
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
            cbar_label = "Log(1 + Real Expected Birds)"
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

    snapshots = [1970, 1990, 2010, 2020]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    for ax, year in zip(axes.flatten(), snapshots):
        t_idx = np.where(years == year)[0][0]
        na = Na_grid[t_idx]
        nj = Nj_grid[t_idx]
        
        valid_pop = (na + nj) > 0.01
        rho_spatial = np.where(valid_pop, nj / (na + nj + 1e-9), np.nan)
        rho_masked = np.ma.masked_where((land_mask == 0) | ~valid_pop, rho_spatial)
        
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
    
    R0 = (Fmax_avg * Sj_avg) / (1.0 - Sa_avg + 1e-6)
    R0_masked = np.ma.masked_where(land_mask == 0, R0)
    
    plt.figure(figsize=(10, 8))
    binary_map = (R0_masked > 1.0).astype(float)
    cmap_binary = mcolors.ListedColormap(['#d73027', '#4575b4']) 
    plt.imshow(binary_map, cmap=cmap_binary, origin='upper')
    
    legend_elements = [
        mpatches.Patch(color='#4575b4', label='Source ($R_0 > 1$)'),
        mpatches.Patch(color='#d73027', label='Sink ($R_0 \leq 1$)')
    ]
    plt.legend(handles=legend_elements, loc='lower right', frameon=True)
    plt.title("Binary Source-Sink Landscape")
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, "analysis_source_sink_binary.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(10, 8))
    upper_thresh, lower_thresh = 1.1, 0.9
    
    ternary_map = np.zeros_like(R0_masked)
    ternary_map = np.where(R0_masked > upper_thresh, 2, ternary_map)
    ternary_map = np.where((R0_masked >= lower_thresh) & (R0_masked <= upper_thresh), 1, ternary_map)
    ternary_map = np.where(R0_masked < lower_thresh, 0, ternary_map)
    
    ternary_masked = np.ma.masked_where(land_mask == 0, ternary_map)
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

    print("Calculating true path-integrated outgoing risk...")
    
    if Q_grid.ndim == 4:
        Q_spatial = np.mean(Q_grid, axis=0) # (Ny, Nx, K)
    else:
        Q_spatial = Q_grid
        
    k_stack_fft = data['juvenile_fft_kernel_stack'] # (K, Ly, Lx)
    edge_corr = data['juvenile_edge_correction_stack'] # (K, Ny, Nx)
    
    Ny, Nx, num_k = Q_spatial.shape
    Ly, Lx = k_stack_fft.shape[1], k_stack_fft.shape[2]
    pad_y, pad_x = Ly - Ny, Lx - Nx
    
    expected_survivals = []
    
    for k in range(num_k):
        Q_k = Q_spatial[..., k] * land_mask
        Q_k_pad = jnp.pad(Q_k, ((0, pad_y), (0, pad_x)), mode='constant', constant_values=0.0)
        
        Q_k_fft = jnp.fft.fft2(Q_k_pad)
        cross_corr = jnp.real(jnp.fft.ifft2(Q_k_fft * jnp.conj(k_stack_fft[k])))
        
        Q_k_pulled = cross_corr[:Ny, :Nx]
        Q_k_expected = Q_k_pulled / (edge_corr[k] + 1e-6)
        expected_survivals.append(Q_k_expected)
        
    avg_expected_survival = jnp.mean(jnp.stack(expected_survivals, axis=0), axis=0)
    mortality = 1.0 - avg_expected_survival
    risk_masked = np.ma.masked_where(land_mask == 0, mortality)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(risk_masked, cmap='YlOrRd', origin='upper', vmin=0, vmax=1)
    plt.title("Juvenile Migrant Risk\n(Average Mortality for Dispersers Leaving Pixel)")
    plt.colorbar(label="Probability of Death During Dispersal")
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, "analysis_migrant_risk.png"), dpi=200)
    plt.close()

def plot_spatial_residuals(obs_grid, density, output_dir, land_mask):
    print("Generating Spatial Residual Map...")
    
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
    plt.title("Spatial Residuals (Average across Time)\nRed = Under-predicted | Blue = Over-predicted")
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, "diagnostics_spatial_residuals.png"), dpi=200)
    plt.close()

def diagnose_st_weights(sim, data, output_dir):
    print("Diagnosing Spatio-Temporal Regularization (Spatial Confounding)...")
    st_weights = sim['st_weights']
    
    beta_s = sim['w_env'][:, 0]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].hist(st_weights, bins=50, color='purple', edgecolor='black', log=True)
    axes[0].set_title("Distribution of Spatio-Temporal Weights\n(Healthy = Massive spike at 0)")
    axes[0].set_xlabel("Learned Weight Value")
    axes[0].set_ylabel("Frequency (Log Scale)")
    
    t_idx = data['time'] // 2
    
    z_t = np.array(data['Z_gathered'][t_idx]) 
    st_basis_t = np.array(data['st_basis'][:, t_idx, :]) 
    
    H_env = np.dot(z_t, beta_s)                     
    H_st = np.dot(st_basis_t.T, st_weights)         
    
    std_env = np.std(H_env)
    std_st = np.std(H_st)
    
    bars = axes[1].bar(
        ['Environmental Features\n(Climate & Land)', 'Spatio-Temporal Noise\n(The Escape Hatch)'], 
        [std_env, std_st], 
        color=['forestgreen', 'gray']
    )
    axes[1].set_title("Variance Contribution to Latent Manifold")
    axes[1].set_ylabel("Standard Deviation of Spatial Field")
    
    ratio = std_st / (std_env + 1e-9)
    axes[1].text(0.5, max(std_env, std_st) * 0.9, f"Noise-to-Signal Ratio: {ratio:.2f}x", 
                 ha='center', va='top', fontsize=12, bbox=dict(facecolor='white', alpha=0.9))
                 
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "diagnostics_st_weights_impact.png"), dpi=200)
    plt.close()

# --- MAIN ---

def plot_results():
    data = load_data_to_gpu(INPUT_DIR, precision=PRECISION)
    with open(os.path.join(RESULT_DIR, "map_params.pkl"), 'rb') as f: params = pickle.load(f)

    # --- Load Kernel Labels ---
    PATH_INTEGRATION_DIR = "/home/breallis/processed_data/datasets/latent_avian_path_diagnostics"
    disp_files = glob.glob(os.path.join(PATH_INTEGRATION_DIR, "Z_disp_*.npz"))
    
    if not disp_files:
        print("Warning: No Z_disp files found. Labels will be generic.")
        K_kernels = data['juvenile_fft_kernel_stack'].shape[0]
        labels = [f"Kernel_{i}" for i in range(K_kernels)] 
    else:
        with np.load(disp_files[0]) as loader:
            labels = [str(lbl) for lbl in loader['labels']]
    # --------------------------

    # 1. Standard Run
    sim = reconstruct_simulation(data, params)
    
    # 2. Counter-Factual Run (No Invasion)
    params_no_inv = params.copy()
    inv_key = 'inv_eta_auto_loc' 
    if inv_key in params_no_inv:
        params_no_inv[inv_key] = jnp.full_like(params_no_inv[inv_key], -100.0)
    else:
        print(f"Key {inv_key} not found. Available keys: {list(params_no_inv.keys())}")

    sim_cf = reconstruct_simulation(data, params_no_inv)

    # 3. Diagnostic Counter-Factual (No Invasion + No Spatial Noise)
    params_diagnostic = params.copy()
    
    if inv_key in params_diagnostic:
        params_diagnostic[inv_key] = jnp.full_like(params_diagnostic[inv_key], -100.0)
        
    params_diagnostic['st_weights_auto_loc'] = jnp.zeros_like(params_diagnostic['st_weights_auto_loc'])
    
    sim_diagnostic = reconstruct_simulation(data, params_diagnostic)
    
    Ny, Nx = data['Ny'], data['Nx']
    rows, cols = data['land_rows'], data['land_cols']
    years = data['years']
    
    density = np.array(sim['simulated_density']) * data['pop_scalar']
    density_cf = np.array(sim_cf['simulated_density']) * data['pop_scalar']
    density_cf_diag = np.array(sim_diagnostic['simulated_density']) * data['pop_scalar']
    
    Sa_grid = scatter_to_grid_robust(np.array(sim['Sa_flat']), rows, cols, (Ny, Nx))
    Sj_grid = scatter_to_grid_robust(np.array(sim['Sj_flat']), rows, cols, (Ny, Nx))
    Fmax_grid = scatter_to_grid_robust(np.array(sim['Fmax_flat']), rows, cols, (Ny, Nx))
    K_grid = scatter_to_grid_robust(np.array(sim['K_flat']), rows, cols, (Ny, Nx))
    Q_grid = scatter_to_grid_robust(np.array(sim['Q_flat']), rows, cols, (Ny, Nx))
    
    obs_grid = scatter_observations_to_grid(data['observed_results'], data['obs_time_indices'], data['obs_rows'], data['obs_cols'], (Ny, Nx), data['time'])

    start_idx = np.where(years >= 1960)[0][0]
    
    Na_grid, Nj_grid = rebuild_age_pools(sim, data, params)

    # --- Execute Diagnostics ---

    # 1. Cleanly map the exact point estimates for the environmental profiles
    sim['beta_s'] = sim['w_env'][:, 0]
    sim['beta_r'] = sim['w_env'][:, 1]
    
    # 2. Call the suite.
    characterize_Z_dependence_suite(sim, data, OUTPUT_PLOT_DIR)
    # ----------------------
    
    create_animation(density_cf_diag, obs_grid, years, OUTPUT_PLOT_DIR, data['land_mask'], "evolution_counterfactual_diagnostic_logscale.mp4", logscale=True)
    create_animation(density_cf_diag, obs_grid, years, OUTPUT_PLOT_DIR, data['land_mask'], "evolution_counterfactual_diagnostic.mp4")
    create_animation(density, obs_grid, years, OUTPUT_PLOT_DIR, data['land_mask'])
    create_animation(density_cf, obs_grid, years, OUTPUT_PLOT_DIR, data['land_mask'], "evolution_counterfactual.mp4")

    diagnose_st_weights(sim, data, OUTPUT_PLOT_DIR)
    plot_demographic_timeseries(data, years[start_idx:], Sa_grid[start_idx:], Sj_grid[start_idx:], Fmax_grid[start_idx:], K_grid[start_idx:], OUTPUT_PLOT_DIR)
    plot_habitat_shift_map(data, Sa_grid[start_idx:], Sj_grid[start_idx:], Fmax_grid[start_idx:], OUTPUT_PLOT_DIR)

    explore_manifold_correlation(sim, Sj_grid[start_idx:], Fmax_grid[start_idx:], OUTPUT_PLOT_DIR, data['land_mask'])
    plot_spatial_residuals(obs_grid, density, OUTPUT_PLOT_DIR, data['land_mask'])
    plot_directional_asymmetry(data, Sa_grid[start_idx:], Sj_grid[start_idx:], Fmax_grid[start_idx:], Q_grid[start_idx:], labels, OUTPUT_PLOT_DIR)

    plot_demographic_baselines(data, Sa_grid[start_idx:], Sj_grid[start_idx:], Fmax_grid[start_idx:], K_grid[start_idx:], OUTPUT_PLOT_DIR)
    plot_age_structure_dynamics(Na_grid, Nj_grid, years, OUTPUT_PLOT_DIR, data['land_mask'])
    analyze_source_sink_mortality(data, Sa_grid[start_idx:], Sj_grid[start_idx:], Fmax_grid[start_idx:], Q_grid[start_idx:], OUTPUT_PLOT_DIR)

    print(f"All diagnostics complete. Results at: {OUTPUT_PLOT_DIR}")

if __name__ == "__main__":
    plot_results()