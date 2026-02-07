import jax.numpy as jnp
import jax.nn as jnn
import numpyro
import numpyro.distributions as dist

from src.model.fields import project_and_scatter_efficient
from src.model.forward import forward_sim_2d

def sample_priors(anneal=1.0, M_features=None, N_basis=None, time=None):
    priors = {}
    
    # --- 1. BIOLOGICAL SCALARS ---
    priors['r_mean'] = numpyro.sample("r_mean", dist.Normal(0.3, 0.5 * anneal))
    priors['K_raw_mean'] = numpyro.sample("K_raw_mean", dist.Normal(0.5, 0.5 * anneal))
    priors['allee_intercept'] = numpyro.sample("allee_intercept", dist.Normal(0.0, 1.0 * anneal))
    priors['allee_slope'] = jnn.softplus(numpyro.sample("allee_slope_raw", dist.Normal(1.0, 0.5 * anneal)))
    
    # --- 2. DISPERSAL & SURVIVAL SCALARS ---
    priors['dispersal_intercept'] = numpyro.sample("dispersal_intercept", dist.Normal(0., 0.5 * anneal))
    priors['dispersal_logit_intercept'] = numpyro.sample("dispersal_logit_intercept", dist.Normal(2.0, 0.5 * anneal))
    priors['dispersal_logit_slope'] = numpyro.sample("dispersal_logit_slope", dist.Normal(4.0, 0.5 * anneal))
    priors['dispersal_survival_threshold'] = numpyro.sample("dispersal_survival_threshold", dist.Normal(1.0, 0.5 * anneal))
    
    # --- 3. THE SMOOTH LATENT NUDGE (3D Spectral Weights) ---
    # Using Laplace (Double Exponential) for sparsity/shrinkage. 
    # This automatically "mutes" frequencies that don't help explain the data.
    priors['st_weights'] = numpyro.sample("st_weights", 
                                          dist.Laplace(0.0, 1e-6 * anneal).expand([N_basis, 3]))
    
    # --- 4. COREGIONALIZED FIELDS ---
    priors['beta'] = numpyro.sample("beta", dist.Normal(0.0, 0.5).expand([M_features, 3]))
    
    scales = numpyro.sample("scales", dist.HalfNormal(1.0).expand([3]))
    corr_matrix = numpyro.sample("corr_matrix", dist.LKJ(dimension=3, concentration=0.2))
    loadings = jnp.diag(scales) @ jnp.linalg.cholesky(corr_matrix)

    priors['L_r']    = loadings[0, :]
    priors['L_K']    = loadings[1, :]
    priors['l_surv'] = loadings[2, :]
        
    # Temporal Annual Noise (Leave only the dispersal wiggle as a random effect)
    # r_spatial and r_temp are now handled by st_weights
    priors['dispersal_random'] = numpyro.sample("dispersal_random", dist.Normal(0., 0.001 * anneal), sample_shape=(time,)) # Update shape to match 'time'
    
    return priors

def build_model_2d(data, anneal=0.1):
    Nx, Ny = data['Nx'], data['Ny']
    time = data['time']
    land_rows, land_cols = data['land_rows'], data['land_cols']
    M = data['Z_gathered'].shape[-1]
    
    # 1. Sample Parameters (Including the spectral basis count)
    priors = sample_priors(anneal, M, data['N_basis'], time)
    
    inv_pop = jnn.softplus(numpyro.sample(
        "inv_eta", dist.Normal(-1.0, 1.0 * anneal), sample_shape=(data['inv_window'],)
    ))
    allee_scalar = data['pop_scalar'] * priors['allee_slope']

    # 2. Compute Biological Fields
    r_flat, K_flat, Q_flat, s_flat = project_and_scatter_efficient(
        time, Ny, Nx, land_rows, land_cols,
        data['Z_gathered'], data['Z_disp_gathered'], 
        data['st_basis'],       # Pass the BASIS instead of the CUBE
        priors['st_weights'],   # Pass the WEIGHTS
        priors['beta'], 
        priors['L_r'], priors['L_K'], priors['l_surv'],
        priors['r_mean'], priors['K_raw_mean'], priors['dispersal_survival_threshold']
    )
        
    # Save fields for viz
    numpyro.deterministic("r_flat", r_flat)
    numpyro.deterministic("K_flat", K_flat)
    numpyro.deterministic("Q_flat", Q_flat)
    numpyro.deterministic("s_flat", s_flat)
    
    # 3. Forward Simulation
    densities = forward_sim_2d(
        r_flat, K_flat, Q_flat, s_flat, 
        land_rows, land_cols,           
        data['land_mask'],
        data['adult_fft_kernel'], data['juvenile_fft_kernel_stack'],
        data['adult_edge_correction'], data['juvenile_edge_correction_stack'],
        data['initpop_latent'], priors['dispersal_random'], inv_pop,
        time, data['inv_location'], data['inv_timestep'],
        priors['dispersal_logit_intercept'], priors['dispersal_logit_slope'],
        priors['dispersal_intercept'], allee_scalar, priors['allee_intercept'],
        data['pseudo_zero']
    )

    numpyro.deterministic("simulated_density", densities)

    # 4. Likelihood
    t_idx, rows, cols = data["obs_time_indices"], data["obs_rows"], data["obs_cols"]
    
    densities_obs = jnp.maximum(densities[t_idx, rows, cols] * data["pop_scalar"], 1e-6)
    
    numpyro.deterministic("expected_obs", densities_obs)
    numpyro.sample("obs", dist.Poisson(densities_obs), obs=data["observed_results"])