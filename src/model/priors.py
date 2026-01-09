import jax.numpy as jnp
import jax.nn as jnn
import numpyro
import numpyro.distributions as dist
from fields import precompute_r_K, compute_fields_from_Z, compute_Q_from_Z, scatter_Q
from forward import forward_sim_2d

def sample_priors(anneal=0.1, M_features=None, N_land=None, time=None):
    """
    Sample all scalar and vector priors for the model.
    """
    priors = {}
    
    # Biological scalars
    priors['r_mean'] = numpyro.sample("r_mean", dist.Normal(0.0, 1.0 * anneal))
    priors['K_raw_mean'] = numpyro.sample("K_raw_mean", dist.Normal(-1.0, 1.0 * anneal))
    priors['allee_intercept'] = numpyro.sample("allee_intercept", dist.Normal(0, 1.0 * anneal))
    priors['allee_slope'] = jnn.softplus(numpyro.sample("allee_slope_raw", dist.Normal(2.0, 1.0 * anneal)))
    
    # Dispersal scalars
    priors['dispersal_intercept'] = numpyro.sample("dispersal_intercept", dist.Normal(0., 1.0 * anneal))
    priors['dispersal_logit_intercept'] = numpyro.sample("dispersal_logit_intercept", dist.Normal(2.0, 1.0 * anneal))
    priors['dispersal_logit_slope'] = numpyro.sample("dispersal_logit_slope", dist.Normal(4.0, 1.0 * anneal))
    # Note: dispersal_survival_slope removed as Q is now direct regression
    priors['dispersal_survival_threshold'] = numpyro.sample("dispersal_survival_threshold", dist.Normal(-1.0, 1.0 * anneal))
    
    # Z-Regression Weights (Rank-2)
    # Shared between Growth and Dispersal Survival
    priors['beta'] = numpyro.sample("beta", dist.Normal(0.0, 1.0 * anneal).expand([M_features, 2]))
    
    # Coregionalization Matrix L (for Growth r, K)
    priors['L'] = numpyro.sample("L", dist.Normal(0.0, 0.1 * anneal).expand([2, 2]))
    
    # Survival Coregionalization Vector (for Q)
    # Maps the 2 latent factors to survival probability
    priors['l_surv'] = numpyro.sample("l_surv", dist.Normal(0.0, 1.0 * anneal).expand([2]))
    
    # Random Effects
    priors['r_spatial'] = numpyro.sample("r_spatial", dist.Normal(0., 0.01 * anneal), sample_shape=(N_land,))
    priors['r_temp'] = numpyro.sample("r_temp", dist.Normal(0., 0.01 * anneal), sample_shape=(time,))
    priors['dispersal_random'] = numpyro.sample("dispersal_random", dist.Normal(0., 0.01 * anneal), sample_shape=(time,))
    
    return priors

def build_model_2d(data, anneal=0.1):
    Nx = data['Nx']
    Ny = data['Ny']
    time = data['time']
    land_rows = data['land_rows']
    land_cols = data['land_cols']
    N_land = len(land_rows)
    
    # Z: Latent Mercer Feature Cube (time, N_land, M_features)
    # Z_disp: Path-summarized Feature Cube for dispersal
    Z = data['Z']
    Z_disp = data['Z_disp']
    M_features = Z.shape[-1]

    # 1. Sample Priors
    priors = sample_priors(anneal, M_features, N_land, time)
    
    inv_pop = jnn.softplus(numpyro.sample(
        "inv_eta",
        dist.Normal(-5.0, 1.0 * anneal),
        sample_shape=(data['inv_window'],)
    ))

    allee_scalar = data['pop_scalar'] * priors['allee_slope']

    # 2. Compute Growth Fields (r, K) from Z
    r_Z, logK_Z = compute_fields_from_Z(Z, priors['beta'], priors['L'])

    # 3. Compute Survival Field (Q) from Z_disp
    # Note: We skip the FFT convolution here because Z_disp already encodes path info
    Q_flat = compute_Q_from_Z(Z_disp, priors['beta'], priors['l_surv'], priors['dispersal_survival_threshold'])
    
    # 4. Scatter to Arrays
    r_array, K_array = precompute_r_K(
        time=time,
        Ny=Ny,
        Nx=Nx,
        land_rows=land_rows,
        land_cols=land_cols,
        r_mean=priors['r_mean'],
        r_Z=r_Z,
        r_spatial=priors['r_spatial'],
        r_temp=priors['r_temp'],
        K_raw_mean=priors['K_raw_mean'],
        logK_Z=logK_Z
    )
    
    Q_array = scatter_Q(time, Ny, Nx, land_rows, land_cols, Q_flat)
    
    # s_array (local survival) is technically Q_array here since we assume Z_disp 
    # accounts for the path. If we need local s for non-dispersers, we can 
    # use Q_array as a proxy or compute it from Z using the same logic.
    # For now, passing Q_array as s_array.
    s_array = Q_array 

    # 5. Forward Simulation
    densities = forward_sim_2d(
        time=time,
        initial_pop=data['initpop_latent'],
        r_array=r_array,
        K_array=K_array,
        dispersal_logit_intercept=priors['dispersal_logit_intercept'],
        dispersal_logit_slope=priors['dispersal_logit_slope'],
        dispersal_intercept=priors['dispersal_intercept'],
        dispersal_random=priors['dispersal_random'],
        adult_fft_kernel=data['adult_fft_kernel'],
        juvenile_fft_kernel_stack=data['juvenile_fft_kernel_stack'],
        adult_edge_correction=data['adult_edge_correction'],
        juvenile_edge_correction_stack=data['juvenile_edge_correction_stack'],
        Q_array=Q_array,
        s_array=s_array, 
        land_mask=data['land_mask'],
        inv_pop=inv_pop,
        inv_location=data['inv_location'],
        inv_timestep=data['inv_timestep'],
        allee_scalar=allee_scalar,
        allee_intercept=priors['allee_intercept'],
        pseudo_zero=data['pseudo_zero']
    )

    # 6. Likelihood
    t_idx = data["obs_time_indices"]
    rows = data["obs_rows"]
    cols = data["obs_cols"]

    densities_obs = densities[t_idx, rows, cols] * data["pop_scalar"]

    numpyro.sample("obs", dist.Poisson(densities_obs), obs=data["observed_results"])