import jax.numpy as jnp
import jax.nn as jnn
import numpyro
import numpyro.distributions as dist

from src.model.age_fields import project_and_scatter_age_structured
from src.model.age_forward import forward_sim_age_structured

def sample_priors(anneal=1.0, M_features=None, N_basis=None, time=None):
    priors = {}
    
    # --- 1. CORRELATED 2D HABITAT MANIFOLD WEIGHTS ---
    # LKJ Cholesky prior for correlation between Survival and Reproduction environments
    # concentration=2.0 gently pushes the prior toward a diagonal matrix to prevent 
    # assuming perfect correlation before seeing the data.
    L_corr = numpyro.sample("L_corr", dist.LKJCholesky(dimension=2, concentration=2.0))
    w_scale = numpyro.sample("w_scale", dist.HalfNormal(1.0).expand([2]))
    
    # Scale the correlation matrix by the variance
    L_cov = w_scale[..., None] * L_corr
    
    # Draw the correlated weights for all M features
    with numpyro.plate("env_features", M_features):
        w_env = numpyro.sample(
            "w_env", 
            dist.MultivariateNormal(loc=jnp.zeros(2), scale_tril=L_cov)
        )
        
    priors['beta_s'] = w_env[:, 0]  # Survival Suitability Weights
    priors['beta_r'] = w_env[:, 1]  # Reproductive Suitability Weights
    
    # 1D spectral weights (Spatio-temporal random effects)
    priors['st_weights'] = numpyro.sample("st_weights", 
                                          dist.Laplace(0.0, 1e-6 * anneal).expand([N_basis]))
    
    # --- 2. DEMOGRAPHIC INTERCEPTS (Alphas) ---
    # Adult survival baseline > Juvenile survival baseline
    priors['alpha_a'] = numpyro.sample("alpha_a", dist.Normal(1.5, 0.5 * anneal)) # ~80%
    priors['alpha_j'] = numpyro.sample("alpha_j", dist.Normal(-0.5, 0.5 * anneal)) # ~38%
    priors['alpha_f'] = numpyro.sample("alpha_f", dist.Normal(1.0, 0.5 * anneal))  # Fecundity
    priors['alpha_k'] = numpyro.sample("alpha_k", dist.Normal(0.5, 0.5 * anneal))  # Capacity
    
    # --- 3. DEMOGRAPHIC SLOPES (Gammas) ---
    # Enforce positive slopes: better habitat = higher survival/fecundity
    # Enforce Rule 5: Juvenile survival is more sensitive to environment than adult
    gamma_a_raw = numpyro.sample("gamma_a_raw", dist.Normal(0.5, 0.5 * anneal))
    gamma_j_diff = numpyro.sample("gamma_j_diff", dist.HalfNormal(1.0 * anneal))
    
    priors['gamma_a'] = jnn.softplus(gamma_a_raw)
    priors['gamma_j'] = priors['gamma_a'] + gamma_j_diff 
    
    priors['gamma_f'] = jnn.softplus(numpyro.sample("gamma_f_raw", dist.Normal(1.0, 0.5 * anneal)))
    priors['gamma_k'] = jnn.softplus(numpyro.sample("gamma_k_raw", dist.Normal(1.0, 0.5 * anneal)))
    
    # --- 4. ALLEE & DISPERSAL SCALARS ---
    priors['allee_intercept'] = numpyro.sample("allee_intercept", dist.Normal(0.0, 1.0 * anneal))
    priors['allee_slope'] = jnn.softplus(numpyro.sample("allee_slope_raw", dist.Normal(1.0, 0.5 * anneal)))
    
    priors['dispersal_logit_intercept'] = numpyro.sample("dispersal_logit_intercept", dist.Normal(2.0, 0.5 * anneal))
    priors['dispersal_logit_slope'] = numpyro.sample("dispersal_logit_slope", dist.Normal(4.0, 0.5 * anneal))
    
    # Temporal Annual Noise (Maintained for dispersal probability fluctuations)
    priors['dispersal_random'] = numpyro.sample("dispersal_random", dist.Normal(0., 0.001 * anneal), sample_shape=(time,))
    
    return priors


def build_model_2d(data, anneal=0.1):
    Nx, Ny = data['Nx'], data['Ny']
    time = data['time']
    land_rows, land_cols = data['land_rows'], data['land_cols']
    M = data['Z_gathered'].shape[-1]
    
    # 1. Sample Parameters
    priors = sample_priors(anneal, M, data['N_basis'], time)
    
    inv_pop = jnn.softplus(numpyro.sample(
        "inv_eta", dist.Normal(-1.0, 1.0 * anneal), sample_shape=(data['inv_window'],)
    ))
    allee_scalar = data['pop_scalar'] * priors['allee_slope']

    # --- IDENTIFIABILITY CONSTRAINT: 50/50 STABLE AGE STRUCTURE ---
    # Calculate global means based on intercepts
    S_a_mean = jnn.sigmoid(priors['alpha_a'])
    S_j_mean = jnn.sigmoid(priors['alpha_j'])
    F_mean = jnp.exp(priors['alpha_f'])

    # Calculate theoretical stable age structure (dominant eigenvalue of Leslie matrix)
    lambda_mean = (S_a_mean + jnp.sqrt(S_a_mean**2 + 4 * F_mean * S_j_mean)) / 2.0
    juvenile_fraction = F_mean / (F_mean + lambda_mean)

    # Soft constraint pulling the model's global baselines toward 50/50
    numpyro.factor("age_structure_constraint", dist.Normal(0.5, 0.05).log_prob(juvenile_fraction))

    # 2. Compute Biological Fields (2D Manifold -> Demographic Rates)
    # Notice we now pass beta_s and beta_r instead of a single beta_h
    Sa_flat, Sj_flat, Fmax_flat, K_flat, Q_flat = project_and_scatter_age_structured(
        time, Ny, Nx, land_rows, land_cols,
        data['Z_gathered'], data['Z_disp_gathered'], 
        data['st_basis'], priors['st_weights'], 
        priors['beta_s'], priors['beta_r'],  # <--- CHANGED HERE
        priors['alpha_a'], priors['gamma_a'],
        priors['alpha_j'], priors['gamma_j'],
        priors['alpha_f'], priors['gamma_f'],
        priors['alpha_k'], priors['gamma_k']
    )
        
    # Save fields for viz
    numpyro.deterministic("Sa_flat", Sa_flat)
    numpyro.deterministic("Sj_flat", Sj_flat)
    numpyro.deterministic("Fmax_flat", Fmax_flat)
    numpyro.deterministic("K_flat", K_flat)
    numpyro.deterministic("Q_flat", Q_flat)
    
    # 3. Forward Simulation
    densities = forward_sim_age_structured(
        Sa_flat, Sj_flat, Fmax_flat, K_flat, Q_flat, 
        land_rows, land_cols,           
        data['land_mask'],
        data['adult_fft_kernel'], data['juvenile_fft_kernel_stack'],
        data['adult_edge_correction'], data['juvenile_edge_correction_stack'],
        data['initpop_latent'], priors['dispersal_random'], inv_pop,
        time, data['inv_location'], data['inv_timestep'],
        priors['dispersal_logit_intercept'], priors['dispersal_logit_slope'],
        allee_scalar, priors['allee_intercept'],
        data['pseudo_zero']
    )

    numpyro.deterministic("simulated_density", densities)

    # 4. Likelihood
    t_idx, rows, cols = data["obs_time_indices"], data["obs_rows"], data["obs_cols"]
    
    # densities output should be the sum of adult + juvenile (N_total)
    densities_obs = jnp.maximum(densities[t_idx, rows, cols] * data["pop_scalar"], 1e-6)
    
    numpyro.deterministic("expected_obs", densities_obs)
    numpyro.sample("obs", dist.Poisson(densities_obs), obs=data["observed_results"])