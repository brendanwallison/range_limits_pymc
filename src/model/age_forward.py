import jax.numpy as jnp
import jax.nn as jnn
from jax import lax
import jax

# --- SAFETY HELPERS ---
def clip_safe(x, min_val=-10.0, max_val=10.0):
    return jnp.clip(x, min_val, max_val)

def rightpad(A, Lx, Ly, pad_value=1e-9):
    pad_matrix = jnp.ones((Ly, Lx)) * pad_value
    pad_matrix = pad_matrix.at[:A.shape[0], :A.shape[1]].set(A)
    return pad_matrix

def rightpad_convolution(pop, dispersal_kernel_pad):
    Ly, Lx = dispersal_kernel_pad.shape
    pop_pad = rightpad(pop, Lx, Ly, 1e-9)
    conv = jnp.fft.ifft2(jnp.fft.fft2(pop_pad) * dispersal_kernel_pad)
    return jnp.real(conv)[:pop.shape[0], :pop.shape[1]]

def juvenile_dispersal_vectorized(
    juvenile_dispersers: jnp.ndarray,       
    juvenile_fft_kernels: jnp.ndarray,      
    Q: jnp.ndarray,                         
    juvenile_edge_correction_stack: jnp.ndarray, 
    eps: float = 1e-6
):
    """
    Performs dispersal with SOURCE-BASED edge correction.
    """
    def single_kernel_prop(kernel_fft, land_fraction_map):
        boosted_source = juvenile_dispersers / (land_fraction_map + eps)
        settled = rightpad_convolution(boosted_source, kernel_fft)
        return settled

    potential_settlers = jax.vmap(single_kernel_prop)(
        juvenile_fft_kernels, 
        juvenile_edge_correction_stack
    )
    
    successful_settlers = potential_settlers * Q
    return jnp.sum(successful_settlers, axis=0)

def dispersal_step_age_structured(
    N_a, N_j, K, 
    dispersal_logit_intercept, dispersal_logit_slope, target_fraction,
    adult_edge_correction, juvenile_edge_correction_stack,
    adult_fft_kernel, juvenile_fft_kernel_stack,
    Q_grid,
    eps=1e-6
):
    # Ensure valid inputs
    N_a = jnp.maximum(N_a, 0.0)
    N_j = jnp.maximum(N_j, 0.0)
    N_total = N_a + N_j
    
    # -----------------------------
    # 1. Total Dispersal Probability (Density Dependent)
    # -----------------------------
    K_safe = K + eps
    z_total = dispersal_logit_intercept + dispersal_logit_slope * (N_total / K_safe - target_fraction)
    z_total = clip_safe(z_total) 
    p_total = jnn.sigmoid(z_total)
    
    # -----------------------------
    # 2. Split and Disperse Pools
    # -----------------------------
    
    # A. ADULTS (Short-range, no extra path mortality)
    adult_dispersers = N_a * p_total
    adult_stayers = N_a * (1.0 - p_total)
    
    adult_boosted = adult_dispersers / (adult_edge_correction + eps)
    adult_arriving = rightpad_convolution(adult_boosted, adult_fft_kernel)
    
    N_a_post = adult_stayers + adult_arriving
    
    # B. JUVENILES (Long-range, Q path mortality)
    juvenile_dispersers = N_j * p_total
    juvenile_stayers = N_j * (1.0 - p_total)
    
    juvenile_arriving = juvenile_dispersal_vectorized(
        juvenile_dispersers,
        juvenile_fft_kernel_stack,
        Q_grid,
        juvenile_edge_correction_stack,
        eps
    )
    
    # Note: Q is applied inside the vectorized function to movers.
    # Stayers don't pay path mortality.
    N_j_post = juvenile_stayers + juvenile_arriving

    return N_a_post, N_j_post

def reproduction_age_structured(
    N_a_post, N_j_post, 
    S_a, S_j, F_max, K, 
    allee_scalar, allee_intercept, 
    eps=1e-12
):
    N_total_post = N_a_post + N_j_post
    K_safe = jnp.maximum(K, eps)

    # 1. Calculate Density-Dependent Fecundity (Beverton-Holt style)
    # The c parameter ensures F_eff drops to replacement rate when N == K
    c = (F_max * S_j) / (1.0 - S_a + eps) - 1.0
    c = jnp.maximum(c, 0.0) # Safety clip
    
    F_eff = F_max / (1.0 + c * (N_total_post / K_safe))
    
    # 2. Allee effect (applies to reproduction)
    allee_factor = jnn.sigmoid(N_total_post * allee_scalar + allee_intercept)
    F_actual = F_eff * allee_factor
    
    # 3. State Transitions
    # Adults: Surviving adults + surviving juveniles that matured
    N_a_new = (N_a_post * S_a) + (N_j_post * S_j)
    
    # Juveniles: New offspring produced by surviving adults
    # (Assuming post-breeding census where adults breed after surviving)
    N_j_new = (N_a_post * S_a) * F_actual
    
    return N_a_new, N_j_new

def forward_sim_age_structured(
    Sa_flat, Sj_flat, Fmax_flat, K_flat, Q_flat, 
    land_rows, land_cols,           
    land_mask,
    adult_fft_kernel, juvenile_fft_kernel_stack,
    adult_edge_correction, juvenile_edge_correction_stack,
    initpop_latent, dispersal_random, inv_pop,
    time, inv_location, inv_timestep,
    dispersal_logit_intercept, dispersal_logit_slope,
    allee_scalar, allee_intercept,
    pseudo_zero, target_fraction=0.8
):
    Ny, Nx = land_mask.shape
    row, col = inv_location
    
    # Split the initial total population equally between adults and juveniles
    # (Or use the theoretical stable stage distribution if preferred)
    init_N_a = initpop_latent * 0.5
    init_N_j = initpop_latent * 0.5
    
    def scatter_t(Sa_t, Sj_t, Fmax_t, K_t, Q_t):
        Sa_g = jnp.zeros((Ny, Nx)).at[land_rows, land_cols].set(Sa_t)
        Sj_g = jnp.zeros((Ny, Nx)).at[land_rows, land_cols].set(Sj_t)
        Fmax_g = jnp.zeros((Ny, Nx)).at[land_rows, land_cols].set(Fmax_t)
        K_g = jnp.zeros((Ny, Nx)).at[land_rows, land_cols].set(K_t)
        
        K_kernels = Q_t.shape[-1]
        Q_temp = jnp.zeros((Ny, Nx, K_kernels)).at[land_rows, land_cols, :].set(Q_t)
        Q_g = Q_temp.transpose(2, 0, 1)
        
        return Sa_g, Sj_g, Fmax_g, K_g, Q_g

    def step(pools, t):
        N_a, N_j = pools
        
        # 1. Invasion (Add to adults for simplicity, or split)
        k = t - inv_timestep
        is_invading = (k >= 0) & (k < inv_pop.shape[0])
        val = jnp.where(is_invading, inv_pop[jnp.minimum(jnp.maximum(0, k), inv_pop.shape[0]-1)], 0.0)
        N_a = N_a.at[row, col].add(val)

        # 2. Scatter Parameters
        Sa_g, Sj_g, Fmax_g, K_g, Q_g = scatter_t(
            Sa_flat[t], Sj_flat[t], Fmax_flat[t], K_flat[t], Q_flat[t]
        )

        # 3. Dispersal
        N_a_post, N_j_post = dispersal_step_age_structured(
            N_a, N_j, K_g, 
            dispersal_logit_intercept, dispersal_logit_slope, target_fraction,
            adult_edge_correction, juvenile_edge_correction_stack,
            adult_fft_kernel, juvenile_fft_kernel_stack,
            Q_grid=Q_g, eps=1e-6
        )
        
        # 4. Survival & Reproduction (Age-Structured Update)
        N_a_new, N_j_new = reproduction_age_structured(
            N_a_post, N_j_post, 
            Sa_g, Sj_g, Fmax_g, K_g, 
            allee_scalar, allee_intercept, 
            eps=1e-12
        )
        
        # 5. Mask & Final Clip
        N_a_new = jnp.maximum(N_a_new * land_mask, 0.0)
        N_j_new = jnp.maximum(N_j_new * land_mask, 0.0)
        
        N_total_new = N_a_new + N_j_new
        
        return (N_a_new, N_j_new), N_total_new

    # We carry the tuple of (N_a, N_j), but we only return the combined density for the loss function
    _, total_densities = lax.scan(step, (init_N_a, init_N_j), jnp.arange(time))
    
    return total_densities