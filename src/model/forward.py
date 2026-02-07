import jax.numpy as jnp
import jax.nn as jnn
from jax import lax
import jax

# --- SAFETY HELPERS ---
def clip_safe(x, min_val=-10.0, max_val=10.0):
    return jnp.clip(x, min_val, max_val)

def reproduction_safe(N0, r, K, allee_scalar, allee_intercept, eps=1e-12):
    # Clamp Growth Rate
    r_safe = clip_safe(r, -5.0, 5.0)
    K_safe = jnp.maximum(K, 0.1)
    
    g = jnp.exp(r_safe)
    denom = K_safe + N0 * jnp.expm1(r_safe) + eps 
    
    # Beverton-Holt update
    pop_new = g * N0 * K_safe / denom
    
    # Allee effect
    factor = jnn.sigmoid(N0 * allee_scalar + allee_intercept)
    return factor * pop_new

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
        # 1. Boost Source Mass to conserve probability on land
        # land_fraction_map contains values 0.0 to 1.0 (Fraction of kernel on land).
        # We divide by it to scale up the source density: Pop' = Pop / Fraction
        boosted_source = juvenile_dispersers / (land_fraction_map + eps)
        
        # 2. Convolve the BOOSTED population with the kernel
        # rightpad_convolution handles the padding and FFT/IFFT.
        settled = rightpad_convolution(boosted_source, kernel_fft)
        
        return settled

    # vmap over the stack of kernels (axis 0) and correction maps (axis 0)
    # This runs the logic above K times in parallel
    potential_settlers = jax.vmap(single_kernel_prop)(
        juvenile_fft_kernels, 
        juvenile_edge_correction_stack
    )
    
    # potential_settlers shape is now (K, Ny, Nx)
    
    # 3. Apply Path Survival (Q)
    # Q broadcasts from (Ny, Nx) to (K, Ny, Nx)
    successful_settlers = potential_settlers * Q
    
    # 4. Sum over all kernels to get total settled population
    return jnp.sum(successful_settlers, axis=0)

def sliding_window_rearrange_with_mask(
    arriving_mass, 
    established_mass, 
    K, 
    land_mask, 
    window_size=5, 
    alpha=5.0, 
    eps=1e-6
):
    """
    Differentiable habitat selection that respects the ocean/land boundary.
    """
    # 1. Clean Inputs
    K_safe = jnp.maximum(K, eps)
    land_binary = land_mask.astype(jnp.float32)
    
    # 2. Calculate Soft Vacancy and Apply Mask
    # relative_vacancy: 1.0 (empty) to 0.0 (full)
    relative_vacancy = (K_safe - established_mass) / K_safe
    v_soft = jnn.sigmoid(alpha * relative_vacancy)
    
    # CRITICAL: Ocean pixels have ZERO attraction
    v_soft = v_soft * land_binary
    
    # 3. Neighborhood Context (Accounting for the Coastline)
    sort_kernel = jnp.ones((window_size, window_size))
    
    # Sum of attraction in the neighborhood
    v_neighborhood_sum = jax.scipy.signal.convolve2d(v_soft, sort_kernel, mode='same')
    
    # Count of LAND pixels in the neighborhood (The Ocean Fix)
    # This prevents coastal pixels from having an artificially low denominator
    land_in_neighborhood = jax.scipy.signal.convolve2d(land_binary, sort_kernel, mode='same')
    
    # 4. Pull Factor (Local Attraction vs. Local *Land* Average)
    v_avg = v_neighborhood_sum / (land_in_neighborhood + eps)
    pull_factor = v_soft / (v_avg + eps)
    
    # 5. Redistribute
    # Arriving mass * pull_factor ensures birds only land on available land
    return arriving_mass * pull_factor

def dispersal_step(
    pop, K, 
    dispersal_logit_intercept, dispersal_logit_slope, target_fraction,
    dispersal_intercept, dispersal_random,
    adult_edge_correction, juvenile_edge_correction_stack,
    adult_fft_kernel, juvenile_fft_kernel_stack,
    land_mask,
    Q_grid, s_grid, 
    eps=1e-6
):
    # Ensure valid inputs
    pop = jnp.maximum(pop, 0.0)
    
    # -----------------------------
    # 1. Total Dispersal Probability (Density Dependent)
    # -----------------------------
    K_safe = K + eps
    z_total = dispersal_logit_intercept + dispersal_logit_slope * (pop / K_safe - target_fraction)
    # Safety clip logit
    z_total = clip_safe(z_total) 
    p_total = jnn.sigmoid(z_total)
    
    # Split Biomass into Dispersers vs Stayers
    total_dispersers = pop * p_total
    total_stayers = pop * (1.0 - p_total)

    # -----------------------------
    # 2. Juvenile vs Adult Split (Composition)
    # -----------------------------
    # We use this ratio to split BOTH the dispersers and the stayers
    z_juvenile = dispersal_intercept + dispersal_random
    z_juvenile = clip_safe(z_juvenile)
    p_juvenile = jnn.sigmoid(z_juvenile) # Fraction of biomass that is Juvenile

    # A. STAYERS POOL
    # Adults: Stay put, Survival = 1.0 (Implicit in r)
    adult_stayers = total_stayers * (1.0 - p_juvenile)
    
    # Juveniles: Stay put, Survival = s (Establishment Cost)
    juvenile_stayers = total_stayers * p_juvenile
    juvenile_stayers_surviving = juvenile_stayers * s_grid
    
    # B. DISPERSERS POOL
    # Adults: Move (Short Distance), Survival = 1.0 (Implicit in r)
    adult_dispersers = total_dispersers * (1.0 - p_juvenile)
    
    adult_boosted = adult_dispersers / (adult_edge_correction + eps)
    adults_final = rightpad_convolution(adult_boosted, adult_fft_kernel)
    
    # Juveniles: Move (Long Distance), Survival = Q (Path Cost)
    juvenile_dispersers = total_dispersers * p_juvenile
    
    juveniles_arriving = juvenile_dispersal_vectorized(
        juvenile_dispersers,
        juvenile_fft_kernel_stack,
        Q_grid,
        juvenile_edge_correction_stack,
        eps
    )
    
    # -----------------------------
    # 3. Sum Total
    # -----------------------------
    # Adult Stayers + Juvenile Survivor Stayers + Adult Movers + Juvenile Movers
    # total_pop = adult_stayers + juvenile_stayers_surviving + adults_final + juveniles_arriving
    
    arriving_mass = juveniles_arriving + adults_final
    established_mass = adult_stayers+juvenile_stayers_surviving

    # arriving_sorted = sliding_window_rearrange_with_mask(
    #     arriving_mass, 
    #     established_mass, 
    #     K, 
    #     land_mask,
    #     window_size=5, 
    #     alpha=5.0
    # )
    
    # return arriving_sorted
    return jnp.maximum(1e-12, arriving_mass+established_mass)

def forward_sim_2d(
    r_flat, K_flat, Q_flat, s_flat, 
    land_rows, land_cols,           
    land_mask,
    adult_fft_kernel, juvenile_fft_kernel_stack,
    adult_edge_correction, juvenile_edge_correction_stack,
    initpop_latent, dispersal_random, inv_pop,
    time, inv_location, inv_timestep,
    dispersal_logit_intercept, dispersal_logit_slope,
    dispersal_intercept, allee_scalar, allee_intercept,
    pseudo_zero, target_fraction=0.8
):
    Ny, Nx = land_mask.shape
    row, col = inv_location
    
    def scatter_t(r_t, K_t, s_t, Q_t):
        r_g = jnp.zeros((Ny, Nx)).at[land_rows, land_cols].set(r_t)
        K_g = jnp.zeros((Ny, Nx)).at[land_rows, land_cols].set(K_t)
        s_g = jnp.zeros((Ny, Nx)).at[land_rows, land_cols].set(s_t)
        
        K_kernels = Q_t.shape[-1]
        Q_temp = jnp.zeros((Ny, Nx, K_kernels)).at[land_rows, land_cols, :].set(Q_t)
        Q_g = Q_temp.transpose(2, 0, 1)
        
        return r_g, K_g, s_g, Q_g

    def step(pop, t):
        # 1. Invasion
        k = t - inv_timestep
        is_invading = (k >= 0) & (k < inv_pop.shape[0])
        val = jnp.where(is_invading, inv_pop[jnp.minimum(jnp.maximum(0, k), inv_pop.shape[0]-1)], 0.0)
        pop = pop.at[row, col].add(val)

        # 2. Scatter Parameters
        r_g, K_g, s_g, Q_g = scatter_t(r_flat[t], K_flat[t], s_flat[t], Q_flat[t])

        # 3. Dispersal (Aligned with Old Logic + Mortality Rules)
        pop = dispersal_step(
            pop=pop, K=K_g,
            dispersal_logit_intercept=dispersal_logit_intercept,
            dispersal_logit_slope=dispersal_logit_slope,
            target_fraction=target_fraction,
            dispersal_intercept=dispersal_intercept,
            dispersal_random=dispersal_random[t],
            adult_edge_correction=adult_edge_correction,
            juvenile_edge_correction_stack=juvenile_edge_correction_stack,
            adult_fft_kernel=adult_fft_kernel,
            juvenile_fft_kernel_stack=juvenile_fft_kernel_stack,
            land_mask=land_mask,
            Q_grid=Q_g, s_grid=s_g, eps=1e-6
        )
        
        # 4. Growth
        pop = reproduction_safe(pop, r_g, K_g, allee_scalar, allee_intercept)
        
        # 5. Mask & Final Clip
        pop = pop * land_mask
        pop = jnp.maximum(pop, 0.0)
        
        return pop, pop

    _, densities = lax.scan(step, initpop_latent, jnp.arange(time))
    return densities