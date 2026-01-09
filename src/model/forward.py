import jax.numpy as jnp
import jax.nn as jnn
from jax import lax
import jax

def allee_factor(pop, allee_scalar, allee_intercept):
    """Allee effect using a sigmoid."""
    return jnn.sigmoid(pop * allee_scalar + allee_intercept)

def reproduction_safe(N0, r, K, allee_scalar, allee_intercept, eps=1e-12):
    g = jnp.exp(r)
    denom = K + N0 * jnp.expm1(r) + eps 
    pop_new = g * N0 * K / denom
    return allee_factor(N0, allee_scalar, allee_intercept) * pop_new

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
    juvenile_edge_corrections: jnp.ndarray, 
    Q: jnp.ndarray,                         
    dirichlet_weights: jnp.ndarray = None,
    pad_value: float = 1e-9,
):
    """Vectorized juvenile dispersal."""
    Ny, Nx = juvenile_dispersers.shape
    Nd = juvenile_fft_kernels.shape[0] 

    if dirichlet_weights is None:
        dirichlet_weights = jnp.ones(Nd) / Nd
    dirichlet_weights = dirichlet_weights.reshape(Nd, 1, 1) 
    dirichlet_weights *= Nd 
    pop_split = juvenile_dispersers[None, :, :] * dirichlet_weights * juvenile_edge_corrections 

    Ly, Lx = juvenile_fft_kernels.shape[1:]
    pop_pad = jnp.pad(pop_split, ((0,0), (0, Ly-Ny), (0, Lx-Nx)), constant_values=pad_value) 

    fft_pop = jnp.fft.fft2(pop_pad, axes=(-2,-1))
    conv = jnp.fft.ifft2(fft_pop * juvenile_fft_kernels, axes=(-2,-1))
    conv_real = jnp.real(conv)[:, :Ny, :Nx] 

    # Apply Q (survival). Q is (Ny, Nx), broadcasts to (Nd, Ny, Nx)
    conv_corrected = conv_real * Q  

    pop_after = jnp.sum(conv_corrected, axis=0)

    return pop_after

def dispersal_step(pop, K,
                   dispersal_logit_intercept=0.0,
                   dispersal_logit_slope=10.0,
                   target_fraction=0.8,
                   dispersal_intercept=0.0,
                   dispersal_random=None,
                   adult_edge_correction=1.0,
                   juvenile_edge_correction_stack=1.0,
                   adult_fft_kernel=None,
                   juvenile_fft_kernel_stack=None,
                   Q_array=None,
                   s=None,
                   eps=1e-6,
                   clip_z=30.0,
                   min_p=1e-8):
    """Full dispersal step."""
    
    K_safe = K + eps
    z_total = dispersal_logit_intercept + dispersal_logit_slope * (pop / K_safe - target_fraction)
    z_total = jnp.clip(z_total, -clip_z, clip_z)
    p_total = jnn.sigmoid(z_total)
    p_total = jnp.clip(p_total, min_p, 1 - min_p)
    
    dispersers = p_total * pop
    stayers = pop - dispersers

    if dispersal_random is None:
        dispersal_random = jnp.zeros_like(pop)
        
    z_juvenile = dispersal_intercept + dispersal_random
    z_juvenile = jnp.clip(z_juvenile, -clip_z, clip_z)
    p_juvenile = jnn.sigmoid(z_juvenile)
    p_juvenile = jnp.clip(p_juvenile, min_p, 1 - min_p)

    juvenile_dispersers = dispersers * p_juvenile
    adult_dispersers = dispersers - juvenile_dispersers

    if s is not None:
        # s applies to non-dispersing juveniles
        stayer_survival_factor = 1.0 - p_juvenile * (1.0 - s)
        stayers = stayers * stayer_survival_factor

    adult_update = rightpad_convolution(adult_dispersers * adult_edge_correction, adult_fft_kernel)
    
    juvenile_update = juvenile_dispersal_vectorized(
        juvenile_dispersers=juvenile_dispersers,
        juvenile_fft_kernels = juvenile_fft_kernel_stack, 
        juvenile_edge_corrections=juvenile_edge_correction_stack,
        Q=Q_array,
    )
    pop_new = stayers + adult_update + juvenile_update

    return pop_new

def forward_sim_2d(time, initial_pop, r_array, K_array,
                    dispersal_logit_intercept, dispersal_logit_slope,
                    dispersal_intercept, dispersal_random,
                    adult_fft_kernel, juvenile_fft_kernel_stack,
                    adult_edge_correction, juvenile_edge_correction_stack,
                    Q_array,
                    s_array, 
                    land_mask,
                    inv_pop, inv_location, inv_timestep, 
                    allee_scalar, allee_intercept,
                    pseudo_zero, target_fraction=0.8):
    """2D native simulation."""

    row, col = inv_location
    
    def step(pop, t):
        def invasion_amount(t):
            k = t - inv_timestep
            return jnp.where(
                (k >= 0) & (k < inv_pop.shape[0]),
                inv_pop[k],
                0.0,
            )

        add_val = invasion_amount(t)
        pop = pop.at[row, col].add(add_val)

        r = r_array[t] 
        K = K_array[t] 
        Q = Q_array[t] 
        s = s_array[t]

        pop = dispersal_step(
            pop=pop,
            K=K,
            dispersal_logit_intercept=dispersal_logit_intercept,
            dispersal_logit_slope=dispersal_logit_slope,
            target_fraction=target_fraction,
            dispersal_intercept=dispersal_intercept,
            dispersal_random=dispersal_random[t],
            adult_edge_correction=adult_edge_correction,
            juvenile_edge_correction_stack=juvenile_edge_correction_stack,
            adult_fft_kernel=adult_fft_kernel,
            juvenile_fft_kernel_stack=juvenile_fft_kernel_stack,
            Q_array=Q,
            s=s,
            eps=1e-6,
            clip_z=30.0,
            min_p=1e-8
        )

        pop = land_mask * (reproduction_safe(pop, r, K, allee_scalar, allee_intercept)) + pseudo_zero
        return pop, pop

    _, land_over_time = lax.scan(step, initial_pop, jnp.arange(time))
    return land_over_time