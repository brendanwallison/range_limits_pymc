import jax.numpy as jnp
import jax.nn as jnn
from jax import lax, checkpoint

def project_and_scatter_efficient(
    time, Ny, Nx, 
    land_rows, land_cols,
    Z_gathered, Z_disp_gathered,
    st_basis, st_weights, 
    beta, L_r, L_K, l_surv, 
    r_mean, K_raw_mean, dispersal_survival_threshold
):
    # We use checkpointing to tell JAX: "Don't store the huge intermediate 
    # activations of this function for the backward pass. Re-compute them."
    # This is a huge VRAM saver during SVI.
    @checkpoint
    def process_year(carry, t_idx):
        # Use jnp.take to slice the CPU-backed NumPy arrays
        # This correctly handles the JAX Tracers during compilation
        
        # 1. Pull slices from CPU RAM -> GPU VRAM
        # axis=0 corresponds to the 'time' dimension
        z_t = jnp.take(Z_gathered, t_idx, axis=0)
        z_disp_t = jnp.take(Z_disp_gathered, t_idx, axis=0)
        
        # For the basis, we are taking from the middle dimension (Time)
        # Shape is (N_basis, Time, N_land) -> axis=1
        st_basis_t = jnp.take(st_basis, t_idx, axis=1) 
        
        # 2. Rest of the logic remains the same
        z_smooth = jnp.dot(st_basis_t.T, st_weights)
        latents_local = jnp.dot(z_t, beta) + z_smooth
        
        # ... biological projections ...
        
        # 3. Map Latents to Biological Parameters
        r_Z = jnp.dot(latents_local, L_r)
        logK_Z = jnp.dot(latents_local, L_K)
        
        # Survival (s)
        s_logit = jnp.dot(latents_local, l_surv) + dispersal_survival_threshold
        s_val = jnn.sigmoid(s_logit)
        
        # Path Survival (Q)
        latents_disp = jnp.einsum('nkm,mj->nkj', z_disp_t, beta)
        Q_logit = jnp.dot(latents_disp, l_surv) + dispersal_survival_threshold
        Q_val = jnn.sigmoid(Q_logit)
        
        r_val = r_mean + r_Z
        K_val = jnp.exp(K_raw_mean + logK_Z)
        
        return None, (r_val, K_val, Q_val, s_val)

    # We scan over the range of time indices
    t_indices = jnp.arange(time)
    _, (r_flat, K_flat, Q_flat, s_flat) = lax.scan(process_year, None, t_indices)
    
    return r_flat, K_flat, Q_flat, s_flat