import jax.numpy as jnp
import jax.nn as jnn
from jax import lax, checkpoint

def project_and_scatter_age_structured(
    time, Ny, Nx, 
    land_rows, land_cols,
    Z_gathered, Z_disp_gathered,
    st_basis, st_weights, 
    beta_s,           # 1D feature weights for Survival Suitability (Shape: M)
    beta_r,           # 1D feature weights for Reproductive Suitability (Shape: M)
    alpha_a, gamma_a, # Adult survival intercept & slope
    alpha_j, gamma_j, # Juvenile survival intercept & slope
    alpha_f, gamma_f, # Max fecundity intercept & slope
    alpha_k, gamma_k  # Carrying capacity intercept & slope
):
    # We use checkpointing to tell JAX: "Don't store the huge intermediate 
    # activations of this function for the backward pass. Re-compute them."
    @checkpoint
    def process_year(carry, t_idx):
        # 1. Pull slices from CPU RAM -> GPU VRAM
        z_t = jnp.take(Z_gathered, t_idx, axis=0)
        z_disp_t = jnp.take(Z_disp_gathered, t_idx, axis=0)
        st_basis_t = jnp.take(st_basis, t_idx, axis=1) 
        
        # Spatio-temporal random effects (Baseline noise applied to both fields)
        z_smooth = jnp.dot(st_basis_t.T, st_weights)
        
        # 2. Compute the 2D Correlated Habitat Manifolds (H_s and H_r)
        H_s_local = jnp.dot(z_t, beta_s) + z_smooth
        H_r_local = jnp.dot(z_t, beta_r) + z_smooth
        
        # 3. Path-Integrated Survival Suitability
        # z_disp_t is (N_land, K_kernels, M) -> dot with beta_s (M,) gives (N_land, K_kernels)
        H_s_disp = jnp.dot(z_disp_t, beta_s)
        
        # 4. Map H_s and H_r to Demographic Rates using Intercepts and Slopes
        # Survival listens to H_s
        S_a_val = jnn.sigmoid(alpha_a + gamma_a * H_s_local)
        S_j_val = jnn.sigmoid(alpha_j + gamma_j * H_s_local)
        
        # Reproduction listens to H_r
        F_max_val = jnp.exp(alpha_f + gamma_f * H_r_local)
        K_val = jnp.exp(alpha_k + gamma_k * H_r_local)
        
        # 5. Map Path Habitat (H_s) to Journey Survival (Q) using juvenile rules
        # This perfectly links movement mortality to local survival mortality
        Q_val = jnn.sigmoid(alpha_j + gamma_j * H_s_disp)
        
        return None, (S_a_val, S_j_val, F_max_val, K_val, Q_val)

    # We scan over the range of time indices
    t_indices = jnp.arange(time)
    _, (Sa_flat, Sj_flat, Fmax_flat, K_flat, Q_flat) = lax.scan(process_year, None, t_indices)
    
    return Sa_flat, Sj_flat, Fmax_flat, K_flat, Q_flat