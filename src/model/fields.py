import jax
import jax.numpy as jnp
import jax.nn as jnn
from jax import lax

from .hsgp import hsgp_spd_se, hsgp_field

# ============================================================
# r / K precomputation
# ============================================================

def precompute_r_K(
    time,
    Ny,
    Nx,
    land_rows,
    land_cols,
    r_mean,
    r_bioclim,
    r_pop,
    r_spatial,
    r_temp,
    K_raw_mean,
    logK_bio,
    logK_pop,
):
    N_land = land_rows.shape[0]

    r_pop = r_pop.reshape(time, N_land)
    logK_pop = logK_pop.reshape(time, N_land)

    r_vals = (
        r_mean
        + r_bioclim[None, :]
        + r_spatial[None, :]
        + r_pop
        + r_temp[:, None]
    )

    K_vals = jnp.exp(
        K_raw_mean
        + logK_bio[None, :]
        + logK_pop
    )

    r_array = jnp.zeros((time, Ny, Nx))
    K_array = jnp.zeros((time, Ny, Nx))

    t_idx = jnp.arange(time)[:, None]
    r_array = r_array.at[t_idx, land_rows, land_cols].set(r_vals)
    K_array = K_array.at[t_idx, land_rows, land_cols].set(K_vals)

    return r_array, K_array


# ============================================================
# Survival and Q
# ============================================================

def precompute_s_Q_vectorized(
    r_array,
    juvenile_fft_kernel_stack,
    a,
    r0,
    pad_value=1.0,
):
    time, Ny, Nx = r_array.shape
    Nd, Ly, Lx = juvenile_fft_kernel_stack.shape

    s_array = jnn.sigmoid(a * (r_array - r0))

    s_pad = jnp.pad(
        s_array,
        ((0, 0), (0, Ly - Ny), (0, Lx - Nx)),
        constant_values=pad_value,
    )

    S_fft = jnp.fft.fft2(s_pad, axes=(1, 2))
    Q_fft = juvenile_fft_kernel_stack[None, :, :, :] * S_fft[:, None, :, :]
    Q = jnp.real(jnp.fft.ifft2(Q_fft, axes=(2, 3)))[:, :, :Ny, :Nx]

    return s_array, Q


# ============================================================
# Bioclim + pop HSGP fields
# ============================================================

def build_bioclim_fields(params, data):
    spd = hsgp_spd_se(
        params.alpha_bio_raw,
        params.length_bio_raw,
        ell=data["ell_bioclim"],
        m=data["m_bioclim"],
        dim=data["d_bioclim"],
    )

    f1 = hsgp_field(data["phi_bioclim"], spd, params.f1_bio_beta)
    f2 = hsgp_field(data["phi_bioclim"], spd, params.f2_bio_beta)

    r_bio, logK_bio = params.L_bio @ jnp.stack([f1, f2])
    return r_bio, logK_bio


def build_pop_fields(params, data):
    spd = hsgp_spd_se(
        params.alpha_pop_raw,
        params.length_pop_raw,
        ell=data["ell_pop"],
        m=data["m_pop"],
        dim=1,
    )

    f_pop = hsgp_field(data["phi_pop"], spd, params.f_pop_beta)
    r_pop = params.b_r * f_pop
    logK_pop = params.b_K * f_pop

    return r_pop, logK_pop


# ============================================================
# Full deterministic forward fields
# ============================================================

def build_fields_from_latents(params, data):
    r_bio, logK_bio = build_bioclim_fields(params, data)
    r_pop, logK_pop = build_pop_fields(params, data)

    r_array, K_array = precompute_r_K(
        time=data["time"],
        Ny=data["Ny"],
        Nx=data["Nx"],
        land_rows=data["land_rows"],
        land_cols=data["land_cols"],
        r_mean=params.r_mean,
        r_bioclim=r_bio,
        r_pop=r_pop,
        r_spatial=params.r_spatial,
        r_temp=params.r_temp,
        K_raw_mean=params.K_raw_mean,
        logK_bio=logK_bio,
        logK_pop=logK_pop,
    )

    s_array, Q_array = precompute_s_Q_vectorized(
        r_array,
        data["juvenile_fft_kernel_stack"],
        jnn.softplus(params.dispersal_survival_slope_raw),
        params.dispersal_survival_threshold,
    )

    allee_scalar = data["pop_scalar"] * jnn.softplus(params.allee_slope_raw)

    return r_array, K_array, s_array, Q_array, allee_scalar
