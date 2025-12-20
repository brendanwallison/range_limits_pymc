import jax.numpy as jnp
import jax.nn as jnn

# ============================================================
# Allee + reproduction
# ============================================================

def allee_factor(pop, allee_scalar, allee_intercept):
    """Allee effect using a sigmoid."""
    return jnn.sigmoid(pop * allee_scalar + allee_intercept)


def reproduction_safe(N0, r, K, allee_scalar, allee_intercept, eps=1e-12):
    """
    Numerically safe Beverton–Holt–style reproduction with Allee effect.
    """
    g = jnp.exp(r)
    denom = K + N0 * jnp.expm1(r) + eps
    pop_new = g * N0 * K / denom
    return allee_factor(N0, allee_scalar, allee_intercept) * pop_new


# ============================================================
# FFT helpers
# ============================================================

def rightpad(A, Lx, Ly, pad_value=1e-9):
    pad = jnp.ones((Ly, Lx)) * pad_value
    return pad.at[:A.shape[0], :A.shape[1]].set(A)


def rightpad_convolution(pop, kernel_fft):
    Ly, Lx = kernel_fft.shape
    pop_pad = rightpad(pop, Lx, Ly)
    conv = jnp.fft.ifft2(jnp.fft.fft2(pop_pad) * kernel_fft)
    return jnp.real(conv)[:pop.shape[0], :pop.shape[1]]


# ============================================================
# Juvenile dispersal (vectorized, directional)
# ============================================================

def juvenile_dispersal_vectorized(
    juvenile_dispersers,            # (Ny, Nx)
    juvenile_fft_kernels,           # (Nd, Ly, Lx)
    juvenile_edge_corrections,      # (Nd, Ny, Nx)
    Q,                              # (Nd, Ny, Nx)
    dirichlet_weights=None,
    pad_value=1e-9,
):
    Ny, Nx = juvenile_dispersers.shape
    Nd = juvenile_fft_kernels.shape[0]

    if dirichlet_weights is None:
        dirichlet_weights = jnp.ones(Nd) / Nd

    w = dirichlet_weights[:, None, None] * Nd
    pop_split = juvenile_dispersers[None, :, :] * w * juvenile_edge_corrections

    Ly, Lx = juvenile_fft_kernels.shape[1:]
    pop_pad = jnp.pad(
        pop_split,
        ((0, 0), (0, Ly - Ny), (0, Lx - Nx)),
        constant_values=pad_value,
    )

    fft_pop = jnp.fft.fft2(pop_pad, axes=(-2, -1))
    conv = jnp.fft.ifft2(fft_pop * juvenile_fft_kernels, axes=(-2, -1))
    conv = jnp.real(conv)[:, :Ny, :Nx]

    return jnp.sum(conv * Q, axis=0)


# ============================================================
# Dispersal step
# ============================================================

def dispersal_step(
    pop,
    K,
    dispersal_logit_intercept,
    dispersal_logit_slope,
    target_fraction,
    dispersal_intercept,
    dispersal_random,
    adult_edge_correction,
    juvenile_edge_correction_stack,
    adult_fft_kernel,
    juvenile_fft_kernel_stack,
    Q_array,
    eps=1e-6,
    clip_z=30.0,
    min_p=1e-8,
):
    K_safe = K + eps
    z = dispersal_logit_intercept + dispersal_logit_slope * (
        pop / K_safe - target_fraction
    )
    z = jnp.clip(z, -clip_z, clip_z)
    p_total = jnp.clip(jnn.sigmoid(z), min_p, 1 - min_p)
    dispersers = p_total * pop

    z_j = dispersal_intercept + dispersal_random
    z_j = jnp.clip(z_j, -clip_z, clip_z)
    p_j = jnp.clip(jnn.sigmoid(z_j), min_p, 1 - min_p)

    juvenile = dispersers * p_j
    adult = dispersers - juvenile

    adult_update = rightpad_convolution(
        adult * adult_edge_correction,
        adult_fft_kernel,
    )

    juvenile_update = juvenile_dispersal_vectorized(
        juvenile,
        juvenile_fft_kernel_stack,
        juvenile_edge_correction_stack,
        Q_array,
    )

    return pop + adult_update + juvenile_update - dispersers
