# ============================================================
# Dispersal preprocessing (run ONCE, outside NumPyro model)
# ============================================================

import jax.numpy as jnp
from jax.numpy.fft import fft2, ifft2
from jax.scipy.special import gamma

# ------------------------------------------------------------
# Toroidal distance grid
# ------------------------------------------------------------
def toroidal_distance_grid(Lx: int, Ly: int, distance_scale: float) -> jnp.ndarray:
    if (Lx % 2 == 0) or (Ly % 2 == 0):
        raise ValueError("Lx and Ly must be odd")

    max_x = Lx // 2
    max_y = Ly // 2

    x = jnp.concatenate([
        jnp.arange(max_x + 1),
        jnp.arange(max_x, 0, -1),
    ])
    y = jnp.concatenate([
        jnp.arange(max_y + 1),
        jnp.arange(max_y, 0, -1),
    ])

    x_steps = jnp.tile(x, (Ly, 1))
    y_steps = jnp.tile(y[:, None], (1, Lx))

    return jnp.sqrt(x_steps**2 + y_steps**2) * distance_scale


# ------------------------------------------------------------
# Dispersal kernel FFT (no pseudo-zero, normalized)
# ------------------------------------------------------------
def dispersal_kernel_fft(
    scale: float,
    shape: float,
    r_dist: jnp.ndarray,
) -> jnp.ndarray:
    kernel = jnp.exp(-(r_dist / scale) ** shape)
    kernel = kernel / jnp.sum(kernel)
    return fft2(kernel)


# ------------------------------------------------------------
# Edge correction from FFT kernel
# ------------------------------------------------------------
def edge_correction_from_fft(
    fft_land: jnp.ndarray,
    fft_kernel: jnp.ndarray,
    land_mask: jnp.ndarray,
    Ny: int,
    Nx: int,
    eps: float = 1e-12,
) -> jnp.ndarray:

    fraction_land = jnp.real(ifft2(fft_land * fft_kernel))[:Ny, :Nx]
    fraction_land = jnp.maximum(fraction_land, eps)



    correction = 1.0 / fraction_land
    correction = jnp.where(land_mask, correction, 1.0)

    return correction


# ------------------------------------------------------------
# Main preprocessing entry point
# ------------------------------------------------------------
def prepare_dispersal_constants(
    land: jnp.ndarray,               # (Ny, Nx), 1=land, 0=ocean
    mean_dispersal_distance: float,
    mean_local_dispersal_distance: float,
    unit_distance: float,
    adult_shape: float,
    juvenile_shape: float,
    distance_scale: float,
):
    """
    Precompute FFT dispersal kernels and edge corrections.
    Intended to be run ONCE (Stan-style transformed data).
    """

    Ny, Nx = land.shape
    Lx = 2 * Nx - 1
    Ly = 2 * Ny - 1

    land = jnp.asarray(land)
    land_mask = land.astype(bool)

    # Pad land mask for linear convolution
    padded_land = jnp.zeros((Ly, Lx))
    padded_land = padded_land.at[:Ny, :Nx].set(land)
    fft_land = fft2(padded_land)

    # Distance grid
    r_dist = toroidal_distance_grid(Lx, Ly, distance_scale)

    # Mean -> scale conversion for 2D kernel
    def scale_from_mean(mean_dist, shape):
        return (mean_dist / unit_distance) * gamma(2.0 / shape) / gamma(3.0 / shape)

    adult_scale = scale_from_mean(mean_dispersal_distance, adult_shape)
    juvenile_scale = scale_from_mean(mean_local_dispersal_distance, juvenile_shape)

    # FFT kernels
    adult_fft_kernel = dispersal_kernel_fft(adult_scale, adult_shape, r_dist)
    juvenile_fft_kernel = dispersal_kernel_fft(juvenile_scale, juvenile_shape, r_dist)

    # Edge corrections
    adult_edge = edge_correction_from_fft(
        fft_land, adult_fft_kernel, land_mask, Ny, Nx
    )
    juvenile_edge = edge_correction_from_fft(
        fft_land, juvenile_fft_kernel, land_mask, Ny, Nx
    )

    return {
        # shared
        "fft_land": fft_land,
        "Ny": Ny,
        "Nx": Nx,
        "Ly": Ly,
        "Lx": Lx,

        # adult dispersal
        "adult_fft_kernel": adult_fft_kernel,
        "adult_edge_correction": adult_edge,

        # juvenile dispersal
        "juvenile_fft_kernel": juvenile_fft_kernel,
        "juvenile_edge_correction": juvenile_edge,
    }
