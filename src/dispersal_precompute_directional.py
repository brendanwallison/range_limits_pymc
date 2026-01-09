# ============================================================
# Dispersal preprocessing (run ONCE, outside NumPyro model)
# ============================================================

import jax.numpy as jnp
from jax.numpy.fft import fft2, ifft2
from jax.scipy.special import gamma
import numpy as np 

# ------------------------------------------------------------
# Toroidal distance grid
# ------------------------------------------------------------
def toroidal_distance_grid(Lx: int, Ly: int, distance_scale: float) -> jnp.ndarray:
    if (Lx % 2 == 0) or (Ly % 2 == 0):
        raise ValueError("Lx and Ly must be odd")

    x = jnp.concatenate([jnp.arange(Lx//2 + 1), jnp.arange(Lx//2, 0, -1)])
    y = jnp.concatenate([jnp.arange(Ly//2 + 1), jnp.arange(Ly//2, 0, -1)])

    x_steps = jnp.tile(x, (Ly, 1))
    y_steps = jnp.tile(y[:, None], (1, Lx))

    return jnp.sqrt(x_steps**2 + y_steps**2) * distance_scale

# ------------------------------------------------------------
# Edge correction from FFT kernel
# ------------------------------------------------------------
def edge_correction_from_fft(fft_land, fft_kernel, land_mask, Ny, Nx, eps=1e-12):
    fraction_land = jnp.real(ifft2(fft_land * fft_kernel))[:Ny, :Nx]
    fraction_land = jnp.maximum(fraction_land, eps)
    correction = 1.0 / fraction_land
    correction = jnp.where(land_mask, correction, 1.0)
    return correction

# ------------------------------------------------------------
# Toroidal-aware angular weights
# ------------------------------------------------------------

def angular_weights_toroidal(Lx: int, Ly: int):
    """
    Return smooth angular weights for four cardinal directions.
    The origin is at (0,0) (top-left corner), consistent with toroidal FFT conventions.
    Each kernel sums to 0.25 (so total mass = 1).
    """
    # Generate grid with origin at (0,0)
    y_idx, x_idx = jnp.meshgrid(jnp.arange(Ly), jnp.arange(Lx), indexing="ij")
    
    # Centered coordinates for toroidal FFT conventions
    # y_idx=1 -> dy=1 (South 1 unit in matrix coords)
    dx = jnp.where(x_idx <= Lx // 2, x_idx, x_idx - Lx)
    dy = jnp.where(y_idx <= Ly // 2, y_idx, y_idx - Ly)
    
    angles = jnp.arctan2(dy, dx)  # [-pi, pi]

    # PYTHON MATRIX CONVENTION:
    # y increases downwards. dy > 0 is SOUTH. dy < 0 is NORTH.
    # Angles: East=0, South=pi/2, West=pi, North=-pi/2
    directions = {
        'to_NORTH': -jnp.pi/2, 
        'to_SOUTH':  jnp.pi/2, 
        'to_EAST':   0.0, 
        'to_WEST':   jnp.pi
    }
    width = jnp.pi  # ±90° taper

    w_dict = {}
    for d, target_angle in directions.items():
        # Difference wrapped to [-pi, pi]
        diff = jnp.mod(angles - target_angle + jnp.pi, 2*jnp.pi) - jnp.pi
        # Smooth cosine taper: 1 at 0 diff, 0 at ±width/2
        taper = jnp.clip(diff, -width/2, width/2)  # restrict to ±width/2
        weight = 0.5 * (1 + jnp.cos(jnp.pi * taper / (width/2)))
        # Ensure origin (0,0) is evenly distributed
        weight = jnp.where((dx == 0) & (dy == 0), 0.25, weight)
        # Normalize so sum over this kernel = 0.25
        total = jnp.sum(weight)
        total_safe = jnp.where(total == 0, 1.0, total)
        weight = 0.25 * weight / total_safe
        w_dict[d] = weight

    return w_dict

# ------------------------------------------------------------
# Main preprocessing entry point
# ------------------------------------------------------------
def prepare_dispersal_constants(
    land: jnp.ndarray,  # (Ny, Nx), 1=land, 0=ocean
    mean_dispersal_distance: float,
    mean_local_dispersal_distance: float,
    unit_distance: float,
    adult_shape: float,
    juvenile_shape: float,
    distance_scale: float,
):
    Ny, Nx = land.shape
    Lx = 2 * Nx - 1
    Ly = 2 * Ny - 1

    land_mask = land.astype(bool)
    padded_land = jnp.zeros((Ly, Lx)).at[:Ny, :Nx].set(land)
    fft_land = fft2(padded_land)

    r_dist = toroidal_distance_grid(Lx, Ly, distance_scale)

    def scale_from_mean(mean_dist, shape):
        return (mean_dist / unit_distance) * gamma(2.0 / shape) / gamma(3.0 / shape)

    adult_scale = scale_from_mean(mean_dispersal_distance, adult_shape)
    juvenile_scale = scale_from_mean(mean_local_dispersal_distance, juvenile_shape)

    # Adult dispersal
    adult_kernel = jnp.exp(-(r_dist / adult_scale) ** adult_shape)
    adult_kernel = adult_kernel / jnp.sum(adult_kernel)
    adult_fft_kernel = fft2(adult_kernel)
    adult_edge_correction = edge_correction_from_fft(fft_land, adult_fft_kernel, land_mask, Ny, Nx)

    # Juvenile dispersal (symmetric)
    juvenile_kernel = jnp.exp(-(r_dist / juvenile_scale) ** juvenile_shape)
    juvenile_kernel = juvenile_kernel / jnp.sum(juvenile_kernel)

    # Angular directional weights
    angular_w = angular_weights_toroidal(Lx, Ly)
    
    # Store in lists to stack later (ensuring deterministic order)
    # Recommended order for 'juvenile_fft_kernel_stack': to_NORTH, to_SOUTH, to_EAST, to_WEST
    stack_keys = ['to_NORTH', 'to_SOUTH', 'to_EAST', 'to_WEST']
    
    juvenile_fft_list = []
    juvenile_edge_list = []
    
    # We still return the dicts if the legacy code needs them, 
    # but the primary output for the new model is the STACK.
    juvenile_fft_kernels = {}
    juvenile_edge_corrections = {}
    
    for d in stack_keys:
        w = angular_w[d]
        k_dir = juvenile_kernel * w
        k_dir = k_dir / jnp.sum(k_dir)
        
        fft_k = fft2(k_dir)
        edge_c = edge_correction_from_fft(fft_land, fft_k, land_mask, Ny, Nx)
        
        # Populate lists for stack
        juvenile_fft_list.append(fft_k)
        juvenile_edge_list.append(edge_c)
        
        # Populate legacy dicts
        juvenile_fft_kernels[d] = fft_k
        juvenile_edge_corrections[d] = edge_c

    # Create stacks for vectorized processing
    juvenile_fft_stack = jnp.stack(juvenile_fft_list, axis=0)
    juvenile_edge_stack = jnp.stack(juvenile_edge_list, axis=0)

    return {
        "fft_land": fft_land,
        "Ny": Ny,
        "Nx": Nx,
        "Ly": Ly,
        "Lx": Lx,
        "adult_fft_kernel": adult_fft_kernel,
        "adult_edge_correction": adult_edge_correction,
        "juvenile_fft_kernels": juvenile_fft_kernels, # Legacy dict
        "juvenile_edge_corrections": juvenile_edge_corrections, # Legacy dict
        "juvenile_fft_kernel_stack": juvenile_fft_stack, # New Model Input (4, Ly, Lx)
        "juvenile_edge_correction_stack": juvenile_edge_stack, # New Model Input (4, Ny, Nx)
    }