import jax.numpy as jnp
import jax.nn
from jax.numpy.fft import fft2, ifft2
from scipy.special import gamma, gammaincinv

# =========================================================
# 1. CORE GEOMETRY & MATH
# =========================================================

def toroidal_distance_grid(Lx: int, Ly: int, cell_size: float) -> jnp.ndarray:
    if (Lx % 2 == 0) or (Ly % 2 == 0):
        raise ValueError("Lx and Ly must be odd")

    x = jnp.concatenate([jnp.arange(Lx//2 + 1), jnp.arange(Lx//2, 0, -1)])
    y = jnp.concatenate([jnp.arange(Ly//2 + 1), jnp.arange(Ly//2, 0, -1)])

    x_steps = jnp.tile(x, (Ly, 1))
    y_steps = jnp.tile(y[:, None], (1, Lx))

    return jnp.sqrt(x_steps**2 + y_steps**2) * cell_size

def angular_weights_toroidal(Lx: int, Ly: int):
    y_idx, x_idx = jnp.meshgrid(jnp.arange(Ly), jnp.arange(Lx), indexing="ij")
    
    # Centered coordinates for toroidal FFT conventions
    dx = jnp.where(x_idx <= Lx // 2, x_idx, x_idx - Lx)
    dy = jnp.where(y_idx <= Ly // 2, y_idx, y_idx - Ly)
    
    angles = jnp.arctan2(dy, dx) 

    directions = {
        'to_NORTH': -jnp.pi/2, 
        'to_SOUTH':  jnp.pi/2, 
        'to_EAST':   0.0, 
        'to_WEST':   jnp.pi
    }
    width = jnp.pi 

    w_dict = {}
    for d, target_angle in directions.items():
        diff = jnp.mod(angles - target_angle + jnp.pi, 2*jnp.pi) - jnp.pi
        taper = jnp.clip(diff, -width/2, width/2) 
        weight = 0.5 * (1 + jnp.cos(jnp.pi * taper / (width/2)))
        weight = jnp.where((dx == 0) & (dy == 0), 1.0, weight)
        w_dict[d] = weight

    return w_dict

def edge_correction_from_fft(fft_land, fft_kernel, land_mask, Ny, Nx, eps=1e-12):
    """
    Calculates the Fraction of the kernel that lands on valid habitat.
    
    FIX: Returns the FRACTION (denominator), not the reciprocal.
    forward.py divides by this value: Result = Conv / Fraction.
    """
    # Cross-Correlation in Fourier Domain = F(A) * conj(F(B))
    fraction_land = jnp.real(ifft2(fft_land * jnp.conj(fft_kernel)))[:Ny, :Nx]
    fraction_land = jnp.maximum(fraction_land, eps)
    
    # On water pixels, we don't care (mask later), but set to 1.0 to avoid NaNs
    fraction_land = jnp.where(land_mask, fraction_land, 1.0)
    
    return fraction_land

def get_gamma_scale(mean_dist, shape):
    return mean_dist * gamma(2.0 / shape) / gamma(3.0 / shape)

def get_dispersal_quantiles(mean_dist, shape_param, quantiles=[0.33, 0.66]):
    g2 = gamma(2.0 / shape_param)
    g3 = gamma(3.0 / shape_param)
    scale = mean_dist / (g3 / g2)
    
    radii = []
    for q in quantiles:
        val = gammaincinv(2.0 / shape_param, q)
        r = scale * (val ** (1.0 / shape_param))
        radii.append(r)
    return radii

# =========================================================
# 2. KERNEL BUILDERS (The Factories)
# =========================================================

def make_radial_directional_kernels(
    Lx, Ly, 
    cell_size, 
    base_kernel_grid,   # [NEW] The Master Gamma PDF
    radii_splits, 
    smoothness_km=None
):
    """
    Splits the base_kernel_grid into 12 wedges using soft masking.
    Does NOT re-normalize. The sum of all kernels equals base_kernel_grid.
    """
    if smoothness_km is None:
        smoothness_km = 2.0 * cell_size

    r_dist = toroidal_distance_grid(Lx, Ly, cell_size)
    angular_w = angular_weights_toroidal(Lx, Ly)
    
    kernels = []
    labels = []
    
    slope = 4.0 / smoothness_km
    
    radial_cdfs = []
    for r_boundary in radii_splits:
        if r_boundary <= 1e-6:
            cdf = jnp.ones_like(r_dist)
        elif r_boundary >= 1e9:
            cdf = jnp.zeros_like(r_dist)
        else:
            cdf = jax.nn.sigmoid(slope * (r_dist - r_boundary))
        radial_cdfs.append(cdf)
        
    direction_order = ['to_NORTH', 'to_SOUTH', 'to_EAST', 'to_WEST']
    
    for d in direction_order:
        w_dir = angular_w[d]
        
        for i in range(len(radii_splits) - 1):
            r_min_val = radii_splits[i]
            r_max_val = radii_splits[i+1]
            
            # 1. Calculate the Partition of Unity Mask
            # (Which fraction of space belongs to this bin?)
            mask_radial = radial_cdfs[i] - radial_cdfs[i+1]
            mask_combined = w_dir * mask_radial
            
            # 2. Apply Mask to Base PDF (Scenario B)
            # This preserves the probability mass of the donut.
            k = base_kernel_grid * mask_combined
            
            # [REMOVED] Normalization Step
            # total = jnp.sum(k)
            # k = k / total
            
            kernels.append(k)
            labels.append(f"{d}_{r_min_val:.0f}-{r_max_val:.0f}")

    return jnp.stack(kernels, axis=0), labels


def build_simulation_struct(
    land: jnp.ndarray,
    cell_size: float,
    mean_dispersal_distance: float,
    mean_local_dispersal_distance: float,
    adult_shape: float,
    juvenile_shape: float,
    radii_splits=None
):
    """
    Builds simulation structure with mass-conservative weighted kernels.
    """
    Ny, Nx = land.shape
    Lx, Ly = 2 * Nx - 1, 2 * Ny - 1
    land_mask = land.astype(bool)
    
    padded_land = jnp.zeros((Ly, Lx)).at[:Ny, :Nx].set(land)
    fft_land = fft2(padded_land)
    
    # 1. Grid (in km)
    r_dist = toroidal_distance_grid(Lx, Ly, cell_size)
    
    # 2. Adult (Isotropic)
    adult_scale = get_gamma_scale(mean_dispersal_distance, adult_shape)
    adult_kernel = jnp.exp(-(r_dist / adult_scale) ** adult_shape)
    adult_kernel /= jnp.sum(adult_kernel) # Normalize Master
    adult_fft_kernel = fft2(adult_kernel)
    
    # Adult Edge Correction (Standard)
    adult_edge_correction = edge_correction_from_fft(fft_land, adult_fft_kernel, land_mask, Ny, Nx)

    # 3. Juvenile (12 Cohorts)
    if radii_splits is None:
        radii_splits = [0.0] + get_dispersal_quantiles(
            mean_local_dispersal_distance, juvenile_shape, [0.33, 0.66]
        ) + [1e9]

    # A. Generate Master Juvenile Kernel (Gamma)
    juv_scale = get_gamma_scale(mean_local_dispersal_distance, juvenile_shape)
    juv_master = jnp.exp(-(r_dist / juv_scale) ** juvenile_shape)
    juv_master /= jnp.sum(juv_master) # Normalize Master to 1.0
    
    # B. Split into Weighted Kernels
    juv_kernels, labels = make_radial_directional_kernels(
        Lx, Ly, cell_size, 
        juv_master, # Pass the PDF!
        radii_splits
    )
    
    fft_list = []
    edge_list = []
    
    # C. Calculate Edge Corrections for Weighted Kernels
    for i in range(juv_kernels.shape[0]):
        k_weighted = juv_kernels[i]
        
        # 1. Store Weighted Kernel for Simulation
        fft_k = fft2(k_weighted)
        fft_list.append(fft_k)
        
        # 2. Calculate Edge Correction using NORMALIZED shape
        # Edge correction asks: "If I throw this SHAPE at land, what fraction hits?"
        # We must temporarily normalize to answer that question correctly.
        weight = jnp.sum(k_weighted)
        
        # Safety for empty kernels (remote possibility)
        k_normalized = jnp.where(weight > 1e-12, k_weighted / weight, k_weighted)
        fft_k_norm = fft2(k_normalized)
        
        # Returns FRACTION (0.0 to 1.0)
        edge_c = edge_correction_from_fft(fft_land, fft_k_norm, land_mask, Ny, Nx)
        edge_list.append(edge_c)

    return {
        "fft_land": fft_land,
        "adult_fft_kernel": adult_fft_kernel,
        "adult_edge_correction": adult_edge_correction,
        
        "juvenile_fft_kernel_stack": jnp.stack(fft_list, axis=0),        # (12, Ly, Lx)
        "juvenile_edge_correction_stack": jnp.stack(edge_list, axis=0),  # (12, Ny, Nx)
        
        "labels": labels,
        "radii_splits": radii_splits
    }