import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingest_directional import build_data_jax

import jax.numpy as jnp
from jax.numpy.fft import fft2, ifft2
import matplotlib.pyplot as plt

# -------------------------------
# FFT convolution helper
# -------------------------------
def fft_convolve(pop, fft_kernel, edge_correction, Ny, Nx):
    Ly, Lx = fft_kernel.shape
    # pop *= edge_correction
    padded_pop = jnp.zeros((Ly, Lx))
    padded_pop = padded_pop.at[:Ny, :Nx].set(pop)
    conv_result = jnp.real(ifft2(fft2(padded_pop) * fft_kernel))[:Ny, :Nx]
    return conv_result

# -------------------------------
# Visualize & save heatmap
# -------------------------------
def save_heatmap(matrix, filename, title="Matrix"):
    plt.figure(figsize=(6,5))
    plt.imshow(matrix, origin="lower", cmap="viridis")
    plt.colorbar(label="Population")
    plt.title(title)
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved {filename}")

# -------------------------------
# Build test population: single spike
# -------------------------------
data_jax = build_data_jax("data/stan_data_for_python.npz")
Ny, Nx = data_jax["initpop_latent"].shape

pop_before = jnp.zeros((Ny, Nx))
center_y, center_x = Ny // 2, Nx // 2
pop_before = pop_before.at[center_y, center_x].set(1.0)

land_mask = data_jax["land_mask"]
distance_scale = 15.26926  # same as previous

# -------------------------------
# Juvenile directional kernels
# -------------------------------
direction_keys = ["N", "S", "E", "W"]

pop_total = jnp.zeros_like(pop_before)

y_idx, x_idx = jnp.meshgrid(jnp.arange(Ny), jnp.arange(Nx), indexing="ij")
center_y, center_x = Ny // 2, Nx // 2
distance_from_origin = jnp.sqrt((y_idx - center_y)**2 + (x_idx - center_x)**2) * distance_scale

for d in direction_keys:
    fft_kernel = data_jax["juvenile_fft_kernels"][d]
    edge_correction = data_jax["juvenile_edge_corrections"][d]

    pop_after = fft_convolve(
        pop_before,
        fft_kernel,
        edge_correction,
        Ny,
        Nx
    )

    # accumulate total
    pop_total += pop_after

    # save individual directional heatmaps
    save_heatmap(pop_after, f"pop_after_spike_{d}.png", title=f"After Juvenile Dispersal ({d})")
    save_heatmap(pop_after - pop_before, f"pop_diff_spike_{d}.png", title=f"Displacement ({d})")

    # mean dispersal distance (land only, moved population)
    displacement_only = jnp.maximum(pop_after - pop_before, 0) * land_mask
    total_moved = jnp.sum(displacement_only)
    mean_disp = jnp.sum(displacement_only * distance_from_origin) / total_moved
    print(f"Mean dispersal distance ({d} kernel, moved only): {mean_disp:.3f}")

# ---- After accumulating all directions ----
save_heatmap(pop_total, "pop_after_spike_total.png", title="After Juvenile Dispersal (Total)")

displacement_total_only = jnp.maximum(pop_total - pop_before, 0) * land_mask
total_moved_total = jnp.sum(displacement_total_only)
mean_disp_total = jnp.sum(displacement_total_only * distance_from_origin) / total_moved_total
print(f"Mean dispersal distance (total, moved only): {mean_disp_total:.3f}")


