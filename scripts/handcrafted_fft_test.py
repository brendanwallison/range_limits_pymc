import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import your preprocessing function
from src.ingest import build_data_jax

import jax.numpy as jnp
from jax.numpy.fft import fft2, ifft2
import matplotlib.pyplot as plt

# -------------------------------
# FFT convolution helper
# -------------------------------
def fft_convolve(pop, fft_kernel, edge_correction, Ny, Nx):
    Ly, Lx = fft_kernel.shape
    pop *= edge_correction
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
pop_before = pop_before.at[center_y, center_x].set(1.0)  # single individual in the middle

distance_scale = 15.26926
land_mask = data_jax["land_mask"]  # 1=land, 0=ocean

# -------------------------------
# Juvenile dispersal convolution
# -------------------------------
pop_after = land_mask * fft_convolve(
    pop_before,
    data_jax["adult_fft_kernel"],
    data_jax["adult_edge_correction"],
    Ny,
    Nx
)

# -------------------------------
# Save heatmaps
# -------------------------------
save_heatmap(pop_before, "pop_before_spike.png", title="Initial Population (Spike)")
save_heatmap(pop_after, "pop_after_spike.png", title="After Juvenile Dispersal")
save_heatmap(pop_after - pop_before, "pop_diff_spike.png", title="Displacement (After - Before)")

# -------------------------------
# Correct mean dispersal distance (land only, only mass that moved)
# -------------------------------
y_idx, x_idx = jnp.meshgrid(jnp.arange(Ny), jnp.arange(Nx), indexing="ij")
distance_from_origin = jnp.sqrt((y_idx - center_y)**2 + (x_idx - center_x)**2) * distance_scale

# Only count population that actually moved
displacement_only = jnp.maximum(pop_after - pop_before, 0)
displacement_only = displacement_only * land_mask  # zero out ocean cells

total_moved = jnp.sum(displacement_only)
mean_disp = jnp.sum(displacement_only * distance_from_origin) / total_moved

print("Mean dispersal distance (single spike test, moved population only):", mean_disp)


slice = pop_after[center_y-22:center_y+22, center_x-22:center_x+22]

slice = pop_after[center_y-44:center_y+44, center_x-44:center_x+44]
