#!/usr/bin/env python3
"""
Sanity check for 2D population model.
Generates population maps over time, overlays observed points, and saves r/K maps.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from src.ingest import build_data_jax
from src.model import build_model_2d, forward_sim_2d, precompute_r_K

# ---------------------------
# Helper functions
# ---------------------------
def save_heatmap(matrix, filename, title="Matrix", cmap="viridis", overlay_points=None):
    """
    Save a heatmap with optional overlay points.
    overlay_points: list of (row, col) tuples
    """
    plt.figure(figsize=(6,5))
    plt.imshow(matrix, origin="lower", cmap=cmap)
    plt.colorbar(label="Population")
    plt.title(title)
    if overlay_points is not None and len(overlay_points) > 0:
        ys, xs = zip(*overlay_points)
        plt.scatter(xs, ys, c="red", s=20, marker="x")
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved {filename}")

# ---------------------------
# Main function
# ---------------------------
def main():
    # -----------------------
    # Load data
    # -----------------------
    data_path = "data/stan_data_for_python.npz"
    data = build_data_jax(data_path)

    Ny, Nx = data["Ny"], data["Nx"]
    time = data["time"]

    # -----------------------
    # Build default model params for sanity check
    # -----------------------
    densities = forward_sim_2d(
        time=time,
        initial_pop=data["initpop_latent"],
        r_array=jnp.ones((time, Ny, Nx)) * 0.1,  # simple default
        K_array=jnp.ones((time, Ny, Nx)) * 10.0,
        dispersal_logit_intercept=0.0,
        dispersal_logit_slope=0.0,
        dispersal_intercept=0.0,
        dispersal_random=jnp.zeros(time),
        adult_fft_kernel=data["adult_fft_kernel"],
        juvenile_fft_kernel=data["juvenile_fft_kernel"],
        adult_edge_correction=data["adult_edge_correction"],
        juvenile_edge_correction=data["juvenile_edge_correction"],
        land_mask=data["land_mask"],
        inv_pop=0.0,
        inv_location=(0,0),
        inv_timestep=-1,
        allee_scalar=data["pop_scalar"],
        allee_intercept=0.0,
        pseudo_zero=data["pseudo_zero"],
    )

    # -----------------------
    # Prepare output folders
    # -----------------------
    out_root = "sanity_check_outputs"
    os.makedirs(out_root, exist_ok=True)
    pop_folder = os.path.join(out_root, "population")
    r_folder = os.path.join(out_root, "r_maps")
    K_folder = os.path.join(out_root, "K_maps")
    for folder in [pop_folder, r_folder, K_folder]:
        os.makedirs(folder, exist_ok=True)

    # -----------------------
    # Loop through timesteps and save maps
    # -----------------------
    for t in range(time):
        # Population map
        pop_map = densities[t, :, :]
        # Overlay observed points for this timestep
        obs_mask = data["obs_time_indices"] == t
        overlay_points = list(zip(data["obs_rows"][obs_mask], data["obs_cols"][obs_mask]))
        save_heatmap(
            pop_map,
            filename=os.path.join(pop_folder, f"pop_t{t:03d}.png"),
            title=f"Population t={t}",
            overlay_points=overlay_points
        )

        # r and K maps
        r_map = data.get("r_array", jnp.ones((time, Ny, Nx)) * 0.1)[t, :, :]
        K_map = data.get("K_array", jnp.ones((time, Ny, Nx)) * 10.0)[t, :, :]
        save_heatmap(r_map, os.path.join(r_folder, f"r_t{t:03d}.png"), title=f"r t={t}")
        save_heatmap(K_map, os.path.join(K_folder, f"K_t{t:03d}.png"), title=f"K t={t}")

    print("All sanity check maps saved.")

# ---------------------------
# Entry point
# ---------------------------
if __name__ == "__main__":
    main()
