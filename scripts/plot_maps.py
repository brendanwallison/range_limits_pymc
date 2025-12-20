#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#!/usr/bin/env python3

import numpy as np
import jax.numpy as jnp
import jax.nn as jnn
import matplotlib.pyplot as plt

from src.ingest import build_data_jax
from src.model import (
    precompute_r_K,
    forward_sim_2d,
)


# ---------------------------
# Plot helper
# ---------------------------

def save_map(field, t, outdir, name, vmin, vmax):
    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=(6, 5))
    plt.imshow(field[t], origin="lower", cmap="viridis",
               vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title(f"{name}, t={t}")
    plt.tight_layout()
    plt.savefig(f"{outdir}/{name}_t{t:03d}.png", dpi=150)
    plt.close()


# ---------------------------
# Main
# ---------------------------

def main():

    # ---------------------------
    # Load data
    # ---------------------------
    data = build_data_jax("data/stan_data_for_python.npz")

    Ny, Nx = data["initpop_latent"].shape
    time = data["time"]

    # ---------------------------
    # Load MAP estimate
    # ---------------------------
    map_np = np.load("map_estimate.npz")
    params = {k: jnp.asarray(map_np[k]) for k in map_np.files}

    # ---------------------------
    # Derived parameters (EXACTLY as in model)
    # ---------------------------
    # Allee effect
    allee_slope = jnn.softplus(params["allee_slope_raw"])
    allee_scalar = data["pop_scalar"] * allee_slope

    # Invasion intensity
    inv_pop = jnn.sigmoid(params["inv_eta"])

    # ---------------------------
    # Reconstruct r_array and K_array
    # ---------------------------
    r_array, K_array = precompute_r_K(
        time=time,
        Ny=Ny,
        Nx=Nx,
        land_rows=data["land_rows"],
        land_cols=data["land_cols"],
        r_mean=params["r_mean"],
        r_bioclim=params["r_bioclim"],
        r_pop=params["r_pop"],
        r_spatial=params["r_spatial"],
        r_temp=params["r_temp"],
        K_raw_mean=params["K_raw_mean"],
        logK_bio=params["logK_bio"],
        logK_pop=params["logK_pop"],
    )

    # ---------------------------
    # Forward simulation (MAP)
    # ---------------------------
    densities = forward_sim_2d(
        time=time,
        initial_pop=data["initpop_latent"],
        r_array=r_array,
        K_array=K_array,
        dispersal_logit_intercept=params["dispersal_logit_intercept"],
        dispersal_logit_slope=params["dispersal_logit_slope"],
        dispersal_intercept=params["dispersal_intercept"],
        dispersal_random=params["dispersal_random"],
        adult_fft_kernel=data["adult_fft_kernel"],
        juvenile_fft_kernel=data["juvenile_fft_kernel"],
        adult_edge_correction=data["adult_edge_correction"],
        juvenile_edge_correction=data["juvenile_edge_correction"],
        land_mask=data["land_mask"],
        inv_pop=inv_pop,
        inv_location=data["inv_location"],
        inv_timestep=data["inv_timestep"],
        allee_scalar=allee_scalar,
        allee_intercept=params["allee_intercept"],
        pseudo_zero=data["pseudo_zero"],
    )

    # ---------------------------
    # Output directories
    # ---------------------------
    base = "sanity_maps_MAP"
    r_dir = os.path.join(base, "r")
    K_dir = os.path.join(base, "K")
    pop_dir = os.path.join(base, "population")

    # ---------------------------
    # Fixed color scales
    # ---------------------------
    rmin, rmax = float(jnp.min(r_array)), float(jnp.max(r_array))
    Kmin, Kmax = float(jnp.min(K_array)), float(jnp.max(K_array))
    pmin, pmax = float(jnp.min(densities)), float(jnp.max(densities))

    print("r range:", rmin, rmax)
    print("K range:", Kmin, Kmax)
    print("population range:", pmin, pmax)

    # ---------------------------
    # Save maps
    # ---------------------------
    for t in range(time):
        save_map(r_array, t, r_dir, "r", rmin, rmax)
        save_map(K_array, t, K_dir, "K", Kmin, Kmax)
        save_map(densities, t, pop_dir, "population", pmin, pmax)

    print(f"Saved all maps to {base}/")


if __name__ == "__main__":
    main()
