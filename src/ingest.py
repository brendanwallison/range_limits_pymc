import numpy as np
import jax.numpy as jnp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
from numpyro.contrib.hsgp.laplacian import eigenfunctions
from src.dispersal_precompute import prepare_dispersal_constants

# --------------------------------------------------
# Index helpers (unchanged)
# --------------------------------------------------

def colmajor_to_rows_and_cols(indices, nrows, ncols):
    """
    Convert flat column-major indices to 2D row/column coordinates.
    """
    indices = np.asarray(indices) - 1  # R 1-based → 0-based
    cols = indices // nrows
    rows = indices % nrows
    return rows, cols


def rows_and_cols_to_colmajor(rows, cols, nrows, ncols):
    """
    Convert 2D row/column coordinates to column-major flat indices.
    """
    return cols * nrows + rows + 1


# --------------------------------------------------
# Main builder
# --------------------------------------------------

def build_data_jax(
    npz_path="data/stan_data_for_python.npz",
):
    """
    Load R-exported Stan data and construct the JAX-native
    data dictionary used by the NumPyro model.
    """

    # -----------------------------
    # Load raw data
    # -----------------------------
    data_raw = np.load(npz_path, allow_pickle=True)

    Nx = int(data_raw["Nx"])
    Ny = int(data_raw["Ny"])
    unit_distance = 1
    distance_scale = data_raw["unit_distance"] # Named wrong in R
    # distance_scale = 1
    N_obs = int(data_raw["N_obs"])
    N_pseudo = int(data_raw["N_pseudo"])
    # N_ocean_indices = int(data_raw["N_ocean_indices"])
    # N_land_indices = int(data_raw["N_land_indices"])
    # N_initpop_indices = int(data_raw["N_initpop_indices"])

    ocean_indices = data_raw["ocean_indices"].astype(int)
    land_indices = data_raw["land_indices"].astype(int)
    initpop_indices = data_raw["initpop_indices"].astype(int)

    time_true = int(data_raw["time"])
    pop_scalar = float(data_raw["pop_scalar"])

    ocean = jnp.array(data_raw["ocean"])
    land = jnp.array(data_raw["land"])

    observed_results = jnp.array(data_raw["observed_results"])
    obs_years = jnp.array(data_raw["obs_year"])
    obs_indices = jnp.array(data_raw["obs_indices"])

    inv_location = int(data_raw["inv_location"])
    inv_year = int(data_raw["inv_year"])

    pseudo_zero = data_raw["pseudo_zero"]
    western_initpop = data_raw["western_initpop"]

    # -----------------------------
    # Index reconstruction
    # -----------------------------

    # Land cells
    land_rows, land_cols = colmajor_to_rows_and_cols(
        land_indices, Ny, Nx
    )

    land_bool = jnp.array(land, dtype=bool)

    ocean_rows, ocean_cols = colmajor_to_rows_and_cols(
        ocean_indices, Ny, Nx
    )

    ocean_bool = jnp.array(ocean, dtype=bool)

    # Observations
    obs_rows, obs_cols = colmajor_to_rows_and_cols(
        obs_indices, Ny, Nx
    )

    init_year = obs_years.min()
    obs_time_indices = obs_years - init_year

    # Invasion timing: calendar year → model timestep
    inv_timestep = int(inv_year - init_year)
    inv_timestep = 45

    # Invasion location: flat → (row, col)
    inv_row, inv_col = colmajor_to_rows_and_cols(
        inv_location, Ny, Nx
    )
    inv_location_tuple = (inv_row, inv_col)

    # -----------------------------
    # Initial population (2D-native)
    # -----------------------------

    initpop_latent = jnp.full(
        (Ny, Nx),
        pseudo_zero
    )

    init_rows, init_cols = colmajor_to_rows_and_cols(
        initpop_indices, Ny, Nx
    )

    initpop_latent = initpop_latent.at[
        init_rows, init_cols
    ].set(western_initpop)

    # -----------------------------
    # Population time series for HSGP
    # -----------------------------
    # Flattened vector with population for each land index by year, just like R
    r_pop_input = jnp.array(data_raw["pop_ts"])[:, None]  # shape [N_land * N_years, 1]

    # -----------------------------
    # Transform Bioclim (PCA)
    # -----------------------------
    print("Taking PCA of Bioclim")

    X = data_raw["bioclim_flat"]  # (N, 19)

    # ---- Standardize original variables
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ---- PCA
    bioclim_D = 3
    pca = PCA(n_components=bioclim_D, whiten=False)
    bioclim_PC_np = pca.fit_transform(X_scaled)

    # ---- Print variance explained
    var = pca.explained_variance_ratio_
    cumvar = np.cumsum(var)

    print(
        f"PCA complete | variance explained: "
        + ", ".join(f"PC{i+1}={v:.3f}" for i, v in enumerate(var))
        + f" | cumulative={cumvar[-1]:.3f}"
    )

    # ---- Rescale PCs to unit variance (IMPORTANT for HSGP)
    pc_mean = bioclim_PC_np.mean(axis=0)
    pc_std = bioclim_PC_np.std(axis=0)
    bioclim_PC_np = (bioclim_PC_np - pc_mean) / pc_std

    # ---- Convert to JAX
    bioclim_PC = jnp.asarray(bioclim_PC_np) # (N_land, 3)

    # ---- Store PCA artifacts alongside raw data
    data_dir = os.path.dirname(npz_path)
    np.savez(
        os.path.join(data_dir, "bioclim_pca.npz"),
        bioclim_PC=bioclim_PC_np,
        original_mean=scaler.mean_,
        original_scale=scaler.scale_,
        components=pca.components_,
        explained_variance_ratio=var,
        pc_mean=pc_mean,
        pc_std=pc_std,
        n_components=bioclim_D,
    )

    print("Bioclim PCs standardized (mean 0, std 1 per component)")

    ell_bioclim=5
    m_bioclim=5
    ell_pop=5
    m_pop=5
    
    # --- Precompute HSGP basis for Bioclim ---
    phi_bioclim = eigenfunctions(
        x=bioclim_PC,  # JAX array, shape [N_land, D]
        ell=ell_bioclim,
        m=m_bioclim
    )

    phi_pop = eigenfunctions(
        x=r_pop_input,  # JAX array, shape [N_land, D]
        ell=ell_pop,
        m=m_pop
    )

    md_adult=100
    md_juvenile=330
    adult_shape=0.859
    juvenile_shape = 0.468

    dispersal_constants = prepare_dispersal_constants(
        land=land,
        mean_dispersal_distance=md_adult,
        mean_local_dispersal_distance=md_juvenile,
        unit_distance=unit_distance,
        adult_shape=adult_shape,
        juvenile_shape=juvenile_shape,
        distance_scale=distance_scale,
    )


    # -----------------------------
    # Package for NumPyro
    # -----------------------------

    data_jax = dict(
        Nx=Nx,
        Ny=Ny,
        N_obs=N_obs,
        N_pseudo=N_pseudo,
        initpop_latent=initpop_latent,
        time=time_true,
        pop_scalar=pop_scalar,
        ocean=ocean,
        land=land,
        land_rows=land_rows,
        land_cols=land_cols,
        observed_results=observed_results,
        obs_rows=obs_rows,
        obs_cols=obs_cols,
        obs_time_indices=obs_time_indices,
        inv_location=inv_location_tuple,
        inv_year=inv_year,
        inv_timestep=inv_timestep,
        inv_window=20,
        phi_bioclim=phi_bioclim,
        ell_bioclim=ell_bioclim,
        m_bioclim=m_bioclim,
        d_bioclim=bioclim_D,
        phi_pop=phi_pop,
        ell_pop=ell_pop,
        m_pop=m_pop,
        land_mask=land,
        pseudo_zero=pseudo_zero,
        adult_fft_kernel=dispersal_constants["adult_fft_kernel"],
        adult_edge_correction=dispersal_constants["adult_edge_correction"],
        juvenile_fft_kernel=dispersal_constants["juvenile_fft_kernel"],
        juvenile_edge_correction=dispersal_constants["juvenile_edge_correction"],
    )

    return data_jax
