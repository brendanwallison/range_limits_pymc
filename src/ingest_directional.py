import numpy as np
import jax.numpy as jnp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
from numpyro.contrib.hsgp.laplacian import eigenfunctions
from src.dispersal_precompute_directional import prepare_dispersal_constants
from scipy.spatial import KDTree

# --------------------------------------------------
# Index helpers (unchanged)
# --------------------------------------------------

def colmajor_to_rows_and_cols(indices, nrows, ncols):
    indices = np.asarray(indices) - 1  # R 1-based → 0-based
    cols = indices // nrows
    rows = indices % nrows
    return rows, cols

def rows_and_cols_to_colmajor(rows, cols, nrows, ncols):
    return cols * nrows + rows + 1

def precompute_ocean_nn_jax(
    ocean_rows: np.ndarray,
    ocean_cols: np.ndarray,
    land_rows: np.ndarray,
    land_cols: np.ndarray,
    k: int = 20
):
    """
    Precompute the k nearest land neighbors for each ocean cell.
    
    Args:
        ocean_rows, ocean_cols: (Nocean,) arrays of ocean cell indices
        land_rows, land_cols: (Nland,) arrays of land cell indices
        k: number of nearest neighbors to store

    Returns:
        nn_rows: (Nocean, k) jnp.array of neighbor row indices
        nn_cols: (Nocean, k) jnp.array of neighbor column indices
        nn_weights: (Nocean, k) jnp.array of normalized inverse-distance weights
    """
    # Stack land and ocean coordinates for KD-tree query
    land_coords = np.stack([land_rows, land_cols], axis=1)      # shape (Nland, 2)
    ocean_coords = np.stack([ocean_rows, ocean_cols], axis=1)  # shape (Nocean, 2)

    tree = KDTree(land_coords)
    dists, idxs = tree.query(ocean_coords, k=k)

    # Avoid zero distances
    dists = np.maximum(dists, 1e-6)

    # Inverse-distance weighting
    weights = 1.0 / dists
    weights /= weights.sum(axis=1, keepdims=True)

    # Map indices to row/col coordinates
    nn_rows = land_rows[idxs]
    nn_cols = land_cols[idxs]

    # Convert everything to jnp arrays for JAX
    return jnp.array(nn_rows, dtype=jnp.int32), \
           jnp.array(nn_cols, dtype=jnp.int32), \
           jnp.array(weights, dtype=jnp.float32)


# --------------------------------------------------
# Main builder 
# --------------------------------------------------

def build_data_jax(npz_path="data/stan_data_for_python.npz", k_ocean_nn=20):
    data_raw = np.load(npz_path, allow_pickle=True)

    Nx = int(data_raw["Nx"])
    Ny = int(data_raw["Ny"])
    unit_distance = 1
    distance_scale = data_raw["unit_distance"]
    N_obs = int(data_raw["N_obs"])
    N_pseudo = int(data_raw["N_pseudo"])

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

    # --------------------------------------------------
    # Index reconstruction
    # --------------------------------------------------
    land_rows, land_cols = colmajor_to_rows_and_cols(land_indices, Ny, Nx)
    ocean_rows, ocean_cols = colmajor_to_rows_and_cols(ocean_indices, Ny, Nx)
    obs_rows, obs_cols = colmajor_to_rows_and_cols(obs_indices, Ny, Nx)

    land_bool = jnp.array(land, dtype=bool)
    ocean_bool = jnp.array(ocean, dtype=bool)

    init_year = obs_years.min()
    obs_time_indices = obs_years - init_year
    inv_timestep = int(inv_year - init_year)
    inv_row, inv_col = colmajor_to_rows_and_cols(inv_location, Ny, Nx)
    inv_location_tuple = (inv_row, inv_col)

    # --------------------------------------------------
    # Initial population
    # --------------------------------------------------
    initpop_latent = jnp.full((Ny, Nx), pseudo_zero)
    init_rows, init_cols = colmajor_to_rows_and_cols(initpop_indices, Ny, Nx)
    initpop_latent = initpop_latent.at[init_rows, init_cols].set(western_initpop)

    # --------------------------------------------------
    # Population time series for HSGP
    # --------------------------------------------------
    r_pop_input = jnp.array(data_raw["pop_ts"])[:, None]

    # --------------------------------------------------
    # Bioclim PCA
    # --------------------------------------------------
    X = data_raw["bioclim_flat"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    bioclim_D = 3
    pca = PCA(n_components=bioclim_D, whiten=False)
    bioclim_PC_np = pca.fit_transform(X_scaled)
    var = pca.explained_variance_ratio_
    cumvar = np.cumsum(var)
    pc_mean = bioclim_PC_np.mean(axis=0)
    pc_std = bioclim_PC_np.std(axis=0)
    bioclim_PC_np = (bioclim_PC_np - pc_mean) / pc_std
    bioclim_PC = jnp.asarray(bioclim_PC_np)

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

    ell_bioclim = 5
    m_bioclim = 5
    ell_pop = 5
    m_pop = 5

    phi_bioclim = eigenfunctions(x=bioclim_PC, ell=ell_bioclim, m=m_bioclim)
    phi_pop = eigenfunctions(x=r_pop_input, ell=ell_pop, m=m_pop)

    # --------------------------------------------------
    # Dispersal constants (adults + four juvenile directions)
    # --------------------------------------------------
    md_adult = 100
    md_juvenile = 330
    adult_shape = 0.859
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
    
    direction_keys = ["N", "S", "E", "W"]
    juvenile_fft_kernel_stack = jnp.stack([dispersal_constants["juvenile_fft_kernels"][d] for d in direction_keys], axis=0)
    juvenile_edge_correction_stack = jnp.stack([dispersal_constants["juvenile_edge_corrections"][d] for d in direction_keys], axis=0)
    # shape = [4, Ly, Lx]

    # --------------------------------------------------
    # Ocean nearest-land neighbors
    # --------------------------------------------------
    nn_rows, nn_cols, nn_weights = precompute_ocean_nn_jax(
        ocean_rows=ocean_rows,
        ocean_cols=ocean_cols,
        land_rows=land_rows,
        land_cols=land_cols,
        k=20
    )

    # --------------------------------------------------
    # Package for NumPyro
    # --------------------------------------------------
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
        # Adult dispersal
        adult_fft_kernel=dispersal_constants["adult_fft_kernel"],
        adult_edge_correction=dispersal_constants["adult_edge_correction"],
        # Juvenile directional dispersal (nested dicts)
        juvenile_fft_kernel_stack=juvenile_fft_kernel_stack,
        juvenile_edge_correction_stack=juvenile_edge_correction_stack,
        # Ocean nearest neighbors
        ocean_rows=ocean_rows,
        ocean_cols=ocean_cols,
        ocean_nn_rows=nn_rows,
        ocean_nn_cols=nn_cols,
        ocean_nn_weights=nn_weights
    )

    return data_jax
