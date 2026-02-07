import sys
import os
import glob
import pickle
import warnings
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as project_coords

# --- Setup Paths ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import jax.numpy as jnp
from src.model.build_kernels import build_simulation_struct

# --- CONFIGURATION ---
RAW_Z_DIR = "/home/breallis/processed_data/datasets/latent_avian_paths"
BBS_DATA_NPZ = "/home/breallis/datasets/bbs_2024_release/bbs_data_for_python.npz"
MASK_FILE = "/home/breallis/datasets/latent_avian_community_similarities/ocean_mask_4km.tif"
OUTPUT_DIR = "/home/breallis/processed_data/model_inputs/numpyro_input"

# --- PARAMETERS ---
AGG_FACTOR = 4          
N_PCA_COMPONENTS = 16    
START_YEAR = 1900
END_YEAR = 2023

# --- SPATIOTEMPORAL BASIS SETTINGS ---
N_FREQ_SPACE = 4  # ~600km regional resolution
N_FREQ_TIME = 8   # ~15-year decadal resolution

# --- INVASION PARAMETERS ---
INV_LAT = 40.6106
INV_LON = -73.4445

def generate_spatiotemporal_basis(Ny, Nx, Time, land_rows, land_cols, n_freq_space=4, n_freq_time=8):
    """
    Generates a 3D Spectral Basis (Cosine series).
    Space: n_freq_space=4 captures regional patterns.
    Time: n_freq_time=8 captures decadal cycles.
    """
    print(f"  Constructing 3D Basis: Space={n_freq_space}, Time={n_freq_time}...")
    
    # Create normalized coordinate grids [0, 1]
    t_coord = np.linspace(0, 1, Time)[:, None] # (Time, 1)
    y_coord = np.linspace(0, 1, Ny)[land_rows] # (N_land,)
    x_coord = np.linspace(0, 1, Nx)[land_cols] # (N_land,)
    
    basis_list = []
    
    for k in range(n_freq_time + 1):
        t_wave = np.cos(k * np.pi * t_coord) # (Time, 1)
        
        for i in range(n_freq_space + 1):
            for j in range(n_freq_space + 1):
                if i == 0 and j == 0 and k == 0:
                    continue # Skip the constant offset
                
                # Spatial component
                s_wave = np.cos(i * np.pi * y_coord) * np.cos(j * np.pi * x_coord) # (N_land,)
                
                # Outer product creates (Time, N_land) volume
                st_volume = (t_wave * s_wave[None, :]).astype(np.float32)
                basis_list.append(st_volume)
    
    st_basis = np.stack(basis_list, axis=0) # (N_basis, Time, N_land)
    return st_basis


def downsample_grid(array, factor, method='mean'):
    Ny, Nx = array.shape[:2]
    new_ny = Ny // factor
    new_nx = Nx // factor
    trimmed = array[:new_ny*factor, :new_nx*factor]
    new_shape = (new_ny, factor, new_nx, factor) + trimmed.shape[2:]
    reshaped = trimmed.reshape(new_shape)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if method == 'mean': return np.nanmean(reshaped, axis=(1, 3))
        elif method == 'max': return np.nanmax(reshaped, axis=(1, 3))
        else: raise ValueError("Unknown method")

def get_log_spaced_splits(min_dist, max_dist, n_bins):
    start = np.log10(max(min_dist, 1.0))
    end = np.log10(max_dist)
    log_points = np.logspace(start, end, n_bins + 1)
    splits = [0.0] + list(log_points[1:])
    splits[-1] = 1e9 
    return splits

def load_land_metadata(tif_path):
    with rasterio.open(tif_path) as src:
        res_x = src.res[0]
        if (src.crs and 'metre' in src.crs.linear_units.lower()) or (res_x > 100):
            cell_size_km = res_x / 1000.0
        else:
            cell_size_km = res_x * 111.0
    return cell_size_km

def get_grid_location(tif_path, lat, lon):
    with rasterio.open(tif_path) as src:
        if src.crs != 'EPSG:4326':
            xs, ys = project_coords('EPSG:4326', src.crs, [lon], [lat])
            x, y = xs[0], ys[0]
        else:
            x, y = lon, lat
        row, col = src.index(x, y)
        return int(row), int(col)

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
def ingest_data():
    print(f"--- Starting Data Ingestion (Agg={AGG_FACTOR}, PCA={N_PCA_COMPONENTS}) ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Raw Data (Fine Grid)
    if not os.path.exists(BBS_DATA_NPZ):
        raise FileNotFoundError(f"BBS data not found: {BBS_DATA_NPZ}")
        
    bbs_data = np.load(BBS_DATA_NPZ)
    raw_land_mask = bbs_data['land'].astype(np.float32)
    raw_ocean_bool = (raw_land_mask < 0.5)
    raw_init_density = bbs_data['initpop_density'] # This exists now!

    # 2. Downsample Geometry (Target Grid)
    print("  Downsampling Grid and Initialization Map...")
    land_mask = downsample_grid(raw_land_mask, AGG_FACTOR, method='max')
    land_mask = (land_mask > 0).astype(int)
    
    # Downsample Init Map using MAX to preserve the 0.5/0.05 values
    # If we used mean, the density would dilute.
    initpop_map = downsample_grid(raw_init_density, AGG_FACTOR, method='max')
    # Mask out any init that fell into the ocean during downsampling
    initpop_map = initpop_map * land_mask
    
    Ny, Nx = land_mask.shape
    land_rows, land_cols = np.where(land_mask)
    N_land = len(land_rows)
    print(f"  New Grid: {Ny}x{Nx}, Land Pixels: {N_land}")

    # 3. Process Observations
    print("  Processing Observations...")
    orig_rows = bbs_data['obs_rows']
    orig_cols = bbs_data['obs_cols']
    orig_years = bbs_data['obs_year']
    orig_counts = bbs_data['observed_results']
    n_pseudo_orig = int(bbs_data['N_pseudo'])
    
    # Split Real vs Pseudo
    real_indices = slice(n_pseudo_orig, None)
    pseudo_indices = slice(0, n_pseudo_orig)
    
    # -- Real Data --
    r_rows_coarse = orig_rows[real_indices] // AGG_FACTOR
    r_cols_coarse = orig_cols[real_indices] // AGG_FACTOR
    r_years = orig_years[real_indices]
    r_counts = orig_counts[real_indices]
    
    # -- Pseudo Data Subsampling --
    # a. Calculate Density of Real Data
    real_locs = np.vstack((r_rows_coarse, r_cols_coarse)).T
    unique_real_locs = np.unique(real_locs, axis=0)
    sampling_density = len(unique_real_locs) / N_land
    
    # b. Get Unique Coarse Locations of Pseudo Data
    p_rows_fine = orig_rows[pseudo_indices]
    p_cols_fine = orig_cols[pseudo_indices]
    p_years_fine = orig_years[pseudo_indices]
    
    p_rows_coarse = p_rows_fine // AGG_FACTOR
    p_cols_coarse = p_cols_fine // AGG_FACTOR
    
    pseudo_locs = np.vstack((p_rows_coarse, p_cols_coarse)).T
    unique_pseudo_locs = np.unique(pseudo_locs, axis=0)
    
    # c. Subsample
    n_target = int(len(unique_pseudo_locs) * sampling_density)
    n_target = max(n_target, 50)
    
    print(f"  Subsampling Pseudo-Zeros: Target {n_target} sites (Density {sampling_density:.4f})")
    
    rng = np.random.default_rng(42)
    if len(unique_pseudo_locs) > n_target:
        chosen_indices = rng.choice(len(unique_pseudo_locs), n_target, replace=False)
        chosen_locs = unique_pseudo_locs[chosen_indices]
    else:
        chosen_locs = unique_pseudo_locs
    
    # d. Expand Chosen Locs over Years
    # Note: p_years_fine contains all years. We just need the unique years range.
    years_range = np.unique(p_years_fine)
    
    final_p_rows, final_p_cols, final_p_years = [], [], []
    for yr in years_range:
        final_p_rows.append(chosen_locs[:, 0])
        final_p_cols.append(chosen_locs[:, 1])
        final_p_years.append(np.full(len(chosen_locs), yr))
        
    final_p_rows = np.concatenate(final_p_rows)
    final_p_cols = np.concatenate(final_p_cols)
    final_p_years = np.concatenate(final_p_years)
    final_p_counts = np.zeros_like(final_p_years)

    # -- Merge --
    obs_rows = np.concatenate([final_p_rows, r_rows_coarse])
    obs_cols = np.concatenate([final_p_cols, r_cols_coarse])
    obs_year = np.concatenate([final_p_years, r_years])
    observed_results = np.concatenate([final_p_counts, r_counts])
    
    # Bounds Check
    valid_locs = (obs_rows < Ny) & (obs_cols < Nx)
    obs_rows = obs_rows[valid_locs]
    obs_cols = obs_cols[valid_locs]
    obs_year = obs_year[valid_locs]
    observed_results = observed_results[valid_locs]
    
    print(f"  Final Observations: {len(observed_results)}")

    # 4. Stream Z Data
    z_files = sorted(glob.glob(os.path.join(RAW_Z_DIR, "Z_disp_*.npz")))
    file_map = {int(os.path.basename(f).split('_')[2].split('.')[0]): f for f in z_files}
    sorted_years = sorted(file_map.keys())
    start_year_model, end_year_model = min(sorted_years), max(sorted_years)
    model_years = np.array(sorted_years)
    Time = len(model_years)
    print(f"  Timeline: {start_year_model}-{end_year_model} ({Time} years)")

    peek = np.load(file_map[start_year_model])
    M = min(peek['Z_raw'].shape[-1], N_PCA_COMPONENTS)
    K = peek['Z_disp'].shape[-1]
    
    z_gather_path = os.path.join(OUTPUT_DIR, "Z_gathered.dat")
    Z_gathered = np.memmap(z_gather_path, dtype='float32', mode='w+', shape=(Time, N_land, M))
    z_disp_path = os.path.join(OUTPUT_DIR, "Z_disp_gathered.dat")
    Z_disp_gathered = np.memmap(z_disp_path, dtype='float32', mode='w+', shape=(Time, N_land, K, M))

    print("  Streaming Z Data...")
    for t, year in enumerate(model_years):
        data = np.load(file_map[year])
        raw_z = data['Z_raw'][0]; raw_z[raw_ocean_bool] = np.nan
        raw_disp = data['Z_disp'][0].transpose(0, 1, 3, 2); raw_disp[raw_ocean_bool] = np.nan
        
        coarse_z = np.nan_to_num(downsample_grid(raw_z, AGG_FACTOR, 'mean'))
        coarse_disp = np.nan_to_num(downsample_grid(raw_disp, AGG_FACTOR, 'mean'))
        
        Z_gathered[t] = coarse_z[land_rows, land_cols, :M]
        Z_disp_gathered[t] = coarse_disp[land_rows, land_cols, :, :M]
        if t % 5 == 0: print(f"    Processed {year}...", end='\r')

    Z_gathered.flush(); Z_disp_gathered.flush()
    print("\n  Data Streaming Complete.")

    # 5. Generate 3D Spatiotemporal Basis
    st_basis = generate_spatiotemporal_basis(Ny, Nx, Time, land_rows, land_cols, 
                                             n_freq_space=N_FREQ_SPACE, 
                                             n_freq_time=N_FREQ_TIME)
    N_basis = st_basis.shape[0]
    print(f"  Basis Footprint: {st_basis.nbytes / 1e6:.2f} MB")

    # 6. Build Kernels
    raw_cell_size = load_land_metadata(MASK_FILE)
    cell_size_km = raw_cell_size * AGG_FACTOR
    print(f"  Cell Size: {cell_size_km:.2f} km")
    
    splits = get_log_spaced_splits(min_dist=50.0, max_dist=1500.0, n_bins=3)
    sim_struct = build_simulation_struct(
        land=jnp.array(land_mask),
        cell_size=cell_size_km,
        mean_dispersal_distance=330.0, mean_local_dispersal_distance=330.0,
        adult_shape=0.468, juvenile_shape=0.468, radii_splits=splits
    )

    raw_inv_row, raw_inv_col = get_grid_location(MASK_FILE, INV_LAT, INV_LON)
    inv_row = raw_inv_row // AGG_FACTOR
    inv_col = raw_inv_col // AGG_FACTOR

    valid_obs_mask = (obs_year >= start_year_model) & (obs_year <= end_year_model)
    final_obs_time_idx = obs_year[valid_obs_mask] - start_year_model
    
    model_metadata = {
        "Ny": Ny, "Nx": Nx,
        "land_mask": np.array(land_mask).astype(int),
        "land_rows": np.array(land_rows), "land_cols": np.array(land_cols),
        "time": Time, "years": model_years,
        "M": M, "K": K, "N_land": N_land,
        "st_basis": st_basis, 
        "N_basis": N_basis,
        "z_gathered_path": "Z_gathered.dat", "z_disp_gathered_path": "Z_disp_gathered.dat",
        "adult_fft_kernel": np.array(sim_struct['adult_fft_kernel']),
        "juvenile_fft_kernel_stack": np.array(sim_struct['juvenile_fft_kernel_stack']),
        "adult_edge_correction": np.array(sim_struct['adult_edge_correction']),
        "juvenile_edge_correction_stack": np.array(sim_struct['juvenile_edge_correction_stack']),
        "obs_time_indices": np.array(final_obs_time_idx),
        "obs_rows": np.array(obs_rows[valid_obs_mask]),
        "obs_cols": np.array(obs_cols[valid_obs_mask]),
        "observed_results": np.array(observed_results[valid_obs_mask]),
        "initpop_latent": initpop_map,
        "pseudo_zero": 1e-12, "pop_scalar": 210.0,
        "inv_location": (inv_row, inv_col), "inv_timestep": 40, "inv_window": 10
    }
    
    meta_path = os.path.join(OUTPUT_DIR, "metadata.pkl")
    print(f"Saving metadata to {meta_path}...")
    with open(meta_path, 'wb') as f: pickle.dump(model_metadata, f)
    print("Success. Data ingested to disk.")

if __name__ == "__main__":
    ingest_data()