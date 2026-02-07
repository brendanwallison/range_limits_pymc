import os
import glob
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
from shapely.geometry import Point, MultiPoint
import geopandas as gpd

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
BBS_PARENT_DIR = "/home/breallis/datasets/bbs_2024_release" 

# 1. Weather Files
WEATHER_FILES = [
    os.path.join(BBS_PARENT_DIR, "Weather.csv"),
    os.path.join(BBS_PARENT_DIR, "Weather_Mexico.csv")
]

# 2. Route Files
ROUTE_FILES = [
    os.path.join(BBS_PARENT_DIR, "Routes.csv"),
    os.path.join(BBS_PARENT_DIR, "Routes_Mexico.csv")
]

# 3. States Directory
BBS_STATES_DIR = os.path.join(BBS_PARENT_DIR, "States")

# 4. Mask Path
MASK_PATH = "/home/breallis/datasets/land_mask/ocean_mask_4km.tif"

# 5. Parameters
HOUSE_FINCH_AOU = 5190
RPID_FILTER = 101
START_YEAR = 1900
END_YEAR = 2023
PSEUDO_ZERO_END_YEAR = 1939
BUFFER_DISTANCE_METERS = 1000 * 1000  # 1000 km buffer

# -----------------------------------------------------------------------------
# 1. Load Reference Grid
# -----------------------------------------------------------------------------
def load_grid_reference(mask_path):
    with rasterio.open(mask_path) as src:
        data = src.read(1)
        # Python: True=Land, False=Ocean (inverse of TIF 1=Ocean)
        ocean_mask = (data == 1)
        land_mask = (data == 0)
        transform = src.transform
        crs = src.crs
        ny, nx = data.shape
        
    print(f"Grid loaded: {ny}x{nx}, CRS: {crs}")
    return land_mask, ocean_mask, transform, crs, nx, ny

# -----------------------------------------------------------------------------
# 2. Load and Filter BBS Data
# -----------------------------------------------------------------------------
def load_bbs_data(states_dir, weather_files):
    print(f"Loading BBS data from {states_dir}...")
    
    if not os.path.exists(states_dir):
        raise FileNotFoundError(f"States directory not found: {states_dir}")

    # --- A. Load State Counts ---
    state_files = glob.glob(os.path.join(states_dir, "*.csv"))
    cols = ["CountryNum", "StateNum", "Route", "RPID", "Year", "AOU", "SpeciesTotal"]
    
    df_list = []
    for f in state_files:
        try:
            temp = pd.read_csv(f, usecols=cols)
            df_list.append(temp)
        except Exception as e:
            print(f"  Skipping file {f} due to error: {e}")
            continue 
            
    if not df_list:
        raise ValueError("No valid BBS State CSVs could be loaded.")
        
    bbs_full = pd.concat(df_list, ignore_index=True)
    
    for c in cols: bbs_full[c] = pd.to_numeric(bbs_full[c], errors='coerce')
    bbs_full.dropna(subset=["CountryNum", "StateNum", "Route", "RPID", "Year", "AOU"], inplace=True)
    int_cols = ["CountryNum", "StateNum", "Route", "RPID", "Year", "AOU"]
    bbs_full[int_cols] = bbs_full[int_cols].astype(int)

    # --- B. Load Weather (QC) ---
    print("Loading Weather/QC files...")
    weather_list = []
    w_cols = ["CountryNum", "StateNum", "Route", "RPID", "Year", "RunType"]
    
    for f in weather_files:
        if os.path.exists(f):
            try:
                temp_w = pd.read_csv(f, usecols=w_cols)
                weather_list.append(temp_w)
            except: pass

    qc_routes = pd.concat(weather_list, ignore_index=True)
    for c in w_cols: qc_routes[c] = pd.to_numeric(qc_routes[c], errors='coerce')
    qc_routes.dropna(inplace=True)
    qc_routes[w_cols] = qc_routes[w_cols].astype(int)

    # Filter QC and RPID
    qc_routes = qc_routes[(qc_routes['RunType'] == 1) & (qc_routes['RPID'] == RPID_FILTER)]
    bbs_full = bbs_full[bbs_full['RPID'] == RPID_FILTER]

    # --- C. Merge ---
    join_keys = ["CountryNum", "StateNum", "Route", "RPID", "Year"]
    finch_obs = bbs_full[bbs_full['AOU'] == HOUSE_FINCH_AOU].copy()
    
    merged = pd.merge(qc_routes, finch_obs, on=join_keys, how="left")
    merged['SpeciesTotal'] = merged['SpeciesTotal'].fillna(0).astype(int)
    merged = merged[(merged['Year'] >= START_YEAR) & (merged['Year'] <= END_YEAR)]
    
    print(f"  Final Merged Data: {len(merged)} rows.")
    return merged

# -----------------------------------------------------------------------------
# 3. Spatial Processing
# -----------------------------------------------------------------------------
def map_routes_to_grid(df, route_files_list, grid_transform, grid_crs, nx, ny, land_mask):
    print("Mapping routes to grid...")
    
    route_dfs = []
    for r_file in route_files_list:
        if os.path.exists(r_file):
            try: route_dfs.append(pd.read_csv(r_file))
            except UnicodeDecodeError: route_dfs.append(pd.read_csv(r_file, encoding="latin1"))

    routes_geo = pd.concat(route_dfs, ignore_index=True)
    
    # Create Geometry
    gdf = gpd.GeoDataFrame(
        routes_geo, 
        geometry=gpd.points_from_xy(routes_geo['Longitude'], routes_geo['Latitude']),
        crs="EPSG:4326" 
    )
    
    # Reproject to Model Grid CRS (Meters)
    gdf_proj = gdf.to_crs(grid_crs)
    
    coords = np.array([(p.x, p.y) for p in gdf_proj.geometry])
    rows, cols = rasterio.transform.rowcol(grid_transform, coords[:, 0], coords[:, 1])
    
    gdf_proj['row'] = rows
    gdf_proj['col'] = cols
    
    valid_indices = (gdf_proj['row'] >= 0) & (gdf_proj['row'] < ny) & (gdf_proj['col'] >= 0) & (gdf_proj['col'] < nx)
    gdf_valid = gdf_proj[valid_indices].copy()
    
    is_land = land_mask[gdf_valid['row'].values, gdf_valid['col'].values]
    gdf_final = gdf_valid[is_land].copy()
    
    join_keys = ["CountryNum", "StateNum", "Route"]
    df[join_keys] = df[join_keys].astype(int)
    gdf_final[join_keys] = gdf_final[join_keys].astype(int)

    final_df = pd.merge(df, gdf_final[join_keys + ['row', 'col', 'geometry']], on=join_keys, how='inner')
    return final_df

# -----------------------------------------------------------------------------
# 4. CORE VS MARGIN GENERATION
# -----------------------------------------------------------------------------
def generate_core_margin_initialization(obs_df, ny, nx, transform, land_mask):
    """
    1. Identifies Native Range (Pre-1970).
    2. Defines Core (75th percentile) and Margin (All points).
    3. Creates a DENSITY MAP (0.5 Core, 0.05 Margin).
    4. Buffers Margin by 1000km to find Uninvaded East.
    5. Generates Pseudo-Zeros for Uninvaded East.
    """
    print("Generating Core/Margin Map and Pseudo-Zeros...")
    
    western_limit_col = int(nx * 0.66)
    
    # 1. Native Range Data
    hist_obs = obs_df[
        (obs_df['Year'] <= 1970) & 
        (obs_df['SpeciesTotal'] > 0) &
        (obs_df['col'] < western_limit_col) 
    ].copy()
    
    if hist_obs.empty:
        raise ValueError("No observations found before 1970 in the West.")

    unique_locs = hist_obs.drop_duplicates(subset=['row', 'col'])
    
    # --- 2. HULL GENERATION ---
    # Margin (All Points)
    points_all = unique_locs['geometry'].tolist()
    hull_margin = MultiPoint(points_all).convex_hull
    
    # Core (High Density)
    threshold = unique_locs['SpeciesTotal'].quantile(0.75)
    print(f"  Core Threshold (75th percentile): {threshold:.1f}")
    
    core_locs = unique_locs[unique_locs['SpeciesTotal'] > threshold]
    points_core = core_locs['geometry'].tolist()
    hull_core = MultiPoint(points_core).convex_hull
    
    # --- 3. RASTERIZATION ---
    # We use the 'transform' to map Geometry (Meters) -> Pixels
    
    # Margin Mask
    mask_margin = rasterio.features.rasterize(
        [(hull_margin, 1)], out_shape=(ny, nx), transform=transform, default_value=0, dtype=np.uint8
    )
    mask_margin = (mask_margin == 1) & land_mask
    
    # Core Mask
    mask_core = rasterio.features.rasterize(
        [(hull_core, 1)], out_shape=(ny, nx), transform=transform, default_value=0, dtype=np.uint8
    )
    mask_core = (mask_core == 1) & land_mask
    
    # Density Map Construction
    initpop_density = np.zeros((ny, nx), dtype=np.float32)
    initpop_density[mask_margin] = 0.001
    initpop_density[mask_core] = 0.1  # Overwrites margin
    
    print(f"  Init Map created. Core Pixels: {np.sum(mask_core)}, Margin Pixels: {np.sum(mask_margin)}")

    # --- 4. BUFFERING (Uninvaded East) ---
    print(f"  Buffering Native Range by {BUFFER_DISTANCE_METERS/1000:.0f} km...")
    hull_buffer = hull_margin.buffer(BUFFER_DISTANCE_METERS)
    
    inside_buffer_mask = rasterio.features.rasterize(
        [(hull_buffer, 1)], out_shape=(ny, nx), transform=transform, default_value=0, dtype=np.uint8
    )
    # Uninvaded = Land AND Not in Buffer
    uninvaded_mask = land_mask & (inside_buffer_mask == 0)
    
    # --- 5. PSEUDO-ZEROS ---
    ui_rows, ui_cols = np.where(uninvaded_mask)
    n_ui = len(ui_rows)
    print(f"  Found {n_ui} uninvaded pixels for pseudo-zeros.")
    
    pseudo_rows, pseudo_cols, pseudo_years = [], [], []
    for year in range(START_YEAR, PSEUDO_ZERO_END_YEAR + 1):
        pseudo_rows.append(ui_rows)
        pseudo_cols.append(ui_cols)
        pseudo_years.append(np.full(n_ui, year))
        
    if pseudo_rows:
        p_rows = np.concatenate(pseudo_rows)
        p_cols = np.concatenate(pseudo_cols)
        p_years = np.concatenate(pseudo_years)
        p_counts = np.zeros_like(p_years)
    else:
        p_rows, p_cols, p_years, p_counts = [], [], [], []
        
    return initpop_density, p_rows, p_cols, p_years, p_counts

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    land_mask, ocean_mask, transform, crs, nx, ny = load_grid_reference(MASK_PATH)
    
    bbs_df = load_bbs_data(BBS_STATES_DIR, WEATHER_FILES)
    
    bbs_mapped = map_routes_to_grid(bbs_df, ROUTE_FILES, transform, crs, nx, ny, land_mask)

    # Use the new logic
    init_density_map, p_rows, p_cols, p_years, p_counts = generate_core_margin_initialization(
        bbs_mapped, ny, nx, transform, land_mask
    )

    real_rows = bbs_mapped['row'].values
    real_cols = bbs_mapped['col'].values
    real_years = bbs_mapped['Year'].values
    real_counts = bbs_mapped['SpeciesTotal'].values

    # Concatenate
    final_rows = np.concatenate([p_rows, real_rows])
    final_cols = np.concatenate([p_cols, real_cols])
    final_years = np.concatenate([p_years, real_years])
    final_counts = np.concatenate([p_counts, real_counts])

    # Save to NPZ
    # Note: We now save 'initpop_density' which is the FULL FLOAT GRID
    np.savez(
        "/home/breallis/datasets/bbs_2024_release/bbs_data_for_python.npz",
        Nx=nx,
        Ny=ny,
        land=land_mask.astype(int),
        ocean=ocean_mask.astype(int),
        obs_rows=final_rows.astype(int),
        obs_cols=final_cols.astype(int),
        obs_year=final_years.astype(int),
        observed_results=final_counts.astype(int),
        
        # New: Save the full density map
        initpop_density=init_density_map,
        
        # Legacy: Save indices of non-zero (union of core/margin) for backward compat
        initpop_rows=np.where(init_density_map > 0)[0],
        initpop_cols=np.where(init_density_map > 0)[1],
        
        N_obs=len(real_counts),
        N_pseudo=len(p_counts),
        unit_distance=1000.0,
        time=END_YEAR - START_YEAR + 1
    )

    print("Done. Saved bbs_data_for_python.npz")