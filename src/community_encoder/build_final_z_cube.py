import os
import glob
import numpy as np
import rasterio
import torch
from torch import nn
import torch.nn.functional as F
from scipy.interpolate import griddata
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

# ============================================================
# 1. MODEL ARCHITECTURE (Must match training script)
# ============================================================
class BMLPBlock(nn.Module):
    def __init__(self, m, k=4, dropout=0.5):
        super().__init__()
        self.ln = nn.LayerNorm(m)
        self.fc1 = nn.Linear(m, m*k)
        self.fc2 = nn.Linear(m*k, m)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        z = self.ln(x)
        z = F.gelu(self.fc1(z))
        z = self.drop(z)
        z = self.fc2(z)
        return x + z

class MultiInputAutoencoder(nn.Module):
    def __init__(self, prism_dim, bui_dim, latent_dim):
        super().__init__()
        h = max(128, latent_dim * 4) 
        self.prism_enc = nn.Sequential(
            nn.Linear(prism_dim, h), nn.GELU(),
            BMLPBlock(h), BMLPBlock(h),
        )
        self.bui_enc = nn.Sequential(
            nn.Linear(bui_dim, h), nn.GELU(),
            BMLPBlock(h), BMLPBlock(h),
        )
        self.mixer = nn.Sequential(
            nn.Linear(2*h, 2*h), nn.GELU(),
            nn.Linear(2*h, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 2*h), nn.GELU(),
            nn.Linear(2*h, prism_dim + bui_dim),
        )

    def forward(self, prism, bui):
        h = torch.cat([self.prism_enc(prism), self.bui_enc(bui)], dim=1)
        z_pred = self.mixer(h)
        return z_pred, self.decoder(z_pred)

# ============================================================
# 2. GAP FILLING LOGIC
# ============================================================
def fill_gaps_stage1_spatial(z_cube, valid_mask, land_mask, radius_px=25):
    """
    Fills land gaps within `radius_px` (100km @ 4km/px) using Linear Interpolation.
    """
    H, W, D = z_cube.shape
    
    # 1. Identify Target Pixels: Land + NaN + Close to Valid
    # Calculate Euclidean distance from every pixel to the nearest VALID pixel
    # distance_transform_edt calculates distance to the nearest ZERO, so we invert valid_mask
    dist_map = distance_transform_edt(~valid_mask)
    
    # Target: Is Land, Is currently Invalid (NaN), Is within 100km
    target_mask = land_mask & (~valid_mask) & (dist_map <= radius_px)
    
    if target_mask.sum() == 0:
        return z_cube, valid_mask
        
    print(f"   -> Stage 1 (Spatial): Interpolating {target_mask.sum()} pixels within 100km...")

    # 2. Prepare Interpolation coords
    # Sources: All currently valid pixels
    y_valid, x_valid = np.where(valid_mask)
    points = np.column_stack((y_valid, x_valid))
    
    # Targets: The selected gap pixels
    y_target, x_target = np.where(target_mask)
    
    # 3. Interpolate channel by channel
    # Linear is best for "smooth continuous" fields like climate/Z
    z_filled = z_cube.copy()
    
    for d in range(D):
        values = z_cube[y_valid, x_valid, d]
        interp_vals = griddata(points, values, (y_target, x_target), method='linear')
        z_filled[y_target, x_target, d] = interp_vals
        
    # Update validity mask (some griddata might fail if hull is convex, but mostly good)
    new_valid = ~np.isnan(z_filled).any(axis=-1)
    
    return z_filled, new_valid

def fill_gaps_stage2_static(z_cube, valid_mask, land_mask, z_static_ref, z_static_mask):
    """
    Fills remaining land gaps using the static 2023 eBird Z values.
    """
    # Target: Land + Still Invalid + We actually have eBird data there
    target_mask = land_mask & (~valid_mask) & z_static_mask
    
    count = target_mask.sum()
    if count == 0:
        return z_cube, valid_mask

    print(f"   -> Stage 2 (Static): Backfilling {count} pixels with 2023 eBird Community Z...")
    
    z_filled = z_cube.copy()
    z_filled[target_mask] = z_static_ref[target_mask]
    
    new_valid = valid_mask | target_mask
    return z_filled, new_valid

def fill_gaps_stage3_nearest(z_cube, valid_mask, land_mask):
    """
    Fills ANY remaining land gaps with the nearest neighbor.
    """
    target_mask = land_mask & (~valid_mask)
    
    if target_mask.sum() == 0:
        return z_cube
        
    print(f"   -> Stage 3 (Cleanup): NN filling remaining {target_mask.sum()} pixels...")
    
    # Use NearestNDInterpolator (or griddata nearest)
    y_valid, x_valid = np.where(valid_mask)
    points = np.column_stack((y_valid, x_valid))
    
    y_target, x_target = np.where(target_mask)
    
    z_filled = z_cube.copy()
    for d in range(z_cube.shape[2]):
        values = z_cube[y_valid, x_valid, d]
        interp_vals = griddata(points, values, (y_target, x_target), method='nearest')
        z_filled[y_target, x_target, d] = interp_vals
        
    return z_filled

# ============================================================
# 3. MAIN EXECUTION
# ============================================================
def build_spacetime_cube():
    # --- CONFIG ---
    DATA_DIR = "/home/breallis/datasets"
    HIST_DIR = f"{DATA_DIR}/smoothed_prism_bui/yearly_states"
    
    # Weights and Reference Z
    Z_DIR = "/home/breallis/dev/range_limits_pymc/misc_outputs/ruzicka_sweep_global/sigma_1.5/" 
    MODEL_PATH = os.path.join(Z_DIR, "results", "env_model_semisup.pth")
    Z_REF_PATH = os.path.join(Z_DIR, "Z.npy")
    MASK_REF_PATH = os.path.join(Z_DIR, "valid_mask.npy")
    
    # Ocean Mask
    WATER_MASK_PATH = f"{DATA_DIR}/land_mask/ocean_mask_4km.tif"
    
    OUTPUT_DIR = os.path.join(Z_DIR, "results", "spacetime_cube")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Reference Grids
    print("Loading masks and reference data...")
    with rasterio.open(WATER_MASK_PATH) as src:
        water_data = src.read(1)
        LAND_MASK = (water_data == 0) # 1=Water, 0=Land
        H, W = LAND_MASK.shape

    # Load 2023 Static Z for Stage 2 Filling
    z_ref_flat = np.load(Z_REF_PATH)
    z_ref_mask = np.load(MASK_REF_PATH)
    
    # Project flat Z back to 2D grid
    Z_DIM = z_ref_flat.shape[1]
    Z_STATIC_GRID = np.full((H, W, Z_DIM), np.nan, dtype=np.float32)
    Z_STATIC_GRID[z_ref_mask] = z_ref_flat
    Z_STATIC_VALID = ~np.isnan(Z_STATIC_GRID).any(axis=-1)

    # 2. Derive Normalization Stats (Crucial: Must match training)
    print("Loading 2023 state to derive normalization stats...")
    state_2023 = np.load(os.path.join(HIST_DIR, "state_2023_bio_ema10.npz"))
    
    # Apply the intersection mask used in training script to get exact same stats
    # (Recreating the training logic briefly)
    p_temp = state_2023['prism']
    b_temp = state_2023['bui']
    
    # Intersection of all 3 (Env + BUI + Z_ref)
    mask_intersect = (~np.isnan(p_temp).any(-1)) & \
                     (~np.isnan(b_temp).any(-1)) & \
                     z_ref_mask
                     
    p_flat = p_temp[mask_intersect]
    b_flat = b_temp[mask_intersect]
    
    # FIX: Cast to float32 immediately
    p_mu = p_flat.mean(0).astype(np.float32)
    p_sd = p_flat.std(0).astype(np.float32)
    
    b_mu = (b_flat**0.1).mean(0).astype(np.float32)
    b_sd = (b_flat**0.1).std(0).astype(np.float32)
    
    print("Normalization stats derived.")
    
    # 3. Load Model
    print(f"Loading model from {MODEL_PATH}...")
    model = MultiInputAutoencoder(
        prism_dim=p_temp.shape[2], 
        bui_dim=b_temp.shape[2], 
        latent_dim=Z_DIM
    ).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    # 4. Processing Loop
    year_files = sorted(glob.glob(os.path.join(HIST_DIR, "state_*.npz")))
    
    for fpath in tqdm(year_files, desc="Processing Years"):
        # Extract year from filename: state_1900_bio_ema10.npz
        fname = os.path.basename(fpath)
        year = int(fname.split('_')[1])
        
        # Load Raw Data
        data = np.load(fpath)
        p_raw = data['prism'] # (H, W, 84)
        b_raw = data['bui']   # (H, W, 7)
        
        # --- STAGE 0: INFERENCE ---
        # Identify pixels that have both P and B data
        valid_pixels = (~np.isnan(p_raw).any(-1)) & (~np.isnan(b_raw).any(-1))
        
        # Create output container
        z_year = np.full((H, W, Z_DIM), np.nan, dtype=np.float32)
        
        if valid_pixels.sum() > 0:
            # Extract valid vectors
            p_in = torch.tensor(p_raw[valid_pixels], dtype=torch.float32)
            b_in = torch.tensor(b_raw[valid_pixels], dtype=torch.float32)
            
            # Normalize
            p_in = (p_in - p_mu) / (p_sd + 1e-6)
            b_in = (b_in**0.1 - b_mu) / (b_sd + 1e-6)
            
            # Infer
            with torch.no_grad():
                z_out, _ = model(p_in.to(DEVICE), b_in.to(DEVICE))
                
            # Place back
            z_year[valid_pixels] = z_out.cpu().numpy()
            
        # --- STAGE 1: SPATIAL INTERPOLATION (100km radius) ---
        # 100km / 4km pixel = 25 pixels
        z_s1, mask_s1 = fill_gaps_stage1_spatial(z_year, valid_pixels, LAND_MASK, radius_px=25)
        
        # --- STAGE 2: STATIC BACKFILL (Canada/Mexico eBird) ---
        z_s2, mask_s2 = fill_gaps_stage2_static(z_s1, mask_s1, LAND_MASK, Z_STATIC_GRID, Z_STATIC_VALID)
        
        # --- STAGE 3: FINAL CLEANUP (Nearest Neighbor) ---
        # Fills holes inside CONUS or islands that were missed by previous steps
        # STRICT CONSTRAINT: Only fill if it is LAND.
        z_final = fill_gaps_stage3_nearest(z_s2, mask_s2, LAND_MASK)
        
        # Ensure Ocean is NaN (Safety check)
        z_final[~LAND_MASK] = np.nan
        
        # Save
        out_name = f"Z_latent_{year}.npy"
        np.save(os.path.join(OUTPUT_DIR, out_name), z_final.astype(np.float32))

    print("Spatiotemporal Cube Generation Complete.")

if __name__ == "__main__":
    build_spacetime_cube()