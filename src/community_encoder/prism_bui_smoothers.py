import os
import glob
import re
import numpy as np
import rasterio
from tqdm import tqdm

# ============================================================
# 1. BUI STREAMER (Multi-Band + Optional Interpolation)
# ============================================================
class BuiStreamer:
    def __init__(self, bui_dir, start_year, end_year, alpha, interpolate=False):
        self.bui_dir = bui_dir
        self.years = range(start_year, end_year + 1)
        self.alpha = alpha
        self.state = None
        self.interpolate = interpolate
        self.bands = range(1, 8) # Bands 1 through 7

        # Index available files by Year and Band
        # Expected: 1810_BUI_4km_interp_band1.png
        self.anchors = {} 
        
        print(f"[BUI] Scanning for multi-band files in {bui_dir}...")
        pattern = re.compile(r".*(\d{4})_BUI_4km_interp_band(\d)\.png$")
        
        files = glob.glob(os.path.join(bui_dir, "*_BUI_4km_interp_band*.png"))
        for f in files:
            fname = os.path.basename(f)
            match = pattern.match(fname)
            if match:
                y = int(match.group(1))
                b = int(match.group(2))
                if y not in self.anchors: self.anchors[y] = {}
                self.anchors[y][b] = f

        self.sorted_years = sorted(self.anchors.keys())
        print(f"[BUI] Found {len(self.sorted_years)} anchor years.")

    def _load_year_stack(self, year):
        """Loads bands 1-7 for a specific year. Returns (H, W, 7) or None."""
        if year not in self.anchors:
            return None
        
        # Ensure all 7 bands exist
        year_files = self.anchors[year]
        if len(year_files) != 7:
            return None
            
        stack = []
        try:
            for b in self.bands:
                fpath = year_files[b]
                with rasterio.open(fpath) as src:
                    stack.append(src.read(1).astype(np.float32))
            return np.stack(stack, axis=-1)
        except Exception as e:
            print(f"Error loading BUI {year}: {e}")
            return None

    def _get_data_for_year(self, year):
        """Handles interpolation or raw loading."""
        # 1. Try Direct Load
        raw = self._load_year_stack(year)
        if raw is not None:
            return raw
            
        # 2. If Interpolation is OFF, return None (Persistence)
        if not self.interpolate:
            return None
            
        # 3. Interpolation Logic
        past = [y for y in self.sorted_years if y < year]
        future = [y for y in self.sorted_years if y > year]
        
        if not past and not future: return None
        if not past: return self._load_year_stack(future[0])
        if not future: return self._load_year_stack(past[-1])
        
        y1, y2 = past[-1], future[0]
        data1 = self._load_year_stack(y1)
        data2 = self._load_year_stack(y2)
        
        if data1 is None or data2 is None: return None
        
        w = (year - y1) / (y2 - y1)
        return (1 - w) * data1 + w * data2

    def __iter__(self):
        print(f"[BUI] Stream started ({self.years.start}-{self.years.stop-1})...")
        for year in self.years:
            curr = self._get_data_for_year(year)
            
            # Update EMA State
            if self.state is None:
                if curr is not None:
                    self.state = curr
            else:
                if curr is not None:
                    self.state = self.alpha * curr + (1 - self.alpha) * self.state
                
            yield year, self.state

# ============================================================
# 2. PRISM STREAMER (Strict Variables + Bio-Year)
# ============================================================
class PrismStreamer:
    def __init__(self, prism_dir, start_year, end_year, alpha):
        self.prism_dir = prism_dir
        self.years = range(start_year, end_year + 1)
        self.alpha = alpha
        self.state = None
        
        self.VARS = ['ppt', 'tdmean', 'tmax', 'tmean', 'tmin', 'vpdmax', 'vpdmin']
        self.file_template = "prism_{var}_us_25m_{date}_bui4km.tif"

    def _load_month_stack(self, yyyymm):
        """Loads all 7 variables for a specific month."""
        stack = []
        for var in self.VARS:
            fname = self.file_template.format(var=var, date=yyyymm)
            fpath = os.path.join(self.prism_dir, fname)
            
            if not os.path.exists(fpath):
                raise FileNotFoundError(f"Missing required PRISM file: {fname}")
                
            with rasterio.open(fpath) as src:
                stack.append(src.read(1).astype(np.float32))
        
        return np.stack(stack, axis=-1)

    def _get_bio_year_stack(self, target_year):
        """Constructs stack for Aug(T-1) -> Jul(T)."""
        prev_year = target_year - 1
        
        # Months: Aug-Dec (Prev) + Jan-Jul (Curr)
        month_keys = []
        for m in ["08", "09", "10", "11", "12"]:
            month_keys.append(f"{prev_year}{m}")
        for m in ["01", "02", "03", "04", "05", "06", "07"]:
            month_keys.append(f"{target_year}{m}")
            
        full_year_stack = []
        try:
            for date_key in month_keys:
                m_stack = self._load_month_stack(date_key)
                full_year_stack.append(m_stack)
        except FileNotFoundError:
            return None
            
        return np.concatenate(full_year_stack, axis=-1)

    def __iter__(self):
        print(f"[PRISM] Stream started ({self.years.start}-{self.years.stop-1})...")
        print(f"[PRISM] Variables: {self.VARS} (Total 84 channels/year)")
        
        for year in self.years:
            curr = self._get_bio_year_stack(year)
            
            if self.state is None:
                if curr is not None:
                    self.state = curr
            else:
                if curr is not None:
                    self.state = self.alpha * curr + (1 - self.alpha) * self.state
                
            yield year, self.state

# ============================================================
# 3. SYNCHRONIZED EXECUTION
# ============================================================
def run_simulation(prism_dir, bui_dir, out_dir, interpolate_bui=False):
    os.makedirs(out_dir, exist_ok=True)
    states_out_dir = os.path.join(out_dir, "yearly_states")
    os.makedirs(states_out_dir, exist_ok=True)
    
    # Configuration
    START_YEAR = 1896  # Needs Aug 1895
    END_YEAR = 2024
    SAMPLE_START = 1900
    EMA_TAU = 10.0
    ALPHA = 1.0 - np.exp(-1.0 / EMA_TAU)
    SAMPLES_PER_YEAR = 20000
    
    # 1. Setup Mask
    ref_files = glob.glob(os.path.join(prism_dir, "prism_ppt_*.tif"))
    if not ref_files:
        raise FileNotFoundError("Could not find any PRISM files for mask generation.")
    ref_path = ref_files[0]
        
    print(f"Using reference file for mask: {os.path.basename(ref_path)}")
    with rasterio.open(ref_path) as src:
        ref_data = src.read(1)
        valid_y, valid_x = np.where(ref_data > -1000)
        valid_coords = list(zip(valid_y, valid_x))
    
    print(f"Valid Land Pixels: {len(valid_coords)}")
    
    # 2. Initialize Generators
    gen_prism = PrismStreamer(prism_dir, START_YEAR, END_YEAR, ALPHA)
    gen_bui = BuiStreamer(bui_dir, START_YEAR, END_YEAR, ALPHA, interpolate=interpolate_bui)
    
    all_vectors = []
    
    # 3. Lockstep Iteration
    print(f"Starting Simulation. Saving yearly states to {states_out_dir}...")
    
    for (y_p, s_p), (y_b, s_b) in zip(gen_prism, gen_bui):
        assert y_p == y_b, f"Sync Error: {y_p} != {y_b}"
        year = y_p
        
        # Proceed only if both states are initialized
        if s_p is not None and s_b is not None:
            
            if year >= SAMPLE_START:
                # --- A. Random Sampling (for Autoencoder Training) ---
                indices = np.random.choice(len(valid_coords), SAMPLES_PER_YEAR, replace=False)
                rows = [valid_coords[i][0] for i in indices]
                cols = [valid_coords[i][1] for i in indices]
                
                p_vec = s_p[rows, cols] # (N, 84)
                b_vec = s_b[rows, cols] # (N, 7)
                
                # Strict NaN Filter
                mask = ~np.isnan(p_vec).any(axis=1) & ~np.isnan(b_vec).any(axis=1)
                
                if mask.sum() > 0:
                    combined = np.concatenate([p_vec[mask], b_vec[mask]], axis=1)
                    all_vectors.append(combined)

                # --- B. Save Whole Grid State (for Inference Cube) ---
                # We save every valid year to build the full spacetime cube later
                # Format: state_YYYY_bio_ema10.npz
                state_fname = f"state_{year}_bio_ema10.npz"
                np.savez_compressed(
                    os.path.join(states_out_dir, state_fname),
                    prism=s_p, 
                    bui=s_b
                )
        
        if year % 5 == 0:
            print(f"Processed {year}...")

    # 4. Save Final Training Bag
    if all_vectors:
        full_arr = np.concatenate(all_vectors, axis=0)
        print(f"Saving {len(full_arr)} historical training vectors to {out_dir}...")
        np.save(os.path.join(out_dir, "history_vectors_bio_ema10.npy"), full_arr.astype(np.float32))
    else:
        print("WARNING: No vectors sampled! Check data availability.")

if __name__ == "__main__":
    DATA_DIR = "/home/breallis/datasets"
    run_simulation(
        prism_dir=f"{DATA_DIR}/prism_monthly_4km_albers",
        bui_dir=f"{DATA_DIR}/HBUI/BUI_4km_interp",
        out_dir=f"{DATA_DIR}/smoothed_prism_bui",
        interpolate_bui=False 
    )