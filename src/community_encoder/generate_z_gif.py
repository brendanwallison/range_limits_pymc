import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import imageio.v2 as imageio

def generate_rgb_timeseries():
    # --- CONFIG ---
    Z_DIR = "/home/breallis/dev/range_limits_pymc/misc_outputs/ruzicka_sweep_global/sigma_1.5/results/spacetime_cube"
    OUT_DIR = os.path.join(Z_DIR, "rgb_visualization")
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # Files
    files = sorted(glob.glob(os.path.join(Z_DIR, "Z_latent_*.npy")))
    if not files:
        print("No Z-latent files found.")
        return

    print(f"Found {len(files)} files. Computing global statistics for contrast scaling...")

    # 1. PASS 1: COMPUTE ROBUST MIN/MAX (Global Percentiles)
    # We sample every 10th year to speed up stats calculation
    sample_vals = {0: [], 1: [], 2: []}
    
    for fpath in tqdm(files[::10], desc="Scanning for stats"):
        z = np.load(fpath) # (H, W, D)
        # Flatten valid pixels
        mask = ~np.isnan(z).any(axis=-1)
        valid_z = z[mask]
        
        if len(valid_z) > 0:
            for d in range(3):
                sample_vals[d].append(valid_z[:, d])
    
    # Calculate 2nd and 98th percentiles for each channel
    bounds = {}
    for d in range(3):
        all_vals = np.concatenate(sample_vals[d])
        vmin = np.percentile(all_vals, 2)
        vmax = np.percentile(all_vals, 98)
        bounds[d] = (vmin, vmax)
        print(f"  Ch {d} (RGB index {d}): range [{vmin:.2f}, {vmax:.2f}]")

    # 2. PASS 2: RENDER FRAMES
    print("Rendering RGB frames...")
    
    for fpath in tqdm(files, desc="Saving Images"):
        year = int(os.path.basename(fpath).split('_')[2].split('.')[0])
        z = np.load(fpath)
        H, W, D = z.shape
        
        # Create RGB Container (Initialize with 0/Black)
        rgb_img = np.zeros((H, W, 3), dtype=np.float32)
        
        # Determine valid mask (Land)
        valid_mask = ~np.isnan(z).any(axis=-1)
        
        # Normalize each channel independently
        for d in range(3):
            val = z[:, :, d]
            vmin, vmax = bounds[d]
            
            # Clip and Scale to 0-1
            norm = np.clip(val, vmin, vmax)
            norm = (norm - vmin) / (vmax - vmin + 1e-6)
            
            rgb_img[:, :, d] = norm
            
        # 3. HANDLE BACKGROUND
        # Set NaNs (Ocean) to a neutral dark gray or keep black
        # We make ocean pure black (0,0,0) by default since we init with zeros.
        # But let's mask explicitly to be safe.
        rgb_img[~valid_mask] = 0.0

        # 4. PLOT AND SAVE
        plt.figure(figsize=(10, 6))
        plt.imshow(rgb_img, interpolation='nearest')
        plt.title(f"Community State (RGB = Z0, Z1, Z2) | Year: {year}", fontsize=14, color='white')
        
        # Aesthetics for dark mode look
        plt.axis('off')
        
        # Save
        out_path = os.path.join(OUT_DIR, f"frame_{year}.png")
        # Save with dark background
        plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='black')
        plt.close()

    # 3. CREATE GIF (Optional)
    print("Compiling GIF...")
    png_files = sorted(glob.glob(os.path.join(OUT_DIR, "frame_*.png")))
    if png_files:
        gif_path = os.path.join(OUT_DIR, "evolution_z_rgb.gif")
        with imageio.get_writer(gif_path, mode='I', duration=0.1) as writer: # 0.1s = 10 fps
            for filename in png_files:
                image = imageio.imread(filename)
                writer.append_data(image)
        print(f"Animation saved to {gif_path}")

if __name__ == "__main__":
    generate_rgb_timeseries()