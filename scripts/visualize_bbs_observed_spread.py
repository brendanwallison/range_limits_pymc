import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import gaussian_filter
from matplotlib.lines import Line2D

def create_smoothed_gif(
    npz_path="/home/breallis/datasets/bbs_2024_release/bbs_data_for_python.npz", 
    output_file="/home/breallis/datasets/bbs_2024_release/spread_smoothed.gif",
    sigma=2.0,         # Spatial smoothing (Gaussian sigma in pixels)
    window_radius=2    # Temporal smoothing (+/- years. 2 = 5-year window)
):
    print(f"Loading {npz_path}...")
    data = np.load(npz_path)

    # Extract dimensions
    Nx, Ny = data["Nx"], data["Ny"]
    land_mask = data["land"].astype(bool)
    
    # Extract observations
    obs_rows = data["obs_rows"]
    obs_cols = data["obs_cols"]
    obs_year = data["obs_year"]
    obs_counts = data["observed_results"]
    
    # Western Hull for context
    init_rows = data["initpop_rows"]
    init_cols = data["initpop_cols"]
    western_mask = np.zeros((Ny, Nx), dtype=bool)
    western_mask[init_rows, init_cols] = True

    # Use Full Range (No cropping)
    min_obs_year = obs_year.min()
    max_obs_year = obs_year.max()
    years = np.arange(min_obs_year, max_obs_year + 1)
    
    print(f"Generating animation for {min_obs_year}-{max_obs_year}")
    print(f"  Spatial Sigma: {sigma}")
    print(f"  Temporal Window: +/- {window_radius} years (Total {window_radius*2 + 1}y)")

    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Colormap
    cmap = plt.cm.RdYlBu_r 
    cmap.set_bad('lightgrey') 

    # Base land mask
    base_grid = np.full((Ny, Nx), np.nan)
    masked_base = np.ma.masked_where(~land_mask, base_grid)
    
    im = ax.imshow(masked_base, cmap=cmap, vmin=0, vmax=1.0, origin='upper')
    
    # Context layers
    ax.contour(western_mask, levels=[0.5], colors='black', linewidths=1.5, linestyles='--')
    title_text = ax.text(0.5, 1.02, "", transform=ax.transAxes, ha="center", fontsize=14)
    
    # Legend
    legend_elements = [
        Line2D([0], [0], color='black', lw=1.5, linestyle='--', label='Native Hull'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=cmap(0.0), markersize=10, label='Absence'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=cmap(1.0), markersize=10, label='Presence'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    ax.set_axis_off()

    def get_smoothed_frame(center_year):
        counts_grid = np.zeros((Ny, Nx))
        visits_grid = np.zeros((Ny, Nx))

        # Loop through the temporal window
        for offset in range(-window_radius, window_radius + 1):
            target_year = center_year + offset
            
            # MIRRORING LOGIC: Reflect data at boundaries
            if target_year < min_obs_year:
                eff_year = min_obs_year + (min_obs_year - target_year)
            elif target_year > max_obs_year:
                eff_year = max_obs_year - (target_year - max_obs_year)
            else:
                eff_year = target_year
                
            # Clamp in case reflection goes out of bounds (for very large windows)
            eff_year = max(min_obs_year, min(eff_year, max_obs_year))

            # Grab data for this effective year
            mask = (obs_year == eff_year)
            y_rows = obs_rows[mask]
            y_cols = obs_cols[mask]
            y_counts = obs_counts[mask]
            
            if len(y_rows) > 0:
                np.add.at(counts_grid, (y_rows, y_cols), y_counts)
                np.add.at(visits_grid, (y_rows, y_cols), 1)
        
        # Spatial Smoothing (Gaussian Blur)
        smooth_counts = gaussian_filter(counts_grid, sigma=sigma)
        smooth_visits = gaussian_filter(visits_grid, sigma=sigma)
        
        # Calculate Density
        with np.errstate(divide='ignore', invalid='ignore'):
            density = smooth_counts / smooth_visits
            
            # Mask areas with low visit weight (prevents extrapolation into unsampled zones)
            # We scale the cutoff by sigma to account for the blur spreading the "visit mass"
            density[smooth_visits < (0.05 / sigma)] = np.nan
            
        # Normalize (0 to 1 based on a cap of 50 birds/route)
        density = np.clip(density / 50.0, 0, 1) 
        
        return density

    def update(year):
        grid = get_smoothed_frame(year)
        grid_masked = np.ma.masked_where(~land_mask, grid)
        im.set_data(grid_masked)
        title_text.set_text(f"Year: {year} (±{window_radius}y, σ={sigma})")
        return [im, title_text]

    ani = animation.FuncAnimation(
        fig, update, frames=years, interval=150, blit=True
    )
    
    ani.save(output_file, writer='pillow', fps=3)
    print(f"Saved smoothed animation to {output_file}")

if __name__ == "__main__":
    create_smoothed_gif(sigma=3.0, window_radius=3)