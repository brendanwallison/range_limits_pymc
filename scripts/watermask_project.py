import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
import matplotlib.pyplot as plt

def reproject_water_mask(mask_path, bui_path, out_path="/home/breallis/datasets/land_mask/watermask_reproj.tif"):
    """
    Reproject the water mask (EPSG:4326) into the BUI CRS/grid.
    Saves the reprojected mask as a GeoTIFF and returns the array.
    """

    # Read BUI metadata (target grid)
    with rasterio.open(bui_path) as bui_src:
        bui_profile = bui_src.profile.copy()
        bui_shape = (bui_src.height, bui_src.width)
        bui_crs = bui_src.crs
        bui_transform = bui_src.transform

    # Read water mask (source grid)
    with rasterio.open(mask_path) as mask_src:
        mask = mask_src.read(1)
        mask_transform = mask_src.transform
        mask_crs = rasterio.crs.CRS.from_epsg(4326)  # override missing CRS

    # Prepare output array
    mask_reproj = np.zeros(bui_shape, dtype=np.uint8)

    # Reproject using nearest neighbor (categorical)
    reproject(
        source=mask,
        destination=mask_reproj,
        src_transform=mask_transform,
        src_crs=mask_crs,
        dst_transform=bui_transform,
        dst_crs=bui_crs,
        resampling=Resampling.nearest
    )

    # Save as GeoTIFF
    out_profile = bui_profile.copy()
    out_profile.update(
        dtype="uint8",
        count=1,
        nodata=None,
        compress="lzw"
    )

    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(mask_reproj, 1)

    return mask_reproj

def extract_ocean_bui(bui_path, mask_reproj):
    with rasterio.open(bui_path) as src:
        bui = src.read(1).astype("float32")

    # Ocean = mask == 1
    ocean_vals = bui[mask_reproj == 1]

    # Drop NaNs (BUI may already have NaNs over ocean)
    ocean_vals = ocean_vals[~np.isnan(ocean_vals)]

    return ocean_vals


def visualize_ocean_bui(ocean_vals, log=False, power=None, png="ocean_bui_hist.png"):
    vals = ocean_vals.copy()

    # Power-law transform
    if power is not None:
        if not (0 < power <= 1):
            raise ValueError("power must be in (0,1]")
        positive = vals > 0
        vals = vals.astype("float32")
        vals[positive] = vals[positive] ** power
        vals[~positive] = 0.0

    # Log transform (only if power not used)
    elif log:
        positive = vals > 0
        vals = vals.astype("float32")
        vals = np.log10(vals[positive])  # drop zeros

    plt.figure(figsize=(10,6))
    plt.hist(vals, bins=100, color="steelblue", alpha=0.8)
    plt.title("Distribution of BUI values over ocean")
    plt.xlabel("BUI (transformed)" if (log or power) else "BUI")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(png, dpi=150)
    plt.close()


def count_ocean_bui(ocean_vals):
    """
    Given a 1-D array of BUI values extracted over ocean,
    count how many are > 0 and return summary stats.
    """

    # Drop NaNs (BUI may already have NaNs over ocean)
    vals = ocean_vals[~np.isnan(ocean_vals)]

    # Count positives
    positive_mask = vals > 0
    n_positive = np.sum(positive_mask)
    n_total = vals.size

    # Summary stats for the positive values
    if n_positive > 0:
        stats = {
            "min": float(np.min(vals[positive_mask])),
            "max": float(np.max(vals[positive_mask])),
            "mean": float(np.mean(vals[positive_mask])),
            "median": float(np.median(vals[positive_mask])),
        }
    else:
        stats = None

    return n_positive, n_total, stats

def visualize_mask(mask_array, png_path="watermask_reproj.png"):
    """
    Visualize a reprojected water mask (0 = land, 1 = water).
    Saves a PNG and returns nothing.
    """

    plt.figure(figsize=(10, 6))
    plt.imshow(mask_array, cmap="Blues", interpolation="nearest")
    plt.title("Reprojected Water Mask (1 = water, 0 = land)")
    plt.colorbar(label="Mask value")
    plt.axis("off")

    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    mask_path = "/home/breallis/datasets/land_mask/watermask_2025.nc"
    bui_path = "/home/breallis/datasets/HBUI/BUI/2020_BUI.tif"

    mask_reproj = reproject_water_mask(
        mask_path=mask_path,
        bui_path=bui_path,
        out_path="/home/breallis/datasets/land_mask/watermask_2025_in_bui_space.tif"
    )

    visualize_mask(mask_reproj, png_path="watermask_reproj.png")

    ocean_vals = extract_ocean_bui(bui_path, mask_reproj)

    n_positive, n_total, stats = count_ocean_bui(ocean_vals)

    print("Total ocean pixels with valid BUI:", n_total)
    print("Ocean pixels with BUI > 0:", n_positive)

    if stats:
        print("Stats for BUI > 0 over ocean:", stats)
    else:
        print("No positive BUI values over ocean.")

    # visualize_ocean_bui(
    #     ocean_vals,
    #     log=False,
    #     power=0.5,   # or None
    #     png="ocean_bui_hist.png"
    # )


