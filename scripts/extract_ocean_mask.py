#!/usr/bin/env python3

"""
Create a 4 km ocean mask from any aggregated BUI 4 km raster.

Convention:
    1 = ocean
    0 = land

Logic:
    A 4 km cell is ocean if all quantile bands are NaN.

Also generates a PNG quick-look using a categorical colormap.
"""

import rasterio
import numpy as np
import matplotlib.pyplot as plt

# Path to any aggregated BUI 4 km file
BUI_4KM_PATH = "/home/breallis/datasets/HBUI/BUI/2020_BUI_4km.tif"

# Output mask path
OUT_MASK = "/home/breallis/datasets/land_mask/ocean_mask_4km.tif"

# PNG quick-look path
OUT_PNG = "/home/breallis/datasets/land_mask/ocean_mask_4km.png"


def save_mask_png(mask, out_path):
    """
    Save a PNG for a binary mask using a categorical colormap.

    - 0 = land (light gray)
    - 1 = ocean (blue)
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(mask, cmap="Blues")  # ocean = blue, land = white
    plt.axis("off")
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def main():
    # Load any BUI 4 km raster
    with rasterio.open(BUI_4KM_PATH) as src:
        arr = src.read()  # shape: (7, H, W)
        profile = src.profile

    # Ocean = all bands NaN
    ocean = np.all(np.isnan(arr), axis=0).astype("uint8")

    # Update profile for single-band mask
    profile.update(count=1, dtype="uint8")

    # Save mask
    with rasterio.open(OUT_MASK, "w", **profile) as dst:
        dst.write(ocean, 1)

    print(f"Saved 4 km ocean mask → {OUT_MASK}")

    # Save PNG quick-look
    save_mask_png(ocean, OUT_PNG)
    print(f"Saved PNG quick-look → {OUT_PNG}")


if __name__ == "__main__":
    main()