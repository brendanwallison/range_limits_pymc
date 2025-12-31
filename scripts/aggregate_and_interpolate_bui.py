#!/usr/bin/env python3

"""
Aggregate BUI rasters from 250 m to 4 km using strict ocean masking,
then linearly interpolate annual values between 5-year snapshots.

Now includes PNG quick-look generation using a global power transform
(x^PNG_POWER), which ensures PNGs are visually comparable across years.

Quantiles per 4 km cell:
    q = [0.05, 0.25, 0.50, 0.75, 0.90, 0.97, 0.99]

PNG generation:
    - One PNG per band per year
    - Uses a global monotone power transform (default: 0.25)
    - No percentile stretching
    - PNGs are comparable across years
"""

# ============================================================
# ======================= DEFAULTS ============================
# ============================================================

BUI_DIR = "/home/breallis/datasets/HBUI/BUI"
WATER_MASK_PATH = "/home/breallis/datasets/land_mask/watermask_2025_in_bui_space.tif"

BLOCK_SIZE = 16  # 4 km = 16 × 250 m pixels

QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.90, 0.97, 0.99]

# Power transform exponent for PNG visualization
PNG_POWER = 0.25

# ============================================================
# ======================= IMPORTS =============================
# ============================================================

import os
import glob
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# ============================================================
# =================== PNG HELPER ==============================
# ============================================================

def save_png(array, out_path, cmap="viridis"):
    """
    Save a PNG using a global monotone power transform.

    - Applies x^PNG_POWER
    - NaNs rendered as zero
    - No percentile stretching (globally comparable across years)
    """
    arr = array.astype(float)
    arr[arr < 0] = 0
    vis = np.power(arr, PNG_POWER)
    vis[np.isnan(vis)] = 0
    plt.imsave(out_path, vis, cmap=cmap)

# ============================================================
# =================== AGGREGATION HELPERS =====================
# ============================================================

def aggregate_block(values):
    vals = values[~np.isnan(values)]
    if vals.size == 0:
        return [np.nan] * len(QUANTILES)
    return [float(np.quantile(vals, q)) for q in QUANTILES]


def aggregate_bui_to_4km(bui_path, watermask, out_path):
    with rasterio.open(bui_path) as src:
        profile = src.profile
        h, w = src.height, src.width

        H4 = h // BLOCK_SIZE
        W4 = w // BLOCK_SIZE

        out = np.full((len(QUANTILES), H4, W4), np.nan, dtype="float32")

        for i in range(H4):
            for j in range(W4):
                win = Window(
                    col_off=j * BLOCK_SIZE,
                    row_off=i * BLOCK_SIZE,
                    width=BLOCK_SIZE,
                    height=BLOCK_SIZE,
                )
                block = src.read(1, window=win)
                mask_block = watermask[
                    i * BLOCK_SIZE : (i + 1) * BLOCK_SIZE,
                    j * BLOCK_SIZE : (j + 1) * BLOCK_SIZE,
                ]

                block = np.where(mask_block == 1, np.nan, block)
                out[:, i, j] = aggregate_block(block.flatten())

        new_transform = profile["transform"] * Affine.scale(BLOCK_SIZE, BLOCK_SIZE)
        new_profile = profile.copy()
        new_profile.update(
            height=H4,
            width=W4,
            count=len(QUANTILES),
            transform=new_transform,
            dtype="float32",
        )

        with rasterio.open(out_path, "w", **new_profile) as dst:
            for k in range(out.shape[0]):
                dst.write(out[k], k + 1)

        # ---- PNG generation for aggregated ----
        for k in range(out.shape[0]):
            png_path = out_path.replace(".tif", f"_band{k+1}.png")
            save_png(out[k], png_path)

def load_water_mask(path):
    with rasterio.open(path) as src:
        return src.read(1)

# ============================================================
# =================== INTERPOLATION HELPERS ===================
# ============================================================

def linear_interpolate_features(years, data):
    years = np.array(years)
    all_years = np.arange(years.min(), years.max() + 1)

    nY, nF, H, W = data.shape
    out = np.full((len(all_years), nF, H, W), np.nan, dtype="float32")

    for f in range(nF):
        for i in range(H):
            for j in range(W):
                ts = data[:, f, i, j]
                if np.all(np.isnan(ts)):
                    continue
                valid = ~np.isnan(ts)
                if valid.sum() == 1:
                    out[:, f, i, j] = ts[valid][0]
                    continue
                f_interp = interp1d(
                    years[valid],
                    ts[valid],
                    kind="linear",
                    bounds_error=False,
                    fill_value="extrapolate",
                )
                out[:, f, i, j] = f_interp(all_years)

    # Enforce quantile ordering by sorting across features
    for t in range(len(all_years)):
        slice_t = out[t].reshape(nF, -1)
        slice_t_sorted = np.sort(slice_t, axis=0)
        out[t] = slice_t_sorted.reshape(nF, H, W)

    return all_years, out

# ============================================================
# ========================= MAIN =============================
# ============================================================

def main():
    watermask = load_water_mask(WATER_MASK_PATH)

    bui_files = sorted(glob.glob(os.path.join(BUI_DIR, "*_BUI.tif")))

    aggregated_paths = []
    years = []

    # -------------------------------
    # 1. AGGREGATION
    # -------------------------------
    for path in bui_files:
        fname = os.path.basename(path)
        try:
            year = int(fname.split("_")[0])
        except ValueError:
            continue

        years.append(year)

        out_path = os.path.join(BUI_DIR, f"{year}_BUI_4km.tif")
        if os.path.exists(out_path):
            aggregated_paths.append(out_path)
            continue

        print(f"Aggregating {fname} → {out_path}")
        aggregate_bui_to_4km(path, watermask, out_path)
        aggregated_paths.append(out_path)

    years, aggregated_paths = zip(*sorted(zip(years, aggregated_paths)))

    # -------------------------------
    # 2. INTERPOLATION
    # -------------------------------
    arrays = []
    for path in aggregated_paths:
        with rasterio.open(path) as src:
            arrays.append(src.read())

    data = np.stack(arrays, axis=0)

    all_years, interp_data = linear_interpolate_features(np.array(years), data)

    with rasterio.open(aggregated_paths[0]) as ref:
        profile = ref.profile

    for idx, yr in enumerate(all_years):
        out_path = os.path.join(BUI_DIR, f"{yr}_BUI_4km_interp.tif")
        if os.path.exists(out_path):
            continue

        print(f"Saving interpolated {yr} → {out_path}")
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(interp_data[idx])

        # ---- PNG generation for interpolated ----
        for k in range(interp_data.shape[1]):
            png_path = out_path.replace(".tif", f"_band{k+1}.png")
            save_png(interp_data[idx, k], png_path)


if __name__ == "__main__":
    main()