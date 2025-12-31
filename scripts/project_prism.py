#!/usr/bin/env python3
"""
Reproject all PRISM monthly .nc files onto the canonical 4 km BUI grid.

Features:
    - Proper handling of PRISM nodata (_FillValue, missing_value, common sentinels)
    - Convert nodata → NaN AFTER reprojection
    - Preserve NaNs through reprojection
    - Apply ocean mask cleanly to all bands
    - Assert no negative values remain in output
    - Save GeoTIFF + PNG quick-look
    - Robust to single-band (2D) or multi-band (3D) rasters
"""

import os
import glob
import numpy as np
import xarray as xr
import rioxarray as rxr
import rasterio
from rasterio.crs import CRS
import matplotlib.pyplot as plt

# -----------------------------
# Configuration
# -----------------------------
PRISM_DIR = "/home/breallis/datasets/prism_monthly_4km"
BUI_REF = "/home/breallis/datasets/HBUI/BUI/2020_BUI_4km.tif"
OCEAN_MASK = "/home/breallis/datasets/land_mask/ocean_mask_4km.tif"

PNG_POWER = 0.25
OUT_DIR = "/home/breallis/datasets/prism_monthly_4km_albers"

EXPECTED_CRS = CRS.from_epsg(4269)  # NAD83 geographic
SENTINELS = [-9999, -999, -99]      # common sentinel values


# -----------------------------
# Utility functions
# -----------------------------
def check_prism_crs(ds, nc_path):
    """Accept any CRS that is NAD83 geographic."""
    if "crs" in ds.data_vars:
        crs_da = ds["crs"]
        wkt = crs_da.attrs.get("spatial_ref") or crs_da.attrs.get("crs_wkt")
        if wkt is not None:
            file_crs = CRS.from_wkt(wkt)
            if file_crs == EXPECTED_CRS or (file_crs.is_geographic and "NAD83" in file_crs.to_wkt()):
                return file_crs
            raise ValueError(f"Unrecognized CRS in {nc_path}: {file_crs}. Expected NAD83 geographic.")
    print(f"Warning: No usable CRS metadata found in {nc_path}; assuming EPSG:4269 (NAD83 geographic).")
    return EXPECTED_CRS


def save_png(array, out_path, cmap="viridis"):
    """Save a quick-look PNG from an array."""
    arr = array.astype(float)
    arr[np.isnan(arr)] = 0
    plt.imsave(out_path, arr, cmap=cmap)


# -----------------------------
# Main processing
# -----------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load reference BUI grid
    bui_ref = rxr.open_rasterio(BUI_REF)

    # Load ocean mask
    with rasterio.open(OCEAN_MASK) as src:
        ocean_mask = src.read(1)

    # Find all PRISM .nc files
    nc_files = sorted(glob.glob(os.path.join(PRISM_DIR, "*.nc")))

    for nc_path in nc_files:
        fname = os.path.basename(nc_path)
        out_tif = os.path.join(OUT_DIR, fname.replace(".nc", "_bui4km.tif"))
        out_png = os.path.join(OUT_DIR, fname.replace(".nc", "_bui4km.png"))

        if os.path.exists(out_tif):
            print(f"Skipping existing: {out_tif}")
            continue

        print(f"Processing {fname}")

        # -----------------------------
        # Open NetCDF and select Band1
        # -----------------------------
        ds = xr.open_dataset(nc_path)
        if "Band1" not in ds.data_vars:
            raise ValueError(f"'Band1' not found in {nc_path}")
        da = ds["Band1"]

        # Handle nodata values
        nodata_vals = []
        fill = da.attrs.get("_FillValue")
        missing = da.attrs.get("missing_value")
        if fill is not None:
            nodata_vals.append(float(fill))
        if missing is not None and missing != fill:
            nodata_vals.append(float(missing))

        if nodata_vals:
            mask = np.isin(da.values, nodata_vals)
            da = da.where(~mask)

        # Determine and attach CRS
        file_crs = check_prism_crs(ds, nc_path)
        da = da.rio.write_crs(file_crs, inplace=False)

        # -----------------------------
        # Reproject to BUI 4 km grid
        # -----------------------------
        da_reproj = da.rio.reproject_match(bui_ref)

        # Convert common sentinel values to NaN
        for s in SENTINELS:
            da_reproj = da_reproj.where(da_reproj != s, np.nan)

        # -----------------------------
        # Apply ocean mask robustly
        # -----------------------------
        data = da_reproj.values.astype("float32")

        if data.shape[-2:] != ocean_mask.shape:
            raise ValueError(
                f"Shape mismatch after reprojection for {nc_path}: "
                f"da_reproj.shape={data.shape}, ocean_mask.shape={ocean_mask.shape}"
            )

        if data.ndim == 2:  # single band
            data[ocean_mask == 1] = np.nan
        elif data.ndim == 3:  # multi-band
            for b in range(data.shape[0]):
                data[b, ocean_mask == 1] = np.nan
        else:
            raise ValueError(f"Unexpected data shape: {data.shape}")

        # Assign cleaned values back to DataArray
        da_reproj.values = data

        # -----------------------------
        # Safety check
        # -----------------------------
        neg_mask = (data < -1000) & ~np.isnan(data)
        if np.any(neg_mask):
            raise ValueError(
                f"Negative values found after sentinel replacement and masking: "
                f"min={np.nanmin(data)}, example={data[neg_mask][0]}"
            )

        # -----------------------------
        # Save GeoTIFF
        # -----------------------------
        da_reproj.rio.to_raster(out_tif)

        # -----------------------------
        # Save PNG quick-look (band 0 for 3D, or 2D array)
        # -----------------------------
        save_png(data[0] if data.ndim == 3 else data, out_png)

        print(f"Saved → {out_tif}")
        print(f"Saved → {out_png}")


if __name__ == "__main__":
    main()
