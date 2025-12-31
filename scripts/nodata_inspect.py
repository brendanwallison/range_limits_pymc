#!/usr/bin/env python3

import rasterio
import numpy as np
import xarray as xr
from pathlib import Path

# ------------------------------------------------------------
# NetCDF (PRISM) nodata inspection
# ------------------------------------------------------------

def inspect_netcdf_nodata(nc_path, var_name=None):
    """
    Inspect nodata / fill values in a PRISM-style NetCDF file.
    Handles cases where some variables are strings.
    """
    print(f"\n=== NetCDF inspection: {nc_path} ===")
    ds = xr.open_dataset(nc_path)

    # List all variables
    print("Variables in file:")
    for v in ds.data_vars:
        print(f"  - {v}: dtype={ds[v].dtype}")

    # If user didn't specify a variable, pick the first numeric one
    if var_name is None:
        numeric_vars = [v for v in ds.data_vars if np.issubdtype(ds[v].dtype, np.number)]
        if not numeric_vars:
            print("No numeric data variables found.")
            return
        var_name = numeric_vars[0]
        print(f"\nAuto-selected numeric variable: {var_name}")

    da = ds[var_name]
    attrs = da.attrs

    fill = attrs.get("_FillValue", None)
    missing = attrs.get("missing_value", None)

    print(f"\nInspecting variable: {var_name}")
    print(f"_FillValue: {fill}")
    print(f"missing_value: {missing}")

    arr = da.values

    # If the array is not numeric, bail out safely
    if not np.issubdtype(arr.dtype, np.number):
        print(f"Variable {var_name} is not numeric (dtype={arr.dtype}). Cannot compute min/max.")
        return

    # Build mask
    mask = np.ones(arr.shape, dtype=bool)
    for fv in [fill, missing]:
        if fv is not None:
            mask &= (arr != fv)

    valid = arr[mask]

    if valid.size == 0:
        print("No valid data found (all nodata?).")
    else:
        print(f"Valid data range: min={valid.min()}, max={valid.max()}")

    ds.close()


def inspect_netcdf_numeric_detail(nc_path, var_name="Band1"):
    print(f"\n=== Detailed NetCDF numeric inspection: {nc_path} ===")
    ds = xr.open_dataset(nc_path)
    da = ds[var_name]
    arr = da.values

    print(f"Variable: {var_name}, dtype={arr.dtype}, shape={arr.shape}")

    # Count NaNs
    if np.issubdtype(arr.dtype, np.floating):
        nan_mask = np.isnan(arr)
        n_nan = nan_mask.sum()
        n_total = arr.size
        n_valid = n_total - n_nan
        print(f"Total pixels: {n_total}")
        print(f"NaN pixels:   {n_nan}")
        print(f"Valid pixels: {n_valid}")
        print(f"NaN fraction: {n_nan / n_total:.4f}")

        if n_valid > 0:
            valid_vals = arr[~nan_mask]
            print(f"Valid value range (ignoring NaNs): "
                  f"min={np.nanmin(valid_vals)}, max={np.nanmax(valid_vals)}")
        else:
            print("All pixels are NaN (no valid values).")
    else:
        print("Variable is not floating type; NaN analysis not applicable.")

    ds.close()

# ------------------------------------------------------------
# GeoTIFF (eBird, BUI) nodata inspection
# ------------------------------------------------------------

def inspect_geotiff_nodata(tif_path):
    """
    Inspect nodata in a GeoTIFF (eBird or BUI).
    Handles the case where nodata is NaN.
    """
    print(f"\n=== GeoTIFF inspection: {tif_path} ===")
    with rasterio.open(tif_path) as src:
        nodata = src.nodata
        print(f"Nodata tag: {nodata}")

        arr = src.read(1)
        total = arr.size

        # Case 1: nodata is explicitly NaN
        if nodata is not None and isinstance(nodata, float) and np.isnan(nodata):
            nodata_mask = np.isnan(arr)
            n_nodata = nodata_mask.sum()
            valid_mask = ~nodata_mask
        # Case 2: nodata is a finite value
        elif nodata is not None:
            nodata_mask = (arr == nodata)
            n_nodata = nodata_mask.sum()
            valid_mask = ~nodata_mask
        # Case 3: no nodata defined
        else:
            print("No nodata tag found.")
            n_nodata = 0
            valid_mask = np.ones_like(arr, dtype=bool)

        n_valid = valid_mask.sum()
        print(f"Nodata pixel count: {n_nodata}")
        print(f"Valid pixel count: {n_valid}")

        if n_valid > 0:
            valid_vals = arr[valid_mask]
            # Robust range ignoring NaNs inside valid_vals
            vmin = np.nanmin(valid_vals)
            vmax = np.nanmax(valid_vals)
            print(f"Valid data range (ignoring NaNs): min={vmin}, max={vmax}")
        else:
            print("No valid pixels found.")


# ------------------------------------------------------------
# Folder helper
# ------------------------------------------------------------

def inspect_folder(path, pattern="*.tif", limit=3):
    """
    Inspect a few files in a folder.
    """
    files = sorted(Path(path).glob(pattern))
    if not files:
        print(f"No files found in {path} matching {pattern}")
        return

    for f in files[:limit]:
        if f.suffix == ".tif":
            inspect_geotiff_nodata(f)
        elif f.suffix == ".nc":
            inspect_netcdf_nodata(f)
        else:
            print(f"Skipping unsupported file: {f}")


# ------------------------------------------------------------
# Example usage (edit paths as needed)
# ------------------------------------------------------------
if __name__ == "__main__":
    print("\n--- Example usage ---")

    # PRISM example
    # inspect_netcdf_nodata("/home/breallis/datasets/prism_monthly_4km/prism_tdmean_us_25m_200311.nc")
    # inspect_netcdf_numeric_detail(
    # "/home/breallis/datasets/prism_monthly_4km/prism_tdmean_us_25m_200311.nc")

    # eBird example
    # inspect_folder("/home/breallis/datasets/ebird_abundances", "*.tif", limit=3)

    # BUI example
    inspect_folder("/home/breallis/datasets/HBUI/BUI", "*.tif", limit=3)

    print("\nEdit the paths in the __main__ block to inspect your datasets.\n")