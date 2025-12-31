import rasterio
from pathlib import Path

folder = Path("/home/breallis/datasets/ebird_abundances")

meta = []

for tif in folder.glob("*.tif"):
    with rasterio.open(tif) as src:
        meta.append({
            "file": tif.name,
            "crs": src.crs,
            "res": src.res,
            "bounds": src.bounds,
            "width": src.width,
            "height": src.height,
            "transform": src.transform
        })

# Quick check: do all resolutions match?
resolutions = {m["res"] for m in meta}
print("Unique resolutions:", resolutions)

# Check extents
bounds = {m["bounds"] for m in meta}
print("Unique bounds:", len(bounds))

crs_set = {m["crs"] for m in meta}
print("Unique CRS:", crs_set)

transform_set = {m["transform"] for m in meta}
print("Unique transforms:", len(transform_set))
