import rasterio
from rasterio.warp import transform
from rasterio.crs import CRS

mask_path = "/home/breallis/datasets/land_mask/watermask_2025.nc"

with rasterio.open(mask_path) as src:
    # Override CRS manually (the file has no CRS metadata)
    src_crs = CRS.from_epsg(4326)

    test_points = {
        "deep_ocean_1": (0.0, -30.0),
        "deep_ocean_2": (0.0, -150.0),
        "interior_us": (40.0, -100.0),
        "sahara": (23.0, 10.0),
        "europe": (48.0, 10.0),
    }

    arr = src.read(1)

    for name, (lat, lon) in test_points.items():
        # transform() needs CRS objects, not strings
        xs, ys = transform(CRS.from_epsg(4326), src_crs, [lon], [lat])
        x, y = xs[0], ys[0]

        row, col = src.index(x, y)
        val = arr[row, col]

        print(f"{name}: {val}")