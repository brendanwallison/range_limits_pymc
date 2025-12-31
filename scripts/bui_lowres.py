import argparse
import numpy as np
import rasterio
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Hardcoded defaults (used when no CLI args are provided)
# ---------------------------------------------------------
DEFAULT_BUI_PATH = "/home/breallis/datasets/HBUI/BUI/2020_BUI.tif"
DEFAULT_FACTOR = 16          # 16 → ~4 km
DEFAULT_PNG_PATH = "/home/breallis/datasets/HBUI/bui_2020_lowres_powerdot1.png"
DEFAULT_LOGSCALE = False     # default: linear scale
DEFAULT_POWER = 0.1         # None = no power transform


def visualize_bui_lowres(bui_path, factor=16, png_path="bui_lowres.png",
                         logscale=False, power=None):
    """
    Load a BUI raster, downsample by an integer factor using nanmean,
    optionally apply log or power-law scaling, display it, and save a PNG.
    """

    with rasterio.open(bui_path) as src:
        bui = src.read(1).astype("float32")
        h, w = bui.shape

    # Trim edges so dimensions are divisible by factor
    H = h // factor
    W = w // factor
    bui = bui[:H*factor, :W*factor]

    # Reshape into blocks: (H, factor, W, factor)
    blocks = bui.reshape(H, factor, W, factor)

    # Compute block means ignoring NaNs
    bui_lowres = np.nanmean(blocks, axis=(1, 3))

    # ---------------------------------------------------------
    # Power-law transform (takes precedence over log)
    # ---------------------------------------------------------
    if power is not None:
        if not (0 < power <= 1):
            raise ValueError("Power exponent must be between 0 and 1.")

        positive = bui_lowres > 0
        bui_pow = np.full_like(bui_lowres, np.nan, dtype="float32")
        bui_pow[positive] = bui_lowres[positive] ** power
        bui_pow[bui_lowres == 0] = 0.0  # preserve zeros

        bui_lowres = bui_pow

    # ---------------------------------------------------------
    # Log10 transform (only if power is not used)
    # ---------------------------------------------------------
    elif logscale:
        positive = bui_lowres > 0
        bui_log = np.full_like(bui_lowres, np.nan, dtype="float32")
        bui_log[positive] = np.log10(bui_lowres[positive])
        # zeros and NaNs remain NaN
        bui_lowres = bui_log

    # ---------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.imshow(bui_lowres, cmap="viridis", interpolation="nearest")

    if power is not None:
        plt.colorbar(label=f"BUI^(power={power})")
        title_suffix = f" (power={power})"
    elif logscale:
        plt.colorbar(label="log10(BUI)")
        title_suffix = " (log10)"
    else:
        plt.colorbar(label="BUI (coarse mean)")
        title_suffix = ""

    plt.title(f"BUI downsampled by factor {factor}{title_suffix}")
    plt.axis("off")

    # Save PNG
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()

    return bui_lowres


# ---------------------------------------------------------
# Main block
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Downsample a BUI raster and save a PNG visualization."
    )

    parser.add_argument("--bui", type=str, help="Path to the BUI GeoTIFF (250 m Albers).")
    parser.add_argument("--factor", type=int, help="Downsampling factor (e.g., 16 → ~4 km).")
    parser.add_argument("--png", type=str, help="Output PNG path.")
    parser.add_argument("--log", action="store_true", help="Visualize in log10 scale.")
    parser.add_argument("--power", type=float,
                        help="Apply a power-law transform x^a (0 < a <= 1). Overrides --log.")

    args = parser.parse_args()

    # Fallback logic
    bui_path = args.bui if args.bui else DEFAULT_BUI_PATH
    factor = args.factor if args.factor else DEFAULT_FACTOR
    png_path = args.png if args.png else DEFAULT_PNG_PATH
    logscale = args.log if args.log else DEFAULT_LOGSCALE
    power = args.power if args.power is not None else DEFAULT_POWER

    print(f"Using BUI path: {bui_path}")
    print(f"Downsampling factor: {factor}")
    print(f"Saving PNG to: {png_path}")
    print(f"Log scale: {logscale}")
    print(f"Power transform: {power}")

    visualize_bui_lowres(
        bui_path=bui_path,
        factor=factor,
        png_path=png_path,
        logscale=logscale,
        power=power
    )