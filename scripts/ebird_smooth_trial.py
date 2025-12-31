#!/usr/bin/env python3
"""
Strict-mask version with full per-anchor plots restored.
A pixel is valid iff:
    1) All 12 PRISM months are finite
    2) BUI Band 6 is finite
"""

import os
import numpy as np
import rioxarray as rxr
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from rasterio.windows import Window

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

EBIRD_DIR = "/home/breallis/datasets/ebird_weekly_2023_albers"
SPECIES = ["amegfi", "whcspa", "houspa"]
SIGMAS = [0.0, 1.0, 2.5, 4.0]
POWER = 0.5

PRISM_MONTHLY = "/home/breallis/datasets/prism_monthly_4km_albers"
BUI_FILE = "/home/breallis/datasets/HBUI/BUI/2020_BUI_4km_interp.tif"

OUTDIR = "/home/breallis/datasets/ebird_trial_outputs"
os.makedirs(OUTDIR, exist_ok=True)

SITES = {
    "site_center": {
        "window": Window(col_off=200, row_off=200, width=150, height=150),
        "anchors": [(75, 75), (50, 50), (100, 100)],
    },
    "site_north": {
        "window": Window(col_off=200, row_off=50, width=150, height=150),
        "anchors": [(75, 75)],
    },
    "site_south": {
        "window": Window(col_off=200, row_off=350, width=150, height=150),
        "anchors": [(75, 75)],
    },
}

# ---------------------------------------------------------
# UTILITY FUNCTIONS
# ---------------------------------------------------------

def load_prism_monthly(window):
    files = sorted([
        f for f in os.listdir(PRISM_MONTHLY)
        if f.startswith("prism_ppt_us_25m_2023") and f.endswith("_bui4km.tif")
    ])
    assert len(files) == 12

    stack = []
    for f in files:
        da = rxr.open_rasterio(os.path.join(PRISM_MONTHLY, f))
        da_win = da.rio.isel_window(window)
        stack.append(da_win.values[0])
    return np.stack(stack, axis=0)  # (12, H, W)


def load_bui_band6_raw(window):
    da = rxr.open_rasterio(BUI_FILE)
    da_win = da.rio.isel_window(window)
    return da_win.values[5].astype("float32")  # (H, W)


def load_weekly_stack(species, window, valid_mask):
    files = sorted([
        f for f in os.listdir(EBIRD_DIR)
        if species in f and f.endswith(".tif")
    ])
    assert len(files) == 52

    stack = []
    for f in files:
        da = rxr.open_rasterio(os.path.join(EBIRD_DIR, f))
        da = da.rio.write_nodata(np.nan, inplace=False)
        da_win = da.rio.isel_window(window)
        arr = da_win.values[0].astype("float32")

        out = np.zeros_like(arr, dtype="float32")
        out[valid_mask] = np.nan_to_num(arr[valid_mask], nan=0.0)
        stack.append(out)

    return np.stack(stack, axis=0)  # (52, H, W)


def circular_gaussian_smooth(ts, sigma):
    if sigma == 0:
        return ts.copy()
    return gaussian_filter1d(ts, sigma=sigma, axis=0, mode="wrap")


def power_transform(arr, exponent):
    return np.power(np.clip(arr, 0, None), exponent)


def flatten_features(species_smoothed):
    feats = []
    for arr in species_smoothed.values():
        feats.append(arr.reshape(52, -1))
    return np.concatenate(feats, axis=0).T  # (pixels, features)


def compute_kernel(X):
    return X @ X.T


def build_similarity_from_1d(x_flat, valid_idx):
    x = x_flat[valid_idx].astype("float32")
    diff = np.abs(x[:, None] - x[None, :])
    max_diff = diff.max()
    return (1.0 - diff / max_diff).astype("float32") if max_diff > 0 else np.ones_like(diff)


def mantel_style_correlations(K, S_env, valid_idx):
    K_sub = K[np.ix_(valid_idx, valid_idx)]
    triu = np.triu_indices_from(K_sub, k=1)
    sim_corr = np.corrcoef(K_sub[triu], S_env[triu])[0, 1]

    K_min, K_max = K_sub.min(), K_sub.max()
    D_K = 1.0 - (K_sub - K_min) / (K_max - K_min) if K_max > K_min else np.ones_like(K_sub)
    D_env = 1.0 - S_env
    dist_corr = np.corrcoef(D_K[triu], D_env[triu])[0, 1]
    return sim_corr, dist_corr


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    for site_name, site_cfg in SITES.items():
        window = site_cfg["window"]
        anchors = site_cfg["anchors"]
        site_outdir = os.path.join(OUTDIR, site_name)
        os.makedirs(site_outdir, exist_ok=True)
        print(f"\n=== Site: {site_name} ===")

        # Load raw PRISM + BUI
        prism = load_prism_monthly(window)       # (12, H, W)
        bui_raw = load_bui_band6_raw(window)     # (H, W)
        H, W = bui_raw.shape

        # Strict valid mask
        valid_mask = np.all(np.isfinite(prism), axis=0) & np.isfinite(bui_raw)
        valid_idx = np.where(valid_mask.reshape(-1))[0]

        # PRISM annual mean
        ppt_mean = np.full((H, W), np.nan, dtype="float32")
        ppt_mean[valid_mask] = prism[:, valid_mask].mean(axis=0)
        ppt_mean_flat = ppt_mean.reshape(-1)

        # Plot annual mean ppt
        plt.figure()
        plt.imshow(ppt_mean, cmap="Blues")
        plt.colorbar()
        plt.title(f"PRISM annual mean ppt (2023) - {site_name}")
        plt.savefig(os.path.join(site_outdir, "prism_ppt_mean_map.png"), dpi=150)
        plt.close()

        climate_sim = build_similarity_from_1d(ppt_mean_flat, valid_idx)

        # BUI transformed
        bui = np.full((H, W), np.nan, dtype="float32")
        # bui[valid_mask] = np.power(np.clip(bui_raw[valid_mask], 0, None), 0.01)
        bui[valid_mask] = np.log(np.clip(bui_raw[valid_mask], 0, None) + 1)
        bui_flat = bui.reshape(-1)

        plt.figure()
        plt.imshow(bui, cmap="inferno")
        plt.colorbar()
        plt.title(f"BUI Band 6^0.1 (2020) - {site_name}")
        plt.savefig(os.path.join(site_outdir, "bui_band6_p01_map.png"), dpi=150)
        plt.close()

        bui_sim = build_similarity_from_1d(bui_flat, valid_idx)

        # Load eBird
        weekly = {sp: load_weekly_stack(sp, window, valid_mask) for sp in SPECIES}

        # Loop over smoothing scales
        for sigma in SIGMAS:
            print(f"  σ = {sigma}")
            smoothed = {sp: circular_gaussian_smooth(weekly[sp], sigma) for sp in SPECIES}
            smoothed_pt = {sp: power_transform(smoothed[sp], POWER) for sp in SPECIES}
            X = flatten_features(smoothed_pt)
            K = compute_kernel(X)

            sim_corr_clim, dist_corr_clim = mantel_style_correlations(K, climate_sim, valid_idx)
            sim_corr_bui, dist_corr_bui = mantel_style_correlations(K, bui_sim, valid_idx)

            print(f"    Climate similarity corr: {sim_corr_clim:.3f}")
            print(f"    Climate distance   corr: {dist_corr_clim:.3f}")
            print(f"    BUI similarity     corr: {sim_corr_bui:.3f}")
            print(f"    BUI distance       corr: {dist_corr_bui:.3f}")

            # -----------------------------------------
            # Per-anchor plots (restored)
            # -----------------------------------------
            for ar, ac in anchors:
                if not (0 <= ar < H and 0 <= ac < W): 
                    continue
                anchor_idx = ar * W + ac
                anchor_tag = f"r{ar}_c{ac}"

                # PRISM phenology
                ppt_anchor = prism[:, ar, ac]
                months = np.arange(1, 13)
                plt.figure()
                plt.plot(months, ppt_anchor, marker="o")
                plt.xticks(months)
                plt.xlabel("Month (2023)")
                plt.ylabel("PRISM ppt")
                plt.title(f"PRISM ppt phenology at ({ar}, {ac}) - {site_name}")
                plt.savefig(os.path.join(site_outdir, f"prism_ppt_phenology_{anchor_tag}.png"), dpi=150)
                plt.close()

                # PRISM similarity
                if valid_mask[ar, ac]:
                    clim_sim_anchor = np.full(H*W, np.nan, dtype="float32")
                    anchor_valid_pos = np.where(valid_idx == anchor_idx)[0]
                    if anchor_valid_pos.size > 0:
                        clim_sim_anchor[valid_idx] = climate_sim[anchor_valid_pos[0]]
                        plt.figure()
                        plt.imshow(clim_sim_anchor.reshape(H, W), cmap="Blues")
                        plt.colorbar()
                        plt.title(f"Climate similarity (ppt_mean) to ({ar},{ac}) - {site_name}")
                        plt.savefig(os.path.join(site_outdir, f"prism_similarity_anchor_{anchor_tag}.png"), dpi=150)
                        plt.close()

                # BUI similarity
                if valid_mask[ar, ac]:
                    bui_sim_anchor = np.full(H*W, np.nan, dtype="float32")
                    anchor_valid_pos = np.where(valid_idx == anchor_idx)[0]
                    if anchor_valid_pos.size > 0:
                        bui_sim_anchor[valid_idx] = bui_sim[anchor_valid_pos[0]]
                        plt.figure()
                        plt.imshow(bui_sim_anchor.reshape(H, W), cmap="viridis")
                        plt.colorbar()
                        plt.title(f"BUI^0.1 similarity to ({ar},{ac}) - {site_name}")
                        plt.savefig(os.path.join(site_outdir, f"bui_similarity_anchor_{anchor_tag}.png"), dpi=150)
                        plt.close()

                # Species phenology curves
                for sp in SPECIES:
                    raw = weekly[sp][:, ar, ac]
                    sm = smoothed[sp][:, ar, ac]
                    plt.figure()
                    plt.plot(raw, label="raw")
                    plt.plot(sm, label=f"smoothed σ={sigma}")
                    plt.xlabel("Week")
                    plt.ylabel("eBird abundance")
                    plt.title(f"{sp} phenology at ({ar},{ac}) - {site_name}")
                    plt.legend()
                    plt.savefig(os.path.join(site_outdir, f"phenology_{sp}_sigma{sigma}_{anchor_tag}.png"), dpi=150)
                    plt.close()

                # Kernel similarity
                sim_map = K[anchor_idx].reshape(H, W)
                plt.figure()
                plt.imshow(sim_map, cmap="magma")
                plt.colorbar()
                plt.title(f"Kernel similarity to ({ar},{ac}) (σ={sigma}) - {site_name}")
                plt.savefig(os.path.join(site_outdir, f"kernel_anchor_sigma{sigma}_{anchor_tag}.png"), dpi=150)
                plt.close()


if __name__ == "__main__":
    main()
