import requests
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import xarray as xr

# ------------------------------------------
# Config
# ------------------------------------------
BASE_URL = "https://services.nacse.org/prism/data/get/us/4km"
OUTPUT_DIR = Path("prism_monthly_4km")
VARIABLES = ["ppt", "tmin", "tmax", "tmean", "tdmean", "vpdmin", "vpdmax"]
START_YEAR = 1895
END_YEAR = 2024
FORMAT = "nc"

MAX_RETRIES = 5
BACKOFF = 5  # seconds
MAX_WORKERS = 3  # safer for NACSE

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------
# Helper Functions
# ------------------------------------------

def generate_month_list(start_year, end_year):
    out = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            out.append(f"{year:04d}{month:02d}")
    return out

def expected_filename(variable, yyyymm):
    return OUTPUT_DIR / f"prism_{variable}_{yyyymm}.zip"

def construct_url(variable, yyyymm):
    url = f"{BASE_URL}/{variable}/{yyyymm}"
    if FORMAT:
        url += f"?format={FORMAT}"
    return url

def stream_download(url, dest):
    """Robust streaming download with retries and content-type validation."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with requests.get(url, stream=True, timeout=300) as r:
                r.raise_for_status()

                ctype = r.headers.get("Content-Type", "").lower()
                if "zip" not in ctype:
                    raise ValueError(f"Unexpected content type: {ctype}")

                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

            # Validate file size > 10 KB
            if dest.stat().st_size < 10_000:
                raise ValueError("Downloaded file too small; likely error page.")

            # Validate NetCDF
            try:
                xr.open_dataset(dest).close()
            except Exception:
                raise ValueError("Downloaded file is not a valid NetCDF.")

            return True

        except Exception as e:
            print(f"[WARN] {url}: attempt {attempt} failed: {e}")
            time.sleep(BACKOFF * attempt)

    return False

def download_task(variable, yyyymm):
    dest = expected_filename(variable, yyyymm)
    if dest.exists():
        return (variable, yyyymm, "exists")

    url = construct_url(variable, yyyymm)
    ok = stream_download(url, dest)
    return (variable, yyyymm, "ok" if ok else "fail")

# ------------------------------------------
# Verification Logic
# ------------------------------------------

def verify_file(variable, yyyymm):
    """Return True if file exists and is valid NetCDF."""
    f = expected_filename(variable, yyyymm)
    if not f.exists():
        return False

    # Size check
    if f.stat().st_size < 10_000:
        return False

    # Try opening
    try:
        xr.open_dataset(f).close()
    except Exception:
        return False

    return True

def scan_all():
    """Return lists of missing and corrupt files."""
    months = generate_month_list(START_YEAR, END_YEAR)
    missing = []
    corrupt = []

    for var in VARIABLES:
        for yyyymm in months:
            f = expected_filename(var, yyyymm)
            if not f.exists():
                missing.append(f)
            else:
                if not verify_file(var, yyyymm):
                    corrupt.append(f)

    return missing, corrupt

# ------------------------------------------
# Main
# ------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-only", action="store_true",
                        help="List missing files without downloading.")
    parser.add_argument("--verify", action="store_true",
                        help="Verify all files and list corrupt ones.")
    parser.add_argument("--resume", action="store_true",
                        help="Re-download missing or corrupt files.")
    args = parser.parse_args()

    # -------------------------
    # SCAN ONLY
    # -------------------------
    if args.scan_only:
        missing, corrupt = scan_all()
        print("\n=== Missing Files ===")
        for f in missing:
            print(f)
        print(f"Total missing: {len(missing)}")

        print("\n=== Corrupt Files ===")
        for f in corrupt:
            print(f)
        print(f"Total corrupt: {len(corrupt)}")
        exit(0)

    # -------------------------
    # VERIFY
    # -------------------------
    if args.verify:
        missing, corrupt = scan_all()
        print("\n=== Missing Files ===")
        for f in missing:
            print(f)
        print(f"Total missing: {len(missing)}")

        print("\n=== Corrupt Files ===")
        for f in corrupt:
            print(f)
        print(f"Total corrupt: {len(corrupt)}")
        exit(0)

    # -------------------------
    # RESUME
    # -------------------------
    if args.resume:
        missing, corrupt = scan_all()
        todo = missing + corrupt

        print(f"Re-downloading {len(todo)} files...")

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = []
            for f in todo:
                # parse variable + yyyymm from filename
                parts = f.stem.split("_")
                var = parts[1]
                yyyymm = parts[2]
                futures.append(ex.submit(download_task, var, yyyymm))

            for fut in tqdm(as_completed(futures), total=len(futures)):
                var, yyyymm, status = fut.result()
                if status != "ok":
                    print(f"[ERROR] {var} {yyyymm} failed.")

        print("Re-download complete. Re-verifying...")

        missing2, corrupt2 = scan_all()
        print("\n=== Remaining Missing ===")
        for f in missing2:
            print(f)
        print(f"Total missing: {len(missing2)}")

        print("\n=== Remaining Corrupt ===")
        for f in corrupt2:
            print(f)
        print(f"Total corrupt: {len(corrupt2)}")

        exit(0)

    # -------------------------
    # DEFAULT: FULL DOWNLOAD
    # -------------------------
    months = generate_month_list(START_YEAR, END_YEAR)
    total_tasks = len(VARIABLES) * len(months)

    print(f"Downloading {total_tasks} monthly grids...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for var in VARIABLES:
            for yyyymm in months:
                futures.append(executor.submit(download_task, var, yyyymm))

        for f in tqdm(as_completed(futures), total=total_tasks):
            var, yyyymm, status = f.result()
            if status not in ("ok", "exists"):
                print(f"[ERROR] {var} {yyyymm} failed.")

    print("Done!")