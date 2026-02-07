import sys
import os
import pickle
import numpy as np
import jax
import jax.numpy as jnp
import numpyro
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta
from numpyro.optim import Adam
import matplotlib.pyplot as plt

# --- SINGLE POINT OF CONTROL ---
PRECISION = 'float32' # Options: 'float32' or 'float64'
# -------------------------------

# This must happen immediately after import and before build_model_2d is touched
jax.config.update("jax_enable_x64", True if PRECISION == 'float64' else False)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.model.priors import build_model_2d

# --- CONFIGURATION ---
INPUT_DIR = "/home/breallis/processed_data/model_inputs/numpyro_input"
OUTPUT_DIR = f"/home/breallis/processed_data/model_results/map_{PRECISION}_run_01"

def load_data_to_gpu(input_dir, precision='float32'):
    meta_path = os.path.join(input_dir, "metadata.pkl")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    
    # Use NumPy types for CPU-bound data, JAX types for GPU-bound
    f_type_cpu = np.float32 if precision == 'float32' else np.float64
    f_type_gpu = jnp.float32 if precision == 'float32' else jnp.float64
    i_type_gpu = jnp.int32 if precision == 'float32' else jnp.int64

    # The "Heavy Three" that we want to keep on CPU/System RAM
    streaming_keys = {'Z_gathered', 'Z_disp_gathered', 'st_basis'}

    # 1. Handle Memmaps first
    z_shape = (meta['time'], meta['N_land'], meta['M'])
    z_mem = np.memmap(os.path.join(input_dir, meta['z_gathered_path']), 
                      dtype='float32', mode='r', shape=z_shape)
    
    z_disp_shape = (meta['time'], meta['N_land'], meta['K'], meta['M'])
    z_disp_mem = np.memmap(os.path.join(input_dir, meta['z_disp_gathered_path']), 
                           dtype='float32', mode='r', shape=z_disp_shape)
    
    # Store them as NumPy arrays (CPU)
    meta['Z_gathered'] = np.array(z_mem).astype(f_type_cpu)
    meta['Z_disp_gathered'] = np.array(z_disp_mem).astype(f_type_cpu)

    print(f"Iterating through metadata and casting to {precision}...")
    for key, value in meta.items():
        if isinstance(value, np.ndarray):
            # Is it one of our giant streaming arrays?
            if key in streaming_keys:
                # Keep on CPU, just ensure correct precision
                meta[key] = value.astype(f_type_cpu)
                print(f"  [CPU] {key}: {meta[key].nbytes / 1e9:.2f} GB")
            else:
                # Everything else goes to GPU
                if np.issubdtype(value.dtype, np.floating):
                    meta[key] = jnp.array(value).astype(f_type_gpu)
                elif np.issubdtype(value.dtype, np.integer):
                    meta[key] = jnp.array(value).astype(i_type_gpu)
                else:
                    meta[key] = jnp.array(value)
                print(f"  [GPU] {key}: {meta[key].nbytes / 1e6:.1f} MB")

    if precision == 'float32' and meta.get('pseudo_zero', 0) < 1e-7:
        meta['pseudo_zero'] = 1e-7

    vram_gb = sum(x.nbytes for x in meta.values() if isinstance(x, jnp.ndarray)) / 1e9
    print(f"--- Total Resident VRAM: {vram_gb:.2f} GB ---")
    
    return meta

def run_map():
    print(f"--- Starting {PRECISION.upper()} Optimization ---")
    print(f"Platform: {jax.devices()[0].platform}")
    
    # Load Data with current precision
    data_dict = load_data_to_gpu(INPUT_DIR, precision=PRECISION)

    guide = AutoDelta(build_model_2d)
    optimizer = Adam(step_size=0.001) 
    svi = SVI(build_model_2d, guide, optimizer, loss=Trace_ELBO())
    
    print("Compiling model...")
    rng_key = jax.random.PRNGKey(41)
    num_steps = 20000
    
    # Ensure anneal is passed correctly
    svi_result = svi.run(rng_key, num_steps, data=data_dict, anneal=1.0, progress_bar=True)
    
    # Save results to a precision-specific folder
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    params = svi_result.params
    losses = svi_result.losses
    
    with open(os.path.join(OUTPUT_DIR, "map_params.pkl"), 'wb') as f:
        pickle.dump(params, f)
        
    # --- LOSS PLOTTING ---
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    
    # Apply log scale to see the actual convergence progress
    plt.yscale('log')
    
    plt.title(f"MAP Optimization Loss ({PRECISION})")
    plt.xlabel("Iteration")
    plt.ylabel("Loss (Log Scale)")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve_log.png"))
    plt.close()
    
    # Also print the last 10 steps to check for "jitter"
    print(f"Final Loss: {losses[-1]:.4f}")
    print(f"Recent Loss Delta: {losses[-10] - losses[-1]:.4f}")

if __name__ == "__main__":
    run_map()