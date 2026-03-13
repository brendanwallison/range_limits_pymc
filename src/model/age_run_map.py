import sys
import os
import pickle
import numpy as np
import jax
import jax.numpy as jnp
import numpyro
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta
import matplotlib.pyplot as plt
import optax
from tqdm import tqdm

# --- SINGLE POINT OF CONTROL ---
PRECISION = 'float32' # Options: 'float32' or 'float64'
# -------------------------------

jax.config.update("jax_enable_x64", True if PRECISION == 'float64' else False)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.model.age_priors import build_model_2d

# --- CONFIGURATION ---
INPUT_DIR = "/home/breallis/processed_data/model_inputs/numpyro_input"
OUTPUT_DIR = f"/home/breallis/processed_data/model_results/age_map_{PRECISION}_run_15"

def load_data_to_gpu(input_dir, precision='float32'):
    meta_path = os.path.join(input_dir, "metadata.pkl")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    
    f_type_cpu = np.float32 if precision == 'float32' else np.float64
    f_type_gpu = jnp.float32 if precision == 'float32' else jnp.float64
    i_type_gpu = jnp.int32 if precision == 'float32' else jnp.int64

    streaming_keys = {'Z_gathered', 'Z_disp_gathered', 'st_basis'}

    z_shape = (meta['time'], meta['N_land'], meta['M'])
    z_mem = np.memmap(os.path.join(input_dir, meta['z_gathered_path']), 
                      dtype='float32', mode='r', shape=z_shape)
    
    z_disp_shape = (meta['time'], meta['N_land'], meta['K'], meta['M'])
    z_disp_mem = np.memmap(os.path.join(input_dir, meta['z_disp_gathered_path']), 
                           dtype='float32', mode='r', shape=z_disp_shape)
    
    meta['Z_gathered'] = np.array(z_mem).astype(f_type_cpu)
    meta['Z_disp_gathered'] = np.array(z_disp_mem).astype(f_type_cpu)

    print(f"Iterating through metadata and casting to {precision}...")
    for key, value in meta.items():
        if isinstance(value, np.ndarray):
            if key in streaming_keys:
                meta[key] = value.astype(f_type_cpu)
                print(f"  [CPU] {key}: {meta[key].nbytes / 1e9:.2f} GB")
            else:
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
    print(f"--- Starting {PRECISION.upper()} Optimization (Age-Structured) ---")
    data_dict = load_data_to_gpu(INPUT_DIR, precision=PRECISION)

    guide = AutoDelta(build_model_2d)

    total_steps = 1000
    anneal_epochs = [0.1, 0.5, 1.0]
    steps_per_epoch = total_steps // len(anneal_epochs)

    scheduler = optax.cosine_decay_schedule(init_value=0.01, decay_steps=total_steps, alpha=0.1)

    optimizer = numpyro.optim.optax_to_numpyro(
        optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=scheduler, weight_decay=1e-4, eps=1e-7)
        )
    )
    svi = SVI(build_model_2d, guide, optimizer, loss=Trace_ELBO())
    
    print("Compiling model...")
    rng_key = jax.random.PRNGKey(41)
    
    # Initialize state
    svi_state = svi.init(rng_key, data=data_dict, anneal=anneal_epochs[0])
    all_losses = []

    # --- THE EPOCH LOOP ---
    # Using a Python loop here avoids the XLA memory explosion 
    # that happens when scanning over thousands of high-res steps at once.
    # --- THE EPOCH LOOP ---
    for anneal_level in anneal_epochs:
        desc = f"Epoch (Anneal={anneal_level})"
        epoch_losses = []
        
        # Wrap the range in tqdm and assign to pbar
        pbar = tqdm(range(steps_per_epoch), desc=desc)
        for i in pbar:
            svi_state, loss = svi.update(svi_state, data=data_dict, anneal=anneal_level)
            
            # Convert loss to float for the progress bar
            loss_val = float(loss)
            epoch_losses.append(loss_val)
            
            # Update the progress bar every 10 steps to keep the display snappy
            if i % 10 == 0:
                pbar.set_postfix({"loss": f"{loss_val:.4f}"})

        all_losses.append(np.array(epoch_losses))

    losses = np.concatenate(all_losses)
    params = svi.get_params(svi_state)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "map_params.pkl"), 'wb') as f:
        pickle.dump(params, f)
        
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.yscale('log')
    plt.title(f"MAP Optimization Loss - Epoch-Based Annealing ({PRECISION})")
    plt.xlabel("Iteration")
    plt.ylabel("Loss (Log Scale)")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve_log.png"))
    plt.close()
    
    print(f"Final Loss: {losses[-1]:.4f}")

if __name__ == "__main__":
    run_map()