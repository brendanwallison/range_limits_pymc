import os
import pickle
import sys
import jax
import jax.numpy as jnp
import numpyro
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta
from numpyro.optim import Minimize

# --- Configuration ---
PRECISION = 'float32'
jax.config.update("jax_enable_x64", True if PRECISION == 'float64' else False)

RESULT_DIR = f"/home/breallis/processed_data/model_results/age_map_{PRECISION}_run_11"
INPUT_DIR = "/home/breallis/processed_data/model_inputs/numpyro_input"

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.model.age_priors import build_model_2d
from src.model.run_map import load_data_to_gpu

def run_polish():
    print(f"--- Loading Results for L-BFGS Polishing ---")
    
    # 1. Load Data and MAP Params
    data_dict = load_data_to_gpu(INPUT_DIR, precision=PRECISION)
    with open(os.path.join(RESULT_DIR, "map_params.pkl"), 'rb') as f:
        initial_params = pickle.load(f)

    # 2. Setup Model & Guide
    model = build_model_2d
    guide = AutoDelta(model)

    # 3. Setup the Minimize Optimizer (L-BFGS)
    # This wrapper is the standard way to do MAP polishing in NumPyro
    optimizer = Minimize(method='bfgs', options={'maxiter': 500, 'gtol': 1e-7})

    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    print("Initializing SVI state from previous Adam results...")
    # Warm start: Initialize, then replace the state params with your Adam results
    svi_state = svi.init(jax.random.PRNGKey(0), data=data_dict, anneal=1.0)
    
    # In NumPyro's Minimize/SVI, the 'optim_state' holds the current parameter values
    svi_state = svi_state._replace(optim_state=initial_params)

    print("Starting L-BFGS (this may take a few minutes as it calculates curvature)...")
    # For Minimize, svi.run only needs to be called with num_steps=1 
    # because the optimizer handles its own internal iterations.
    svi_result = svi.run(
        jax.random.PRNGKey(1), 
        num_steps=1, 
        data=data_dict, 
        anneal=1.0, 
        progress_bar=True
    )

    # 4. Save the polished results
    polished_params = svi_result.params
    output_path = os.path.join(RESULT_DIR, "map_params_polished.pkl")
    
    with open(output_path, 'wb') as f:
        pickle.dump(polished_params, f)
    
    print(f"Polishing complete. Final Loss: {svi_result.losses[-1]:.4f}")
    print(f"Saved polished parameters to: {output_path}")

if __name__ == "__main__":
    run_polish()