import os
import jax
import jax.numpy as jnp
import numpy as np
import pickle

import numpyro
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta
from numpyro.optim import Adam

from src.ingest_directional import build_data_jax
from src.model import build_model_2d

# -----------------------------
# Checkpoint helpers
# -----------------------------
def save_checkpoint(path, svi_state, step, rng_key, config):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(
            {
                "svi_state": svi_state,
                "step": step,
                "rng_key": rng_key,
                "config": config,
            },
            f,
        )

def load_checkpoint(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# -----------------------------
# MAP runner with rolling checkpoints
# -----------------------------
def run_map(
    data_path="data/stan_data_for_python.npz",
    num_steps=1000,
    lr=1e-2,
    seed=135,
    checkpoint_dir="checkpoints",
    checkpoint_every=100,
    resume=False,
):
    # -----------------------------
    # Load data
    # -----------------------------
    data = build_data_jax(data_path)

    # -----------------------------
    # Model + MAP guide
    # -----------------------------
    model = build_model_2d
    guide = AutoDelta(model)
    optimizer = Adam(lr)
    svi = SVI(model=model, guide=guide, optim=optimizer, loss=Trace_ELBO())

    # -----------------------------
    # Initialize or resume
    # -----------------------------
    rng_key = jax.random.PRNGKey(seed)
    start_step = 0

    if resume:
        # Find latest checkpoint in directory
        if not os.path.exists(checkpoint_dir):
            raise ValueError(f"Checkpoint directory {checkpoint_dir} does not exist.")
        ckpt_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pkl")]
        if not ckpt_files:
            raise ValueError("No checkpoints found to resume from.")
        latest_ckpt = max(ckpt_files, key=lambda x: int(x.split("step")[-1].split(".pkl")[0]))
        ckpt = load_checkpoint(os.path.join(checkpoint_dir, latest_ckpt))
        svi_state = ckpt["svi_state"]
        rng_key = ckpt["rng_key"]
        start_step = ckpt["step"] + 1
        print(f"Resuming from step {start_step}")
    else:
        svi_state = svi.init(rng_key, data)

    # -----------------------------
    # Optimization loop
    # -----------------------------
    for step in range(start_step, num_steps):
        svi_state, loss = svi.update(svi_state, data)

        # NaN/Inf check
        if not jnp.isfinite(loss):
            print(f"Non-finite loss at step {step}, stopping.")
            break

        if step % 25 == 0:
            print(f"step {step:5d} | loss = {loss:.6f}")

        # Save rolling checkpoint
        if checkpoint_every > 0 and step % checkpoint_every == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"map_ckpt_step{step}.pkl")
            save_checkpoint(
                ckpt_path,
                svi_state=svi_state,
                step=step,
                rng_key=rng_key,
                config={
                    "lr": lr,
                    "seed": seed,
                    "data_path": data_path,
                    "num_steps": num_steps,
                },
            )

    # -----------------------------
    # Extract MAP estimate
    # -----------------------------
    params = svi.get_params(svi_state)
    map_estimate = guide.median(params)

    return map_estimate, svi_state

# -----------------------------
# Script entry point
# -----------------------------
if __name__ == "__main__":
    map_estimate, svi_state = run_map(
        num_steps=30000,
        lr=1e-3,
        checkpoint_dir="checkpoints",
        checkpoint_every=1000,
        resume=False
    )

    # Save MAP parameters only
    np.savez(
        "map_estimate_4.npz",
        **{k: np.asarray(v) for k, v in map_estimate.items()},
    )

    REPORT_KEYS = [
        "r_mean",
        "b_raw",
        "allee_intercept",
        "allee_slope",
        "dispersal_intercept",
        "dispersal_logit_intercept",
        "dispersal_logit_slope",
        "dispersal_survival_threshold",
        "dispersal_survival_slope",
    ]

    print("\nMAP summary (selected parameters):")
    for k in REPORT_KEYS:
        if k in map_estimate:
            print(f"{k:30s}: {float(map_estimate[k]): .4f}")
