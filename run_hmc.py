import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro.infer import MCMC, NUTS, init_to_value
from src.ingest import build_data_jax
from src.model import build_model_2d

# ---------------------------
# Load data
# ---------------------------
data_path = "data/stan_data_for_python.npz"
data = build_data_jax(data_path)

# ---------------------------
# Load MAP estimate
# ---------------------------
map_file = "map_estimate.npz"
map_data = np.load(map_file)
map_estimate = {k: jnp.array(v) for k, v in map_data.items()}

# ---------------------------
# Random key
# ---------------------------
rng_key = jax.random.PRNGKey(0)

# ---------------------------
# NUTS kernel
# ---------------------------
nuts_kernel = NUTS(
    build_model_2d,
    init_strategy=init_to_value(values=map_estimate),
    max_tree_depth=10,
)

# ---------------------------
# MCMC
# ---------------------------
mcmc = MCMC(
    nuts_kernel,
    num_warmup=250,
    num_samples=750,
    num_chains=1
)

# ---------------------------
# Run MCMC: pass data explicitly
# ---------------------------
mcmc.run(rng_key, data=data)

# ---------------------------
# Extract and save posterior draws
# ---------------------------
posterior_samples = mcmc.get_samples()
np.savez("posterior_draws.npz", **{k: np.asarray(v) for k, v in posterior_samples.items()})

# ---------------------------
# Summary
# ---------------------------
mcmc.print_summary()
