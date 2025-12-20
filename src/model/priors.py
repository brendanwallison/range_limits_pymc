import numpyro
import numpyro.distributions as dist

def sample_hsgp_beta(name, m):
    with numpyro.plate(f"{name}_basis", m):
        return numpyro.sample(
            f"{name}/beta",
            dist.Normal(0.0, 1.0),
        )


def sample_positive(name, scale=1.0):
    return numpyro.sample(
        name,
        dist.LogNormal(0.0, scale),
    )
