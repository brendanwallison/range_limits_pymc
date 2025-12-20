import jax.numpy as jnp
import jax.nn as jnn
from numpyro.contrib.hsgp.spectral_densities import (
    diag_spectral_density_squared_exponential,
)

def hsgp_spd_se(
    alpha_raw,
    length_raw,
    ell,
    m,
    dim,
):
    """
    Square root spectral density for SE kernel.
    """
    alpha = jnn.softplus(alpha_raw)
    length = jnn.softplus(length_raw)

    return jnp.sqrt(
        diag_spectral_density_squared_exponential(
            alpha=alpha,
            length=length,
            ell=ell,
            m=m,
            dim=dim,
        )
    )


def hsgp_field(phi, spd, beta):
    """
    Deterministic HSGP field reconstruction.
    """
    return phi @ (spd * beta)
