import jax.numpy as jnp
from jax import lax
import jax.nn as jnn
import numpyro
import numpyro.distributions as dist
from numpyro.contrib.hsgp.approximation import linear_approximation
from numpyro.contrib.hsgp.spectral_densities import diag_spectral_density_squared_exponential
from numpyro.handlers import scope
import jax

# ---------------------------
# Helper functions
# ---------------------------

def allee_factor(pop, allee_scalar, allee_intercept):
    """Allee effect using a sigmoid."""
    return jnn.sigmoid(pop * allee_scalar + allee_intercept)

def debug_one_nan(tag, **arrays):
    """
    arrays: named arrays, all same shape
    Prints values at the first NaN location (if any).
    """

    # pick a reference array to locate NaNs
    ref = next(iter(arrays.values()))
    nan_mask = jnp.isnan(ref)

    # index of first NaN (flattened)
    idx = jnp.argmax(nan_mask)

    # convert flat index to 2D
    ny, nx = ref.shape
    iy = idx // nx
    ix = idx % nx

    has_nan = jnp.any(nan_mask)

    jax.debug.print(
        "\n[{tag}] NaN found={has_nan} at (y={iy}, x={ix})",
        tag=tag,
        has_nan=has_nan,
        iy=iy,
        ix=ix,
    )

    for name, arr in arrays.items():
        jax.debug.print(
            "[{tag}] {name}[y,x] = {val}",
            tag=tag,
            name=name,
            val=arr[iy, ix],
        )

# def reproduction(N0, r, K, allee_scalar, allee_intercept):
#     """Density-dependent growth with Allee effect."""
#     g = jnp.exp(r)
#     denom = g * N0 + K - N0
#     pop_new = g * N0 * K / denom
#     return allee_factor(N0, allee_scalar, allee_intercept) * pop_new

def reproduction_safe(N0, r, K, allee_scalar, allee_intercept, eps=1e-12):
    g = jnp.exp(r)
    denom = K + N0 * jnp.expm1(r) + eps  # multiply through by K and add tiny eps
    pop_new = g * N0 * K / denom
    return allee_factor(N0, allee_scalar, allee_intercept) * pop_new

def rightpad(A, Lx, Ly, pad_value=1e-9):
    """Pad a 2D matrix to size (Ly, Lx)."""
    pad_matrix = jnp.ones((Ly, Lx)) * pad_value
    pad_matrix = pad_matrix.at[:A.shape[0], :A.shape[1]].set(A)
    return pad_matrix

def rightpad_convolution(pop, dispersal_kernel_pad):
    """FFT-based convolution, extract real part."""
    Ly, Lx = dispersal_kernel_pad.shape
    pop_pad = rightpad(pop, Lx, Ly, 1e-9)
    conv = jnp.fft.ifft2(jnp.fft.fft2(pop_pad) * dispersal_kernel_pad)
    return jnp.real(conv)[:pop.shape[0], :pop.shape[1]]

def juvenile_dispersal_vectorized(
    juvenile_dispersers: jnp.ndarray,       # (Ny, Nx)
    juvenile_fft_kernels: jnp.ndarray,      # (4, Ly, Lx)
    juvenile_edge_corrections: jnp.ndarray, # (4, Ny, Nx)
    Q: jnp.ndarray,                         # (4, Ny, Nx)
    dirichlet_weights: jnp.ndarray = None,
    pad_value: float = 1e-9,
):
    """
    Vectorized juvenile dispersal across four directional kernels with mortality.

    Returns
    -------
    pop_after: (Ny, Nx) array after dispersal and survival
    """
    Ny, Nx = juvenile_dispersers.shape
    Nd = juvenile_fft_kernels.shape[0]  # should be 4

    # ----------------------------
    # Determine directional subpopulations
    # ----------------------------
    if dirichlet_weights is None:
        dirichlet_weights = jnp.ones(Nd) / Nd
    dirichlet_weights = dirichlet_weights.reshape(Nd, 1, 1)  # broadcastable
    dirichlet_weights *= Nd # Quick fix as directional kernels each have 1/Nd mass
    pop_split = juvenile_dispersers[None, :, :] * dirichlet_weights * juvenile_edge_corrections  # (4, Ny, Nx)

    # ----------------------------
    # Pad pop for FFT
    # ----------------------------
    Ly, Lx = juvenile_fft_kernels.shape[1:]
    pop_pad = jnp.pad(pop_split, ((0,0), (0, Ly-Ny), (0, Lx-Nx)), constant_values=pad_value)  # (4, Ly, Lx)

    # ----------------------------
    # FFT-based convolution (all directions at once)
    # ----------------------------
    fft_pop = jnp.fft.fft2(pop_pad, axes=(-2,-1))
    conv = jnp.fft.ifft2(fft_pop * juvenile_fft_kernels, axes=(-2,-1))
    conv_real = jnp.real(conv)[:, :Ny, :Nx]  # crop back to original size

    # ----------------------------
    # Apply Q (survival)
    # ----------------------------
    conv_corrected = conv_real * Q  # all broadcasted

    # ----------------------------
    # Sum over directions
    # ----------------------------
    pop_after = jnp.sum(conv_corrected, axis=0)

    return pop_after

def dispersal_step(pop, K,
                   dispersal_logit_intercept=0.0,
                   dispersal_logit_slope=10.0,
                   target_fraction=0.8,
                   dispersal_intercept=0.0,
                   dispersal_random=None,
                   adult_edge_correction=1.0,
                   juvenile_edge_correction_stack=1.0,
                   adult_fft_kernel=None,
                   juvenile_fft_kernel_stack=None,
                   Q_array=None,
                   eps=1e-6,
                   clip_z=30.0,
                   min_p=1e-8):
    """
    Full dispersal step with numerical stability, juvenile/adult split, and convolution update.
    
    Parameters
    ----------
    pop : array
        Current population at all grid cells.
    K : array
        Carrying capacity at all grid cells.
    dispersal_logit_intercept : float
        Baseline logit for total dispersal probability.
    dispersal_logit_slope : float
        Steepness of total dispersal probability vs relative density.
    target_fraction : float
        Fraction of K where dispersal probability ~0.5.
    dispersal_intercept : float
        Baseline juvenile dispersal probability (logit) modifier.
    dispersal_random : array
        Random temporal variation for juvenile dispersal (same shape as pop).
    adult_edge_correction : float or array
        Edge correction multiplier for adults.
    juvenile_edge_correction : float or array
        Edge correction multiplier for juveniles.
    adult_fft_kernel : array
        Convolution kernel for adult dispersal.
    juvenile_fft_kernel : array
        Convolution kernel for juvenile dispersal.
    eps : float
        Small number added to K to avoid division by zero.
    clip_z : float
        Maximum absolute value of sigmoid input for stability.
    min_p : float
        Minimum/maximum probability to avoid exact 0 or 1.
    
    Returns
    -------
    pop_new : array
        Updated population after dispersal step.
    dispersers : array
        Total number of dispersers (for diagnostics if needed).
    """
    # -----------------------------
    # Total dispersal probability
    # -----------------------------
    K_safe = K + eps
    z_total = dispersal_logit_intercept + dispersal_logit_slope * (pop / K_safe - target_fraction)
    z_total = jnp.clip(z_total, -clip_z, clip_z)
    p_total = jnn.sigmoid(z_total)
    p_total = jnp.clip(p_total, min_p, 1 - min_p)
    
    dispersers = p_total * pop

    # -----------------------------
    # Juvenile vs adult split
    # -----------------------------
    if dispersal_random is None:
        dispersal_random = jnp.zeros_like(pop)
        
    z_juvenile = dispersal_intercept + dispersal_random
    z_juvenile = jnp.clip(z_juvenile, -clip_z, clip_z)
    p_juvenile = jnn.sigmoid(z_juvenile)
    p_juvenile = jnp.clip(p_juvenile, min_p, 1 - min_p)

    juvenile_dispersers = dispersers * p_juvenile
    adult_dispersers = dispersers - juvenile_dispersers

    # -----------------------------
    # Convolution for dispersal movement
    # -----------------------------
    adult_update = rightpad_convolution(adult_dispersers * adult_edge_correction, adult_fft_kernel)
    # juvenile_update = rightpad_convolution(juvenile_dispersers * juvenile_edge_correction, juvenile_fft_kernel)

    juvenile_update = juvenile_dispersal_vectorized(
        juvenile_dispersers=juvenile_dispersers,
        juvenile_fft_kernels = juvenile_fft_kernel_stack, 
        juvenile_edge_corrections=juvenile_edge_correction_stack,
        Q=Q_array,
    )
    pop_update = adult_update + juvenile_update - dispersers

    # -----------------------------
    # Update population
    # -----------------------------
    pop_new = pop + pop_update

    return pop_new


def precompute_r_K(
    time,
    Ny,
    Nx,
    land_rows,
    land_cols,
    r_mean,
    r_bioclim,     # (N_land,)
    r_pop,         # (time * N_land,)
    r_spatial,     # (N_land,)
    r_temp,        # (time,)
    K_raw_mean,
    logK_bio,      # (N_land,)
    logK_pop,      # (time * N_land,)
):
    """
    Precompute r_array and K_array for 2D-native forward simulation.

    Returns
    -------
    r_array : (time, Ny, Nx)
    K_array : (time, Ny, Nx)
    """

    N_land = land_rows.shape[0]

    # -----------------------------
    # Validate shapes (cheap, safe)
    # -----------------------------
    assert r_bioclim.shape == (N_land,)
    assert r_spatial.shape == (N_land,)
    assert logK_bio.shape == (N_land,)

    assert r_pop.size == time * N_land
    assert logK_pop.size == time * N_land
    assert r_temp.shape == (time,)

    # -----------------------------
    # Reshape time-varying terms
    # -----------------------------
    r_pop = r_pop.reshape(time, N_land)
    logK_pop = logK_pop.reshape(time, N_land)

    # -----------------------------
    # Combine growth-rate terms
    # -----------------------------
    r_vals = (
        r_mean
        + r_bioclim[None, :]
        + r_spatial[None, :]
        + r_pop
        + r_temp[:, None]
    )                           # (time, N_land)

    # -----------------------------
    # Combine carrying-capacity terms
    # -----------------------------
    K_vals = jnp.exp(           # enforce positivity
        K_raw_mean
        + logK_bio[None, :]
        + logK_pop
    )                           # (time, N_land)

    # -----------------------------
    # Scatter into 2D grids
    # -----------------------------
    r_array = jnp.zeros((time, Ny, Nx))
    K_array = jnp.zeros((time, Ny, Nx))

    t_idx = jnp.arange(time)[:, None]

    r_array = r_array.at[t_idx, land_rows, land_cols].set(r_vals)
    K_array = K_array.at[t_idx, land_rows, land_cols].set(K_vals)

    return r_array, K_array

def precompute_s_Q_vectorized(
    r_array,                    # (time, Ny, Nx)
    juvenile_fft_kernel_stack,  # (Nd, Ly, Lx)
    a,                          # slope for survival
    r0,                         # threshold for survival
    pad_value: float = 1.0,     # neutral survival outside domain
):
    """
    Vectorized precompute s and Q for all directions simultaneously.
    
    Parameters
    ----------
    r_array : jnp.ndarray
        Growth rate array, shape (time, Ny, Nx)
    juvenile_fft_kernel_stack : jnp.ndarray
        FFT of directional juvenile kernels, shape (Nd, Ly, Lx)
    a : float
        Survival slope
    r0 : float
        Survival threshold
    pad_value : float
        Value to pad outside of grid to avoid artifacts (default 1.0)
    
    Returns
    -------
    s_array : jnp.ndarray
        Survival probability array, shape (time, Ny, Nx)
    Q_array : jnp.ndarray
        Directional survival expectation, shape (time, Nd, Ny, Nx)
    """

    time, Ny, Nx = r_array.shape
    Nd, Ly, Lx = juvenile_fft_kernel_stack.shape

    # -----------------------------
    # Compute survival s
    # -----------------------------
    s_array = 1 / (1 + jnp.exp(-a * (r_array - r0)))  # (time, Ny, Nx)

    # -----------------------------
    # Pad s_array to FFT kernel size
    # -----------------------------
    pad_y = Ly - Ny
    pad_x = Lx - Nx
    s_pad = jnp.pad(
        s_array,
        ((0, 0), (0, pad_y), (0, pad_x)),
        constant_values=pad_value
    )  # (time, Ly, Lx)

    # -----------------------------
    # FFT of s (all timesteps)
    # -----------------------------
    S_fft = jnp.fft.fft2(s_pad, axes=(1, 2))  # shape (time, Ly, Lx)

    # -----------------------------
    # Broadcast for directions
    # -----------------------------
    # S_fft: (1, time, Ly, Lx), kernels: (Nd, 1, Ly, Lx)
    S_fft_b = S_fft[:, None, :, :]                     # (time, 1, Ly, Lx)
    K_fft_b = juvenile_fft_kernel_stack[None, :, :, :]  # (1, Nd, Ly, Lx)

    # -----------------------------
    # Multiply in Fourier space
    # -----------------------------
    Q_fft = K_fft_b * S_fft_b                           # (time, Nd, Ly, Lx)

    # -----------------------------
    # Inverse FFT to get directional Q
    # -----------------------------
    Q_full = jnp.fft.ifft2(Q_fft, axes=(2, 3))
    Q_array = jnp.real(Q_full)[:, :, :Ny, :Nx]          # crop to original grid

    return s_array, Q_array


# ---------------------------
# Forward simulation (2D native)
# ---------------------------

def forward_sim_2d(time, initial_pop, r_array, K_array,
                    dispersal_logit_intercept, dispersal_logit_slope,
                    dispersal_intercept, dispersal_random,
                    adult_fft_kernel, juvenile_fft_kernel_stack,
                    adult_edge_correction, juvenile_edge_correction_stack,
                    Q_array,
                    land_mask,
                    inv_pop, inv_location, inv_timestep, 
                    allee_scalar, allee_intercept,
                    pseudo_zero, target_fraction=0.8):
    """
    2D native simulation with precomputed r and K arrays.
    r_array: [time, Ny, Nx] growth rate
    K_array: [time, Ny, Nx] carrying capacity
    land_mask: 0/1 masks
    """

    row, col = inv_location
    # jax.debug.print("min/max N0={}/{}", jnp.min(initial_pop), jnp.max(initial_pop))
    def step(pop, t):
        
        # jax.debug.print("t={}: Starting pop={}", t, jnp.sum(pop))
        # -----------------------------
        # Invasion event
        # -----------------------------
        def invasion_amount(t):
            k = t - inv_timestep
            return jnp.where(
                (k >= 0) & (k < inv_pop.shape[0]),
                inv_pop[k],
                0.0,
            )

        add_val = invasion_amount(t)
        pop = pop.at[row, col].add(add_val)

        # -----------------------------
        # Extract r, K and Q for this timestep
        # -----------------------------
        r = r_array[t]  # [Ny, Nx]
        K = K_array[t]  # [Ny, Nx]
        Q = Q_array[t]  # [Nd, Ny, Nx]          

        pop = dispersal_step(
            pop=pop,
            K=K,
            dispersal_logit_intercept=dispersal_logit_intercept,
            dispersal_logit_slope=dispersal_logit_slope,
            target_fraction=target_fraction,
            dispersal_intercept=dispersal_intercept,
            dispersal_random=dispersal_random[t],
            adult_edge_correction=adult_edge_correction,
            juvenile_edge_correction_stack=juvenile_edge_correction_stack,
            adult_fft_kernel=adult_fft_kernel,
            juvenile_fft_kernel_stack=juvenile_fft_kernel_stack,
            Q_array=Q,
            eps=1e-6,
            clip_z=30.0,
            min_p=1e-8
        )

        # Compute reproduction across the full 2D array, but only affects land cells
        pop = land_mask * (reproduction_safe(pop, r, K, allee_scalar, allee_intercept)) + pseudo_zero

        # jax.debug.print("t={}: pop after reproduction={}", t, jnp.sum(pop))

        return pop, pop

    _, land_over_time = lax.scan(step, initial_pop, jnp.arange(time))
    # jax.debug.print("min/max N123={}/{}", jnp.min(land_over_time[123,:,:]), jnp.max(land_over_time[123,:,:]))
    # jax.debug.print("min/max R={}/{}", jnp.min(r_array), jnp.max(r_array))
    # jax.debug.print("min/max K={}/{}", jnp.min(K_array), jnp.max(K_array))
    return land_over_time  # shape: [time, Ny, Nx]


# ---------------------------
# Full NumPyro model
# ---------------------------

def build_model_2d(data, anneal=0.1, ):
    Nx = data['Nx']
    Ny = data['Ny']
    time = data['time']
    land_rows = data['land_rows']
    land_cols = data['land_cols']
    ocean_rows = data['ocean_rows']
    ocean_cols = data['ocean_cols']   
    N_land = len(land_rows)
    phi_bioclim = data['phi_bioclim']
    ell_bioclim = data['ell_bioclim']
    m_bio_per_dim = data['m_bioclim']
    d_bioclim = data['d_bioclim']
    phi_pop=data['phi_pop']
    ell_pop=data['ell_pop']
    m_pop_per_dim=data['m_pop']

    # ---------------------------
    # Scalar priors
    # ---------------------------
    r_mean = numpyro.sample("r_mean", dist.Normal(0.0, 1.0 * anneal))
    K_raw_mean = numpyro.sample("K_raw_mean", dist.Normal(-1.0, 1.0 * anneal))
    allee_intercept = numpyro.sample("allee_intercept", dist.Normal(0, 1.0 * anneal))
    allee_slope = jnn.softplus(numpyro.sample("allee_slope_raw", dist.Normal(2.0, 1.0 * anneal)))
    dispersal_intercept = numpyro.sample("dispersal_intercept", dist.Normal(0., 1.0 * anneal))
    dispersal_logit_intercept = numpyro.sample("dispersal_logit_intercept", dist.Normal(2.0, 1.0 * anneal))
    dispersal_logit_slope = numpyro.sample("dispersal_logit_slope", dist.Normal(4.0, 1.0 * anneal))
    dispersal_survival_threshold = numpyro.sample("dispersal_survival_threshold", dist.Normal(-1.0, 1.0 * anneal))
    dispersal_survival_slope = jnn.softplus(numpyro.sample("dispersal_survival_slope_raw", dist.Normal(1.0, 1.0 * anneal)))

    # ------------------------------------------------------------
    # 1. BIOCLIM HSGP (rank-2 coregionalization)
    # ------------------------------------------------------------
    M_bio_total = phi_bioclim.shape[1]
    M_pop_total = phi_pop.shape[1]

    # Kernel hyperparameters
    alpha_bio = jnn.softplus(numpyro.sample("alpha_bio_raw", dist.Normal(0., 1.0 * anneal)))
    length_bio = jnn.softplus(numpyro.sample("length_bio_raw", dist.Normal(0., 1.0 * anneal)))

    # Spectral density square-root: spd_bio has shape (M_bio,)
    spd_bio = jnp.sqrt(
        diag_spectral_density_squared_exponential(
            alpha=alpha_bio,
            length=length_bio,
            ell=ell_bioclim,
            m=m_bio_per_dim,
            dim=d_bioclim,  # 3D bioclim space (e.g. PC1, PC2, PC3)
        )
    )

    with scope(prefix="f1_bio"):
        # Two independent latent bioclim GPs over the same basis
        # Each call returns shape (N_land,)
        f_bio_1 = linear_approximation(
            phi=phi_bioclim,  # (N_land, M_bio)
            spd=spd_bio,      # (M_bio,)
            m=M_bio_total,
            non_centered=True,
        )

    with scope(prefix="f2_bio"):
        f_bio_2 = linear_approximation(
            phi=phi_bioclim,
            spd=spd_bio,
            m=M_bio_total,
            non_centered=True,
        )

    # Stack into (rank=2, N_land)
    f_bio = jnp.stack([f_bio_1, f_bio_2], axis=0)

    # Coregionalization matrix: shape (2 outputs, 2 latent)
    # outputs = [r_bio, logK_bio]
    L_bio = numpyro.sample(
        "L_bio",
        dist.Normal(0.0, 1.0 * anneal).expand([2, 2])
    )

    # Output-specific fields: shape (2, N_land)
    # [r_bio, logK_bio] = L_bio @ f_bio
    r_bio, logK_bio = L_bio @ f_bio  # each (N_land,)

    # ------------------------------------------------------------
    # 2. HUMAN-POPULATION HSGP (rank-1 coregionalization)
    # ------------------------------------------------------------
    with scope(prefix="pop"):

        # Kernel hyperparameters
        alpha_pop = jnn.softplus(numpyro.sample("alpha_pop_raw", dist.Normal(-1., 1.0 * anneal)))
        length_pop = jnn.softplus(numpyro.sample("length_pop_raw", dist.Normal(0., 1.0 * anneal)))

        # Spectral density square-root: spd_bio has shape (M_pop,)
        spd_pop = jnp.sqrt(
            diag_spectral_density_squared_exponential(
                alpha=alpha_pop,
                length=length_pop,
                ell=ell_pop,
                m=m_pop_per_dim,
                dim=1,  # 1D pop space
            )
        )

        # One latent GP field over space × time
        # f_pop has shape (1, N_land * N_years)
        f_pop = linear_approximation(
            phi=phi_pop,
            spd=spd_pop,
            m=M_pop_total,
            non_centered=True,
        ).squeeze()                                 # returns (N_st)

        # Rank-1 mixing: two scalars
        b_r = numpyro.sample("b_r", dist.Normal(0., 1.0 * anneal))
        b_K = numpyro.sample("b_K", dist.Normal(0., 1.0 * anneal))

        # Human-pop contributions
        r_pop = b_r * f_pop              # (N_st,)
        logK_pop = b_K * f_pop           # (N_st,)

    # ---------------------------
    # Other growth
    # ---------------------------

    inv_pop = jnn.softplus(numpyro.sample(
        "inv_eta",
        dist.Normal(-5.0, 1.0 * anneal),
        sample_shape=(data['inv_window'],)
    ))
    allee_scalar = data['pop_scalar'] * allee_slope

    # ---------------------------
    # Temporal/spatial effects
    # ---------------------------
    r_spatial = numpyro.sample("r_spatial", dist.Normal(0., 0.01 * anneal), sample_shape=(N_land,))
    r_temp = numpyro.sample("r_temp", dist.Normal(0., 0.01 * anneal), sample_shape=(time,))
    dispersal_random = numpyro.sample("dispersal_random", dist.Normal(0., 0.01 * anneal), sample_shape=(time,))

    # ---------------------------
    # Precompute r_array and K_array
    # ---------------------------
    r_array, K_array = precompute_r_K(
        time=time,
        Ny=Ny,
        Nx=Nx,
        land_rows=land_rows,
        land_cols=land_cols,
        r_mean=r_mean,
        r_bioclim=r_bio,
        r_pop=r_pop,
        r_spatial=r_spatial,
        r_temp=r_temp,
        K_raw_mean=K_raw_mean,
        logK_bio=logK_bio,
        logK_pop=logK_pop
    )

    s_array, Q_array = precompute_s_Q_vectorized(
        r_array,
        data['juvenile_fft_kernel_stack'],      # (4, Ny, Nx), directions first
        dispersal_survival_slope,               # slope for survival of dispersing juveniles
        dispersal_survival_threshold            # threshold for survival of dispersing juveniles
    )

    # ---------------------------
    # Forward simulation
    # ---------------------------
    densities = forward_sim_2d(
        time=time,
        initial_pop=data['initpop_latent'],
        r_array=r_array,
        K_array=K_array,
        dispersal_logit_intercept=dispersal_logit_intercept,
        dispersal_logit_slope=dispersal_logit_slope,
        dispersal_intercept=dispersal_intercept,
        dispersal_random=dispersal_random,
        adult_fft_kernel=data['adult_fft_kernel'],
        juvenile_fft_kernel_stack=data['juvenile_fft_kernel_stack'],
        adult_edge_correction=data['adult_edge_correction'],
        juvenile_edge_correction_stack=data['juvenile_edge_correction_stack'],
        Q_array=Q_array,
        land_mask=data['land_mask'],
        inv_pop=inv_pop,
        inv_location=data['inv_location'],
        inv_timestep = data['inv_timestep'],
        allee_scalar=allee_scalar,
        allee_intercept=allee_intercept,
        pseudo_zero=data['pseudo_zero']
    )

    # ---------------------------
    # Likelihood
    # ---------------------------
    # Assume data provides obs_years, obs_rows, obs_cols, and observed_results
    # All are 1D arrays of length N_obs
    t_idx = data["obs_time_indices"]    # 0-based indices into time
    rows = data["obs_rows"]             # 0-based row indices
    cols = data["obs_cols"]             # 0-based column indices

    # Extract densities at observed locations
    densities_obs = densities[t_idx, rows, cols] * data["pop_scalar"]

    # Single vectorized Poisson likelihood
    numpyro.sample("obs", dist.Poisson(densities_obs), obs=data["observed_results"])


