import numpy as np
import matplotlib.pyplot as plt

# 1. Setup Temperature Range
temps = np.linspace(15, 50, 400)

# 2. Simulate Species Abundance ("Bio-Sensors")
def sigmoid(x, center, k=3.0):
    return 100 / (1 + np.exp(k * (x - center)))

np.random.seed(42)
species_data = []
for _ in range(5):
    species_data.append(sigmoid(temps, center=32 + np.random.uniform(-1, 1), k=3.0)) 
for _ in range(5):
    species_data.append(100 - sigmoid(temps, center=32 + np.random.uniform(-1, 1), k=3.0)) 

abundances = np.stack(species_data, axis=-1)

# 3. Calculate Similarity (Original Plots Logic)
ref_idx = np.argmin(np.abs(temps - 25))
ref_temp = temps[ref_idx]
ref_abund = abundances[ref_idx]

l_temp = 5.0
sim_se = np.exp(-0.5 * (temps - ref_temp)**2 / l_temp**2)

def calculate_ruzicka(u, v):
    return np.sum(np.minimum(u, v)) / (np.sum(np.maximum(u, v)) + 1e-8)

sim_ruzicka = np.array([calculate_ruzicka(ref_abund, abundances[i]) for i in range(len(temps))])

# --- 4. GP PREDICTION WITH CENTERED PRIOR ---
target_ice = (temps < 32).astype(float) * 100
# Sparse indices to force the GP to "guess" between points
train_indices = [30, 80, 220, 360] 
train_x_temp = temps[train_indices]
train_x_abund = abundances[train_indices]
train_y = target_ice[train_indices]

def gp_posterior(K_train, K_test_train, K_test_test, y_train, prior_mu=50.0, noise=1e-2):
    """GP with a centered prior mean and proper variance scaling."""
    # Center the data around the prior mean
    y_centered = y_train - prior_mu
    
    K_inv = np.linalg.inv(K_train + np.eye(len(K_train)) * noise)
    
    # Mean: Prior + K_* @ K_inv @ (y - Prior)
    mu = prior_mu + K_test_train @ K_inv @ y_centered
    
    # Variance: K_** - K_* @ K_inv @ K_*.T
    # This is where the 'fans' come from
    post_cov = K_test_test - K_test_train @ K_inv @ K_test_train.T
    std = np.sqrt(np.diag(post_cov).clip(min=0))
    return mu, std

# Kernel Matrices (Temp-based)
# Outputscale (v) is set to 2500 because (range/2)^2 = 50^2
v_scale = 2500.0
K_temp = v_scale * np.exp(-0.5 * (train_x_temp[:, None] - train_x_temp[None, :])**2 / l_temp**2)
Ks_temp = v_scale * np.exp(-0.5 * (temps[:, None] - train_x_temp[None, :])**2 / l_temp**2)
Kss_temp = v_scale * np.exp(-0.5 * (temps[:, None] - temps[None, :])**2 / l_temp**2)
mu_temp, std_temp = gp_posterior(K_temp, Ks_temp, Kss_temp, train_y)

# Kernel Matrices (Ruzicka-based)
def ruz_matrix(A, B):
    return v_scale * np.array([[calculate_ruzicka(a, b) for b in B] for a in A])

K_ruz = ruz_matrix(train_x_abund, train_x_abund)
Ks_ruz = ruz_matrix(abundances, train_x_abund)
Kss_ruz = ruz_matrix(abundances, abundances)
mu_ruz, std_ruz = gp_posterior(K_ruz, Ks_ruz, Kss_ruz, train_y)

# --- 5. FINAL UNIFIED VISUALIZATION ---
fig, axes = plt.subplots(4, 1, figsize=(11, 18), sharex=True)

# PLOT 1: Biological Sensors
axes[0].plot(temps, abundances[:, :5], color='blue', alpha=0.3)
axes[0].plot(temps, abundances[:, 5:], color='red', alpha=0.3)
axes[0].axvline(32, color='black', linestyle='--', label='32°F Threshold')
axes[0].set_title("1. Biological 'Sensor' Response (Abundances)", fontweight='bold')
axes[0].set_ylabel("Abundance")

# PLOT 2: Similarity Measures
axes[1].plot(temps, sim_se, color='gray', lw=2, linestyle=':', label='SE Kernel (Temp)')
axes[1].plot(temps, sim_ruzicka, color='forestgreen', lw=3, label='Ruzicka Kernel (Bio)')
axes[1].axvline(32, color='black', linestyle='--')
axes[1].scatter(ref_temp, 1.0, color='purple', s=80, zorder=5, label='Ref (25°F)')
axes[1].set_title("2. Similarity to Frozen Day: Temp vs. Bio-Features", fontweight='bold')
axes[1].set_ylabel("Similarity Score")

# PLOT 3: Temp-based GP (The "Smeared" Prediction)
axes[2].plot(temps, target_ice, 'k--', alpha=0.3, label='True Ice State')
axes[2].scatter(train_x_temp, train_y, color='black', s=40, zorder=5)
axes[2].plot(temps, mu_temp, color='gray', lw=2, label='GP Mean (Temp)')
axes[2].fill_between(temps, mu_temp - 2*std_temp, mu_temp + 2*std_temp, color='gray', alpha=0.2, label='95% CI')
axes[2].set_title("3. GP Prediction: SE on Temperature (Smeared)", fontweight='bold', color='gray')
axes[2].set_ylabel("% Frozen")
axes[2].set_ylim(-20, 120)

# PLOT 4: Bio-based GP (The "Sharp" Prediction)
axes[3].plot(temps, target_ice, 'k--', alpha=0.3, label='True Ice State')
axes[3].scatter(train_x_temp, train_y, color='black', s=40, zorder=5)
axes[3].plot(temps, mu_ruz, color='forestgreen', lw=3, label='GP Mean (Bio)')
axes[3].fill_between(temps, mu_ruz - 2*std_ruz, mu_ruz + 2*std_ruz, color='forestgreen', alpha=0.2, label='95% CI')
axes[3].set_title("4. GP Prediction: Ruzicka Abundance Prior (Sharp)", fontweight='bold', color='forestgreen')
axes[3].set_xlabel("Temperature (°F)")
axes[3].set_ylabel("% Frozen")
axes[3].set_ylim(-20, 120)

plt.tight_layout()

plt.savefig("test_ponds.png")
plt.show()