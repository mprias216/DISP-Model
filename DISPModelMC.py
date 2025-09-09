import pyreadstat
import pandas as pd
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

# =====================
# General Column Detection Utilities
# =====================

def auto_detect_column(columns, keywords):
    for col in columns:
        for key in keywords:
            if key in col.lower():
                return col
    return None

def load_data(file_path):
    if file_path.endswith(".sav"):
        df, _ = pyreadstat.read_sav(file_path)
    elif file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file format.")
    df.columns = [col.strip() for col in df.columns]
    return df

# === Load file ===
file_path = "Data_Experiment 2.sav"  # Replace with your file
real_df = load_data(file_path)

# Auto-detect relevant columns
subject_col = auto_detect_column(real_df.columns, ["subject", "participant", "id"])
rt_col = auto_detect_column(real_df.columns, ["rt", "response_time", "latency"])

if not subject_col or not rt_col:
    raise ValueError("Could not detect subject or RT column. Please rename or specify.")

print(f"Detected subject column: {subject_col}")
print(f"Detected RT column: {rt_col}")

# =====================
# DISP Parameters and Simulation
# =====================

params = {
    "C_High": 1.4,
    "mu_R": 0.8,
    "mu_F": 0.2,
    "D_R": 0.05,
    "D_F": 0.05,
    "tau": 0.01,
    "B_r": 0.6,
    "x0": -0.1,
    "mu_R0": 0.1,
    "mu_F0": 0.1,
    "delta_r": 0.1,
    "t0": 0.25
}

def simulate_disp_trial(C_High, mu_R, mu_F, D_R, D_F, tau, B_r, x0, t0, delta_r, max_time=5.0, dt=0.01):
    time = 0
    evidence = x0
    boundary = 1.0
    while abs(evidence) < boundary and time < max_time:
        noise = np.random.normal(0, np.sqrt(D_R + D_F))
        drift = mu_R + mu_F
        evidence += drift * dt + noise * np.sqrt(dt)
        boundary = max(0.01, boundary - tau * dt)
        time += dt
    decision = 1 if evidence >= B_r else 0
    confidence = 2 if evidence >= C_High else 1 if evidence >= 0 else 0
    rt = time + t0 + delta_r
    return decision, confidence, rt

def simulate_disp_trials(num_trials, params):
    results = []

    # Only extract the parameters that simulate_disp_trial() actually uses
    sim_params = {k: params[k] for k in [
        "C_High", "mu_R", "mu_F", "D_R", "D_F", "tau", "B_r", "x0", "t0", "delta_r"
    ]}

    for _ in range(num_trials):
        _, _, rt = simulate_disp_trial(**sim_params)
        results.append(rt)

    return np.array(results)

def disp_loss(param_vector):
    mu_R, mu_F, D_R, D_F, tau, B_r, x0, C_High, t0, delta_r = param_vector
    sim_rts = simulate_disp_trials(1000, {
        "mu_R": mu_R, "mu_F": mu_F, "D_R": D_R, "D_F": D_F,
        "tau": tau, "B_r": B_r, "x0": x0, "C_High": C_High,
        "t0": t0, "delta_r": delta_r
    })
    actual_q = np.quantile(actual_rts, [0.1, 0.3, 0.5, 0.7, 0.9])
    sim_q = np.quantile(sim_rts, [0.1, 0.3, 0.5, 0.7, 0.9])
    return np.mean((actual_q - sim_q) ** 2)

# =====================
# Fit Model Per Subject
# =====================

labels = ["mu_R", "mu_F", "D_R", "D_F", "tau", "B_r", "x0", "C_High", "t0", "delta_r"]
initial_guess = [params[k] for k in labels]
bounds = [
    (0.05, 1.5),  # mu_R
    (0.05, 1.5),  # mu_F
    (0.01, 0.5),  # D_R
    (0.01, 0.5),  # D_F
    (0.0, 0.2),   # tau
    (0.5, 1.0),   # B_r
    (-0.5, 0.5),  # x0
    (0.5, 2.0),   # C_High
    (0.1, 1.0),   # t0
    (0.0, 1.0)    # delta_r
]


subject_ids = real_df[subject_col].dropna().unique()[:4]

for i, sub_id in enumerate(subject_ids, start=1):
    print(f"\nRunning model fit for Participant S{i} (Subject ID {sub_id})")

    df_sub = real_df[real_df[subject_col] == sub_id]
    actual_rts = df_sub[rt_col].dropna().values

    if len(actual_rts) < 10:
        print("  Skipping (not enough RTs).")
        continue

    result = scipy.optimize.minimize(disp_loss, initial_guess, bounds=bounds, method="L-BFGS-B")
    fitted_params = dict(zip(labels, result.x))
    params.update(fitted_params)

    print("\nBest-Fitting DISP Parameters:")
    for name in labels:
        print(f"{name}: {fitted_params[name]:.4f}")
    print(f"MSE Loss: {result.fun:.5f}")

    # Simulate and compute NLL
    simulated_rts = simulate_disp_trials(len(actual_rts) * 10, params)

    # Estimate PDF of simulated RTs
    kde = gaussian_kde(simulated_rts, bw_method='scott')  # or try 'silverman'

    # Compute likelihoods for each actual RT
    likelihoods = kde(actual_rts)

    # Clip to avoid log(0)
    likelihoods = np.clip(likelihoods, 1e-12, None)

    # Compute NLL
    nll = -np.sum(np.log(likelihoods))

    print(f"Negative Log Likelihood (NLL) for S{i}: {nll:.2f}")

    # Plot comparison
    print("First 5 actual RTs:", actual_rts[:5])
    print("First 5 simulated RTs:", simulated_rts[:5])

    plt.figure(figsize=(10, 4))
    plt.hist(actual_rts, bins=50, alpha=0.5, label='Actual RTs')
    plt.hist(simulated_rts, bins=50, alpha=0.5, label='Simulated RTs')
    plt.title(f"S{i} RT Distribution")
    plt.xlabel("RT (s)")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.show()
