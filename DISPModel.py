import pyreadstat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# === Load Data ===
real_df, meta = pyreadstat.read_sav("Masterfile Experiment 1.sav")

# === Model Parameters ===
params = {
    "mu_R": 0.6,
    "mu_F": 0.4,
    "D_R": 0.05,
    "D_F": 0.09,
    "x0": -0.1,
    "boundary": 1.0
}
mu = params["mu_R"] + params["mu_F"]
D = params["D_R"] + params["D_F"]
x0 = params["x0"]
boundary = params["boundary"]

# === Grid Setup (static across participants) ===
dx = 0.02
dt = 0.001
x = np.arange(-1.5, 1.5 + dx, dx)
x0_idx = np.argmin(np.abs(x - x0))
upper_idx = np.where(x >= boundary)[0][0]
lower_idx = np.where(x <= -boundary)[0][-1]

# === NLL Function ===
def compute_nll(probs):
    return -np.sum(np.log(probs))

# === Subject-wise NLL Computation ===
subject_column = "subject"  # Adjust this if your column is named differently
rt_column = "old_new_decision.rt"
nll_results = {}

unique_subjects = real_df[subject_column].dropna().unique()
unique_subjects.sort()

for subject in unique_subjects:
    print(f"\nRunning model fit for Participant {int(subject)} (Subject ID {subject})")

    subject_rts = real_df[real_df[subject_column] == subject][rt_column].dropna().values
    if len(subject_rts) == 0:
        print(f"  Skipping subject {subject} (no RTs).")
        continue

    T_max = subject_rts.max() + 0.5
    t = np.arange(0, T_max + dt, dt)

    # === Fokker-Planck Initialization ===
    P = np.zeros((len(t), len(x)))
    P[0, x0_idx] = 1.0 / dx
    decision_upper = np.zeros(len(t))
    decision_lower = np.zeros(len(t))

    # === Fokker-Planck Solver ===
    for n in range(len(t) - 1):
        drift = -mu * (P[n, 2:] - P[n, :-2]) / (2 * dx)
        diffusion = D * (P[n, 2:] - 2 * P[n, 1:-1] + P[n, :-2]) / (dx ** 2)
        P[n + 1, 1:-1] = P[n, 1:-1] + dt * (drift + diffusion)

        decision_upper[n + 1] = P[n + 1, upper_idx]
        decision_lower[n + 1] = P[n + 1, lower_idx]
        P[n + 1, upper_idx:] = 0
        P[n + 1, :lower_idx + 1] = 0

    # === Normalize model RTs into PDF ===
    model_pdf = decision_upper / (np.sum(decision_upper) * dt)

    # === Interpolate to actual RTs ===
    interpolator = interp1d(t, model_pdf, bounds_error=False, fill_value=1e-12)
    model_probs = np.clip(interpolator(subject_rts), 1e-12, None)

    # === NLL ===
    nll = compute_nll(model_probs)
    nll_results[subject] = nll
    print(f"  Negative Log-Likelihood (NLL) for S{int(subject)}: {nll:.2f}")

# === Optional: Save all NLLs to CSV ===
# pd.DataFrame.from_dict(nll_results, orient="index", columns=["NLL"]).to_csv("nll_by_subject.csv")

# === Optional: Plot one subject's model fit ===
example_subject = unique_subjects[0]
example_rts = real_df[real_df[subject_column] == example_subject][rt_column].dropna().values
T_max = example_rts.max() + 0.5
t = np.arange(0, T_max + dt, dt)
P = np.zeros((len(t), len(x)))
P[0, x0_idx] = 1.0 / dx
decision_upper = np.zeros(len(t))
decision_lower = np.zeros(len(t))
for n in range(len(t) - 1):
    drift = -mu * (P[n, 2:] - P[n, :-2]) / (2 * dx)
    diffusion = D * (P[n, 2:] - 2 * P[n, 1:-1] + P[n, :-2]) / (dx ** 2)
    P[n + 1, 1:-1] = P[n, 1:-1] + dt * (drift + diffusion)
    decision_upper[n + 1] = P[n + 1, upper_idx]
    decision_lower[n + 1] = P[n + 1, lower_idx]
    P[n + 1, upper_idx:] = 0
    P[n + 1, :lower_idx + 1] = 0

model_pdf = decision_upper / (np.sum(decision_upper) * dt)

plt.figure(figsize=(12, 6))
plt.hist(example_rts, bins=50, density=True, alpha=0.5, color='gray', label="Actual RTs")
plt.plot(t, model_pdf, color='blue', label="Model RTs (Upper Boundary)")
plt.xlabel("Response Time (s)")
plt.ylabel("Probability Density")
plt.title(f"Model vs. Actual RTs (Subject ID {int(example_subject)})")
plt.legend()
plt.tight_layout()
plt.show()
