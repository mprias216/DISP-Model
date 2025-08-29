import pyreadstat
import pandas as pd
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import seaborn as sns


# LOAD EXPERIMENT 1 DATA
real_df, meta = pyreadstat.read_sav("Masterfile Experiment 1.sav")
actual_rts = real_df["old_new_decision.rt"].dropna().values  # Check this column name if it gives errors


# STARTING PARAMETERS (USED FOR SIMULATION BEFORE MLE)
params = {
    "C_High": 1.0,
    "mu_R": 0.35,
    "mu_F": 0.25,
    "D_R": 0.1,
    "D_F": 0.1,
    "tau": 0.05,
    "B_r": 0.8,
    "x0": -0.1,
    "mu_R0": 0.15,  # Unused
    "mu_F0": 0.05,  # Unused
    "delta_r": 0.2,
    "t0": 0.3
}


# DISP TRIAL SIMULATION
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
    for _ in range(num_trials):
        _, _, rt = simulate_disp_trial(
            C_High=params["C_High"], mu_R=params["mu_R"], mu_F=params["mu_F"],
            D_R=params["D_R"], D_F=params["D_F"], tau=params["tau"],
            B_r=params["B_r"], x0=params["x0"], t0=params["t0"], delta_r=params["delta_r"]
        )
        results.append(rt)
    return np.array(results)


# LOSS FUNCTION FOR MLE (Mean Squared Error between actual and simulated RTs)
def disp_loss(param_vector):
    mu_R, mu_F, D_R, D_F, tau, B_r, x0, C_High, t0, delta_r = param_vector
    sim_rts = simulate_disp_trials(300, {
        "mu_R": mu_R, "mu_F": mu_F, "D_R": D_R, "D_F": D_F,
        "tau": tau, "B_r": B_r, "x0": x0, "C_High": C_High,
        "t0": t0, "delta_r": delta_r,
        "mu_R0": 0.1, "mu_F0": 0.1
    })
   # min_len = min(len(actual_rts), len(sim_rts))
   # return np.mean((actual_rts[:min_len] - sim_rts[:min_len])**2)
    quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]


    # Compute Vincentized RTs
    actual_q = np.quantile(actual_rts, quantiles)
    sim_q = np.quantile(sim_rts, quantiles)


    # Mean squared error between quantiles
    return np.mean((actual_q - sim_q) ** 2)


# RUN MLE OPTIMIZATION
initial_guess = [params["mu_R"], params["mu_F"], params["D_R"], params["D_F"],
                 params["tau"], params["B_r"], params["x0"], params["C_High"],
                 params["t0"], params["delta_r"]]


bounds = [(0.01, 1.0)] * len(initial_guess)


result = scipy.optimize.minimize(disp_loss, initial_guess, bounds=bounds, method="L-BFGS-B")


# PRINT BEST-FIT PARAMETERS
labels = ["mu_R", "mu_F", "D_R", "D_F", "tau", "B_r", "x0", "C_High", "t0", "delta_r"]
print("\n Best-Fitting DISP Parameters (from MLE):\n")
for name, value in zip(labels, result.x):
    print(f"{name}: {value:.4f}")
print(f"\nMSE Loss: {result.fun:.5f}")


# UPDATE YOUR PARAMS DICTIONARY FOR REUSE
params.update(dict(zip(labels, result.x)))


# PLOT ACTUAL vs. SIMULATED RTs WITH BEST-FIT PARAMETERS (YOU CAN CHANGE THIS TO ANOTHER VISUAL PLOT OR GRAPH WITH DATA YOU WANT TO COMPARE)
simulated_rts = simulate_disp_trials(1900, params)




# FOR Negative Log Likelihood (NLL)
from scipy.stats import gaussian_kde


# Make sure actual_rts and simulated_rts are both numpy arrays
actual_rts = np.array(actual_rts)
simulated_rts = np.array(simulated_rts)


subject_ids = real_df["subject"].dropna().unique()[:4]  # First 4 subjects


for i, sub_id in enumerate(subject_ids, start=1):
    print(f"\n Running model fit for Participant S{i} (Subject ID {sub_id})")


    df_sub = real_df[real_df["subject"] == sub_id]
    actual_rts = df_sub["old_new_decision.rt"].dropna().values


    # Run the MLE fitting like you did before
    result = scipy.optimize.minimize(disp_loss, initial_guess, bounds=bounds, method="L-BFGS-B")


    fitted_params = dict(zip(labels, result.x))
    params.update(fitted_params)


    simulated_rts = simulate_disp_trials(len(actual_rts), params)


    # Compute NLL (you can replace with your own NLL function if it’s custom)
    epsilon = 1e-9
    residuals = actual_rts - simulated_rts
    sigma = np.std(residuals)
    log_likelihoods = -0.5 * np.log(2 * np.pi * sigma ** 2 + epsilon) - (residuals ** 2) / (2 * sigma ** 2 + epsilon)
    nll = -np.sum(log_likelihoods)


    print(f"Negative Log Likelihood (NLL) for S{i}: {nll:.2f}")




import matplotlib.pyplot as plt


plt.hist(actual_rts, bins=100, alpha=0.5, label='Actual RTs')
plt.hist(simulated_rts, bins=100, alpha=0.5, label='Simulated RTs')
plt.legend()
plt.title("RT Distribution Check")
plt.show()


'''
plt.figure(figsize=(10, 5))
sns.histplot(actual_rts, bins=40, kde=True, label="Actual RTs", color="skyblue", alpha=0.6)
sns.histplot(simulated_rts, bins=40, kde=True, label="Simulated RTs (Best Fit)", color="orange", alpha=0.6)
plt.title("Actual vs Simulated Reaction Time Distribution (MLE-Fitted)")
plt.xlabel("RT (seconds)")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()
'''



