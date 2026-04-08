"""
Bayesian Laplace distributional regression with hierarchical transfer from AAPL to COIN.

Model:
  y_i ~ Laplace(mu_i, b_i)
  mu_i = beta_0 + beta . X_i   (location, standardized features)
  b_i  = max(gamma_0 + gamma . X_i, floor)   (linear scale, matching frequentist GAMLSS)

Three fits:
  1. AAPL with uninformative priors
  2. COIN with uninformative priors
  3. COIN with AAPL-informed priors (hierarchical transfer)

Two types of prediction intervals:
  - MAP-only: use MAP point estimates of mu, b with closed-form Laplace quantiles
  - Full posterior predictive: sample parameters from posterior, then sample y|params

Method: MAP + Laplace approximation (posterior = N(MAP, H^{-1}))

Output:
  - aapl_bayesian_posteriors.png
  - coin_hierarchical_comparison.png
  - bayesian_calibration.png
  - data/bayesian_results.csv
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

FEATURES = [
    "dollar_value", "log_dollar_value", "participation_rate",
    "roll_spread_500", "roll_vol_500", "exchange_id",
]
COEF_NAMES = ["intercept"] + FEATURES
N_COEFS = len(COEF_NAMES)
P = len(FEATURES)
B_FLOOR = 0.1  # Minimum scale value

# -- Helpers ------------------------------------------------------------------

def load_and_prep(tr_file, te_file):
    df_tr = pd.read_parquet(tr_file)
    df_te = pd.read_parquet(te_file)
    df_tr["abs_impact"] = df_tr["impact_vwap_bps"].abs()
    df_te["abs_impact"] = df_te["impact_vwap_bps"].abs()
    df_tr = df_tr.sort_values("date").reset_index(drop=True)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(df_tr[FEATURES].to_numpy(dtype=np.float64))
    X_te = scaler.transform(df_te[FEATURES].to_numpy(dtype=np.float64))
    y_tr = df_tr["abs_impact"].to_numpy(dtype=np.float64)
    y_te = df_te["abs_impact"].to_numpy(dtype=np.float64)
    return X_tr, y_tr, X_te, y_te, scaler


# Smooth max(x, floor) using softplus: floor + softplus(x - floor)
# softplus(z) = log(1 + exp(z)), smooth approximation of max(z, 0)
SP_SHARPNESS = 10.0  # Higher = closer to hard clip


def soft_floor(x, floor=B_FLOOR, k=SP_SHARPNESS):
    """Smooth approximation of max(x, floor)."""
    z = k * (x - floor)
    # Numerically stable softplus
    return floor + np.where(z > 20, x - floor, np.log1p(np.exp(z)) / k)


def soft_floor_grad(x, floor=B_FLOOR, k=SP_SHARPNESS):
    """Gradient of soft_floor w.r.t. x: sigmoid(k*(x-floor))."""
    z = k * (x - floor)
    return np.where(z > 20, 1.0, np.where(z < -20, 0.0, 1.0 / (1.0 + np.exp(-z))))


def neg_log_posterior(theta, X_c, y, beta_prior_mu, beta_prior_sd,
                      gamma_prior_mu, gamma_prior_sd):
    """Negative log-posterior for Laplace GAMLSS with linear scale link.

    b_i = soft_floor(gamma . X_c_i, floor=0.1)
    """
    p1 = P + 1
    beta = theta[:p1]
    gamma = theta[p1:]

    mu = X_c @ beta
    b_raw = X_c @ gamma
    b = soft_floor(b_raw)

    resid = np.abs(y - mu)
    ll = np.sum(-np.log(2.0 * b) - resid / b)

    lp_beta = -0.5 * np.sum(((beta - beta_prior_mu) / beta_prior_sd) ** 2)
    lp_gamma = -0.5 * np.sum(((gamma - gamma_prior_mu) / gamma_prior_sd) ** 2)

    return -(ll + lp_beta + lp_gamma)


def neg_log_posterior_grad(theta, X_c, y, beta_prior_mu, beta_prior_sd,
                           gamma_prior_mu, gamma_prior_sd):
    """Gradient of negative log-posterior."""
    p1 = P + 1
    beta = theta[:p1]
    gamma = theta[p1:]

    mu = X_c @ beta
    b_raw = X_c @ gamma
    b = soft_floor(b_raw)
    db_draw = soft_floor_grad(b_raw)  # d(b)/d(b_raw)

    resid = y - mu
    abs_resid = np.abs(resid)
    sign_resid = np.sign(resid)

    # d/d_beta: sum(sign(y-mu)/b * X_c)
    grad_beta_ll = X_c.T @ (sign_resid / b)

    # d/d_gamma: chain rule through b = soft_floor(gamma.X)
    # dll/db = -1/b + |y-mu|/b^2
    dll_db = -1.0 / b + abs_resid / (b ** 2)
    # dll/d_gamma = dll/db * db/d(b_raw) * X_c
    grad_gamma_ll = X_c.T @ (dll_db * db_draw)

    grad_beta_prior = -(beta - beta_prior_mu) / (beta_prior_sd ** 2)
    grad_gamma_prior = -(gamma - gamma_prior_mu) / (gamma_prior_sd ** 2)

    grad = np.concatenate([
        -(grad_beta_ll + grad_beta_prior),
        -(grad_gamma_ll + grad_gamma_prior),
    ])
    return grad


def fit_bayesian_laplace(X_tr, y_tr, beta_prior_mu=None, beta_prior_sd=None,
                         gamma_prior_mu=None, gamma_prior_sd=None,
                         label=""):
    n = len(y_tr)
    p1 = P + 1

    if beta_prior_mu is None:
        beta_prior_mu = np.zeros(p1)
        beta_prior_sd = np.full(p1, 5.0)
    if gamma_prior_mu is None:
        gamma_prior_mu = np.zeros(p1)
        gamma_prior_sd = np.full(p1, 5.0)

    X_c = np.column_stack([np.ones(n), X_tr])

    # Initial guess
    from numpy.linalg import lstsq
    beta_init = lstsq(X_c, y_tr, rcond=None)[0]
    resid_init = np.abs(y_tr - X_c @ beta_init)
    gamma_init = lstsq(X_c, np.clip(resid_init, B_FLOOR, None), rcond=None)[0]
    theta0 = np.concatenate([beta_init, gamma_init])

    args = (X_c, y_tr, beta_prior_mu, beta_prior_sd, gamma_prior_mu, gamma_prior_sd)

    print(f"  Optimizing MAP ({label})...", flush=True)
    result = minimize(
        neg_log_posterior, theta0, args=args,
        jac=neg_log_posterior_grad,
        method="L-BFGS-B", options={"maxiter": 5000, "ftol": 1e-12, "gtol": 1e-8},
    )

    if not result.success:
        print(f"  Warning: optimizer message: {result.message}")

    theta_map = result.x
    beta_map = theta_map[:p1]
    gamma_map = theta_map[p1:]

    print(f"  Computing Hessian for Laplace approximation...", flush=True)
    eps = 1e-5
    n_params = len(theta_map)
    hessian = np.zeros((n_params, n_params))
    g0 = neg_log_posterior_grad(theta_map, *args)
    for j in range(n_params):
        theta_pert = theta_map.copy()
        theta_pert[j] += eps
        g1 = neg_log_posterior_grad(theta_pert, *args)
        hessian[:, j] = (g1 - g0) / eps

    hessian = 0.5 * (hessian + hessian.T)

    try:
        cov = np.linalg.inv(hessian)
        post_std = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    except np.linalg.LinAlgError:
        print("  Warning: Hessian inversion failed, using diagonal approx")
        cov = np.diag(1.0 / np.maximum(np.diag(hessian), 1e-12))
        post_std = np.sqrt(np.diag(cov))

    return {
        "beta_map": beta_map,
        "gamma_map": gamma_map,
        "theta_map": theta_map,
        "cov": cov,
        "post_std": post_std,
        "beta_std": post_std[:p1],
        "gamma_std": post_std[p1:],
        "nll": result.fun,
    }


def posterior_samples(fit, n_samples=4000):
    rng = np.random.default_rng(42)
    samples = rng.multivariate_normal(fit["theta_map"], fit["cov"], size=n_samples)
    p1 = P + 1
    return samples[:, :p1], samples[:, p1:]


# -- MAP-only intervals (closed-form Laplace quantiles) -----------------------

def map_only_predict(fit, X_te):
    X_te_c = np.column_stack([np.ones(len(X_te)), X_te])
    mu_map = X_te_c @ fit["beta_map"]
    b_raw = X_te_c @ fit["gamma_map"]
    b_map = np.clip(b_raw, B_FLOOR, None)  # Hard clip at prediction time
    return mu_map, b_map


def map_only_coverage(y_te, mu_map, b_map, level):
    z = np.log(1.0 / (1.0 - level))
    lo = np.maximum(mu_map - z * b_map, 0.0)
    hi = mu_map + z * b_map
    covered = ((y_te >= lo) & (y_te <= hi)).mean()
    width = (hi - lo).mean()
    return covered, width


# -- Full posterior predictive intervals --------------------------------------

def posterior_predictive(fit, X_te, n_samples=4000):
    beta_samp, gamma_samp = posterior_samples(fit, n_samples)
    X_te_c = np.column_stack([np.ones(len(X_te)), X_te])

    mu_all = X_te_c @ beta_samp.T
    b_raw_all = X_te_c @ gamma_samp.T
    b_all = np.clip(b_raw_all, B_FLOOR, None)

    rng = np.random.default_rng(123)
    y_pred = np.zeros_like(mu_all)
    for j in range(mu_all.shape[1]):
        y_pred[:, j] = rng.laplace(mu_all[:, j], b_all[:, j])

    return mu_all, b_all, y_pred


def coverage_from_samples(y_te, y_pred_samples, level):
    lo_pct = (1.0 - level) / 2.0
    hi_pct = 1.0 - lo_pct
    lo = np.percentile(y_pred_samples, lo_pct * 100, axis=1)
    hi = np.percentile(y_pred_samples, hi_pct * 100, axis=1)
    lo = np.maximum(lo, 0.0)
    covered = ((y_te >= lo) & (y_te <= hi)).mean()
    width = (hi - lo).mean()
    return covered, width


# =============================================================================
# STEP 2: Load data
# =============================================================================
print("Loading data...", flush=True)
X_tr_aapl, y_tr_aapl, X_te_aapl, y_te_aapl, scaler_aapl = load_and_prep(
    "data/lit_buy_features_v2.parquet", "data/lit_buy_features_v2_sep.parquet")
X_tr_coin, y_tr_coin, X_te_coin, y_te_coin, scaler_coin = load_and_prep(
    "data/coin_lit_buy_features_train.parquet", "data/coin_lit_buy_features_test.parquet")

print(f"AAPL: train={len(y_tr_aapl):,}, test={len(y_te_aapl):,}")
print(f"COIN: train={len(y_tr_coin):,}, test={len(y_te_coin):,}")

# =============================================================================
# STEP 3: Fit Bayesian Laplace on AAPL
# =============================================================================
print("\n" + "=" * 80)
print("STEP 3: Fitting Bayesian Laplace on AAPL (MAP + Laplace approx)...")
print("=" * 80, flush=True)

fit_aapl = fit_bayesian_laplace(X_tr_aapl, y_tr_aapl, label="AAPL")

print(f"\nAAPL MAP estimates:")
print(f"  {'Coef':<22} {'beta (location)':>16} {'gamma (scale)':>14}")
print(f"  {'-'*22} {'-'*16} {'-'*14}")
for i, name in enumerate(COEF_NAMES):
    print(f"  {name:<22} {fit_aapl['beta_map'][i]:>+16.6f} {fit_aapl['gamma_map'][i]:>+14.6f}")

print(f"\nAAPL Posterior std:")
for i, name in enumerate(COEF_NAMES):
    print(f"  {name:<22}  beta_std={fit_aapl['beta_std'][i]:.6f}  gamma_std={fit_aapl['gamma_std'][i]:.6f}")

# =============================================================================
# STEP 4: Evaluate AAPL
# =============================================================================
print("\n" + "=" * 80)
print("STEP 4: Evaluating AAPL on test set...")
print("=" * 80, flush=True)

mu_map_aapl, b_map_aapl = map_only_predict(fit_aapl, X_te_aapl)
mae_aapl = np.mean(np.abs(y_te_aapl - mu_map_aapl))
print(f"AAPL Test MAE (MAP mu): {mae_aapl:.4f}")
print(f"AAPL b_map: mean={b_map_aapl.mean():.4f}, median={np.median(b_map_aapl):.4f}, "
      f"p95={np.percentile(b_map_aapl, 95):.4f}, max={b_map_aapl.max():.4f}")

print("\n  MAP-only intervals (closed-form Laplace quantiles):")
for level in [0.50, 0.80, 0.90]:
    cov, width = map_only_coverage(y_te_aapl, mu_map_aapl, b_map_aapl, level)
    print(f"    {level:.0%} coverage: {cov:.4f}, mean width: {width:.4f}")

mu_all_aapl, b_all_aapl, y_pred_aapl = posterior_predictive(fit_aapl, X_te_aapl)
print("\n  Full posterior predictive intervals:")
for level in [0.50, 0.80, 0.90]:
    cov, width = coverage_from_samples(y_te_aapl, y_pred_aapl, level)
    print(f"    {level:.0%} coverage: {cov:.4f}, mean width: {width:.4f}")

# =============================================================================
# STEP 5: Extract AAPL posterior for transfer
# =============================================================================
beta_mean_aapl = fit_aapl["beta_map"]
beta_std_aapl = np.maximum(fit_aapl["beta_std"], 0.05)
gamma_mean_aapl = fit_aapl["gamma_map"]
gamma_std_aapl = np.maximum(fit_aapl["gamma_std"], 0.05)

print("\nAAPL posterior for transfer:")
for i, name in enumerate(COEF_NAMES):
    print(f"  beta[{name}]: mean={beta_mean_aapl[i]:+.4f}, std={beta_std_aapl[i]:.4f}")
for i, name in enumerate(COEF_NAMES):
    print(f"  gamma[{name}]: mean={gamma_mean_aapl[i]:+.4f}, std={gamma_std_aapl[i]:.4f}")

# =============================================================================
# STEP 6: Fit COIN with uninformative priors
# =============================================================================
print("\n" + "=" * 80)
print("STEP 6: Fitting COIN with uninformative priors...")
print("=" * 80, flush=True)

fit_coin_uninf = fit_bayesian_laplace(X_tr_coin, y_tr_coin, label="COIN uninformative")

print(f"\nCOIN (uninformative) MAP estimates:")
for i, name in enumerate(COEF_NAMES):
    print(f"  {name:<22}  beta={fit_coin_uninf['beta_map'][i]:>+12.6f}  "
          f"gamma={fit_coin_uninf['gamma_map'][i]:>+12.6f}")

mu_map_coin_u, b_map_coin_u = map_only_predict(fit_coin_uninf, X_te_coin)
mae_coin_u = np.mean(np.abs(y_te_coin - mu_map_coin_u))
mu_all_coin_u, b_all_coin_u, y_pred_coin_u = posterior_predictive(fit_coin_uninf, X_te_coin)

print(f"\nCOIN (uninformative) Test MAE: {mae_coin_u:.4f}")
print(f"COIN b_map: mean={b_map_coin_u.mean():.4f}, median={np.median(b_map_coin_u):.4f}")
print("  MAP-only intervals:")
for level in [0.50, 0.80, 0.90]:
    cov, width = map_only_coverage(y_te_coin, mu_map_coin_u, b_map_coin_u, level)
    print(f"    {level:.0%} coverage: {cov:.4f}, mean width: {width:.4f}")
print("  Full posterior predictive intervals:")
for level in [0.50, 0.80, 0.90]:
    cov, width = coverage_from_samples(y_te_coin, y_pred_coin_u, level)
    print(f"    {level:.0%} coverage: {cov:.4f}, mean width: {width:.4f}")

# =============================================================================
# STEP 7: Fit COIN with AAPL-informed priors
# =============================================================================
print("\n" + "=" * 80)
print("STEP 7: Fitting COIN with AAPL-informed priors (hierarchical transfer)...")
print("=" * 80, flush=True)

fit_coin_hier = fit_bayesian_laplace(
    X_tr_coin, y_tr_coin,
    beta_prior_mu=beta_mean_aapl,
    beta_prior_sd=3.0 * beta_std_aapl,
    gamma_prior_mu=gamma_mean_aapl,
    gamma_prior_sd=3.0 * gamma_std_aapl,
    label="COIN hierarchical",
)

print(f"\nCOIN (hierarchical) MAP estimates:")
for i, name in enumerate(COEF_NAMES):
    print(f"  {name:<22}  beta={fit_coin_hier['beta_map'][i]:>+12.6f}  "
          f"gamma={fit_coin_hier['gamma_map'][i]:>+12.6f}")

mu_map_coin_h, b_map_coin_h = map_only_predict(fit_coin_hier, X_te_coin)
mae_coin_h = np.mean(np.abs(y_te_coin - mu_map_coin_h))
mu_all_coin_h, b_all_coin_h, y_pred_coin_h = posterior_predictive(fit_coin_hier, X_te_coin)

print(f"\nCOIN (hierarchical) Test MAE: {mae_coin_h:.4f}")
print(f"COIN b_map: mean={b_map_coin_h.mean():.4f}, median={np.median(b_map_coin_h):.4f}")
print("  MAP-only intervals:")
for level in [0.50, 0.80, 0.90]:
    cov, width = map_only_coverage(y_te_coin, mu_map_coin_h, b_map_coin_h, level)
    print(f"    {level:.0%} coverage: {cov:.4f}, mean width: {width:.4f}")
print("  Full posterior predictive intervals:")
for level in [0.50, 0.80, 0.90]:
    cov, width = coverage_from_samples(y_te_coin, y_pred_coin_h, level)
    print(f"    {level:.0%} coverage: {cov:.4f}, mean width: {width:.4f}")

# =============================================================================
# STEP 8: Comparison table
# =============================================================================
print("\n" + "=" * 80)
print("STEP 8: Test Set Comparison")
print("=" * 80)

csv_rows = []

all_results = [
    ("AAPL", "Bayesian", y_te_aapl, mu_map_aapl, b_map_aapl, y_pred_aapl, mae_aapl),
    ("COIN", "Bayesian (uninf.)", y_te_coin, mu_map_coin_u, b_map_coin_u, y_pred_coin_u, mae_coin_u),
    ("COIN", "Bayesian (hier.)", y_te_coin, mu_map_coin_h, b_map_coin_h, y_pred_coin_h, mae_coin_h),
]

print(f"\n{'Model':<24} {'Type':<12} {'90% Cov':>8} {'80% Cov':>8} {'50% Cov':>8} "
      f"{'MAE':>8} {'90% Width':>10}")
print("-" * 86)

for ticker, label, y_te, mu_map, b_map, y_pred, mae in all_results:
    # MAP-only row
    row_map = {"ticker": ticker, "model": label, "type": "MAP-only", "mae": round(mae, 4)}
    parts = [f"{label:<24}", "MAP-only    "]
    for level in [0.90, 0.80, 0.50]:
        cov, width = map_only_coverage(y_te, mu_map, b_map, level)
        row_map[f"cov_{int(level*100)}"] = round(cov, 4)
        row_map[f"width_{int(level*100)}"] = round(width, 4)
        parts.append(f"{cov:>8.4f}")
    parts.append(f"{mae:>8.4f}")
    parts.append(f"{row_map['width_90']:>10.4f}")
    print(" ".join(parts))
    csv_rows.append(row_map)

    # Full posterior predictive row
    row_pp = {"ticker": ticker, "model": label, "type": "Post. pred.", "mae": round(mae, 4)}
    parts = [f"{'':24}", "Post. pred. "]
    for level in [0.90, 0.80, 0.50]:
        cov, width = coverage_from_samples(y_te, y_pred, level)
        row_pp[f"cov_{int(level*100)}"] = round(cov, 4)
        row_pp[f"width_{int(level*100)}"] = round(width, 4)
        parts.append(f"{cov:>8.4f}")
    parts.append(f"{'':>8}")
    parts.append(f"{row_pp['width_90']:>10.4f}")
    print(" ".join(parts))
    csv_rows.append(row_pp)
    print()

csv_df = pd.DataFrame(csv_rows)
csv_df.to_csv("data/bayesian_results.csv", index=False)
print(f"Saved -> data/bayesian_results.csv")

# =============================================================================
# STEP 9: Figures
# =============================================================================
print("\nGenerating figures...", flush=True)

beta_samp_aapl, gamma_samp_aapl = posterior_samples(fit_aapl, 4000)
beta_samp_coin_u, gamma_samp_coin_u = posterior_samples(fit_coin_uninf, 4000)
beta_samp_coin_h, gamma_samp_coin_h = posterior_samples(fit_coin_hier, 4000)

# -- Figure 1: AAPL posterior distributions -----------------------------------
fig1 = plt.figure(figsize=(28, 8))
gs1 = gridspec.GridSpec(2, 7, figure=fig1, wspace=0.35, hspace=0.45)

for i, name in enumerate(COEF_NAMES):
    ax = fig1.add_subplot(gs1[0, i])
    ax.hist(beta_samp_aapl[:, i], bins=50, density=True, alpha=0.7, color="#2563eb",
            edgecolor="white", linewidth=0.3)
    x_range = np.linspace(beta_samp_aapl[:, i].min() - 1, beta_samp_aapl[:, i].max() + 1, 200)
    ax.plot(x_range, norm.pdf(x_range, 0, 5), color="black", lw=1.5, ls="--",
            alpha=0.6, label="Prior N(0,5)")
    ax.axvline(fit_aapl["beta_map"][i], color="#dc2626", lw=1.5, ls="-",
               alpha=0.8, label=f"MAP={fit_aapl['beta_map'][i]:.3f}")
    ax.set_title(f"beta[{name}]", fontsize=8, fontweight="bold")
    ax.set_xlabel("Value", fontsize=7)
    if i == 0:
        ax.set_ylabel("Density", fontsize=8)
    ax.legend(fontsize=5.5)
    ax.grid(True, alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = fig1.add_subplot(gs1[1, i])
    ax.hist(gamma_samp_aapl[:, i], bins=50, density=True, alpha=0.7, color="#16a34a",
            edgecolor="white", linewidth=0.3)
    x_range_g = np.linspace(gamma_samp_aapl[:, i].min() - 0.5, gamma_samp_aapl[:, i].max() + 0.5, 200)
    ax.plot(x_range_g, norm.pdf(x_range_g, 0, 5), color="black", lw=1.5, ls="--",
            alpha=0.6, label="Prior N(0,5)")
    ax.axvline(fit_aapl["gamma_map"][i], color="#dc2626", lw=1.5, ls="-",
               alpha=0.8, label=f"MAP={fit_aapl['gamma_map'][i]:.3f}")
    ax.set_title(f"gamma[{name}]", fontsize=8, fontweight="bold")
    ax.set_xlabel("Value", fontsize=7)
    if i == 0:
        ax.set_ylabel("Density", fontsize=8)
    ax.legend(fontsize=5.5)
    ax.grid(True, alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig1.suptitle("AAPL Bayesian Laplace GAMLSS: Posterior Distributions (Laplace Approximation)\n"
              "Top: location coefficients (beta)  |  Bottom: scale coefficients (gamma, linear link)",
              fontsize=13, fontweight="bold", y=1.03)
plt.savefig("aapl_bayesian_posteriors.png", dpi=150, bbox_inches="tight")
print("Saved -> aapl_bayesian_posteriors.png")

# -- Figure 2: COIN hierarchical comparison ----------------------------------
fig2 = plt.figure(figsize=(28, 8))
gs2 = gridspec.GridSpec(2, 7, figure=fig2, wspace=0.35, hspace=0.45)

for i, name in enumerate(COEF_NAMES):
    ax = fig2.add_subplot(gs2[0, i])
    ax.hist(beta_samp_aapl[:, i], bins=40, density=True, alpha=0.4, color="#2563eb",
            edgecolor="none", label="AAPL posterior")
    ax.hist(beta_samp_coin_u[:, i], bins=40, density=True, alpha=0.4, color="#dc2626",
            edgecolor="none", label="COIN uninform.")
    ax.hist(beta_samp_coin_h[:, i], bins=40, density=True, alpha=0.4, color="#16a34a",
            edgecolor="none", label="COIN hierarchical")
    ax.set_title(f"beta[{name}]", fontsize=8, fontweight="bold")
    ax.set_xlabel("Value", fontsize=7)
    if i == 0:
        ax.set_ylabel("Density", fontsize=8)
    if i == 6:
        ax.legend(fontsize=5.5)
    ax.grid(True, alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = fig2.add_subplot(gs2[1, i])
    ax.hist(gamma_samp_aapl[:, i], bins=40, density=True, alpha=0.4, color="#2563eb",
            edgecolor="none", label="AAPL posterior")
    ax.hist(gamma_samp_coin_u[:, i], bins=40, density=True, alpha=0.4, color="#dc2626",
            edgecolor="none", label="COIN uninform.")
    ax.hist(gamma_samp_coin_h[:, i], bins=40, density=True, alpha=0.4, color="#16a34a",
            edgecolor="none", label="COIN hierarchical")
    ax.set_title(f"gamma[{name}]", fontsize=8, fontweight="bold")
    ax.set_xlabel("Value", fontsize=7)
    if i == 0:
        ax.set_ylabel("Density", fontsize=8)
    if i == 6:
        ax.legend(fontsize=5.5)
    ax.grid(True, alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig2.suptitle("COIN Hierarchical Transfer: AAPL Posterior (blue) vs COIN Uninformative (red) "
              "vs COIN AAPL-Informed (green)\n"
              "Top: location coefficients (beta)  |  Bottom: scale coefficients (gamma, linear link)",
              fontsize=12, fontweight="bold", y=1.03)
plt.savefig("coin_hierarchical_comparison.png", dpi=150, bbox_inches="tight")
print("Saved -> coin_hierarchical_comparison.png")

# -- Figure 3: Calibration plots (2x3) — MAP-only (top) vs Post.Pred. (bottom)
fig3, axes = plt.subplots(2, 3, figsize=(18, 12))

cal_levels = np.linspace(0.05, 0.99, 50)

plot_data = [
    ("AAPL", y_te_aapl, mu_map_aapl, b_map_aapl, y_pred_aapl),
    ("COIN Uninformative", y_te_coin, mu_map_coin_u, b_map_coin_u, y_pred_coin_u),
    ("COIN Hierarchical", y_te_coin, mu_map_coin_h, b_map_coin_h, y_pred_coin_h),
]

for col, (label, y_te, mu_map, b_map, y_pred) in enumerate(plot_data):
    # Top row: MAP-only
    ax_top = axes[0, col]
    cal_map = [map_only_coverage(y_te, mu_map, b_map, lv)[0] for lv in cal_levels]
    ax_top.plot(cal_levels, cal_map, color="#2563eb", lw=2.2, marker="o", markersize=3,
                label="MAP-only")
    ax_top.plot([0, 1], [0, 1], color="black", lw=1.2, ls="--", alpha=0.6,
                label="Perfect calibration")
    for lv in [0.50, 0.80, 0.90]:
        cov, width = map_only_coverage(y_te, mu_map, b_map, lv)
        ax_top.annotate(f"{lv:.0%}: {cov:.1%}\nw={width:.2f}", xy=(lv, cov),
                        textcoords="offset points", xytext=(10, -14),
                        fontsize=7.5, color="#dc2626", fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color="#dc2626", lw=0.8))
    ax_top.set_xlabel("Nominal coverage level", fontsize=10)
    ax_top.set_ylabel("Actual coverage", fontsize=10)
    ax_top.set_title(f"{label}\nMAP-only intervals", fontsize=11, fontweight="bold")
    ax_top.legend(fontsize=8)
    ax_top.set_xlim(0, 1); ax_top.set_ylim(0, 1)
    ax_top.set_aspect("equal")
    ax_top.grid(True, alpha=0.15)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)

    # Bottom row: Full posterior predictive
    ax_bot = axes[1, col]
    cal_pp = [coverage_from_samples(y_te, y_pred, lv)[0] for lv in cal_levels]
    ax_bot.plot(cal_levels, cal_pp, color="#7c3aed", lw=2.2, marker="s", markersize=3,
                label="Posterior predictive")
    ax_bot.plot([0, 1], [0, 1], color="black", lw=1.2, ls="--", alpha=0.6,
                label="Perfect calibration")
    for lv in [0.50, 0.80, 0.90]:
        cov, width = coverage_from_samples(y_te, y_pred, lv)
        ax_bot.annotate(f"{lv:.0%}: {cov:.1%}\nw={width:.1f}", xy=(lv, cov),
                        textcoords="offset points", xytext=(10, -14),
                        fontsize=7.5, color="#dc2626", fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color="#dc2626", lw=0.8))
    ax_bot.set_xlabel("Nominal coverage level", fontsize=10)
    ax_bot.set_ylabel("Actual coverage", fontsize=10)
    ax_bot.set_title(f"{label}\nFull posterior predictive", fontsize=11, fontweight="bold")
    ax_bot.legend(fontsize=8)
    ax_bot.set_xlim(0, 1); ax_bot.set_ylim(0, 1)
    ax_bot.set_aspect("equal")
    ax_bot.grid(True, alpha=0.15)
    ax_bot.spines["top"].set_visible(False)
    ax_bot.spines["right"].set_visible(False)

fig3.suptitle("Bayesian Laplace GAMLSS: Calibration Comparison (linear scale link)\n"
              "Top: MAP-only intervals (no parameter uncertainty)  |  "
              "Bottom: Full posterior predictive (includes parameter uncertainty)",
              fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("bayesian_calibration.png", dpi=150, bbox_inches="tight")
print("Saved -> bayesian_calibration.png")

print("\nDone!")
