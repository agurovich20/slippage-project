"""
GAMLSS Laplace with Gamma GLM scale model.

Stage 1: Location = QuantReg LAD (tau=0.5)  — same as before
Stage 2: Scale = Gamma GLM with identity link on |residuals|
          This is the exact MLE for Laplace scale b, since
          |y - mu| ~ Exp(b) = Gamma(shape=1, scale=b).

Output:
  - calibration_gamma_overlay.png   (AAPL vs COIN, Gamma GLM GAMLSS)
  - calibration_ols_vs_gamma.png    (OLS vs Gamma GLM, per ticker)
  - data/gamlss_gamma_results.csv
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FEATURES = [
    "dollar_value", "log_dollar_value", "participation_rate",
    "roll_spread_500", "roll_vol_500", "exchange_id",
]

cal_levels = np.linspace(0.05, 0.99, 50)


def laplace_coverage(y, mu, b, level):
    z = np.log(1.0 / (1.0 - level))
    lo = np.maximum(mu - z * b, 0.0)
    hi = mu + z * b
    return ((y >= lo) & (y <= hi)).mean()


def fit_location(X_tr, y_tr, X_te):
    """QuantReg LAD (tau=0.5) for conditional median."""
    X_tr_c = sm.add_constant(X_tr)
    X_te_c = sm.add_constant(X_te)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        qr = sm.QuantReg(y_tr, X_tr_c).fit(q=0.5, max_iter=5000)
    return qr.predict(X_tr_c), qr.predict(X_te_c), qr


def fit_scale_ols(X_tr, abs_resid, X_te):
    """OLS on |residuals| (baseline)."""
    X_tr_c = np.column_stack([np.ones(len(X_tr)), X_tr])
    X_te_c = np.column_stack([np.ones(len(X_te)), X_te])
    gamma, _, _, _ = np.linalg.lstsq(X_tr_c, abs_resid, rcond=None)
    return np.clip(X_te_c @ gamma, 0.1, None)


def fit_scale_gamma_glm(X_tr, abs_resid, X_te):
    """Gamma GLM with identity link on |residuals| — exact Laplace scale MLE."""
    # Floor tiny residuals to avoid Gamma issues at zero
    abs_resid_safe = np.maximum(abs_resid, 1e-4)
    X_tr_c = sm.add_constant(X_tr)
    X_te_c = sm.add_constant(X_te)
    gamma_fam = sm.families.Gamma(link=sm.families.links.Identity())
    # Use OLS solution as starting values for stable convergence
    ols_beta, _, _, _ = np.linalg.lstsq(X_tr_c, abs_resid_safe, rcond=None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        glm = sm.GLM(abs_resid_safe, X_tr_c, family=gamma_fam)
        res = glm.fit(start_params=ols_beta, maxiter=500)
    b_te = np.clip(res.predict(X_te_c), 0.1, None)
    return b_te, res


# -- Load data ----------------------------------------------------------------
print("Loading data...", flush=True)

datasets = {
    "AAPL": ("data/lit_buy_features_v2.parquet", "data/lit_buy_features_v2_sep.parquet"),
    "COIN": ("data/coin_lit_buy_features_train.parquet", "data/coin_lit_buy_features_test.parquet"),
}

data = {}
for ticker, (tr_f, te_f) in datasets.items():
    df_tr = pd.read_parquet(tr_f)
    df_te = pd.read_parquet(te_f)
    df_tr["abs_impact"] = df_tr["impact_vwap_bps"].abs()
    df_te["abs_impact"] = df_te["impact_vwap_bps"].abs()
    df_tr = df_tr.sort_values("date").reset_index(drop=True)
    data[ticker] = {
        "X_tr": df_tr[FEATURES].to_numpy(dtype=np.float64),
        "y_tr": df_tr["abs_impact"].to_numpy(dtype=np.float64),
        "X_te": df_te[FEATURES].to_numpy(dtype=np.float64),
        "y_te": df_te["abs_impact"].to_numpy(dtype=np.float64),
    }

# -- Fit models ----------------------------------------------------------------
results = {}
csv_rows = []

for ticker in ["AAPL", "COIN"]:
    d = data[ticker]
    print(f"\n{'='*60}")
    print(f"  {ticker}")
    print(f"{'='*60}")

    # Stage 1: Location (shared across both scale methods)
    mu_tr, mu_te, qr_model = fit_location(d["X_tr"], d["y_tr"], d["X_te"])
    abs_resid = np.abs(d["y_tr"] - mu_tr)

    mae_te = np.mean(np.abs(d["y_te"] - mu_te))
    print(f"  Test MAE: {mae_te:.4f}")

    # Stage 2a: OLS scale (baseline)
    b_ols = fit_scale_ols(d["X_tr"], abs_resid, d["X_te"])

    # Stage 2b: Gamma GLM scale
    b_gamma, glm_res = fit_scale_gamma_glm(d["X_tr"], abs_resid, d["X_te"])

    print(f"\n  Gamma GLM converged: {glm_res.converged}")
    print(f"  Gamma GLM deviance:  {glm_res.deviance:.4f}")
    coef_names = ["intercept"] + FEATURES
    print(f"\n  {'Feature':<22} {'OLS coef':>12} {'Gamma coef':>12}")
    # Get OLS coefficients for comparison
    X_tr_c = np.column_stack([np.ones(len(d["X_tr"])), d["X_tr"]])
    ols_gamma, _, _, _ = np.linalg.lstsq(X_tr_c, abs_resid, rcond=None)
    for name, c_ols, c_glm in zip(coef_names, ols_gamma, glm_res.params):
        print(f"  {name:<22} {c_ols:>+12.6f} {c_glm:>+12.6f}")

    print(f"\n  Test b_hat (OLS):   mean={b_ols.mean():.4f}, median={np.median(b_ols):.4f}")
    print(f"  Test b_hat (Gamma): mean={b_gamma.mean():.4f}, median={np.median(b_gamma):.4f}")

    # Calibration curves
    cal_ols = [laplace_coverage(d["y_te"], mu_te, b_ols, lv) for lv in cal_levels]
    cal_gamma = [laplace_coverage(d["y_te"], mu_te, b_gamma, lv) for lv in cal_levels]

    # Key level coverage
    print(f"\n  {'Level':>8} {'OLS cov':>10} {'OLS width':>12} {'Gamma cov':>12} {'Gamma width':>14}")
    for lv in [0.50, 0.80, 0.90]:
        cov_ols = laplace_coverage(d["y_te"], mu_te, b_ols, lv)
        cov_gam = laplace_coverage(d["y_te"], mu_te, b_gamma, lv)
        z = np.log(1.0 / (1.0 - lv))
        w_ols = (2 * z * b_ols).mean()
        w_gam = (2 * z * b_gamma).mean()
        print(f"  {lv:>8.0%} {cov_ols:>10.4f} {w_ols:>12.4f} {cov_gam:>12.4f} {w_gam:>14.4f}")
        csv_rows.append({
            "ticker": ticker, "model": "Gamma_GLM_GAMLSS",
            "level": lv, "nominal_coverage": lv,
            "actual_coverage": round(cov_gam, 6),
            "mean_interval_width": round(w_gam, 4),
            "test_mae": round(mae_te, 4),
            "n_train": len(d["y_tr"]), "n_test": len(d["y_te"]),
        })

    results[ticker] = {
        "mu_te": mu_te, "y_te": d["y_te"],
        "b_ols": b_ols, "b_gamma": b_gamma,
        "cal_ols": np.array(cal_ols), "cal_gamma": np.array(cal_gamma),
    }

# Save CSV
pd.DataFrame(csv_rows).to_csv("data/gamlss_gamma_results.csv", index=False)
print("\nSaved -> data/gamlss_gamma_results.csv")

# -- Colors -------------------------------------------------------------------
AAPL_COLOR = "#2563eb"
COIN_COLOR = "#dc2626"

# -- Figure 1: Gamma GLM GAMLSS calibration overlay (AAPL vs COIN) -----------
print("\nPlotting Gamma GLM calibration overlay...", flush=True)

fig1, ax1 = plt.subplots(figsize=(8, 8))

ax1.plot(cal_levels, results["AAPL"]["cal_gamma"], color=AAPL_COLOR, lw=2.5,
         marker="o", markersize=4, label="AAPL (9,152 test trades)")
ax1.plot(cal_levels, results["COIN"]["cal_gamma"], color=COIN_COLOR, lw=2.5,
         marker="s", markersize=4, label="COIN (959 test trades)")
ax1.plot([0, 1], [0, 1], color="black", lw=1.2, ls="--", alpha=0.5,
         label="Perfect calibration")

for ticker, color, yoff in [("AAPL", AAPL_COLOR, 12), ("COIN", COIN_COLOR, -16)]:
    r = results[ticker]
    for lv in [0.50, 0.80, 0.90]:
        cov = laplace_coverage(r["y_te"], r["mu_te"], r["b_gamma"], lv)
        z = np.log(1.0 / (1.0 - lv))
        w = (2 * z * r["b_gamma"]).mean()
        ax1.annotate(f"{lv:.0%}: {cov:.1%} (w={w:.1f})", xy=(lv, cov),
                     textcoords="offset points", xytext=(12, yoff),
                     fontsize=8, color=color, fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color=color, lw=0.8))

ax1.set_xlabel("Nominal coverage level", fontsize=12)
ax1.set_ylabel("Actual coverage", fontsize=12)
ax1.set_title("Gamma GLM GAMLSS Calibration: AAPL vs COIN\n"
              "Location = QuantReg (LAD)  |  Scale = Gamma GLM (identity link)",
              fontsize=13, fontweight="bold")
ax1.legend(fontsize=10, loc="upper left")
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_aspect("equal")
ax1.grid(True, alpha=0.15)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("calibration_gamma_overlay.png", dpi=150, bbox_inches="tight")
print("Saved -> calibration_gamma_overlay.png")

# -- Figure 2: OLS vs Gamma GLM comparison (side by side per ticker) ----------
print("Plotting OLS vs Gamma GLM comparison...", flush=True)

fig2, axes = plt.subplots(1, 2, figsize=(16, 8))

for ax, ticker, color in zip(axes, ["AAPL", "COIN"], [AAPL_COLOR, COIN_COLOR]):
    r = results[ticker]
    n_test = len(r["y_te"])

    ax.plot(cal_levels, r["cal_ols"], color="#6b7280", lw=2.2,
            marker="^", markersize=4, label="OLS scale", alpha=0.85)
    ax.plot(cal_levels, r["cal_gamma"], color=color, lw=2.5,
            marker="o", markersize=4, label="Gamma GLM scale")
    ax.plot([0, 1], [0, 1], color="black", lw=1.2, ls="--", alpha=0.5,
            label="Perfect calibration")

    # Annotate differences at key levels
    for lv in [0.50, 0.80, 0.90]:
        cov_ols = laplace_coverage(r["y_te"], r["mu_te"], r["b_ols"], lv)
        cov_gam = laplace_coverage(r["y_te"], r["mu_te"], r["b_gamma"], lv)
        z = np.log(1.0 / (1.0 - lv))
        w_ols = (2 * z * r["b_ols"]).mean()
        w_gam = (2 * z * r["b_gamma"]).mean()
        ax.annotate(f"OLS: {cov_ols:.1%} (w={w_ols:.1f})", xy=(lv, cov_ols),
                    textcoords="offset points", xytext=(-80, 18),
                    fontsize=7.5, color="#6b7280", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#6b7280", lw=0.7))
        ax.annotate(f"Gamma: {cov_gam:.1%} (w={w_gam:.1f})", xy=(lv, cov_gam),
                    textcoords="offset points", xytext=(12, -14),
                    fontsize=7.5, color=color, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=color, lw=0.7))

    ax.set_xlabel("Nominal coverage level", fontsize=12)
    ax.set_ylabel("Actual coverage", fontsize=12)
    ax.set_title(f"{ticker} ({n_test:,} test trades)\n"
                 f"OLS vs Gamma GLM Scale Model",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig2.suptitle("Scale Model Comparison: OLS vs Gamma GLM (Identity Link)\n"
              "Both use QuantReg LAD location  |  Laplace prediction intervals",
              fontsize=14, fontweight="bold", y=1.04)

plt.tight_layout()
plt.savefig("calibration_ols_vs_gamma.png", dpi=150, bbox_inches="tight")
print("Saved -> calibration_ols_vs_gamma.png")

print("\nDone!")
