"""
Two calibration overlay plots:
  1. Linear GAMLSS: AAPL vs COIN on same axes
  2. XGB GAMLSS: AAPL vs COIN on same axes

Output:
  - calibration_linear_overlay.png
  - calibration_xgb_overlay.png
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from xgboost import XGBRegressor
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


def fit_linear_gamlss(X_tr, y_tr, X_te):
    X_tr_c = sm.add_constant(X_tr)
    X_te_c = sm.add_constant(X_te)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        qr = sm.QuantReg(y_tr, X_tr_c).fit(q=0.5, max_iter=5000)
    mu_tr = qr.predict(X_tr_c)
    mu_te = qr.predict(X_te_c)
    abs_r = np.abs(y_tr - mu_tr)
    Xg_tr = np.column_stack([np.ones(len(X_tr)), X_tr])
    Xg_te = np.column_stack([np.ones(len(X_te)), X_te])
    gamma, _, _, _ = np.linalg.lstsq(Xg_tr, abs_r, rcond=None)
    b_te = np.clip(Xg_te @ gamma, 0.1, None)
    return mu_te, b_te


def fit_xgb_gamlss(X_tr, y_tr, X_te):
    loc = XGBRegressor(objective="reg:absoluteerror", tree_method="hist", verbosity=0,
                       random_state=42, n_jobs=1, max_depth=3, n_estimators=200,
                       learning_rate=0.07, min_child_weight=5, reg_alpha=10, reg_lambda=10)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loc.fit(X_tr, y_tr)
    mu_tr = np.maximum(loc.predict(X_tr), 0.0)
    mu_te = np.maximum(loc.predict(X_te), 0.0)
    abs_r = np.abs(y_tr - mu_tr)
    sc = XGBRegressor(objective="reg:squarederror", tree_method="hist", verbosity=0,
                      random_state=42, n_jobs=1, max_depth=5, n_estimators=50,
                      min_child_weight=20, learning_rate=0.1, reg_alpha=1, reg_lambda=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc.fit(X_tr, abs_r)
    b_te = np.clip(sc.predict(X_te), 0.1, None)
    return mu_te, b_te


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

# -- Fit models and compute calibration curves --------------------------------
results = {}

for ticker in ["AAPL", "COIN"]:
    d = data[ticker]
    print(f"Fitting {ticker}...", flush=True)

    mu_lin, b_lin = fit_linear_gamlss(d["X_tr"], d["y_tr"], d["X_te"])
    mu_xgb, b_xgb = fit_xgb_gamlss(d["X_tr"], d["y_tr"], d["X_te"])

    cal_lin = [laplace_coverage(d["y_te"], mu_lin, b_lin, lv) for lv in cal_levels]
    cal_xgb = [laplace_coverage(d["y_te"], mu_xgb, b_xgb, lv) for lv in cal_levels]

    results[ticker] = {
        "cal_lin": np.array(cal_lin),
        "cal_xgb": np.array(cal_xgb),
        "mu_lin": mu_lin, "b_lin": b_lin,
        "mu_xgb": mu_xgb, "b_xgb": b_xgb,
        "y_te": d["y_te"],
    }

AAPL_COLOR = "#2563eb"
COIN_COLOR = "#dc2626"

# -- Figure 1: Linear GAMLSS — AAPL vs COIN ----------------------------------
print("Plotting Linear GAMLSS overlay...", flush=True)

fig1, ax1 = plt.subplots(figsize=(8, 8))

ax1.plot(cal_levels, results["AAPL"]["cal_lin"], color=AAPL_COLOR, lw=2.5,
         marker="o", markersize=4, label="AAPL (9,152 test trades)")
ax1.plot(cal_levels, results["COIN"]["cal_lin"], color=COIN_COLOR, lw=2.5,
         marker="s", markersize=4, label="COIN (959 test trades)")
ax1.plot([0, 1], [0, 1], color="black", lw=1.2, ls="--", alpha=0.5,
         label="Perfect calibration")

for ticker, color, yoff in [("AAPL", AAPL_COLOR, 12), ("COIN", COIN_COLOR, -16)]:
    r = results[ticker]
    for lv in [0.50, 0.80, 0.90]:
        cov = laplace_coverage(r["y_te"], r["mu_lin"], r["b_lin"], lv)
        _, width = lv, None  # just need cov
        z = np.log(1.0 / (1.0 - lv))
        w = (2 * z * r["b_lin"]).mean()
        ax1.annotate(f"{lv:.0%}: {cov:.1%} (w={w:.1f})", xy=(lv, cov),
                     textcoords="offset points", xytext=(12, yoff),
                     fontsize=8, color=color, fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color=color, lw=0.8))

ax1.set_xlabel("Nominal coverage level", fontsize=12)
ax1.set_ylabel("Actual coverage", fontsize=12)
ax1.set_title("Two-Stage Linear Regression Calibration: AAPL and COIN",
              fontsize=13, fontweight="bold")
ax1.legend(fontsize=10, loc="upper left")
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_aspect("equal")
ax1.grid(True, alpha=0.15)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("calibration_linear_overlay.png", dpi=150, bbox_inches="tight")
print("Saved -> calibration_linear_overlay.png")

# -- Figure 2: XGB GAMLSS — AAPL vs COIN -------------------------------------
print("Plotting XGB GAMLSS overlay...", flush=True)

fig2, ax2 = plt.subplots(figsize=(8, 8))

ax2.plot(cal_levels, results["AAPL"]["cal_xgb"], color=AAPL_COLOR, lw=2.5,
         marker="o", markersize=4, label="AAPL (9,152 test trades)")
ax2.plot(cal_levels, results["COIN"]["cal_xgb"], color=COIN_COLOR, lw=2.5,
         marker="s", markersize=4, label="COIN (959 test trades)")
ax2.plot([0, 1], [0, 1], color="black", lw=1.2, ls="--", alpha=0.5,
         label="Perfect calibration")

for ticker, color, yoff in [("AAPL", AAPL_COLOR, 12), ("COIN", COIN_COLOR, -16)]:
    r = results[ticker]
    for lv in [0.50, 0.80, 0.90]:
        cov = laplace_coverage(r["y_te"], r["mu_xgb"], r["b_xgb"], lv)
        z = np.log(1.0 / (1.0 - lv))
        w = (2 * z * r["b_xgb"]).mean()
        ax2.annotate(f"{lv:.0%}: {cov:.1%} (w={w:.1f})", xy=(lv, cov),
                     textcoords="offset points", xytext=(12, yoff),
                     fontsize=8, color=color, fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color=color, lw=0.8))

ax2.set_xlabel("Nominal coverage level", fontsize=12)
ax2.set_ylabel("Actual coverage", fontsize=12)
ax2.set_title("Two-Stage XGBoost Calibration: AAPL and COIN",
              fontsize=13, fontweight="bold")
ax2.legend(fontsize=10, loc="upper left")
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_aspect("equal")
ax2.grid(True, alpha=0.15)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("calibration_xgb_overlay.png", dpi=150, bbox_inches="tight")
print("Saved -> calibration_xgb_overlay.png")

print("\nDone!")
