"""
Standalone prediction interval visualizations.

Output:
  - pred_intervals_fan.png       2x2 fan charts (50/80/90%) sorted by predicted median
  - pred_intervals_by_size.png   2x2 intervals binned by dollar_value
  - pred_intervals_linear_vs_xgb.png  AAPL and COIN: Linear vs XGB side by side
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
        "dollar_te": df_te["dollar_value"].to_numpy(dtype=np.float64),
    }

# -- Fit all models ------------------------------------------------------------
print("Fitting models...", flush=True)
results = {}
for ticker in ["AAPL", "COIN"]:
    d = data[ticker]
    print(f"  {ticker}...", flush=True)
    mu_lin, b_lin = fit_linear_gamlss(d["X_tr"], d["y_tr"], d["X_te"])
    mu_xgb, b_xgb = fit_xgb_gamlss(d["X_tr"], d["y_tr"], d["X_te"])
    results[ticker] = {
        "y_te": d["y_te"], "dollar_te": d["dollar_te"],
        "mu_lin": mu_lin, "b_lin": b_lin,
        "mu_xgb": mu_xgb, "b_xgb": b_xgb,
    }

rng = np.random.default_rng(42)
AAPL_COLOR = "#2563eb"
COIN_COLOR = "#dc2626"
BAND_COLORS = {"90%": "#bfdbfe", "80%": "#93c5fd", "50%": "#60a5fa"}
BAND_COLORS_COIN = {"90%": "#fecaca", "80%": "#fca5a5", "50%": "#f87171"}

# =============================================================================
# Figure 1: Fan charts — 50/80/90% intervals sorted by predicted median
# =============================================================================
print("Plotting fan charts...", flush=True)

fig1, axes1 = plt.subplots(2, 2, figsize=(20, 14))

for col, model_key, model_name in [(0, "lin", "Two-Stage Linear Regression"), (1, "xgb", "Two-Stage XGBoost")]:
    for row, ticker in [(0, "AAPL"), (1, "COIN")]:
        ax = axes1[row, col]
        r = results[ticker]
        mu = r[f"mu_{model_key}"]
        b = r[f"b_{model_key}"]
        y = r["y_te"]

        n_show = min(300, len(y))
        idx = rng.choice(len(y), size=n_show, replace=False)
        order = np.argsort(mu[idx])
        idx_sorted = idx[order]

        x = np.arange(n_show)
        mu_s = mu[idx_sorted]
        b_s = b[idx_sorted]
        y_s = y[idx_sorted]

        colors = BAND_COLORS if ticker == "AAPL" else BAND_COLORS_COIN
        base_color = AAPL_COLOR if ticker == "AAPL" else COIN_COLOR

        for level, label in [(0.90, "90%"), (0.80, "80%"), (0.50, "50%")]:
            z = np.log(1.0 / (1.0 - level))
            lo = np.maximum(mu_s - z * b_s, 0.0)
            hi = mu_s + z * b_s
            ax.fill_between(x, lo, hi, alpha=0.7, color=colors[label], label=f"{label} interval")

        ax.plot(x, mu_s, color=base_color, lw=1.5, label="Predicted median", zorder=3)
        ax.scatter(x, y_s, s=10, color="black", alpha=0.45, zorder=4, label="Actual")

        ax.set_xlabel("Trade index (sorted by predicted median)", fontsize=10)
        ax.set_ylabel("|slippage| (bps)", fontsize=10)
        ax.set_title(f"{ticker} {model_name} (300 random test trades)",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="upper left", ncol=2)
        ax.grid(True, alpha=0.12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

fig1.suptitle("Prediction Interval Fan Charts: 50% / 80% / 90% Laplace Intervals",
              fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("pred_intervals_fan.png", dpi=150, bbox_inches="tight")
print("Saved -> pred_intervals_fan.png")

# =============================================================================
# Figure 2: Intervals binned by dollar value (trade size)
# =============================================================================
print("Plotting intervals by trade size...", flush=True)

fig2, axes2 = plt.subplots(2, 2, figsize=(20, 14))

for col, model_key, model_name in [(0, "lin", "Two-Stage Linear Regression"), (1, "xgb", "Two-Stage XGBoost")]:
    for row, ticker in [(0, "AAPL"), (1, "COIN")]:
        ax = axes2[row, col]
        r = results[ticker]
        mu = r[f"mu_{model_key}"]
        b = r[f"b_{model_key}"]
        y = r["y_te"]
        dv = r["dollar_te"]

        base_color = AAPL_COLOR if ticker == "AAPL" else COIN_COLOR

        n_bins = 15
        bin_edges = np.percentile(dv, np.linspace(0, 100, n_bins + 1))
        bin_edges = np.unique(bin_edges)
        actual_bins = len(bin_edges) - 1

        bin_centers = []
        bin_mu = []
        bin_y_mean = []
        bin_y_med = []
        bin_lo90 = []
        bin_hi90 = []
        bin_lo80 = []
        bin_hi80 = []
        bin_lo50 = []
        bin_hi50 = []

        for i in range(actual_bins):
            mask = (dv >= bin_edges[i]) & (dv < bin_edges[i + 1])
            if i == actual_bins - 1:
                mask = (dv >= bin_edges[i]) & (dv <= bin_edges[i + 1])
            if mask.sum() < 5:
                continue
            bin_centers.append(dv[mask].mean())
            bin_mu.append(mu[mask].mean())
            bin_y_mean.append(y[mask].mean())
            bin_y_med.append(np.median(y[mask]))
            for level, lo_list, hi_list in [
                (0.90, bin_lo90, bin_hi90),
                (0.80, bin_lo80, bin_hi80),
                (0.50, bin_lo50, bin_hi50),
            ]:
                z = np.log(1.0 / (1.0 - level))
                lo_list.append(np.maximum(mu[mask] - z * b[mask], 0).mean())
                hi_list.append((mu[mask] + z * b[mask]).mean())

        bc = np.array(bin_centers)
        colors_set = BAND_COLORS if ticker == "AAPL" else BAND_COLORS_COIN

        ax.fill_between(bc, bin_lo90, bin_hi90, alpha=0.6, color=colors_set["90%"], label="90% interval")
        ax.fill_between(bc, bin_lo80, bin_hi80, alpha=0.7, color=colors_set["80%"], label="80% interval")
        ax.fill_between(bc, bin_lo50, bin_hi50, alpha=0.8, color=colors_set["50%"], label="50% interval")
        ax.plot(bc, bin_mu, color=base_color, lw=2.2, marker="o", markersize=5,
                label="Predicted median", zorder=3)
        ax.scatter(bc, bin_y_med, s=40, color="black", marker="x", linewidths=1.5,
                   zorder=5, label="Actual median")

        ax.set_xlabel("Dollar value (trade size)", fontsize=10)
        ax.set_ylabel("|slippage| (bps)", fontsize=10)
        ax.set_title(f"{ticker} {model_name}\nPrediction Intervals by Trade Size (15 bins)",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

fig2.suptitle("Prediction Intervals by Trade Size: How Uncertainty Scales with Dollar Value",
              fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("pred_intervals_by_size.png", dpi=150, bbox_inches="tight")
print("Saved -> pred_intervals_by_size.png")

# =============================================================================
# Figure 3: Linear vs XGB side by side (direct comparison)
# =============================================================================
print("Plotting Linear vs XGB comparison...", flush=True)

fig3, axes3 = plt.subplots(1, 2, figsize=(20, 8))

for ax, ticker in zip(axes3, ["AAPL", "COIN"]):
    r = results[ticker]
    y = r["y_te"]
    base_color = AAPL_COLOR if ticker == "AAPL" else COIN_COLOR

    n_show = 250
    idx = rng.choice(len(y), size=min(n_show, len(y)), replace=False)
    # Sort by XGB predicted median for consistent ordering
    order = np.argsort(r["mu_xgb"][idx])
    idx_sorted = idx[order]
    x = np.arange(len(idx_sorted))

    z90 = np.log(10.0)

    # Linear intervals (background, gray)
    mu_lin_s = r["mu_lin"][idx_sorted]
    b_lin_s = r["b_lin"][idx_sorted]
    lo_lin = np.maximum(mu_lin_s - z90 * b_lin_s, 0.0)
    hi_lin = mu_lin_s + z90 * b_lin_s

    # XGB intervals (foreground, colored)
    mu_xgb_s = r["mu_xgb"][idx_sorted]
    b_xgb_s = r["b_xgb"][idx_sorted]
    lo_xgb = np.maximum(mu_xgb_s - z90 * b_xgb_s, 0.0)
    hi_xgb = mu_xgb_s + z90 * b_xgb_s

    y_s = y[idx_sorted]

    ax.fill_between(x, lo_lin, hi_lin, alpha=0.2, color="#9ca3af",
                    label=f"Linear 90% (mean w={hi_lin.mean()-lo_lin.mean():.1f})")
    ax.fill_between(x, lo_xgb, hi_xgb, alpha=0.3, color=base_color,
                    label=f"XGB 90% (mean w={hi_xgb.mean()-lo_xgb.mean():.1f})")
    ax.plot(x, mu_lin_s, color="#6b7280", lw=1.0, ls="--", alpha=0.7, label="Linear median")
    ax.plot(x, mu_xgb_s, color=base_color, lw=1.5, label="XGB median", zorder=3)
    ax.scatter(x, y_s, s=10, color="black", alpha=0.4, zorder=4, label="Actual")

    ax.set_xlabel("Trade index (sorted by XGB predicted median)", fontsize=10)
    ax.set_ylabel("|slippage| (bps)", fontsize=10)
    ax.set_title(f"{ticker} Linear vs Two-Stage XGBoost (90% intervals, 250 random test trades)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8.5, loc="upper left")
    ax.grid(True, alpha=0.12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig3.suptitle("Linear vs Two-Stage XGBoost: 90% Prediction Interval Comparison",
              fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("pred_intervals_linear_vs_xgb.png", dpi=150, bbox_inches="tight")
print("Saved -> pred_intervals_linear_vs_xgb.png")

print("\nDone!")
