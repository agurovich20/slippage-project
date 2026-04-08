"""
Calibration and coverage plots for the pooled XGB GAMLSS (6 stocks).

Output:
  - calibration_pooled_6stocks.png      All 6 stocks on one calibration curve
  - coverage_deviation_pooled.png       Coverage deviation (actual - nominal)
  - calibration_pooled_vs_perstock.png  Side-by-side: per-stock vs pooled calibration
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import warnings
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FEATURES_3 = ["roll_spread_500", "roll_vol_500", "participation_rate"]
IMPACT_CAP_BPS = 500
cal_levels = np.linspace(0.05, 0.99, 50)

DATASETS = {
    "AAPL": ("data/lit_buy_features_v2.parquet", "data/lit_buy_features_v2_sep.parquet"),
    "COIN": ("data/coin_lit_buy_features_train.parquet", "data/coin_lit_buy_features_test.parquet"),
    "NVDA": ("data/nvda_lit_buy_features_train.parquet", "data/nvda_lit_buy_features_test.parquet"),
    "AMD":  ("data/amd_lit_buy_features_train.parquet", "data/amd_lit_buy_features_test.parquet"),
    "AMZN": ("data/amzn_lit_buy_features_train.parquet", "data/amzn_lit_buy_features_test.parquet"),
    "TSLA": ("data/tsla_lit_buy_features_train.parquet", "data/tsla_lit_buy_features_test.parquet"),
}

COLORS = {
    "AAPL": "#2563eb", "COIN": "#dc2626", "NVDA": "#16a34a",
    "AMD": "#f59e0b", "AMZN": "#7c3aed", "TSLA": "#ec4899",
}
MARKERS = {
    "AAPL": "o", "COIN": "s", "NVDA": "D", "AMD": "^", "AMZN": "v", "TSLA": "P",
}

LOC_PARAMS = dict(max_depth=3, n_estimators=200, learning_rate=0.07,
                  min_child_weight=5, reg_alpha=10, reg_lambda=10)
SCALE_PARAMS = dict(max_depth=3, n_estimators=50, learning_rate=0.1,
                    min_child_weight=20, reg_alpha=1, reg_lambda=1)


def laplace_coverage(y, mu, b, level):
    z = np.log(1.0 / (1.0 - level))
    lo = np.maximum(mu - z * b, 0.0)
    hi = mu + z * b
    return ((y >= lo) & (y <= hi)).mean()


# ═════════════════════════════════════════════════════════════════════════════
# Load data and build pooled training set
# ═════════════════════════════════════════════════════════════════════════════
print("Loading data and normalizing...", flush=True)

stock_data = {}
pooled_X, pooled_y = [], []

for ticker, (tr_f, te_f) in DATASETS.items():
    df_tr = pd.read_parquet(tr_f)
    df_te = pd.read_parquet(te_f)
    df_tr = df_tr[df_tr["impact_vwap_bps"].abs() <= IMPACT_CAP_BPS]
    df_te = df_te[df_te["impact_vwap_bps"].abs() <= IMPACT_CAP_BPS]
    df_tr["abs_impact"] = df_tr["impact_vwap_bps"].abs()
    df_te["abs_impact"] = df_te["impact_vwap_bps"].abs()

    medians = {}
    for feat in FEATURES_3:
        medians[feat] = df_tr[feat].median()
        if medians[feat] == 0 or np.isnan(medians[feat]):
            medians[feat] = df_tr[feat].mean()
            if medians[feat] == 0:
                medians[feat] = 1.0

    median_impact = df_tr["abs_impact"].median()
    if median_impact == 0 or np.isnan(median_impact):
        median_impact = df_tr["abs_impact"].mean()

    X_tr_norm = np.column_stack([df_tr[feat].values / medians[feat] for feat in FEATURES_3])
    y_tr_norm = df_tr["abs_impact"].values / median_impact
    X_te_norm = np.column_stack([df_te[feat].values / medians[feat] for feat in FEATURES_3])

    stock_data[ticker] = {
        "medians": medians, "median_impact": median_impact,
        "X_tr_raw": df_tr[FEATURES_3].to_numpy(dtype=np.float64),
        "y_tr_raw": df_tr["abs_impact"].to_numpy(dtype=np.float64),
        "X_te_raw": df_te[FEATURES_3].to_numpy(dtype=np.float64),
        "y_te_raw": df_te["abs_impact"].to_numpy(dtype=np.float64),
        "X_tr_norm": X_tr_norm, "y_tr_norm": y_tr_norm,
        "X_te_norm": X_te_norm,
        "n_test": len(df_te),
    }
    pooled_X.append(X_tr_norm)
    pooled_y.append(y_tr_norm)

pooled_X = np.vstack(pooled_X)
pooled_y = np.concatenate(pooled_y)
print(f"  Pooled training: {len(pooled_y):,} trades")


# ═════════════════════════════════════════════════════════════════════════════
# Fit pooled model
# ═════════════════════════════════════════════════════════════════════════════
print("Fitting pooled location + scale models...", flush=True)

loc_model = XGBRegressor(objective="reg:absoluteerror", tree_method="hist",
                         verbosity=0, random_state=42, n_jobs=1, **LOC_PARAMS)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    loc_model.fit(pooled_X, pooled_y)

mu_pooled_tr = np.maximum(loc_model.predict(pooled_X), 0.0)
abs_resid = np.abs(pooled_y - mu_pooled_tr)

scale_model = XGBRegressor(objective="reg:squarederror", tree_method="hist",
                           verbosity=0, random_state=42, n_jobs=1, **SCALE_PARAMS)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    scale_model.fit(pooled_X, abs_resid)


# ═════════════════════════════════════════════════════════════════════════════
# Fit per-stock models and compute calibration for both
# ═════════════════════════════════════════════════════════════════════════════
print("Computing calibration curves...", flush=True)

results_pooled = {}
results_perstock = {}

for ticker in DATASETS:
    d = stock_data[ticker]
    y_te = d["y_te_raw"]
    scale = d["median_impact"]

    # --- Pooled predictions ---
    mu_pool_norm = np.maximum(loc_model.predict(d["X_te_norm"]), 0.0)
    b_pool_norm = np.clip(scale_model.predict(d["X_te_norm"]), 0.01, None)
    mu_pool = mu_pool_norm * scale
    b_pool = np.clip(b_pool_norm * scale, 0.1, None)

    cal_pool = np.array([laplace_coverage(y_te, mu_pool, b_pool, lv) for lv in cal_levels])
    results_pooled[ticker] = {
        "cal": cal_pool, "mu_te": mu_pool, "b_te": b_pool,
        "y_te": y_te, "n_test": len(y_te),
    }

    # --- Per-stock predictions ---
    loc_ps = XGBRegressor(objective="reg:absoluteerror", tree_method="hist",
                          verbosity=0, random_state=42, n_jobs=1, **LOC_PARAMS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loc_ps.fit(d["X_tr_raw"], d["y_tr_raw"])
    mu_tr_ps = np.maximum(loc_ps.predict(d["X_tr_raw"]), 0.0)
    mu_te_ps = np.maximum(loc_ps.predict(d["X_te_raw"]), 0.0)

    abs_r_ps = np.abs(d["y_tr_raw"] - mu_tr_ps)
    sc_ps = XGBRegressor(objective="reg:squarederror", tree_method="hist",
                         verbosity=0, random_state=42, n_jobs=1, **SCALE_PARAMS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc_ps.fit(d["X_tr_raw"], abs_r_ps)
    b_te_ps = np.clip(sc_ps.predict(d["X_te_raw"]), 0.1, None)

    cal_ps = np.array([laplace_coverage(y_te, mu_te_ps, b_te_ps, lv) for lv in cal_levels])
    results_perstock[ticker] = {
        "cal": cal_ps, "mu_te": mu_te_ps, "b_te": b_te_ps,
        "y_te": y_te, "n_test": len(y_te),
    }

    print(f"  {ticker}: 90% pooled={cal_pool[cal_levels >= 0.89][0]:.3f}, "
          f"per-stock={cal_ps[cal_levels >= 0.89][0]:.3f}")

TICKERS = list(DATASETS.keys())


# ═════════════════════════════════════════════════════════════════════════════
# Figure 1: Pooled calibration — all 6 stocks
# ═════════════════════════════════════════════════════════════════════════════
print("\nPlotting pooled calibration curve...", flush=True)

fig1, ax1 = plt.subplots(figsize=(9, 9))


for ticker in TICKERS:
    r = results_pooled[ticker]
    ax1.plot(cal_levels, r["cal"], color=COLORS[ticker], lw=2.5,
             marker=MARKERS[ticker], markersize=4, markevery=3,
             label=f"{ticker} ({r['n_test']:,} test trades)")

ax1.plot([0, 1], [0, 1], color="black", lw=1.2, ls="--", alpha=0.5,
         label="Perfect calibration")

for lv in [0.50, 0.80, 0.90]:
    ax1.axvline(lv, color="gray", lw=0.8, ls=":", alpha=0.4, zorder=1)
    ax1.annotate(f"{lv:.0%}", xy=(lv, lv), textcoords="offset points",
                 xytext=(-14, -10), fontsize=9, color="black", fontweight="bold",
                 alpha=0.6)
    for ticker in TICKERS:
        r = results_pooled[ticker]
        cov = laplace_coverage(r["y_te"], r["mu_te"], r["b_te"], lv)
        ax1.plot(lv, cov, marker="_", markersize=10, markeredgewidth=2.5,
                 color=COLORS[ticker], zorder=5)

ax1.set_xlabel("Nominal coverage level", fontsize=12)
ax1.set_ylabel("Actual coverage", fontsize=12)
ax1.set_title("Pooled XGB GAMLSS Calibration (6 stocks)",
              fontsize=13, fontweight="bold")
ax1.legend(fontsize=9, loc="upper left")
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_aspect("equal")
ax1.grid(True, alpha=0.15)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

plt.tight_layout()
fig1.savefig("calibration_pooled_6stocks.png", dpi=150, bbox_inches="tight")
print("Saved -> calibration_pooled_6stocks.png")
plt.close(fig1)


# ═════════════════════════════════════════════════════════════════════════════
# Figure 2: Coverage deviation — pooled model
# ═════════════════════════════════════════════════════════════════════════════
print("Plotting coverage deviation...", flush=True)

fig2, ax2 = plt.subplots(figsize=(10, 6))

ax2.axhline(0, color="black", lw=1.5, ls="--", alpha=0.6, zorder=1)
ax2.fill_between(cal_levels, -0.05, 0.05, color="orange", alpha=0.05, label="\u00b15% band")

for ticker in TICKERS:
    r = results_pooled[ticker]
    deviation = r["cal"] - cal_levels
    ax2.plot(cal_levels, deviation, color=COLORS[ticker], lw=2.5,
             marker=MARKERS[ticker], markersize=5, markevery=3,
             label=f"{ticker} (n={r['n_test']:,})", zorder=3)

ax2.set_xlabel("Nominal Coverage Level", fontsize=13)
ax2.set_ylabel("Actual \u2212 Nominal Coverage", fontsize=13)
ax2.set_title("Coverage Deviation: Pooled XGB GAMLSS (6 stocks)\n"
              "Positive = over-coverage (conservative), Negative = under-coverage",
              fontsize=13, fontweight="bold")
ax2.set_xlim(0.05, 0.95)
ax2.set_ylim(-0.12, 0.15)
ax2.legend(fontsize=9, loc="upper left", framealpha=0.9)
ax2.grid(True, alpha=0.2)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

plt.tight_layout()
fig2.savefig("coverage_deviation_pooled.png", dpi=150, bbox_inches="tight")
print("Saved -> coverage_deviation_pooled.png")
plt.close(fig2)


# ═════════════════════════════════════════════════════════════════════════════
# Figure 3: Per-stock vs Pooled calibration — side by side
# ═════════════════════════════════════════════════════════════════════════════
print("Plotting per-stock vs pooled comparison...", flush=True)

fig3, (ax_ps, ax_pl) = plt.subplots(1, 2, figsize=(18, 8.5))

for ax, res, title in [
    (ax_ps, results_perstock, "Per-Stock XGB GAMLSS"),
    (ax_pl, results_pooled, "Pooled XGB GAMLSS (6 stocks)"),
]:

    for ticker in TICKERS:
        r = res[ticker]
        ax.plot(cal_levels, r["cal"], color=COLORS[ticker], lw=2.5,
                marker=MARKERS[ticker], markersize=4, markevery=3,
                label=f"{ticker} ({r['n_test']:,})")

    ax.plot([0, 1], [0, 1], color="black", lw=1.2, ls="--", alpha=0.5,
            label="Perfect")

    for lv in [0.50, 0.80, 0.90]:
        ax.axvline(lv, color="gray", lw=0.8, ls=":", alpha=0.4, zorder=1)
        for ticker in TICKERS:
            r = res[ticker]
            cov = laplace_coverage(r["y_te"], r["mu_te"], r["b_te"], lv)
            ax.plot(lv, cov, marker="_", markersize=8, markeredgewidth=2,
                    color=COLORS[ticker], zorder=5)

    ax.set_xlabel("Nominal coverage level", fontsize=11)
    ax.set_ylabel("Actual coverage", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig3.suptitle("Calibration Comparison: Per-Stock vs Pooled Model",
              fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
fig3.savefig("calibration_pooled_vs_perstock.png", dpi=150, bbox_inches="tight")
print("Saved -> calibration_pooled_vs_perstock.png")
plt.close(fig3)

print("\nDone!")
