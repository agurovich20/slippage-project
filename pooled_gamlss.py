"""
Pooled XGB GAMLSS: single model trained on normalized, pooled data from 6 stocks.
Tests on each stock's September holdout individually.

Key idea: normalize each stock's features by its training-set median so that
"median spread for COIN" and "median spread for AAPL" both map to 1.0.
The target is also normalized by median |impact|.

Output:
  - pooled_vs_perstock.png
  - Comparison table: per-stock vs pooled
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

DATASETS = {
    "AAPL": ("data/lit_buy_features_v2.parquet", "data/lit_buy_features_v2_sep.parquet"),
    "COIN": ("data/coin_lit_buy_features_train.parquet", "data/coin_lit_buy_features_test.parquet"),
    "NVDA": ("data/nvda_lit_buy_features_train.parquet", "data/nvda_lit_buy_features_test.parquet"),
    "AMD":  ("data/amd_lit_buy_features_train.parquet", "data/amd_lit_buy_features_test.parquet"),
    "AMZN": ("data/amzn_lit_buy_features_train.parquet", "data/amzn_lit_buy_features_test.parquet"),
    "TSLA": ("data/tsla_lit_buy_features_train.parquet", "data/tsla_lit_buy_features_test.parquet"),
}

LOC_PARAMS = dict(max_depth=3, n_estimators=200, learning_rate=0.07,
                  min_child_weight=5, reg_alpha=10, reg_lambda=10)
SCALE_PARAMS = dict(max_depth=3, n_estimators=50, learning_rate=0.1,
                    min_child_weight=20, reg_alpha=1, reg_lambda=1)


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1: Load and pool training data
# ═════════════════════════════════════════════════════════════════════════════
print("STEP 1: Loading and normalizing training data...")

stock_data = {}
pooled_X = []
pooled_y = []

for ticker, (tr_f, te_f) in DATASETS.items():
    df_tr = pd.read_parquet(tr_f)
    df_te = pd.read_parquet(te_f)
    df_tr = df_tr[df_tr["impact_vwap_bps"].abs() <= IMPACT_CAP_BPS]
    df_te = df_te[df_te["impact_vwap_bps"].abs() <= IMPACT_CAP_BPS]
    df_tr["abs_impact"] = df_tr["impact_vwap_bps"].abs()
    df_te["abs_impact"] = df_te["impact_vwap_bps"].abs()

    # Training-set medians for normalization
    medians = {}
    for feat in FEATURES_3:
        medians[feat] = df_tr[feat].median()
        # Guard against zero median
        if medians[feat] == 0 or np.isnan(medians[feat]):
            medians[feat] = df_tr[feat].mean()
            if medians[feat] == 0:
                medians[feat] = 1.0

    median_impact = df_tr["abs_impact"].median()
    if median_impact == 0 or np.isnan(median_impact):
        median_impact = df_tr["abs_impact"].mean()

    # Normalize training features and target
    X_tr_norm = np.column_stack([
        df_tr[feat].values / medians[feat] for feat in FEATURES_3
    ])
    y_tr_norm = df_tr["abs_impact"].values / median_impact

    # Normalize test features (same medians from training)
    X_te_norm = np.column_stack([
        df_te[feat].values / medians[feat] for feat in FEATURES_3
    ])
    y_te_raw = df_te["abs_impact"].values

    stock_data[ticker] = {
        "medians": medians,
        "median_impact": median_impact,
        "X_tr_norm": X_tr_norm,
        "y_tr_norm": y_tr_norm,
        "X_te_norm": X_te_norm,
        "y_te_raw": y_te_raw,
        "n_train": len(df_tr),
        "n_test": len(df_te),
    }

    pooled_X.append(X_tr_norm)
    pooled_y.append(y_tr_norm)

pooled_X = np.vstack(pooled_X)
pooled_y = np.concatenate(pooled_y)

print(f"\n  {'Ticker':<8} {'n_train':>8} {'med_impact':>11} {'med_spread':>11} "
      f"{'med_vol':>10} {'med_prate':>10}")
print(f"  {'-'*8} {'-'*8} {'-'*11} {'-'*11} {'-'*10} {'-'*10}")
for ticker in DATASETS:
    d = stock_data[ticker]
    m = d["medians"]
    print(f"  {ticker:<8} {d['n_train']:>8,} {d['median_impact']:>11.4f} "
          f"{m['roll_spread_500']:>11.4f} {m['roll_vol_500']:>10.4f} "
          f"{m['participation_rate']:>10.6f}")
print(f"\n  Pooled training set: {len(pooled_y):,} trades")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2: Fit pooled location model
# ═════════════════════════════════════════════════════════════════════════════
print("\nSTEP 2: Fitting pooled location model (XGB LAD)...")

loc_model = XGBRegressor(
    objective="reg:absoluteerror", tree_method="hist",
    verbosity=0, random_state=42, n_jobs=1, **LOC_PARAMS,
)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    loc_model.fit(pooled_X, pooled_y)

mu_pooled_tr = np.maximum(loc_model.predict(pooled_X), 0.0)
mae_tr_norm = np.mean(np.abs(pooled_y - mu_pooled_tr))
print(f"  Pooled train MAE (normalized): {mae_tr_norm:.4f}")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 3: Fit pooled scale model
# ═════════════════════════════════════════════════════════════════════════════
print("\nSTEP 3: Fitting pooled scale model (XGB MSE on |residuals|)...")

abs_resid_tr = np.abs(pooled_y - mu_pooled_tr)
scale_model = XGBRegressor(
    objective="reg:squarederror", tree_method="hist",
    verbosity=0, random_state=42, n_jobs=1, **SCALE_PARAMS,
)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    scale_model.fit(pooled_X, abs_resid_tr)

print(f"  Scale model fitted on {len(abs_resid_tr):,} residuals")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 4: Evaluate per stock
# ═════════════════════════════════════════════════════════════════════════════
print("\nSTEP 4: Evaluating pooled model on each stock's September holdout...\n")

pooled_results = {}

for ticker in DATASETS:
    d = stock_data[ticker]
    X_te_norm = d["X_te_norm"]
    y_te = d["y_te_raw"]
    scale = d["median_impact"]

    # Predict in normalized space, then un-normalize
    mu_te_norm = np.maximum(loc_model.predict(X_te_norm), 0.0)
    b_te_norm = np.clip(scale_model.predict(X_te_norm), 0.01, None)

    mu_te = mu_te_norm * scale
    b_te = np.clip(b_te_norm * scale, 0.1, None)

    mae_te = np.mean(np.abs(y_te - mu_te))

    cov_data = {}
    for level in [0.50, 0.80, 0.90]:
        z = np.log(1.0 / (1.0 - level))
        lo = np.maximum(mu_te - z * b_te, 0.0)
        hi = mu_te + z * b_te
        cov = ((y_te >= lo) & (y_te <= hi)).mean()
        width = (hi - lo).mean()
        cov_data[level] = (cov, width)

    pooled_results[ticker] = {
        "mae_te": mae_te,
        "coverage_data": cov_data,
        "n_test": len(y_te),
    }

    print(f"  {ticker}: MAE={mae_te:.4f}, "
          f"90% cov={cov_data[0.90][0]:.4f}, "
          f"80% cov={cov_data[0.80][0]:.4f}, "
          f"50% cov={cov_data[0.50][0]:.4f}, "
          f"90% width={cov_data[0.90][1]:.2f}")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 5: Fit per-stock models for comparison
# ═════════════════════════════════════════════════════════════════════════════
print("\nFitting per-stock models for comparison...")

perstock_results = {}

for ticker, (tr_f, te_f) in DATASETS.items():
    df_tr = pd.read_parquet(tr_f)
    df_te = pd.read_parquet(te_f)
    df_tr = df_tr[df_tr["impact_vwap_bps"].abs() <= IMPACT_CAP_BPS]
    df_te = df_te[df_te["impact_vwap_bps"].abs() <= IMPACT_CAP_BPS]
    df_tr["abs_impact"] = df_tr["impact_vwap_bps"].abs()
    df_te["abs_impact"] = df_te["impact_vwap_bps"].abs()

    X_tr = df_tr[FEATURES_3].to_numpy(dtype=np.float64)
    y_tr = df_tr["abs_impact"].to_numpy(dtype=np.float64)
    X_te = df_te[FEATURES_3].to_numpy(dtype=np.float64)
    y_te = df_te["abs_impact"].to_numpy(dtype=np.float64)

    loc_ps = XGBRegressor(objective="reg:absoluteerror", tree_method="hist",
                          verbosity=0, random_state=42, n_jobs=1, **LOC_PARAMS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loc_ps.fit(X_tr, y_tr)
    mu_tr_ps = np.maximum(loc_ps.predict(X_tr), 0.0)
    mu_te_ps = np.maximum(loc_ps.predict(X_te), 0.0)

    abs_r_ps = np.abs(y_tr - mu_tr_ps)
    sc_ps = XGBRegressor(objective="reg:squarederror", tree_method="hist",
                         verbosity=0, random_state=42, n_jobs=1, **SCALE_PARAMS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc_ps.fit(X_tr, abs_r_ps)
    b_te_ps = np.clip(sc_ps.predict(X_te), 0.1, None)

    mae_ps = np.mean(np.abs(y_te - mu_te_ps))

    cov_data_ps = {}
    for level in [0.50, 0.80, 0.90]:
        z = np.log(1.0 / (1.0 - level))
        lo = np.maximum(mu_te_ps - z * b_te_ps, 0.0)
        hi = mu_te_ps + z * b_te_ps
        cov = ((y_te >= lo) & (y_te <= hi)).mean()
        width = (hi - lo).mean()
        cov_data_ps[level] = (cov, width)

    perstock_results[ticker] = {
        "mae_te": mae_ps,
        "coverage_data": cov_data_ps,
    }


# ═════════════════════════════════════════════════════════════════════════════
# STEP 5: Comparison table
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 95}")
print("  COMPARISON: Per-Stock XGB GAMLSS vs Pooled XGB GAMLSS")
print(f"{'=' * 95}")
print(f"  {'Ticker':<8} {'n_test':>7} "
      f"{'PS 90%':>7} {'Pool 90%':>8} "
      f"{'PS MAE':>7} {'Pool MAE':>9} "
      f"{'PS Width':>9} {'Pool Width':>11}")
print(f"  {'-'*8} {'-'*7} "
      f"{'-'*7} {'-'*8} "
      f"{'-'*7} {'-'*9} "
      f"{'-'*9} {'-'*11}")

for ticker in DATASETS:
    ps = perstock_results[ticker]
    pl = pooled_results[ticker]
    ps_cd = ps["coverage_data"]
    pl_cd = pl["coverage_data"]
    print(f"  {ticker:<8} {pl['n_test']:>7,} "
          f"{ps_cd[0.90][0]:>7.4f} {pl_cd[0.90][0]:>8.4f} "
          f"{ps['mae_te']:>7.4f} {pl['mae_te']:>9.4f} "
          f"{ps_cd[0.90][1]:>9.2f} {pl_cd[0.90][1]:>11.2f}")

print(f"\n  Nominal 90% target: 0.9000")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 6: Figure
# ═════════════════════════════════════════════════════════════════════════════
print("\nPlotting comparison...")

tickers_ordered = list(DATASETS.keys())
x = np.arange(len(tickers_ordered))
bw = 0.35

COLORS_PS = "#2563eb"
COLORS_PL = "#dc2626"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))

# Left panel: 90% coverage
cov_ps = [perstock_results[t]["coverage_data"][0.90][0] for t in tickers_ordered]
cov_pl = [pooled_results[t]["coverage_data"][0.90][0] for t in tickers_ordered]

bars1 = ax1.bar(x - bw/2, cov_ps, width=bw, color=COLORS_PS, alpha=0.8,
                edgecolor="white", linewidth=0.8, label="Per-Stock")
bars2 = ax1.bar(x + bw/2, cov_pl, width=bw, color=COLORS_PL, alpha=0.8,
                edgecolor="white", linewidth=0.8, label="Pooled")

ax1.axhline(0.90, color="black", lw=1.2, ls="--", alpha=0.5, label="Nominal 90%")

for i, (v1, v2) in enumerate(zip(cov_ps, cov_pl)):
    ax1.text(i - bw/2, v1 + 0.005, f"{v1:.3f}", ha="center", fontsize=7.5, color=COLORS_PS)
    ax1.text(i + bw/2, v2 + 0.005, f"{v2:.3f}", ha="center", fontsize=7.5, color=COLORS_PL)

ax1.set_xticks(x)
ax1.set_xticklabels(tickers_ordered, fontsize=10, fontweight="bold")
ax1.set_ylabel("Actual Coverage", fontsize=11)
ax1.set_title("90% Coverage: Per-Stock vs Pooled", fontsize=13, fontweight="bold")
ax1.legend(fontsize=9, loc="lower right")
ax1.set_ylim(0.75, 1.0)
ax1.grid(axis="y", alpha=0.2)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# Right panel: MAE
mae_ps = [perstock_results[t]["mae_te"] for t in tickers_ordered]
mae_pl = [pooled_results[t]["mae_te"] for t in tickers_ordered]

bars3 = ax2.bar(x - bw/2, mae_ps, width=bw, color=COLORS_PS, alpha=0.8,
                edgecolor="white", linewidth=0.8, label="Per-Stock")
bars4 = ax2.bar(x + bw/2, mae_pl, width=bw, color=COLORS_PL, alpha=0.8,
                edgecolor="white", linewidth=0.8, label="Pooled")

for i, (v1, v2) in enumerate(zip(mae_ps, mae_pl)):
    ax2.text(i - bw/2, v1 + 0.03, f"{v1:.3f}", ha="center", fontsize=7.5, color=COLORS_PS)
    ax2.text(i + bw/2, v2 + 0.03, f"{v2:.3f}", ha="center", fontsize=7.5, color=COLORS_PL)

ax2.set_xticks(x)
ax2.set_xticklabels(tickers_ordered, fontsize=10, fontweight="bold")
ax2.set_ylabel("MAE (bps)", fontsize=11)
ax2.set_title("Location MAE: Per-Stock vs Pooled", fontsize=13, fontweight="bold")
ax2.legend(fontsize=9)
ax2.grid(axis="y", alpha=0.2)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

fig.suptitle("Pooled XGB GAMLSS (median-normalized, 6 stocks) vs Per-Stock Models\n"
             "3 Features | Fixed AAPL Hyperparameters | Train: Jun-Aug 2024 | Test: Sep 2024",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig("pooled_vs_perstock.png", dpi=150, bbox_inches="tight")
print("Saved -> pooled_vs_perstock.png")
