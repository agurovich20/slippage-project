"""
Calibration curves for XGB GAMLSS Laplace prediction intervals across 6 stocks.
Plots nominal coverage vs actual coverage at many levels.
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

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

COLORS = {
    "AAPL": "#2563eb", "COIN": "#dc2626", "NVDA": "#16a34a",
    "AMD":  "#f59e0b", "AMZN": "#7c3aed", "TSLA": "#ec4899",
}

NOMINAL_LEVELS = np.arange(0.05, 1.00, 0.05)  # 5%, 10%, ..., 95%


def fit_gamlss(train_df, test_df):
    X_tr = train_df[FEATURES_3].to_numpy(dtype=np.float64)
    y_tr = train_df["impact_vwap_bps"].abs().to_numpy(dtype=np.float64)
    X_te = test_df[FEATURES_3].to_numpy(dtype=np.float64)
    y_te = test_df["impact_vwap_bps"].abs().to_numpy(dtype=np.float64)

    loc_model = XGBRegressor(
        objective="reg:absoluteerror", tree_method="hist",
        max_depth=3, n_estimators=200, learning_rate=0.07,
        min_child_weight=5, reg_alpha=10, reg_lambda=10,
        verbosity=0, random_state=42, n_jobs=1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loc_model.fit(X_tr, y_tr)

    mu_hat_tr = np.maximum(loc_model.predict(X_tr), 0.0)
    mu_hat_te = np.maximum(loc_model.predict(X_te), 0.0)

    abs_resid_tr = np.abs(y_tr - mu_hat_tr)
    scale_model = XGBRegressor(
        objective="reg:squarederror", tree_method="hist",
        max_depth=3, n_estimators=50, learning_rate=0.1,
        min_child_weight=20, reg_alpha=1, reg_lambda=1,
        verbosity=0, random_state=42, n_jobs=1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scale_model.fit(X_tr, abs_resid_tr)

    b_hat_te = np.clip(scale_model.predict(X_te), 0.1, None)

    return y_te, mu_hat_te, b_hat_te


def compute_calibration(y_te, mu_hat_te, b_hat_te, levels):
    actual_coverages = []
    for level in levels:
        z = np.log(1.0 / (1.0 - level))
        lo = np.maximum(mu_hat_te - z * b_hat_te, 0.0)
        hi = mu_hat_te + z * b_hat_te
        cov = ((y_te >= lo) & (y_te <= hi)).mean()
        actual_coverages.append(cov)
    return np.array(actual_coverages)


# Fit models and compute calibration for each stock
results = {}
for ticker, (tr_path, te_path) in DATASETS.items():
    train_df = pd.read_parquet(tr_path)
    test_df = pd.read_parquet(te_path)
    train_df = train_df[train_df["impact_vwap_bps"].abs() <= IMPACT_CAP_BPS]
    test_df = test_df[test_df["impact_vwap_bps"].abs() <= IMPACT_CAP_BPS]

    y_te, mu_hat_te, b_hat_te = fit_gamlss(train_df, test_df)
    actual = compute_calibration(y_te, mu_hat_te, b_hat_te, NOMINAL_LEVELS)
    results[ticker] = actual
    print(f"  {ticker}: done (n_test={len(y_te):,})")


# ─── Figure ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(20, 9))

# Panel 1: Calibration curves (all stocks overlaid)
ax = axes[0]
ax.plot([0, 1], [0, 1], color="black", lw=1.5, ls="--", alpha=0.6,
        label="Perfect calibration", zorder=1)
ax.fill_between([0, 1], [0, 1], [0.05, 1.05], color="gray", alpha=0.06)
ax.fill_between([0, 1], [-0.05, 0.95], [0, 1], color="gray", alpha=0.06)

for ticker in DATASETS:
    ax.plot(NOMINAL_LEVELS, results[ticker], color=COLORS[ticker],
            lw=2.5, marker="o", markersize=5, label=ticker, zorder=3)

ax.set_xlabel("Nominal Coverage Level", fontsize=13)
ax.set_ylabel("Actual Coverage", fontsize=13)
ax.set_title("Calibration Curves: XGB GAMLSS with Laplace Intervals\n"
             "6 Stocks, 3 Features, Fixed AAPL Hyperparameters",
             fontsize=14, fontweight="bold")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("equal")
ax.legend(fontsize=10, loc="lower right", framealpha=0.9)
ax.grid(True, alpha=0.2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Panel 2: Deviation from nominal (actual - nominal)
ax2 = axes[1]
ax2.axhline(0, color="black", lw=1.5, ls="--", alpha=0.6, zorder=1)
ax2.fill_between(NOMINAL_LEVELS, -0.02, 0.02, color="green", alpha=0.08,
                 label="\u00b12% band")
ax2.fill_between(NOMINAL_LEVELS, -0.05, 0.05, color="orange", alpha=0.05,
                 label="\u00b15% band")

for ticker in DATASETS:
    deviation = results[ticker] - NOMINAL_LEVELS
    ax2.plot(NOMINAL_LEVELS, deviation, color=COLORS[ticker],
             lw=2.5, marker="o", markersize=5, label=ticker, zorder=3)

ax2.set_xlabel("Nominal Coverage Level", fontsize=13)
ax2.set_ylabel("Actual \u2212 Nominal Coverage", fontsize=13)
ax2.set_title("Coverage Deviation from Nominal\n"
              "Positive = over-coverage (conservative), Negative = under-coverage",
              fontsize=14, fontweight="bold")
ax2.set_xlim(0, 1)
ax2.set_ylim(-0.15, 0.15)
ax2.legend(fontsize=9, loc="upper left", framealpha=0.9, ncol=2)
ax2.grid(True, alpha=0.2)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

fig.suptitle(
    "XGB GAMLSS Calibration: Block Trade Impact Prediction Intervals\n"
    "Train: Jun\u2013Aug 2024 | Test: Sep 2024 | Laplace distribution",
    fontsize=15, fontweight="bold", y=1.03,
)

plt.tight_layout()
fig.savefig("calibration_curves.png", dpi=150, bbox_inches="tight")
print("\nSaved -> calibration_curves.png")
