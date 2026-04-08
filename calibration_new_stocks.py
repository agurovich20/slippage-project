"""
Calibration curves and coverage deviation for NVDA, AMD, AMZN, TSLA.
Two-Stage XGBoost with Laplace intervals, 3 features, fixed AAPL hyperparameters.

Output:
  - calibration_xgb_nvda_amd.png
  - calibration_xgb_amzn_tsla.png
  - coverage_deviation_4stocks.png
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
    "NVDA": ("data/nvda_lit_buy_features_train.parquet", "data/nvda_lit_buy_features_test.parquet"),
    "AMD":  ("data/amd_lit_buy_features_train.parquet", "data/amd_lit_buy_features_test.parquet"),
    "AMZN": ("data/amzn_lit_buy_features_train.parquet", "data/amzn_lit_buy_features_test.parquet"),
    "TSLA": ("data/tsla_lit_buy_features_train.parquet", "data/tsla_lit_buy_features_test.parquet"),
}

COLORS = {
    "NVDA": "#16a34a", "AMD": "#f59e0b", "AMZN": "#7c3aed", "TSLA": "#ec4899",
}
MARKERS = {
    "NVDA": "o", "AMD": "s", "AMZN": "D", "TSLA": "^",
}


def laplace_coverage(y, mu, b, level):
    z = np.log(1.0 / (1.0 - level))
    lo = np.maximum(mu - z * b, 0.0)
    hi = mu + z * b
    return ((y >= lo) & (y <= hi)).mean()


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
                      random_state=42, n_jobs=1, max_depth=3, n_estimators=50,
                      min_child_weight=20, learning_rate=0.1, reg_alpha=1, reg_lambda=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc.fit(X_tr, abs_r)
    b_te = np.clip(sc.predict(X_te), 0.1, None)
    return mu_te, b_te


# -- Load and fit all 4 stocks ------------------------------------------------
results = {}
for ticker, (tr_f, te_f) in DATASETS.items():
    print(f"Fitting {ticker}...", flush=True)
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

    mu_te, b_te = fit_xgb_gamlss(X_tr, y_tr, X_te)
    cal = np.array([laplace_coverage(y_te, mu_te, b_te, lv) for lv in cal_levels])

    results[ticker] = {
        "cal": cal, "mu_te": mu_te, "b_te": b_te,
        "y_te": y_te, "n_test": len(y_te),
    }
    print(f"  {ticker}: n_test={len(y_te):,}")


def plot_pair(ticker1, ticker2, filename):
    """Calibration curve for a pair of stocks, same style as calibration_xgb_overlay.png."""
    fig, ax = plt.subplots(figsize=(8, 8))

    c1, c2 = COLORS[ticker1], COLORS[ticker2]
    r1, r2 = results[ticker1], results[ticker2]

    ax.plot(cal_levels, r1["cal"], color=c1, lw=2.5,
            marker=MARKERS[ticker1], markersize=4,
            label=f"{ticker1} ({r1['n_test']:,} test trades)")
    ax.plot(cal_levels, r2["cal"], color=c2, lw=2.5,
            marker=MARKERS[ticker2], markersize=4,
            label=f"{ticker2} ({r2['n_test']:,} test trades)")
    ax.plot([0, 1], [0, 1], color="black", lw=1.2, ls="--", alpha=0.5,
            label="Perfect calibration")

    for ticker, color, yoff in [(ticker1, c1, 12), (ticker2, c2, -16)]:
        r = results[ticker]
        for lv in [0.50, 0.80, 0.90]:
            cov = laplace_coverage(r["y_te"], r["mu_te"], r["b_te"], lv)
            z = np.log(1.0 / (1.0 - lv))
            w = (2 * z * r["b_te"]).mean()
            ax.annotate(f"{lv:.0%}: {cov:.1%} (w={w:.1f})", xy=(lv, cov),
                        textcoords="offset points", xytext=(12, yoff),
                        fontsize=8, color=color, fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color=color, lw=0.8))

    ax.set_xlabel("Nominal coverage level", fontsize=12)
    ax.set_ylabel("Actual coverage", fontsize=12)
    ax.set_title(f"Two-Stage XGBoost Calibration: {ticker1} and {ticker2}",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Saved -> {filename}")
    plt.close(fig)


# -- Figure 1: All 4 stocks on one calibration curve --------------------------
fig, ax = plt.subplots(figsize=(9, 9))


for ticker in ["NVDA", "AMD", "AMZN", "TSLA"]:
    r = results[ticker]
    ax.plot(cal_levels, r["cal"], color=COLORS[ticker], lw=2.5,
            marker=MARKERS[ticker], markersize=4, markevery=3,
            label=f"{ticker} ({r['n_test']:,} test trades)")

ax.plot([0, 1], [0, 1], color="black", lw=1.2, ls="--", alpha=0.5,
        label="Perfect calibration")

# Vertical reference lines at 50%, 80%, 90% with label on the diagonal
for lv in [0.50, 0.80, 0.90]:
    ax.axvline(lv, color="gray", lw=0.8, ls=":", alpha=0.4, zorder=1)
    ax.annotate(f"{lv:.0%}", xy=(lv, lv), textcoords="offset points",
                xytext=(-14, -10), fontsize=9, color="black", fontweight="bold",
                alpha=0.6)
    # Small tick marks on each curve at these levels
    for ticker in ["NVDA", "AMD", "AMZN", "TSLA"]:
        r = results[ticker]
        cov = laplace_coverage(r["y_te"], r["mu_te"], r["b_te"], lv)
        ax.plot(lv, cov, marker="_", markersize=10, markeredgewidth=2.5,
                color=COLORS[ticker], zorder=5)

ax.set_xlabel("Nominal coverage level", fontsize=12)
ax.set_ylabel("Actual coverage", fontsize=12)
ax.set_title("Two-Stage XGBoost Calibration: NVDA, AMD, AMZN, TSLA",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10, loc="upper left")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("equal")
ax.grid(True, alpha=0.15)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
fig.savefig("calibration_xgb_4stocks.png", dpi=150, bbox_inches="tight")
print("Saved -> calibration_xgb_4stocks.png")
plt.close(fig)

# -- Figure 3: Coverage deviation for all 4 stocks ----------------------------
fig3, ax3 = plt.subplots(figsize=(10, 6))

ax3.axhline(0, color="black", lw=1.5, ls="--", alpha=0.6, zorder=1)
ax3.fill_between(cal_levels, -0.05, 0.05, color="orange", alpha=0.05, label="\u00b15% band")

for ticker in ["NVDA", "AMD", "AMZN", "TSLA"]:
    r = results[ticker]
    deviation = r["cal"] - cal_levels
    ax3.plot(cal_levels, deviation, color=COLORS[ticker], lw=2.5,
             marker=MARKERS[ticker], markersize=5, markevery=3,
             label=f"{ticker} (n={r['n_test']:,})", zorder=3)

ax3.set_xlabel("Nominal Coverage Level", fontsize=13)
ax3.set_ylabel("Actual \u2212 Nominal Coverage", fontsize=13)
ax3.set_title("Coverage Deviation: Two-Stage XGBoost (NVDA, AMD, AMZN, TSLA)\n"
              "Positive = over-coverage (conservative), Negative = under-coverage",
              fontsize=13, fontweight="bold")
ax3.set_xlim(0.05, 0.95)
ax3.set_ylim(-0.12, 0.15)
ax3.legend(fontsize=10, loc="upper left", framealpha=0.9)
ax3.grid(True, alpha=0.2)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)

plt.tight_layout()
fig3.savefig("coverage_deviation_4stocks.png", dpi=150, bbox_inches="tight")
print("Saved -> coverage_deviation_4stocks.png")

print("\nDone!")
