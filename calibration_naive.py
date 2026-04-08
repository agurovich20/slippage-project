"""
Naive baseline calibration: global mean + global std, Gaussian intervals.
mu = training mean of |impact|, sigma = training std of |impact|.
Same interval for every trade.

Output: calibration_naive.png
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    "AAPL": "#2563eb", "COIN": "#dc2626",
    "NVDA": "#16a34a", "AMD": "#f59e0b", "AMZN": "#7c3aed", "TSLA": "#ec4899",
}
MARKERS = {
    "AAPL": "o", "COIN": "X",
    "NVDA": "D", "AMD": "s", "AMZN": "^", "TSLA": "v",
}


def normal_coverage(y, mu, sigma, level):
    """Coverage of a symmetric Gaussian interval [mu - z*sigma, mu + z*sigma], clipped at 0."""
    z = stats.norm.ppf(0.5 + level / 2.0)
    lo = np.maximum(mu - z * sigma, 0.0)
    hi = mu + z * sigma
    return ((y >= lo) & (y <= hi)).mean()


results = {}
for ticker, (tr_f, te_f) in DATASETS.items():
    df_tr = pd.read_parquet(tr_f)
    df_te = pd.read_parquet(te_f)
    df_tr = df_tr[df_tr["impact_vwap_bps"].abs() <= IMPACT_CAP_BPS]
    df_te = df_te[df_te["impact_vwap_bps"].abs() <= IMPACT_CAP_BPS]

    y_tr = df_tr["impact_vwap_bps"].abs().to_numpy(dtype=np.float64)
    y_te = df_te["impact_vwap_bps"].abs().to_numpy(dtype=np.float64)

    mu = y_tr.mean()
    sigma = y_tr.std()

    cal = np.array([normal_coverage(y_te, mu, sigma, lv) for lv in cal_levels])

    results[ticker] = {
        "cal": cal, "mu": mu, "sigma": sigma,
        "y_te": y_te, "n_test": len(y_te),
    }
    print(f"{ticker}: mu={mu:.2f}  sigma={sigma:.2f}  n_test={len(y_te):,}")


# -- Calibration curve ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 9))

for ticker in DATASETS:
    r = results[ticker]
    ax.plot(cal_levels, r["cal"], color=COLORS[ticker], lw=2.5,
            marker=MARKERS[ticker], markersize=4, markevery=3,
            label=f"{ticker} (n={r['n_test']:,})")

ax.plot([0, 1], [0, 1], color="black", lw=1.2, ls="--", alpha=0.5,
        label="Perfect calibration")

ax.set_xlabel("Nominal coverage level", fontsize=12)
ax.set_ylabel("Actual coverage", fontsize=12)
ax.set_title("Naive Gaussian Calibration",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10, loc="upper left")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("equal")
ax.grid(True, alpha=0.15)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
fig.savefig("calibration_naive.png", dpi=150, bbox_inches="tight")
print("Saved -> calibration_naive.png")

# -- Coverage deviation --------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(10, 6))

ax2.axhline(0, color="black", lw=1.5, ls="--", alpha=0.6, zorder=1)
ax2.fill_between(cal_levels, -0.05, 0.05, color="orange", alpha=0.05, label="\u00b15% band")

for ticker in DATASETS:
    r = results[ticker]
    deviation = r["cal"] - cal_levels
    ax2.plot(cal_levels, deviation, color=COLORS[ticker], lw=2.5,
             marker=MARKERS[ticker], markersize=5, markevery=3,
             label=f"{ticker} (n={r['n_test']:,})", zorder=3)

ax2.set_xlabel("Nominal Coverage Level", fontsize=13)
ax2.set_ylabel("Actual \u2212 Nominal Coverage", fontsize=13)
ax2.set_title("Coverage Deviation: Naive Gaussian",
              fontsize=13, fontweight="bold")
ax2.set_xlim(0.05, 0.95)
ax2.set_ylim(-0.12, 0.15)
ax2.legend(fontsize=10, loc="upper left", framealpha=0.9)
ax2.grid(True, alpha=0.2)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

plt.tight_layout()
fig2.savefig("coverage_deviation_naive.png", dpi=150, bbox_inches="tight")
print("Saved -> coverage_deviation_naive.png")

print("\nDone!")
