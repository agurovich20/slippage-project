"""
OLS |slippage| scatter plots rotating over features as x-axis.

One figure per ticker, each with a panel per feature showing:
  - Train scatter (subsampled)
  - Test scatter
  - Binned mean trend line

Output:
  - ols_unsigned_aapl_by_feature.png
  - ols_unsigned_coin_by_feature.png
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FEATURES = [
    "dollar_value", "log_dollar_value", "participation_rate",
    "roll_spread_500", "roll_vol_500", "exchange_id",
]

FEATURE_LABELS = {
    "dollar_value": "Dollar Value",
    "log_dollar_value": "Log Dollar Value",
    "participation_rate": "Participation Rate",
    "roll_spread_500": "Roll Spread 500 (bps)",
    "roll_vol_500": "Roll Volatility 500 (bps)",
    "exchange_id": "Exchange ID",
}

COLORS = {"AAPL": "#2563eb", "COIN": "#dc2626"}

datasets = {
    "AAPL": ("data/lit_buy_features_v2.parquet", "data/lit_buy_features_v2_sep.parquet"),
    "COIN": ("data/coin_lit_buy_features_train.parquet", "data/coin_lit_buy_features_test.parquet"),
}

rng = np.random.default_rng(42)

for ticker, (tr_f, te_f) in datasets.items():
    print(f"Processing {ticker}...", flush=True)
    df_tr = pd.read_parquet(tr_f)
    df_te = pd.read_parquet(te_f)
    df_tr["abs_impact"] = df_tr["impact_vwap_bps"].abs()
    df_te["abs_impact"] = df_te["impact_vwap_bps"].abs()
    df_tr = df_tr.sort_values("date").reset_index(drop=True)

    y_tr = df_tr["abs_impact"].values
    y_te = df_te["abs_impact"].values
    color = COLORS[ticker]

    fig, axes = plt.subplots(2, 3, figsize=(22, 13))
    axes = axes.ravel()

    for i, feat in enumerate(FEATURES):
        ax = axes[i]
        x_tr = df_tr[feat].values
        x_te = df_te[feat].values

        # Subsample train
        n_plot = min(4000, len(x_tr))
        idx = rng.choice(len(x_tr), size=n_plot, replace=False)

        ax.scatter(x_tr[idx], y_tr[idx], s=6, alpha=0.12,
                   color=color, edgecolors="none", zorder=2, label=f"Train ({len(x_tr):,})")
        ax.scatter(x_te, y_te, s=10, alpha=0.25, color="#f59e0b",
                   edgecolors="none", zorder=2, label=f"Test ({len(x_te):,})")

        # Binned mean trend (train)
        n_bins = 25
        lo_pct, hi_pct = np.percentile(x_tr, [2, 98])
        edges = np.linspace(lo_pct, hi_pct, n_bins + 1)
        bx, by = [], []
        for j in range(n_bins):
            mask = (x_tr >= edges[j]) & (x_tr < edges[j + 1])
            if j == n_bins - 1:
                mask = (x_tr >= edges[j]) & (x_tr <= edges[j + 1])
            if mask.sum() > 10:
                bx.append((edges[j] + edges[j + 1]) / 2)
                by.append(y_tr[mask].mean())
        ax.plot(bx, by, color="black", lw=2.2, zorder=4, marker="o", markersize=3,
                label="Binned mean")

        # Correlation
        valid = np.isfinite(x_tr) & np.isfinite(y_tr)
        corr = np.corrcoef(x_tr[valid], y_tr[valid])[0, 1]
        ax.text(0.97, 0.97, f"r = {corr:.3f}",
                transform=ax.transAxes, fontsize=10, family="monospace",
                va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#94a3b8", alpha=0.92))

        ax.set_xlabel(FEATURE_LABELS[feat], fontsize=11)
        ax.set_ylabel("|slippage| (bps)", fontsize=11)
        ax.set_title(f"|slippage| vs {FEATURE_LABELS[feat]}",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9)
        ax.grid(True, alpha=0.18)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Cap y-axis
        y_hi = np.percentile(np.concatenate([y_tr, y_te]), 98)
        ax.set_ylim(0, y_hi * 1.15)
        ax.set_xlim(np.percentile(x_tr, 1), np.percentile(x_tr, 99))

    fig.suptitle(f"{ticker} OLS Feature Exploration: |slippage| vs Each Feature\n"
                 f"Train: {len(df_tr):,} trades (Jun-Aug)  |  Test: {len(df_te):,} trades (Sep)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fname = f"ols_unsigned_{ticker.lower()}_by_feature.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved -> {fname}")

print("\nDone!")
