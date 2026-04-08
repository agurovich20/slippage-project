"""
Scatter of |slippage| vs Roll Estimated Spread with OLS line.

Output: ols_fitted_line.png
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

datasets = {
    "AAPL": ("data/lit_buy_features_v2.parquet", "data/lit_buy_features_v2_sep.parquet"),
    "COIN": ("data/coin_lit_buy_features_train.parquet", "data/coin_lit_buy_features_test.parquet"),
}

COLORS = {"AAPL": "#2563eb", "COIN": "#dc2626"}
rng = np.random.default_rng(42)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

for ax, (ticker, (tr_f, te_f)) in zip(axes, datasets.items()):
    df_tr = pd.read_parquet(tr_f)
    df_te = pd.read_parquet(te_f)
    df_tr["abs_impact"] = df_tr["impact_vwap_bps"].abs()
    df_te["abs_impact"] = df_te["impact_vwap_bps"].abs()

    x_tr = df_tr["roll_spread_500"].values
    y_tr = df_tr["abs_impact"].values
    x_te = df_te["roll_spread_500"].values
    y_te = df_te["abs_impact"].values

    # Univariate OLS
    X_tr = np.column_stack([x_tr, np.ones(len(x_tr))])
    beta, _, _, _ = np.linalg.lstsq(X_tr, y_tr, rcond=None)

    color = COLORS[ticker]

    n_plot = min(5000, len(x_tr))
    idx = rng.choice(len(x_tr), size=n_plot, replace=False)
    ax.scatter(x_tr[idx], y_tr[idx], s=8, alpha=0.1, color=color,
               edgecolors="none", label="Train", rasterized=True)
    ax.scatter(x_te, y_te, s=12, alpha=0.2, color="#f59e0b",
               edgecolors="none", label="Test", rasterized=True)

    x_line = np.linspace(0, np.percentile(np.concatenate([x_tr, x_te]), 99), 300)
    y_line = beta[0] * x_line + beta[1]
    ax.plot(x_line, y_line, color="black", lw=3, zorder=5,
            label=f"OLS: {beta[0]:.2f}x + {beta[1]:.2f}")

    ax.set_xlabel("Roll Estimated Spread (bps)", fontsize=12)
    ax.set_ylabel("|slippage| (bps)", fontsize=12)
    ax.set_title(f"{ticker} OLS: |slippage| vs Roll Estimated Spread",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    y_cap = np.percentile(np.concatenate([y_tr, y_te]), 98)
    ax.set_ylim(0, y_cap * 1.1)
    ax.set_xlim(0, x_line[-1])

plt.tight_layout()
plt.savefig("ols_fitted_line.png", dpi=150, bbox_inches="tight")
print("Saved -> ols_fitted_line.png")
