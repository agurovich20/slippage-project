import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Load data
coin_tr = pd.read_parquet("data/coin_lit_buy_features_train.parquet")
coin_te = pd.read_parquet("data/coin_lit_buy_features_test.parquet")

x_tr = coin_tr["roll_spread_500"].values.astype(np.float64)
y_tr = coin_tr["impact_vwap_bps"].values.astype(np.float64)
x_te = coin_te["roll_spread_500"].values.astype(np.float64)
y_te = coin_te["impact_vwap_bps"].values.astype(np.float64)

# OLS fit
X_tr = np.column_stack([x_tr, np.ones(len(x_tr))])
beta, *_ = np.linalg.lstsq(X_tr, y_tr, rcond=None)

# Plot
fig, ax = plt.subplots(figsize=(10, 6.5))

rng = np.random.default_rng(42)
mask_tr = x_tr <= 20
mask_te = x_te <= 20
idx_tr = np.where(mask_tr)[0]
n_plot = min(4000, len(idx_tr))
idx_plot = rng.choice(idx_tr, size=n_plot, replace=False)

ax.scatter(x_tr[idx_plot], y_tr[idx_plot], s=10, alpha=0.4,
           color="#dc2626", edgecolors="none", zorder=2, label="Train")
ax.scatter(x_te[mask_te], y_te[mask_te], s=14, alpha=0.55,
           color="#d97706", edgecolors="none", zorder=3, label="Test (Sep)")

xg = np.linspace(0, 20, 300)
yg = beta[0] * xg + beta[1]
ax.plot(xg, yg, color="black", lw=2.2, zorder=4,
        label=f"OLS: {beta[0]:+.4f}x {beta[1]:+.4f}")

ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.5, zorder=1)

ax.set_xlim(0, 20)
ax.set_ylim(-25, 30)

ax.set_xlabel("Roll Spread Estimate (bps)", fontsize=12)
ax.set_ylabel("Slippage (bps)", fontsize=12)
ax.set_title("COIN OLS Fit", fontsize=14, fontweight="bold")
ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
ax.grid(True, alpha=0.18)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("coin_signed_ols.png", dpi=150, bbox_inches="tight")
print("Saved -> coin_signed_ols.png")
