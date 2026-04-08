"""
COIN block trades: raw slippage (pre-tick-test) vs trade size with OLS fit.
slippage_bps = (post_price - pre_price) / pre_price * 10_000  — no sign flipping.
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Load block trades, filter to COIN
blocks = pd.read_parquet("data/block_trades.parquet")
coin = blocks[blocks["ticker"] == "COIN"].dropna(subset=["slippage_bps"]).copy()
print(f"COIN block trades with slippage: {len(coin):,}")

x = coin["dollar_value"].values / 1e6  # trade size in $M for readability
y = coin["slippage_bps"].values

# OLS fit: slippage = beta0 + beta1 * dollar_value
X = np.column_stack([x, np.ones(len(x))])
beta, *_ = np.linalg.lstsq(X, y, rcond=None)
print(f"OLS: slippage = {beta[0]:+.4f} * size_M {beta[1]:+.4f}")

# Plot
fig, ax = plt.subplots(figsize=(10, 6.5))

ax.scatter(x, y, s=10, alpha=0.3, color="#dc2626", edgecolors="none", zorder=2,
           label=f"COIN blocks (n={len(coin):,})")

xg = np.linspace(x.min(), x.max(), 300)
yg = beta[0] * xg + beta[1]
ax.plot(xg, yg, color="black", lw=2.2, zorder=4,
        label=f"OLS: {beta[0]:+.4f}x {beta[1]:+.4f}")

ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.5, zorder=1)

ax.set_xlabel("Trade Size ($M)", fontsize=12)
ax.set_ylabel("Slippage (bps)", fontsize=12)
ax.set_title("COIN OLS Fit", fontsize=14, fontweight="bold")
ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
ax.grid(True, alpha=0.18)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Cap y-axis at 1st/99th percentile
y_lo, y_hi = np.percentile(y, [1, 99])
margin = (y_hi - y_lo) * 0.1
ax.set_ylim(y_lo - margin, y_hi + margin)

plt.tight_layout()
plt.savefig("coin_raw_ols.png", dpi=150, bbox_inches="tight")
print("Saved -> coin_raw_ols.png")
