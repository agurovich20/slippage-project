"""
Standalone ICE plot for roll_vol_500 from Random Forest MSE model.

Output: ice_roll_vol.png
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
from sklearn.ensemble import RandomForestRegressor

FEATURES = [
    "dollar_value", "log_dollar_value", "participation_rate",
    "roll_spread_500", "roll_vol_500", "exchange_id",
]
VOL_IDX = FEATURES.index("roll_vol_500")

# -- Load and train ------------------------------------------------------------
df_tr = pd.read_parquet("data/lit_buy_features_v2.parquet")
df_te = pd.read_parquet("data/lit_buy_features_v2_sep.parquet")
df_tr["abs_impact"] = df_tr["impact_vwap_bps"].abs()
df_te["abs_impact"] = df_te["impact_vwap_bps"].abs()
df_tr = df_tr.sort_values("date").reset_index(drop=True)

X_tr = df_tr[FEATURES].to_numpy(dtype=np.float64)
y_tr = df_tr["abs_impact"].to_numpy(dtype=np.float64)
X_te = df_te[FEATURES].to_numpy(dtype=np.float64)

print("Training RF...", flush=True)
model = RandomForestRegressor(
    max_depth=30, n_estimators=50, min_samples_leaf=20,
    max_features=0.33, bootstrap=False, random_state=42, n_jobs=-1,
)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model.fit(X_tr, y_tr)

# -- Compute ICE ---------------------------------------------------------------
print("Computing ICE curves...", flush=True)
rng = np.random.default_rng(42)
n_ice = 100
ice_idx = rng.choice(len(X_te), size=n_ice, replace=False)
X_ice = X_te[ice_idx]

vol_vals = X_te[:, VOL_IDX]
grid = np.linspace(np.percentile(vol_vals, 1), np.percentile(vol_vals, 99), 200)

ice = np.zeros((n_ice, len(grid)))
for gi, gval in enumerate(grid):
    X_mod = X_ice.copy()
    X_mod[:, VOL_IDX] = gval
    ice[:, gi] = np.maximum(model.predict(X_mod), 0.0)

pdp_mean = ice.mean(axis=0)

# -- Plot ----------------------------------------------------------------------
print("Plotting...", flush=True)
fig, ax = plt.subplots(figsize=(10, 7))

for row in range(n_ice):
    ax.plot(grid, ice[row], color="#7c3aed", alpha=0.08, lw=0.8)

ax.plot(grid, pdp_mean, color="black", lw=2.5, label="PDP (mean)", zorder=5)

ax.set_xlabel("500 Trade Rolling Volatility (bps)", fontsize=12)
ax.set_ylabel("|slippage| (bps)", fontsize=12)
ax.set_title("AAPL Random Forest ICE: Rolling Volatility",
             fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.15)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xlim(grid[0], grid[-1])

plt.tight_layout()
plt.savefig("ice_roll_vol.png", dpi=150, bbox_inches="tight")
print("Saved -> ice_roll_vol.png")
