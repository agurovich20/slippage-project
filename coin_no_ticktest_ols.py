"""
Same as coin_signed_ols_plot.py but WITHOUT tick-test filtering.
Uses ALL COIN lit block trades (buy + sell + unknown), raw impact_vwap_bps.
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config (same as build_coin_features.py) ──────────────────────────────────
DARK_IDS   = {4, 6, 16, 62, 201, 202, 203}
COIN_MID   = 222.0
WINDOW     = 500
MIN_TICKS  = 10

def roll_spread_500(px_window):
    dp = np.diff(px_window.astype(np.float64))
    if len(dp) < 2:
        return np.nan
    cov_mat = np.cov(dp[1:], dp[:-1], ddof=1)
    cov1 = cov_mat[0, 1]
    if cov1 < 0:
        return 2.0 * np.sqrt(-cov1)
    return np.nan

# ── 1. Load ALL COIN lit block trades (no tick-test filter) ──────────────────
bt = pd.read_parquet(
    "data/block_trades.parquet",
    columns=["ticker", "date", "timestamp_ns", "price", "size",
             "dollar_value", "exchange", "impact_vwap_bps"],
    filters=[("ticker", "==", "COIN")],
)

# Lit only, must have impact
trades = bt[
    (~bt["exchange"].isin(DARK_IDS)) &
    bt["impact_vwap_bps"].notna()
].copy()
trades = trades.sort_values(["date", "timestamp_ns"]).reset_index(drop=True)
print(f"COIN lit block trades (all sides): {len(trades):,}")

# ── 2. Compute roll_spread_500 for each trade ───────────────────────────────
roll_spread_arr = np.full(len(trades), np.nan)

days_processed = 0
for date, grp in trades.groupby("date"):
    tick_path = f"data/COIN/{date}.parquet"
    if not os.path.exists(tick_path):
        continue

    ticks = pd.read_parquet(tick_path, columns=["sip_timestamp", "price"])
    ticks = ticks.sort_values("sip_timestamp").reset_index(drop=True)

    ts = ticks["sip_timestamp"].to_numpy(dtype=np.int64)
    px = ticks["price"].to_numpy(dtype=np.float64)

    block_ts = grp["timestamp_ns"].to_numpy(dtype=np.int64)
    hi_idx = np.searchsorted(ts, block_ts, side="left")

    for i, (idx, hi) in enumerate(zip(grp.index, hi_idx)):
        lo = max(0, hi - WINDOW)
        if hi - lo < MIN_TICKS:
            continue
        spread_raw = roll_spread_500(px[lo:hi])
        if not np.isnan(spread_raw):
            roll_spread_arr[idx] = spread_raw / COIN_MID * 1e4

    days_processed += 1
    if days_processed % 50 == 0:
        print(f"  Processed {days_processed} days...")

print(f"  Processed {days_processed} days total")

trades["roll_spread_500"] = roll_spread_arr

# ── 3. Drop NaN, split train/test ───────────────────────────────────────────
feat = trades.dropna(subset=["roll_spread_500", "impact_vwap_bps"]).copy()
feat = feat.reset_index(drop=True)
print(f"Rows after dropping NaN: {len(feat):,}")

train = feat[feat["date"] < "2024-09-01"].reset_index(drop=True)
test  = feat[feat["date"] >= "2024-09-01"].reset_index(drop=True)
print(f"Train: {len(train):,}  Test: {len(test):,}")

# ── 4. OLS fit ───────────────────────────────────────────────────────────────
x_tr = train["roll_spread_500"].values.astype(np.float64)
y_tr = train["impact_vwap_bps"].values.astype(np.float64)
x_te = test["roll_spread_500"].values.astype(np.float64)
y_te = test["impact_vwap_bps"].values.astype(np.float64)

X_tr = np.column_stack([x_tr, np.ones(len(x_tr))])
beta, *_ = np.linalg.lstsq(X_tr, y_tr, rcond=None)
print(f"OLS: impact = {beta[0]:+.4f} * spread {beta[1]:+.4f}")

# ── 5. Plot (same style/zoom as coin_signed_ols_plot.py) ─────────────────────
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
ax.set_title("Signed COIN OLS Fit", fontsize=14, fontweight="bold")
ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
ax.grid(True, alpha=0.18)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("coin_no_ticktest_ols.png", dpi=150, bbox_inches="tight")
print("Saved -> coin_no_ticktest_ols.png")
