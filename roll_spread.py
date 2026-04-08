import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from glob import glob

AAPL_MID = 190.0   # approx midprice for bps conversion

# ── 1. Roll (1984) spread estimator per day ───────────────────────────────────
# Cov1 = first-order autocovariance of price changes
# spread = 2 * sqrt(-Cov1)  if Cov1 < 0, else NaN

records = []
tick_files = sorted(glob("data/AAPL/*.parquet"))

for path in tick_files:
    date = os.path.basename(path).replace(".parquet", "")
    ticks = pd.read_parquet(path, columns=["sip_timestamp", "price"])
    ticks = ticks.sort_values("sip_timestamp")

    px = ticks["price"].to_numpy(dtype=np.float64)
    dp = np.diff(px)               # trade-to-trade price changes

    if len(dp) < 2:
        records.append({"date": date, "roll_cov1": np.nan,
                        "spread_raw": np.nan, "spread_bps": np.nan,
                        "n_ticks": len(px)})
        continue

    # first-order autocovariance: E[(dp_t)(dp_{t-1})]
    # use sample cov with mean correction (Roll uses zero-mean assumption;
    # we use the unbiased sample version)
    cov1 = np.cov(dp[1:], dp[:-1], ddof=1)[0, 1]

    if cov1 < 0:
        spread_raw = 2.0 * np.sqrt(-cov1)          # in $ per share
        spread_bps = spread_raw / AAPL_MID * 1e4
    else:
        spread_raw = np.nan
        spread_bps = np.nan

    records.append({
        "date":       date,
        "roll_cov1":  cov1,
        "spread_raw": spread_raw,
        "spread_bps": spread_bps,
        "n_ticks":    len(px),
    })
    print(f"  {date}: cov1={cov1:.6f}  spread={spread_bps:.3f} bps  ({len(px):,} ticks)")

spread_df = pd.DataFrame(records)
spread_df.to_parquet("data/aapl_daily_spread.parquet", index=False)
print(f"\nSaved {len(spread_df)} daily estimates -> data/aapl_daily_spread.parquet")
print(spread_df[["date","spread_bps"]].to_string(index=False))

# ── 2. Attach spread to buy block trades ─────────────────────────────────────
bt = pd.read_parquet("data/block_trades.parquet",
    columns=["date", "timestamp_ns", "side_label", "impact_vwap_bps"])
buys = bt[(bt.side_label == "buy") & bt.impact_vwap_bps.notna()].copy()

buys = buys.merge(spread_df[["date", "spread_bps"]], on="date", how="left")
valid = buys.dropna(subset=["spread_bps", "impact_vwap_bps"]).copy()
print(f"\n{len(valid):,} buys with valid spread estimate (of {len(buys):,} total)")

# ── 3. OLS: impact_bps = c1 * spread_bps + c2 ────────────────────────────────
y = valid["impact_vwap_bps"].to_numpy()
X = valid["spread_bps"].to_numpy()

# Design matrix [spread_bps | 1]
Xm = np.column_stack([X, np.ones(len(X))])

# beta = (X'X)^{-1} X'y
beta, *_ = np.linalg.lstsq(Xm, y, rcond=None)
c1, c2 = beta

y_hat    = Xm @ beta
residuals = y - y_hat
ss_res   = float(residuals @ residuals)
ss_tot   = float(((y - y.mean()) ** 2).sum())
r2       = 1.0 - ss_res / ss_tot

n, k = len(y), 2
s2   = ss_res / (n - k)                         # MSE
var_beta = s2 * np.linalg.inv(Xm.T @ Xm)
se   = np.sqrt(np.diag(var_beta))
t    = beta / se

print("\n" + "="*52)
print("OLS: impact_bps = c1 * spread_bps + c2")
print("="*52)
print(f"  c1 (spread_bps)  = {c1:+.6f}   SE={se[0]:.6f}   t={t[0]:.2f}")
print(f"  c2 (intercept)   = {c2:+.6f}   SE={se[1]:.6f}   t={t[1]:.2f}")
print(f"  R²               = {r2:.6f}")
print(f"  n                = {n:,}")
print("="*52)

# ── 4. Scatter + fitted line ──────────────────────────────────────────────────
CLIP = 15
sample = valid.sample(n=min(10_000, len(valid)), random_state=42)

x_line = np.linspace(valid["spread_bps"].min(), valid["spread_bps"].max(), 200)
y_line = c1 * x_line + c2

fig, ax = plt.subplots(figsize=(9, 6))

ax.scatter(sample["spread_bps"],
           sample["impact_vwap_bps"].clip(-CLIP, CLIP),
           alpha=0.08, s=7, color="#2563eb", linewidths=0,
           label=f"Trades (10k sample of {len(valid):,})")

ax.plot(x_line, y_line,
        color="#dc2626", lw=2.2, zorder=5,
        label=f"OLS fit: impact = {c1:+.4f}×spread {c2:+.4f}\n$R^2$={r2:.4f}")

ax.axhline(0, color="black", lw=1, ls="--", alpha=0.55)
ax.set_xlabel("Daily Roll spread estimate (bps)", fontsize=12)
ax.set_ylabel("VWAP-bar impact (bps, clipped ±15)", fontsize=12)
ax.set_title("AAPL buy block trades — impact vs Roll spread", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig("aapl_slippage_vs_spread.png", dpi=150, bbox_inches="tight")
print("saved -> aapl_slippage_vs_spread.png")
