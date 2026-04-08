import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── 1. Load buy block trades ──────────────────────────────────────────────────
bt = pd.read_parquet("data/block_trades.parquet",
    columns=["date", "timestamp_ns", "side_label", "impact_vwap_bps"])
buys = bt[(bt.side_label == "buy") & bt.impact_vwap_bps.notna()].copy()
buys = buys.sort_values(["date", "timestamp_ns"]).reset_index(drop=True)
print(f"{len(buys):,} AAPL buy block trades across {buys.date.nunique()} days")

ONE_MIN_NS = 60 * 1_000_000_000

# ── 2. Vectorized trailing features per day via prefix sums ──────────────────
#
# For a window of ticks at indices [lo, hi):
#   - volume       = cumsum_size[hi] - cumsum_size[lo]
#   - diffs        = price[1:] - price[:-1]  (length N-1 for N ticks)
#   - n_diffs      = (hi-1) - lo  = hi - lo - 1
#   - sum_d        = cumsum_d[hi-1]   - cumsum_d[lo]
#   - sum_d2       = cumsum_d2[hi-1]  - cumsum_d2[lo]
#   - std(diffs)   = sqrt((sum_d2 - sum_d**2/n_diffs) / (n_diffs-1))

results = []

for date, grp in buys.groupby("date"):
    tick_path = f"data/AAPL/{date}.parquet"
    if not os.path.exists(tick_path):
        continue
    ticks = pd.read_parquet(tick_path, columns=["sip_timestamp", "price", "size"])
    ticks = ticks.sort_values("sip_timestamp").reset_index(drop=True)

    ts  = ticks["sip_timestamp"].to_numpy(dtype=np.int64)
    px  = ticks["price"].to_numpy(dtype=np.float64)
    sz  = ticks["size"].to_numpy(dtype=np.int64)
    N   = len(ts)

    # prefix sums (length N+1, index 0 = 0)
    cum_sz  = np.empty(N + 1, dtype=np.int64);   cum_sz[0]  = 0; np.cumsum(sz, out=cum_sz[1:])
    diffs   = np.diff(px)                           # length N-1
    cum_d   = np.empty(N, dtype=np.float64);  cum_d[0]  = 0.0; np.cumsum(diffs, out=cum_d[1:])
    cum_d2  = np.empty(N, dtype=np.float64);  cum_d2[0] = 0.0; np.cumsum(diffs**2, out=cum_d2[1:])

    block_ts = grp["timestamp_ns"].to_numpy(dtype=np.int64)
    lo_idx   = np.searchsorted(ts, block_ts - ONE_MIN_NS, side="left")
    hi_idx   = np.searchsorted(ts, block_ts,              side="left")

    # volume: shares in [lo, hi)
    vol_shares = (cum_sz[hi_idx] - cum_sz[lo_idx]).astype(float)

    # volatility: std of price diffs in window [lo, hi)
    #   diffs in window are at diff-indices [lo, hi-1)
    #   diff-prefix-sum index: cum_d[0..N], where cum_d[k] = sum(diffs[0:k])
    n_diffs  = (hi_idx - lo_idx - 1).astype(float)   # may be 0 or negative
    sum_d    = cum_d[hi_idx - 1]  - cum_d[lo_idx]     # hi_idx-1 safe: hi<=N, so hi-1<=N-1
    sum_d2   = cum_d2[hi_idx - 1] - cum_d2[lo_idx]

    # guard: need >=2 ticks (>=1 diff) for std; n_diffs>=2 for ddof=1
    mask_ok  = (n_diffs >= 2) & (hi_idx > 0)
    variance = np.where(
        mask_ok,
        (sum_d2 - sum_d**2 / np.where(n_diffs > 0, n_diffs, 1)) /
        np.where(n_diffs > 1, n_diffs - 1, 1),
        np.nan
    )
    variance = np.where(variance < 0, 0.0, variance)   # float noise guard
    trail_std = np.where(mask_ok, np.sqrt(variance), np.nan)

    # set vol_shares to nan when window is empty
    vol_shares = np.where(hi_idx > lo_idx, vol_shares, np.nan)

    tmp = grp[["impact_vwap_bps"]].copy()
    tmp["trail_vol_px"]     = trail_std   # price-change std in $ per share
    tmp["trail_vol_shares"] = vol_shares
    results.append(tmp)
    print(f"  {date}: {len(grp):4d} buys, tick rows={N:,}")

buys_feat = pd.concat(results)
print(f"\nTotal: {len(buys_feat):,} trades")

AAPL_MID = 190.0
buys_feat["trail_vol_bps"] = buys_feat["trail_vol_px"] / AAPL_MID * 1e4

valid = buys_feat.dropna(subset=["trail_vol_bps", "trail_vol_shares", "impact_vwap_bps"])
print(f"Trades with full features: {len(valid):,}")
print(f"  trail_vol_bps  — median={valid.trail_vol_bps.median():.2f}  p99={valid.trail_vol_bps.quantile(0.99):.2f}")
print(f"  trail_vol_sh   — median={valid.trail_vol_shares.median():.0f}  p99={valid.trail_vol_shares.quantile(0.99):.0f}")

# ── 3. Helper: binned mean ────────────────────────────────────────────────────
CLIP = 15

def make_bins(df, xcol, q=30):
    df = df.copy()
    df["impact_c"] = df["impact_vwap_bps"].clip(-CLIP, CLIP)
    df["bin"] = pd.qcut(df[xcol], q=q, duplicates="drop")
    binned = df.groupby("bin", observed=True).agg(
        x_mid=(xcol, "median"),
        mean_impact=("impact_vwap_bps", "mean"),
    ).reset_index()
    binned["mean_impact"] = binned["mean_impact"].clip(-CLIP, CLIP)
    return df, binned

# ── 4. Plot ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# — Panel A: impact vs trailing volatility —
df_v, bin_v = make_bins(valid, "trail_vol_bps")
sample_v = df_v.sample(n=min(10_000, len(df_v)), random_state=42)

ax = axes[0]
ax.scatter(sample_v["trail_vol_bps"], sample_v["impact_c"],
           alpha=0.10, s=7, color="#2563eb", linewidths=0,
           label=f"Trades (10k sample of {len(df_v):,})")
ax.plot(bin_v["x_mid"], bin_v["mean_impact"],
        color="#f59e0b", lw=2.2, marker="o", ms=5, zorder=5,
        label="Binned mean (all trades)")
ax.axhline(0, color="black", lw=1, ls="--", alpha=0.55)
ax.set_xlabel("Trailing 1-min volatility (price-change std, bps)", fontsize=11)
ax.set_ylabel("VWAP-bar impact (bps, clipped ±15)", fontsize=11)
ax.set_title("Impact vs Trailing Volatility", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

# — Panel B: impact vs trailing volume —
df_s, bin_s = make_bins(valid, "trail_vol_shares")
sample_s = df_s.sample(n=min(10_000, len(df_s)), random_state=42)

ax = axes[1]
ax.scatter(sample_s["trail_vol_shares"], sample_s["impact_c"],
           alpha=0.10, s=7, color="#2563eb", linewidths=0,
           label=f"Trades (10k sample of {len(df_s):,})")
ax.plot(bin_s["x_mid"], bin_s["mean_impact"],
        color="#f59e0b", lw=2.2, marker="o", ms=5, zorder=5,
        label="Binned mean (all trades)")
ax.axhline(0, color="black", lw=1, ls="--", alpha=0.55)
ax.set_xscale("log")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(
    lambda v, _: f"{v/1e6:.1f}M" if v >= 1e6 else f"{v/1e3:.0f}k"))
ax.set_xlabel("Trailing 1-min volume (shares)", fontsize=11)
ax.set_ylabel("VWAP-bar impact (bps, clipped ±15)", fontsize=11)
ax.set_title("Impact vs Trailing Volume", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

fig.suptitle("AAPL buy block trades — impact drivers", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("aapl_impact_drivers.png", dpi=150, bbox_inches="tight")
print("saved -> aapl_impact_drivers.png")
