import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import gc
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from glob import glob

DARK_IDS     = {4, 6, 16, 62, 201, 202, 203}
GAP_NS       = 100_000_000        # 100 ms in nanoseconds
MIN_DOLLAR   = 200_000

# ── 1. Find sweep clusters per day ───────────────────────────────────────────
all_clusters = []
total_raw_lit_buys = 0

tick_files = sorted(glob("data/AAPL/*.parquet"))

for path in tick_files:
    date = os.path.basename(path).replace(".parquet", "")

    # Read with pyarrow, extract numpy arrays directly — no pandas DataFrame
    # Use float32/int32 to halve memory vs float64/int64
    tbl = pq.read_table(path, columns=["sip_timestamp", "price", "size", "exchange"])
    ts_all = tbl.column("sip_timestamp").to_numpy(zero_copy_only=False)          # int64
    px_all = tbl.column("price").to_numpy(zero_copy_only=False).astype(np.float32)
    sz_all = tbl.column("size").to_numpy(zero_copy_only=False).astype(np.int32)
    ex_all = tbl.column("exchange").to_numpy(zero_copy_only=False).astype(np.int16)
    del tbl; gc.collect()

    # Sort by sip_timestamp with numpy argsort (avoids pandas temp index allocs)
    order  = np.argsort(ts_all, kind="stable")
    ts_all = ts_all[order]
    px_all = px_all[order]
    sz_all = sz_all[order]
    ex_all = ex_all[order]
    del order; gc.collect()

    # ── Filter to lit exchanges only ─────────────────────────────────────────
    lit_mask = ~np.isin(ex_all, list(DARK_IDS))
    ts  = ts_all[lit_mask]
    px  = px_all[lit_mask]
    sz  = sz_all[lit_mask]
    ex  = ex_all[lit_mask]
    del ts_all, px_all, sz_all, ex_all, lit_mask; gc.collect()

    if len(ts) < 2:
        continue

    # ── Tick test: classify each lit trade as buy / sell ─────────────────────
    dp      = np.diff(px, prepend=px[0])       # first trade gets dp=0
    raw_dir = np.sign(dp).astype(float)
    raw_dir[raw_dir == 0] = np.nan             # zero-ticks: forward-fill
    # pandas ffill handles NaN streaks correctly
    direction = pd.Series(raw_dir).ffill().fillna(1.0).to_numpy()  # leading zeros → buy

    # ── Keep only uptick (buy) trades ────────────────────────────────────────
    buy_mask = direction > 0
    b_ts = ts[buy_mask]
    b_px = px[buy_mask]
    b_sz = sz[buy_mask]
    b_ex = ex[buy_mask]
    n_buys = buy_mask.sum()
    total_raw_lit_buys += n_buys

    if n_buys < 2:
        continue

    # ── Cluster: new cluster when gap > 100 ms OR exchange changes ───────────
    gap_arr  = np.empty(n_buys, dtype=np.int64)
    gap_arr[0] = GAP_NS + 1                   # first trade always starts new cluster
    gap_arr[1:] = np.diff(b_ts)
    ex_chg   = np.concatenate([[True], b_ex[1:] != b_ex[:-1]])
    new_cls  = (gap_arr > GAP_NS) | ex_chg
    cls_id   = new_cls.cumsum()               # 1-indexed cluster labels

    # ── Aggregate per cluster ─────────────────────────────────────────────────
    # Use numpy group-by via sorting (already sorted by time, so cls_id is monotone)
    unique_cls, first_pos = np.unique(cls_id, return_index=True)
    last_pos  = np.concatenate([first_pos[1:] - 1, [n_buys - 1]])

    dv_arr        = b_px * b_sz
    cum_dv        = np.concatenate([[0], np.cumsum(dv_arr)])
    cum_sz        = np.concatenate([[0], np.cumsum(b_sz)])
    cluster_dv    = cum_dv[last_pos + 1] - cum_dv[first_pos]
    cluster_sh    = cum_sz[last_pos + 1] - cum_sz[first_pos]
    cluster_n     = last_pos - first_pos + 1
    cluster_fp    = b_px[first_pos]
    cluster_lp    = b_px[last_pos]
    cluster_ts    = b_ts[first_pos]
    cluster_ex    = b_ex[first_pos]
    cluster_span  = (b_ts[last_pos] - b_ts[first_pos]) / 1_000_000   # ns → ms
    with np.errstate(invalid="ignore", divide="ignore"):
        cluster_impact = (cluster_lp - cluster_fp) / cluster_fp * 10_000

    df_day = pd.DataFrame({
        "date":          date,
        "cluster_ts":    cluster_ts,
        "exchange":      cluster_ex,
        "n_fills":       cluster_n,
        "total_shares":  cluster_sh,
        "dollar_value":  cluster_dv,
        "first_price":   cluster_fp,
        "last_price":    cluster_lp,
        "impact_bps":    cluster_impact,
        "time_span_ms":  cluster_span,
    })
    all_clusters.append(df_day)
    print(f"  {date}: {len(df_day):5,} clusters from {n_buys:6,} lit buy ticks")

clusters = pd.concat(all_clusters, ignore_index=True)
large    = clusters[clusters["dollar_value"] >= MIN_DOLLAR].copy()

print(f"\n{'='*55}")
print(f"Clusters found (dv >= $200k)  : {len(large):,}")
print(f"Average fills per cluster     : {large['n_fills'].mean():.2f}")
print(f"Median cluster dollar value   : ${large['dollar_value'].median():,.0f}")
print(f"{'='*55}")
print(f"  (all sizes before filter)   : {len(clusters):,}")
print(f"  Total lit buy ticks used    : {total_raw_lit_buys:,}")
print(f"  Multi-fill clusters (n>1)   : {(large['n_fills'] > 1).sum():,}  "
      f"({100*(large['n_fills']>1).mean():.1f}%)")
print(f"  Median fills per cluster    : {large['n_fills'].median():.0f}")
print(f"  Median time span (ms)       : {large['time_span_ms'].median():.2f}")
print(f"  Median total shares         : {large['total_shares'].median():,.0f}")

imp = large["impact_bps"]
print(f"\nCluster impact_bps (dv >= $200k):")
print(f"  Median : {imp.median():+.4f}")
print(f"  Mean   : {imp.mean():+.4f}")
print(f"  Std    : {imp.std():.4f}")

# ── 2. Scatter + binned means ─────────────────────────────────────────────────
CLIP    = 20
N_BINS  = 30

large["impact_c"] = large["impact_bps"].clip(-CLIP, CLIP)
large["dv_bin"]   = pd.qcut(large["dollar_value"], q=N_BINS, duplicates="drop")

binned = (
    large.groupby("dv_bin", observed=True)
    .agg(
        dv_mid     =("dollar_value", "median"),
        mean_impact=("impact_bps",   "mean"),
        n          =("impact_bps",   "count"),
    )
    .reset_index(drop=True)
)
binned["mean_impact_c"] = binned["mean_impact"].clip(-CLIP, CLIP)

sample = large.sample(n=min(10_000, len(large)), random_state=42)

fig, ax = plt.subplots(figsize=(10, 6))

# Raw scatter
ax.scatter(
    sample["dollar_value"], sample["impact_c"],
    alpha=0.09, s=7, color="#2563eb", linewidths=0,
    label=f"Sweep clusters (10k sample of {len(large):,})",
)

# Binned mean line + dots
ax.plot(
    binned["dv_mid"], binned["mean_impact_c"],
    color="#f59e0b", lw=2.2, zorder=5,
)
ax.scatter(
    binned["dv_mid"], binned["mean_impact_c"],
    color="#f59e0b", s=55, zorder=6, edgecolors="white", linewidths=0.8,
    label=f"Binned mean ({N_BINS} quantile bins)",
)

ax.axhline(0, color="black", lw=1, ls="--", alpha=0.55)
ax.set_xscale("log")
ax.set_xlabel("Cluster total dollar value  (log scale)", fontsize=12)
ax.set_ylabel(
    "Cluster impact_bps  =  (last_px − first_px) / first_px × 10,000\n"
    "(clipped ±20 bps)",
    fontsize=11,
)
ax.set_title(
    "AAPL lit buy sweep clusters — cluster_impact_bps vs cluster_dollar_value\n"
    f"100-ms window, same exchange  |  {len(large):,} clusters with dv ≥ $200k",
    fontsize=12, fontweight="bold",
)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(
    lambda v, _: (f"${v/1e6:.1f}M" if v >= 1e6 else f"${v/1e3:.0f}k")
))
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("aapl_sweep_impact.png", dpi=150, bbox_inches="tight")
print("saved -> aapl_sweep_impact.png")
