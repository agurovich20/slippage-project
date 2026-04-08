"""
Forward price impact of AAPL lit buy sweep clusters.

For each cluster with dv >= $200k, find the first trade in the full tick stream
(lit + dark) that occurs >= T seconds after the cluster's last fill timestamp.
Forward impact = (forward_price - first_fill_price) / first_fill_price * 10,000 bps.

Horizons: T = 1s and T = 5s.

Uses the same clustering logic as sweep_clusters.py.  The forward price lookup
keeps the full (lit + dark) sorted tick arrays alive so searchsorted can find
the first post-horizon trade regardless of exchange.

Output: aapl_sweep_forward.png  (two panels: 1s and 5s)
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"]      = "1"

import gc
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from glob import glob

DARK_IDS   = {4, 6, 16, 62, 201, 202, 203}
GAP_NS     = 100_000_000          # 100 ms cluster gap
MIN_DOLLAR = 200_000
NS_1S      = 1_000_000_000        # 1 second in ns
NS_5S      = 5_000_000_000        # 5 seconds in ns

all_clusters = []
tick_files   = sorted(glob("data/AAPL/*.parquet"))

for path in tick_files:
    date = os.path.basename(path).replace(".parquet", "")

    # ── Load ALL trades (lit + dark) for forward price lookup ─────────────────
    tbl    = pq.read_table(path, columns=["sip_timestamp", "price", "size", "exchange"])
    ts_raw = tbl.column("sip_timestamp").to_numpy(zero_copy_only=False)
    px_raw = tbl.column("price").to_numpy(zero_copy_only=False).astype(np.float32)
    sz_raw = tbl.column("size").to_numpy(zero_copy_only=False).astype(np.int32)
    ex_raw = tbl.column("exchange").to_numpy(zero_copy_only=False).astype(np.int16)
    del tbl; gc.collect()

    # Sort all trades once
    order  = np.argsort(ts_raw, kind="stable")
    ts_all = ts_raw[order]     # ALL trades, sorted — kept for forward lookup
    px_all = px_raw[order]
    sz_all = sz_raw[order]
    ex_all = ex_raw[order]
    del ts_raw, px_raw, sz_raw, ex_raw, order; gc.collect()

    N_ALL = len(ts_all)

    # ── Lit-only arrays for clustering ────────────────────────────────────────
    lit_mask = ~np.isin(ex_all, list(DARK_IDS))
    ts = ts_all[lit_mask]
    px = px_all[lit_mask]
    sz = sz_all[lit_mask]
    ex = ex_all[lit_mask]

    if len(ts) < 2:
        del ts_all, px_all, sz_all, ex_all, lit_mask; gc.collect()
        continue

    # ── Tick test on lit trades ───────────────────────────────────────────────
    dp        = np.diff(px, prepend=px[0])
    raw_dir   = np.sign(dp).astype(float)
    raw_dir[raw_dir == 0] = np.nan
    direction = pd.Series(raw_dir).ffill().fillna(1.0).to_numpy()

    buy_mask = direction > 0
    b_ts = ts[buy_mask];  b_px = px[buy_mask]
    b_sz = sz[buy_mask];  b_ex = ex[buy_mask]
    n_buys = buy_mask.sum()

    if n_buys < 2:
        del ts_all, px_all, sz_all, ex_all; gc.collect()
        continue

    # ── Cluster assignment ────────────────────────────────────────────────────
    gap_arr     = np.empty(n_buys, dtype=np.int64)
    gap_arr[0]  = GAP_NS + 1
    gap_arr[1:] = np.diff(b_ts)
    ex_chg      = np.concatenate([[True], b_ex[1:] != b_ex[:-1]])
    new_cls     = (gap_arr > GAP_NS) | ex_chg
    cls_id      = new_cls.cumsum()

    # ── Per-cluster aggregation ───────────────────────────────────────────────
    unique_cls, first_pos = np.unique(cls_id, return_index=True)
    last_pos  = np.concatenate([first_pos[1:] - 1, [n_buys - 1]])

    dv_arr       = b_px.astype(np.float64) * b_sz
    cum_dv       = np.concatenate([[0.0], np.cumsum(dv_arr)])
    cum_sz       = np.concatenate([[0],   np.cumsum(b_sz)])

    cluster_dv   = cum_dv[last_pos + 1] - cum_dv[first_pos]
    cluster_sh   = cum_sz[last_pos + 1] - cum_sz[first_pos]
    cluster_n    = last_pos - first_pos + 1
    cluster_fp   = b_px[first_pos].astype(np.float64)
    cluster_lp   = b_px[last_pos].astype(np.float64)
    cluster_fts  = b_ts[first_pos]    # first-fill timestamp
    cluster_lts  = b_ts[last_pos]     # last-fill timestamp
    cluster_span = (cluster_lts - cluster_fts) / 1_000_000   # ms
    with np.errstate(invalid="ignore", divide="ignore"):
        cluster_impact = (cluster_lp - cluster_fp) / cluster_fp * 10_000

    # ── Forward price lookup (ALL trades, vectorised) ─────────────────────────
    def forward_impact(horizon_ns):
        """
        For each cluster, find the first trade in ts_all at
        >= cluster_lts + horizon_ns, then compute
        (that_price - first_fill_price) / first_fill_price * 10_000.
        Returns float64 array of length n_clusters; NaN where no trade found.
        """
        targets = cluster_lts + horizon_ns
        idx     = np.searchsorted(ts_all, targets, side="left")   # first trade >= target
        valid   = idx < N_ALL
        safe    = np.where(valid, idx, 0)                          # avoid OOB index
        fwd_px  = np.where(valid, px_all[safe].astype(np.float64), np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            fwd_imp = np.where(
                valid,
                (fwd_px - cluster_fp) / cluster_fp * 10_000,
                np.nan,
            )
        return fwd_imp

    fwd_1s = forward_impact(NS_1S)
    fwd_5s = forward_impact(NS_5S)

    del ts_all, px_all, sz_all, ex_all; gc.collect()

    df_day = pd.DataFrame({
        "date":           date,
        "n_fills":        cluster_n,
        "total_shares":   cluster_sh,
        "dollar_value":   cluster_dv,
        "first_price":    cluster_fp,
        "last_price":     cluster_lp,
        "impact_bps":     cluster_impact,
        "time_span_ms":   cluster_span,
        "fwd_impact_1s":  fwd_1s,
        "fwd_impact_5s":  fwd_5s,
    })
    all_clusters.append(df_day)
    n_valid_1s = np.sum(~np.isnan(fwd_1s))
    n_valid_5s = np.sum(~np.isnan(fwd_5s))
    print(f"  {date}: {len(df_day):5,} clusters  "
          f"fwd_1s={n_valid_1s:,}  fwd_5s={n_valid_5s:,}")

# ── Combine and filter ─────────────────────────────────────────────────────────
clusters = pd.concat(all_clusters, ignore_index=True)
large    = clusters[clusters["dollar_value"] >= MIN_DOLLAR].copy()

print(f"\n{'='*60}")
print(f"Clusters (dv >= $200k)         : {len(large):,}")
print(f"  with valid 1s forward price  : {large['fwd_impact_1s'].notna().sum():,}")
print(f"  with valid 5s forward price  : {large['fwd_impact_5s'].notna().sum():,}")
print(f"{'='*60}")

for label, col in [("1s forward impact", "fwd_impact_1s"),
                   ("5s forward impact", "fwd_impact_5s"),
                   ("within-cluster impact", "impact_bps")]:
    s = large[col].dropna()
    print(f"\n{label} (bps):")
    print(f"  Mean   : {s.mean():+.4f}")
    print(f"  Median : {s.median():+.4f}")
    print(f"  Std    : {s.std():.4f}")


# ── Plot ───────────────────────────────────────────────────────────────────────
CLIP   = 15
N_BINS = 30

def make_panel(ax, dv, impact, clip, n_bins, color_scatter, color_bin,
               title, ylabel):
    valid = ~np.isnan(impact)
    dv_v  = dv[valid]
    imp_v = impact[valid]
    imp_c = np.clip(imp_v, -clip, clip)

    # quantile bins on dollar value
    dv_s  = pd.Series(dv_v)
    bins  = pd.qcut(dv_s, q=n_bins, duplicates="drop")
    binned = (
        pd.DataFrame({"dv": dv_v, "imp": imp_v, "bin": bins})
        .groupby("bin", observed=True)
        .agg(dv_mid=("dv", "median"), mean_imp=("imp", "mean"), n=("imp", "count"))
        .reset_index(drop=True)
    )
    binned["mean_imp_c"] = binned["mean_imp"].clip(-clip, clip)

    # sample for scatter (cap at 10k)
    rng   = np.random.default_rng(42)
    idx_s = rng.choice(len(dv_v), size=min(10_000, len(dv_v)), replace=False)

    ax.scatter(dv_v[idx_s], imp_c[idx_s],
               alpha=0.08, s=6, color=color_scatter, linewidths=0,
               label=f"Clusters ({len(dv_v):,} with valid forward price)")

    ax.plot(binned["dv_mid"], binned["mean_imp_c"],
            color=color_bin, lw=2.2, zorder=5)
    ax.scatter(binned["dv_mid"], binned["mean_imp_c"],
               color=color_bin, s=55, zorder=6,
               edgecolors="white", linewidths=0.8,
               label=f"Binned mean ({n_bins} quantile bins)")

    ax.axhline(0, color="black", lw=1.0, ls="--", alpha=0.55)
    ax.set_xscale("log")
    ax.set_xlabel("Cluster dollar value  (log scale)", fontsize=10.5)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: (f"${v/1e6:.1f}M" if v >= 1e6 else f"${v/1e3:.0f}k")))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # annotate mean and median
    s = pd.Series(imp_v)
    ax.text(0.97, 0.97,
            f"mean  {s.mean():+.3f} bps\nmedian {s.median():+.3f} bps",
            transform=ax.transAxes, fontsize=8.5, family="monospace",
            ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="#94a3b8", alpha=0.9))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

make_panel(
    ax1,
    large["dollar_value"].to_numpy(),
    large["fwd_impact_1s"].to_numpy(),
    clip=CLIP, n_bins=N_BINS,
    color_scatter="#2563eb", color_bin="#f59e0b",
    title="1-second forward impact\n"
          "(next trade >= 1s after last fill vs first fill price)",
    ylabel="forward_impact_bps  (clipped ±15)",
)

make_panel(
    ax2,
    large["dollar_value"].to_numpy(),
    large["fwd_impact_5s"].to_numpy(),
    clip=CLIP, n_bins=N_BINS,
    color_scatter="#16a34a", color_bin="#dc2626",
    title="5-second forward impact\n"
          "(next trade >= 5s after last fill vs first fill price)",
    ylabel="forward_impact_bps  (clipped ±15)",
)

fig.suptitle(
    "AAPL lit buy sweep clusters — forward price impact vs cluster size\n"
    f"100-ms window, same-exchange clusters  |  dv >= $200k  |  {len(large):,} clusters",
    fontsize=12, fontweight="bold", y=1.01,
)
plt.tight_layout()
plt.savefig("aapl_sweep_forward.png", dpi=150, bbox_inches="tight")
print("\nsaved -> aapl_sweep_forward.png")
