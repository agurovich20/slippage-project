"""
For each block trade, find the immediately adjacent trades in the full sequence
(no time window) and compute:
    impact_bps = (post_price_adj - pre_price_adj) / pre_price_adj * 10000

This measures how much the block actually moved the market, uncontaminated by
the ±1-second window used in slippage_bps.

Output: adds pre_adj, post_adj, impact_bps to block_trades.parquet
        saves adjacent_impact.png
"""

import gc
import logging
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_DIR = Path("data")
BLK_PATH = DATA_DIR / "block_trades.parquet"


# ---------------------------------------------------------------------------
# Adjacent-price lookup for one ticker-day file
# ---------------------------------------------------------------------------

def adjacent_prices(trade_path: Path, blk_ts: np.ndarray,
                    blk_px: np.ndarray, blk_sz: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Load one trade file, sort by (participant_timestamp, sequence_number),
    find each block trade's position via searchsorted, return
    (pre_prices, post_prices) as float64 arrays (NaN where unavailable).
    """
    tbl = pq.read_table(
        trade_path,
        columns=["participant_timestamp", "price", "size", "sequence_number"],
    )
    ts_arr    = tbl.column("participant_timestamp").to_pylist()
    seq_arr   = tbl.column("sequence_number").to_pylist()
    price_col = tbl.column("price")
    size_col  = tbl.column("size")

    # Sort order by (timestamp, sequence_number)
    order = np.lexsort((
        np.array(seq_arr,  dtype=np.int64),
        np.array(ts_arr,   dtype=np.int64),
    ))

    ts_s    = np.array(ts_arr,            dtype=np.int64)[order]
    price_s = price_col.to_pylist()
    price_s = np.array(price_s,           dtype=np.float64)[order]
    size_s  = size_col.to_pylist()
    size_s  = np.array(size_s,            dtype=np.int64)[order]
    n       = len(ts_s)

    del tbl, ts_arr, seq_arr, price_col, size_col, order
    gc.collect()

    # Vectorised searchsorted for all block trades at once
    lo_arr = np.searchsorted(ts_s, blk_ts, side="left")
    hi_arr = np.searchsorted(ts_s, blk_ts, side="right")

    pos_arr = lo_arr.copy()  # will be refined for multi-trade timestamps

    # Resolve multi-trade timestamps (rare) by matching price + size
    for i in np.where(hi_arr - lo_arr != 1)[0]:
        lo, hi = lo_arr[i], hi_arr[i]
        if lo >= hi:
            continue
        hits = np.where(
            (price_s[lo:hi] == blk_px[i]) & (size_s[lo:hi] == blk_sz[i])
        )[0]
        if len(hits):
            pos_arr[i] = lo + hits[0]

    # Adjacent prices
    pre_pos  = pos_arr - 1
    post_pos = pos_arr + 1

    safe_pre  = np.clip(pre_pos,  0, n - 1)
    safe_post = np.clip(post_pos, 0, n - 1)

    pre_prices  = np.where(pre_pos  >= 0,     price_s[safe_pre],  np.nan)
    post_prices = np.where(post_pos <  n,     price_s[safe_post], np.nan)

    del ts_s, price_s, size_s
    gc.collect()

    return pre_prices, post_prices


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    blocks = pq.read_table(BLK_PATH).to_pandas()
    log.info("Loaded %d block trades", len(blocks))

    blocks.sort_values(["ticker", "date", "timestamp_ns"], inplace=True, ignore_index=True)

    pre_adj  = np.full(len(blocks), np.nan)
    post_adj = np.full(len(blocks), np.nan)

    for (ticker, date_str), grp in blocks.groupby(["ticker", "date"]):
        trade_path = DATA_DIR / ticker / f"{date_str}.parquet"
        if not trade_path.exists():
            log.warning("Missing %s", trade_path)
            continue

        idx      = grp.index.to_numpy()
        blk_ts   = grp["timestamp_ns"].to_numpy(np.int64)
        blk_px   = grp["price"].to_numpy(np.float64)
        blk_sz   = grp["size"].to_numpy(np.int64)

        pre, post = adjacent_prices(trade_path, blk_ts, blk_px, blk_sz)
        pre_adj[idx]  = pre
        post_adj[idx] = post

        n_valid = (~np.isnan(pre) & ~np.isnan(post)).sum()
        log.info("  %s %s  %d/%d with both neighbors", ticker, date_str, n_valid, len(grp))

    blocks["pre_adj"]    = pre_adj
    blocks["post_adj"]   = post_adj
    with np.errstate(invalid="ignore", divide="ignore"):
        blocks["impact_bps"] = (
            (blocks["post_adj"] - blocks["pre_adj"]) / blocks["pre_adj"] * 10_000
        )

    pq.write_table(pa.Table.from_pandas(blocks, preserve_index=False), BLK_PATH)
    log.info("Saved block_trades.parquet with pre_adj / post_adj / impact_bps")

    # ── Analytics ─────────────────────────────────────────────────────────
    ana = blocks[
        (blocks["side_label"] != "unknown") &
        blocks["impact_bps"].notna()
    ].copy()

    bins   = [200_000, 500_000, 1_000_000, 5_000_000, np.inf]
    dv_labels = ["$200k-500k", "$500k-1M", "$1M-5M", "$5M+"]
    ana["dv_bucket"] = pd.cut(ana["dollar_value"], bins=bins, labels=dv_labels, right=False)

    def tbl(g):
        s = g["impact_bps"]
        return pd.Series({
            "n":         len(g),
            "median":    s.median(),
            "mean":      s.mean(),
            "p25":       s.quantile(0.25),
            "p75":       s.quantile(0.75),
            "p95":       s.quantile(0.95),
            "std":       s.std(),
        })

    print(f"\n{'='*65}")
    print("ADJACENT-TRADE IMPACT BY SIDE  (bps)")
    print(f"{'='*65}")
    by_side = ana.groupby("side_label", observed=True).apply(tbl, include_groups=False)
    print(by_side.to_string(float_format="{:.3f}".format))

    print(f"\n{'='*65}")
    print("ADJACENT-TRADE IMPACT BY DOLLAR BUCKET  (bps, both sides)")
    print(f"{'='*65}")
    by_dv = ana.groupby("dv_bucket", observed=True).apply(tbl, include_groups=False)
    print(by_dv.to_string(float_format="{:.3f}".format))

    print(f"\n{'='*65}")
    print("MEDIAN IMPACT (bps) — bucket x side")
    print(f"{'='*65}")
    pivot = ana.pivot_table(
        values="impact_bps", index="dv_bucket", columns="side_label",
        aggfunc="median", observed=True,
    )[["buy", "sell"]]
    print(pivot.to_string(float_format="{:.4f}".format))

    # ── Histogram ─────────────────────────────────────────────────────────
    CLIP = 30   # bps — cuts <0.1% of extremes for readability

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("AAPL Block-Trade Adjacent-Trade Impact", fontsize=12, fontweight="bold")

    # Left: overlapping histograms, buy vs sell
    ax = axes[0]
    COLORS = {"buy": "#3a86ff", "sell": "#e63946"}
    for side, color in COLORS.items():
        data = ana.loc[ana["side_label"] == side, "impact_bps"].clip(-CLIP, CLIP)
        ax.hist(data, bins=120, range=(-CLIP, CLIP),
                color=color, alpha=0.55, label=side, density=True)

    ax.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.6)
    ax.set_xlabel("Impact (bps, clipped ±30)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Buy vs Sell", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # Right: by dollar bucket, stacked/overlapping on log-y
    ax2 = axes[1]
    BUCKET_COLORS = ["#4878d0", "#ee854a", "#6acc65", "#d65f5f"]
    for bucket, color in zip(dv_labels, BUCKET_COLORS):
        data = ana.loc[ana["dv_bucket"] == bucket, "impact_bps"].clip(-CLIP, CLIP)
        if len(data) == 0:
            continue
        ax2.hist(data, bins=80, range=(-CLIP, CLIP),
                 color=color, alpha=0.55, label=bucket, density=True)

    ax2.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.6)
    ax2.set_yscale("log")
    ax2.set_xlabel("Impact (bps, clipped ±30)", fontsize=11)
    ax2.set_ylabel("Density (log)", fontsize=11)
    ax2.set_title("By dollar bucket", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("adjacent_impact.png", dpi=150, bbox_inches="tight")
    log.info("Plot saved -> adjacent_impact.png")


if __name__ == "__main__":
    main()
