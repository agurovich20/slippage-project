"""
For each block trade compute price impact two ways:

  Method 1 — adjacent tick (already in block_trades.parquet as impact_bps):
      (post_adj - pre_adj) / pre_adj * 10_000

  Method 2 — VWAP bar reference:
      (post_adj - vwap_ref) / vwap_ref * 10_000
      where vwap_ref is the VWAP of the 1-second bar ending just before
      the second containing the block trade.

Fetches 1-second bars from Polygon, caches at data/sec_aggs/TICKER/YYYY-MM-DD.parquet.
Compares variance, zero-impact fraction, and distributions of both methods.
"""

import time
import logging
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from polygon import RESTClient

from fetch_trades import API_KEY

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_DIR    = Path("data")
AGGS_DIR    = DATA_DIR / "sec_aggs"
BLK_PATH    = DATA_DIR / "block_trades.parquet"

SEC_SCHEMA = pa.schema([
    ("t",  pa.int64()),    # bar start, milliseconds since epoch
    ("vw", pa.float64()),  # VWAP
    ("o",  pa.float64()),
    ("h",  pa.float64()),
    ("l",  pa.float64()),
    ("c",  pa.float64()),
    ("v",  pa.float64()),  # volume (shares)
    ("n",  pa.int32()),    # number of trades in bar
])


# ---------------------------------------------------------------------------
# Fetch & cache 1-second bars
# ---------------------------------------------------------------------------

def fetch_sec_aggs(client: RESTClient, ticker: str, date_str: str, out_path: Path) -> int:
    aggs = client.get_aggs(
        ticker, 1, "second", date_str, date_str,
        adjusted=True, limit=100_000,
    )
    if not aggs:
        return 0
    rows = [{"t": a.timestamp, "vw": a.vwap, "o": a.open,  "h": a.high,
             "l": a.low,       "c": a.close, "v": a.volume, "n": a.transactions}
            for a in aggs]
    pq.write_table(pa.Table.from_pylist(rows, schema=SEC_SCHEMA), out_path, compression="snappy")
    return len(rows)


def ensure_sec_aggs(ticker_dates: list[tuple[str, str]]) -> None:
    client = RESTClient(api_key=API_KEY)
    AGGS_DIR.mkdir(parents=True, exist_ok=True)
    for ticker, date_str in ticker_dates:
        out_path = AGGS_DIR / ticker / f"{date_str}.parquet"
        if out_path.exists():
            log.info("SKIP  %s %s", ticker, date_str)
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        n = fetch_sec_aggs(client, ticker, date_str, out_path)
        log.info("FETCH %s %s  -> %d bars", ticker, date_str, n)
        time.sleep(0.1)


# ---------------------------------------------------------------------------
# VWAP lookup for one ticker-day
# ---------------------------------------------------------------------------

def load_vwap_index(ticker: str, date_str: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (bar_start_sec, vwap) as sorted int64/float64 arrays."""
    path = AGGS_DIR / ticker / f"{date_str}.parquet"
    tbl  = pq.read_table(path, columns=["t", "vw"])
    t_ms = tbl.column("t").to_pylist()
    vw   = tbl.column("vw").to_pylist()
    t_s  = np.array(t_ms, dtype=np.int64) // 1000   # ms -> seconds
    vw_a = np.array(vw,   dtype=np.float64)
    order = np.argsort(t_s)
    return t_s[order], vw_a[order]


def lookup_vwap_ref(bar_ts_s: np.ndarray, bar_vwap: np.ndarray,
                    blk_ts_ns: np.ndarray) -> np.ndarray:
    """
    For each block trade at blk_ts_ns, find the VWAP of the bar whose
    interval ends just before the trade's second:
        ref_bar_second = floor(blk_ts_ns / 1e9) - 1
    Falls back to the most recent earlier bar if that exact second had no trades.
    """
    ref_s = blk_ts_ns // 1_000_000_000 - 1   # second just before the block's second

    # searchsorted 'right' gives the insertion point after any matching bar_ts_s == ref_s
    # so idx - 1 is the last bar with start <= ref_s
    idx = np.searchsorted(bar_ts_s, ref_s, side="right") - 1

    valid = idx >= 0
    result = np.full(len(blk_ts_ns), np.nan)
    result[valid] = bar_vwap[idx[valid]]
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    blocks = pq.read_table(BLK_PATH).to_pandas()
    log.info("Loaded %d block trades", len(blocks))

    # Unique (ticker, date) pairs present in block_trades
    ticker_dates = (
        blocks[["ticker", "date"]].drop_duplicates()
        .apply(lambda r: (r["ticker"], r["date"]), axis=1)
        .tolist()
    )

    # ── Step 1: fetch / cache 1-second bars ───────────────────────────────
    log.info("Ensuring 1-second bars for %d ticker-days ...", len(ticker_dates))
    ensure_sec_aggs(ticker_dates)

    # ── Step 2: compute vwap_ref for each block trade ─────────────────────
    blocks.sort_values(["ticker", "date", "timestamp_ns"], inplace=True, ignore_index=True)
    vwap_ref = np.full(len(blocks), np.nan)

    for (ticker, date_str), grp in blocks.groupby(["ticker", "date"]):
        try:
            bar_ts_s, bar_vwap = load_vwap_index(ticker, date_str)
        except Exception as e:
            log.warning("Cannot load bars for %s %s: %s", ticker, date_str, e)
            continue

        vwap_ref[grp.index] = lookup_vwap_ref(
            bar_ts_s, bar_vwap, grp["timestamp_ns"].to_numpy(np.int64)
        )

    # ── Step 3: compute impacts ───────────────────────────────────────────
    with np.errstate(invalid="ignore", divide="ignore"):
        blocks["impact_vwap_bps"] = (
            (blocks["post_adj"] - vwap_ref) / vwap_ref * 10_000
        )

    # Method 1 already stored as impact_bps (adjacent-tick)
    # Save
    pq.write_table(pa.Table.from_pandas(blocks, preserve_index=False), BLK_PATH)
    log.info("Saved block_trades.parquet with impact_vwap_bps")

    # ── Step 4: comparison stats ──────────────────────────────────────────
    ana = blocks[blocks["side_label"] != "unknown"].copy()

    m1 = ana["impact_bps"].dropna()
    m2 = ana["impact_vwap_bps"].dropna()
    both = ana.dropna(subset=["impact_bps", "impact_vwap_bps"])

    print(f"\n{'='*60}")
    print(f"{'Metric':<30} {'Adj-tick':>12} {'VWAP-bar':>12}")
    print(f"{'='*60}")
    stats = [
        ("n with value",         m1.count(),                   m2.count()),
        ("Zero impact  (%)",     100*(m1==0).mean(),           100*(m2==0).mean()),
        ("Median (bps)",         m1.median(),                  m2.median()),
        ("Mean   (bps)",         m1.mean(),                    m2.mean()),
        ("Std    (bps)",         m1.std(),                     m2.std()),
        ("Variance (bps^2)",     m1.var(),                     m2.var()),
        ("IQR    (bps)",         m1.quantile(.75)-m1.quantile(.25),
                                 m2.quantile(.75)-m2.quantile(.25)),
        ("p5  (bps)",            m1.quantile(.05),             m2.quantile(.05)),
        ("p95 (bps)",            m1.quantile(.95),             m2.quantile(.95)),
        ("MAD (bps)",            (m1 - m1.median()).abs().median(),
                                 (m2 - m2.median()).abs().median()),
    ]
    for label, v1, v2 in stats:
        print(f"  {label:<28} {v1:>12.3f} {v2:>12.3f}")

    # By side
    print(f"\n{'='*60}")
    print("MEAN IMPACT BY SIDE (bps)")
    print(f"{'='*60}")
    print(f"  {'Side':<10} {'Adj-tick mean':>14} {'VWAP-bar mean':>14}")
    for side in ["buy", "sell"]:
        g = both[both["side_label"] == side]
        print(f"  {side:<10} {g['impact_bps'].mean():>14.3f} {g['impact_vwap_bps'].mean():>14.3f}")

    # By dollar bucket
    bins   = [200_000, 500_000, 1_000_000, 5_000_000, np.inf]
    labels = ["$200k-500k", "$500k-1M", "$1M-5M", "$5M+"]
    both = both.copy()
    both["dv_bucket"] = pd.cut(both["dollar_value"], bins=bins, labels=labels, right=False)

    print(f"\n{'='*60}")
    print("STD OF IMPACT BY DOLLAR BUCKET (bps) — lower = more precise")
    print(f"{'='*60}")
    print(f"  {'Bucket':<14} {'n':>7} {'Adj-tick std':>13} {'VWAP-bar std':>13} {'Zero%(adj)':>10} {'Zero%(vwap)':>11}")
    for bucket in labels:
        g = both[both["dv_bucket"] == bucket]
        if len(g) == 0:
            continue
        print(f"  {bucket:<14} {len(g):>7,} "
              f"{g['impact_bps'].std():>13.3f} "
              f"{g['impact_vwap_bps'].std():>13.3f} "
              f"{100*(g['impact_bps']==0).mean():>10.1f}% "
              f"{100*(g['impact_vwap_bps']==0).mean():>10.1f}%")

    # ── Step 5: side-by-side distribution plots ───────────────────────────
    CLIP   = 20
    BINS   = 150
    RANGE  = (-CLIP, CLIP)
    COLORS = {"buy": "#3a86ff", "sell": "#e63946", "all": "#555555"}

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Impact method comparison: adjacent tick vs 1-second VWAP bar",
                 fontsize=13, fontweight="bold")

    titles = ["Method 1 — Adjacent-tick impact", "Method 2 — VWAP-bar impact"]
    cols   = ["impact_bps", "impact_vwap_bps"]

    # Top row: full distribution, buy vs sell
    for ax, col, title in zip(axes[0], cols, titles):
        for side, color in [("buy", COLORS["buy"]), ("sell", COLORS["sell"])]:
            data = both.loc[both["side_label"] == side, col].clip(RANGE[0], RANGE[1])
            ax.hist(data, bins=BINS, range=RANGE, density=True,
                    color=color, alpha=0.55, label=side)
        ax.axvline(0, color="black", lw=1, ls="--", alpha=0.6)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Impact (bps, clipped ±20)", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        # annotate key stats
        s = both[col].dropna()
        ax.text(0.97, 0.95,
                f"std={s.std():.2f}\nIQR={s.quantile(.75)-s.quantile(.25):.2f}\nzero={100*(s==0).mean():.1f}%",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    # Bottom row: log-scale to show tails
    for ax, col, title in zip(axes[1], cols, titles):
        data_all = both[col].clip(RANGE[0], RANGE[1]).dropna()
        ax.hist(data_all, bins=BINS, range=RANGE, density=True,
                color=COLORS["all"], alpha=0.75)
        ax.axvline(0, color="black", lw=1, ls="--", alpha=0.6)
        ax.set_yscale("log")
        ax.set_title(f"{title} — log scale", fontsize=11)
        ax.set_xlabel("Impact (bps, clipped ±20)", fontsize=10)
        ax.set_ylabel("Density (log)", fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("impact_method_comparison.png", dpi=150, bbox_inches="tight")
    log.info("Plot saved -> impact_method_comparison.png")


if __name__ == "__main__":
    main()
