"""
Find block trades (dollar_value >= $200k), look up pre/post prices within 1 second,
compute slippage in bps. Saves to data/block_trades.parquet.

Usage:  python block_trades.py
"""

import logging
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR       = Path("data")
OUT_PATH       = DATA_DIR / "block_trades.parquet"
MIN_DOLLAR     = 200_000
ONE_SEC_NS     = 1_000_000_000   # 1 second in nanoseconds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RESULT_SCHEMA = pa.schema([
    ("ticker",          pa.string()),
    ("date",            pa.string()),
    ("timestamp_ns",    pa.int64()),
    ("price",           pa.float64()),
    ("size",            pa.int64()),
    ("dollar_value",    pa.float64()),
    ("pre_price",       pa.float64()),
    ("post_price",      pa.float64()),
    ("slippage_bps",    pa.float64()),
    ("exchange",        pa.int16()),
    # "conditions" appended separately to preserve its native list type
])


def find_blocks_in_file(ticker: str, path: Path) -> pa.Table | None:
    """
    Load one ticker-day file, identify block trades, compute pre/post slippage.
    Returns a PyArrow table of results, or None if no blocks found.
    """
    # Read numeric columns + exchange normally; keep conditions in PyArrow to
    # avoid the slow Python-list deserialisation on all ~700k rows per day.
    tbl = pq.read_table(
        path, columns=["participant_timestamp", "price", "size", "exchange", "conditions"]
    )

    # Sort by timestamp — stay in Arrow so conditions column is never converted
    order  = np.argsort(tbl.column("participant_timestamp").to_numpy(zero_copy_only=False), stable=True)
    tbl    = tbl.take(order.tolist())

    ts        = tbl.column("participant_timestamp").to_numpy(zero_copy_only=False).astype(np.int64)
    prices    = tbl.column("price").to_numpy(zero_copy_only=False).astype(np.float64)
    sizes     = tbl.column("size").to_numpy(zero_copy_only=False).astype(np.int64)
    exchanges = tbl.column("exchange").to_numpy(zero_copy_only=False).astype(np.int16)
    dv        = prices * sizes

    block_mask = dv >= MIN_DOLLAR
    n_blocks   = block_mask.sum()
    if n_blocks == 0:
        return None

    log.info("  %s %s : %d block trades from %d total", ticker, path.stem, n_blocks, len(ts))

    block_idx = np.where(block_mask)[0]
    block_ts  = ts[block_idx]

    # ── Pre-price: last trade strictly before block, within 1 second ──────
    # searchsorted 'left': all ts[:i] < block_ts at position i
    pre_ends = np.searchsorted(ts, block_ts, side="left")
    pre_idx  = pre_ends - 1
    pre_ok   = (pre_idx >= 0) & (ts[np.clip(pre_idx, 0, len(ts) - 1)] >= block_ts - ONE_SEC_NS)
    pre_prices = np.where(pre_ok, prices[np.clip(pre_idx, 0, len(prices) - 1)], np.nan)

    # ── Post-price: first trade strictly after block, within 1 second ─────
    # searchsorted 'right': all ts[i:] > block_ts at position i
    post_starts = np.searchsorted(ts, block_ts, side="right")
    post_ok     = (post_starts < len(ts)) & (
        ts[np.clip(post_starts, 0, len(ts) - 1)] <= block_ts + ONE_SEC_NS
    )
    post_prices = np.where(post_ok, prices[np.clip(post_starts, 0, len(prices) - 1)], np.nan)

    # ── Slippage ──────────────────────────────────────────────────────────
    with np.errstate(invalid="ignore", divide="ignore"):
        slippage_bps = (post_prices - pre_prices) / pre_prices * 10_000

    # Extract conditions for block rows only — stays in Arrow, no Python-list overhead
    conditions_col = tbl.column("conditions").take(block_idx.tolist())

    result = pa.table({
        "ticker":       pa.array([ticker] * n_blocks,              type=pa.string()),
        "date":         pa.array([path.stem] * n_blocks,           type=pa.string()),
        "timestamp_ns": pa.array(block_ts,                         type=pa.int64()),
        "price":        pa.array(prices[block_idx],                type=pa.float64()),
        "size":         pa.array(sizes[block_idx],                 type=pa.int64()),
        "dollar_value": pa.array(dv[block_idx],                    type=pa.float64()),
        "pre_price":    pa.array(pre_prices,                       type=pa.float64()),
        "post_price":   pa.array(post_prices,                      type=pa.float64()),
        "slippage_bps": pa.array(slippage_bps,                     type=pa.float64()),
        "exchange":     pa.array(exchanges[block_idx],             type=pa.int16()),
    }, schema=RESULT_SCHEMA)

    # Append conditions preserving its native list type from the source file
    result = result.append_column("conditions", conditions_col)

    return result


def main():
    files = sorted(
        p for p in DATA_DIR.rglob("*.parquet")
        if ".tmp." not in p.name and p.name != "daily_ohlcv.parquet"
           and p.name != "block_trades.parquet"
           and p.name != "aapl_daily_spread.parquet"
    )
    log.info("Found %d completed trade files", len(files))

    tables = []
    for path in files:
        ticker = path.parent.name
        try:
            t = find_blocks_in_file(ticker, path)
            if t is not None:
                tables.append(t)
        except Exception as exc:
            log.error("ERROR %s %s: %s", ticker, path.stem, exc)

    if not tables:
        log.warning("No block trades found.")
        return

    combined = pa.concat_tables(tables)
    pq.write_table(combined, OUT_PATH, compression="snappy")
    log.info("Saved %d block trades to %s", len(combined), OUT_PATH)

    # ── Summary stats ─────────────────────────────────────────────────────
    df = combined.to_pandas()

    print("\n=== Block trades by ticker ===")
    by_ticker = (
        df.groupby("ticker")
          .agg(blocks=("dollar_value", "count"),
               median_dv_k=("dollar_value", lambda x: x.median() / 1000),
               max_dv_m=("dollar_value", lambda x: x.max() / 1e6))
          .sort_values("blocks", ascending=False)
    )
    print(by_ticker.to_string())

    slip = df["slippage_bps"].dropna()
    print(f"\n=== Slippage distribution (bps)  [n={len(slip):,} with both pre & post] ===")
    for label, val in [
        ("Min",   slip.min()),
        ("p1",    slip.quantile(.01)),
        ("p5",    slip.quantile(.05)),
        ("p25",   slip.quantile(.25)),
        ("Median",slip.median()),
        ("p75",   slip.quantile(.75)),
        ("p95",   slip.quantile(.95)),
        ("p99",   slip.quantile(.99)),
        ("Max",   slip.max()),
    ]:
        print(f"  {label:<8} {val:>10.2f}")
    print(f"  Std dev  {slip.std():>10.2f}")
    print(f"\n  Missing pre  : {df['pre_price'].isna().sum():,}")
    print(f"  Missing post : {df['post_price'].isna().sum():,}")


if __name__ == "__main__":
    main()
