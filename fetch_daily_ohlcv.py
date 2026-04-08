"""
Fetch daily OHLCV aggregates from Polygon.io for 50 tickers.
Covers Jan 2023 – Aug 2024. Saves to data/daily_ohlcv.parquet.

Usage:  python fetch_daily_ohlcv.py
"""

import time
import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from polygon import RESTClient

from fetch_trades import API_KEY, TICKERS_50

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATE_START = "2023-01-01"
DATE_END   = "2024-08-30"
OUT_PATH   = Path("data/daily_ohlcv.parquet")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SCHEMA = pa.schema([
    ("ticker",     pa.string()),
    ("date",       pa.date32()),          # calendar date
    ("open",       pa.float64()),
    ("high",       pa.float64()),
    ("low",        pa.float64()),
    ("close",      pa.float64()),
    ("volume",     pa.float64()),
    ("vwap",       pa.float64()),
    ("num_trades", pa.int64()),
    ("timestamp",  pa.int64()),           # ms since epoch (Polygon native)
])


def fetch_all(client: RESTClient) -> pa.Table:
    tables = []

    for i, ticker in enumerate(TICKERS_50, 1):
        log.info("[%d/%d] %s", i, len(TICKERS_50), ticker)
        try:
            aggs = client.get_aggs(
                ticker,
                1, "day",
                DATE_START, DATE_END,
                adjusted=True,
                limit=50_000,
            )
        except Exception as exc:
            log.error("  ERROR %s: %s", ticker, exc)
            continue

        rows = [
            {
                "ticker":     ticker,
                "date":       __import__("datetime").date.fromtimestamp(a.timestamp / 1000),
                "open":       a.open,
                "high":       a.high,
                "low":        a.low,
                "close":      a.close,
                "volume":     a.volume,
                "vwap":       a.vwap,
                "num_trades": a.transactions,
                "timestamp":  a.timestamp,
            }
            for a in aggs
        ]
        log.info("  %d rows", len(rows))
        if rows:
            tables.append(pa.Table.from_pylist(rows, schema=SCHEMA))

        time.sleep(0.1)   # stay well within rate limits

    return pa.concat_tables(tables)


if __name__ == "__main__":
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    client = RESTClient(api_key=API_KEY)

    table = fetch_all(client)
    pq.write_table(table, OUT_PATH, compression="snappy")

    log.info("Saved %d rows → %s", len(table), OUT_PATH)
