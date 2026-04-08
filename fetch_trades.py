"""
Fetch tick-level trade data from Polygon.io and save as parquet.

Usage:
    python fetch_trades.py                        # test: AAPL on 2024-06-03
    python fetch_trades.py MSFT 2024-06-04        # single ticker/date
    python fetch_trades.py --all                  # full 50-ticker, June-Aug 2024 run

Output: data/<TICKER>/<YYYY-MM-DD>.parquet
"""

import sys
import time
import logging
from datetime import date, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from polygon import RESTClient

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_KEY = "fvBzlF_z4PLBp8l0aikf6HCcI1rIn5A0"

TICKERS_50 = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK.B", "JPM", "UNH",
    "LLY", "V",    "XOM",  "AVGO", "JNJ",  "MA",   "PG",   "HD",   "COST","MRK",
    "ABBV", "CVX",  "KO",   "PEP",  "ADBE", "WMT",  "BAC",  "MCD",  "CRM", "ACN",
    "AMD",  "LIN",  "NFLX", "TMO",  "CSCO", "ABT",  "TXN",  "WFC",  "DHR", "DIS",
    "AMGN", "VZ",   "INTU", "QCOM", "IBM",  "CAT",  "GE",   "RTX",  "SPGI","UNP",
]

# Date range for the full run
DATE_START = date(2024, 6, 3)   # first trading day of June 2024
DATE_END   = date(2024, 8, 30)

# API page size (max allowed)
PAGE_LIMIT = 50_000

# Write a pyarrow batch every N trades to cap RAM usage
BATCH_SIZE = 200_000

# Pause between tickers (seconds) to be polite to the API
INTER_TICKER_SLEEP = 0.2

DATA_DIR = Path("data")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema – matches Polygon v3 trades fields
# ---------------------------------------------------------------------------
SCHEMA = pa.schema([
    ("participant_timestamp", pa.int64()),   # nanoseconds since epoch
    ("sip_timestamp",         pa.int64()),
    ("price",                 pa.float64()),
    ("size",                  pa.int64()),
    ("conditions",            pa.list_(pa.int32())),
    ("exchange",              pa.int16()),
    ("tape",                  pa.int8()),
    ("trf_timestamp",         pa.int64()),
    ("sequence_number",       pa.int64()),
    ("id",                    pa.string()),
])


def _trade_to_row(t) -> dict:
    """Convert a polygon Trade object to a plain dict matching SCHEMA."""
    return {
        "participant_timestamp": getattr(t, "participant_timestamp", None),
        "sip_timestamp":         getattr(t, "sip_timestamp",         None),
        "price":                 getattr(t, "price",                 None),
        "size":                  getattr(t, "size",                  None),
        "conditions":            getattr(t, "conditions",            None) or [],
        "exchange":              getattr(t, "exchange",              None),
        "tape":                  getattr(t, "tape",                  None),
        "trf_timestamp":         getattr(t, "trf_timestamp",         None),
        "sequence_number":       getattr(t, "sequence_number",       None),
        "id":                    getattr(t, "id",                    None),
    }


def fetch_day(client: RESTClient, ticker: str, day: date, out_path: Path) -> int:
    """
    Stream all trades for `ticker` on `day`, writing incrementally to `out_path`.
    Returns total trade count.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".tmp.parquet")

    date_str = day.isoformat()
    total = 0
    batch: list[dict] = []

    writer: pq.ParquetWriter | None = None

    def flush(rows: list[dict]) -> pq.ParquetWriter:
        nonlocal writer
        table = pa.Table.from_pylist(rows, schema=SCHEMA)
        if writer is None:
            writer = pq.ParquetWriter(tmp_path, SCHEMA, compression="snappy")
        writer.write_table(table)
        return writer

    try:
        for trade in client.list_trades(
            ticker,
            timestamp_gte=f"{date_str}T00:00:00Z",
            timestamp_lte=f"{date_str}T23:59:59.999999999Z",
            limit=PAGE_LIMIT,
            sort="participant_timestamp",
            order="asc",
        ):
            batch.append(_trade_to_row(trade))
            total += 1

            if len(batch) >= BATCH_SIZE:
                flush(batch)
                batch.clear()

        if batch:
            flush(batch)

    finally:
        if writer is not None:
            writer.close()

    if total == 0:
        log.warning("%s %s – 0 trades (market closed or no data)", ticker, date_str)
        if tmp_path.exists():
            tmp_path.unlink()
        return 0

    tmp_path.rename(out_path)
    return total


def run(tickers: list[str], dates: list[date]) -> None:
    client = RESTClient(api_key=API_KEY)
    total_files = len(tickers) * len(dates)
    done = 0

    for ticker in tickers:
        for day in dates:
            out_path = DATA_DIR / ticker / f"{day.isoformat()}.parquet"
            if out_path.exists():
                log.info("SKIP  %s %s (already exists)", ticker, day)
                done += 1
                continue

            log.info("FETCH %s %s  [%d/%d]", ticker, day, done + 1, total_files)
            t0 = time.monotonic()
            try:
                n = fetch_day(client, ticker, day, out_path)
                elapsed = time.monotonic() - t0
                if n:
                    log.info("  -> %s trades in %.1fs  (%s)", f"{n:,}", elapsed, out_path)
            except Exception as exc:
                log.error("  ERROR %s %s: %s", ticker, day, exc)

            done += 1
            time.sleep(INTER_TICKER_SLEEP)


def trading_days(start: date, end: date) -> list[date]:
    """Return weekdays between start and end (inclusive). No holiday removal."""
    days = []
    cur = start
    while cur <= end:
        if cur.weekday() < 5:   # Mon-Fri
            days.append(cur)
        cur += timedelta(days=1)
    return days


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    args = sys.argv[1:]

    if "--all" in args:
        tickers = TICKERS_50
        dates   = trading_days(DATE_START, DATE_END)
        log.info("Full run: %d tickers × %d days = %d files", len(tickers), len(dates), len(tickers)*len(dates))

    elif len(args) == 2:
        tickers = [args[0].upper()]
        dates   = [date.fromisoformat(args[1])]

    else:
        # Default test: AAPL for one day
        tickers = ["AAPL"]
        dates   = [date(2024, 6, 3)]
        log.info("Test mode: AAPL 2024-06-03")

    run(tickers, dates)
