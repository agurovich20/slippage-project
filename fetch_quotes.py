"""
Fetch NBBO quote data from Polygon.io and save as parquet.

NOTE: Requires a Polygon plan that includes tick-level quote data.
      /v3/quotes returns NOT_AUTHORIZED on the Starter plan.
      Upgrade at https://polygon.io/dashboard/billing

Usage:
    python fetch_quotes.py                        # test: AAPL on 2024-06-03
    python fetch_quotes.py MSFT 2024-06-04        # single ticker/date
    python fetch_quotes.py --existing             # all ticker/dates with trade data
    python fetch_quotes.py --all                  # full 50-ticker, June-Aug 2024 run

Output: data/quotes/<TICKER>/<YYYY-MM-DD>.parquet
"""

import sys
import time
import logging
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from polygon import RESTClient

from fetch_trades import API_KEY, TICKERS_50, PAGE_LIMIT, BATCH_SIZE, INTER_TICKER_SLEEP, trading_days, DATE_START, DATE_END

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
QUOTES_DIR = Path("data/quotes")
TRADES_DIR = Path("data")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema — Polygon v3 quotes fields needed for Lee-Ready
# ---------------------------------------------------------------------------
SCHEMA = pa.schema([
    ("sip_timestamp",         pa.int64()),   # nanoseconds; primary key for trade matching
    ("participant_timestamp", pa.int64()),
    ("sequence_number",       pa.int64()),
    ("bid_price",             pa.float64()),
    ("bid_size",              pa.int32()),
    ("ask_price",             pa.float64()),
    ("ask_size",              pa.int32()),
    ("bid_exchange",          pa.int16()),
    ("ask_exchange",          pa.int16()),
    ("tape",                  pa.int8()),
    ("conditions",            pa.list_(pa.int32())),
    ("indicators",            pa.list_(pa.int32())),
])


def _quote_to_row(q) -> dict:
    return {
        "sip_timestamp":         getattr(q, "sip_timestamp",         None),
        "participant_timestamp": getattr(q, "participant_timestamp", None),
        "sequence_number":       getattr(q, "sequence_number",       None),
        "bid_price":             getattr(q, "bid_price",             None),
        "bid_size":              getattr(q, "bid_size",              None),
        "ask_price":             getattr(q, "ask_price",             None),
        "ask_size":              getattr(q, "ask_size",              None),
        "bid_exchange":          getattr(q, "bid_exchange",          None),
        "ask_exchange":          getattr(q, "ask_exchange",          None),
        "tape":                  getattr(q, "tape",                  None),
        "conditions":            getattr(q, "conditions",            None) or [],
        "indicators":            getattr(q, "indicators",            None) or [],
    }


def fetch_day(client: RESTClient, ticker: str, day: date, out_path: Path) -> int:
    """
    Stream all NBBO quotes for `ticker` on `day`, writing incrementally.
    Returns total quote count.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".tmp.parquet")

    date_str = day.isoformat()
    total = 0
    batch: list[dict] = []
    writer: pq.ParquetWriter | None = None

    def flush(rows: list[dict]) -> None:
        nonlocal writer
        table = pa.Table.from_pylist(rows, schema=SCHEMA)
        if writer is None:
            writer = pq.ParquetWriter(tmp_path, SCHEMA, compression="snappy")
        writer.write_table(table)

    try:
        for quote in client.list_quotes(
            ticker,
            timestamp_gte=f"{date_str}T00:00:00Z",
            timestamp_lte=f"{date_str}T23:59:59.999999999Z",
            limit=PAGE_LIMIT,
            sort="sip_timestamp",
            order="asc",
        ):
            batch.append(_quote_to_row(quote))
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
        log.warning("%s %s - 0 quotes (market closed or no data)", ticker, date_str)
        if tmp_path.exists():
            tmp_path.unlink()
        return 0

    tmp_path.rename(out_path)
    return total


def existing_trade_pairs() -> list[tuple[str, date]]:
    """Return (ticker, date) pairs that have completed trade files."""
    pairs = []
    for p in sorted(TRADES_DIR.rglob("*.parquet")):
        if ".tmp." in p.name or p.parent == TRADES_DIR:
            continue
        ticker = p.parent.name
        if ticker == "quotes":
            continue
        try:
            pairs.append((ticker, date.fromisoformat(p.stem)))
        except ValueError:
            pass
    return pairs


def run(pairs: list[tuple[str, date]]) -> None:
    client = RESTClient(api_key=API_KEY)
    for i, (ticker, day) in enumerate(pairs, 1):
        out_path = QUOTES_DIR / ticker / f"{day.isoformat()}.parquet"
        if out_path.exists():
            log.info("SKIP  %s %s (already exists)", ticker, day)
            continue

        log.info("FETCH %s %s  [%d/%d]", ticker, day, i, len(pairs))
        t0 = time.monotonic()
        try:
            n = fetch_day(client, ticker, day, out_path)
            if n:
                log.info("  -> %s quotes in %.1fs  (%s)", f"{n:,}", time.monotonic() - t0, out_path)
        except Exception as exc:
            log.error("  ERROR %s %s: %s", ticker, day, exc)

        time.sleep(INTER_TICKER_SLEEP)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    args = sys.argv[1:]

    if "--all" in args:
        days  = trading_days(DATE_START, DATE_END)
        pairs = [(t, d) for t in TICKERS_50 for d in days]
        log.info("Full run: %d pairs", len(pairs))

    elif "--existing" in args:
        pairs = existing_trade_pairs()
        log.info("Fetching quotes for %d existing trade files", len(pairs))

    elif len(args) == 2:
        pairs = [(args[0].upper(), date.fromisoformat(args[1]))]

    else:
        pairs = [("AAPL", date(2024, 6, 3))]
        log.info("Test mode: AAPL 2024-06-03")

    run(pairs)
