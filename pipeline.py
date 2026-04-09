"""
Consolidated data pipeline and visualization scripts.

Functions:
  run_fetch_trades()            -- Fetch tick-level trades from Polygon.io
  run_get_exchanges()           -- List Polygon exchange IDs and metadata
  run_block_trades()            -- Find block trades, compute pre/post slippage
  run_check_exchange()          -- Exchange distribution of block trades
  run_cross_stock_pipeline()    -- Cross-stock validation pipeline (fetch, feature, GAMLSS)
  run_summary_visualizations()  -- Summary dashboard: model comparison, GAMLSS calibration, EDA
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"]      = "1"

import sys
import time
import logging
import warnings
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import statsmodels.api as sm
from xgboost import XGBRegressor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from polygon import RESTClient


def run_fetch_trades():
    """
    Fetch tick-level trade data from Polygon.io and save as parquet.

    Usage (when run as main):
        python pipeline.py                        # test: AAPL on 2024-06-03
        python pipeline.py MSFT 2024-06-04        # single ticker/date
        python pipeline.py --all                  # full 50-ticker, June-Aug 2024 run

    Output: data/<TICKER>/<YYYY-MM-DD>.parquet
    """

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


def run_get_exchanges():
    API_KEY = "fvBzlF_z4PLBp8l0aikf6HCcI1rIn5A0"
    client = RESTClient(api_key=API_KEY)
    exchanges = client.get_exchanges(asset_class="stocks")

    print(f"{'ID':>4}  {'MIC':<12}  {'Name':<40}  {'Type'}")
    print("-" * 75)
    for ex in sorted(exchanges, key=lambda e: e.id or 0):
        print(f"{ex.id or '?':>4}  {ex.mic or ''::<12}  {ex.name or ''::<40}  {ex.type or ''}")


def run_block_trades():
    """
    Find block trades (dollar_value >= $200k), look up pre/post prices within 1 second,
    compute slippage in bps. Saves to data/block_trades.parquet.
    """

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


def run_check_exchange():
    # Confirmed via client.get_exchanges()
    EXCHANGE = {
        1:   ("XASE", "NYSE American",              "exchange"),
        2:   ("XBOS", "Nasdaq BX",                  "exchange"),
        3:   ("XCIS", "NYSE National",              "exchange"),
        4:   ("XADF", "FINRA ADF",                  "TRF"),
        5:   ("",     "Unlisted Trading Privileges", "SIP"),
        6:   ("XISE", "ISE Stocks",                  "TRF"),
        7:   ("EDGA", "Cboe EDGA",                   "exchange"),
        8:   ("EDGX", "Cboe EDGX",                   "exchange"),
        9:   ("XCHI", "NYSE Chicago",               "exchange"),
        10:  ("XNYS", "NYSE",                        "exchange"),
        11:  ("ARCX", "NYSE Arca",                   "exchange"),
        12:  ("XNAS", "Nasdaq",                      "exchange"),
        14:  ("LTSE", "Long-Term Stock Exchange",    "exchange"),
        15:  ("IEXG", "IEX",                         "exchange"),
        16:  ("CBSX", "Cboe Stock Exchange",          "TRF"),
        17:  ("XPHL", "Nasdaq PHLX",                 "exchange"),
        18:  ("BATY", "Cboe BYX",                    "exchange"),
        19:  ("BATS", "Cboe BZX",                    "exchange"),
        20:  ("EPRL", "MIAX Pearl",                  "exchange"),
        21:  ("MEMX", "Members Exchange",            "exchange"),
        22:  ("24EQ", "24X National Exchange",       "exchange"),
        62:  ("OOTC", "OTC Equity",                  "ORF"),
        201: ("FINY", "FINRA NYSE TRF",              "TRF"),
        202: ("FINN", "FINRA Nasdaq TRF Carteret",   "TRF"),
        203: ("FINC", "FINRA Nasdaq TRF Chicago",    "TRF"),
    }
    DARK_TYPES = {"TRF", "ORF"}

    df = pq.read_table("data/block_trades.parquet",
                       columns=["exchange", "dollar_value"]).to_pandas()
    large = df[df["dollar_value"] >= 200_000]

    print(f"=== Exchange distribution (dollar_value >= $200k, n={len(large):,}) ===\n")
    print(f"{'ID':>4}  {'MIC':<6}  {'Name':<36}  {'Type':<10}  {'L/D':<5}  {'Count':>8}  {'%':>6}")
    print("-" * 85)

    counts = large["exchange"].value_counts().sort_values(ascending=False)
    total = len(large)
    for ex_id, cnt in counts.items():
        mic, name, etype = EXCHANGE.get(int(ex_id), ("?", f"unknown-{ex_id}", "?"))
        dark = "DARK" if etype in DARK_TYPES else "lit"
        pct  = 100 * cnt / total
        print(f"{int(ex_id):>4}  {mic:<6}  {name:<36}  {etype:<10}  {dark:<5}  {cnt:>8,}  {pct:>5.1f}%")

    dark_n = large[large["exchange"].map(lambda x: EXCHANGE.get(int(x), ("","","?"))[2] in DARK_TYPES)].shape[0]
    lit_n  = total - dark_n
    print(f"\n{'─'*85}")
    print(f"  Lit  (exchange): {lit_n:>8,}  ({100*lit_n/total:.1f}%)")
    print(f"  Dark (TRF/ORF):  {dark_n:>8,}  ({100*dark_n/total:.1f}%)")


def run_cross_stock_pipeline():
    """
    Cross-stock validation pipeline: fetch data, build features, run XGB GAMLSS
    for GOOG, NVDA, HOOD, AMD, then compare with existing AAPL/COIN results.

    Pipeline per ticker:
      1. Fetch tick-level trades from Polygon (Jun-Sep 2024)
      2. Identify block trades (>=$200k)
      3. Classify sides (tick test)
      4. Fetch 1-sec bars, compute VWAP impact
      5. Build features (roll_spread_500, roll_vol_500, participation_rate)
      6. Split train (Jun-Aug) / test (Sep)
      7. Run 2-stage XGB GAMLSS with fixed AAPL hyperparameters
      8. Compute Laplace prediction intervals

    Output:
      - Per-ticker feature parquet files in data/
      - cross_stock_validation.png
      - Console summary table for all 6 stocks
    """

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(__name__)

    # ─── Config ──────────────────────────────────────────────────────────────────
    API_KEY = "fvBzlF_z4PLBp8l0aikf6HCcI1rIn5A0"
    DATA_DIR = Path("data")
    NEW_TICKERS = ["TSLA"]
    DATE_START = date(2024, 6, 3)
    DATE_END = date(2024, 9, 30)
    PAGE_LIMIT = 50_000
    BATCH_SIZE = 200_000
    MIN_DOLLAR = 200_000
    ONE_SEC_NS = 1_000_000_000
    ONE_MIN_NS = 60 * ONE_SEC_NS
    DARK_IDS = {4, 6, 16, 62, 201, 202, 203}
    OPEN_SEC = 9 * 3600 + 30 * 60
    WINDOW = 500
    MIN_TICKS = 10

    FEATURES_3 = ["roll_spread_500", "roll_vol_500", "participation_rate"]

    TRADE_SCHEMA = pa.schema([
        ("participant_timestamp", pa.int64()),
        ("sip_timestamp", pa.int64()),
        ("price", pa.float64()),
        ("size", pa.int64()),
        ("conditions", pa.list_(pa.int32())),
        ("exchange", pa.int16()),
        ("tape", pa.int8()),
        ("trf_timestamp", pa.int64()),
        ("sequence_number", pa.int64()),
        ("id", pa.string()),
    ])

    SEC_SCHEMA = pa.schema([
        ("t", pa.int64()),
        ("vw", pa.float64()),
        ("o", pa.float64()),
        ("h", pa.float64()),
        ("l", pa.float64()),
        ("c", pa.float64()),
        ("v", pa.float64()),
        ("n", pa.int32()),
    ])

    def trading_days(start, end):
        days = []
        cur = start
        while cur <= end:
            if cur.weekday() < 5:
                days.append(cur)
            cur += timedelta(days=1)
        return days

    # ═════════════════════════════════════════════════════════════════════════════
    # STEP 1: Fetch tick trades from Polygon
    # ═════════════════════════════════════════════════════════════════════════════

    def _trade_to_row(t):
        return {
            "participant_timestamp": getattr(t, "participant_timestamp", None),
            "sip_timestamp": getattr(t, "sip_timestamp", None),
            "price": getattr(t, "price", None),
            "size": getattr(t, "size", None),
            "conditions": getattr(t, "conditions", None) or [],
            "exchange": getattr(t, "exchange", None),
            "tape": getattr(t, "tape", None),
            "trf_timestamp": getattr(t, "trf_timestamp", None),
            "sequence_number": getattr(t, "sequence_number", None),
            "id": getattr(t, "id", None),
        }

    def fetch_day(client, ticker, day, out_path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(".tmp.parquet")
        date_str = day.isoformat()
        total = 0
        batch = []
        writer = None

        def flush(rows):
            nonlocal writer
            table = pa.Table.from_pylist(rows, schema=TRADE_SCHEMA)
            if writer is None:
                writer = pq.ParquetWriter(tmp_path, TRADE_SCHEMA, compression="snappy")
            writer.write_table(table)

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
            if tmp_path.exists():
                tmp_path.unlink()
            return 0

        tmp_path.rename(out_path)
        return total

    def fetch_sec_aggs(client, ticker, date_str, out_path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        aggs = client.get_aggs(ticker, 1, "second", date_str, date_str,
                               adjusted=True, limit=100_000)
        if not aggs:
            return 0
        rows = [{"t": a.timestamp, "vw": a.vwap, "o": a.open, "h": a.high,
                 "l": a.low, "c": a.close, "v": a.volume, "n": a.transactions}
                for a in aggs]
        pq.write_table(pa.Table.from_pylist(rows, schema=SEC_SCHEMA),
                        out_path, compression="snappy")
        return len(rows)

    def fetch_all_trades(tickers, dates):
        client = RESTClient(api_key=API_KEY)
        total_files = len(tickers) * len(dates)
        done = 0
        for ticker in tickers:
            for day in dates:
                out_path = DATA_DIR / ticker / f"{day.isoformat()}.parquet"
                if out_path.exists():
                    done += 1
                    continue
                log.info("FETCH %s %s  [%d/%d]", ticker, day, done + 1, total_files)
                try:
                    n = fetch_day(client, ticker, day, out_path)
                    if n:
                        log.info("  -> %s trades", f"{n:,}")
                except Exception as exc:
                    log.error("  ERROR %s %s: %s", ticker, day, exc)
                done += 1
                time.sleep(0.15)

    # ═════════════════════════════════════════════════════════════════════════════
    # STEP 2: Find block trades for a ticker
    # ═════════════════════════════════════════════════════════════════════════════

    def find_blocks_for_ticker(ticker):
        """Find all block trades for a ticker from its daily parquet files."""
        ticker_dir = DATA_DIR / ticker
        files = sorted(ticker_dir.glob("????-??-??.parquet"))
        if not files:
            log.warning("No trade files for %s", ticker)
            return pd.DataFrame()

        all_blocks = []
        for path in files:
            date_str = path.stem
            try:
                tbl = pq.read_table(path, columns=[
                    "participant_timestamp", "price", "size", "exchange"])
            except Exception:
                continue

            order = np.argsort(
                tbl.column("participant_timestamp").to_numpy(zero_copy_only=False),
                stable=True)
            tbl = tbl.take(order.tolist())

            ts = tbl.column("participant_timestamp").to_numpy(zero_copy_only=False).astype(np.int64)
            prices = tbl.column("price").to_numpy(zero_copy_only=False).astype(np.float64)
            sizes = tbl.column("size").to_numpy(zero_copy_only=False).astype(np.int64)
            exchanges = tbl.column("exchange").to_numpy(zero_copy_only=False).astype(np.int16)
            dv = prices * sizes

            block_mask = dv >= MIN_DOLLAR
            if not block_mask.any():
                continue

            block_idx = np.where(block_mask)[0]
            block_ts = ts[block_idx]

            # Pre-price
            pre_ends = np.searchsorted(ts, block_ts, side="left")
            pre_idx = pre_ends - 1
            pre_ok = (pre_idx >= 0) & (
                ts[np.clip(pre_idx, 0, len(ts) - 1)] >= block_ts - ONE_SEC_NS)
            pre_prices = np.where(
                pre_ok, prices[np.clip(pre_idx, 0, len(prices) - 1)], np.nan)

            # Post-price
            post_starts = np.searchsorted(ts, block_ts, side="right")
            post_ok = (post_starts < len(ts)) & (
                ts[np.clip(post_starts, 0, len(ts) - 1)] <= block_ts + ONE_SEC_NS)
            post_prices = np.where(
                post_ok, prices[np.clip(post_starts, 0, len(prices) - 1)], np.nan)

            with np.errstate(invalid="ignore", divide="ignore"):
                slippage_bps = (post_prices - pre_prices) / pre_prices * 10_000

            df = pd.DataFrame({
                "ticker": ticker,
                "date": date_str,
                "timestamp_ns": block_ts,
                "price": prices[block_idx],
                "size": sizes[block_idx],
                "dollar_value": dv[block_idx],
                "pre_price": pre_prices,
                "post_price": post_prices,
                "slippage_bps": slippage_bps,
                "exchange": exchanges[block_idx],
            })
            all_blocks.append(df)

        if not all_blocks:
            return pd.DataFrame()

        result = pd.concat(all_blocks, ignore_index=True)
        log.info("%s: %d block trades found across %d days",
                 ticker, len(result), len(files))
        return result

    # ═════════════════════════════════════════════════════════════════════════════
    # STEP 3: Classify sides (tick test)
    # ═════════════════════════════════════════════════════════════════════════════

    def classify_sides(blocks):
        raw = np.sign(blocks["price"].to_numpy() - blocks["pre_price"].to_numpy()).astype(float)
        raw[blocks["pre_price"].isna().to_numpy()] = np.nan
        dirs = pd.Series(raw, index=blocks.index).replace(0.0, np.nan)
        dirs = (
            blocks.groupby(["ticker", "date"], sort=False, group_keys=False)
            .apply(lambda g: dirs.loc[g.index].ffill(), include_groups=False)
        )
        blocks["side"] = dirs.fillna(0).astype(np.int8).to_numpy()
        blocks["side_label"] = np.where(
            blocks["side"] == 1, "buy",
            np.where(blocks["side"] == -1, "sell", "unknown"))
        return blocks

    # ═════════════════════════════════════════════════════════════════════════════
    # STEP 4: Compute VWAP impact
    # ═════════════════════════════════════════════════════════════════════════════

    def compute_vwap_impact(blocks, client):
        """Fetch 1-sec bars and compute VWAP-referenced impact for each block."""
        aggs_dir = DATA_DIR / "sec_aggs"
        impact = np.full(len(blocks), np.nan)

        for (ticker, date_str), grp in blocks.groupby(["ticker", "date"]):
            # Fetch/cache 1-sec bars
            agg_path = aggs_dir / ticker / f"{date_str}.parquet"
            if not agg_path.exists():
                agg_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    fetch_sec_aggs(client, ticker, date_str, agg_path)
                    time.sleep(0.1)
                except Exception as e:
                    log.warning("Failed to fetch 1-sec bars for %s %s: %s",
                                ticker, date_str, e)
                    continue

            if not agg_path.exists():
                continue

            try:
                tbl = pq.read_table(agg_path, columns=["t", "vw"])
                t_ms = np.array(tbl.column("t").to_pylist(), dtype=np.int64)
                vw = np.array(tbl.column("vw").to_pylist(), dtype=np.float64)
            except Exception:
                continue

            t_s = t_ms // 1000
            order = np.argsort(t_s)
            t_s = t_s[order]
            vw = vw[order]

            blk_ts = grp["timestamp_ns"].to_numpy(np.int64)
            ref_s = blk_ts // 1_000_000_000 - 1

            idx = np.searchsorted(t_s, ref_s, side="right") - 1
            valid = idx >= 0
            vwap_ref = np.full(len(blk_ts), np.nan)
            vwap_ref[valid] = vw[idx[valid]]

            post_p = grp["post_price"].to_numpy(np.float64)
            with np.errstate(invalid="ignore", divide="ignore"):
                imp = (post_p - vwap_ref) / vwap_ref * 10_000
            impact[grp.index] = imp

        blocks["impact_vwap_bps"] = impact
        return blocks

    # ═════════════════════════════════════════════════════════════════════════════
    # STEP 5: Build features
    # ═════════════════════════════════════════════════════════════════════════════

    def roll_spread_calc(px_window):
        dp = np.diff(px_window.astype(np.float64))
        if len(dp) < 2:
            return np.nan
        cov_mat = np.cov(dp[1:], dp[:-1], ddof=1)
        cov1 = cov_mat[0, 1]
        if cov1 < 0:
            return 2.0 * np.sqrt(-cov1)
        return np.nan

    def build_features_for_ticker(ticker, blocks_df):
        """Build feature table for lit buy blocks of a specific ticker."""
        buys = blocks_df[
            (blocks_df["ticker"] == ticker) &
            (~blocks_df["exchange"].isin(DARK_IDS)) &
            (blocks_df["side_label"] == "buy") &
            blocks_df["impact_vwap_bps"].notna()
        ].copy()
        buys = buys.sort_values(["date", "timestamp_ns"]).reset_index(drop=True)

        if len(buys) == 0:
            log.warning("%s: no lit buy blocks with VWAP impact", ticker)
            return pd.DataFrame(), pd.DataFrame()

        log.info("%s: %d lit buy blocks to process", ticker, len(buys))

        # Compute median price for bps normalization
        mid_price = buys["price"].median()
        log.info("%s: median block price = %.2f (used for bps normalization)", ticker, mid_price)

        # time_of_day
        et_times = (
            pd.to_datetime(buys["timestamp_ns"], unit="ns", utc=True)
            .dt.tz_convert("America/New_York")
        )
        buys["time_of_day"] = (
            et_times.dt.hour * 3600 + et_times.dt.minute * 60 + et_times.dt.second
            - OPEN_SEC
        )
        buys["day_of_week"] = pd.to_datetime(buys["date"]).dt.dayofweek

        roll_spread_arr = np.full(len(buys), np.nan)
        roll_vol_arr = np.full(len(buys), np.nan)
        trail_vol_arr = np.full(len(buys), np.nan)

        days_processed = 0
        for dt, grp in buys.groupby("date"):
            tick_path = DATA_DIR / ticker / f"{dt}.parquet"
            if not tick_path.exists():
                continue

            ticks = pd.read_parquet(str(tick_path),
                                    columns=["sip_timestamp", "price", "size"])
            ticks = ticks.sort_values("sip_timestamp").reset_index(drop=True)

            ts = ticks["sip_timestamp"].to_numpy(dtype=np.int64)
            px = ticks["price"].to_numpy(dtype=np.float64)
            sz = ticks["size"].to_numpy(dtype=np.int64)
            N = len(ts)

            cum_sz = np.empty(N + 1, dtype=np.int64)
            cum_sz[0] = 0
            np.cumsum(sz, out=cum_sz[1:])

            block_ts = grp["timestamp_ns"].to_numpy(dtype=np.int64)
            hi_idx = np.searchsorted(ts, block_ts, side="left")
            lo_idx_1min = np.searchsorted(ts, block_ts - ONE_MIN_NS, side="left")

            vol_sh = (cum_sz[np.minimum(hi_idx, N)] -
                      cum_sz[lo_idx_1min]).astype(float)
            vol_sh = np.where(hi_idx > lo_idx_1min, vol_sh, np.nan)
            trail_vol_arr[grp.index] = vol_sh

            for idx, hi in zip(grp.index, hi_idx):
                lo = max(0, hi - WINDOW)
                if hi - lo < MIN_TICKS:
                    continue
                px_win = px[lo:hi]

                spread_raw = roll_spread_calc(px_win)
                if not np.isnan(spread_raw):
                    roll_spread_arr[idx] = spread_raw / mid_price * 1e4

                dp_win = np.diff(px_win)
                if len(dp_win) >= 2:
                    roll_vol_arr[idx] = dp_win.std(ddof=1) / mid_price * 1e4

            days_processed += 1

        log.info("%s: processed %d days for features", ticker, days_processed)

        buys["roll_spread_500"] = roll_spread_arr
        buys["roll_vol_500"] = roll_vol_arr
        buys["trail_1min_volume"] = trail_vol_arr
        buys["participation_rate"] = buys["size"] / buys["trail_1min_volume"]
        buys["log_dollar_value"] = np.log(buys["dollar_value"])
        buys = buys.rename(columns={"exchange": "exchange_id"})

        ALL_FEATURES = [
            "dollar_value", "log_dollar_value", "participation_rate",
            "roll_spread_500", "roll_vol_500", "time_of_day", "exchange_id",
            "day_of_week",
        ]
        TARGET = "impact_vwap_bps"

        feat_df = buys[["date"] + ALL_FEATURES + [TARGET]].copy()
        feat_df = feat_df.dropna(subset=ALL_FEATURES + [TARGET])
        feat_df = feat_df[feat_df["participation_rate"].between(0, 1, inclusive="neither")]
        feat_df = feat_df[feat_df["roll_vol_500"] > 0]

        # Remove extreme outliers (e.g. NVDA stock split on 2024-06-10 causes
        # pre-split trades to have ~90,000 bps "impact" due to split-adjusted bars)
        IMPACT_CAP_BPS = 500
        n_before = len(feat_df)
        feat_df = feat_df[feat_df[TARGET].abs() <= IMPACT_CAP_BPS]
        n_removed = n_before - len(feat_df)
        if n_removed > 0:
            log.info("%s: removed %d rows with |impact| > %d bps (%.2f%%)",
                     ticker, n_removed, IMPACT_CAP_BPS, 100 * n_removed / n_before)

        feat_df = feat_df.reset_index(drop=True)

        # Split train/test
        train_df = feat_df[feat_df["date"] < "2024-09-01"].reset_index(drop=True)
        test_df = feat_df[feat_df["date"] >= "2024-09-01"].reset_index(drop=True)

        log.info("%s: features built — train=%d, test=%d", ticker, len(train_df), len(test_df))
        return train_df, test_df

    # ═════════════════════════════════════════════════════════════════════════════
    # STEP 6: XGB GAMLSS with fixed hyperparameters
    # ═════════════════════════════════════════════════════════════════════════════

    def run_gamlss(train_df, test_df, ticker):
        """Run 2-stage XGB GAMLSS and return results dict."""
        if len(train_df) < 50 or len(test_df) < 10:
            log.warning("%s: insufficient data (train=%d, test=%d)",
                        ticker, len(train_df), len(test_df))
            return None

        X_tr = train_df[FEATURES_3].to_numpy(dtype=np.float64)
        y_tr = train_df["impact_vwap_bps"].abs().to_numpy(dtype=np.float64)
        X_te = test_df[FEATURES_3].to_numpy(dtype=np.float64)
        y_te = test_df["impact_vwap_bps"].abs().to_numpy(dtype=np.float64)

        # Stage 1: Location (XGB LAD)
        loc_model = XGBRegressor(
            objective="reg:absoluteerror", tree_method="hist",
            max_depth=3, n_estimators=200, learning_rate=0.07,
            min_child_weight=5, reg_alpha=10, reg_lambda=10,
            verbosity=0, random_state=42, n_jobs=1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loc_model.fit(X_tr, y_tr)

        mu_hat_tr = np.maximum(loc_model.predict(X_tr), 0.0)
        mu_hat_te = np.maximum(loc_model.predict(X_te), 0.0)
        loc_mae = np.mean(np.abs(y_te - mu_hat_te))

        # Stage 2: Scale (XGB MSE on |residuals|)
        abs_resid_tr = np.abs(y_tr - mu_hat_tr)
        scale_model = XGBRegressor(
            objective="reg:squarederror", tree_method="hist",
            max_depth=3, n_estimators=50, learning_rate=0.1,
            min_child_weight=20, reg_alpha=1, reg_lambda=1,
            verbosity=0, random_state=42, n_jobs=1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scale_model.fit(X_tr, abs_resid_tr)

        b_hat_te = np.clip(scale_model.predict(X_te), 0.1, None)

        # Coverage at 50%, 80%, 90%
        coverages = {}
        for level in [0.50, 0.80, 0.90]:
            z = np.log(1.0 / (1.0 - level))
            lo = np.maximum(mu_hat_te - z * b_hat_te, 0.0)
            hi = mu_hat_te + z * b_hat_te
            cov = ((y_te >= lo) & (y_te <= hi)).mean()
            width = (hi - lo).mean()
            coverages[level] = (cov, width)

        return {
            "ticker": ticker,
            "n_train": len(train_df),
            "n_test": len(test_df),
            "mean_abs_impact": y_te.mean(),
            "loc_mae": loc_mae,
            "cov_90": coverages[0.90][0],
            "cov_80": coverages[0.80][0],
            "cov_50": coverages[0.50][0],
            "width_90": coverages[0.90][1],
        }

    # ═════════════════════════════════════════════════════════════════════════════
    # STEP 7: Load existing AAPL/COIN results
    # ═════════════════════════════════════════════════════════════════════════════

    def load_existing_results():
        """Run GAMLSS on existing AAPL and COIN data."""
        results = []

        datasets = {
            "AAPL": ("data/lit_buy_features_v2.parquet",
                     "data/lit_buy_features_v2_sep.parquet"),
            "COIN": ("data/coin_lit_buy_features_train.parquet",
                     "data/coin_lit_buy_features_test.parquet"),
        }

        for ticker, (tr_path, te_path) in datasets.items():
            if not os.path.exists(tr_path) or not os.path.exists(te_path):
                log.warning("Missing data for %s", ticker)
                continue

            train_df = pd.read_parquet(tr_path)
            test_df = pd.read_parquet(te_path)
            r = run_gamlss(train_df, test_df, ticker)
            if r:
                results.append(r)
                log.info("%s (existing): train=%d, test=%d, MAE=%.4f, 90%% cov=%.4f",
                         ticker, r["n_train"], r["n_test"], r["loc_mae"], r["cov_90"])

        return results

    # ═════════════════════════════════════════════════════════════════════════════
    # MAIN
    # ═════════════════════════════════════════════════════════════════════════════

    dates = trading_days(DATE_START, DATE_END)
    log.info("Date range: %s to %s (%d trading days)", DATE_START, DATE_END, len(dates))

    # Step 1: Fetch tick data for new tickers
    log.info("=" * 70)
    log.info("STEP 1: Fetching tick trades for %s", NEW_TICKERS)
    log.info("=" * 70)
    fetch_all_trades(NEW_TICKERS, dates)

    # Step 2-4: Process each new ticker
    client = RESTClient(api_key=API_KEY)
    new_results = []

    for ticker in NEW_TICKERS:
        log.info("=" * 70)
        log.info("PROCESSING %s", ticker)
        log.info("=" * 70)

        # Find block trades
        blocks = find_blocks_for_ticker(ticker)
        if blocks.empty:
            log.warning("%s: no block trades found, skipping", ticker)
            continue

        # Classify sides
        blocks = blocks.sort_values(["ticker", "date", "timestamp_ns"]).reset_index(drop=True)
        blocks = classify_sides(blocks)
        n_buys = (blocks["side_label"] == "buy").sum()
        n_sells = (blocks["side_label"] == "sell").sum()
        log.info("%s: %d buys, %d sells, %d unknown",
                 ticker, n_buys, n_sells,
                 (blocks["side_label"] == "unknown").sum())

        # Compute VWAP impact
        blocks = compute_vwap_impact(blocks, client)
        n_with_impact = blocks["impact_vwap_bps"].notna().sum()
        log.info("%s: %d/%d blocks have VWAP impact", ticker, n_with_impact, len(blocks))

        # Build features
        train_df, test_df = build_features_for_ticker(ticker, blocks)
        if train_df.empty or test_df.empty:
            log.warning("%s: empty train or test set, skipping", ticker)
            continue

        # Save feature files
        tr_path = f"data/{ticker.lower()}_lit_buy_features_train.parquet"
        te_path = f"data/{ticker.lower()}_lit_buy_features_test.parquet"
        train_df.to_parquet(tr_path, index=False, compression="snappy")
        test_df.to_parquet(te_path, index=False, compression="snappy")
        log.info("%s: saved %s (%d) and %s (%d)",
                 ticker, tr_path, len(train_df), te_path, len(test_df))

        # Run GAMLSS
        r = run_gamlss(train_df, test_df, ticker)
        if r:
            new_results.append(r)

    # Step 5: Load existing AAPL/COIN results
    log.info("=" * 70)
    log.info("Loading existing AAPL/COIN results")
    log.info("=" * 70)
    existing_results = load_existing_results()

    all_results = existing_results + new_results

    if not all_results:
        log.error("No results to display!")
        return

    # ═════════════════════════════════════════════════════════════════════════
    # SUMMARY TABLE
    # ═════════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 100}")
    print(f"  CROSS-STOCK XGB GAMLSS VALIDATION (3 features: spread, vol, participation)")
    print(f"  Location: XGB LAD (depth=3, n=200) | Scale: XGB MSE (depth=3, n=50)")
    print(f"  Train: Jun-Aug 2024 | Test: Sep 2024 | Laplace intervals")
    print(f"{'=' * 100}")
    print(f"  {'Ticker':<8} {'n_train':>8} {'n_test':>7} {'Mean|imp|':>10} "
          f"{'Loc MAE':>8} {'90% Cov':>8} {'80% Cov':>8} {'50% Cov':>8} "
          f"{'90% Width':>10}")
    print(f"  {'-' * 8} {'-' * 8} {'-' * 7} {'-' * 10} {'-' * 8} "
          f"{'-' * 8} {'-' * 8} {'-' * 8} {'-' * 10}")

    for r in all_results:
        print(f"  {r['ticker']:<8} {r['n_train']:>8,} {r['n_test']:>7,} "
              f"{r['mean_abs_impact']:>10.4f} {r['loc_mae']:>8.4f} "
              f"{r['cov_90']:>8.4f} {r['cov_80']:>8.4f} {r['cov_50']:>8.4f} "
              f"{r['width_90']:>10.4f}")

    print(f"\n  Nominal targets:  90% -> 0.9000   80% -> 0.8000   50% -> 0.5000")

    # ═════════════════════════════════════════════════════════════════════════
    # VISUALIZATION
    # ═════════════════════════════════════════════════════════════════════════
    results_df = pd.DataFrame(all_results)
    n_stocks = len(results_df)
    tickers_ordered = results_df["ticker"].tolist()
    colors = ["#2563eb", "#dc2626", "#16a34a", "#f59e0b", "#7c3aed", "#ec4899"][:n_stocks]

    fig = plt.figure(figsize=(24, 14))
    gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.32, hspace=0.42)

    x = np.arange(n_stocks)
    bar_w = 0.55

    # Panel 1: Coverage at 90%, 80%, 50% (grouped bars)
    ax1 = fig.add_subplot(gs[0, 0])
    bw = 0.22
    for j, (level, label) in enumerate([(0.90, "90%"), (0.80, "80%"), (0.50, "50%")]):
        col_name = f"cov_{int(level*100):d}"
        vals = results_df[col_name].values
        offset = (j - 1) * bw
        bars = ax1.bar(x + offset, vals, width=bw, label=f"{label} actual",
                       alpha=0.85, edgecolor="white", linewidth=0.5)
        ax1.axhline(level, color="gray", lw=0.8, ls=":", alpha=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(tickers_ordered, fontsize=10, fontweight="bold")
    ax1.set_ylabel("Actual Coverage", fontsize=11)
    ax1.set_title("Coverage by Stock\n(dashed = nominal)", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.2)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.set_ylim(0, 1.05)

    # Panel 2: Location MAE
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(x, results_df["loc_mae"].values, color=colors, width=bar_w,
            edgecolor="white", linewidth=0.8)
    for i, v in enumerate(results_df["loc_mae"].values):
        ax2.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(tickers_ordered, fontsize=10, fontweight="bold")
    ax2.set_ylabel("MAE (bps)", fontsize=11)
    ax2.set_title("Location Model MAE (OOS)", fontsize=12, fontweight="bold")
    ax2.grid(axis="y", alpha=0.2)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Panel 3: Mean absolute impact
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.bar(x, results_df["mean_abs_impact"].values, color=colors, width=bar_w,
            edgecolor="white", linewidth=0.8)
    for i, v in enumerate(results_df["mean_abs_impact"].values):
        ax3.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(tickers_ordered, fontsize=10, fontweight="bold")
    ax3.set_ylabel("|impact| (bps)", fontsize=11)
    ax3.set_title("Mean |impact| on Test Set", fontsize=12, fontweight="bold")
    ax3.grid(axis="y", alpha=0.2)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    # Panel 4: 90% interval width
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.bar(x, results_df["width_90"].values, color=colors, width=bar_w,
            edgecolor="white", linewidth=0.8)
    for i, v in enumerate(results_df["width_90"].values):
        ax4.text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")
    ax4.set_xticks(x)
    ax4.set_xticklabels(tickers_ordered, fontsize=10, fontweight="bold")
    ax4.set_ylabel("Width (bps)", fontsize=11)
    ax4.set_title("Mean 90% Interval Width", fontsize=12, fontweight="bold")
    ax4.grid(axis="y", alpha=0.2)
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)

    # Panel 5: Coverage deviation from nominal (heatmap)
    ax5 = fig.add_subplot(gs[1, 1])
    cov_matrix = np.array([
        [r["cov_90"] - 0.90, r["cov_80"] - 0.80, r["cov_50"] - 0.50]
        for r in all_results
    ])
    lim = max(abs(cov_matrix.min()), abs(cov_matrix.max())) * 1.1
    im = ax5.imshow(cov_matrix, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax5.set_xticks([0, 1, 2])
    ax5.set_xticklabels(["90%", "80%", "50%"], fontsize=10)
    ax5.set_yticks(range(n_stocks))
    ax5.set_yticklabels(tickers_ordered, fontsize=10, fontweight="bold")
    for i in range(n_stocks):
        for j in range(3):
            v = cov_matrix[i, j]
            text_color = "white" if abs(v) > lim * 0.5 else "black"
            ax5.text(j, i, f"{v:+.3f}", ha="center", va="center",
                     fontsize=9.5, fontweight="bold", color=text_color)
    cbar = plt.colorbar(im, ax=ax5, fraction=0.04, pad=0.04)
    cbar.set_label("Coverage - Nominal", fontsize=9)
    ax5.set_title("Coverage Deviation from Nominal\n(blue=under, red=over)",
                  fontsize=12, fontweight="bold")

    # Panel 6: Sample size
    ax6 = fig.add_subplot(gs[1, 2])
    train_sizes = results_df["n_train"].values
    test_sizes = results_df["n_test"].values
    ax6.bar(x - 0.15, train_sizes, width=0.3, color="#64748b", label="Train",
            edgecolor="white", linewidth=0.5)
    ax6.bar(x + 0.15, test_sizes, width=0.3, color="#f59e0b", label="Test",
            edgecolor="white", linewidth=0.5)
    for i, (tr, te) in enumerate(zip(train_sizes, test_sizes)):
        ax6.text(i - 0.15, tr + 50, f"{tr:,}", ha="center", fontsize=7.5,
                 fontweight="bold", rotation=45)
        ax6.text(i + 0.15, te + 50, f"{te:,}", ha="center", fontsize=7.5,
                 fontweight="bold", rotation=45)
    ax6.set_xticks(x)
    ax6.set_xticklabels(tickers_ordered, fontsize=10, fontweight="bold")
    ax6.set_ylabel("Number of trades", fontsize=11)
    ax6.set_title("Dataset Size (Train vs Test)", fontsize=12, fontweight="bold")
    ax6.legend(fontsize=9)
    ax6.grid(axis="y", alpha=0.2)
    ax6.spines["top"].set_visible(False)
    ax6.spines["right"].set_visible(False)

    fig.suptitle(
        "Cross-Stock XGB GAMLSS Validation: Block Trade Impact Prediction\n"
        "3 features (spread, volatility, participation rate) | "
        "Fixed AAPL hyperparameters | Train: Jun-Aug, Test: Sep 2024",
        fontsize=14, fontweight="bold", y=1.01,
    )

    fig.savefig("cross_stock_validation.png", dpi=150, bbox_inches="tight")
    log.info("Saved -> cross_stock_validation.png")

    # Save results CSV
    results_df.to_csv("data/cross_stock_results.csv", index=False)
    log.info("Saved -> data/cross_stock_results.csv")

    print("\nDone!")


def run_summary_visualizations():
    """
    Summary visualizations for the market impact project.

    Produces three figures:
      1. model_comparison.png — OOS metrics for all 6 point-prediction models (AAPL + COIN)
      2. gamlss_calibration_overlay.png — Linear vs XGB vs Bayesian calibration overlaid
      3. eda_overview.png — Data EDA: target distributions, feature correlations, feature vs impact
    """

    FEATURES = [
        "dollar_value", "log_dollar_value", "participation_rate",
        "roll_spread_500", "roll_vol_500", "exchange_id",
    ]
    FEAT_SHORT = ["dollar_val", "log_dollar", "prate", "spread", "vol", "exch_id"]

    # =============================================================================
    # FIGURE 1: Model Comparison Dashboard
    # =============================================================================
    print("Building Figure 1: Model Comparison Dashboard...", flush=True)

    ols = pd.read_csv("data/ols_results.csv")
    oos = ols[ols["sample"] == "out_of_sample"].copy()

    MODEL_ORDER = ["OLS", "OLS_LAD", "XGBoost_MSE", "XGBoost_LAD", "RF_MSE", "RF_LAD"]
    MODEL_LABELS = ["OLS\n(MSE)", "OLS\n(LAD)", "XGB\n(MSE)", "XGB\n(LAD)", "RF\n(MSE)", "RF\n(LAD)"]
    COLORS_AAPL = "#2563eb"
    COLORS_COIN = "#dc2626"

    fig1 = plt.figure(figsize=(22, 14))
    gs1 = gridspec.GridSpec(2, 3, figure=fig1, wspace=0.30, hspace=0.45)

    metrics = [
        ("MAE", "MAE (bps) — lower is better"),
        ("MedAE", "Median AE (bps) — lower is better"),
        ("RMSE", "RMSE (bps) — lower is better"),
        ("R2", "R² — higher is better"),
    ]

    for idx, (col, ylabel) in enumerate(metrics):
        row, c = divmod(idx, 3) if idx < 3 else (1, idx - 3)
        if idx == 3:
            row, c = 1, 0
        elif idx < 3:
            row, c = 0, idx
        ax = fig1.add_subplot(gs1[row, c])

        x = np.arange(len(MODEL_ORDER))
        w = 0.35

        vals_aapl = []
        vals_coin = []
        for m in MODEL_ORDER:
            v_a = oos[(oos["ticker"] == "AAPL") & (oos["model"] == m)][col].values
            v_c = oos[(oos["ticker"] == "COIN") & (oos["model"] == m)][col].values
            vals_aapl.append(v_a[0] if len(v_a) > 0 else np.nan)
            vals_coin.append(v_c[0] if len(v_c) > 0 else np.nan)

        bars_a = ax.bar(x - w/2, vals_aapl, w, color=COLORS_AAPL, alpha=0.8, label="AAPL",
                        edgecolor="white", linewidth=0.5)
        bars_c = ax.bar(x + w/2, vals_coin, w, color=COLORS_COIN, alpha=0.8, label="COIN",
                        edgecolor="white", linewidth=0.5)

        # Value labels
        for bar, val in zip(bars_a, vals_aapl):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 * abs(bar.get_height()),
                        f"{val:.3f}" if abs(val) < 1 else f"{val:.2f}",
                        ha="center", va="bottom", fontsize=7, fontweight="bold", color=COLORS_AAPL)
        for bar, val in zip(bars_c, vals_coin):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 * abs(bar.get_height()),
                        f"{val:.3f}" if abs(val) < 1 else f"{val:.2f}",
                        ha="center", va="bottom", fontsize=7, fontweight="bold", color=COLORS_COIN)

        ax.set_xticks(x)
        ax.set_xticklabels(MODEL_LABELS, fontsize=8.5)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f"Out-of-Sample {col}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Highlight best
        if col == "R2":
            best_a = max(vals_aapl)
            best_c = max(vals_coin)
        else:
            best_a = min(v for v in vals_aapl if not np.isnan(v))
            best_c = min(v for v in vals_coin if not np.isnan(v))
        for i, (va, vc) in enumerate(zip(vals_aapl, vals_coin)):
            if va == best_a:
                bars_a[i].set_edgecolor("#000000")
                bars_a[i].set_linewidth(2)
            if vc == best_c:
                bars_c[i].set_edgecolor("#000000")
                bars_c[i].set_linewidth(2)

    # Panel 5: In-sample vs OOS MAE (overfitting check)
    ax5 = fig1.add_subplot(gs1[1, 1])
    ins = ols[ols["sample"] == "in_sample"]

    for i, m in enumerate(MODEL_ORDER):
        mae_in_a = ins[(ins["ticker"] == "AAPL") & (ins["model"] == m)]["MAE"].values
        mae_oos_a = oos[(oos["ticker"] == "AAPL") & (oos["model"] == m)]["MAE"].values
        if len(mae_in_a) > 0 and len(mae_oos_a) > 0:
            ax5.scatter(mae_in_a[0], mae_oos_a[0], s=100, color=COLORS_AAPL, zorder=3,
                        edgecolors="white", linewidth=1)
            ax5.annotate(MODEL_ORDER[i].replace("_", "\n"), xy=(mae_in_a[0], mae_oos_a[0]),
                         textcoords="offset points", xytext=(8, -4), fontsize=6.5, color=COLORS_AAPL)

        mae_in_c = ins[(ins["ticker"] == "COIN") & (ins["model"] == m)]["MAE"].values
        mae_oos_c = oos[(oos["ticker"] == "COIN") & (oos["model"] == m)]["MAE"].values
        if len(mae_in_c) > 0 and len(mae_oos_c) > 0:
            ax5.scatter(mae_in_c[0], mae_oos_c[0], s=100, color=COLORS_COIN, zorder=3,
                        edgecolors="white", linewidth=1, marker="s")
            ax5.annotate(MODEL_ORDER[i].replace("_", "\n"), xy=(mae_in_c[0], mae_oos_c[0]),
                         textcoords="offset points", xytext=(8, -4), fontsize=6.5, color=COLORS_COIN)

    ax5.plot([0, 10], [0, 10], color="black", lw=1, ls="--", alpha=0.4, label="No overfitting line")
    ax5.set_xlabel("In-sample MAE (bps)", fontsize=10)
    ax5.set_ylabel("Out-of-sample MAE (bps)", fontsize=10)
    ax5.set_title("Overfitting Check: In-sample vs OOS MAE\n(above diagonal = overfitting)",
                  fontsize=11, fontweight="bold")
    ax5.legend(fontsize=8, handles=[
        Patch(facecolor=COLORS_AAPL, label="AAPL"),
        Patch(facecolor=COLORS_COIN, label="COIN"),
        plt.Line2D([0], [0], color="black", ls="--", lw=1, label="No overfit"),
    ])
    ax5.grid(True, alpha=0.15)
    ax5.spines["top"].set_visible(False)
    ax5.spines["right"].set_visible(False)

    # Panel 6: MSE vs LAD objective comparison
    ax6 = fig1.add_subplot(gs1[1, 2])

    pairs = [("OLS", "OLS_LAD"), ("XGBoost_MSE", "XGBoost_LAD"), ("RF_MSE", "RF_LAD")]
    pair_labels = ["OLS", "XGBoost", "RF"]
    pair_colors = ["#7c3aed", "#f59e0b", "#16a34a"]

    for j, (mse_m, lad_m) in enumerate(pairs):
        for ticker, marker in [("AAPL", "o"), ("COIN", "s")]:
            mse_mae = oos[(oos["ticker"] == ticker) & (oos["model"] == mse_m)]["MAE"].values
            lad_mae = oos[(oos["ticker"] == ticker) & (oos["model"] == lad_m)]["MAE"].values
            if len(mse_mae) > 0 and len(lad_mae) > 0:
                ax6.scatter(mse_mae[0], lad_mae[0], s=120, color=pair_colors[j], marker=marker,
                            edgecolors="white", linewidth=1, zorder=3)
                ax6.annotate(f"{pair_labels[j]}\n{ticker}", xy=(mse_mae[0], lad_mae[0]),
                             textcoords="offset points", xytext=(8, -4), fontsize=6.5,
                             color=pair_colors[j])

    lim = [1.2, 4.5]
    ax6.plot(lim, lim, color="black", lw=1, ls="--", alpha=0.4, label="Equal MAE")
    ax6.set_xlabel("MSE objective — OOS MAE (bps)", fontsize=10)
    ax6.set_ylabel("LAD objective — OOS MAE (bps)", fontsize=10)
    ax6.set_title("MSE vs LAD Objective: OOS MAE\n(below diagonal = LAD wins)",
                  fontsize=11, fontweight="bold")
    ax6.legend(fontsize=8, handles=[
        Patch(facecolor=pair_colors[0], label="OLS"),
        Patch(facecolor=pair_colors[1], label="XGBoost"),
        Patch(facecolor=pair_colors[2], label="RF"),
        plt.Line2D([0], [0], color="black", ls="--", lw=1, label="Equal"),
    ])
    ax6.grid(True, alpha=0.15)
    ax6.spines["top"].set_visible(False)
    ax6.spines["right"].set_visible(False)

    fig1.suptitle(
        "Model Comparison Dashboard: Out-of-Sample Performance on |impact_vwap_bps|\n"
        "AAPL (35K train, 9.2K test)  |  COIN (15.7K train, 959 test)  |  6 features  |  "
        "Bold border = best per ticker",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
    print("Saved -> model_comparison.png")

    # =============================================================================
    # FIGURE 2: GAMLSS Calibration Overlay
    # =============================================================================
    print("Building Figure 2: GAMLSS Calibration Overlay...", flush=True)

    # Recompute calibration curves from the raw data
    df_tr_a = pd.read_parquet("data/lit_buy_features_v2.parquet")
    df_te_a = pd.read_parquet("data/lit_buy_features_v2_sep.parquet")
    df_tr_c = pd.read_parquet("data/coin_lit_buy_features_train.parquet")
    df_te_c = pd.read_parquet("data/coin_lit_buy_features_test.parquet")

    for df in [df_tr_a, df_te_a, df_tr_c, df_te_c]:
        df["abs_impact"] = df["impact_vwap_bps"].abs()
    df_tr_a = df_tr_a.sort_values("date").reset_index(drop=True)
    df_tr_c = df_tr_c.sort_values("date").reset_index(drop=True)

    cal_levels = np.linspace(0.05, 0.99, 50)

    def laplace_coverage(y, mu, b, level):
        z = np.log(1.0 / (1.0 - level))
        lo = np.maximum(mu - z * b, 0.0)
        hi = mu + z * b
        return ((y >= lo) & (y <= hi)).mean()

    def fit_linear_gamlss(X_tr, y_tr, X_te, y_te):
        """LAD location + OLS scale (same as gamlss_laplace.py)."""
        X_tr_c = sm.add_constant(X_tr)
        X_te_c = sm.add_constant(X_te)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            qr = sm.QuantReg(y_tr, X_tr_c).fit(q=0.5, max_iter=5000)
        mu_tr = qr.predict(X_tr_c)
        mu_te = qr.predict(X_te_c)
        abs_r = np.abs(y_tr - mu_tr)
        Xg_tr = np.column_stack([np.ones(len(X_tr)), X_tr])
        Xg_te = np.column_stack([np.ones(len(X_te)), X_te])
        gamma, _, _, _ = np.linalg.lstsq(Xg_tr, abs_r, rcond=None)
        b_te = np.clip(Xg_te @ gamma, 0.1, None)
        return mu_te, b_te

    def fit_xgb_gamlss(X_tr, y_tr, X_te, y_te):
        """XGB LAD location + XGB MSE scale (same as gamlss_xgb.py)."""
        loc = XGBRegressor(objective="reg:absoluteerror", tree_method="hist", verbosity=0,
                           random_state=42, n_jobs=1, max_depth=3, n_estimators=200,
                           learning_rate=0.07, min_child_weight=5, reg_alpha=10, reg_lambda=10)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loc.fit(X_tr, y_tr)
        mu_tr = np.maximum(loc.predict(X_tr), 0.0)
        mu_te = np.maximum(loc.predict(X_te), 0.0)
        abs_r = np.abs(y_tr - mu_tr)
        sc = XGBRegressor(objective="reg:squarederror", tree_method="hist", verbosity=0,
                          random_state=42, n_jobs=1, max_depth=5, n_estimators=50,
                          min_child_weight=20, learning_rate=0.1, reg_alpha=1, reg_lambda=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sc.fit(X_tr, abs_r)
        b_te = np.clip(sc.predict(X_te), 0.1, None)
        return mu_te, b_te

    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 7))

    for ax, (ticker, df_tr, df_te) in zip(axes2, [
        ("AAPL", df_tr_a, df_te_a),
        ("COIN", df_tr_c, df_te_c),
    ]):
        X_tr = df_tr[FEATURES].to_numpy(dtype=np.float64)
        y_tr = df_tr["abs_impact"].to_numpy(dtype=np.float64)
        X_te = df_te[FEATURES].to_numpy(dtype=np.float64)
        y_te = df_te["abs_impact"].to_numpy(dtype=np.float64)

        # Linear GAMLSS
        mu_lin, b_lin = fit_linear_gamlss(X_tr, y_tr, X_te, y_te)
        cal_lin = [laplace_coverage(y_te, mu_lin, b_lin, lv) for lv in cal_levels]

        # XGB GAMLSS
        mu_xgb, b_xgb = fit_xgb_gamlss(X_tr, y_tr, X_te, y_te)
        cal_xgb = [laplace_coverage(y_te, mu_xgb, b_xgb, lv) for lv in cal_levels]

        ax.plot(cal_levels, cal_lin, color="#2563eb", lw=2.5, marker="o", markersize=3,
                label="Linear GAMLSS (QuantReg + OLS scale)", alpha=0.9)
        ax.plot(cal_levels, cal_xgb, color="#dc2626", lw=2.5, marker="s", markersize=3,
                label="XGB GAMLSS (XGB location + XGB scale)", alpha=0.9)
        ax.plot([0, 1], [0, 1], color="black", lw=1.2, ls="--", alpha=0.5,
                label="Perfect calibration")

        # Annotate key levels
        for lv in [0.50, 0.90]:
            cv_l = laplace_coverage(y_te, mu_lin, b_lin, lv)
            cv_x = laplace_coverage(y_te, mu_xgb, b_xgb, lv)
            ax.annotate(f"Linear {cv_l:.1%}", xy=(lv, cv_l), textcoords="offset points",
                        xytext=(-50, 10), fontsize=7.5, color="#2563eb", fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color="#2563eb", lw=0.7))
            ax.annotate(f"XGB {cv_x:.1%}", xy=(lv, cv_x), textcoords="offset points",
                        xytext=(10, -15), fontsize=7.5, color="#dc2626", fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color="#dc2626", lw=0.7))

        ax.set_xlabel("Nominal coverage level", fontsize=11)
        ax.set_ylabel("Actual coverage", fontsize=11)
        ax.set_title(f"{ticker}", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig2.suptitle(
        "GAMLSS Calibration Comparison: Linear (QuantReg location + OLS scale) vs XGBoost (both stages)\n"
        "Closer to diagonal = better calibrated  |  Laplace(mu, b) prediction intervals",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.savefig("gamlss_calibration_overlay.png", dpi=150, bbox_inches="tight")
    print("Saved -> gamlss_calibration_overlay.png")

    # =============================================================================
    # FIGURE 3: EDA Overview
    # =============================================================================
    print("Building Figure 3: EDA Overview...", flush=True)

    fig3 = plt.figure(figsize=(26, 16))
    gs3 = gridspec.GridSpec(3, 4, figure=fig3, wspace=0.30, hspace=0.45)

    rng = np.random.default_rng(42)

    # -- Row 1, Col 1-2: Target distribution (AAPL and COIN) ---------------------
    for col_idx, (ticker, df_tr, df_te) in enumerate([
        ("AAPL", df_tr_a, df_te_a), ("COIN", df_tr_c, df_te_c)
    ]):
        ax = fig3.add_subplot(gs3[0, col_idx])
        y_all = pd.concat([df_tr["abs_impact"], df_te["abs_impact"]])
        clip = np.percentile(y_all, 98)
        ax.hist(df_tr["abs_impact"].clip(upper=clip), bins=80, density=True, alpha=0.6,
                color="#2563eb", edgecolor="white", linewidth=0.3, label="Train")
        ax.hist(df_te["abs_impact"].clip(upper=clip), bins=80, density=True, alpha=0.5,
                color="#dc2626", edgecolor="white", linewidth=0.3, label="Test")
        ax.axvline(df_tr["abs_impact"].median(), color="#2563eb", lw=1.5, ls="--",
                   label=f"Train median: {df_tr['abs_impact'].median():.2f}")
        ax.axvline(df_te["abs_impact"].median(), color="#dc2626", lw=1.5, ls="--",
                   label=f"Test median: {df_te['abs_impact'].median():.2f}")
        ax.set_xlabel("|impact_vwap_bps|", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title(f"{ticker} Target Distribution\n(clipped at p98={clip:.1f} bps)",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=7.5)
        ax.grid(True, alpha=0.15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # -- Row 1, Col 3: Target stats table ----------------------------------------
    ax_table = fig3.add_subplot(gs3[0, 2])
    ax_table.axis("off")

    stats_data = []
    for ticker, df_tr, df_te in [("AAPL", df_tr_a, df_te_a), ("COIN", df_tr_c, df_te_c)]:
        for label, df in [("Train", df_tr), ("Test", df_te)]:
            y = df["abs_impact"]
            stats_data.append([
                f"{ticker} {label}", f"{len(df):,}", f"{y.mean():.2f}", f"{y.median():.2f}",
                f"{y.std():.2f}", f"{y.skew():.1f}", f"{y.kurtosis():.0f}",
            ])

    col_labels = ["Dataset", "N", "Mean", "Median", "Std", "Skew", "Kurt"]
    table = ax_table.table(cellText=stats_data, colLabels=col_labels,
                           cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#e0e7ff")
            cell.set_text_props(fontweight="bold")
        cell.set_edgecolor("#d1d5db")
    ax_table.set_title("Target Summary Statistics\n|impact_vwap_bps|",
                        fontsize=11, fontweight="bold", pad=20)

    # -- Row 1, Col 4: Train/test time split -------------------------------------
    ax_ts = fig3.add_subplot(gs3[0, 3])
    for ticker, df_tr, df_te, color in [
        ("AAPL", df_tr_a, df_te_a, "#2563eb"),
        ("COIN", df_tr_c, df_te_c, "#dc2626"),
    ]:
        tr_dates = pd.to_datetime(df_tr["date"])
        te_dates = pd.to_datetime(df_te["date"])
        tr_daily = tr_dates.dt.date.value_counts().sort_index()
        te_daily = te_dates.dt.date.value_counts().sort_index()
        ax_ts.bar(tr_daily.index, tr_daily.values, color=color, alpha=0.4, width=1, label=f"{ticker} train")
        ax_ts.bar(te_daily.index, te_daily.values, color=color, alpha=0.8, width=1, label=f"{ticker} test")
    ax_ts.set_xlabel("Date", fontsize=10)
    ax_ts.set_ylabel("Trades per day", fontsize=10)
    ax_ts.set_title("Temporal Train/Test Split\n(walk-forward holdout)", fontsize=11, fontweight="bold")
    ax_ts.legend(fontsize=7.5)
    ax_ts.tick_params(axis="x", rotation=30, labelsize=7)
    ax_ts.grid(axis="y", alpha=0.2)
    ax_ts.spines["top"].set_visible(False)
    ax_ts.spines["right"].set_visible(False)

    # -- Row 2: Feature distributions (AAPL train) --------------------------------
    for i, (feat, short) in enumerate(zip(FEATURES, FEAT_SHORT)):
        ax = fig3.add_subplot(gs3[1, i]) if i < 4 else None
        if ax is None:
            break
        vals_a = df_tr_a[feat].values
        vals_c = df_tr_c[feat].values
        clip_lo = np.percentile(np.concatenate([vals_a, vals_c]), 1)
        clip_hi = np.percentile(np.concatenate([vals_a, vals_c]), 99)
        ax.hist(np.clip(vals_a, clip_lo, clip_hi), bins=60, density=True, alpha=0.6,
                color="#2563eb", edgecolor="white", linewidth=0.3, label="AAPL")
        ax.hist(np.clip(vals_c, clip_lo, clip_hi), bins=60, density=True, alpha=0.5,
                color="#dc2626", edgecolor="white", linewidth=0.3, label="COIN")
        ax.set_xlabel(feat, fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.set_title(f"{feat}", fontsize=10, fontweight="bold")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # -- Row 3, Col 1: Feature correlation heatmap (AAPL) -------------------------
    ax_corr = fig3.add_subplot(gs3[2, 0])
    corr = df_tr_a[FEATURES + ["abs_impact"]].corr()
    im = ax_corr.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    labels_corr = FEAT_SHORT + ["impact"]
    ax_corr.set_xticks(range(len(labels_corr)))
    ax_corr.set_xticklabels(labels_corr, fontsize=7.5, rotation=45, ha="right")
    ax_corr.set_yticks(range(len(labels_corr)))
    ax_corr.set_yticklabels(labels_corr, fontsize=7.5)
    for i2 in range(len(labels_corr)):
        for j2 in range(len(labels_corr)):
            ax_corr.text(j2, i2, f"{corr.iloc[i2, j2]:.2f}", ha="center", va="center",
                         fontsize=6.5, color="white" if abs(corr.iloc[i2, j2]) > 0.5 else "black")
    plt.colorbar(im, ax=ax_corr, fraction=0.046, pad=0.04)
    ax_corr.set_title("AAPL Feature Correlations\n(Pearson)", fontsize=10, fontweight="bold")

    # -- Row 3, Col 2: Feature correlation heatmap (COIN) -------------------------
    ax_corr2 = fig3.add_subplot(gs3[2, 1])
    corr2 = df_tr_c[FEATURES + ["abs_impact"]].corr()
    im2 = ax_corr2.imshow(corr2, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax_corr2.set_xticks(range(len(labels_corr)))
    ax_corr2.set_xticklabels(labels_corr, fontsize=7.5, rotation=45, ha="right")
    ax_corr2.set_yticks(range(len(labels_corr)))
    ax_corr2.set_yticklabels(labels_corr, fontsize=7.5)
    for i2 in range(len(labels_corr)):
        for j2 in range(len(labels_corr)):
            ax_corr2.text(j2, i2, f"{corr2.iloc[i2, j2]:.2f}", ha="center", va="center",
                          fontsize=6.5, color="white" if abs(corr2.iloc[i2, j2]) > 0.5 else "black")
    plt.colorbar(im2, ax=ax_corr2, fraction=0.046, pad=0.04)
    ax_corr2.set_title("COIN Feature Correlations\n(Pearson)", fontsize=10, fontweight="bold")

    # -- Row 3, Col 3: Scatter: participation_rate vs impact ----------------------
    ax_sc1 = fig3.add_subplot(gs3[2, 2])
    samp_a = rng.choice(len(df_tr_a), size=min(3000, len(df_tr_a)), replace=False)
    samp_c = rng.choice(len(df_tr_c), size=min(3000, len(df_tr_c)), replace=False)
    ax_sc1.scatter(df_tr_a["participation_rate"].iloc[samp_a],
                   df_tr_a["abs_impact"].iloc[samp_a], s=6, alpha=0.1,
                   color="#2563eb", linewidths=0, rasterized=True, label="AAPL")
    ax_sc1.scatter(df_tr_c["participation_rate"].iloc[samp_c],
                   df_tr_c["abs_impact"].iloc[samp_c], s=6, alpha=0.1,
                   color="#dc2626", linewidths=0, rasterized=True, label="COIN")
    clip_y = max(np.percentile(df_tr_a["abs_impact"], 98), np.percentile(df_tr_c["abs_impact"], 98))
    ax_sc1.set_ylim(0, clip_y)
    ax_sc1.set_xlabel("participation_rate", fontsize=10)
    ax_sc1.set_ylabel("|impact_vwap_bps|", fontsize=10)
    ax_sc1.set_title("Participation Rate vs Impact\n(3K random trades each)",
                     fontsize=10, fontweight="bold")
    ax_sc1.legend(fontsize=8, markerscale=5)
    ax_sc1.grid(True, alpha=0.15)
    ax_sc1.spines["top"].set_visible(False)
    ax_sc1.spines["right"].set_visible(False)

    # -- Row 3, Col 4: Scatter: roll_vol_500 vs impact ----------------------------
    ax_sc2 = fig3.add_subplot(gs3[2, 3])
    ax_sc2.scatter(df_tr_a["roll_vol_500"].iloc[samp_a],
                   df_tr_a["abs_impact"].iloc[samp_a], s=6, alpha=0.1,
                   color="#2563eb", linewidths=0, rasterized=True, label="AAPL")
    ax_sc2.scatter(df_tr_c["roll_vol_500"].iloc[samp_c],
                   df_tr_c["abs_impact"].iloc[samp_c], s=6, alpha=0.1,
                   color="#dc2626", linewidths=0, rasterized=True, label="COIN")
    ax_sc2.set_ylim(0, clip_y)
    ax_sc2.set_xlabel("roll_vol_500", fontsize=10)
    ax_sc2.set_ylabel("|impact_vwap_bps|", fontsize=10)
    ax_sc2.set_title("Rolling Volatility vs Impact\n(3K random trades each)",
                     fontsize=10, fontweight="bold")
    ax_sc2.legend(fontsize=8, markerscale=5)
    ax_sc2.grid(True, alpha=0.15)
    ax_sc2.spines["top"].set_visible(False)
    ax_sc2.spines["right"].set_visible(False)

    fig3.suptitle(
        "Data Exploration: AAPL and COIN Lit Buy Block Trades\n"
        "Target = |impact_vwap_bps|  |  6 features  |  Jun–Sep 2024",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.savefig("eda_overview.png", dpi=150, bbox_inches="tight")
    print("Saved -> eda_overview.png")

    print("\nAll visualizations complete!")


if __name__ == "__main__":
    run_fetch_trades()
    run_get_exchanges()
    run_block_trades()
    run_check_exchange()
    run_cross_stock_pipeline()
    run_summary_visualizations()
