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

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from polygon import RESTClient
from xgboost import XGBRegressor

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

def main():
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


if __name__ == "__main__":
    main()
