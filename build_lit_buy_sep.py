"""
Build feature table for AAPL lit-only buy block trades — September 2024 only.

Same logic as build_lit_buy_features_v2.py; filters to September 2024 dates.
Output: data/lit_buy_features_v2_sep.parquet
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd

DARK_IDS   = {4, 6, 16, 62, 201, 202, 203}
ONE_MIN_NS = 60 * 1_000_000_000
AAPL_MID   = 190.0    # same reference as training data for feature-scale consistency
OPEN_SEC   = 9 * 3600 + 30 * 60
WINDOW     = 500
MIN_TICKS  = 10

# ── Load September AAPL lit buy block trades ───────────────────────────────────
bt = pd.read_parquet(
    "data/block_trades.parquet",
    columns=["ticker", "date", "timestamp_ns", "price", "size",
             "dollar_value", "exchange", "impact_vwap_bps", "side_label"],
    filters=[
        ("ticker",     "==", "AAPL"),
        ("date",       ">=", "2024-09-01"),
        ("date",       "<=", "2024-09-30"),
    ],
)

buys = bt[
    (~bt["exchange"].isin(DARK_IDS)) &
    (bt["side_label"] == "buy") &
    bt["impact_vwap_bps"].notna()
].copy()
buys = buys.sort_values(["date", "timestamp_ns"]).reset_index(drop=True)
print(f"{len(buys):,} AAPL lit buy block trades in September across {buys['date'].nunique()} days")

# ── time_of_day ────────────────────────────────────────────────────────────────
et_times = (
    pd.to_datetime(buys["timestamp_ns"], unit="ns", utc=True)
    .dt.tz_convert("America/New_York")
)
buys["time_of_day"] = (
    et_times.dt.hour * 3600 + et_times.dt.minute * 60 + et_times.dt.second - OPEN_SEC
)

# ── day_of_week ────────────────────────────────────────────────────────────────
buys["day_of_week"] = pd.to_datetime(buys["date"]).dt.dayofweek

# ── Rolling 500-trade features + trailing 1-min volume ────────────────────────
roll_spread_arr = np.full(len(buys), np.nan)
roll_vol_arr    = np.full(len(buys), np.nan)
trail_vol_arr   = np.full(len(buys), np.nan)

days_processed = 0
days_missing   = 0


def roll_spread_500(px_window):
    dp = np.diff(px_window.astype(np.float64))
    if len(dp) < 2:
        return np.nan
    cov_mat = np.cov(dp[1:], dp[:-1], ddof=1)
    cov1 = cov_mat[0, 1]
    if cov1 < 0:
        return 2.0 * np.sqrt(-cov1)
    return np.nan


for date, grp in buys.groupby("date"):
    tick_path = f"data/AAPL/{date}.parquet"
    if not os.path.exists(tick_path):
        days_missing += 1
        print(f"  MISSING tick file: {tick_path}")
        continue

    ticks = pd.read_parquet(tick_path, columns=["sip_timestamp", "price", "size"])
    ticks = ticks.sort_values("sip_timestamp").reset_index(drop=True)

    ts = ticks["sip_timestamp"].to_numpy(dtype=np.int64)
    px = ticks["price"].to_numpy(dtype=np.float64)
    sz = ticks["size"].to_numpy(dtype=np.int64)
    N  = len(ts)

    cum_sz = np.empty(N + 1, dtype=np.int64)
    cum_sz[0] = 0
    np.cumsum(sz, out=cum_sz[1:])

    block_ts = grp["timestamp_ns"].to_numpy(dtype=np.int64)
    hi_idx   = np.searchsorted(ts, block_ts, side="left")

    lo_idx_1min = np.searchsorted(ts, block_ts - ONE_MIN_NS, side="left")
    vol_sh = (cum_sz[hi_idx] - cum_sz[lo_idx_1min]).astype(float)
    vol_sh = np.where(hi_idx > lo_idx_1min, vol_sh, np.nan)
    trail_vol_arr[grp.index] = vol_sh

    for i, (idx, hi) in enumerate(zip(grp.index, hi_idx)):
        lo = max(0, hi - WINDOW)
        window_len = hi - lo
        if window_len < MIN_TICKS:
            continue
        px_win = px[lo:hi]

        spread_raw = roll_spread_500(px_win)
        if not np.isnan(spread_raw):
            roll_spread_arr[idx] = spread_raw / AAPL_MID * 1e4

        dp_win = np.diff(px_win)
        if len(dp_win) >= 2:
            roll_vol_arr[idx] = dp_win.std(ddof=1) / AAPL_MID * 1e4

    days_processed += 1
    print(f"  {date}: {len(grp):,} buys processed")

print(f"  Processed {days_processed} days  |  missing: {days_missing}")

buys["roll_spread_500"]   = roll_spread_arr
buys["roll_vol_500"]      = roll_vol_arr
buys["trail_1min_volume"] = trail_vol_arr
buys["participation_rate"] = buys["size"] / buys["trail_1min_volume"]
buys["log_dollar_value"]   = np.log(buys["dollar_value"])
buys = buys.rename(columns={"exchange": "exchange_id"})

FEATURES = ["dollar_value", "log_dollar_value", "participation_rate",
            "roll_spread_500", "roll_vol_500", "time_of_day", "exchange_id", "day_of_week"]
TARGET   = "impact_vwap_bps"

feat_df = buys[["date"] + FEATURES + [TARGET]].copy()
n_raw = len(feat_df)
feat_df = feat_df.dropna(subset=FEATURES + [TARGET])
feat_df = feat_df[feat_df["participation_rate"].between(0, 1, inclusive="neither")].copy()
feat_df = feat_df[feat_df["roll_vol_500"] > 0].copy()
feat_df = feat_df.reset_index(drop=True)

print(f"\nRows: {n_raw:,} raw -> {len(feat_df):,} after dropping NaN/invalid")

out_path = "data/lit_buy_features_v2_sep.parquet"
feat_df.to_parquet(out_path, index=False, compression="snappy")
print(f"Saved -> {out_path}")

print(f"\nSeptember summary:")
print(f"  Dates: {feat_df['date'].nunique()} trading days  ({feat_df['date'].min()} .. {feat_df['date'].max()})")
print(f"  Trades: {len(feat_df):,}")
print(f"  roll_spread_500: mean={feat_df['roll_spread_500'].mean():.4f}  std={feat_df['roll_spread_500'].std():.4f}")
print(f"  roll_vol_500:    mean={feat_df['roll_vol_500'].mean():.4f}  std={feat_df['roll_vol_500'].std():.4f}")
print(f"  prate:           mean={feat_df['participation_rate'].mean():.4f}  std={feat_df['participation_rate'].std():.4f}")
print(f"  impact_vwap_bps: mean={feat_df['impact_vwap_bps'].mean():+.4f}  std={feat_df['impact_vwap_bps'].std():.4f}")
