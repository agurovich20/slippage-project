"""
Build feature table v2 for AAPL lit-only buy block trades.

Changes vs v1:
  - daily_roll_spread    → roll_spread_500  (Roll 1984, last 500 ticks before the trade)
  - trail_1min_volatility → roll_vol_500   (std of tick-to-tick price changes, last 500 ticks)

Both are computed from the 500 trades (ticks) immediately preceding each block trade in
the full tick stream — not from the full day or a fixed time window.

Unchanged features:
  dollar_value, log_dollar_value, participation_rate (still uses trailing 1-min volume),
  time_of_day, exchange_id, day_of_week

Target: impact_vwap_bps (signed)

Output: data/lit_buy_features_v2.parquet
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd

# ── Config ─────────────────────────────────────────────────────────────────────
DARK_IDS   = {4, 6, 16, 62, 201, 202, 203}
ONE_MIN_NS = 60 * 1_000_000_000
AAPL_MID   = 190.0
OPEN_SEC   = 9 * 3600 + 30 * 60
WINDOW     = 500   # number of prior ticks for rolling estimates
MIN_TICKS  = 10    # minimum ticks needed to produce an estimate

# ── 1. Load AAPL lit buy block trades ─────────────────────────────────────────
bt = pd.read_parquet(
    "data/block_trades.parquet",
    columns=["ticker", "date", "timestamp_ns", "price", "size",
             "dollar_value", "exchange", "impact_vwap_bps", "side_label"],
    filters=[("ticker", "==", "AAPL")],
)

buys = bt[
    (~bt["exchange"].isin(DARK_IDS)) &
    (bt["side_label"] == "buy") &
    bt["impact_vwap_bps"].notna()
].copy()
buys = buys.sort_values(["date", "timestamp_ns"]).reset_index(drop=True)
print(f"{len(buys):,} AAPL lit buy block trades across {buys['date'].nunique()} days")

# ── 2. time_of_day (seconds since 9:30 ET) ───────────────────────────────────
et_times = (
    pd.to_datetime(buys["timestamp_ns"], unit="ns", utc=True)
    .dt.tz_convert("America/New_York")
)
buys["time_of_day"] = (
    et_times.dt.hour * 3600
    + et_times.dt.minute * 60
    + et_times.dt.second
    - OPEN_SEC
)

# ── 3. day_of_week ────────────────────────────────────────────────────────────
buys["day_of_week"] = pd.to_datetime(buys["date"]).dt.dayofweek

# ── 4. Per-day: compute rolling 500-trade features + trailing 1-min volume ────
roll_spread_arr = np.full(len(buys), np.nan)
roll_vol_arr    = np.full(len(buys), np.nan)
trail_vol_arr   = np.full(len(buys), np.nan)

days_processed = 0
days_missing   = 0


def roll_spread_500(px_window):
    """
    Roll (1984) spread from a price window.
    dp = first differences; cov1 = first-order autocovariance of dp.
    spread = 2*sqrt(-cov1)  if cov1 < 0, else NaN.
    Returns spread in same units as price (dollars).
    """
    dp = np.diff(px_window.astype(np.float64))
    if len(dp) < 2:
        return np.nan
    # First-order autocovariance using numpy (unbiased)
    cov_mat = np.cov(dp[1:], dp[:-1], ddof=1)
    cov1 = cov_mat[0, 1]
    if cov1 < 0:
        return 2.0 * np.sqrt(-cov1)
    return np.nan


for date, grp in buys.groupby("date"):
    tick_path = f"data/AAPL/{date}.parquet"
    if not os.path.exists(tick_path):
        days_missing += 1
        continue

    ticks = pd.read_parquet(tick_path, columns=["sip_timestamp", "price", "size"])
    ticks = ticks.sort_values("sip_timestamp").reset_index(drop=True)

    ts = ticks["sip_timestamp"].to_numpy(dtype=np.int64)
    px = ticks["price"].to_numpy(dtype=np.float64)
    sz = ticks["size"].to_numpy(dtype=np.int64)
    N  = len(ts)

    # Prefix sums for trailing 1-min volume (unchanged from v1)
    cum_sz = np.empty(N + 1, dtype=np.int64)
    cum_sz[0] = 0
    np.cumsum(sz, out=cum_sz[1:])

    block_ts = grp["timestamp_ns"].to_numpy(dtype=np.int64)

    # For each block trade, find its position in the tick array
    # hi_idx = first tick at or after the block trade timestamp
    hi_idx = np.searchsorted(ts, block_ts, side="left")

    # Trailing 1-min volume
    lo_idx_1min = np.searchsorted(ts, block_ts - ONE_MIN_NS, side="left")
    vol_sh = (cum_sz[hi_idx] - cum_sz[lo_idx_1min]).astype(float)
    vol_sh = np.where(hi_idx > lo_idx_1min, vol_sh, np.nan)
    trail_vol_arr[grp.index] = vol_sh

    # Rolling 500-trade Roll spread and realized vol
    for i, (idx, hi) in enumerate(zip(grp.index, hi_idx)):
        lo = max(0, hi - WINDOW)
        window_len = hi - lo
        if window_len < MIN_TICKS:
            continue

        px_win = px[lo:hi]

        # Roll spread (bps)
        spread_raw = roll_spread_500(px_win)
        if not np.isnan(spread_raw):
            roll_spread_arr[idx] = spread_raw / AAPL_MID * 1e4

        # Realized vol: std of tick-to-tick price changes (bps)
        dp_win = np.diff(px_win)
        if len(dp_win) >= 2:
            roll_vol_arr[idx] = dp_win.std(ddof=1) / AAPL_MID * 1e4

    days_processed += 1

print(f"  Processed {days_processed} days  |  missing tick files: {days_missing}")

buys["roll_spread_500"]    = roll_spread_arr
buys["roll_vol_500"]       = roll_vol_arr
buys["trail_1min_volume"]  = trail_vol_arr

# ── 5. participation_rate ─────────────────────────────────────────────────────
buys["participation_rate"] = buys["size"] / buys["trail_1min_volume"]

# ── 6. Derived features ───────────────────────────────────────────────────────
buys["log_dollar_value"] = np.log(buys["dollar_value"])

# ── 7. Assemble final feature table ───────────────────────────────────────────
FEATURES = [
    "dollar_value",
    "log_dollar_value",
    "participation_rate",
    "roll_spread_500",
    "roll_vol_500",
    "time_of_day",
    "exchange_id",
    "day_of_week",
]
TARGET = "impact_vwap_bps"

buys = buys.rename(columns={"exchange": "exchange_id"})
feat_df = buys[["date"] + FEATURES + [TARGET]].copy()

# ── 8. Drop rows missing any feature or target ────────────────────────────────
n_raw = len(feat_df)
feat_df = feat_df.dropna(subset=FEATURES + [TARGET])
feat_df = feat_df[feat_df["participation_rate"] <= 1.0].copy()
feat_df = feat_df[feat_df["participation_rate"] > 0].copy()
feat_df = feat_df[feat_df["roll_vol_500"] > 0].copy()
feat_df = feat_df.reset_index(drop=True)

print(f"\nRows: {n_raw:,} raw -> {len(feat_df):,} after dropping NaN/invalid")

# ── 9. Save ───────────────────────────────────────────────────────────────────
out_path = "data/lit_buy_features_v2.parquet"
feat_df.to_parquet(out_path, index=False, compression="snappy")
print(f"Saved -> {out_path}")

# ── 10. Summary stats ─────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Shape: {feat_df.shape}  ({feat_df.shape[0]:,} rows x {feat_df.shape[1]} cols)")
print(f"{'='*60}")

num_df = feat_df.drop(columns=["date"])
print("\n--- Summary statistics ---")
desc = num_df.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
desc.columns = [c.replace("%", "p") for c in desc.columns]
with pd.option_context("display.float_format", "{:.4f}".format,
                        "display.max_columns", 20, "display.width", 120):
    print(desc.to_string())

print(f"\n--- Pearson correlations with {TARGET} ---")
corr = num_df.corr()[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
print(corr.to_string(float_format="{:+.4f}".format))

print(f"\n--- Compare new vs old feature coverage ---")
print(f"  roll_spread_500 non-NaN: {(~np.isnan(roll_spread_arr)).sum():,} / {len(buys):,}")
print(f"  roll_vol_500    non-NaN: {(~np.isnan(roll_vol_arr)).sum():,} / {len(buys):,}")
