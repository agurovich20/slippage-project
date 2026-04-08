"""
Build feature table for AAPL lit-only buy block trades.

Features:
  dollar_value          — raw notional ($)
  log_dollar_value      — natural log of dollar_value
  participation_rate    — block size / trailing 1-min volume (shares)
  daily_roll_spread     — Roll (1984) spread estimate for that day (bps)
  trail_1min_volatility — std of tick-to-tick price changes in prior 1-min window (bps)
  time_of_day           — seconds since 9:30 ET
  exchange_id           — Polygon exchange ID
  day_of_week           — 0=Monday … 4=Friday

Target:
  impact_vwap_bps       — signed VWAP-bar impact (positive = price rose)

Output: data/lit_buy_features.parquet
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd

# ── Config ─────────────────────────────────────────────────────────────────────
DARK_IDS    = {4, 6, 16, 62, 201, 202, 203}   # exchange IDs treated as dark
ONE_MIN_NS  = 60 * 1_000_000_000
AAPL_MID    = 190.0          # reference price for bps conversion of vol
OPEN_SEC    = 9 * 3600 + 30 * 60   # 9:30 ET in seconds since midnight

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

# ── 3. day_of_week (0=Mon … 4=Fri) ───────────────────────────────────────────
buys["day_of_week"] = pd.to_datetime(buys["date"]).dt.dayofweek

# ── 4. Per-day: trailing 1-min volume and volatility via prefix sums ──────────
trail_vol_shares = np.full(len(buys), np.nan)
trail_vol_bps    = np.full(len(buys), np.nan)

days_processed = 0
days_missing   = 0

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

    # prefix sums for volume
    cum_sz = np.empty(N + 1, dtype=np.int64)
    cum_sz[0] = 0
    np.cumsum(sz, out=cum_sz[1:])

    # prefix sums for tick-to-tick price diffs
    diffs  = np.diff(px)                       # length N-1
    cum_d  = np.empty(N, dtype=np.float64);  cum_d[0]  = 0.0
    cum_d2 = np.empty(N, dtype=np.float64);  cum_d2[0] = 0.0
    np.cumsum(diffs,    out=cum_d[1:])
    np.cumsum(diffs**2, out=cum_d2[1:])

    block_ts = grp["timestamp_ns"].to_numpy(dtype=np.int64)
    lo_idx   = np.searchsorted(ts, block_ts - ONE_MIN_NS, side="left")
    hi_idx   = np.searchsorted(ts, block_ts,              side="left")

    # volume in window [lo, hi)
    vol_sh = (cum_sz[hi_idx] - cum_sz[lo_idx]).astype(float)
    vol_sh = np.where(hi_idx > lo_idx, vol_sh, np.nan)

    # volatility: sample std of price diffs in window [lo, hi)
    n_diffs = (hi_idx - lo_idx - 1).astype(float)
    sum_d   = cum_d[hi_idx - 1]  - cum_d[lo_idx]
    sum_d2  = cum_d2[hi_idx - 1] - cum_d2[lo_idx]

    mask_ok  = (n_diffs >= 2) & (hi_idx > 0)
    variance = np.where(
        mask_ok,
        (sum_d2 - sum_d**2 / np.where(n_diffs > 0, n_diffs, 1))
        / np.where(n_diffs > 1, n_diffs - 1, 1),
        np.nan,
    )
    variance = np.where(variance < 0, 0.0, variance)
    px_std   = np.where(mask_ok, np.sqrt(variance), np.nan)

    trail_vol_shares[grp.index] = vol_sh
    trail_vol_bps[grp.index]    = px_std / AAPL_MID * 1e4

    days_processed += 1

print(f"  Processed {days_processed} days  |  missing tick files: {days_missing}")

buys["trail_1min_volume"]     = trail_vol_shares
buys["trail_1min_volatility"] = trail_vol_bps

# ── 5. participation_rate ─────────────────────────────────────────────────────
buys["participation_rate"] = buys["size"] / buys["trail_1min_volume"]

# ── 6. daily Roll spread ──────────────────────────────────────────────────────
spread_df = pd.read_parquet("data/aapl_daily_spread.parquet",
                            columns=["date", "spread_bps"])
buys = buys.merge(spread_df.rename(columns={"spread_bps": "daily_roll_spread"}),
                  on="date", how="left")

# ── 7. Derived features ───────────────────────────────────────────────────────
buys["log_dollar_value"] = np.log(buys["dollar_value"])

# ── 8. Assemble final feature table ───────────────────────────────────────────
FEATURES = [
    "dollar_value",
    "log_dollar_value",
    "participation_rate",
    "daily_roll_spread",
    "trail_1min_volatility",
    "time_of_day",
    "exchange_id",
    "day_of_week",
]
TARGET = "impact_vwap_bps"

# rename exchange -> exchange_id
buys = buys.rename(columns={"exchange": "exchange_id"})

feat_df = buys[["date"] + FEATURES + [TARGET]].copy()

# ── 9. Drop rows missing any feature or target ────────────────────────────────
n_raw = len(feat_df)
feat_df = feat_df.dropna(subset=FEATURES + [TARGET])

# also drop participation_rate > 1 (data anomaly: block > market volume window)
feat_df = feat_df[feat_df["participation_rate"] <= 1.0].copy()
feat_df = feat_df[feat_df["participation_rate"] > 0].copy()
feat_df = feat_df[feat_df["trail_1min_volatility"] > 0].copy()
feat_df = feat_df.reset_index(drop=True)

print(f"\nRows: {n_raw:,} raw -> {len(feat_df):,} after dropping NaN/invalid")

# ── 10. Save ───────────────────────────────────────────────────────────────────
out_path = "data/lit_buy_features.parquet"
feat_df.to_parquet(out_path, index=False, compression="snappy")
print(f"Saved → {out_path}")

# ── 11. Shape + summary stats ─────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Shape: {feat_df.shape}  ({feat_df.shape[0]:,} rows × {feat_df.shape[1]} cols)")
print(f"{'='*60}")

num_df = feat_df.drop(columns=["date"])
print("\n--- Summary statistics ---")
desc = num_df.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
desc.columns = [c.replace("%", "p") for c in desc.columns]
with pd.option_context("display.float_format", "{:.4f}".format,
                        "display.max_columns", 20, "display.width", 120):
    print(desc.to_string())

# ── 12. Correlation matrix: features vs target ────────────────────────────────
print(f"\n--- Pearson correlations with {TARGET} ---")
corr = num_df.corr()[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
print(corr.to_string(float_format="{:+.4f}".format))

print(f"\n--- Full feature x feature + target correlation matrix ---")
full_corr = num_df.corr().round(3)
with pd.option_context("display.max_columns", 20, "display.width", 140,
                        "display.float_format", "{:+.3f}".format):
    print(full_corr.to_string())
