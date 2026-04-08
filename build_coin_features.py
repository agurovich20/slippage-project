"""
Build feature table for COIN lit-only buy block trades.

Same logic as build_lit_buy_features_v2.py but for COIN instead of AAPL.
Splits into train (January-August 2024) and test (September 2024).

Output:
  data/coin_lit_buy_features_train.parquet
  data/coin_lit_buy_features_test.parquet

Also prints summary stats and side-by-side comparison with AAPL.
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd

# ── Config ─────────────────────────────────────────────────────────────────────
DARK_IDS   = {4, 6, 16, 62, 201, 202, 203}
ONE_MIN_NS = 60 * 1_000_000_000
COIN_MID   = 222.0     # approximate mid-price for COIN over Jun-Sep 2024
OPEN_SEC   = 9 * 3600 + 30 * 60
WINDOW     = 500
MIN_TICKS  = 10

# ── 1. Load COIN block trades (all dates) ──────────────────────────────────────
bt = pd.read_parquet(
    "data/block_trades.parquet",
    columns=["ticker", "date", "timestamp_ns", "price", "size",
             "dollar_value", "exchange", "impact_vwap_bps", "side_label"],
    filters=[("ticker", "==", "COIN")],
)

# Dark vs lit split for stats
n_total_blocks = len(bt)
n_dark  = bt["exchange"].isin(DARK_IDS).sum()
n_lit   = n_total_blocks - n_dark

print(f"COIN total block trades: {n_total_blocks:,}")
print(f"  Dark: {n_dark:,} ({100*n_dark/n_total_blocks:.1f}%)")
print(f"  Lit:  {n_lit:,} ({100*n_lit/n_total_blocks:.1f}%)")

buys = bt[
    (~bt["exchange"].isin(DARK_IDS)) &
    (bt["side_label"] == "buy") &
    bt["impact_vwap_bps"].notna()
].copy()
buys = buys.sort_values(["date", "timestamp_ns"]).reset_index(drop=True)
print(f"\n{len(buys):,} COIN lit buy block trades across {buys['date'].nunique()} days")

# ── 2. time_of_day (seconds since 9:30 ET) ─────────────────────────────────────
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

# ── 3. day_of_week ──────────────────────────────────────────────────────────────
buys["day_of_week"] = pd.to_datetime(buys["date"]).dt.dayofweek

# ── 4. Per-day: compute rolling 500-trade features + trailing 1-min volume ──────
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
    tick_path = f"data/COIN/{date}.parquet"
    if not os.path.exists(tick_path):
        days_missing += 1
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
    hi_idx = np.searchsorted(ts, block_ts, side="left")

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
            roll_spread_arr[idx] = spread_raw / COIN_MID * 1e4

        dp_win = np.diff(px_win)
        if len(dp_win) >= 2:
            roll_vol_arr[idx] = dp_win.std(ddof=1) / COIN_MID * 1e4

    days_processed += 1

print(f"  Processed {days_processed} days  |  missing tick files: {days_missing}")

buys["roll_spread_500"]    = roll_spread_arr
buys["roll_vol_500"]       = roll_vol_arr
buys["trail_1min_volume"]  = trail_vol_arr

# ── 5. participation_rate ───────────────────────────────────────────────────────
buys["participation_rate"] = buys["size"] / buys["trail_1min_volume"]

# ── 6. Derived features ────────────────────────────────────────────────────────
buys["log_dollar_value"] = np.log(buys["dollar_value"])

# ── 7. Assemble final feature table ────────────────────────────────────────────
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

# ── 8. Drop rows missing any feature or target ─────────────────────────────────
n_raw = len(feat_df)
feat_df = feat_df.dropna(subset=FEATURES + [TARGET])
feat_df = feat_df[feat_df["participation_rate"] <= 1.0].copy()
feat_df = feat_df[feat_df["participation_rate"] > 0].copy()
feat_df = feat_df[feat_df["roll_vol_500"] > 0].copy()
feat_df = feat_df.reset_index(drop=True)

print(f"\nRows: {n_raw:,} raw -> {len(feat_df):,} after dropping NaN/invalid")

# ── 9. Split into train (Jun-Aug) and test (Sep) ──────────────────────────────
train_df = feat_df[feat_df["date"] < "2024-09-01"].copy().reset_index(drop=True)
test_df  = feat_df[feat_df["date"] >= "2024-09-01"].copy().reset_index(drop=True)

train_path = "data/coin_lit_buy_features_train.parquet"
test_path  = "data/coin_lit_buy_features_test.parquet"
train_df.to_parquet(train_path, index=False, compression="snappy")
test_df.to_parquet(test_path, index=False, compression="snappy")
print(f"Saved -> {train_path}  ({len(train_df):,} rows)")
print(f"Saved -> {test_path}   ({len(test_df):,} rows)")

# ── 10. Summary stats ─────────────────────────────────────────────────────────
def print_split_stats(label, df):
    abs_imp = df["impact_vwap_bps"].abs()
    print(f"\n  {label}:")
    print(f"    Trading days : {df['date'].nunique()}")
    print(f"    Date range   : {df['date'].min()} .. {df['date'].max()}")
    print(f"    Lit buy blocks: {len(df):,}")
    print(f"    abs(impact)  : mean={abs_imp.mean():.4f}  median={abs_imp.median():.4f}  std={abs_imp.std():.4f} bps")
    print(f"    roll_spread  : mean={df['roll_spread_500'].mean():.4f} bps")
    print(f"    roll_vol     : mean={df['roll_vol_500'].mean():.4f} bps")
    print(f"    prate        : mean={df['participation_rate'].mean():.4f}")

print(f"\n{'='*70}")
print("COIN FEATURE SUMMARY")
print(f"{'='*70}")
print_split_stats("Train (Jan-Aug 2024)", train_df)
print_split_stats("Test  (Sep 2024)",     test_df)

# ── 11. Side-by-side comparison with AAPL ──────────────────────────────────────
# Load AAPL feature data
aapl_tr = pd.read_parquet("data/lit_buy_features_v2.parquet")
aapl_te = pd.read_parquet("data/lit_buy_features_v2_sep.parquet")

# Also compute AAPL dark/lit/block stats from block_trades
aapl_bt = pd.read_parquet(
    "data/block_trades.parquet",
    columns=["ticker", "exchange", "side_label"],
    filters=[("ticker", "==", "AAPL")],
)
aapl_n_total  = len(aapl_bt)
aapl_n_dark   = aapl_bt["exchange"].isin(DARK_IDS).sum()
aapl_n_lit    = aapl_n_total - aapl_n_dark

def stat_row(label, aapl_val, coin_val, fmt=".4f"):
    return f"  {label:<28} {aapl_val:>14{fmt}} {coin_val:>14{fmt}}"

print(f"\n\n{'='*70}")
print(f"{'AAPL vs COIN — SIDE-BY-SIDE COMPARISON':^70}")
print(f"{'='*70}")
print(f"  {'Metric':<28} {'AAPL':>14} {'COIN':>14}")
print(f"  {'-'*56}")

# Block-level stats
print(stat_row("Total blocks",       aapl_n_total,  n_total_blocks,  ",d"))
print(stat_row("Dark blocks",        aapl_n_dark,   n_dark,          ",d"))
print(stat_row("Lit blocks",         aapl_n_lit,    n_lit,           ",d"))
print(stat_row("Dark %",             100*aapl_n_dark/aapl_n_total, 100*n_dark/n_total_blocks, ".1f"))

# Train split
print(f"\n  {'--- TRAIN ---':<28}")
print(stat_row("Trading days",       aapl_tr['date'].nunique(), train_df['date'].nunique(), "d"))
print(stat_row("Lit buy blocks",     len(aapl_tr),              len(train_df),              ",d"))

aapl_abs_tr = aapl_tr["impact_vwap_bps"].abs()
coin_abs_tr = train_df["impact_vwap_bps"].abs()
print(stat_row("abs(impact) mean",   aapl_abs_tr.mean(),        coin_abs_tr.mean()))
print(stat_row("abs(impact) median", aapl_abs_tr.median(),      coin_abs_tr.median()))
print(stat_row("abs(impact) std",    aapl_abs_tr.std(),         coin_abs_tr.std()))
print(stat_row("roll_spread mean",   aapl_tr["roll_spread_500"].mean(), train_df["roll_spread_500"].mean()))
print(stat_row("roll_vol mean",      aapl_tr["roll_vol_500"].mean(),    train_df["roll_vol_500"].mean()))
print(stat_row("prate mean",         aapl_tr["participation_rate"].mean(), train_df["participation_rate"].mean()))

# Test split
print(f"\n  {'--- TEST (Sep) ---':<28}")
print(stat_row("Trading days",       aapl_te['date'].nunique(), test_df['date'].nunique(), "d"))
print(stat_row("Lit buy blocks",     len(aapl_te),              len(test_df),              ",d"))

aapl_abs_te = aapl_te["impact_vwap_bps"].abs()
coin_abs_te = test_df["impact_vwap_bps"].abs()
print(stat_row("abs(impact) mean",   aapl_abs_te.mean(),        coin_abs_te.mean()))
print(stat_row("abs(impact) median", aapl_abs_te.median(),      coin_abs_te.median()))
print(stat_row("abs(impact) std",    aapl_abs_te.std(),         coin_abs_te.std()))
print(stat_row("roll_spread mean",   aapl_te["roll_spread_500"].mean(), test_df["roll_spread_500"].mean()))
print(stat_row("roll_vol mean",      aapl_te["roll_vol_500"].mean(),    test_df["roll_vol_500"].mean()))
print(stat_row("prate mean",         aapl_te["participation_rate"].mean(), test_df["participation_rate"].mean()))

print(f"\n{'='*70}")
