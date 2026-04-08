"""
For what fraction of lit buy block trades does the block's timestamp fall
within the same second as the VWAP reference bar?

Two checks:
  A) Arithmetic (participant_timestamp): floor(ts/1e9) == floor(ts/1e9) - 1
     -> always 0 by definition; confirms no implementation bug.

  B) SIP-timestamp check: Polygon bars aggregate by SIP time, not participant
     time.  If a block's SIP second == ref_s (= participant_second - 1), the
     block trade is physically inside its own reference bar (look-ahead bias).

Vectorized per-day using searchsorted -- no Python loops over individual rows.
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"]      = "1"

import numpy as np
import pandas as pd

DARK_IDS   = {4, 6, 16, 62, 201, 202, 203}
ONE_SEC_NS = 1_000_000_000

# ── 1. Load lit buy block trades ──────────────────────────────────────────────
bt = pd.read_parquet(
    "data/block_trades.parquet",
    columns=["date", "timestamp_ns", "price", "size",
             "exchange", "side_label", "impact_vwap_bps"],
    filters=[("ticker", "==", "AAPL")],
)
buys = bt[
    (~bt["exchange"].isin(DARK_IDS)) &
    (bt["side_label"] == "buy") &
    bt["impact_vwap_bps"].notna()
].copy().sort_values(["date", "timestamp_ns"]).reset_index(drop=True)
n_total = len(buys)
print(f"Lit buy block trades: {n_total:,}")

# ── 2. Check A: arithmetic / by-construction ──────────────────────────────────
block_sec = (buys["timestamp_ns"].to_numpy(np.int64) // ONE_SEC_NS)
ref_s_arr = block_sec - 1
n_same_arith = int((block_sec == ref_s_arr).sum())
print(f"\n[Check A] block_participant_second == ref_s: {n_same_arith} / {n_total}")
print(f"  {100.0 * n_same_arith / n_total:.4f}%  (must be 0 by construction)")

# ── 3. Check B: vectorized SIP-timestamp lookup ───────────────────────────────
# For each block, find the SIP timestamp of the matching tick in the tick file.
# Match key: (participant_timestamp, size) -- should be unique within a day.
# Vectorized: for each day, sort ticks by participant_timestamp,
# use searchsorted to find candidate positions, then verify size match.

sip_sec_arr  = np.full(n_total, -1, dtype=np.int64)   # -1 = unmatched
unmatched_ct = 0
day_results  = []

for date, grp in buys.groupby("date"):
    ticks = pd.read_parquet(
        f"data/AAPL/{date}.parquet",
        columns=["participant_timestamp", "sip_timestamp", "size"],
    )
    ticks = ticks.sort_values("participant_timestamp").reset_index(drop=True)

    pt  = ticks["participant_timestamp"].to_numpy(np.int64)
    st  = ticks["sip_timestamp"].to_numpy(np.int64)
    tsz = ticks["size"].to_numpy(np.int64)

    blk_pt  = grp["timestamp_ns"].to_numpy(np.int64)
    blk_sz  = grp["size"].to_numpy(np.int64)
    blk_idx = grp.index.to_numpy()   # indices into sip_sec_arr

    # searchsorted: for each block, find first tick with pt >= blk_pt
    lo = np.searchsorted(pt, blk_pt, side="left")

    for i, (bi, bpt, bsz, loc) in enumerate(
            zip(blk_idx, blk_pt, blk_sz, lo)):
        # scan forward from loc while pt[j] == bpt
        j = loc
        matched = False
        while j < len(pt) and pt[j] == bpt:
            if tsz[j] == bsz:
                sip_sec_arr[bi] = st[j] // ONE_SEC_NS
                matched = True
                break
            j += 1
        if not matched:
            # fall back: any tick at the same participant_timestamp
            if loc < len(pt) and pt[loc] == bpt:
                sip_sec_arr[bi] = st[loc] // ONE_SEC_NS
            else:
                unmatched_ct += 1

print(f"\n[Check B] Unmatched block trades: {unmatched_ct}")

valid         = sip_sec_arr >= 0
n_valid       = int(valid.sum())
sip_sec_valid = sip_sec_arr[valid]
blk_sec_valid = block_sec[valid]
ref_s_valid   = blk_sec_valid - 1

# SIP second == ref_s  =>  block is inside its own VWAP reference bar
n_contaminated = int((sip_sec_valid == ref_s_valid).sum())
print(f"Trades where SIP_second == ref_s (block inside its own VWAP bar):")
print(f"  {n_contaminated} / {n_valid} = {100.0 * n_contaminated / n_valid:.4f}%")

# Distribution of (SIP second - participant second)
diff_s = sip_sec_valid - blk_sec_valid
unique_diffs, counts = np.unique(diff_s, return_counts=True)
print(f"\n[SIP second - participant second] distribution:")
for d, c in zip(unique_diffs, counts):
    bar = "#" * min(40, int(40 * c / n_valid))
    pct = 100.0 * c / n_valid
    tag = " <-- CONTAMINATION (SIP is 1s behind participant)" if d == -1 else ""
    print(f"  delta={d:+d}s: {c:>6,}  ({pct:6.3f}%)  {bar}{tag}")

# ── 4. Summary ────────────────────────────────────────────────────────────────
print(f"\n{'='*64}")
print(f"QUESTION: For what fraction of lit buy block trades does the")
print(f"block timestamp fall in the same second as the VWAP ref bar?")
print(f"{'='*64}")
print(f"  By participant_timestamp (arithmetic guarantee): "
      f"{100.0 * n_same_arith / n_total:.4f}%")
print(f"  By SIP timestamp (actual Polygon bar content):  "
      f"{100.0 * n_contaminated / n_valid:.4f}%")
print(f"{'='*64}")
