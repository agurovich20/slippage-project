import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import pyarrow.parquet as pq
import pandas as pd
from collections import Counter

path = "data/AAPL/2024-06-03.parquet"

# ── Schema ────────────────────────────────────────────────────────────────────
schema = pq.read_schema(path)
print("=== Parquet schema ===")
for f in schema:
    print(f"  {f.name:30s} {f.type}")

# ── Exchange + tape (fast columns) ───────────────────────────────────────────
tbl = pq.read_table(path, columns=["exchange", "tape"])
df  = tbl.to_pandas()

# Polygon exchange ID → name mapping (from Polygon docs)
EXCHANGE = {
    1:  "NYSE",
    2:  "AMEX",
    3:  "NYSE Arca",
    4:  "BATS",
    5:  "BSX (Boston)",
    6:  "CBSX",
    7:  "CHX",
    8:  "FINRA ADF",
    9:  "ISE",
    10: "CBOE EDGA",
    11: "CBOE EDGX",
    12: "LTSE",
    13: "MIAX",
    14: "NASDAQ",
    15: "NYSE MKT",
    16: "NSX",
    17: "IEX",
    18: "NASDAQ PHLX",
    19: "CBOE BZX",
    20: "NYSE American",
    21: "NASDAQ BX",
    22: "MEMX",
    23: "CBOE",
    24: "Long-Term SE",
    62: "OTC (dark/TRF)",
    63: "FINRA TRF",
}

print(f"\n=== Exchange distribution (2024-06-03, {len(df):,} trades) ===")
ex_counts = df["exchange"].value_counts().sort_values(ascending=False)
for ex_id, cnt in ex_counts.items():
    name = EXCHANGE.get(int(ex_id), f"unknown-{ex_id}")
    pct  = 100 * cnt / len(df)
    print(f"  {int(ex_id):>3}  {name:<22}  {cnt:>8,}  ({pct:5.1f}%)")

print(f"\n=== Tape distribution ===")
print(df["tape"].value_counts().to_string())

# ── Conditions: read in batches to avoid slow list deserialisation ────────────
print("\n=== Condition codes (sampled from first 20k rows) ===")
pf = pq.ParquetFile(path)
all_conds = []
rows_read = 0
for batch in pf.iter_batches(batch_size=5000, columns=["conditions"]):
    chunk = batch.to_pandas()
    for v in chunk["conditions"]:
        if v is not None and len(v) > 0:
            all_conds.extend(v)
    rows_read += len(chunk)
    if rows_read >= 20000:
        break

cond_counts = Counter(all_conds)
# Polygon condition code meanings (common ones)
COND = {
    14: "ODD_LOT",
    15: "DERIVATIVELY_PRICED",
    37: "NASDAQ_XT / Ext-hours",
    41: "FINRA_ADF / OTC",
    12: "REGULAR",
    10: "SOLD_LAST",
    1:  "REGULAR_SALE",
    2:  "ACQUISITION",
    3:  "AMEX_RULE_155_TRADE",
    4:  "AMEX_STOPPED_STOCK",
    7:  "BUNCHED_TRADE",
    16: "PRIOR_REFERENCE_PRICE",
    17: "RULE_127",
    19: "SELLER",
    20: "SOLD_OUT_OF_SEQUENCE",
    21: "STOPPED_STOCK",
    24: "FORM_T",
    29: "EXTENDED_HOURS_TRADE",
    33: "SOLD",
    36: "INTERMARKET_SWEEP",
    40: "MARKET_CENTER_OFFICIAL_CLOSE",
    41: "MARKET_CENTER_OFFICIAL_OPEN",
    52: "CONTINGENT_TRADE",
    53: "QUALIFIED_CONTINGENT_TRADE",
}
for code, cnt in sorted(cond_counts.items(), key=lambda x: -x[1]):
    meaning = COND.get(code, "?")
    print(f"  {code:>4}: {cnt:>7,}   {meaning}")

# ── block_trades exchange breakdown ──────────────────────────────────────────
print("\n=== block_trades.parquet columns ===")
bt_schema = pq.read_schema("data/block_trades.parquet")
for f in bt_schema:
    print(f"  {f.name:30s} {f.type}")
