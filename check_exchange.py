import pyarrow.parquet as pq
import pandas as pd

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
