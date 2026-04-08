import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# TRF / ORF exchange IDs — confirmed via client.get_exchanges()
DARK_IDS = {4, 6, 16, 62, 201, 202, 203}

df = pd.read_parquet("data/block_trades.parquet",
    columns=["side_label", "exchange", "dollar_value", "impact_vwap_bps"])

lit  = df[~df["exchange"].isin(DARK_IDS)]
buys = lit[(lit["side_label"] == "buy") & lit["impact_vwap_bps"].notna()].copy()

n = len(buys)
med = buys["impact_vwap_bps"].median()
mn  = buys["impact_vwap_bps"].mean()
std = buys["impact_vwap_bps"].std()

print(f"Lit buy block trades : {n:,}")
print(f"  Median impact (bps): {med:+.4f}")
print(f"  Mean   impact (bps): {mn:+.4f}")
print(f"  Std    impact (bps): {std:.4f}")

CLIP = 15
buys["impact_c"] = buys["impact_vwap_bps"].clip(-CLIP, CLIP)

sample = buys.sample(n=min(10_000, n), random_state=42)

buys["bin"] = pd.qcut(buys["dollar_value"], q=30, duplicates="drop")
binned = buys.groupby("bin", observed=True).agg(
    dv_mid=("dollar_value", "median"),
    mean_impact=("impact_vwap_bps", "mean"),
).reset_index()
binned["mean_impact"] = binned["mean_impact"].clip(-CLIP, CLIP)

fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(sample["dollar_value"], sample["impact_c"],
           alpha=0.10, s=7, color="#2563eb", linewidths=0,
           label=f"Trades (10k sample of {n:,})")

ax.plot(binned["dv_mid"], binned["mean_impact"],
        color="#f59e0b", lw=2.2, marker="o", ms=5, zorder=5,
        label="Binned mean (all trades)")

ax.axhline(0, color="black", lw=1, ls="--", alpha=0.55)
ax.set_xscale("log")
ax.set_xlabel("Trade size (dollar value)", fontsize=12)
ax.set_ylabel("VWAP-bar impact (bps, clipped +-15)", fontsize=12)
ax.set_title("AAPL lit-only buy block trades — impact vs size", fontsize=13, fontweight="bold")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(
    lambda v, _: f"${v/1e6:.1f}M" if v >= 1e6 else f"${v/1e3:.0f}k"))
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig("aapl_lit_buy_impact.png", dpi=150, bbox_inches="tight")
print("saved -> aapl_lit_buy_impact.png")
