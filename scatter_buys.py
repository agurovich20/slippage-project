import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

df = pd.read_parquet("data/block_trades.parquet",
    columns=["side_label", "dollar_value", "impact_vwap_bps"])
buys = df[(df.side_label == "buy") & df.impact_vwap_bps.notna()].copy()
print(f"{len(buys):,} AAPL buy block trades")

CLIP = 15
buys["impact_c"] = buys["impact_vwap_bps"].clip(-CLIP, CLIP)

# 10k random sample for scatter points
sample = buys.sample(n=min(10_000, len(buys)), random_state=42)

# Binned mean on full data (30 equal-count bins)
buys["bin"] = pd.qcut(buys["dollar_value"], q=30, duplicates="drop")
binned = buys.groupby("bin", observed=True).agg(
    dv_mid=("dollar_value", "median"),
    mean_impact=("impact_vwap_bps", "mean"),
    n=("impact_vwap_bps", "count"),
).reset_index()

fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(sample["dollar_value"], sample["impact_c"],
           alpha=0.10, s=7, color="#2563eb", linewidths=0,
           label=f"Trades (10k sample of {len(buys):,})")

ax.plot(binned["dv_mid"], binned["mean_impact"].clip(-CLIP, CLIP),
        color="#f59e0b", lw=2.2, marker="o", ms=5, zorder=5,
        label="Binned mean (all trades)")

ax.axhline(0, color="black", lw=1, ls="--", alpha=0.55)
ax.set_xscale("log")
ax.set_xlabel("Trade size (dollar value)", fontsize=12)
ax.set_ylabel("Slippage — VWAP-bar impact (bps, clipped ±15)", fontsize=12)
ax.set_title("AAPL block-trade slippage: buys", fontsize=13, fontweight="bold")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(
    lambda v, _: f"${v/1e6:.1f}M" if v >= 1e6 else f"${v/1e3:.0f}k"))
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig("aapl_buy_slippage_scatter.png", dpi=150, bbox_inches="tight")
print("saved -> aapl_buy_slippage_scatter.png")
