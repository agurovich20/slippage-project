"""
Bin AAPL block trades by dollar_value into 20 equal-count quantile buckets.
For each bin, plot mean and median VWAP-bar signed impact (bps) for buys and
sells separately. X-axis is log dollar value.

Signed impact convention:
  buy  -> impact_vwap_bps as-is   (positive = price rose after block = adverse)
  sell -> -impact_vwap_bps         (positive = price fell after block = adverse)
This puts both sides on the same "adverse impact" axis for easy comparison,
and also plots them raw (unsigned) on a second panel for symmetry inspection.
"""

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
df = pq.read_table(
    "data/block_trades.parquet",
    columns=["dollar_value", "side_label", "impact_vwap_bps"],
).to_pandas()

df = df[(df["side_label"] != "unknown") & df["impact_vwap_bps"].notna()].copy()

# Signed impact: positive = price moved against initiator
df["signed_impact"] = np.where(
    df["side_label"] == "buy",
     df["impact_vwap_bps"],
    -df["impact_vwap_bps"],
)

N_BINS = 20

# ---------------------------------------------------------------------------
# Bin by dollar_value using equal-count quantiles (on the full population)
# ---------------------------------------------------------------------------
df["bin"] = pd.qcut(df["dollar_value"], q=N_BINS, labels=False, duplicates="drop")

def bin_stats(sub: pd.DataFrame) -> pd.DataFrame:
    """Aggregate mean, median, SEM, and count per bin per side."""
    rows = []
    for side in ("buy", "sell"):
        g = sub[sub["side_label"] == side]
        for bn, grp in g.groupby("bin", observed=True):
            si = grp["signed_impact"]
            rows.append({
                "side":        side,
                "bin":         bn,
                "n":           len(grp),
                "dv_mean":     grp["dollar_value"].mean(),
                "dv_geo":      np.exp(np.log(grp["dollar_value"]).mean()),  # geometric mean
                "mean":        si.mean(),
                "median":      si.median(),
                "sem":         si.sem(),
                "q25":         si.quantile(0.25),
                "q75":         si.quantile(0.75),
            })
    return pd.DataFrame(rows)

stats = bin_stats(df)
buy  = stats[stats["side"] == "buy" ].sort_values("bin").reset_index(drop=True)
sell = stats[stats["side"] == "sell"].sort_values("bin").reset_index(drop=True)

# ---------------------------------------------------------------------------
# Print table
# ---------------------------------------------------------------------------
print(f"\n{'='*78}")
print(f"{'BIN':>4}  {'DV (geom mean $k)':>18}  "
      f"{'BUY n':>7} {'buy mean':>9} {'buy med':>8}  "
      f"{'SELL n':>7} {'sell mean':>9} {'sell med':>8}")
print(f"{'='*78}")
for _, br, sr in zip(range(N_BINS), buy.itertuples(), sell.itertuples()):
    print(f"{int(br.bin):>4}  ${br.dv_geo/1000:>16,.0f}k  "
          f"{br.n:>7,} {br.mean:>9.3f} {br.median:>8.3f}  "
          f"{sr.n:>7,} {sr.mean:>9.3f} {sr.median:>8.3f}")

# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------
BUY_C  = "#2563eb"   # blue
SELL_C = "#dc2626"   # red

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
fig.suptitle(
    "AAPL block-trade price impact vs. trade size\n"
    "20 equal-count quantile bins  |  VWAP-bar method",
    fontsize=13, fontweight="bold", y=1.01,
)

# ── Panel A: signed adverse impact (both sides on same axis) ──────────────
ax = axes[0]

for side, sdf, color in [("Buy", buy, BUY_C), ("Sell", sell, SELL_C)]:
    x = sdf["dv_geo"].values

    # SEM band around mean
    ax.fill_between(x,
                    sdf["mean"] - 1.96 * sdf["sem"],
                    sdf["mean"] + 1.96 * sdf["sem"],
                    color=color, alpha=0.12)

    ax.plot(x, sdf["mean"],   color=color, lw=2.2, marker="o", ms=5,
            label=f"{side} — mean")
    ax.plot(x, sdf["median"], color=color, lw=1.4, marker="s", ms=4,
            ls="--", alpha=0.85, label=f"{side} — median")

ax.axhline(0, color="black", lw=0.9, ls="--", alpha=0.5)
ax.set_xscale("log")
ax.set_xlabel("Trade size (dollar value, log scale)", fontsize=11)
ax.set_ylabel("Signed adverse impact (bps)\n(positive = price moved against initiator)", fontsize=10)
ax.set_title("Adverse impact by side", fontsize=11)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(
    lambda x, _: f"${x/1e6:.1f}M" if x >= 1e6 else f"${x/1e3:.0f}k"
))
ax.legend(fontsize=9, ncol=2)
ax.grid(True, alpha=0.25)
ax.grid(True, which="minor", alpha=0.1)

# ── Panel B: raw (unsigned) impact — buy positive, sell negative ──────────
ax2 = axes[1]

buy_raw  = df[df["side_label"] == "buy" ].groupby("bin", observed=True)["impact_vwap_bps"]
sell_raw = df[df["side_label"] == "sell"].groupby("bin", observed=True)["impact_vwap_bps"]

# Merge onto bin x-positions from combined stats
dv_x = (
    df.groupby("bin", observed=True)["dollar_value"]
      .apply(lambda s: np.exp(np.log(s).mean()))
      .sort_index()
)

for raw_grp, color, label_stem in [
    (buy_raw,  BUY_C,  "Buy"),
    (sell_raw, SELL_C, "Sell"),
]:
    means   = raw_grp.mean().reindex(dv_x.index)
    medians = raw_grp.median().reindex(dv_x.index)
    sems    = raw_grp.sem().reindex(dv_x.index)
    x       = dv_x.values

    ax2.fill_between(x, means - 1.96*sems, means + 1.96*sems,
                     color=color, alpha=0.12)
    ax2.plot(x, means,   color=color, lw=2.2, marker="o", ms=5,
             label=f"{label_stem} — mean")
    ax2.plot(x, medians, color=color, lw=1.4, marker="s", ms=4,
             ls="--", alpha=0.85, label=f"{label_stem} — median")

ax2.axhline(0, color="black", lw=0.9, ls="--", alpha=0.5)
ax2.set_xscale("log")
ax2.set_xlabel("Trade size (dollar value, log scale)", fontsize=11)
ax2.set_ylabel("Raw impact_vwap_bps\n(buy +, sell −, zero = no price move)", fontsize=10)
ax2.set_title("Raw directional impact by side", fontsize=11)
ax2.xaxis.set_major_formatter(mticker.FuncFormatter(
    lambda x, _: f"${x/1e6:.1f}M" if x >= 1e6 else f"${x/1e3:.0f}k"
))
ax2.legend(fontsize=9, ncol=2)
ax2.grid(True, alpha=0.25)
ax2.grid(True, which="minor", alpha=0.1)

plt.tight_layout()
plt.savefig("impact_by_size.png", dpi=160, bbox_inches="tight")
print("\nChart saved -> impact_by_size.png")
