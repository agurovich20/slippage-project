"""
Classify block trades as buy/sell using the tick test, add 'side' column,
compute signed price impact, and show breakdown by side x dollar bucket.

Tick test (Lee-Ready fallback, derived from block_trades.parquet directly):
  - pre_price is the last trade price within 1s before the block trade, which
    for liquid stocks like AAPL is always the immediately preceding trade.
  - Compare block price vs pre_price:
      price > pre_price  -> uptick   -> buy
      price < pre_price  -> downtick -> sell
      price == pre_price -> zero-tick -> forward-fill from prior block's direction

signed_impact_bps = slippage_bps * side_sign  (+1 buy / -1 sell)
  Positive = price moved adversely for the initiator.
"""

import logging
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_DIR = Path("data")
BLK_PATH = DATA_DIR / "block_trades.parquet"


# ---------------------------------------------------------------------------
# Tick-test — runs entirely on pre_price already in block_trades.parquet
# ---------------------------------------------------------------------------

def classify_blocks(df: pd.DataFrame) -> np.ndarray:
    """
    Classify each block trade using price vs pre_price (the immediately
    preceding trade price, already stored in block_trades.parquet).

    Zero-ticks are resolved by forward-filling the last non-zero direction
    within each (ticker, date) group — the zero-plus/zero-minus rule applied
    across the block-trade subsequence. Roughly 95% of blocks have a non-zero
    tick so this approximation is excellent.
    """
    raw = np.sign(df["price"].to_numpy() - df["pre_price"].to_numpy()).astype(float)
    raw[df["pre_price"].isna().to_numpy()] = np.nan   # no preceding trade -> unknown

    dirs = pd.Series(raw, index=df.index).replace(0.0, np.nan)
    dirs = (
        df.groupby(["ticker", "date"], sort=False, group_keys=False)
          .apply(lambda g: dirs.loc[g.index].ffill(), include_groups=False)
    )
    return dirs.fillna(0).astype(np.int8).to_numpy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    blocks = pq.read_table(BLK_PATH).to_pandas()
    log.info("Loaded %d block trades", len(blocks))

    blocks.sort_values(["ticker", "date", "timestamp_ns"], inplace=True, ignore_index=True)
    side_values = classify_blocks(blocks)

    for (ticker, date_str), grp in blocks.groupby(["ticker", "date"]):
        dirs = side_values[grp.index]
        log.info("  %s %s  buys=%d  sells=%d  unknown=%d",
                 ticker, date_str,
                 (dirs ==  1).sum(), (dirs == -1).sum(), (dirs == 0).sum())

    blocks["side"] = side_values
    blocks["side_label"] = pd.Categorical(
        np.where(side_values ==  1, "buy",
        np.where(side_values == -1, "sell", "unknown")),
        categories=["buy", "sell", "unknown"],
    )
    blocks["signed_impact_bps"] = blocks["slippage_bps"] * blocks["side"].replace(0, np.nan)

    pq.write_table(pa.Table.from_pandas(blocks, preserve_index=False), BLK_PATH)
    log.info("Saved updated block_trades.parquet with side + signed_impact_bps")

    # ── Summary stats ─────────────────────────────────────────────────────
    classified = blocks[blocks["side_label"] != "unknown"].dropna(
        subset=["slippage_bps", "signed_impact_bps"]
    ).copy()

    bins   = [200_000, 500_000, 1_000_000, 5_000_000, np.inf]
    labels = ["$200k-500k", "$500k-1M", "$1M-5M", "$5M+"]
    classified["dv_bucket"] = pd.cut(
        classified["dollar_value"], bins=bins, labels=labels, right=False
    )

    def agg(g):
        si = g["signed_impact_bps"]
        return pd.Series({
            "n":               len(g),
            "med_slip_bps":    g["slippage_bps"].median(),
            "med_impact_bps":  si.median(),
            "mean_impact_bps": si.mean(),
            "p95_impact_bps":  si.quantile(0.95),
            "std_impact_bps":  si.std(),
        })

    def agg_dv(g):
        si = g["signed_impact_bps"]
        return pd.Series({
            "n":               len(g),
            "buy%":            100 * (g["side_label"] == "buy").mean(),
            "med_slip_bps":    g["slippage_bps"].median(),
            "med_impact_bps":  si.median(),
            "mean_impact_bps": si.mean(),
            "p95_impact_bps":  si.quantile(0.95),
            "std_impact_bps":  si.std(),
        })

    print(f"\n{'='*60}")
    print("SIDE DISTRIBUTION")
    print(f"{'='*60}")
    for label, n in blocks["side_label"].value_counts().items():
        print(f"  {label:<10}  {n:>7,}  ({100*n/len(blocks):.1f}%)")

    print(f"\n{'='*60}")
    print("SIGNED IMPACT BY SIDE  (bps)")
    print(f"{'='*60}")
    by_side = classified.groupby("side_label", observed=True).apply(agg, include_groups=False)
    print(by_side.to_string(float_format="{:.3f}".format))

    print(f"\n{'='*60}")
    print("SIGNED IMPACT BY DOLLAR BUCKET  (bps)")
    print(f"{'='*60}")
    by_dv = classified.groupby("dv_bucket", observed=True).apply(agg_dv, include_groups=False)
    print(by_dv.to_string(float_format="{:.3f}".format))

    print(f"\n{'='*60}")
    print("MEDIAN SIGNED IMPACT (bps) — bucket x side")
    print(f"{'='*60}")
    pivot = classified.pivot_table(
        values="signed_impact_bps", index="dv_bucket", columns="side_label",
        aggfunc="median", observed=True,
    )[["buy", "sell"]]
    print(pivot.to_string(float_format="{:.3f}".format))

    # ── Plot ──────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    DV_ORDER = [l for l in labels if l in classified["dv_bucket"].cat.categories]
    COLORS   = {"buy": "#3a86ff", "sell": "#e63946"}
    CLIP     = 15

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("AAPL Block-Trade Signed Price Impact  (tick-test classification)",
                 fontsize=12, fontweight="bold")

    # Left: grouped box plot by dollar bucket x side
    ax = axes[0]
    width, gap, group_gap = 0.32, 0.06, 0.35
    xticks, xlabels = [], []

    for i, bucket in enumerate(DV_ORDER):
        base = i * (2 * width + gap + group_gap)
        for j, side in enumerate(["buy", "sell"]):
            x    = base + j * (width + gap)
            data = classified.loc[
                (classified["dv_bucket"] == bucket) & (classified["side_label"] == side),
                "signed_impact_bps",
            ].clip(-CLIP, CLIP).dropna()

            bp = ax.boxplot(
                data, positions=[x], widths=width, patch_artist=True,
                medianprops=dict(color="black", linewidth=1.8),
                whiskerprops=dict(linewidth=1.0),
                capprops=dict(linewidth=1.0),
                flierprops=dict(marker=".", markersize=1.5, alpha=0.2, linestyle="none"),
            )
            bp["boxes"][0].set_facecolor(COLORS[side])
            bp["boxes"][0].set_alpha(0.75)

        xticks.append(base + (width + gap) / 2)
        xlabels.append(bucket)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_ylabel(f"Signed impact (bps, clipped +/-{CLIP})", fontsize=10)
    ax.set_title("By dollar bucket", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(handles=[Patch(facecolor=c, label=s, alpha=0.75) for s, c in COLORS.items()],
              fontsize=9)

    # Right: heatmap of median signed impact
    ax2 = axes[1]
    heat = pivot[["buy", "sell"]].reindex(DV_ORDER).astype(float)
    lim  = max(abs(heat.values[~np.isnan(heat.values)]).max(), 0.01)
    im   = ax2.imshow(heat.values, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["Buy", "Sell"], fontsize=10)
    ax2.set_yticks(range(len(DV_ORDER)))
    ax2.set_yticklabels(DV_ORDER, fontsize=9)
    ax2.set_title("Median signed impact (bps)\nbucket x side", fontsize=11)
    for ri, row in enumerate(heat.values):
        for ci, val in enumerate(row):
            if not np.isnan(val):
                ax2.text(ci, ri, f"{val:.3f}", ha="center", va="center",
                         fontsize=11, fontweight="bold",
                         color="white" if abs(val) > lim * 0.5 else "black")
    plt.colorbar(im, ax=ax2, shrink=0.7, label="bps")

    plt.tight_layout()
    plt.savefig("signed_impact.png", dpi=150, bbox_inches="tight")
    log.info("Plot saved -> signed_impact.png")


if __name__ == "__main__":
    main()
