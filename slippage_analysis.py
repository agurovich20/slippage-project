"""
Slippage breakdown by dollar-value bucket and time of day.
Saves: slippage_breakdown.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Load & prep
# ---------------------------------------------------------------------------
df = pq.read_table("data/block_trades.parquet").to_pandas()
df = df.dropna(subset=["slippage_bps"])

# Convert nanosecond UTC timestamp -> Eastern time
df["dt_et"] = (
    pd.to_datetime(df["timestamp_ns"], unit="ns", utc=True)
      .dt.tz_convert("America/New_York")
)
df["time_et"] = df["dt_et"].dt.time

import datetime as dt
OPEN      = dt.time(9, 30)
OPEN_30   = dt.time(10, 0)
CLOSE_30  = dt.time(15, 30)
CLOSE     = dt.time(16, 0)

def classify_tod(t):
    if OPEN <= t < OPEN_30:
        return "Open\n(9:30–10:00)"
    elif OPEN_30 <= t < CLOSE_30:
        return "Midday\n(10:00–15:30)"
    elif CLOSE_30 <= t <= CLOSE:
        return "Close\n(15:30–16:00)"
    return "Other"

df["tod"] = df["time_et"].apply(classify_tod)
df = df[df["tod"] != "Other"]          # drop pre/post-market outliers

# Dollar-value buckets
bins   = [200_000, 500_000, 1_000_000, 5_000_000, np.inf]
labels = ["$200k–500k", "$500k–1M", "$1M–5M", "$5M+"]
df["dv_bucket"] = pd.cut(df["dollar_value"], bins=bins, labels=labels, right=False)

# Clip slippage for display (keeps box plots readable; full range in table)
CLIP = 20
df["slip_clip"] = df["slippage_bps"].clip(-CLIP, CLIP)

TOD_ORDER = ["Open\n(9:30–10:00)", "Midday\n(10:00–15:30)", "Close\n(15:30–16:00)"]
DV_ORDER  = labels

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
def summarise(group):
    s = group["slippage_bps"]
    return pd.Series({
        "n":          len(s),
        "median":     s.median(),
        "mean":       s.mean(),
        "IQR":        s.quantile(0.75) - s.quantile(0.25),
        "p95":        s.quantile(0.95),
        "std":        s.std(),
    })

print("=" * 72)
print("SLIPPAGE BY DOLLAR-VALUE BUCKET  (bps)")
print("=" * 72)
tbl_dv = df.groupby("dv_bucket", observed=True).apply(summarise, include_groups=False)
print(tbl_dv.to_string(float_format="{:.2f}".format))

print()
print("=" * 72)
print("SLIPPAGE BY TIME OF DAY  (bps)")
print("=" * 72)
tbl_tod = (df.assign(tod_plain=df["tod"].str.replace("\n", " "))
             .groupby("tod_plain").apply(summarise, include_groups=False)
             .reindex([t.replace("\n", " ") for t in TOD_ORDER]))
print(tbl_tod.to_string(float_format="{:.2f}".format))

print()
print("=" * 72)
print("SLIPPAGE MEDIAN (bps) — dollar bucket × time of day")
print("=" * 72)
pivot_med = df.pivot_table(
    values="slippage_bps", index="dv_bucket", columns="tod",
    aggfunc="median", observed=True
)[TOD_ORDER]
pivot_med.columns = [c.replace("\n", " ") for c in pivot_med.columns]
print(pivot_med.to_string(float_format="{:.3f}".format))

print()
print("=" * 72)
print("COUNT — dollar bucket × time of day")
print("=" * 72)
pivot_n = df.pivot_table(
    values="slippage_bps", index="dv_bucket", columns="tod",
    aggfunc="count", observed=True
)[TOD_ORDER]
pivot_n.columns = [c.replace("\n", " ") for c in pivot_n.columns]
print(pivot_n.to_string())

# ---------------------------------------------------------------------------
# Plot — 2 rows: by DV bucket (top), by time of day (bottom)
# ---------------------------------------------------------------------------
COLORS_DV  = ["#4878d0", "#ee854a", "#6acc65", "#d65f5f"]
COLORS_TOD = ["#956cb4", "#8c613c", "#dc7ec0"]

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
fig.suptitle("AAPL Block-Trade Slippage Breakdown", fontsize=13, fontweight="bold", y=1.01)

def boxplot_groups(ax, df, group_col, order, colors, title, xlabel):
    data   = [df.loc[df[group_col] == g, "slip_clip"].dropna().values for g in order]
    labels = [g.replace("\n", "\n") for g in order]

    bp = ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker=".", markersize=2, alpha=0.3, linestyle="none"),
        widths=0.55,
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Slippage (bps, clipped ±20)", fontsize=10)
    ax.set_ylim(-CLIP * 1.15, CLIP * 1.15)
    ax.grid(axis="y", alpha=0.3)

    # Annotate median + n
    for i, (grp, d) in enumerate(zip(order, data), 1):
        if len(d):
            med = np.median(d)
            ax.text(i, CLIP * 1.05, f"n={len(d):,}", ha="center", va="bottom",
                    fontsize=7.5, color="dimgray")
            ax.text(i, med + 0.7, f"{np.median(df.loc[df[group_col]==grp,'slippage_bps'].dropna()):.2f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")

boxplot_groups(
    axes[0], df, "dv_bucket", DV_ORDER, COLORS_DV,
    "By Trade Size", "Dollar-value bucket"
)
boxplot_groups(
    axes[1], df, "tod", TOD_ORDER, COLORS_TOD,
    "By Time of Day", "Session period"
)

plt.tight_layout()
plt.savefig("slippage_breakdown.png", dpi=150, bbox_inches="tight")
print("\nPlot saved -> slippage_breakdown.png")
