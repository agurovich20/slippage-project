"""
Binned OLS regression: group lit buy trades into 20 quantile bins by each feature,
compute per-bin means, fit OLS on the 20 points, and plot.

Features binned:
  1. roll_spread_500
  2. roll_vol_500
  3. participation_rate

Target: impact_vwap_bps (signed)

Output: aapl_binned_regression.png
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Load ───────────────────────────────────────────────────────────────────────
df = pd.read_parquet("data/lit_buy_features_v2.parquet")
print(f"Loaded {len(df):,} rows")

N_BINS   = 20
TARGET   = "impact_vwap_bps"
FEATURES = [
    ("roll_spread_500",  "Rolling Roll spread (500 ticks, bps)",    "#2563eb"),
    ("roll_vol_500",     "Rolling realized vol (500 ticks, bps)",   "#16a34a"),
    ("participation_rate", "Participation rate (block / 1-min vol)", "#dc2626"),
]


def ols_2d(x, y):
    """Fit y = c1*x + c2 via closed-form OLS. Returns c1, c2, R²."""
    n  = len(x)
    Xm = np.column_stack([x, np.ones(n)])
    beta, *_ = np.linalg.lstsq(Xm, y, rcond=None)
    c1, c2   = beta
    y_hat    = Xm @ beta
    ss_res   = float(((y - y_hat) ** 2).sum())
    ss_tot   = float(((y - y.mean()) ** 2).sum())
    r2       = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return c1, c2, r2


def binned_stats(df, feat_col, target_col, n_bins):
    """
    Assign each row to one of n_bins quantile bins on feat_col.
    Return DataFrame with columns: bin_mean_x, bin_mean_y, bin_count.
    """
    df = df[[feat_col, target_col]].dropna().copy()
    df["bin"] = pd.qcut(df[feat_col], q=n_bins, labels=False, duplicates="drop")
    agg = (
        df.groupby("bin", observed=True)
        .agg(
            bin_mean_x=(feat_col,  "mean"),
            bin_mean_y=(target_col, "mean"),
            bin_count =(feat_col,  "count"),
        )
        .reset_index(drop=True)
    )
    return agg


# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

for ax, (feat_col, feat_label, color) in zip(axes, FEATURES):
    bins = binned_stats(df, feat_col, TARGET, N_BINS)
    x    = bins["bin_mean_x"].to_numpy()
    y    = bins["bin_mean_y"].to_numpy()
    cnt  = bins["bin_count"].to_numpy()

    c1, c2, r2 = ols_2d(x, y)
    print(f"\n{feat_col}")
    print(f"  c1={c1:+.6f}  c2={c2:+.6f}  R²={r2:.4f}  (n={len(x)} bins)")

    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = c1 * x_line + c2

    # Scatter: point size proportional to bin count
    sizes = 40 + 120 * (cnt / cnt.max())
    sc = ax.scatter(x, y, s=sizes, color=color, alpha=0.85,
                    edgecolors="white", linewidths=0.6, zorder=3)

    ax.plot(x_line, y_line, color="black", lw=1.8, zorder=4,
            label=f"OLS: impact = {c1:+.4f}·x {c2:+.4f}\n$R^2$ = {r2:.4f}")

    ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.6)

    ax.set_xlabel(feat_label, fontsize=10)
    ax.set_ylabel("Mean impact_vwap_bps (per bin)", fontsize=10)
    ax.set_title(
        f"20-bin OLS: impact vs {feat_col}\n"
        f"$c_1$={c1:+.4f}   $c_2$={c2:+.4f}   $R^2$={r2:.4f}",
        fontsize=10, fontweight="bold",
    )
    ax.legend(fontsize=8.5, loc="best")
    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.suptitle(
    "AAPL lit buy block trades — binned OLS regression (20 quantile bins)\n"
    "trade-level rolling features (500-tick window)",
    fontsize=12, fontweight="bold", y=1.01,
)
plt.tight_layout()
plt.savefig("aapl_binned_regression.png", dpi=150, bbox_inches="tight")
print("\nsaved -> aapl_binned_regression.png")
