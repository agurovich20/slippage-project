"""
Binned OLS regression using abs(impact_vwap_bps) as target.

Panel A (3 subplots): 20 quantile bins by each of:
  roll_spread_500, roll_vol_500, participation_rate
  → univariate OLS on bin means

Panel B (1 subplot): 5×5 grid by roll_spread_500 × participation_rate
  → multivariate OLS on 25 cell means:
     abs_impact = c1*spread + c2*participation_rate + c3

Output: aapl_binned_regression_abs.png
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ── Load ───────────────────────────────────────────────────────────────────────
df = pd.read_parquet("data/lit_buy_features_v2.parquet")
df["abs_impact"] = df["impact_vwap_bps"].abs()
print(f"Loaded {len(df):,} rows")
print(f"abs_impact: mean={df['abs_impact'].mean():.4f}  "
      f"median={df['abs_impact'].median():.4f}  "
      f"std={df['abs_impact'].std():.4f}")

N_BINS  = 20
N_GRID  = 5
TARGET  = "abs_impact"

UNIVAR_FEATURES = [
    ("roll_spread_500",    "Rolling Roll spread (500 ticks, bps)",    "#2563eb"),
    ("roll_vol_500",       "Rolling realized vol (500 ticks, bps)",   "#16a34a"),
    ("participation_rate", "Participation rate (block / 1-min vol)",  "#dc2626"),
]


# ── OLS helpers ────────────────────────────────────────────────────────────────
def ols_1d(x, y):
    """y = c1*x + c2. Returns c1, c2, R²."""
    Xm = np.column_stack([x, np.ones(len(x))])
    beta, *_ = np.linalg.lstsq(Xm, y, rcond=None)
    y_hat = Xm @ beta
    ss_res = float(((y - y_hat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return beta[0], beta[1], r2


def ols_2d_multi(x1, x2, y):
    """y = c1*x1 + c2*x2 + c3. Returns c1, c2, c3, R²."""
    Xm = np.column_stack([x1, x2, np.ones(len(y))])
    beta, *_ = np.linalg.lstsq(Xm, y, rcond=None)
    y_hat = Xm @ beta
    ss_res = float(((y - y_hat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return beta[0], beta[1], beta[2], r2


def binned_stats_1d(data, feat_col, target_col, n_bins):
    sub = data[[feat_col, target_col]].dropna().copy()
    sub["bin"] = pd.qcut(sub[feat_col], q=n_bins, labels=False, duplicates="drop")
    agg = (
        sub.groupby("bin", observed=True)
        .agg(mean_x=(feat_col, "mean"),
             mean_y=(target_col, "mean"),
             count=(feat_col, "count"))
        .reset_index(drop=True)
    )
    return agg


# ── Figure layout: 2 rows ──────────────────────────────────────────────────────
# Row 0: 3 univariate scatter plots
# Row 1: left = 2-D heatmap of cell means; right = fitted-vs-actual scatter
fig = plt.figure(figsize=(18, 11))
gs  = gridspec.GridSpec(
    2, 4,
    figure=fig,
    height_ratios=[1, 1.1],
    hspace=0.42,
    wspace=0.38,
)

axes_uni = [fig.add_subplot(gs[0, k]) for k in range(3)]
ax_heat  = fig.add_subplot(gs[1, :2])   # heatmap spans cols 0-1
ax_mv    = fig.add_subplot(gs[1, 2:])   # multivar scatter spans cols 2-3


# ── Panel A: univariate binned OLS ────────────────────────────────────────────
print("\n=== Univariate binned OLS (target: abs_impact) ===")
for ax, (feat_col, feat_label, color) in zip(axes_uni, UNIVAR_FEATURES):
    bins = binned_stats_1d(df, feat_col, TARGET, N_BINS)
    x   = bins["mean_x"].to_numpy()
    y   = bins["mean_y"].to_numpy()
    cnt = bins["count"].to_numpy()

    c1, c2, r2 = ols_1d(x, y)
    print(f"  {feat_col:<22}  c1={c1:+.5f}  c2={c2:+.5f}  R²={r2:.4f}  "
          f"(n={len(x)} bins)")

    x_line = np.linspace(x.min(), x.max(), 300)
    y_line = c1 * x_line + c2

    sizes = 40 + 130 * (cnt / cnt.max())
    ax.scatter(x, y, s=sizes, color=color, alpha=0.85,
               edgecolors="white", linewidths=0.6, zorder=3)
    ax.plot(x_line, y_line, color="black", lw=1.8, zorder=4,
            label=f"$c_1$={c1:+.4f},  $c_2$={c2:+.4f}\n$R^2$={r2:.4f}")

    ax.set_xlabel(feat_label, fontsize=9.5)
    ax.set_ylabel("|impact_vwap_bps| (bin mean)", fontsize=9.5)
    ax.set_title(
        f"|impact| vs {feat_col}\n"
        f"$c_1$={c1:+.4f}   $c_2$={c2:+.4f}   $R^2$={r2:.4f}",
        fontsize=9.5, fontweight="bold",
    )
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── Panel B: 5×5 multivariate binned OLS ──────────────────────────────────────
print("\n=== 5×5 multivariate binned OLS (target: abs_impact) ===")

FEAT_S = "roll_spread_500"
FEAT_P = "participation_rate"

sub = df[[FEAT_S, FEAT_P, TARGET]].dropna().copy()
sub["bin_s"] = pd.qcut(sub[FEAT_S], q=N_GRID, labels=False, duplicates="drop")
sub["bin_p"] = pd.qcut(sub[FEAT_P], q=N_GRID, labels=False, duplicates="drop")

grid = (
    sub.groupby(["bin_s", "bin_p"], observed=True)
    .agg(
        mean_spread=(FEAT_S, "mean"),
        mean_prate =(FEAT_P, "mean"),
        mean_abs   =(TARGET,  "mean"),
        count      =(TARGET,  "count"),
    )
    .reset_index()
)

print(f"  Grid cells populated: {len(grid)} / {N_GRID*N_GRID}")
print(f"  Cell count range: {grid['count'].min()} – {grid['count'].max()}")

x1 = grid["mean_spread"].to_numpy()
x2 = grid["mean_prate"].to_numpy()
y  = grid["mean_abs"].to_numpy()

c1, c2, c3, r2_mv = ols_2d_multi(x1, x2, y)
y_hat_mv = c1 * x1 + c2 * x2 + c3

print(f"\n  OLS: abs_impact = c1*spread + c2*prate + c3")
print(f"    c1 (spread)        = {c1:+.6f}")
print(f"    c2 (prate)         = {c2:+.6f}")
print(f"    c3 (intercept)     = {c3:+.6f}")
print(f"    R² (25 cell means) = {r2_mv:.4f}")

# ── Heatmap: mean abs impact on the 5×5 grid ──────────────────────────────────
# Pivot to 2-D array (rows = prate bin, cols = spread bin) for imshow
pivot = grid.pivot(index="bin_p", columns="bin_s", values="mean_abs")
pivot_count = grid.pivot(index="bin_p", columns="bin_s", values="count")

# Axis tick labels: midpoints of each bin
spread_mids = grid.groupby("bin_s")["mean_spread"].mean().sort_index()
prate_mids  = grid.groupby("bin_p")["mean_prate"].mean().sort_index()

im = ax_heat.imshow(
    pivot.to_numpy(),
    origin="lower", aspect="auto",
    cmap="YlOrRd",
    norm=Normalize(vmin=y.min(), vmax=y.max()),
)
plt.colorbar(im, ax=ax_heat, label="|impact_vwap_bps| (bin mean)", shrink=0.85)

# Annotate each cell with mean value and count
for bi_p in range(N_GRID):
    for bi_s in range(N_GRID):
        if bi_p in pivot.index and bi_s in pivot.columns:
            val = pivot.loc[bi_p, bi_s]
            cnt = pivot_count.loc[bi_p, bi_s]
            if not np.isnan(val):
                ax_heat.text(bi_s, bi_p, f"{val:.2f}\n(n={cnt:,})",
                             ha="center", va="center",
                             fontsize=7.5, color="black",
                             fontweight="bold" if val > y.mean() else "normal")

ax_heat.set_xticks(range(len(spread_mids)))
ax_heat.set_xticklabels([f"{v:.2f}" for v in spread_mids.values], fontsize=8)
ax_heat.set_yticks(range(len(prate_mids)))
ax_heat.set_yticklabels([f"{v:.4f}" for v in prate_mids.values], fontsize=8)
ax_heat.set_xlabel("Mean roll_spread_500 (bps, quintile bin)", fontsize=9.5)
ax_heat.set_ylabel("Mean participation_rate (quintile bin)", fontsize=9.5)
ax_heat.set_title(
    f"5×5 cell means: |impact| by spread × prate quintile\n"
    f"(raw cell values; OLS fit shown in right panel)",
    fontsize=9.5, fontweight="bold",
)

# ── Fitted vs actual scatter for the multivariate OLS ─────────────────────────
sizes_mv = 40 + 140 * (grid["count"].to_numpy() / grid["count"].max())
ax_mv.scatter(y_hat_mv, y, s=sizes_mv, color="#7c3aed", alpha=0.85,
              edgecolors="white", linewidths=0.6, zorder=3)

diag = np.array([min(y_hat_mv.min(), y.min()), max(y_hat_mv.max(), y.max())])
ax_mv.plot(diag, diag, "k--", lw=1.5, alpha=0.6, label="Perfect fit")

ax_mv.set_xlabel("OLS fitted |impact| (bps)", fontsize=9.5)
ax_mv.set_ylabel("Cell mean |impact| (bps)", fontsize=9.5)
ax_mv.set_title(
    f"Multivariate OLS: fitted vs actual (25 cells)\n"
    f"abs_impact = {c1:+.4f}·spread {c2:+.4f}·prate {c3:+.4f}\n"
    f"$R^2$ = {r2_mv:.4f}",
    fontsize=9.5, fontweight="bold",
)
ax_mv.legend(fontsize=8.5)
ax_mv.grid(True, alpha=0.2)
ax_mv.spines["top"].set_visible(False)
ax_mv.spines["right"].set_visible(False)

# ── Super-title ────────────────────────────────────────────────────────────────
fig.suptitle(
    "AAPL lit buy block trades — binned OLS on |impact_vwap_bps|\n"
    "trade-level 500-tick rolling features",
    fontsize=12, fontweight="bold", y=1.005,
)

plt.savefig("aapl_binned_regression_abs.png", dpi=150, bbox_inches="tight")
print("\nsaved -> aapl_binned_regression_abs.png")
