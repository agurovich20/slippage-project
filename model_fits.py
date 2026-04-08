"""
Three model fits on 20-bin means (bins by roll_spread_500).

Per-bin mean features: mean_spread, mean_prate, mean_abs_impact (n=20 points).

Models:
  1. Linear    abs_impact = c1*spread + c2
  2. Quadratic abs_impact = c1*spread^2 + c2*spread + c3
  3. Vlad bond abs_impact = c1*spread + c2*prate^0.4 + c3

All three are linear in parameters → closed-form OLS.
R² computed on the 20 bin means.

Output: aapl_model_fits.png
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Load and add abs target ────────────────────────────────────────────────────
df = pd.read_parquet("data/lit_buy_features_v2.parquet")
df["abs_impact"] = df["impact_vwap_bps"].abs()
print(f"Loaded {len(df):,} rows")

N_BINS = 20

# ── Build 20-bin means by roll_spread_500 ─────────────────────────────────────
sub = df[["roll_spread_500", "participation_rate", "abs_impact"]].dropna().copy()
sub["bin"] = pd.qcut(sub["roll_spread_500"], q=N_BINS, labels=False, duplicates="drop")

bins = (
    sub.groupby("bin", observed=True)
    .agg(
        mean_spread=("roll_spread_500",  "mean"),
        mean_prate =("participation_rate","mean"),
        mean_abs   =("abs_impact",        "mean"),
        count      =("abs_impact",        "count"),
    )
    .reset_index(drop=True)
)

s   = bins["mean_spread"].to_numpy()   # shape (20,)
p   = bins["mean_prate"].to_numpy()    # shape (20,)
y   = bins["mean_abs"].to_numpy()      # shape (20,)
cnt = bins["count"].to_numpy()

print(f"\nBin means (n={len(bins)}):")
print(f"  spread range : {s.min():.4f} – {s.max():.4f} bps")
print(f"  prate range  : {p.min():.5f} – {p.max():.5f}")
print(f"  abs_imp range: {y.min():.4f} – {y.max():.4f} bps")


# ── OLS helper ────────────────────────────────────────────────────────────────
def ols_fit(X_design, y):
    """Fit OLS: y = X_design @ beta.  Returns beta, y_hat, R²."""
    beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    y_hat = X_design @ beta
    ss_res = float(((y - y_hat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return beta, y_hat, r2


# ── Model 1: Linear ───────────────────────────────────────────────────────────
X1 = np.column_stack([s, np.ones(len(s))])
b1, yhat1, r2_1 = ols_fit(X1, y)
print(f"\nModel 1 — Linear")
print(f"  abs_impact = {b1[0]:+.6f}*spread {b1[1]:+.6f}")
print(f"  R² = {r2_1:.4f}")

# ── Model 2: Quadratic ────────────────────────────────────────────────────────
X2 = np.column_stack([s**2, s, np.ones(len(s))])
b2, yhat2, r2_2 = ols_fit(X2, y)
print(f"\nModel 2 — Quadratic")
print(f"  abs_impact = {b2[0]:+.6f}*spread² {b2[1]:+.6f}*spread {b2[2]:+.6f}")
print(f"  R² = {r2_2:.4f}")

# ── Model 3: Vlad bond form ───────────────────────────────────────────────────
X3 = np.column_stack([s, p**0.4, np.ones(len(s))])
b3, yhat3, r2_3 = ols_fit(X3, y)
print(f"\nModel 3 — Vlad bond (spread + prate^0.4)")
print(f"  abs_impact = {b3[0]:+.6f}*spread {b3[1]:+.6f}*prate^0.4 {b3[2]:+.6f}")
print(f"  R² = {r2_3:.4f}")

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n" + "="*52)
print(f"{'Model':<22}  {'Parameters':>5}  {'R²':>8}")
print("="*52)
print(f"  {'1. Linear':<20}  {2:>5}  {r2_1:>8.4f}")
print(f"  {'2. Quadratic':<20}  {3:>5}  {r2_2:>8.4f}")
print(f"  {'3. Vlad bond':<20}  {3:>5}  {r2_3:>8.4f}")
print("="*52)


# ── Plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6.5))

# Scatter of 20 bin means — size proportional to count
sizes = 55 + 160 * (cnt / cnt.max())
ax.scatter(s, y, s=sizes, color="#1e293b", zorder=5,
           edgecolors="white", linewidths=0.8,
           label=f"20-bin means (area ∝ trade count)")

# Smooth x-grid for curves 1 & 2 (purely functions of spread)
s_grid = np.linspace(s.min(), s.max(), 400)

# Model 1 curve
y_grid1 = b1[0] * s_grid + b1[1]
ax.plot(s_grid, y_grid1, color="#2563eb", lw=2.2, zorder=4,
        label=f"[1] Linear:    $c_1$={b1[0]:+.4f}*s  $c_2$={b1[1]:+.4f}"
              f"   $R^2$={r2_1:.4f}")

# Model 2 curve
y_grid2 = b2[0] * s_grid**2 + b2[1] * s_grid + b2[2]
ax.plot(s_grid, y_grid2, color="#16a34a", lw=2.2, ls="--", zorder=4,
        label=f"[2] Quadratic: $c_1$={b2[0]:+.4f}*s^2  $c_2$={b2[1]:+.4f}*s"
              f"  $c_3$={b2[2]:+.4f}   $R^2$={r2_2:.4f}")

# Model 3 predictions at the 20 bin points (depends on both spread & prate,
# so we connect them with a step-sorted line rather than a smooth curve).
order = np.argsort(s)
ax.plot(s[order], yhat3[order], color="#dc2626", lw=2.2, ls=":",
        marker="x", markersize=7, markeredgewidth=1.8, zorder=4,
        label=f"[3] Vlad bond: $c_1$={b3[0]:+.4f}*s  $c_2$={b3[1]:+.4f}*p$^{{0.4}}$"
              f"  $c_3$={b3[2]:+.4f}   $R^2$={r2_3:.4f}")

ax.set_xlabel("Mean roll_spread_500 (bps, bin mean)", fontsize=11)
ax.set_ylabel("|impact_vwap_bps| (bin mean, bps)", fontsize=11)
ax.set_title(
    "AAPL lit buy block trades — three model fits on 20-bin means\n"
    "(bins by roll_spread_500;  target = |impact_vwap_bps|)",
    fontsize=12, fontweight="bold",
)

# R² summary box
summary_text = (
    f"$R^2$ summary (n=20 bins)\n"
    f"[1] Linear:      {r2_1:.4f}\n"
    f"[2] Quadratic:  {r2_2:.4f}\n"
    f"[3] Vlad bond:  {r2_3:.4f}"
)
ax.text(
    0.97, 0.05, summary_text,
    transform=ax.transAxes,
    fontsize=9.5, family="monospace",
    verticalalignment="bottom", horizontalalignment="right",
    bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
              edgecolor="#94a3b8", alpha=0.92),
)

ax.legend(fontsize=8.5, loc="upper left", framealpha=0.92)
ax.grid(True, alpha=0.2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("aapl_model_fits.png", dpi=150, bbox_inches="tight")
print("\nsaved -> aapl_model_fits.png")
