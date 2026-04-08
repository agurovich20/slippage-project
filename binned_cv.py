"""
Time-series 5-fold CV at the binned level.

Split logic (same date-block scheme as model_comparison.py):
  - Divide unique dates into 5 consecutive blocks.
  - Walk-forward: train on blocks 0..k-1, test on block k  (k = 1..4 → 4 OOS evals).

Per fold:
  1. Training days  → 50 quantile bins by roll_spread_500 → bin means (spread, prate, abs_impact)
     Fit 3 models on those 50 training-bin means.
  2. Test days      → 50 fresh quantile bins by roll_spread_500 → bin means
     Plug test-bin mean_spread (and mean_prate for Vlad) into fitted model → predicted mean_abs.
     OOS R² on the 50 test-bin (actual, predicted) pairs.

Models:
  [1] Linear:    abs = c1*spread + c2
  [2] Quadratic: abs = c1*spread^2 + c2*spread + c3
  [3] Vlad bond: abs = c1*spread + c2*prate^0.4 + c3

Output: aapl_binned_cv.png
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
df["abs_impact"] = df["impact_vwap_bps"].abs()
df = df.sort_values("date").reset_index(drop=True)

COLS   = ["date", "roll_spread_500", "participation_rate", "abs_impact"]
df     = df[COLS].dropna()
dates  = df["date"].to_numpy()

print(f"Loaded {len(df):,} rows across {df['date'].nunique()} dates")

N_BINS = 50
N_FOLDS = 5

# ── Date-block fold assignment (identical to model_comparison.py) ──────────────
unique_dates = np.array(sorted(df["date"].unique()))
n_days = len(unique_dates)

date_fold = np.digitize(
    np.arange(n_days),
    bins=np.linspace(0, n_days, N_FOLDS + 1)[1:-1],
)
date_to_fold = dict(zip(unique_dates, date_fold))
row_fold = np.array([date_to_fold[d] for d in dates])

print(f"\nFold sizes:")
for f in range(N_FOLDS):
    mask = row_fold == f
    days_in = np.unique(dates[mask])
    print(f"  block {f}: {mask.sum():>6,} rows  |  {len(days_in):2d} days  "
          f"({days_in[0]} .. {days_in[-1]})")

# Walk-forward splits
splits = []
for k in range(1, N_FOLDS):
    tr_idx = np.where(row_fold < k)[0]
    te_idx = np.where(row_fold == k)[0]
    splits.append((tr_idx, te_idx))

print(f"\n{len(splits)} walk-forward OOS splits\n")


# ── Helpers ────────────────────────────────────────────────────────────────────
def make_bins(data_idx, n_bins):
    """
    Compute quantile bins on roll_spread_500 for the given row indices.
    Returns DataFrame with columns: mean_spread, mean_prate, mean_abs, count.
    """
    sub = df.iloc[data_idx][["roll_spread_500", "participation_rate", "abs_impact"]].copy()
    sub["bin"] = pd.qcut(sub["roll_spread_500"], q=n_bins, labels=False, duplicates="drop")
    agg = (
        sub.groupby("bin", observed=True)
        .agg(
            mean_spread=("roll_spread_500",  "mean"),
            mean_prate =("participation_rate","mean"),
            mean_abs   =("abs_impact",        "mean"),
            count      =("abs_impact",        "count"),
        )
        .reset_index(drop=True)
    )
    return agg


def ols_fit(X_design, y):
    """Closed-form OLS. Returns beta, R²."""
    beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    y_hat = X_design @ beta
    ss_res = float(((y - y_hat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return beta, r2


def r2_score(y_true, y_pred):
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


# ── CV loop ────────────────────────────────────────────────────────────────────
model_names = ["Linear", "Quadratic", "Vlad bond"]
oos_r2 = {m: [] for m in model_names}

for fold_idx, (tr_idx, te_idx) in enumerate(splits):
    # ── Training bins ────────────────────────────────────────────────────────
    tr_bins = make_bins(tr_idx, N_BINS)
    s_tr = tr_bins["mean_spread"].to_numpy()
    p_tr = tr_bins["mean_prate"].to_numpy()
    y_tr = tr_bins["mean_abs"].to_numpy()
    n_tr = len(tr_bins)

    # ── Test bins ────────────────────────────────────────────────────────────
    te_bins = make_bins(te_idx, N_BINS)
    s_te = te_bins["mean_spread"].to_numpy()
    p_te = te_bins["mean_prate"].to_numpy()
    y_te = te_bins["mean_abs"].to_numpy()
    n_te = len(te_bins)

    print(f"--- Fold {fold_idx+1}/{len(splits)}  "
          f"(train bins={n_tr}, test bins={n_te}) ---")

    # ── Model 1: Linear ──────────────────────────────────────────────────────
    X1_tr = np.column_stack([s_tr, np.ones(n_tr)])
    b1, r2_tr1 = ols_fit(X1_tr, y_tr)
    y_hat1 = b1[0] * s_te + b1[1]
    r2_oos1 = r2_score(y_te, y_hat1)
    oos_r2["Linear"].append(r2_oos1)
    print(f"  Linear:    train R²={r2_tr1:.4f}  OOS R²={r2_oos1:+.4f}  "
          f"c=({b1[0]:+.4f}, {b1[1]:+.4f})")

    # ── Model 2: Quadratic ───────────────────────────────────────────────────
    X2_tr = np.column_stack([s_tr**2, s_tr, np.ones(n_tr)])
    b2, r2_tr2 = ols_fit(X2_tr, y_tr)
    y_hat2 = b2[0] * s_te**2 + b2[1] * s_te + b2[2]
    r2_oos2 = r2_score(y_te, y_hat2)
    oos_r2["Quadratic"].append(r2_oos2)
    print(f"  Quadratic: train R²={r2_tr2:.4f}  OOS R²={r2_oos2:+.4f}  "
          f"c=({b2[0]:+.4f}, {b2[1]:+.4f}, {b2[2]:+.4f})")

    # ── Model 3: Vlad bond ───────────────────────────────────────────────────
    X3_tr = np.column_stack([s_tr, p_tr**0.4, np.ones(n_tr)])
    b3, r2_tr3 = ols_fit(X3_tr, y_tr)
    y_hat3 = b3[0] * s_te + b3[1] * p_te**0.4 + b3[2]
    r2_oos3 = r2_score(y_te, y_hat3)
    oos_r2["Vlad bond"].append(r2_oos3)
    print(f"  Vlad bond: train R²={r2_tr3:.4f}  OOS R²={r2_oos3:+.4f}  "
          f"c=({b3[0]:+.4f}, {b3[1]:+.4f}, {b3[2]:+.4f})")

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print(f"{'Model':<12}  {'Mean OOS R2':>12}  {'Std OOS R2':>11}  Fold R2s")
print("="*60)
for name in model_names:
    arr = np.array(oos_r2[name])
    fold_str = "  ".join(f"{v:+.4f}" for v in arr)
    print(f"  {name:<10}  {arr.mean():>+12.4f}  {arr.std():>11.4f}  [{fold_str}]")
print("="*60)

# ── Bar chart ─────────────────────────────────────────────────────────────────
means = [np.array(oos_r2[m]).mean() for m in model_names]
stds  = [np.array(oos_r2[m]).std()  for m in model_names]
colors = ["#2563eb", "#16a34a", "#dc2626"]

fig, ax = plt.subplots(figsize=(8, 5.5))

bars = ax.bar(
    model_names, means,
    yerr=stds, capsize=6,
    color=colors, edgecolor="white", linewidth=0.8, width=0.5,
    error_kw=dict(elinewidth=1.8, capthick=1.8, ecolor="black"),
)

# Individual fold dots
for i, name in enumerate(model_names):
    fold_vals = oos_r2[name]
    jitter = np.linspace(-0.12, 0.12, len(fold_vals))
    ax.scatter(
        [i + j for j in jitter], fold_vals,
        color="white", edgecolors=colors[i],
        s=55, zorder=5, linewidths=1.8,
    )

# Value labels
for bar, val, err in zip(bars, means, stds):
    y_pos = val + err + 0.01 if val >= 0 else val - err - 0.01
    va    = "bottom" if val >= 0 else "top"
    ax.text(
        bar.get_x() + bar.get_width() / 2, y_pos,
        f"{val:+.4f}\n(±{err:.4f})",
        ha="center", va=va, fontsize=9, fontweight="bold",
    )

ax.axhline(0, color="black", lw=1, ls="--", alpha=0.5)
ax.set_xticks(range(len(model_names)))
ax.set_xticklabels(
    ["[1] Linear\nc1*s + c2",
     "[2] Quadratic\nc1*s^2 + c2*s + c3",
     "[3] Vlad bond\nc1*s + c2*p^0.4 + c3"],
    fontsize=9.5,
)
ax.set_ylabel("OOS R2 on 50 test-bin means", fontsize=11)
ax.set_title(
    "Binned CV: 3 parametric models on |impact_vwap_bps|\n"
    "5-block walk-forward, 50 quantile bins by roll_spread_500\n"
    "(dots = individual fold OOS R2;  bar = mean;  error bar = std)",
    fontsize=10.5, fontweight="bold",
)
ax.grid(axis="y", alpha=0.25)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("aapl_binned_cv.png", dpi=150, bbox_inches="tight")
print("\nsaved -> aapl_binned_cv.png")
