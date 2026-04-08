"""
Individual-trade LAD vs OLS regression for abs(impact_vwap_bps).

Models
------
  Univariate:    target = c1*spread + c2
  Multivariate:  target = c1*spread + c2*vol + c3*prate + c4

Methods: OLS (sklearn/numpy) and LAD (statsmodels QuantReg, tau=0.5).

5-fold time-series walk-forward CV by date (same scheme as model_comparison.py):
  blocks 0..4 assigned by date order; fold k trains on rows in blocks < k,
  tests on rows in block k; produces 4 OOS evaluations.

OOS metrics computed on individual test trades (not bins):
  - OOS R²  (standard coefficient of determination on squared errors)
  - OOS MAE (mean absolute error)

Output: aapl_lad_vs_ols_trade.png
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"]      = "1"

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg

# ── Load ───────────────────────────────────────────────────────────────────────
df = pd.read_parquet("data/lit_buy_features_v2.parquet")
df["abs_impact"] = df["impact_vwap_bps"].abs()
df = df.sort_values("date").reset_index(drop=True)
print(f"Loaded {len(df):,} trades across {df['date'].nunique()} dates")

# ── Feature matrices ──────────────────────────────────────────────────────────
# Univariate: [spread, 1]
X_uni = np.column_stack([
    df["roll_spread_500"].to_numpy(dtype=np.float64),
    np.ones(len(df)),
])

# Multivariate: [spread, vol, prate, 1]
X_mul = np.column_stack([
    df["roll_spread_500"].to_numpy(dtype=np.float64),
    df["roll_vol_500"].to_numpy(dtype=np.float64),
    df["participation_rate"].to_numpy(dtype=np.float64),
    np.ones(len(df)),
])

y_all = df["abs_impact"].to_numpy(dtype=np.float64)
dates = df["date"].to_numpy()


# ── Metric helpers ─────────────────────────────────────────────────────────────
def r2(ytrue, ypred):
    ss_res = ((ytrue - ypred) ** 2).sum()
    ss_tot = ((ytrue - ytrue.mean()) ** 2).sum()
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

def mae(ytrue, ypred):
    return np.mean(np.abs(ytrue - ypred))


# ── Fit helpers ────────────────────────────────────────────────────────────────
def fit_ols(X_tr, y_tr, X_te):
    beta, *_ = np.linalg.lstsq(X_tr, y_tr, rcond=None)
    return X_te @ beta

def fit_lad(X_tr, y_tr, X_te):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        qr  = QuantReg(y_tr, X_tr)
        res = qr.fit(q=0.5, max_iter=2000, p_tol=1e-6)
    return X_te @ res.params


# ── 5-fold walk-forward CV ─────────────────────────────────────────────────────
N_FOLDS = 5
unique_dates = np.array(sorted(df["date"].unique()))
n_days  = len(unique_dates)
d_fold  = np.digitize(np.arange(n_days),
                      bins=np.linspace(0, n_days, N_FOLDS + 1)[1:-1])
d2f     = dict(zip(unique_dates, d_fold))
row_fold = np.array([d2f[d] for d in dates])

# splits: (train_idx, test_idx) for k=1..4
splits = [(np.where(row_fold < k)[0], np.where(row_fold == k)[0])
          for k in range(1, N_FOLDS)]

print(f"\n5-fold walk-forward CV  ({len(splits)} OOS evaluations)\n")
print(f"{'Model':<22}  {'Fold':>5}  {'OOS R2':>9}  {'OOS MAE':>9}")
print("-" * 52)

# Store per-fold results
fold_results = {
    "OLS-Uni":   {"r2": [], "mae": []},
    "LAD-Uni":   {"r2": [], "mae": []},
    "OLS-Multi": {"r2": [], "mae": []},
    "LAD-Multi": {"r2": [], "mae": []},
}

MODELS = [
    ("OLS-Uni",   X_uni, fit_ols),
    ("LAD-Uni",   X_uni, fit_lad),
    ("OLS-Multi", X_mul, fit_ols),
    ("LAD-Multi", X_mul, fit_lad),
]

for fi, (tr_idx, te_idx) in enumerate(splits):
    y_tr = y_all[tr_idx]
    y_te = y_all[te_idx]

    for name, X, fitter in MODELS:
        X_tr = X[tr_idx]
        X_te = X[te_idx]
        ypred = np.maximum(fitter(X_tr, y_tr, X_te), 0.0)   # clip at 0
        rv  = r2(y_te, ypred)
        mv  = mae(y_te, ypred)
        fold_results[name]["r2"].append(rv)
        fold_results[name]["mae"].append(mv)
        print(f"  {name:<20}  {fi+1:>5}  {rv:>+9.4f}  {mv:>9.4f}")

# ── Summary table ──────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"{'Model':<22}  {'Mean R2':>9}  {'Std R2':>8}  {'Mean MAE':>10}  {'Std MAE':>8}")
print(f"  {'-'*54}")
for name, _ in [(m[0], None) for m in MODELS]:
    r2s  = np.array(fold_results[name]["r2"])
    maes = np.array(fold_results[name]["mae"])
    print(f"  {name:<20}  {r2s.mean():>+9.4f}  {r2s.std():>8.4f}  "
          f"{maes.mean():>10.4f}  {maes.std():>8.4f}")
print(f"{'='*60}")


# ── Full-data fits for the plot ────────────────────────────────────────────────
print("\nFitting full-dataset models for plot...")

# OLS coefficients on full data
beta_ols_uni, *_ = np.linalg.lstsq(X_uni, y_all, rcond=None)
beta_ols_mul, *_ = np.linalg.lstsq(X_mul, y_all, rcond=None)

# LAD on full data
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    qr_uni = QuantReg(y_all, X_uni).fit(q=0.5, max_iter=2000, p_tol=1e-6)
    qr_mul = QuantReg(y_all, X_mul).fit(q=0.5, max_iter=2000, p_tol=1e-6)

beta_lad_uni = qr_uni.params
beta_lad_mul = qr_mul.params

print(f"  OLS-Uni:   c1={beta_ols_uni[0]:+.5f}  c2={beta_ols_uni[1]:+.5f}")
print(f"  LAD-Uni:   c1={beta_lad_uni[0]:+.5f}  c2={beta_lad_uni[1]:+.5f}")
print(f"  OLS-Multi: spread={beta_ols_mul[0]:+.5f}  vol={beta_ols_mul[1]:+.5f}  "
      f"prate={beta_ols_mul[2]:+.5f}  c4={beta_ols_mul[3]:+.5f}")
print(f"  LAD-Multi: spread={beta_lad_mul[0]:+.5f}  vol={beta_lad_mul[1]:+.5f}  "
      f"prate={beta_lad_mul[2]:+.5f}  c4={beta_lad_mul[3]:+.5f}")


# ── Plot ───────────────────────────────────────────────────────────────────────
MODEL_NAMES   = ["OLS-Uni", "LAD-Uni", "OLS-Multi", "LAD-Multi"]
COLORS        = ["#2563eb", "#16a34a", "#7c3aed", "#dc2626"]
MEAN_R2       = [np.mean(fold_results[n]["r2"])  for n in MODEL_NAMES]
STD_R2        = [np.std(fold_results[n]["r2"])   for n in MODEL_NAMES]
MEAN_MAE      = [np.mean(fold_results[n]["mae"]) for n in MODEL_NAMES]
STD_MAE       = [np.std(fold_results[n]["mae"])  for n in MODEL_NAMES]

fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.32)
ax1 = fig.add_subplot(gs[0, 0])   # OOS R² bar
ax2 = fig.add_subplot(gs[0, 1])   # OOS MAE bar
ax3 = fig.add_subplot(gs[1, 0])   # Scatter + lines (univariate, spread)
ax4 = fig.add_subplot(gs[1, 1])   # Per-fold R² by model

xpos = np.arange(len(MODEL_NAMES))

# ── Panel 1: OOS R² bars ──────────────────────────────────────────────────────
bars1 = ax1.bar(xpos, MEAN_R2, yerr=STD_R2, color=COLORS,
                capsize=5, width=0.55, edgecolor="white", linewidth=0.8)
for bar, v in zip(bars1, MEAN_R2):
    ax1.text(bar.get_x() + bar.get_width() / 2, v + (0.003 if v >= 0 else -0.009),
             f"{v:+.4f}", ha="center", va="bottom" if v >= 0 else "top",
             fontsize=9, fontweight="bold")
ax1.axhline(0, color="black", lw=0.9, ls="--", alpha=0.5)
ax1.set_xticks(xpos)
ax1.set_xticklabels(MODEL_NAMES, fontsize=10)
ax1.set_ylabel("OOS R²  (individual trades)", fontsize=10.5)
ax1.set_title("OOS R²: individual-trade predictions\n(mean ± std across 4 folds)",
              fontsize=11, fontweight="bold")
ax1.grid(True, axis="y", alpha=0.2)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# ── Panel 2: OOS MAE bars ─────────────────────────────────────────────────────
bars2 = ax2.bar(xpos, MEAN_MAE, yerr=STD_MAE, color=COLORS,
                capsize=5, width=0.55, edgecolor="white", linewidth=0.8)
for bar, v in zip(bars2, MEAN_MAE):
    ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.002,
             f"{v:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax2.set_xticks(xpos)
ax2.set_xticklabels(MODEL_NAMES, fontsize=10)
ax2.set_ylabel("OOS MAE  (bps, individual trades)", fontsize=10.5)
ax2.set_title("OOS MAE: individual-trade predictions\n(mean ± std across 4 folds)",
              fontsize=11, fontweight="bold")
ax2.grid(True, axis="y", alpha=0.2)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

# ── Panel 3: Scatter abs_impact vs spread + OLS and LAD lines ─────────────────
spread_all = df["roll_spread_500"].to_numpy(dtype=np.float64)
rng   = np.random.default_rng(42)
samp  = rng.choice(len(df), size=min(8000, len(df)), replace=False)

ax3.scatter(spread_all[samp], y_all[samp],
            alpha=0.06, s=5, color="#94a3b8", linewidths=0,
            label=f"Trades (8k sample of {len(df):,})")

s_lo, s_hi = np.percentile(spread_all, 0.5), np.percentile(spread_all, 99.5)
s_grid = np.linspace(s_lo, s_hi, 400)

mu_ols_line = beta_ols_uni[0] * s_grid + beta_ols_uni[1]
mu_lad_line = beta_lad_uni[0] * s_grid + beta_lad_uni[1]

ax3.plot(s_grid, mu_ols_line, color="#2563eb", lw=2.2, zorder=5,
         label=f"OLS: {beta_ols_uni[0]:+.4f}·spread {beta_ols_uni[1]:+.4f}")
ax3.plot(s_grid, mu_lad_line, color="#16a34a", lw=2.2, ls="--", zorder=5,
         label=f"LAD: {beta_lad_uni[0]:+.4f}·spread {beta_lad_uni[1]:+.4f}")

ax3.set_xlim(s_lo, min(s_hi, 12))
ax3.set_ylim(0, np.percentile(y_all, 99))
ax3.set_xlabel("roll_spread_500 (bps)", fontsize=10.5)
ax3.set_ylabel("|impact_vwap_bps|  (bps)", fontsize=10.5)
ax3.set_title("Univariate fit: OLS vs LAD on individual trades\n"
              "(full dataset coefficients shown)",
              fontsize=11, fontweight="bold")
ax3.legend(fontsize=9, loc="upper left")
ax3.grid(True, alpha=0.18)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)

# ── Panel 4: Per-fold OOS R² for each model ───────────────────────────────────
fold_nums = np.arange(1, len(splits) + 1)
for name, color in zip(MODEL_NAMES, COLORS):
    r2s = fold_results[name]["r2"]
    ax4.plot(fold_nums, r2s, color=color, lw=2.0, marker="o", ms=7,
             label=name)
    for fi, rv in enumerate(r2s):
        ax4.annotate(f"{rv:+.3f}", (fold_nums[fi], rv),
                     textcoords="offset points", xytext=(4, 3),
                     fontsize=7.5, color=color)

ax4.axhline(0, color="black", lw=0.9, ls="--", alpha=0.5)
ax4.set_xticks(fold_nums)
ax4.set_xticklabels([f"Fold {k}" for k in fold_nums], fontsize=9.5)
ax4.set_ylabel("OOS R²  (individual trades)", fontsize=10.5)
ax4.set_title("Per-fold OOS R² by model\n(walk-forward, each fold uses more training data)",
              fontsize=11, fontweight="bold")
ax4.legend(fontsize=9.5, loc="lower right")
ax4.grid(True, alpha=0.18)
ax4.spines["top"].set_visible(False)
ax4.spines["right"].set_visible(False)

fig.suptitle(
    "AAPL lit buy block trades — LAD (L1) vs OLS (L2) regression on |impact_vwap_bps|\n"
    "Individual-trade level (35,020 trades)  |  5-fold walk-forward CV by date",
    fontsize=12, fontweight="bold", y=1.01,
)

plt.savefig("aapl_lad_vs_ols_trade.png", dpi=150, bbox_inches="tight")
print("\nsaved -> aapl_lad_vs_ols_trade.png")
