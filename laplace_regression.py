"""
Parametric Laplace regression for abs(impact_vwap_bps).

Three components
----------------
(1) LAD regression (L1 / quantile regression at tau=0.5):
      median(|impact|) = c1*spread + c2
    via exact LP (scipy.linprog HiGHS).  Compare to OLS on same 50-bin means.

(2) Spread-dependent Laplace scale b:
    Fit Laplace(loc, b) to abs(impact) within each of 5 spread quintiles.
    Then fit:  b = a1*spread + a2  (OLS on 5 quintile b-values).

(3) Full probabilistic model: Laplace(mu(spread), b(spread))
    mu(spread) = c1*spread + c2    [from LAD]
    b(spread)  = a1*spread + a2    [from quintile Laplace fits]
    90% PI: [max(0, mu - ln(10)*b),  mu + ln(10)*b]

5-fold time-series CV (same date-block scheme as model_comparison.py):
  Per fold: fit (1) and (2) on TRAINING BINS;
            evaluate OOS R² on TEST BIN MEANS (50 bins);
            evaluate distributional log-likelihood on individual TEST TRADES.

Output: aapl_laplace_regression.png
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"]      = "1"

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linprog
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Load ───────────────────────────────────────────────────────────────────────
df = pd.read_parquet("data/lit_buy_features_v2.parquet")
df["abs_impact"] = df["impact_vwap_bps"].abs()
df = df.sort_values("date").reset_index(drop=True)
df["q5"] = pd.qcut(df["roll_spread_500"], q=5, labels=False, duplicates="drop")

dates = df["date"].to_numpy()
print(f"Loaded {len(df):,} trades, {df['date'].nunique()} dates")


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

N_BINS   = 50
N_FOLDS  = 5
LN10     = np.log(10.0)      # ≈ 2.3026 — Laplace 5th/95th quantile multiplier

def r2(ytrue, ypred):
    ss_res = ((ytrue - ypred)**2).sum()
    ss_tot = ((ytrue - ytrue.mean())**2).sum()
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

def fit_ols(x, y):
    """OLS: y = c1*x + c2. Returns (c1, c2, in-sample R²)."""
    Xm = np.column_stack([x, np.ones(len(x))])
    beta, *_ = np.linalg.lstsq(Xm, y, rcond=None)
    yh = Xm @ beta
    return beta[0], beta[1], r2(y, yh)

def fit_lad(x, y):
    """
    LAD regression via exact LP (HiGHS):
      min  sum |y_i - c1*x_i - c2|
    Variables: [c1, c2, u_0..n-1, l_0..n-1]
    Constraints: c1*x_i + c2 - u_i + l_i = y_i,  u_i, l_i >= 0
    """
    n   = len(x)
    nv  = 2 + 2 * n

    c_cost           = np.zeros(nv)
    c_cost[2:2+n]    = 1.0          # u's
    c_cost[2+n:]     = 1.0          # l's

    A_eq             = np.zeros((n, nv))
    A_eq[:, 0]       = x
    A_eq[:, 1]       = 1.0
    A_eq[np.arange(n), 2+np.arange(n)]   = -1.0   # -u_i
    A_eq[np.arange(n), 2+n+np.arange(n)] =  1.0   # +l_i

    bounds = [(None, None), (None, None)] + [(0, None)] * (2 * n)

    res  = linprog(c_cost, A_eq=A_eq, b_eq=y, bounds=bounds, method="highs")
    c1, c2 = res.x[0], res.x[1]
    yh     = c1 * x + c2
    return c1, c2, r2(y, yh)

def fit_scale_model(data_idx):
    """
    Fit Laplace to abs(impact) in each of 5 spread quintiles (training rows).
    Returns (a1, a2) where b(spread) = a1*spread + a2.
    Also returns arrays (quintile_spread_means, quintile_b_values).
    """
    sub  = df.iloc[data_idx][["roll_spread_500", "abs_impact", "q5"]].copy()
    smids, bs = [], []
    for q in range(5):
        mask = sub["q5"] == q
        if mask.sum() < 20:
            continue
        xi   = sub.loc[mask, "abs_impact"].to_numpy(dtype=np.float64)
        _, b = stats.laplace.fit(xi)
        smids.append(sub.loc[mask, "roll_spread_500"].mean())
        bs.append(b)
    smids, bs = np.array(smids), np.array(bs)
    a1, a2, _ = fit_ols(smids, bs)
    return a1, a2, smids, bs

def make_bins(data_idx):
    sub = df.iloc[data_idx][["roll_spread_500", "abs_impact"]].copy()
    sub["bin"] = pd.qcut(sub["roll_spread_500"], q=N_BINS,
                          labels=False, duplicates="drop")
    return (
        sub.groupby("bin", observed=True)
        .agg(mean_spread=("roll_spread_500", "mean"),
             mean_abs    =("abs_impact",      "mean"),
             count       =("abs_impact",      "count"))
        .reset_index(drop=True)
    )


# ══════════════════════════════════════════════════════════════════════════════
# (1) & (2)  Overall analysis (full dataset)
# ══════════════════════════════════════════════════════════════════════════════
all_idx  = np.arange(len(df))
bins_all = make_bins(all_idx)
s_all    = bins_all["mean_spread"].to_numpy()
y_all    = bins_all["mean_abs"].to_numpy()
cnt_all  = bins_all["count"].to_numpy()

c1_ols, c2_ols, r2_ols = fit_ols(s_all, y_all)
c1_lad, c2_lad, r2_lad = fit_lad(s_all, y_all)
a1_b,   a2_b, smids_q, bs_q = fit_scale_model(all_idx)
_, a2_b_ols, _ = fit_ols(smids_q, bs_q)   # same, just labelled

print(f"\n{'='*64}")
print(f"Component (1): LAD vs OLS on 50-bin means")
print(f"{'='*64}")
print(f"{'':25}  {'c1':>10}  {'c2':>10}  {'in-sample R2':>13}")
print(f"  {'OLS':<23}  {c1_ols:>+10.5f}  {c2_ols:>+10.5f}  {r2_ols:>13.4f}")
print(f"  {'LAD':<23}  {c1_lad:>+10.5f}  {c2_lad:>+10.5f}  {r2_lad:>13.4f}")

print(f"\n{'='*64}")
print(f"Component (2): Spread-dependent Laplace scale  b = a1*spread + a2")
print(f"{'='*64}")
print(f"  {'Quintile':>8}  {'mean_spread':>12}  {'b (Laplace scale)':>18}")
for q, (sm, bv) in enumerate(zip(smids_q, bs_q)):
    b_pred = a1_b * sm + a2_b
    print(f"  {'Q'+str(q+1):>8}  {sm:>12.4f}  {bv:>18.4f}  (predicted: {b_pred:.4f})")
print(f"\n  b(spread) = {a1_b:+.6f}*spread {a2_b:+.6f}")
_, _, r2_b = fit_ols(smids_q, bs_q)
print(f"  R2 of b(spread) model on 5 quintile points: {r2_b:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# (3)  5-fold CV
# ══════════════════════════════════════════════════════════════════════════════
unique_dates = np.array(sorted(df["date"].unique()))
n_days  = len(unique_dates)
d_fold  = np.digitize(np.arange(n_days),
                      bins=np.linspace(0, n_days, N_FOLDS+1)[1:-1])
d2f     = dict(zip(unique_dates, d_fold))
row_fold = np.array([d2f[d] for d in dates])

splits = [(np.where(row_fold < k)[0], np.where(row_fold == k)[0])
          for k in range(1, N_FOLDS)]

print(f"\n{'='*64}")
print(f"5-fold walk-forward CV  ({len(splits)} OOS evaluations)")
print(f"{'='*64}")

cv_r2_ols  = []
cv_r2_lad  = []
cv_ll_dist = []   # per-trade log-likelihood of full Laplace model

for fi, (tr_idx, te_idx) in enumerate(splits):

    # ── Training bins → fit LAD + OLS ───────────────────────────────────────
    tr_bins = make_bins(tr_idx)
    s_tr    = tr_bins["mean_spread"].to_numpy()
    y_tr    = tr_bins["mean_abs"].to_numpy()

    c1_o, c2_o, _ = fit_ols(s_tr, y_tr)
    c1_l, c2_l, _ = fit_lad(s_tr, y_tr)

    # ── Training trades → scale model ────────────────────────────────────────
    a1, a2, *_ = fit_scale_model(tr_idx)

    # ── Test bins → OOS R² ───────────────────────────────────────────────────
    te_bins = make_bins(te_idx)
    s_te    = te_bins["mean_spread"].to_numpy()
    y_te    = te_bins["mean_abs"].to_numpy()

    yhat_ols_te = c1_o * s_te + c2_o
    yhat_lad_te = c1_l * s_te + c2_l

    r2_o = r2(y_te, yhat_ols_te)
    r2_l = r2(y_te, yhat_lad_te)
    cv_r2_ols.append(r2_o)
    cv_r2_lad.append(r2_l)

    # ── Test trades → distributional log-likelihood ──────────────────────────
    te_df     = df.iloc[te_idx]
    sp_trades = te_df["roll_spread_500"].to_numpy(dtype=np.float64)
    ab_trades = te_df["abs_impact"].to_numpy(dtype=np.float64)

    mu_pred = c1_l * sp_trades + c2_l
    b_pred  = np.maximum(a1 * sp_trades + a2, 1e-3)
    ll_i    = stats.laplace.logpdf(ab_trades, mu_pred, b_pred)
    per_trade_ll = ll_i.mean()
    cv_ll_dist.append(per_trade_ll)

    print(f"  fold {fi+1}:  OLS R2={r2_o:+.4f}  LAD R2={r2_l:+.4f}  "
          f"LL/trade={per_trade_ll:.4f}  "
          f"(b model: {a1:+.4f}*s {a2:+.4f})")

print(f"\n{'='*64}")
print(f"{'Metric':<28}  {'Mean':>8}  {'Std':>8}")
print(f"  {'-'*40}")
print(f"  {'OLS OOS R2 (bins)':<26}  {np.mean(cv_r2_ols):>+8.4f}  {np.std(cv_r2_ols):>8.4f}")
print(f"  {'LAD OOS R2 (bins)':<26}  {np.mean(cv_r2_lad):>+8.4f}  {np.std(cv_r2_lad):>8.4f}")
print(f"  {'Laplace LL / trade':<26}  {np.mean(cv_ll_dist):>+8.4f}  {np.std(cv_ll_dist):>8.4f}")
print(f"{'='*64}")


# ══════════════════════════════════════════════════════════════════════════════
# Plot
# ══════════════════════════════════════════════════════════════════════════════
s_grid  = np.linspace(s_all.min() * 0.95, s_all.max() * 1.02, 400)

mu_ols  = c1_ols * s_grid + c2_ols
mu_lad  = c1_lad * s_grid + c2_lad
b_grid  = np.maximum(a1_b * s_grid + a2_b, 1e-3)

pi_lo   = np.maximum(mu_lad - LN10 * b_grid, 0.0)   # clip at 0 (abs values)
pi_hi   = mu_lad + LN10 * b_grid

fig, ax = plt.subplots(figsize=(11, 6.5))

# 90% Laplace PI band
ax.fill_between(s_grid, pi_lo, pi_hi,
                alpha=0.18, color="#dc2626", label="90% Laplace PI  [LAD median ± ln(10)·b(spread)]")
ax.plot(s_grid, pi_lo, color="#dc2626", lw=0.9, ls="--", alpha=0.55)
ax.plot(s_grid, pi_hi, color="#dc2626", lw=0.9, ls="--", alpha=0.55)

# OLS line
ax.plot(s_grid, mu_ols, color="#2563eb", lw=2.2, zorder=5,
        label=f"OLS:  {c1_ols:+.4f}·spread {c2_ols:+.4f}   (CV R2={np.mean(cv_r2_ols):+.3f})")

# LAD line
ax.plot(s_grid, mu_lad, color="#16a34a", lw=2.2, ls="--", zorder=5,
        label=f"LAD:  {c1_lad:+.4f}·spread {c2_lad:+.4f}   (CV R2={np.mean(cv_r2_lad):+.3f})")

# Bin scatter — size proportional to count
sizes = 45 + 130 * (cnt_all / cnt_all.max())
sc = ax.scatter(s_all, y_all, s=sizes, zorder=6,
                color="#1e293b", edgecolors="white", linewidths=0.7,
                label=f"50-bin means  (area proportional to count)")

# Quintile b-values (secondary context)
ax2 = ax.twinx()
ax2.scatter(smids_q, bs_q, color="#f59e0b", s=85, zorder=7,
            marker="D", edgecolors="white", linewidths=0.8, label="Quintile Laplace b")
b_fit_line = a1_b * s_grid + a2_b
ax2.plot(s_grid, b_fit_line, color="#f59e0b", lw=1.6, ls=":",
         label=f"b(spread) = {a1_b:+.4f}·s {a2_b:+.4f}")
ax2.set_ylabel("Laplace scale b  (bps)", color="#b45309", fontsize=10)
ax2.tick_params(axis="y", colors="#b45309")
ax2.set_ylim(0, b_grid.max() * 2.2)
ax2.spines["right"].set_edgecolor("#b45309")

# Combine legends
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=8.5, loc="upper left",
          framealpha=0.93, ncol=1)

ax.set_xlabel("Mean roll_spread_500 per bin (bps)", fontsize=11)
ax.set_ylabel("|impact_vwap_bps| (bin mean, bps)", fontsize=11)
ax.set_title(
    "AAPL lit buy block trades — Laplace regression on |impact_vwap_bps|\n"
    "LAD (L1) vs OLS on 50-bin means  +  90% Laplace PI  [b widens with spread]\n"
    f"Laplace LL/trade = {np.mean(cv_ll_dist):.4f} (mean OOS, 4 folds)",
    fontsize=11, fontweight="bold",
)
ax.set_xlim(s_grid[0], s_grid[-1])
ax.set_ylim(0, pi_hi.max() * 1.05)
ax.grid(True, alpha=0.18)
ax.spines["top"].set_visible(False)

plt.tight_layout()
plt.savefig("aapl_laplace_regression.png", dpi=150, bbox_inches="tight")
print("\nsaved -> aapl_laplace_regression.png")
