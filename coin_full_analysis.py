"""
COIN full analysis — single script covering:

  (1) Distribution fits: Normal, Laplace, Skew-normal, RI(2022) on train signed impact
  (2) Binned regression: 20 bins by roll_spread_500, OLS on bin means
  (3) September holdout: OLS-Uni, LAD-Uni, XGB-MSE, XGB-MAE, RF-MAE
      All hyperparams from AAPL — no re-tuning on COIN.
  (4) Side-by-side AAPL vs COIN comparison table
  (5) Bootstrap 95% CIs on 959 test trades (1000 resamples)

Train: Jan-Aug 2024 (15,681 trades)  Test: Sep 2024 (959 trades)

Output: coin_full_analysis.png (3 panels, bar chart includes bootstrap CIs)
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from statsmodels.regression.quantile_regression import QuantReg
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ══════════════════════════════════════════════════════════════════════════════
# Load data
# ══════════════════════════════════════════════════════════════════════════════
df_tr = pd.read_parquet("data/coin_lit_buy_features_train.parquet")
df_te = pd.read_parquet("data/coin_lit_buy_features_test.parquet")

df_tr["abs_impact"] = df_tr["impact_vwap_bps"].abs()
df_te["abs_impact"] = df_te["impact_vwap_bps"].abs()

df_tr = df_tr.sort_values("date").reset_index(drop=True)
df_te = df_te.sort_values("date").reset_index(drop=True)

print(f"Train: {len(df_tr):,} trades  |  {df_tr['date'].nunique()} days  "
      f"({df_tr['date'].min()} .. {df_tr['date'].max()})")
print(f"Test : {len(df_te):,} trades  |  {df_te['date'].nunique()} days  "
      f"({df_te['date'].min()} .. {df_te['date'].max()})")

# ══════════════════════════════════════════════════════════════════════════════
# (1) Distribution fits on train signed impact_vwap_bps
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*72}")
print("(1) DISTRIBUTION FITS — COIN train impact_vwap_bps (signed)")
print(f"{'='*72}")

x_dist = df_tr["impact_vwap_bps"].to_numpy(dtype=np.float64)
n_dist = len(x_dist)

print(f"n={n_dist:,}  mean={x_dist.mean():.4f}  median={np.median(x_dist):.4f}  "
      f"std={x_dist.std():.4f}")

# ── RI(2022) helpers ─────────────────────────────────────────────────────────
SQRT_PI2 = np.sqrt(np.pi / 2.0)

def ri_neg_ll(params, x):
    mu, ls, lr = params
    s, r = np.exp(ls), np.exp(lr)
    Z  = s * SQRT_PI2 + r
    lm = x <= mu;  rm = ~lm
    return (len(x) * np.log(Z)
            + np.sum((x[lm] - mu)**2) / (2 * s**2)
            + np.sum(x[rm] - mu) / r)

def ri_pdf(x_grid, mu, s, r):
    C = 1.0 / (s * SQRT_PI2 + r)
    return np.where(x_grid <= mu,
                    C * np.exp(-((x_grid - mu)**2) / (2 * s**2)),
                    C * np.exp(-(x_grid - mu) / r))

def fit_ri(x):
    mu0  = np.median(x)
    left = x[x <= mu0];  right = x[x > mu0]
    s0   = max(left.std()  if len(left)  > 1 else 1.0, 0.05)
    r0   = max((right - mu0).mean() if len(right) > 1 else 1.0, 0.05)
    best = None
    for mu_s in (mu0, mu0 - 0.5*s0, mu0 + 0.3*r0):
        for sf in (0.5, 1.0, 2.0):
            for rf in (0.5, 1.0, 2.0):
                res = minimize(ri_neg_ll,
                               x0=[mu_s, np.log(s0*sf), np.log(r0*rf)],
                               args=(x,), method="Nelder-Mead",
                               options={"maxiter": 20_000, "xatol":1e-7, "fatol":1e-7})
                if best is None or res.fun < best.fun:
                    best = res
    return best.x[0], np.exp(best.x[1]), np.exp(best.x[2])

# Fit all 4 distributions
dist_results = []

# (a) Normal
mu_n, sig_n = stats.norm.fit(x_dist)
ll_n = stats.norm.logpdf(x_dist, mu_n, sig_n).sum()
dist_results.append(dict(model="Normal", k=2, ll=ll_n, aic=2*2 - 2*ll_n,
                         pdf_fn=lambda g, p=(mu_n, sig_n): stats.norm.pdf(g, *p)))

# (b) Laplace
loc_l, b_l = stats.laplace.fit(x_dist)
ll_l = stats.laplace.logpdf(x_dist, loc_l, b_l).sum()
dist_results.append(dict(model="Laplace", k=2, ll=ll_l, aic=2*2 - 2*ll_l,
                         pdf_fn=lambda g, p=(loc_l, b_l): stats.laplace.pdf(g, *p)))

# (c) Skew-normal
a_s, xi_s, om_s = stats.skewnorm.fit(x_dist)
ll_s = stats.skewnorm.logpdf(x_dist, a_s, xi_s, om_s).sum()
dist_results.append(dict(model="Skew-normal", k=3, ll=ll_s, aic=2*3 - 2*ll_s,
                         pdf_fn=lambda g, p=(a_s, xi_s, om_s): stats.skewnorm.pdf(g, *p)))

# (d) RI(2022)
mu_r, s_r, r_r = fit_ri(x_dist)
ll_r = -ri_neg_ll([mu_r, np.log(s_r), np.log(r_r)], x_dist)
dist_results.append(dict(model="RI(2022)", k=3, ll=ll_r, aic=2*3 - 2*ll_r,
                         pdf_fn=lambda g, p=(mu_r, s_r, r_r): ri_pdf(g, *p)))

print(f"\n  {'Model':<14} {'k':>2}  {'logL':>12}  {'AIC':>12}")
print(f"  {'-'*44}")
best_aic = min(r["aic"] for r in dist_results)
for r in dist_results:
    tag = " <-- best" if r["aic"] == best_aic else ""
    print(f"  {r['model']:<14} {r['k']:>2}  {r['ll']:>12.2f}  {r['aic']:>12.2f}{tag}")


# ══════════════════════════════════════════════════════════════════════════════
# (2) Binned regression: 20 bins by roll_spread_500
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*72}")
print("(2) BINNED REGRESSION — 20 bins by roll_spread_500 (train)")
print(f"{'='*72}")

N_BINS = 20
df_tr["bin"] = pd.qcut(df_tr["roll_spread_500"], q=N_BINS, labels=False, duplicates="drop")

bins_df = (
    df_tr.groupby("bin", observed=True)
    .agg(mean_spread=("roll_spread_500", "mean"),
         mean_abs   =("abs_impact",      "mean"),
         count      =("abs_impact",      "count"))
    .reset_index(drop=True)
)

s_bins = bins_df["mean_spread"].to_numpy()
y_bins = bins_df["mean_abs"].to_numpy()
cnt_bins = bins_df["count"].to_numpy()

# OLS on bin means
Xm = np.column_stack([s_bins, np.ones(len(s_bins))])
beta_bin, *_ = np.linalg.lstsq(Xm, y_bins, rcond=None)
c1_bin, c2_bin = beta_bin
yhat_bin = Xm @ beta_bin
ss_res = ((y_bins - yhat_bin)**2).sum()
ss_tot = ((y_bins - y_bins.mean())**2).sum()
r2_bin = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

print(f"  OLS: abs_impact = {c1_bin:+.4f} * spread {c2_bin:+.4f}")
print(f"  In-sample R2 (on {len(bins_df)} bin means) = {r2_bin:.4f}")
print(f"  Median count per bin = {int(np.median(cnt_bins))}")


# ══════════════════════════════════════════════════════════════════════════════
# (3) September holdout — 5 models with AAPL's best hyperparams
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*72}")
print("(3) SEPTEMBER HOLDOUT — AAPL hyperparams transferred to COIN")
print(f"{'='*72}")

FEATURES_3 = ["roll_spread_500", "roll_vol_500", "participation_rate"]

# Metric helpers
def r2_fn(ytrue, ypred):
    ss_res = ((ytrue - ypred)**2).sum()
    ss_tot = ((ytrue - ytrue.mean())**2).sum()
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

def mae_fn(ytrue, ypred):
    return np.mean(np.abs(ytrue - ypred))

# Feature arrays
spread_tr = df_tr["roll_spread_500"].to_numpy(dtype=np.float64)
spread_te = df_te["roll_spread_500"].to_numpy(dtype=np.float64)
y_tr_abs  = df_tr["abs_impact"].to_numpy(dtype=np.float64)
y_te_abs  = df_te["abs_impact"].to_numpy(dtype=np.float64)

Xlad_tr = np.column_stack([spread_tr, np.ones(len(df_tr))])
Xlad_te = np.column_stack([spread_te, np.ones(len(df_te))])

X3_tr = df_tr[FEATURES_3].to_numpy(dtype=np.float64)
X3_te = df_te[FEATURES_3].to_numpy(dtype=np.float64)

MODEL_NAMES = ["OLS-Uni", "LAD-Uni", "XGB-MSE", "XGB-MAE", "RF-MAE"]
model_results = {}

# ── (a) OLS-Uni ──────────────────────────────────────────────────────────────
print("\n--- OLS-Uni ---")
beta_ols, *_ = np.linalg.lstsq(Xlad_tr, y_tr_abs, rcond=None)
pred_ols = np.maximum(Xlad_te @ beta_ols, 0.0)
r2_ols_oos = r2_fn(y_te_abs, pred_ols)
mae_ols_oos = mae_fn(y_te_abs, pred_ols)
print(f"  c1={beta_ols[0]:+.5f}  c2={beta_ols[1]:+.5f}  OOS R2={r2_ols_oos:+.4f}  MAE={mae_ols_oos:.4f}")
model_results["OLS-Uni"] = {"r2": r2_ols_oos, "mae": mae_ols_oos, "pred": pred_ols}

# ── (b) LAD-Uni ──────────────────────────────────────────────────────────────
print("--- LAD-Uni ---")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    lad_res = QuantReg(y_tr_abs, Xlad_tr).fit(q=0.5, max_iter=2000, p_tol=1e-6)
beta_lad = lad_res.params
pred_lad = np.maximum(Xlad_te @ beta_lad, 0.0)
r2_lad_oos = r2_fn(y_te_abs, pred_lad)
mae_lad_oos = mae_fn(y_te_abs, pred_lad)
print(f"  c1={beta_lad[0]:+.5f}  c2={beta_lad[1]:+.5f}  OOS R2={r2_lad_oos:+.4f}  MAE={mae_lad_oos:.4f}")
model_results["LAD-Uni"] = {"r2": r2_lad_oos, "mae": mae_lad_oos, "pred": pred_lad}

# ── (c) XGB-MSE ──────────────────────────────────────────────────────────────
print("--- XGB-MSE (AAPL hyperparams) ---")
xgb_mse = XGBRegressor(
    objective="reg:squarederror",
    max_depth=3, n_estimators=50, learning_rate=0.1,
    min_child_weight=1, reg_alpha=10, reg_lambda=10,
    tree_method="hist", verbosity=0, random_state=42, n_jobs=1,
)
xgb_mse.fit(X3_tr, y_tr_abs)
pred_xgb_mse = np.maximum(xgb_mse.predict(X3_te), 0.0)
r2_xm = r2_fn(y_te_abs, pred_xgb_mse)
mae_xm = mae_fn(y_te_abs, pred_xgb_mse)
print(f"  OOS R2={r2_xm:+.4f}  MAE={mae_xm:.4f}")
model_results["XGB-MSE"] = {"r2": r2_xm, "mae": mae_xm, "pred": pred_xgb_mse}

# ── (d) XGB-MAE ──────────────────────────────────────────────────────────────
print("--- XGB-MAE (AAPL hyperparams) ---")
xgb_mae = XGBRegressor(
    objective="reg:absoluteerror",
    max_depth=3, n_estimators=50, learning_rate=0.1,
    min_child_weight=1, reg_alpha=10, reg_lambda=10,
    tree_method="hist", verbosity=0, random_state=42, n_jobs=1,
)
xgb_mae.fit(X3_tr, y_tr_abs)
pred_xgb_mae = np.maximum(xgb_mae.predict(X3_te), 0.0)
r2_xa = r2_fn(y_te_abs, pred_xgb_mae)
mae_xa = mae_fn(y_te_abs, pred_xgb_mae)
print(f"  OOS R2={r2_xa:+.4f}  MAE={mae_xa:.4f}")
model_results["XGB-MAE"] = {"r2": r2_xa, "mae": mae_xa, "pred": pred_xgb_mae}

# ── (e) RF-MAE ───────────────────────────────────────────────────────────────
print("--- RF-MAE (AAPL hyperparams) ---")
rf_mae = RandomForestRegressor(
    criterion="absolute_error",
    max_depth=None, max_features="sqrt", min_samples_leaf=50,
    n_estimators=200, random_state=42, n_jobs=-1,
)
rf_mae.fit(X3_tr, y_tr_abs)
pred_rf_mae = np.maximum(rf_mae.predict(X3_te), 0.0)
r2_rf = r2_fn(y_te_abs, pred_rf_mae)
mae_rf = mae_fn(y_te_abs, pred_rf_mae)
print(f"  OOS R2={r2_rf:+.4f}  MAE={mae_rf:.4f}")
print(f"  Feature importances: {dict(zip(FEATURES_3, rf_mae.feature_importances_.round(4)))}")
model_results["RF-MAE"] = {"r2": r2_rf, "mae": mae_rf, "pred": pred_rf_mae}

# ── Summary table ────────────────────────────────────────────────────────────
print(f"\n{'='*62}")
print(f"  {'Model':<12}  {'OOS R2':>9}  {'OOS MAE':>9}")
print(f"  {'-'*34}")
for nm in MODEL_NAMES:
    info = model_results[nm]
    print(f"  {nm:<12}  {info['r2']:>+9.4f}  {info['mae']:>9.4f}")
print(f"{'='*62}")


# ══════════════════════════════════════════════════════════════════════════════
# (4) AAPL vs COIN side-by-side comparison
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n\n{'='*80}")
print(f"{'AAPL vs COIN — FULL COMPARISON':^80}")
print(f"{'='*80}")

AAPL_RESULTS = {
    "OLS-Uni": {"r2": +0.016, "mae": 1.772},
    "LAD-Uni": {"r2": -0.068, "mae": 1.530},
    "XGB-MSE": {"r2": +0.082, "mae": 1.754},
    "XGB-MAE": {"r2": +0.051, "mae": 1.476},
    "RF-MAE":  {"r2": +0.073, "mae": 1.487},
}

print(f"\n  {'--- Distribution fit (train, signed impact) ---'}")
print(f"  {'Distribution':<14} {'COIN AIC':>14}")
print(f"  {'-'*30}")
for r in dist_results:
    print(f"  {r['model']:<14} {r['aic']:>14.1f}")
best_dist = min(dist_results, key=lambda r: r["aic"])
print(f"  Best: {best_dist['model']}")

print(f"\n  {'--- Binned regression (train, 20 bins) ---'}")
print(f"  {'Metric':<28} {'COIN':>14}")
print(f"  {'-'*44}")
print(f"  {'Slope (c1)':<28} {c1_bin:>+14.4f}")
print(f"  {'Intercept (c2)':<28} {c2_bin:>+14.4f}")
print(f"  {'In-sample R2':<28} {r2_bin:>14.4f}")

print(f"\n  {'--- September holdout (OOS) ---'}")
print(f"  {'Model':<12}  {'AAPL R2':>9}  {'AAPL MAE':>9}  {'COIN R2':>9}  {'COIN MAE':>9}")
print(f"  {'-'*54}")
for nm in MODEL_NAMES:
    ar, am = AAPL_RESULTS[nm]["r2"], AAPL_RESULTS[nm]["mae"]
    cr, cm = model_results[nm]["r2"], model_results[nm]["mae"]
    print(f"  {nm:<12}  {ar:>+9.4f}  {am:>9.4f}  {cr:>+9.4f}  {cm:>9.4f}")

# Dataset comparison
aapl_tr = pd.read_parquet("data/lit_buy_features_v2.parquet")
aapl_te = pd.read_parquet("data/lit_buy_features_v2_sep.parquet")

print(f"\n  {'--- Dataset size ---'}")
print(f"  {'Metric':<28} {'AAPL':>14} {'COIN':>14}")
print(f"  {'-'*58}")
print(f"  {'Train trades':<28} {len(aapl_tr):>14,} {len(df_tr):>14,}")
print(f"  {'Test trades':<28} {len(aapl_te):>14,} {len(df_te):>14,}")
print(f"  {'Train abs(impact) mean':<28} {aapl_tr['impact_vwap_bps'].abs().mean():>14.4f} {df_tr['abs_impact'].mean():>14.4f}")
print(f"  {'Test abs(impact) mean':<28} {aapl_te['impact_vwap_bps'].abs().mean():>14.4f} {df_te['abs_impact'].mean():>14.4f}")
print(f"  {'Train roll_spread mean':<28} {aapl_tr['roll_spread_500'].mean():>14.4f} {df_tr['roll_spread_500'].mean():>14.4f}")
print(f"  {'Test roll_spread mean':<28} {aapl_te['roll_spread_500'].mean():>14.4f} {df_te['roll_spread_500'].mean():>14.4f}")
print(f"{'='*80}")


# ══════════════════════════════════════════════════════════════════════════════
# (5) Bootstrap 95% CIs on September test trades
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n\n{'='*80}")
print(f"(5) BOOTSTRAP 95% CIs — 1,000 resamples of {len(df_te):,} test trades")
print(f"{'='*80}")

N_BOOT = 1000
rng = np.random.default_rng(42)
n_te = len(y_te_abs)
boot_idx = rng.integers(0, n_te, size=(N_BOOT, n_te))

boot_results = {}  # model -> {"mae_lo", "mae_hi", "mae_samples", "r2_lo", "r2_hi", "r2_samples"}

for nm in MODEL_NAMES:
    pred = model_results[nm]["pred"]
    mae_samples = np.empty(N_BOOT)
    r2_samples  = np.empty(N_BOOT)

    for b in range(N_BOOT):
        idx = boot_idx[b]
        y_b = y_te_abs[idx]
        p_b = pred[idx]
        mae_samples[b] = np.mean(np.abs(y_b - p_b))
        ss_res = ((y_b - p_b)**2).sum()
        ss_tot = ((y_b - y_b.mean())**2).sum()
        r2_samples[b] = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    mae_lo, mae_hi = np.percentile(mae_samples, [2.5, 97.5])
    r2_lo,  r2_hi  = np.percentile(r2_samples,  [2.5, 97.5])

    boot_results[nm] = {
        "mae_lo": mae_lo, "mae_hi": mae_hi, "mae_samples": mae_samples,
        "r2_lo": r2_lo, "r2_hi": r2_hi, "r2_samples": r2_samples,
    }

# Print bootstrap table
print(f"\n  {'Model':<12}  {'MAE':>7}  {'MAE 95% CI':>18}  {'CI width':>9}  "
      f"{'R2':>7}  {'R2 95% CI':>20}  {'CI width':>9}")
print(f"  {'-'*90}")
for nm in MODEL_NAMES:
    m  = model_results[nm]
    br = boot_results[nm]
    mae_w = br["mae_hi"] - br["mae_lo"]
    r2_w  = br["r2_hi"]  - br["r2_lo"]
    print(f"  {nm:<12}  {m['mae']:>7.4f}  [{br['mae_lo']:.4f}, {br['mae_hi']:.4f}]  "
          f"{mae_w:>9.4f}  {m['r2']:>+7.4f}  [{br['r2_lo']:+.4f}, {br['r2_hi']:+.4f}]  "
          f"{r2_w:>9.4f}")

# Pairwise overlap analysis
print(f"\n  {'--- Pairwise MAE CI overlap ---'}")
print(f"  Pairs where CIs overlap => difference NOT statistically significant at 95%")
print(f"  {'Model A':<12} {'Model B':<12} {'Overlap?':>10} {'A-B diff':>10}")
print(f"  {'-'*48}")
for i, nm_a in enumerate(MODEL_NAMES):
    for nm_b in MODEL_NAMES[i+1:]:
        a = boot_results[nm_a]
        b = boot_results[nm_b]
        overlap = a["mae_lo"] <= b["mae_hi"] and b["mae_lo"] <= a["mae_hi"]
        diff = model_results[nm_a]["mae"] - model_results[nm_b]["mae"]
        tag = "YES" if overlap else "NO"
        print(f"  {nm_a:<12} {nm_b:<12} {tag:>10} {diff:>+10.4f}")

# Paired bootstrap: fraction of times model A beats model B on MAE
print(f"\n  {'--- Paired bootstrap: P(A < B on MAE) ---'}")
print(f"  {'':>12}", end="")
for nm in MODEL_NAMES:
    print(f"  {nm:>10}", end="")
print()
for nm_a in MODEL_NAMES:
    print(f"  {nm_a:<12}", end="")
    for nm_b in MODEL_NAMES:
        if nm_a == nm_b:
            print(f"  {'---':>10}", end="")
        else:
            p = (boot_results[nm_a]["mae_samples"] < boot_results[nm_b]["mae_samples"]).mean()
            print(f"  {p:>10.3f}", end="")
    print()

print(f"\n{'='*80}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT — 3 panels
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(20, 6.5))
gs_fig = gridspec.GridSpec(1, 3, figure=fig, wspace=0.32)

# ── Panel 1: Distribution fit ────────────────────────────────────────────────
ax1 = fig.add_subplot(gs_fig[0, 0])

lo = max(np.percentile(x_dist, 0.5), -30.0)
hi = min(np.percentile(x_dist, 99.5), 40.0)
x_grid = np.linspace(lo, hi, 600)

ax1.hist(x_dist[(x_dist >= lo) & (x_dist <= hi)], bins=80, density=True,
         color="#cbd5e1", alpha=0.60, edgecolor="none", zorder=1)

DIST_STYLES = {
    "Normal":      dict(color="#2563eb", ls="-",  lw=2.0),
    "Laplace":     dict(color="#16a34a", ls="--", lw=2.0),
    "Skew-normal": dict(color="#f59e0b", ls="-.", lw=2.0),
    "RI(2022)":    dict(color="#dc2626", ls=":",  lw=2.4),
}

for r in dist_results:
    st = DIST_STYLES[r["model"]]
    ax1.plot(x_grid, r["pdf_fn"](x_grid),
             color=st["color"], ls=st["ls"], lw=st["lw"],
             label=f"{r['model']}  AIC={r['aic']:.0f}", zorder=3)

ax1.text(0.97, 0.97, f"Best AIC: {best_dist['model']}",
         transform=ax1.transAxes, fontsize=8, ha="right", va="top",
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#94a3b8", alpha=0.9))

ax1.set_xlim(lo, hi)
ax1.set_xlabel("impact_vwap_bps (signed)", fontsize=10)
ax1.set_ylabel("Density", fontsize=10)
ax1.set_title("Distribution fit on train impact (signed)\nNormal / Laplace / Skew-normal / RI(2022)",
              fontsize=10.5, fontweight="bold")
ax1.legend(fontsize=7.5, loc="upper right", framealpha=0.92)
ax1.grid(True, alpha=0.18)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# ── Panel 2: Binned regression ───────────────────────────────────────────────
ax2 = fig.add_subplot(gs_fig[0, 1])

sizes = 40 + 120 * (cnt_bins / cnt_bins.max())
ax2.scatter(s_bins, y_bins, s=sizes, color="#2563eb", alpha=0.85,
            edgecolors="white", linewidths=0.6, zorder=3)

s_line = np.linspace(s_bins.min(), s_bins.max(), 200)
y_line = c1_bin * s_line + c2_bin
ax2.plot(s_line, y_line, color="black", lw=1.8, zorder=4,
         label=f"OLS: {c1_bin:+.4f}x {c2_bin:+.4f}\n$R^2$ = {r2_bin:.4f}")

ax2.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.6)
ax2.set_xlabel("Mean roll_spread_500 per bin (bps)", fontsize=10)
ax2.set_ylabel("Mean abs(impact_vwap_bps) per bin", fontsize=10)
ax2.set_title(f"Binned regression: 20 bins by spread\n"
              f"OLS R2 = {r2_bin:.4f} (in-sample on bin means)",
              fontsize=10.5, fontweight="bold")
ax2.legend(fontsize=9, loc="upper left")
ax2.grid(True, alpha=0.2)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

# ── Panel 3: Model comparison bar chart with bootstrap CIs ───────────────────
ax3 = fig.add_subplot(gs_fig[0, 2])

COLORS = ["#2563eb", "#16a34a", "#7c3aed", "#dc2626", "#f59e0b"]
xpos = np.arange(len(MODEL_NAMES))
w = 0.35

aapl_maes = [AAPL_RESULTS[nm]["mae"] for nm in MODEL_NAMES]
coin_maes = [model_results[nm]["mae"] for nm in MODEL_NAMES]

# Bootstrap error bars for COIN (asymmetric: [point - lo, hi - point])
coin_errs_lo = [model_results[nm]["mae"] - boot_results[nm]["mae_lo"] for nm in MODEL_NAMES]
coin_errs_hi = [boot_results[nm]["mae_hi"] - model_results[nm]["mae"] for nm in MODEL_NAMES]

bars_a = ax3.bar(xpos - w/2, aapl_maes, width=w, color="#94a3b8",
                 edgecolor="white", linewidth=0.8, label="AAPL")
bars_c = ax3.bar(xpos + w/2, coin_maes, width=w, color=COLORS,
                 edgecolor="white", linewidth=0.8, label="COIN")
ax3.errorbar(xpos + w/2, coin_maes,
             yerr=[coin_errs_lo, coin_errs_hi],
             fmt="none", ecolor="black", elinewidth=1.2, capsize=4, capthick=1.2,
             zorder=5, label="95% CI (bootstrap)")

for bar, v in zip(bars_a, aapl_maes):
    ax3.text(bar.get_x() + bar.get_width()/2, v + 0.05,
             f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, color="#64748b")
for bar, v, br_nm in zip(bars_c, coin_maes, MODEL_NAMES):
    br = boot_results[br_nm]
    ax3.text(bar.get_x() + bar.get_width()/2, br["mae_hi"] + 0.05,
             f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

ax3.set_xticks(xpos)
ax3.set_xticklabels(MODEL_NAMES, fontsize=9)
ax3.set_ylabel("OOS MAE (bps)", fontsize=10)
ax3.set_title("September holdout: OOS MAE\nAAPL (gray) vs COIN (color) + 95% bootstrap CI",
              fontsize=10.5, fontweight="bold")
ax3.legend(fontsize=8)
ax3.grid(True, axis="y", alpha=0.2)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)

# ── Suptitle ─────────────────────────────────────────────────────────────────
best_coin_model = min(model_results, key=lambda k: model_results[k]["mae"])
best_coin_mae = model_results[best_coin_model]["mae"]
best_coin_r2  = model_results[best_coin_model]["r2"]

fig.suptitle(
    f"COIN lit buy blocks — full analysis  |  "
    f"Train: Jan-Aug ({len(df_tr):,} trades)  Test: Sep ({len(df_te):,} trades)\n"
    f"Best distribution: {best_dist['model']}  |  "
    f"Binned R2={r2_bin:.4f}  |  "
    f"Best OOS: {best_coin_model} MAE={best_coin_mae:.4f} R2={best_coin_r2:+.4f}",
    fontsize=12, fontweight="bold", y=1.04,
)

plt.savefig("coin_full_analysis.png", dpi=150, bbox_inches="tight")
print(f"\nSaved -> coin_full_analysis.png")
