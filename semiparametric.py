"""
Partially linear semiparametric model for abs(impact_vwap_bps).

Architecture
------------
  Parametric component  : LAD of abs(impact) on roll_spread_500
                          median = c1*spread + c2   [theory-motivated linear form]
  Nonparametric component: XGBoost on LAD residuals using roll_vol_500 & participation_rate
                          no functional form assumed

Pipeline per fold
-----------------
  1. Fit LAD on training trades: median(abs_impact) = c1*spread + c2
  2. Compute training residuals: resid = abs_impact - LAD_pred
  3. Fit XGBoost(resid ~ vol, prate) via GridSearchCV (3-fold inner CV)
  4. Final prediction = clip(LAD_pred + XGB_resid_pred, 0)

Baselines (same 5-fold CV)
--------------------------
  (A) Pure LAD univariate: median ~ c1*spread + c2
  (B) Pure XGBoost all features: [spread, vol, prate] -> abs_impact, GridSearchCV

GridSearchCV param grid
-----------------------
  max_depth:        [2, 3]
  n_estimators:     [50, 100, 200]
  learning_rate:    [0.05, 0.1]
  min_child_weight: [3, 5]
  scoring:          neg_mean_squared_error (inner CV)

Output: aapl_semiparametric.png
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

from statsmodels.regression.quantile_regression import QuantReg
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

# ── Load ───────────────────────────────────────────────────────────────────────
df = pd.read_parquet("data/lit_buy_features_v2.parquet")
df["abs_impact"] = df["impact_vwap_bps"].abs()
df = df.sort_values("date").reset_index(drop=True)
print(f"Loaded {len(df):,} trades across {df['date'].nunique()} dates")

spread = df["roll_spread_500"].to_numpy(dtype=np.float64)
vol    = df["roll_vol_500"].to_numpy(dtype=np.float64)
prate  = df["participation_rate"].to_numpy(dtype=np.float64)
y_all  = df["abs_impact"].to_numpy(dtype=np.float64)
dates  = df["date"].to_numpy()

# Feature matrices
X_lad_all  = np.column_stack([spread, np.ones(len(df))])      # LAD: spread + intercept
X_xgb_full = np.column_stack([spread, vol, prate])             # pure XGB: all 3 features
X_xgb_res  = np.column_stack([vol, prate])                     # residual XGB: vol & prate


# ── Helpers ────────────────────────────────────────────────────────────────────
def r2(ytrue, ypred):
    ss_res = ((ytrue - ypred) ** 2).sum()
    ss_tot = ((ytrue - ytrue.mean()) ** 2).sum()
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

def mae(ytrue, ypred):
    return np.mean(np.abs(ytrue - ypred))

def fit_lad(X_tr, y_tr):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = QuantReg(y_tr, X_tr).fit(q=0.5, max_iter=2000, p_tol=1e-6)
    return res.params                    # shape (p,)

PARAM_GRID = {
    "max_depth":        [2, 3],
    "n_estimators":     [50, 100, 200],
    "learning_rate":    [0.05, 0.1],
    "min_child_weight": [3, 5],
}

def fit_xgb_gs(X_tr, y_tr, inner_cv=3):
    base = XGBRegressor(
        objective="reg:absoluteerror",
        tree_method="hist", verbosity=0,
        random_state=42, n_jobs=1,
    )
    gs = GridSearchCV(
        base, PARAM_GRID, cv=inner_cv,
        scoring="neg_mean_absolute_error",
        refit=True, n_jobs=1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gs.fit(X_tr, y_tr)
    return gs.best_estimator_, gs.best_params_


# ── 5-fold walk-forward CV by date ─────────────────────────────────────────────
N_FOLDS = 5
unique_dates = np.array(sorted(df["date"].unique()))
n_days  = len(unique_dates)
d_fold  = np.digitize(np.arange(n_days),
                      bins=np.linspace(0, n_days, N_FOLDS + 1)[1:-1])
d2f     = dict(zip(unique_dates, d_fold))
row_fold = np.array([d2f[d] for d in dates])

splits = [(np.where(row_fold < k)[0], np.where(row_fold == k)[0])
          for k in range(1, N_FOLDS)]

print(f"\n5-fold walk-forward CV  ({len(splits)} OOS evaluations)\n")

MODEL_NAMES = ["LAD-Uni", "XGB-MAE", "Semipar-MAE"]
fold_r2  = {n: [] for n in MODEL_NAMES}
fold_mae = {n: [] for n in MODEL_NAMES}
COLORS   = {"LAD-Uni": "#2563eb", "XGB-MAE": "#7c3aed", "Semipar-MAE": "#16a34a"}
best_params_xgb_all = []
best_params_xgb_res = []

# Store OOS residual-model predictions for plot (pooled over all folds)
pool_true_resid = []
pool_pred_resid = []
pool_spread_te  = []
pool_lad_pred   = []
pool_y_te       = []
pool_semi_pred  = []
pool_xgb_pred   = []

for fi, (tr_idx, te_idx) in enumerate(splits):
    y_tr = y_all[tr_idx]
    y_te = y_all[te_idx]

    print(f"Fold {fi+1}  (train={len(tr_idx):,}  test={len(te_idx):,})")

    # ── (A) LAD univariate ────────────────────────────────────────────────────
    lad_params    = fit_lad(X_lad_all[tr_idx], y_tr)
    lad_pred_tr   = X_lad_all[tr_idx] @ lad_params
    lad_pred_te   = np.maximum(X_lad_all[te_idx] @ lad_params, 0.0)

    r2_lad  = r2(y_te, lad_pred_te)
    mae_lad = mae(y_te, lad_pred_te)
    fold_r2["LAD-Uni"].append(r2_lad)
    fold_mae["LAD-Uni"].append(mae_lad)
    print(f"  LAD-Uni :  R2={r2_lad:+.4f}  MAE={mae_lad:.4f}  "
          f"(c1={lad_params[0]:+.5f}  c2={lad_params[1]:+.5f})")

    # ── (B) Pure XGBoost on all 3 features ────────────────────────────────────
    xgb_full, p_full = fit_xgb_gs(X_xgb_full[tr_idx], y_tr)
    xgb_full_pred    = np.maximum(xgb_full.predict(X_xgb_full[te_idx]), 0.0)

    r2_xgb  = r2(y_te, xgb_full_pred)
    mae_xgb = mae(y_te, xgb_full_pred)
    fold_r2["XGB-MAE"].append(r2_xgb)
    fold_mae["XGB-MAE"].append(mae_xgb)
    best_params_xgb_all.append(p_full)
    print(f"  XGB-MAE :  R2={r2_xgb:+.4f}  MAE={mae_xgb:.4f}  {p_full}")

    # ── (C) Semiparametric hybrid ──────────────────────────────────────────────
    resid_tr      = y_tr - lad_pred_tr              # training residuals (may be < 0)
    xgb_res, p_res = fit_xgb_gs(X_xgb_res[tr_idx], resid_tr)

    xgb_res_pred  = xgb_res.predict(X_xgb_res[te_idx])
    semi_pred     = np.maximum(lad_pred_te + xgb_res_pred, 0.0)
    true_resid_te = y_te - lad_pred_te               # for plot

    r2_semi  = r2(y_te, semi_pred)
    mae_semi = mae(y_te, semi_pred)
    fold_r2["Semipar-MAE"].append(r2_semi)
    fold_mae["Semipar-MAE"].append(mae_semi)
    best_params_xgb_res.append(p_res)
    print(f"  Semipar :  R2={r2_semi:+.4f}  MAE={mae_semi:.4f}  {p_res}")

    # Pool for plots
    pool_true_resid.append(true_resid_te)
    pool_pred_resid.append(xgb_res_pred)
    pool_spread_te.append(spread[te_idx])
    pool_lad_pred.append(lad_pred_te)
    pool_y_te.append(y_te)
    pool_semi_pred.append(semi_pred)
    pool_xgb_pred.append(xgb_full_pred)

# ── Summary ────────────────────────────────────────────────────────────────────
print(f"\n{'='*64}")
print(f"{'Model':<22}  {'Mean R2':>9}  {'Std R2':>8}  {'Mean MAE':>10}  {'Std MAE':>8}")
print(f"  {'-'*58}")
for name in MODEL_NAMES:
    r2s  = np.array(fold_r2[name])
    maes = np.array(fold_mae[name])
    print(f"  {name:<20}  {r2s.mean():>+9.4f}  {r2s.std():>8.4f}  "
          f"{maes.mean():>10.4f}  {maes.std():>8.4f}")
print(f"{'='*64}")

# Improvements
mae_lad_mean  = np.mean(fold_mae["LAD-Uni"])
mae_xgb_mean  = np.mean(fold_mae["XGB-MAE"])
mae_semi_mean = np.mean(fold_mae["Semipar-MAE"])
print(f"\n  Semipar-MAE vs LAD-Uni : MAE {(mae_semi_mean-mae_lad_mean)/mae_lad_mean*100:+.2f}%")
print(f"  Semipar-MAE vs XGB-MAE : MAE {(mae_semi_mean-mae_xgb_mean)/mae_xgb_mean*100:+.2f}%")

# Pool arrays
true_resid_pool = np.concatenate(pool_true_resid)
pred_resid_pool = np.concatenate(pool_pred_resid)
y_pool          = np.concatenate(pool_y_te)
spread_pool     = np.concatenate(pool_spread_te)
lad_pool        = np.concatenate(pool_lad_pred)
semi_pool       = np.concatenate(pool_semi_pred)
xgb_pool        = np.concatenate(pool_xgb_pred)


# ── Figure ─────────────────────────────────────────────────────────────────────
fold_nums = np.arange(1, len(splits) + 1)

fig = plt.figure(figsize=(18, 11))
gs_fig = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.36)
ax1 = fig.add_subplot(gs_fig[0, 0])   # OOS R² bars
ax2 = fig.add_subplot(gs_fig[0, 1])   # OOS MAE bars
ax3 = fig.add_subplot(gs_fig[0, 2])   # per-fold R² lines
ax4 = fig.add_subplot(gs_fig[1, 0])   # true vs predicted residual scatter
ax5 = fig.add_subplot(gs_fig[1, 1])   # OOS error distributions
ax6 = fig.add_subplot(gs_fig[1, 2])   # MAE by spread quintile

xpos = np.arange(3)
labels_short = ["LAD-Uni", "XGB-MAE", "Semipar-MAE"]
colors_bar   = [COLORS[n] for n in labels_short]
mean_r2s  = [np.mean(fold_r2[n])  for n in MODEL_NAMES]
std_r2s   = [np.std(fold_r2[n])   for n in MODEL_NAMES]
mean_maes = [np.mean(fold_mae[n]) for n in MODEL_NAMES]
std_maes  = [np.std(fold_mae[n])  for n in MODEL_NAMES]

# ── Panel 1: OOS R² bars ──────────────────────────────────────────────────────
bars = ax1.bar(xpos, mean_r2s, yerr=std_r2s, color=colors_bar,
               capsize=6, width=0.52, edgecolor="white", linewidth=0.8)
for bar, v in zip(bars, mean_r2s):
    offset = 0.004 if v >= 0 else -0.009
    ax1.text(bar.get_x() + bar.get_width() / 2, v + offset,
             f"{v:+.4f}", ha="center", va="bottom" if v >= 0 else "top",
             fontsize=9.5, fontweight="bold")
ax1.axhline(0, color="black", lw=0.9, ls="--", alpha=0.5)
ax1.set_xticks(xpos); ax1.set_xticklabels(labels_short, fontsize=10)
ax1.set_ylabel("OOS R²  (individual trades)", fontsize=10)
ax1.set_title("OOS R²\n(mean ± std, 4 folds)", fontsize=11, fontweight="bold")
ax1.grid(True, axis="y", alpha=0.2)
ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

# ── Panel 2: OOS MAE bars ─────────────────────────────────────────────────────
bars2 = ax2.bar(xpos, mean_maes, yerr=std_maes, color=colors_bar,
                capsize=6, width=0.52, edgecolor="white", linewidth=0.8)
for bar, v in zip(bars2, mean_maes):
    ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
             f"{v:.4f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
ax2.set_xticks(xpos); ax2.set_xticklabels(labels_short, fontsize=10)
ax2.set_ylabel("OOS MAE  (bps)", fontsize=10)
ax2.set_title("OOS MAE\n(mean ± std, 4 folds)", fontsize=11, fontweight="bold")
ax2.grid(True, axis="y", alpha=0.2)
ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

# ── Panel 3: per-fold R² lines ────────────────────────────────────────────────
for name in MODEL_NAMES:
    r2s = fold_r2[name]
    ax3.plot(fold_nums, r2s, color=COLORS[name], lw=2.1, marker="o", ms=7,
             label=name)
    for fi, rv in enumerate(r2s):
        ax3.annotate(f"{rv:+.3f}", (fold_nums[fi], rv),
                     textcoords="offset points", xytext=(5, 3),
                     fontsize=7.5, color=COLORS[name])
ax3.axhline(0, color="black", lw=0.9, ls="--", alpha=0.5)
ax3.set_xticks(fold_nums)
ax3.set_xticklabels([f"Fold {k}" for k in fold_nums], fontsize=9.5)
ax3.set_ylabel("OOS R²", fontsize=10)
ax3.set_title("Per-fold OOS R²\n(walk-forward)", fontsize=11, fontweight="bold")
ax3.legend(fontsize=9.5)
ax3.grid(True, alpha=0.18)
ax3.spines["top"].set_visible(False); ax3.spines["right"].set_visible(False)

# ── Panel 4: True residual vs XGBoost predicted residual ─────────────────────
# Clip to [-10, 10] for readability
clip_r = 10.0
tr_c  = np.clip(true_resid_pool, -clip_r, clip_r)
pr_c  = np.clip(pred_resid_pool, -clip_r, clip_r)
rng   = np.random.default_rng(42)
samp  = rng.choice(len(tr_c), size=min(8000, len(tr_c)), replace=False)

ax4.scatter(tr_c[samp], pr_c[samp],
            alpha=0.07, s=5, color="#7c3aed", linewidths=0)
# LOWESS-style: just show binned mean
bins_r = pd.cut(tr_c[samp], bins=30)
binned_r = (pd.DataFrame({"tr": tr_c[samp], "pr": pr_c[samp], "b": bins_r})
            .groupby("b", observed=True)
            .agg(mid=("tr", "mean"), mn=("pr", "mean"))
            .reset_index(drop=True))
ax4.plot(binned_r["mid"], binned_r["mn"], color="#dc2626", lw=2.2, zorder=5,
         label="Binned mean of XGB prediction")
ax4.axline((0, 0), slope=1, color="black", lw=1.0, ls="--", alpha=0.5,
           label="Perfect prediction (y=x)")
ax4.axhline(0, color="#94a3b8", lw=0.8, ls=":")
ax4.axvline(0, color="#94a3b8", lw=0.8, ls=":")
ax4.set_xlim(-clip_r, clip_r); ax4.set_ylim(-clip_r, clip_r)
ax4.set_xlabel("True residual  (abs_impact - LAD_pred)  bps", fontsize=9.5)
ax4.set_ylabel("XGB predicted residual  (bps)", fontsize=9.5)
ax4.set_title("Nonparametric component quality\nTrue vs XGB-predicted LAD residuals",
              fontsize=11, fontweight="bold")
ax4.legend(fontsize=8.5)
ax4.grid(True, alpha=0.18)
ax4.spines["top"].set_visible(False); ax4.spines["right"].set_visible(False)

# ── Panel 5: OOS absolute error distributions (violin) ───────────────────────
abs_errors = {
    "LAD-Uni":    np.abs(y_pool - lad_pool),
    "XGB-MAE":    np.abs(y_pool - xgb_pool),
    "Semipar-MAE": np.abs(y_pool - semi_pool),
}
clip_e = np.percentile(np.abs(y_pool - lad_pool), 97)   # clip at 97th pct for display
vdata  = [np.clip(abs_errors[n], 0, clip_e) for n in MODEL_NAMES]

vp = ax5.violinplot(vdata, positions=xpos, widths=0.5,
                    showmedians=True, showextrema=False)
for i, (patch, color) in enumerate(zip(vp["bodies"], colors_bar)):
    patch.set_facecolor(color); patch.set_alpha(0.45)
vp["cmedians"].set_color("black"); vp["cmedians"].set_linewidth(1.8)

# Overlay median text
for i, name in enumerate(MODEL_NAMES):
    med = np.median(abs_errors[name])
    ax5.text(i, med + clip_e * 0.02, f"med={med:.3f}", ha="center",
             va="bottom", fontsize=8, fontweight="bold")

ax5.set_xticks(xpos); ax5.set_xticklabels(labels_short, fontsize=10)
ax5.set_ylabel(f"|error|  (bps, clipped at {clip_e:.1f})", fontsize=10)
ax5.set_title("OOS absolute error distribution\n(pooled across 4 folds)",
              fontsize=11, fontweight="bold")
ax5.grid(True, axis="y", alpha=0.2)
ax5.spines["top"].set_visible(False); ax5.spines["right"].set_visible(False)

# ── Panel 6: OOS MAE by spread quintile ──────────────────────────────────────
q5 = pd.qcut(spread_pool, q=5, labels=False, duplicates="drop")
q_mids = []
q_mae_lad  = []; q_mae_xgb  = []; q_mae_semi = []
for q in range(5):
    m = q5 == q
    q_mids.append(spread_pool[m].mean())
    q_mae_lad.append(mae(y_pool[m], lad_pool[m]))
    q_mae_xgb.append(mae(y_pool[m], xgb_pool[m]))
    q_mae_semi.append(mae(y_pool[m], semi_pool[m]))

w = 0.22
q_x = np.arange(5)
ax6.bar(q_x - w, q_mae_lad,  width=w, color=COLORS["LAD-Uni"],
        label="LAD-Uni",  edgecolor="white", linewidth=0.6)
ax6.bar(q_x,     q_mae_xgb,  width=w, color=COLORS["XGB-MAE"],
        label="XGB-MAE",     edgecolor="white", linewidth=0.6)
ax6.bar(q_x + w, q_mae_semi, width=w, color=COLORS["Semipar-MAE"],
        label="Semipar-MAE", edgecolor="white", linewidth=0.6)
ax6.set_xticks(q_x)
ax6.set_xticklabels([f"Q{i+1}\n{q_mids[i]:.2f}bps" for i in range(5)], fontsize=9)
ax6.set_xlabel("roll_spread_500 quintile  (pooled OOS trades)", fontsize=9.5)
ax6.set_ylabel("MAE  (bps)", fontsize=10)
ax6.set_title("OOS MAE by spread quintile\n(which model wins in tight vs wide spread?)",
              fontsize=11, fontweight="bold")
ax6.legend(fontsize=9)
ax6.grid(True, axis="y", alpha=0.2)
ax6.spines["top"].set_visible(False); ax6.spines["right"].set_visible(False)


fig.suptitle(
    "AAPL lit buy block trades — Partially linear semiparametric model  [both components: L1 loss]\n"
    "LAD(spread) parametric  +  XGBoost(reg:absoluteerror)(vol, prate) nonparametric residual\n"
    f"Individual trades ({len(df):,})  |  5-fold walk-forward CV  |  "
    f"Semipar-MAE={mae_semi_mean:.4f} bps",
    fontsize=11.5, fontweight="bold", y=1.02,
)

plt.savefig("aapl_semiparametric_mae.png", dpi=150, bbox_inches="tight")
print("\nsaved -> aapl_semiparametric_mae.png")
