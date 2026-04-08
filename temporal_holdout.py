"""
True temporal holdout evaluation.

Train: June–August 2024  (data/lit_buy_features_v2.parquet,  35,020 trades, 63 days)
Test : September 2024    (data/lit_buy_features_v2_sep.parquet, 9,152 trades, 20 days)

Models evaluated
----------------
  (1) OLS-Uni      : OLS of abs(impact) ~ spread + 1
  (2) LAD-Uni      : LAD of abs(impact) ~ spread + 1  (QuantReg tau=0.5)
  (3) XGB-MSE      : XGBoost (reg:squarederror)  ~ [spread, vol, prate]
  (4) XGB-MAE      : XGBoost (reg:absoluteerror) ~ [spread, vol, prate]
  (5) Semipar-MAE  : LAD(spread) + XGBoost-MAE(residuals ~ vol, prate)

All XGBoost models use 3-fold inner CV on the training data to tune:
  max_depth=[2,3], n_estimators=[50,100,200], learning_rate=[0.05,0.1],
  min_child_weight=[3,5].

OOS metrics (individual September trades): R² and MAE.

Output: aapl_temporal_holdout.png
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

# ── Load training and test sets ────────────────────────────────────────────────
df_tr = pd.read_parquet("data/lit_buy_features_v2.parquet")
df_te = pd.read_parquet("data/lit_buy_features_v2_sep.parquet")

df_tr["abs_impact"] = df_tr["impact_vwap_bps"].abs()
df_te["abs_impact"] = df_te["impact_vwap_bps"].abs()

print(f"Train: {len(df_tr):,} trades  |  {df_tr['date'].nunique()} days  "
      f"({df_tr['date'].min()} .. {df_tr['date'].max()})")
print(f"Test : {len(df_te):,} trades  |  {df_te['date'].nunique()} days  "
      f"({df_te['date'].min()} .. {df_te['date'].max()})")

# ── Feature arrays ─────────────────────────────────────────────────────────────
def make_arrays(df):
    spread = df["roll_spread_500"].to_numpy(dtype=np.float64)
    vol    = df["roll_vol_500"].to_numpy(dtype=np.float64)
    prate  = df["participation_rate"].to_numpy(dtype=np.float64)
    y      = df["abs_impact"].to_numpy(dtype=np.float64)
    X_lad  = np.column_stack([spread, np.ones(len(df))])          # [spread, 1]
    X_full = np.column_stack([spread, vol, prate])                 # all 3
    X_res  = np.column_stack([vol, prate])                         # vol & prate only
    return spread, vol, prate, y, X_lad, X_full, X_res

sp_tr, vo_tr, pr_tr, y_tr, Xlad_tr, Xfull_tr, Xres_tr = make_arrays(df_tr)
sp_te, vo_te, pr_te, y_te, Xlad_te, Xfull_te, Xres_te = make_arrays(df_te)

# ── Metric helpers ─────────────────────────────────────────────────────────────
def r2(ytrue, ypred):
    ss_res = ((ytrue - ypred) ** 2).sum()
    ss_tot = ((ytrue - ytrue.mean()) ** 2).sum()
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

def mae(ytrue, ypred):
    return np.mean(np.abs(ytrue - ypred))

# ── XGBoost grid ──────────────────────────────────────────────────────────────
PARAM_GRID = {
    "max_depth":        [2, 3],
    "n_estimators":     [50, 100, 200],
    "learning_rate":    [0.05, 0.1],
    "min_child_weight": [3, 5],
}

def fit_xgb_gs(X_tr, y_tr, objective, inner_cv=3):
    base = XGBRegressor(
        objective=objective,
        tree_method="hist", verbosity=0,
        random_state=42, n_jobs=1,
    )
    scoring = ("neg_mean_absolute_error" if "absolute" in objective
               else "neg_mean_squared_error")
    gs = GridSearchCV(base, PARAM_GRID, cv=inner_cv,
                      scoring=scoring, refit=True, n_jobs=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gs.fit(X_tr, y_tr)
    return gs.best_estimator_, gs.best_params_

# ── (1) OLS-Uni ───────────────────────────────────────────────────────────────
print("\n--- Fitting OLS-Uni ---")
beta_ols, *_ = np.linalg.lstsq(Xlad_tr, y_tr, rcond=None)
pred_ols     = np.maximum(Xlad_te @ beta_ols, 0.0)
r2_ols, mae_ols = r2(y_te, pred_ols), mae(y_te, pred_ols)
print(f"  c1={beta_ols[0]:+.5f}  c2={beta_ols[1]:+.5f}  "
      f"OOS R2={r2_ols:+.4f}  MAE={mae_ols:.4f}")

# ── (2) LAD-Uni ───────────────────────────────────────────────────────────────
print("--- Fitting LAD-Uni ---")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    lad_res  = QuantReg(y_tr, Xlad_tr).fit(q=0.5, max_iter=2000, p_tol=1e-6)
beta_lad     = lad_res.params
lad_pred_tr  = Xlad_tr @ beta_lad
pred_lad     = np.maximum(Xlad_te @ beta_lad, 0.0)
r2_lad, mae_lad = r2(y_te, pred_lad), mae(y_te, pred_lad)
print(f"  c1={beta_lad[0]:+.5f}  c2={beta_lad[1]:+.5f}  "
      f"OOS R2={r2_lad:+.4f}  MAE={mae_lad:.4f}")

# ── (3) XGB-MSE ───────────────────────────────────────────────────────────────
print("--- Fitting XGB-MSE (GridSearchCV 3-fold) ---")
xgb_mse, params_mse = fit_xgb_gs(Xfull_tr, y_tr, "reg:squarederror")
pred_mse = np.maximum(xgb_mse.predict(Xfull_te), 0.0)
r2_mse, mae_mse = r2(y_te, pred_mse), mae(y_te, pred_mse)
print(f"  best={params_mse}  OOS R2={r2_mse:+.4f}  MAE={mae_mse:.4f}")

# ── (4) XGB-MAE ───────────────────────────────────────────────────────────────
print("--- Fitting XGB-MAE (GridSearchCV 3-fold) ---")
xgb_mae, params_mae = fit_xgb_gs(Xfull_tr, y_tr, "reg:absoluteerror")
pred_mae = np.maximum(xgb_mae.predict(Xfull_te), 0.0)
r2_mae, mae_mae = r2(y_te, pred_mae), mae(y_te, pred_mae)
print(f"  best={params_mae}  OOS R2={r2_mae:+.4f}  MAE={mae_mae:.4f}")

# ── (5) Semipar-MAE ──────────────────────────────────────────────────────────
print("--- Fitting Semipar-MAE ---")
resid_tr = y_tr - lad_pred_tr
xgb_res, params_res = fit_xgb_gs(Xres_tr, resid_tr, "reg:absoluteerror")
xgb_res_pred_te = xgb_res.predict(Xres_te)
pred_semi = np.maximum(pred_lad + xgb_res_pred_te, 0.0)
r2_semi, mae_semi = r2(y_te, pred_semi), mae(y_te, pred_semi)
print(f"  best={params_res}  OOS R2={r2_semi:+.4f}  MAE={mae_semi:.4f}")

# ── Summary ───────────────────────────────────────────────────────────────────
MODEL_NAMES  = ["OLS-Uni", "LAD-Uni", "XGB-MSE", "XGB-MAE", "Semipar-MAE"]
OOS_R2       = [r2_ols, r2_lad, r2_mse, r2_mae, r2_semi]
OOS_MAE      = [mae_ols, mae_lad, mae_mse, mae_mae, mae_semi]
ALL_PREDS    = [pred_ols, pred_lad, pred_mse, pred_mae, pred_semi]

print(f"\n{'='*64}")
print(f"  TEMPORAL HOLDOUT: Train Jun-Aug 2024  ->  Test Sep 2024")
print(f"{'='*64}")
print(f"  {'Model':<22}  {'OOS R2':>9}  {'OOS MAE (bps)':>14}")
print(f"  {'-'*50}")
for name, rv, mv in zip(MODEL_NAMES, OOS_R2, OOS_MAE):
    print(f"  {name:<22}  {rv:>+9.4f}  {mv:>14.4f}")
print(f"{'='*64}")

# Reference: test set naive mean predictor
naive_mae  = mae(y_te, np.full(len(y_te), y_te.mean()))
naive_r2   = 0.0
print(f"\n  Naive mean baseline (predict train mean={y_tr.mean():.3f}): "
      f"MAE={mae(y_te, np.full(len(y_te), y_tr.mean())):.4f}")
print(f"  Test abs_impact: mean={y_te.mean():.4f}  std={y_te.std():.4f}  "
      f"median={np.median(y_te):.4f}")

# ── Plot ───────────────────────────────────────────────────────────────────────
COLORS = ["#2563eb", "#16a34a", "#7c3aed", "#dc2626", "#f59e0b"]

fig = plt.figure(figsize=(18, 10))
gs_fig = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.36)
ax1 = fig.add_subplot(gs_fig[0, 0])   # OOS R² bar
ax2 = fig.add_subplot(gs_fig[0, 1])   # OOS MAE bar
ax3 = fig.add_subplot(gs_fig[0, 2])   # feature distribution train vs test
ax4 = fig.add_subplot(gs_fig[1, 0])   # predictions vs actual scatter (best model)
ax5 = fig.add_subplot(gs_fig[1, 1])   # absolute error distribution violin
ax6 = fig.add_subplot(gs_fig[1, 2])   # MAE by September date

xpos = np.arange(5)

# ── Panel 1: OOS R² bars ──────────────────────────────────────────────────────
bars1 = ax1.bar(xpos, OOS_R2, color=COLORS, width=0.55,
                edgecolor="white", linewidth=0.8)
for bar, v in zip(bars1, OOS_R2):
    ax1.text(bar.get_x() + bar.get_width() / 2,
             v + (0.003 if v >= 0 else -0.006),
             f"{v:+.4f}", ha="center", va="bottom" if v >= 0 else "top",
             fontsize=8.5, fontweight="bold")
ax1.axhline(0, color="black", lw=0.9, ls="--", alpha=0.5)
ax1.set_xticks(xpos); ax1.set_xticklabels(MODEL_NAMES, fontsize=8.5, rotation=10)
ax1.set_ylabel("OOS R²  (Sep 2024)", fontsize=10)
ax1.set_title("OOS R²\n(Sep 2024 holdout)", fontsize=11, fontweight="bold")
ax1.grid(True, axis="y", alpha=0.2)
ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

# ── Panel 2: OOS MAE bars ─────────────────────────────────────────────────────
bars2 = ax2.bar(xpos, OOS_MAE, color=COLORS, width=0.55,
                edgecolor="white", linewidth=0.8)
for bar, v in zip(bars2, OOS_MAE):
    ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
             f"{v:.4f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
ax2.set_xticks(xpos); ax2.set_xticklabels(MODEL_NAMES, fontsize=8.5, rotation=10)
ax2.set_ylabel("OOS MAE  (bps)", fontsize=10)
ax2.set_title("OOS MAE\n(Sep 2024 holdout)", fontsize=11, fontweight="bold")
ax2.grid(True, axis="y", alpha=0.2)
ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

# ── Panel 3: spread distribution train vs test ────────────────────────────────
clip_sp = np.percentile(np.concatenate([sp_tr, sp_te]), 99)
ax3.hist(np.clip(sp_tr, 0, clip_sp), bins=60, density=True,
         color="#2563eb", alpha=0.5, label=f"Train Jun-Aug (n={len(sp_tr):,})")
ax3.hist(np.clip(sp_te, 0, clip_sp), bins=60, density=True,
         color="#dc2626", alpha=0.5, label=f"Test Sep (n={len(sp_te):,})")
ax3.set_xlabel("roll_spread_500 (bps)", fontsize=10)
ax3.set_ylabel("Density", fontsize=10)
ax3.set_title("Feature distribution: roll_spread_500\nTrain vs Test (regime check)",
              fontsize=11, fontweight="bold")
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.18)
ax3.spines["top"].set_visible(False); ax3.spines["right"].set_visible(False)

# ── Panel 4: Predicted vs actual for best MAE model ──────────────────────────
best_name = MODEL_NAMES[np.argmin(OOS_MAE)]
best_pred = ALL_PREDS[np.argmin(OOS_MAE)]
best_color = COLORS[np.argmin(OOS_MAE)]
clip_v = np.percentile(y_te, 98)
rng = np.random.default_rng(42)
samp = rng.choice(len(y_te), size=min(5000, len(y_te)), replace=False)
ax4.scatter(y_te[samp], np.clip(best_pred[samp], 0, clip_v),
            alpha=0.07, s=5, color=best_color, linewidths=0)
ax4.axline((0, 0), slope=1, color="black", lw=1.2, ls="--", alpha=0.6,
           label="Perfect prediction")
ax4.set_xlim(0, clip_v); ax4.set_ylim(0, clip_v)
ax4.set_xlabel("Actual |impact_vwap_bps|  (Sep 2024)", fontsize=10)
ax4.set_ylabel(f"Predicted  [{best_name}]", fontsize=10)
ax4.set_title(f"Predicted vs actual — {best_name}\n"
              f"R²={r2(y_te, best_pred):+.4f}  MAE={mae(y_te, best_pred):.4f} bps",
              fontsize=11, fontweight="bold")
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.18)
ax4.spines["top"].set_visible(False); ax4.spines["right"].set_visible(False)

# ── Panel 5: Absolute error distributions (violin) ────────────────────────────
abs_errs = [np.abs(y_te - p) for p in ALL_PREDS]
clip_e   = np.percentile(abs_errs[0], 97)
vdata    = [np.clip(e, 0, clip_e) for e in abs_errs]
vp = ax5.violinplot(vdata, positions=xpos, widths=0.5,
                    showmedians=True, showextrema=False)
for patch, color in zip(vp["bodies"], COLORS):
    patch.set_facecolor(color); patch.set_alpha(0.45)
vp["cmedians"].set_color("black"); vp["cmedians"].set_linewidth(1.8)
for i, (name, e) in enumerate(zip(MODEL_NAMES, abs_errs)):
    med = np.median(e)
    ax5.text(i, med + clip_e * 0.02, f"med={med:.3f}", ha="center",
             va="bottom", fontsize=7.5, fontweight="bold")
ax5.set_xticks(xpos); ax5.set_xticklabels(MODEL_NAMES, fontsize=8.5, rotation=10)
ax5.set_ylabel(f"|error|  (bps, clipped at {clip_e:.1f})", fontsize=10)
ax5.set_title("Absolute error distribution\n(Sep 2024 holdout, individual trades)",
              fontsize=11, fontweight="bold")
ax5.grid(True, axis="y", alpha=0.2)
ax5.spines["top"].set_visible(False); ax5.spines["right"].set_visible(False)

# ── Panel 6: MAE by September date ────────────────────────────────────────────
dates_te  = df_te["date"].to_numpy()
sep_dates = sorted(set(dates_te))
w = 0.14
d_pos = np.arange(len(sep_dates))

for ci, (name, preds, color) in enumerate(zip(MODEL_NAMES, ALL_PREDS, COLORS)):
    day_mae = [mae(y_te[dates_te == d], preds[dates_te == d]) for d in sep_dates]
    offset  = (ci - 2) * w
    ax6.bar(d_pos + offset, day_mae, width=w, color=color,
            alpha=0.8, label=name, edgecolor="white", linewidth=0.3)

ax6.set_xticks(d_pos)
ax6.set_xticklabels([d[5:] for d in sep_dates], fontsize=7, rotation=45)
ax6.set_ylabel("Daily MAE  (bps)", fontsize=10)
ax6.set_title("Per-day OOS MAE across September 2024\n(spike days reveal regime stress)",
              fontsize=11, fontweight="bold")
ax6.legend(fontsize=7.5, loc="upper right", ncol=2)
ax6.grid(True, axis="y", alpha=0.2)
ax6.spines["top"].set_visible(False); ax6.spines["right"].set_visible(False)

fig.suptitle(
    "AAPL lit buy block trades — True temporal holdout evaluation\n"
    f"Train: Jun–Aug 2024 ({len(df_tr):,} trades, 63 days)  "
    f"->  Test: Sep 2024 ({len(df_te):,} trades, 20 days)",
    fontsize=12, fontweight="bold", y=1.01,
)

plt.savefig("aapl_temporal_holdout.png", dpi=150, bbox_inches="tight")
print("\nsaved -> aapl_temporal_holdout.png")
