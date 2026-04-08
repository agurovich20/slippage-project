"""
Four new semiparametric models for abs(impact_vwap_bps) on individual trades.
Train: Jun-Aug 2024 (35,020 trades)  |  Test: Sep 2024 (9,152 trades)

Models
------
  Baseline  : XGB-MAE   (reg:absoluteerror, 3 features) — 1.476 bps from temporal_holdout.py
  (1) GAM   : LinearGAM with s(spread)+s(vol)+s(prate), lambda grid search
  (2) GBR-Q : GradientBoostingRegressor(loss='quantile', alpha=0.5) + RandomizedSearchCV 100 iter
  (3) Eng-XGB: XGB-MAE + 4 interaction features (spread*prate^0.4, vol*prate^0.5, spread*vol, 1/prate)
  (4) PLQ   : Partially-linear quantile: LAD(spread) + GBR-Q(vol, prate on residuals)

Output: aapl_semiparametric_v2.png
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

from scipy.stats import randint, uniform

import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV

from pygam import LinearGAM, s as gam_s

from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

# ── Load data ──────────────────────────────────────────────────────────────────
tr_df = pd.read_parquet("data/lit_buy_features_v2.parquet")
te_df = pd.read_parquet("data/lit_buy_features_v2_sep.parquet")

tr_df["abs_impact"] = tr_df["impact_vwap_bps"].abs()
te_df["abs_impact"] = te_df["impact_vwap_bps"].abs()

print(f"Train: {len(tr_df):,} trades  |  Test: {len(te_df):,} trades")

FEATURES = ["roll_spread_500", "roll_vol_500", "participation_rate"]

X_tr = tr_df[FEATURES].to_numpy(dtype=np.float64)
X_te = te_df[FEATURES].to_numpy(dtype=np.float64)
y_tr = tr_df["abs_impact"].to_numpy(dtype=np.float64)
y_te = te_df["abs_impact"].to_numpy(dtype=np.float64)

spread_tr = X_tr[:, 0]
vol_tr    = X_tr[:, 1]
prate_tr  = X_tr[:, 2]

spread_te = X_te[:, 0]
vol_te    = X_te[:, 1]
prate_te  = X_te[:, 2]


# ── Metrics ────────────────────────────────────────────────────────────────────
def r2(ytrue, ypred):
    ss_res = ((ytrue - ypred) ** 2).sum()
    ss_tot = ((ytrue - ytrue.mean()) ** 2).sum()
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

def mae(ytrue, ypred):
    return np.mean(np.abs(ytrue - ypred))


# ── LAD helper (for PLQ) ───────────────────────────────────────────────────────
def fit_lad_1d(X_tr, y_tr):
    """Fit LAD and return params."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        qr = QuantReg(y_tr, X_tr)
        res = qr.fit(q=0.5, max_iter=2000, p_tol=1e-6)
    return res.params


# ── XGB-MAE BASELINE (refit on full train) ─────────────────────────────────────
print("\n[Baseline] XGB-MAE (3 features) ...")
XGB_PARAM_GRID = {
    "max_depth":        [2, 3],
    "n_estimators":     [50, 100, 200],
    "learning_rate":    [0.05, 0.1],
    "min_child_weight": [3, 5],
}
base_xgb = XGBRegressor(objective="reg:absoluteerror", tree_method="hist",
                         verbosity=0, random_state=42, n_jobs=1)
gs_base = GridSearchCV(base_xgb, XGB_PARAM_GRID, cv=3,
                       scoring="neg_mean_absolute_error", refit=True, n_jobs=1)
gs_base.fit(X_tr, y_tr)
pred_baseline = np.maximum(gs_base.predict(X_te), 0.0)
r2_baseline  = r2(y_te, pred_baseline)
mae_baseline = mae(y_te, pred_baseline)
print(f"  XGB-MAE baseline  R²={r2_baseline:+.4f}  MAE={mae_baseline:.4f}")
print(f"  Best params: {gs_base.best_params_}")


# ── Model 1: GAM with splines ──────────────────────────────────────────────────
print("\n[Model 1] GAM with splines ...")

# pygam gridsearch for smoothing lambda
lam_grid = np.logspace(-3, 3, 20)
gam = LinearGAM(gam_s(0) + gam_s(1) + gam_s(2))
gam.gridsearch(X_tr, y_tr, lam=lam_grid, objective="GCV", progress=False)
pred_gam = np.maximum(gam.predict(X_te), 0.0)
r2_gam  = r2(y_te, pred_gam)
mae_gam = mae(y_te, pred_gam)
print(f"  GAM  R²={r2_gam:+.4f}  MAE={mae_gam:.4f}")
print(f"  Best lam: {gam.lam}")


# ── Model 2: GBR-Quantile (RandomizedSearchCV 100 iter) ───────────────────────
print("\n[Model 2] GBR-Quantile (RandomizedSearchCV 100 iter) ...")

gbr_param_dist = {
    "n_estimators":      randint(50, 400),
    "max_depth":         randint(2, 6),
    "learning_rate":     uniform(0.02, 0.18),   # 0.02 .. 0.20
    "min_samples_leaf":  randint(5, 50),
    "subsample":         uniform(0.6, 0.4),     # 0.6 .. 1.0
    "max_features":      uniform(0.5, 0.5),     # 0.5 .. 1.0
}

gbr_base = GradientBoostingRegressor(loss="quantile", alpha=0.5, random_state=42)
rs_gbr = RandomizedSearchCV(
    gbr_base, gbr_param_dist,
    n_iter=100, cv=3,
    scoring="neg_mean_absolute_error",
    refit=True, random_state=42, n_jobs=1,
)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    rs_gbr.fit(X_tr, y_tr)

pred_gbr = np.maximum(rs_gbr.predict(X_te), 0.0)
r2_gbr  = r2(y_te, pred_gbr)
mae_gbr = mae(y_te, pred_gbr)
print(f"  GBR-Q  R²={r2_gbr:+.4f}  MAE={mae_gbr:.4f}")
print(f"  Best params: {rs_gbr.best_params_}")


# ── Model 3: Theory-engineered XGB-MAE ────────────────────────────────────────
print("\n[Model 3] Theory-engineered XGB-MAE ...")

def engineer_features(spread, vol, prate):
    """Base [spread, vol, prate] + 4 interactions."""
    inter1 = spread * (prate ** 0.4)          # Almgren-Chriss: sigma * nu^0.4
    inter2 = vol    * (prate ** 0.5)          # vol * sqrt(prate)
    inter3 = spread * vol                     # spread * vol
    inv_p  = np.where(prate > 0, 1.0 / prate, np.nan)  # 1/prate (liquidity)
    return np.column_stack([spread, vol, prate, inter1, inter2, inter3, inv_p])

X_tr_eng = engineer_features(spread_tr, vol_tr, prate_tr)
X_te_eng = engineer_features(spread_te, vol_te, prate_te)

# Drop rows with nan in engineered features (prate==0 gives 1/prate=nan)
valid_tr = np.isfinite(X_tr_eng).all(axis=1)
valid_te = np.isfinite(X_te_eng).all(axis=1)
print(f"  Valid train: {valid_tr.sum():,}  |  Valid test: {valid_te.sum():,}")

eng_xgb = XGBRegressor(objective="reg:absoluteerror", tree_method="hist",
                        verbosity=0, random_state=42, n_jobs=1)
gs_eng = GridSearchCV(eng_xgb, XGB_PARAM_GRID, cv=3,
                      scoring="neg_mean_absolute_error", refit=True, n_jobs=1)
gs_eng.fit(X_tr_eng[valid_tr], y_tr[valid_tr])

pred_eng = np.full(len(y_te), np.nan)
pred_eng[valid_te] = np.maximum(gs_eng.predict(X_te_eng[valid_te]), 0.0)
# For invalid test rows, fall back to baseline prediction
pred_eng[~valid_te] = pred_baseline[~valid_te]

r2_eng  = r2(y_te, pred_eng)
mae_eng = mae(y_te, pred_eng)
print(f"  Eng-XGB  R²={r2_eng:+.4f}  MAE={mae_eng:.4f}")
print(f"  Best params: {gs_eng.best_params_}")


# ── Model 4: Partially Linear Quantile (LAD + GBR-Q on residuals) ─────────────
print("\n[Model 4] Partially linear quantile: LAD(spread) + GBR-Q(vol, prate) ...")

# LAD on spread only
X_spread_tr = np.column_stack([spread_tr, np.ones(len(spread_tr))])
X_spread_te = np.column_stack([spread_te, np.ones(len(spread_te))])
beta_lad = fit_lad_1d(X_spread_tr, y_tr)
lad_pred_tr = X_spread_tr @ beta_lad
lad_pred_te = X_spread_te @ beta_lad
resid_tr = y_tr - lad_pred_tr

# GBR-Quantile on residuals using vol and prate
X_res_tr = np.column_stack([vol_tr, prate_tr])
X_res_te = np.column_stack([vol_te, prate_te])

plq_param_dist = {
    "n_estimators":     randint(50, 400),
    "max_depth":        randint(2, 5),
    "learning_rate":    uniform(0.02, 0.18),
    "min_samples_leaf": randint(5, 50),
    "subsample":        uniform(0.6, 0.4),
}
gbr_res_base = GradientBoostingRegressor(loss="quantile", alpha=0.5, random_state=42)
rs_res = RandomizedSearchCV(
    gbr_res_base, plq_param_dist,
    n_iter=100, cv=3,
    scoring="neg_mean_absolute_error",
    refit=True, random_state=42, n_jobs=1,
)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    rs_res.fit(X_res_tr, resid_tr)

resid_pred_te = rs_res.predict(X_res_te)
pred_plq = np.maximum(lad_pred_te + resid_pred_te, 0.0)
r2_plq  = r2(y_te, pred_plq)
mae_plq = mae(y_te, pred_plq)
print(f"  PLQ  R²={r2_plq:+.4f}  MAE={mae_plq:.4f}")
print(f"  Best params: {rs_res.best_params_}")


# ── Summary table ──────────────────────────────────────────────────────────────
MODEL_NAMES = ["XGB-MAE\n(baseline)", "GAM\n(splines)", "GBR-Q\n(quantile)", "Eng-XGB\n(+interact)", "PLQ\n(LAD+GBR-Q)"]
COLORS      = ["#94a3b8",             "#2563eb",         "#16a34a",            "#7c3aed",              "#dc2626"]
R2S  = [r2_baseline,  r2_gam,  r2_gbr,  r2_eng,  r2_plq]
MAES = [mae_baseline, mae_gam, mae_gbr, mae_eng, mae_plq]

print(f"\n{'='*60}")
print(f"{'Model':<18}  {'OOS R2':>9}  {'OOS MAE':>9}")
print(f"  {'-'*44}")
for nm, rv, mv in zip(MODEL_NAMES, R2S, MAES):
    nm_clean = nm.replace("\n", " ")
    print(f"  {nm_clean:<16}  {rv:>+9.4f}  {mv:>9.4f}")
print(f"{'='*60}")


# ── Plot ───────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 10))
gs_fig = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
ax1 = fig.add_subplot(gs_fig[0, 0])   # OOS R² bar
ax2 = fig.add_subplot(gs_fig[0, 1])   # OOS MAE bar
ax3 = fig.add_subplot(gs_fig[0, 2])   # GAM partial effects
ax4 = fig.add_subplot(gs_fig[1, 0])   # Scatter: predicted vs actual (best MAE model)
ax5 = fig.add_subplot(gs_fig[1, 1])   # Violin of absolute errors
ax6 = fig.add_subplot(gs_fig[1, 2])   # MAE improvement vs baseline bar chart

xpos = np.arange(len(MODEL_NAMES))

# ── Panel 1: OOS R² bars ──────────────────────────────────────────────────────
bars1 = ax1.bar(xpos, R2S, color=COLORS, width=0.55, edgecolor="white", linewidth=0.8)
for bar, v in zip(bars1, R2S):
    ax1.text(bar.get_x() + bar.get_width() / 2,
             v + (0.003 if v >= 0 else -0.009),
             f"{v:+.4f}", ha="center", va="bottom" if v >= 0 else "top",
             fontsize=8.5, fontweight="bold")
ax1.axhline(0, color="black", lw=0.9, ls="--", alpha=0.5)
ax1.set_xticks(xpos)
ax1.set_xticklabels(MODEL_NAMES, fontsize=8.5)
ax1.set_ylabel("OOS R²  (Sep 2024, individual trades)", fontsize=9.5)
ax1.set_title("OOS R²\n(train Jun-Aug → test Sep)", fontsize=10.5, fontweight="bold")
ax1.grid(True, axis="y", alpha=0.2)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# ── Panel 2: OOS MAE bars ─────────────────────────────────────────────────────
bars2 = ax2.bar(xpos, MAES, color=COLORS, width=0.55, edgecolor="white", linewidth=0.8)
for bar, v in zip(bars2, MAES):
    ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
             f"{v:.4f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
# Naive-mean baseline
naive_mae = mae(y_te, np.full_like(y_te, y_te.mean()))
ax2.axhline(naive_mae, color="gray", lw=1.2, ls=":", alpha=0.7, label=f"Naive mean ({naive_mae:.3f})")
ax2.set_xticks(xpos)
ax2.set_xticklabels(MODEL_NAMES, fontsize=8.5)
ax2.set_ylabel("OOS MAE  (bps, individual trades)", fontsize=9.5)
ax2.set_title("OOS MAE\n(lower is better)", fontsize=10.5, fontweight="bold")
ax2.legend(fontsize=8.5)
ax2.grid(True, axis="y", alpha=0.2)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

# ── Panel 3: GAM partial effects for each feature ─────────────────────────────
feature_names = ["roll_spread_500", "roll_vol_500", "participation_rate"]
feat_colors   = ["#2563eb", "#16a34a", "#7c3aed"]
feat_idx = [0, 1, 2]

for fi, (fname, fc) in enumerate(zip(feature_names, feat_colors)):
    XX = gam.generate_X_grid(term=fi)
    pdep, confi = gam.partial_dependence(term=fi, X=XX, width=0.95)
    ax3.plot(XX[:, fi], pdep, color=fc, lw=2.0, label=fname)
    ax3.fill_between(XX[:, fi], confi[:, 0], confi[:, 1],
                     color=fc, alpha=0.12)

ax3.set_xlabel("Feature value", fontsize=9.5)
ax3.set_ylabel("Partial effect on |impact| (bps)", fontsize=9.5)
ax3.set_title("GAM partial effects\n(95% CI, trained on Jun-Aug)", fontsize=10.5, fontweight="bold")
ax3.legend(fontsize=8.5)
ax3.grid(True, alpha=0.18)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)

# ── Panel 4: Predicted vs actual (best-MAE model) ─────────────────────────────
best_idx = int(np.argmin(MAES))
best_name = MODEL_NAMES[best_idx].replace("\n", " ")
best_pred = [pred_baseline, pred_gam, pred_gbr, pred_eng, pred_plq][best_idx]

rng  = np.random.default_rng(0)
samp = rng.choice(len(y_te), size=min(4000, len(y_te)), replace=False)
ax4.scatter(best_pred[samp], y_te[samp],
            alpha=0.12, s=5, color="#94a3b8", linewidths=0)
lo = min(best_pred[samp].min(), y_te[samp].min())
hi = np.percentile(np.concatenate([best_pred[samp], y_te[samp]]), 99)
ax4.plot([lo, hi], [lo, hi], color="red", lw=1.5, ls="--", alpha=0.7)
ax4.set_xlim(lo, hi)
ax4.set_ylim(0, hi)
ax4.set_xlabel(f"Predicted |impact| (bps) — {best_name}", fontsize=9.5)
ax4.set_ylabel("Actual |impact| (bps)", fontsize=9.5)
ax4.set_title(f"Predicted vs Actual ({best_name})\nSep 2024 test (sample of 4k)", fontsize=10.5, fontweight="bold")
ax4.grid(True, alpha=0.18)
ax4.spines["top"].set_visible(False)
ax4.spines["right"].set_visible(False)

# ── Panel 5: Violin of absolute errors ────────────────────────────────────────
preds_list = [pred_baseline, pred_gam, pred_gbr, pred_eng, pred_plq]
abs_errs = [np.abs(y_te - p) for p in preds_list]
vp = ax5.violinplot(abs_errs, positions=xpos, widths=0.5,
                    showmedians=True, showextrema=False)
for pc, color in zip(vp["bodies"], COLORS):
    pc.set_facecolor(color)
    pc.set_alpha(0.55)
vp["cmedians"].set_color("black")
vp["cmedians"].set_linewidth(1.8)
ax5.set_xticks(xpos)
ax5.set_xticklabels(MODEL_NAMES, fontsize=8.5)
ax5.set_ylabel("|prediction error|  (bps)", fontsize=9.5)
ax5.set_title("Distribution of absolute errors\n(Sep 2024 test)", fontsize=10.5, fontweight="bold")
ax5.set_ylim(0, np.percentile(np.concatenate(abs_errs), 98))
ax5.grid(True, axis="y", alpha=0.2)
ax5.spines["top"].set_visible(False)
ax5.spines["right"].set_visible(False)

# ── Panel 6: MAE improvement vs baseline ──────────────────────────────────────
deltas = [100 * (mae_baseline - m) / mae_baseline for m in MAES]  # % improvement (positive = better)
bar_colors = ["#16a34a" if d >= 0 else "#dc2626" for d in deltas]
bars6 = ax6.bar(xpos, deltas, color=bar_colors, width=0.55, edgecolor="white", linewidth=0.8)
for bar, v in zip(bars6, deltas):
    ax6.text(bar.get_x() + bar.get_width() / 2,
             v + (0.05 if v >= 0 else -0.15),
             f"{v:+.2f}%", ha="center", va="bottom" if v >= 0 else "top",
             fontsize=8.5, fontweight="bold")
ax6.axhline(0, color="black", lw=0.9, ls="--", alpha=0.5)
ax6.set_xticks(xpos)
ax6.set_xticklabels(MODEL_NAMES, fontsize=8.5)
ax6.set_ylabel("MAE improvement vs baseline  (%)", fontsize=9.5)
ax6.set_title("MAE improvement vs XGB-MAE baseline\n(positive = better than baseline)", fontsize=10.5, fontweight="bold")
ax6.grid(True, axis="y", alpha=0.2)
ax6.spines["top"].set_visible(False)
ax6.spines["right"].set_visible(False)

fig.suptitle(
    "AAPL lit buy block trades — Semiparametric Models v2\n"
    "Train: Jun-Aug 2024 (35,020 trades)  →  Test: Sep 2024 (9,152 trades)  |  Individual-trade level",
    fontsize=12, fontweight="bold", y=1.01,
)

plt.savefig("aapl_semiparametric_v2.png", dpi=150, bbox_inches="tight")
print("\nsaved -> aapl_semiparametric_v2.png")
