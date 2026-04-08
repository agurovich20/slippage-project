"""
SHAP, feature importance, and residual analytics for Random Forest LAD model.

Best params from LAD-scored grid search (6 features, 3-fold CV):
  max_depth=20, n_estimators=200, min_samples_leaf=20,
  max_features='sqrt', bootstrap=True, criterion='absolute_error'

Output: aapl_rf_lad_analysis.png
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

# -- Load data ----------------------------------------------------------------
df_tr = pd.read_parquet("data/lit_buy_features_v2.parquet")
df_te = pd.read_parquet("data/lit_buy_features_v2_sep.parquet")
df_tr["abs_impact"] = df_tr["impact_vwap_bps"].abs()
df_te["abs_impact"] = df_te["impact_vwap_bps"].abs()
df_tr = df_tr.sort_values("date").reset_index(drop=True)

FEATURES = [
    "dollar_value", "log_dollar_value", "participation_rate",
    "roll_spread_500", "roll_vol_500", "exchange_id",
]
FEAT_SHORT = ["dollar_val", "log_dollar", "prate", "spread", "vol", "exch_id"]

X_tr = df_tr[FEATURES].to_numpy(dtype=np.float64)
y_tr = df_tr["abs_impact"].to_numpy(dtype=np.float64)
X_te = df_te[FEATURES].to_numpy(dtype=np.float64)
y_te = df_te["abs_impact"].to_numpy(dtype=np.float64)

print(f"Train: {len(df_tr):,}  |  Test: {len(df_te):,}")

# -- Train model --------------------------------------------------------------
BEST = dict(max_depth=20, n_estimators=200, min_samples_leaf=20,
            max_features="sqrt", bootstrap=True)

model = RandomForestRegressor(
    criterion="absolute_error", random_state=42, n_jobs=-1, **BEST,
)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model.fit(X_tr, y_tr)

pred_tr = np.maximum(model.predict(X_tr), 0.0)
pred_te = np.maximum(model.predict(X_te), 0.0)

# -- Metrics ------------------------------------------------------------------
def r2(y, yh):
    ss = ((y - yh)**2).sum(); st = ((y - y.mean())**2).sum()
    return 1 - ss/st if st > 0 else np.nan

for label, y, pred in [("In-sample", y_tr, pred_tr), ("OOS (Sep)", y_te, pred_te)]:
    print(f"  {label}: R2={r2(y, pred):+.4f}  MAE={np.mean(np.abs(y - pred)):.4f}  "
          f"RMSE={np.sqrt(np.mean((y - pred)**2)):.4f}")

# -- SHAP on test set ---------------------------------------------------------
rng = np.random.default_rng(42)

n_bg = min(500, len(X_tr))
bg_idx = rng.choice(len(X_tr), size=n_bg, replace=False)

print("\nComputing SHAP values (TreeExplainer)...", flush=True)
explainer = shap.TreeExplainer(model, X_tr[bg_idx])

n_shap = min(3000, len(X_te))
shap_idx = rng.choice(len(X_te), size=n_shap, replace=False)
X_shap = X_te[shap_idx]

shap_vals = explainer.shap_values(X_shap, check_additivity=False)
base_val = explainer.expected_value

print(f"SHAP base value: {base_val:.4f}")
print("Mean |SHAP| per feature:")
mean_abs_shap = np.abs(shap_vals).mean(axis=0)
for feat, ms in sorted(zip(FEATURES, mean_abs_shap), key=lambda x: -x[1]):
    print(f"  {feat:<22} {ms:.5f}")

# -- RF native importances ----------------------------------------------------
imp = model.feature_importances_
print("\nRF Feature Importances (Mean Decrease in Impurity):")
for feat, val in sorted(zip(FEATURES, imp), key=lambda x: -x[1]):
    print(f"  {feat:<22} {val:.4f}")

# -- Permutation importance ----------------------------------------------------
print("\nComputing permutation importance on test set...", flush=True)
perm_result = permutation_importance(model, X_te, y_te, n_repeats=10,
                                      random_state=42, scoring="neg_mean_absolute_error",
                                      n_jobs=-1)
print("Permutation importance (decrease in neg_MAE):")
for feat, mean_imp, std_imp in sorted(
    zip(FEATURES, perm_result.importances_mean, perm_result.importances_std),
    key=lambda x: -x[1]
):
    print(f"  {feat:<22} {mean_imp:.5f} (+/- {std_imp:.5f})")

# -- PLOTS --------------------------------------------------------------------
fig = plt.figure(figsize=(26, 18))
gs = gridspec.GridSpec(3, 3, figure=fig, wspace=0.35, hspace=0.45)

# -- Panel 1: SHAP beeswarm ---------------------------------------------------
ax1 = fig.add_subplot(gs[0, :2])
cmap = cm.coolwarm

order = np.argsort(mean_abs_shap)
for plot_row, fi in enumerate(order):
    sv = shap_vals[:, fi]
    fv = X_shap[:, fi]
    fv_norm = (fv - fv.min()) / ((fv.max() - fv.min()) + 1e-12)
    jitter = rng.uniform(-0.3, 0.3, size=len(sv))
    ax1.scatter(sv, plot_row + jitter, c=cmap(fv_norm), s=12, alpha=0.5,
                edgecolors="none", rasterized=True)

ax1.set_yticks(range(len(order)))
ax1.set_yticklabels([FEATURES[i] for i in order], fontsize=9.5)
ax1.axvline(0, color="black", lw=1, ls="--", alpha=0.5)
ax1.set_xlabel("SHAP value (impact on prediction)", fontsize=10)
ax1.set_title("AAPL SHAP Beeswarm, Random Forest LAD (Sep 2024 test trades)\n"
              "color = feature value (blue=low, red=high)",
              fontsize=11, fontweight="bold")
ax1.grid(axis="x", alpha=0.2)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

sm_obj = cm.ScalarMappable(cmap=cmap, norm=Normalize(0, 1))
sm_obj.set_array([])
cbar = plt.colorbar(sm_obj, ax=ax1, fraction=0.02, pad=0.02)
cbar.set_label("Feature value (low -> high)", fontsize=8)
cbar.set_ticks([0, 0.5, 1])
cbar.set_ticklabels(["low", "mid", "high"], fontsize=7.5)

# -- Panel 2: Mean |SHAP| bar -------------------------------------------------
ax2 = fig.add_subplot(gs[0, 2])
bar_order = np.argsort(mean_abs_shap)[::-1]
bar_colors = plt.cm.Set2(np.linspace(0, 1, len(FEATURES)))

ax2.barh(range(len(FEATURES)), mean_abs_shap[bar_order],
         color=[bar_colors[i] for i in bar_order],
         edgecolor="white", linewidth=0.6, height=0.55)
ax2.set_yticks(range(len(FEATURES)))
ax2.set_yticklabels([FEATURES[i] for i in bar_order], fontsize=9.5)
for j, (val, fi) in enumerate(zip(mean_abs_shap[bar_order], bar_order)):
    ax2.text(val + 0.001, j, f"{val:.4f}", va="center", fontsize=8.5)
ax2.set_xlabel("Mean |SHAP value|", fontsize=10)
ax2.set_title("AAPL SHAP Feature Importance\n(mean |SHAP|)", fontsize=11, fontweight="bold")
ax2.grid(axis="x", alpha=0.2)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

# -- Panel 3-5: SHAP dependence for top 3 ------------------------------------
top3 = bar_order[:3]
for k, fi in enumerate(top3):
    ax = fig.add_subplot(gs[1, k])
    interact_fi = bar_order[1] if fi == bar_order[0] else bar_order[0]
    interact_vals = X_shap[:, interact_fi]
    iv_norm = (interact_vals - interact_vals.min()) / ((interact_vals.max() - interact_vals.min()) + 1e-12)

    ax.scatter(X_shap[:, fi], shap_vals[:, fi],
               c=cm.coolwarm(iv_norm), s=10, alpha=0.45,
               edgecolors="none", rasterized=True)
    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.4)
    ax.set_xlabel(FEATURES[fi], fontsize=10)
    ax.set_ylabel(f"SHAP value for {FEAT_SHORT[fi]}", fontsize=10)
    ax.set_title(f"AAPL SHAP Dependence: {FEATURES[fi]}\n(color = {FEATURES[interact_fi]})",
                 fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# -- Panel 6: Predicted vs Actual ---------------------------------------------
ax6 = fig.add_subplot(gs[2, 0])
clip_v = np.percentile(y_te, 98)
samp = rng.choice(len(y_te), size=min(5000, len(y_te)), replace=False)
ax6.scatter(y_te[samp], np.clip(pred_te[samp], 0, clip_v),
            alpha=0.08, s=8, color="#2563eb", linewidths=0, rasterized=True)
ax6.axline((0, 0), slope=1, color="black", lw=1.2, ls="--", alpha=0.6,
           label="Perfect prediction")
ax6.set_xlim(0, clip_v)
ax6.set_ylim(0, clip_v)
ax6.set_xlabel("Actual |impact_vwap_bps|", fontsize=10)
ax6.set_ylabel("Predicted", fontsize=10)
ax6.set_title(f"AAPL Predicted vs Actual (Sep 2024)\n"
              f"R2={r2(y_te, pred_te):+.4f}  RMSE={np.sqrt(np.mean((y_te - pred_te)**2)):.4f}",
              fontsize=10, fontweight="bold")
ax6.legend(fontsize=9)
ax6.grid(True, alpha=0.18)
ax6.spines["top"].set_visible(False)
ax6.spines["right"].set_visible(False)

# -- Panel 7: Residual distribution -------------------------------------------
ax7 = fig.add_subplot(gs[2, 1])
resid = y_te - pred_te
clip_r = np.percentile(np.abs(resid), 98)
ax7.hist(np.clip(resid, -clip_r, clip_r), bins=80, color="#7c3aed",
         alpha=0.7, edgecolor="white", linewidth=0.3, density=True)
ax7.axvline(0, color="black", lw=1.2, ls="--", alpha=0.6)
ax7.axvline(np.mean(resid), color="#dc2626", lw=1.5, ls="-",
            label=f"Mean residual: {np.mean(resid):.3f}")
ax7.axvline(np.median(resid), color="#16a34a", lw=1.5, ls="--",
            label=f"Median residual: {np.median(resid):.3f}")
ax7.set_xlabel("Residual (actual - predicted, bps)", fontsize=10)
ax7.set_ylabel("Density", fontsize=10)
ax7.set_title(f"AAPL Residual Distribution (Sep 2024)\n"
              f"std={np.std(resid):.3f}  skew={pd.Series(resid).skew():.2f}  "
              f"kurtosis={pd.Series(resid).kurtosis():.1f}",
              fontsize=10, fontweight="bold")
ax7.legend(fontsize=8.5)
ax7.grid(True, alpha=0.18)
ax7.spines["top"].set_visible(False)
ax7.spines["right"].set_visible(False)

# -- Panel 8: MAE by decile --------------------------------------------------
ax8 = fig.add_subplot(gs[2, 2])
decile_labels = pd.qcut(pred_te, q=10, labels=False, duplicates="drop")
n_deciles = len(np.unique(decile_labels))
dec_mae, dec_centers, dec_counts = [], [], []
for d in range(n_deciles):
    mask = decile_labels == d
    dec_mae.append(np.mean(np.abs(y_te[mask] - pred_te[mask])))
    dec_centers.append(np.mean(pred_te[mask]))
    dec_counts.append(mask.sum())

ax8.bar(range(n_deciles), dec_mae, color="#f59e0b",
        edgecolor="white", linewidth=0.6, width=0.7)
for i, (mae_v, cnt) in enumerate(zip(dec_mae, dec_counts)):
    ax8.text(i, mae_v + 0.02, f"{mae_v:.3f}\n(n={cnt})", ha="center",
             va="bottom", fontsize=7, fontweight="bold")
ax8.set_xticks(range(n_deciles))
ax8.set_xticklabels([f"{c:.2f}" for c in dec_centers], fontsize=7.5, rotation=30)
ax8.set_xlabel("Predicted value decile center (bps)", fontsize=10)
ax8.set_ylabel("MAE within decile (bps)", fontsize=10)
ax8.set_title("AAPL MAE by Prediction Decile\n(does the model fail more on large predictions?)",
              fontsize=10, fontweight="bold")
ax8.grid(axis="y", alpha=0.2)
ax8.spines["top"].set_visible(False)
ax8.spines["right"].set_visible(False)

fig.suptitle(
    "AAPL lit buy blocks  Random Forest LAD analytics  (6 features, depth=20)\n"
    f"Train: Jun-Aug ({len(df_tr):,})  Test: Sep ({len(df_te):,})  |  "
    f"OOS R2={r2(y_te, pred_te):+.4f}  MAE={np.mean(np.abs(y_te - pred_te)):.4f} bps",
    fontsize=13, fontweight="bold", y=1.01,
)

plt.savefig("aapl_rf_lad_analysis.png", dpi=150, bbox_inches="tight")
print("\nSaved -> aapl_rf_lad_analysis.png")
