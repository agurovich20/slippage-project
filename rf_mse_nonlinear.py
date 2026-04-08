"""
Non-linearity analysis for Random Forest MSE model (6 features, depth=30).

Plots:
  Row 0: Partial dependence plots (PDP) for all 6 features
  Row 1: ICE plots (Individual Conditional Expectation) for top 3 features,
         SHAP interaction heatmap (2 cols)
  Row 2: Residual vs each feature (5 most important) + permutation importance bar

Output: aapl_rf_mse_nonlinear.png
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
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

# -- Load & train -------------------------------------------------------------
df_tr = pd.read_parquet("data/lit_buy_features_v2.parquet")
df_te = pd.read_parquet("data/lit_buy_features_v2_sep.parquet")
df_tr["abs_impact"] = df_tr["impact_vwap_bps"].abs()
df_te["abs_impact"] = df_te["impact_vwap_bps"].abs()
df_tr = df_tr.sort_values("date").reset_index(drop=True)

FEATURES = [
    "dollar_value", "log_dollar_value", "participation_rate",
    "roll_spread_500", "roll_vol_500", "exchange_id",
]

X_tr = df_tr[FEATURES].to_numpy(dtype=np.float64)
y_tr = df_tr["abs_impact"].to_numpy(dtype=np.float64)
X_te = df_te[FEATURES].to_numpy(dtype=np.float64)
y_te = df_te["abs_impact"].to_numpy(dtype=np.float64)

BEST = dict(max_depth=30, n_estimators=50, min_samples_leaf=20,
            max_features=0.33, bootstrap=False)

model = RandomForestRegressor(random_state=42, n_jobs=-1, **BEST)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model.fit(X_tr, y_tr)

pred_te = np.maximum(model.predict(X_te), 0.0)
resid = y_te - pred_te

# -- SHAP interaction values (on subsample) -----------------------------------
rng = np.random.default_rng(42)
n_shap = 800  # smaller for RF interaction (slow)
shap_idx = rng.choice(len(X_te), size=n_shap, replace=False)
X_shap = X_te[shap_idx]

n_bg = min(300, len(X_tr))
bg_idx = rng.choice(len(X_tr), size=n_bg, replace=False)

print("Computing SHAP interaction values (this may take a while)...", flush=True)
explainer = shap.TreeExplainer(model)
shap_interact = explainer.shap_interaction_values(X_shap)  # (n, 6, 6)
print(f"  shape: {shap_interact.shape}")

mean_interact = np.abs(shap_interact).mean(axis=0)

print("\nMean |SHAP interaction| matrix:")
header = "                     " + "  ".join(f"{f[:8]:>8}" for f in FEATURES)
print(header)
for i, feat in enumerate(FEATURES):
    row = "  ".join(f"{mean_interact[i, j]:>8.4f}" for j in range(len(FEATURES)))
    print(f"  {feat:<20} {row}")

# -- Permutation importance ---------------------------------------------------
print("\nComputing permutation importance...", flush=True)
perm_result = permutation_importance(model, X_te, y_te, n_repeats=10,
                                      random_state=42, scoring="neg_mean_squared_error",
                                      n_jobs=-1)

# -- Feature ranking by SHAP --------------------------------------------------
shap_main = np.abs(shap_interact).sum(axis=2).mean(axis=0)  # marginal SHAP importance
feat_rank = np.argsort(shap_main)[::-1]  # descending

# -- PDP helper ---------------------------------------------------------------
def compute_pdp(model, X_background, feat_idx, grid_n=200):
    feat_vals = X_background[:, feat_idx]
    grid = np.linspace(np.percentile(feat_vals, 1), np.percentile(feat_vals, 99), grid_n)
    pdp_vals = np.zeros(grid_n)
    bg_sub = rng.choice(len(X_background), size=min(2000, len(X_background)), replace=False)
    X_bg = X_background[bg_sub].copy()
    for gi, gval in enumerate(grid):
        X_mod = X_bg.copy()
        X_mod[:, feat_idx] = gval
        pdp_vals[gi] = np.maximum(model.predict(X_mod), 0.0).mean()
    return grid, pdp_vals

# -- ICE helper ---------------------------------------------------------------
def compute_ice(model, X_instances, feat_idx, grid_n=200):
    feat_vals = X_instances[:, feat_idx]
    grid = np.linspace(np.percentile(feat_vals, 1), np.percentile(feat_vals, 99), grid_n)
    ice = np.zeros((len(X_instances), grid_n))
    for gi, gval in enumerate(grid):
        X_mod = X_instances.copy()
        X_mod[:, feat_idx] = gval
        ice[:, gi] = np.maximum(model.predict(X_mod), 0.0)
    return grid, ice

# -- Non-linearity residual test ----------------------------------------------
from numpy.polynomial import polynomial as P

print("\nNon-linearity test: OLS residual vs feature, then fit quadratic")
print(f"  {'Feature':<22} {'Lin coef':>10} {'Quad coef':>10} {'Quad R2 gain':>13}")
for fi in range(len(FEATURES)):
    feat = X_te[:, fi]
    c_lin = np.polyfit(feat, resid, 1)
    pred_lin = np.polyval(c_lin, feat)
    ss_lin = ((resid - pred_lin)**2).sum()
    c_quad = np.polyfit(feat, resid, 2)
    pred_quad = np.polyval(c_quad, feat)
    ss_quad = ((resid - pred_quad)**2).sum()
    ss_tot = ((resid - resid.mean())**2).sum()
    r2_lin = 1 - ss_lin / ss_tot
    r2_quad = 1 - ss_quad / ss_tot
    print(f"  {FEATURES[fi]:<22} {c_lin[0]:>+10.5f} {c_quad[0]:>+10.5f} {r2_quad - r2_lin:>+13.6f}")

# -- PLOT ---------------------------------------------------------------------
fig = plt.figure(figsize=(28, 20))
gs_fig = gridspec.GridSpec(3, 6, figure=fig, wspace=0.38, hspace=0.45)

# -- Row 0: PDP for all 6 features -------------------------------------------
pdp_colors = ["#2563eb", "#16a34a", "#dc2626", "#7c3aed", "#f59e0b", "#06b6d4"]
for k in range(6):
    fi = k
    ax = fig.add_subplot(gs_fig[0, k])
    grid, pdp_vals = compute_pdp(model, X_tr, fi)

    samp = rng.choice(len(X_te), size=min(2000, len(X_te)), replace=False)
    ax.scatter(X_te[samp, fi], y_te[samp], alpha=0.04, s=4, color="#94a3b8",
              linewidths=0, rasterized=True, label="Test data")
    ax.plot(grid, pdp_vals, color=pdp_colors[k], lw=2.5, label="PDP", zorder=5)

    ax.set_xlabel(FEATURES[fi], fontsize=9)
    ax.set_ylabel("Predicted |impact| (bps)", fontsize=9)
    ax.set_title(f"PDP: {FEATURES[fi]}", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(True, alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(np.percentile(X_te[:, fi], 1), np.percentile(X_te[:, fi], 99))

# -- Row 1: ICE for top 3, SHAP interaction heatmap, perm importance ----------
top3_fi = feat_rank[:3]
ice_colors = ["#7c3aed", "#16a34a", "#2563eb"]

for k, fi in enumerate(top3_fi):
    ax = fig.add_subplot(gs_fig[1, k])
    ice_idx = rng.choice(len(X_te), size=100, replace=False)
    grid, ice = compute_ice(model, X_te[ice_idx], fi)

    for row in range(ice.shape[0]):
        ax.plot(grid, ice[row], color=ice_colors[k], alpha=0.08, lw=0.8)
    pdp_mean = ice.mean(axis=0)
    ax.plot(grid, pdp_mean, color="black", lw=2.5, label="PDP (mean)", zorder=5)

    ax.set_xlabel(FEATURES[fi], fontsize=9)
    ax.set_ylabel("Predicted |impact| (bps)", fontsize=9)
    ax.set_title(f"ICE: {FEATURES[fi]}\n(100 instances, black = PDP)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(grid[0], grid[-1])

# SHAP interaction heatmap (spans 2 cols)
ax_heat = fig.add_subplot(gs_fig[1, 3:5])
im = ax_heat.imshow(mean_interact, cmap="YlOrRd", aspect="auto")
ax_heat.set_xticks(range(len(FEATURES)))
ax_heat.set_xticklabels(FEATURES, fontsize=8, rotation=30, ha="right")
ax_heat.set_yticks(range(len(FEATURES)))
ax_heat.set_yticklabels(FEATURES, fontsize=8)

for i in range(len(FEATURES)):
    for j in range(len(FEATURES)):
        val = mean_interact[i, j]
        color = "white" if val > mean_interact.max() * 0.6 else "black"
        ax_heat.text(j, i, f"{val:.4f}", ha="center", va="center",
                     fontsize=7.5, fontweight="bold", color=color)

cb = fig.colorbar(im, ax=ax_heat, shrink=0.8, pad=0.03)
cb.set_label("Mean |SHAP interaction|", fontsize=9)
ax_heat.set_title("SHAP interaction matrix\n(off-diagonal = pairwise interactions)",
                  fontsize=10, fontweight="bold")

# Permutation importance bar
ax_perm = fig.add_subplot(gs_fig[1, 5])
perm_order = np.argsort(perm_result.importances_mean)[::-1]
perm_colors = plt.cm.Set2(np.linspace(0, 1, len(FEATURES)))
ax_perm.barh(range(len(FEATURES)),
             perm_result.importances_mean[perm_order[::-1]],
             xerr=perm_result.importances_std[perm_order[::-1]],
             color=[perm_colors[i] for i in perm_order[::-1]],
             edgecolor="white", linewidth=0.6, height=0.55, capsize=3)
ax_perm.set_yticks(range(len(FEATURES)))
ax_perm.set_yticklabels([FEATURES[i] for i in perm_order[::-1]], fontsize=9)
ax_perm.set_xlabel("Permutation importance\n(decrease in neg MSE)", fontsize=9)
ax_perm.set_title("AAPL Permutation Importance\n(test set, 10 repeats)",
                  fontsize=10, fontweight="bold")
ax_perm.grid(axis="x", alpha=0.2)
ax_perm.spines["top"].set_visible(False)
ax_perm.spines["right"].set_visible(False)

# -- Row 2: Residual vs each feature -----------------------------------------
for k in range(6):
    fi = k
    ax = fig.add_subplot(gs_fig[2, k])
    samp = rng.choice(len(X_te), size=min(3000, len(X_te)), replace=False)
    feat_samp = X_te[samp, fi]
    resid_samp = resid[samp]

    ax.scatter(feat_samp, resid_samp, alpha=0.06, s=6, color="#64748b",
              linewidths=0, rasterized=True)
    ax.axhline(0, color="black", lw=1, ls="--", alpha=0.5)

    # Binned mean residual
    n_bins = 30
    bins = np.linspace(np.percentile(feat_samp, 2), np.percentile(feat_samp, 98), n_bins + 1)
    bin_centers = []
    bin_means = []
    for bi in range(n_bins):
        mask = (feat_samp >= bins[bi]) & (feat_samp < bins[bi + 1])
        if mask.sum() > 10:
            bin_centers.append((bins[bi] + bins[bi + 1]) / 2)
            bin_means.append(resid_samp[mask].mean())
    ax.plot(bin_centers, bin_means, color="#dc2626", lw=2.2, marker="o",
            markersize=3, label="Binned mean residual", zorder=5)

    # Quadratic fit overlay
    c = np.polyfit(feat_samp, resid_samp, 2)
    x_fit = np.linspace(bins[0], bins[-1], 100)
    ax.plot(x_fit, np.polyval(c, x_fit), color="#f59e0b", lw=1.8, ls="--",
            label=f"Quadratic fit (c2={c[0]:+.4f})", zorder=4)

    ax.set_xlabel(FEATURES[fi], fontsize=9)
    ax.set_ylabel("Residual (bps)", fontsize=9)
    ax.set_title(f"Residual vs {FEATURES[fi]}", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(bins[0], bins[-1])
    clip_r = np.percentile(np.abs(resid_samp), 97)
    ax.set_ylim(-clip_r, clip_r)

fig.suptitle(
    "AAPL Random Forest MSE  Non-linearity analysis\n"
    "PDPs, ICE curves, SHAP interactions, residual patterns, permutation importance  |  "
    f"depth=30, 50 trees, 6 features",
    fontsize=14, fontweight="bold", y=1.01,
)

plt.savefig("aapl_rf_mse_nonlinear.png", dpi=150, bbox_inches="tight")
print("\nSaved -> aapl_rf_mse_nonlinear.png")
