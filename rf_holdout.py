"""
Random Forest on individual trades: abs(impact_vwap_bps) ~ spread + vol + prate.
Train: Jun-Aug 2024 (35,020 trades)  |  Test: Sep 2024 (9,152 trades)

Two RF variants
  RF-MSE  : criterion='squared_error'   -- full user grid (720 combos)
  RF-MAE  : criterion='absolute_error'  -- same grid minus max_depth=None (576 combos)
    Note: criterion='absolute_error' sorts targets at every split (O(n log n) per node
    vs O(n) for MSE). max_depth=None with n=500 takes ~35s/fit; excluded to keep
    runtime feasible. All other grid axes match the MSE grid exactly.

Inner CV: GridSearchCV with TimeSeriesSplit(n_splits=5), scoring=neg_mean_absolute_error.
Parallelism: n_jobs=-1 on GridSearchCV (loky backend, 12 CPU cores),
             n_jobs=1 per RF (avoids oversubscription with outer parallelism).

Grid (RF-MSE): n_estimators=[100,200,500], max_depth=[3,5,8,12,None],
               min_samples_leaf=[5,10,20,50], min_samples_split=[5,10,20],
               max_features=['sqrt','log2',0.5,1.0]  -> 720 combinations
Grid (RF-MAE): same, max_depth=[3,5,8,12]  -> 576 combinations

Baselines (from temporal_holdout.py, train Jun-Aug -> test Sep 2024)
  OLS-Uni  : R2=+0.016  MAE=1.772
  LAD-Uni  : R2=-0.068  MAE=1.530
  XGB-MSE  : R2=+0.082  MAE=1.754
  XGB-MAE  : R2=+0.051  MAE=1.476  <- previous best

Output: aapl_rf_holdout.png
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

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

# ── Load data ──────────────────────────────────────────────────────────────────
tr_df = pd.read_parquet("data/lit_buy_features_v2.parquet")
te_df = pd.read_parquet("data/lit_buy_features_v2_sep.parquet")

tr_df = tr_df.sort_values("date").reset_index(drop=True)
te_df = te_df.sort_values("date").reset_index(drop=True)

tr_df["abs_impact"] = tr_df["impact_vwap_bps"].abs()
te_df["abs_impact"] = te_df["impact_vwap_bps"].abs()

print(f"Train: {len(tr_df):,} trades across {tr_df['date'].nunique()} days")
print(f"Test : {len(te_df):,} trades across {te_df['date'].nunique()} days")

FEATURES = ["roll_spread_500", "roll_vol_500", "participation_rate"]

X_tr = tr_df[FEATURES].to_numpy(dtype=np.float64)
X_te = te_df[FEATURES].to_numpy(dtype=np.float64)
y_tr = tr_df["abs_impact"].to_numpy(dtype=np.float64)
y_te = te_df["abs_impact"].to_numpy(dtype=np.float64)

# ── Metrics ────────────────────────────────────────────────────────────────────
def r2(ytrue, ypred):
    ss_res = ((ytrue - ypred) ** 2).sum()
    ss_tot = ((ytrue - ytrue.mean()) ** 2).sum()
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

def mae(ytrue, ypred):
    return np.mean(np.abs(ytrue - ypred))

naive_mae = mae(y_te, np.full_like(y_te, y_te.mean()))
print(f"\nNaive-mean baseline MAE={naive_mae:.4f}")

# ── Grids ──────────────────────────────────────────────────────────────────────
PARAM_GRID_MSE = {
    "n_estimators":      [100, 200, 500],
    "max_depth":         [3, 5, 8, 12, None],
    "min_samples_leaf":  [5, 10, 20, 50],
    "min_samples_split": [5, 10, 20],
    "max_features":      ["sqrt", "log2", 0.5, 1.0],
}  # 3*5*4*3*4 = 720 combos

PARAM_GRID_MAE = {
    "n_estimators":      [100, 200, 500],
    "max_depth":         [3, 5, 8, 12],          # no None: too slow with absolute_error
    "min_samples_leaf":  [5, 10, 20, 50],
    "min_samples_split": [5, 10, 20],
    "max_features":      ["sqrt", "log2", 0.5, 1.0],
}  # 3*4*4*3*4 = 576 combos

inner_cv = TimeSeriesSplit(n_splits=5)

configs = [
    ("RF-MSE", "squared_error",  PARAM_GRID_MSE, 720),
    ("RF-MAE", "absolute_error", PARAM_GRID_MAE, 576),
]

results = {}

for label, criterion, param_grid, n_combos in configs:
    print(f"\n{'-'*60}")
    print(f"Fitting {label}  (criterion='{criterion}') ...")
    print(f"  Grid: {n_combos} combinations x 5 folds = {n_combos*5:,} fits")
    print(f"  Parallelism: n_jobs=-1 outer (12 cores), n_jobs=1 per RF")

    rf_base = RandomForestRegressor(
        criterion=criterion,
        random_state=42,
        n_jobs=1,      # single-threaded RF; outer GridSearchCV handles parallelism
    )

    gs = GridSearchCV(
        rf_base,
        param_grid,
        cv=inner_cv,
        scoring="neg_mean_absolute_error",
        refit=True,
        n_jobs=-1,     # loky backend: distribute (param, fold) pairs across 12 cores
        verbose=0,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gs.fit(X_tr, y_tr)

    best_rf   = gs.best_estimator_
    pred      = np.maximum(best_rf.predict(X_te), 0.0)
    oos_r2    = r2(y_te, pred)
    oos_mae   = mae(y_te, pred)

    print(f"  Best params : {gs.best_params_}")
    print(f"  CV MAE (best, inner): {-gs.best_score_:.4f}")
    print(f"  OOS R2  = {oos_r2:+.4f}")
    print(f"  OOS MAE = {oos_mae:.4f}")

    importances = best_rf.feature_importances_
    print("  Feature importances:")
    for fname, imp in zip(FEATURES, importances):
        print(f"    {fname:<22}: {imp:.4f}")

    results[label] = {
        "pred":        pred,
        "r2":          oos_r2,
        "mae":         oos_mae,
        "best_params": gs.best_params_,
        "cv_mae":      -gs.best_score_,
        "importances": importances,
    }

# ── Full comparison table ──────────────────────────────────────────────────────
BASELINES = {
    "OLS-Uni": {"r2": +0.016, "mae": 1.772},
    "LAD-Uni": {"r2": -0.068, "mae": 1.530},
    "XGB-MSE": {"r2": +0.082, "mae": 1.754},
    "XGB-MAE": {"r2": +0.051, "mae": 1.476},
}

print(f"\n{'='*62}")
print(f"{'Model':<12}  {'OOS R2':>9}  {'OOS MAE':>9}  {'vs XGB-MAE':>12}")
print(f"  {'-'*56}")
for nm, info in BASELINES.items():
    delta = info["mae"] - 1.476
    print(f"  {nm:<10}  {info['r2']:>+9.4f}  {info['mae']:>9.4f}  {delta:>+10.4f} bps")
for nm in ("RF-MSE", "RF-MAE"):
    info = results[nm]
    delta = info["mae"] - 1.476
    print(f"  {nm:<10}  {info['r2']:>+9.4f}  {info['mae']:>9.4f}  {delta:>+10.4f} bps")
print(f"{'='*62}")

# ── Plot ───────────────────────────────────────────────────────────────────────
ALL_MODELS = ["OLS-Uni", "LAD-Uni", "XGB-MSE", "XGB-MAE", "RF-MSE", "RF-MAE"]
ALL_COLORS = ["#94a3b8",  "#64748b",  "#7c3aed",  "#2563eb",  "#f59e0b",  "#dc2626"]
ALL_R2S    = [BASELINES["OLS-Uni"]["r2"], BASELINES["LAD-Uni"]["r2"],
              BASELINES["XGB-MSE"]["r2"], BASELINES["XGB-MAE"]["r2"],
              results["RF-MSE"]["r2"],    results["RF-MAE"]["r2"]]
ALL_MAES   = [BASELINES["OLS-Uni"]["mae"], BASELINES["LAD-Uni"]["mae"],
              BASELINES["XGB-MSE"]["mae"], BASELINES["XGB-MAE"]["mae"],
              results["RF-MSE"]["mae"],    results["RF-MAE"]["mae"]]

rf_mse_pred = results["RF-MSE"]["pred"]
rf_mae_pred = results["RF-MAE"]["pred"]

fig = plt.figure(figsize=(18, 11))
gs_fig = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
ax1 = fig.add_subplot(gs_fig[0, 0])
ax2 = fig.add_subplot(gs_fig[0, 1])
ax3 = fig.add_subplot(gs_fig[0, 2])
ax4 = fig.add_subplot(gs_fig[1, 0])
ax5 = fig.add_subplot(gs_fig[1, 1])
ax6 = fig.add_subplot(gs_fig[1, 2])

xpos = np.arange(len(ALL_MODELS))

# ── Panel 1: OOS R2 ───────────────────────────────────────────────────────────
bars1 = ax1.bar(xpos, ALL_R2S, color=ALL_COLORS, width=0.55,
                edgecolor="white", linewidth=0.8)
for bar, v in zip(bars1, ALL_R2S):
    ax1.text(bar.get_x() + bar.get_width() / 2,
             v + (0.003 if v >= 0 else -0.009),
             f"{v:+.4f}", ha="center", va="bottom" if v >= 0 else "top",
             fontsize=9, fontweight="bold")
ax1.axhline(0, color="black", lw=0.9, ls="--", alpha=0.5)
ax1.set_xticks(xpos)
ax1.set_xticklabels(ALL_MODELS, fontsize=9.5)
ax1.set_ylabel("OOS R2  (Sep 2024, individual trades)", fontsize=9.5)
ax1.set_title("OOS R2\n(train Jun-Aug -> test Sep)", fontsize=10.5, fontweight="bold")
ax1.grid(True, axis="y", alpha=0.2)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# ── Panel 2: OOS MAE ──────────────────────────────────────────────────────────
bars2 = ax2.bar(xpos, ALL_MAES, color=ALL_COLORS, width=0.55,
                edgecolor="white", linewidth=0.8)
for bar, v in zip(bars2, ALL_MAES):
    ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
             f"{v:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax2.axhline(naive_mae, color="gray", lw=1.2, ls=":", alpha=0.7,
            label=f"Naive mean ({naive_mae:.3f})")
ax2.axhline(1.476, color="#2563eb", lw=1.2, ls="--", alpha=0.7,
            label="XGB-MAE (1.476)")
ax2.set_xticks(xpos)
ax2.set_xticklabels(ALL_MODELS, fontsize=9.5)
ax2.set_ylabel("OOS MAE  (bps)", fontsize=9.5)
ax2.set_title("OOS MAE  (lower is better)\n(train Jun-Aug -> test Sep)", fontsize=10.5, fontweight="bold")
ax2.legend(fontsize=8.5)
ax2.grid(True, axis="y", alpha=0.2)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

# ── Panel 3: Feature importances ──────────────────────────────────────────────
feat_x     = np.arange(len(FEATURES))
feat_short = ["spread", "vol", "prate"]
w = 0.35

imp_mse = results["RF-MSE"]["importances"]
imp_mae = results["RF-MAE"]["importances"]

ax3.bar(feat_x - w/2, imp_mse, width=w, color="#f59e0b", label="RF-MSE", edgecolor="white")
ax3.bar(feat_x + w/2, imp_mae, width=w, color="#dc2626", label="RF-MAE", edgecolor="white")
for xi, (v1, v2) in enumerate(zip(imp_mse, imp_mae)):
    ax3.text(xi - w/2, v1 + 0.003, f"{v1:.3f}", ha="center", va="bottom", fontsize=8.5)
    ax3.text(xi + w/2, v2 + 0.003, f"{v2:.3f}", ha="center", va="bottom",
             fontsize=8.5, color="#dc2626")
ax3.set_xticks(feat_x)
ax3.set_xticklabels(feat_short, fontsize=10)
ax3.set_ylabel("Mean decrease in impurity", fontsize=9.5)
ax3.set_title("Feature importances\nRF-MSE vs RF-MAE (trained on Jun-Aug)", fontsize=10.5, fontweight="bold")
ax3.legend(fontsize=9.5)
ax3.set_ylim(0, max(imp_mse.max(), imp_mae.max()) * 1.25)
ax3.grid(True, axis="y", alpha=0.2)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)

# ── Panel 4: Scatter predicted vs actual (best-MAE RF) ────────────────────────
best_rf_label = "RF-MAE" if results["RF-MAE"]["mae"] <= results["RF-MSE"]["mae"] else "RF-MSE"
best_pred_arr = results[best_rf_label]["pred"]

rng  = np.random.default_rng(0)
samp = rng.choice(len(y_te), size=min(4000, len(y_te)), replace=False)
hi   = np.percentile(np.concatenate([best_pred_arr[samp], y_te[samp]]), 99)

ax4.scatter(best_pred_arr[samp], y_te[samp],
            alpha=0.12, s=5, color="#94a3b8", linewidths=0)
ax4.plot([0, hi], [0, hi], color="red", lw=1.5, ls="--", alpha=0.7, label="y = y-hat")
ax4.set_xlim(0, hi)
ax4.set_ylim(0, hi)
ax4.set_xlabel(f"Predicted |impact| -- {best_rf_label} (bps)", fontsize=9.5)
ax4.set_ylabel("Actual |impact| (bps)", fontsize=9.5)
ax4.set_title(f"Predicted vs Actual ({best_rf_label})\nSep 2024 test (4k sample)", fontsize=10.5, fontweight="bold")
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.18)
ax4.spines["top"].set_visible(False)
ax4.spines["right"].set_visible(False)

# ── Panel 5: Violin of absolute errors ────────────────────────────────────────
abs_errs = [np.abs(y_te - rf_mse_pred), np.abs(y_te - rf_mae_pred)]
vp = ax5.violinplot(abs_errs, positions=[0, 1], widths=0.5,
                    showmedians=True, showextrema=False)
for pc, color in zip(vp["bodies"], ["#f59e0b", "#dc2626"]):
    pc.set_facecolor(color)
    pc.set_alpha(0.55)
vp["cmedians"].set_color("black")
vp["cmedians"].set_linewidth(1.8)
clip = np.percentile(np.concatenate(abs_errs), 97)
ax5.axhline(1.476, color="#2563eb", lw=1.4, ls="--", alpha=0.8, label="XGB-MAE 1.476")
ax5.axhline(1.530, color="#64748b", lw=1.2, ls=":",  alpha=0.7, label="LAD-Uni 1.530")
ax5.set_xticks([0, 1])
ax5.set_xticklabels(["RF-MSE", "RF-MAE"], fontsize=10)
ax5.set_ylabel("|prediction error|  (bps)", fontsize=9.5)
ax5.set_title("Distribution of absolute errors\n(Sep 2024 test)", fontsize=10.5, fontweight="bold")
ax5.set_ylim(0, clip)
ax5.legend(fontsize=8.5)
ax5.grid(True, axis="y", alpha=0.2)
ax5.spines["top"].set_visible(False)
ax5.spines["right"].set_visible(False)

# ── Panel 6: Per-day MAE ──────────────────────────────────────────────────────
te_dates     = te_df["date"].to_numpy()
unique_dates = sorted(np.unique(te_dates))

day_mae_mse = []
day_mae_mae = []
for d in unique_dates:
    mask = te_dates == d
    day_mae_mse.append(mae(y_te[mask], rf_mse_pred[mask]))
    day_mae_mae.append(mae(y_te[mask], rf_mae_pred[mask]))

x_days     = np.arange(len(unique_dates))
day_labels = [str(d)[5:] for d in unique_dates]

ax6.plot(x_days, day_mae_mse, color="#f59e0b", lw=1.8, marker="o", ms=4, label="RF-MSE")
ax6.plot(x_days, day_mae_mae, color="#dc2626", lw=1.8, marker="s", ms=4, label="RF-MAE")
ax6.axhline(1.476, color="#2563eb", lw=1.2, ls="--", alpha=0.7, label="XGB-MAE 1.476")
ax6.axhline(1.530, color="#64748b", lw=1.0, ls=":",  alpha=0.6, label="LAD-Uni 1.530")
ax6.set_xticks(x_days[::2])
ax6.set_xticklabels(day_labels[::2], fontsize=7.5, rotation=35, ha="right")
ax6.set_ylabel("Daily MAE  (bps)", fontsize=9.5)
ax6.set_title("Per-day MAE: September 2024 holdout\n(RF-MSE vs RF-MAE)", fontsize=10.5, fontweight="bold")
ax6.legend(fontsize=8.5)
ax6.grid(True, alpha=0.18)
ax6.spines["top"].set_visible(False)
ax6.spines["right"].set_visible(False)

fig.suptitle(
    "AAPL lit buy block trades -- Random Forest  (criterion: squared_error vs absolute_error)\n"
    "Train: Jun-Aug 2024 (35,020 trades)  ->  Test: Sep 2024 (9,152 trades)  |  Individual-trade level",
    fontsize=12, fontweight="bold", y=1.01,
)

plt.savefig("aapl_rf_holdout.png", dpi=150, bbox_inches="tight")
print("\nsaved -> aapl_rf_holdout.png")
