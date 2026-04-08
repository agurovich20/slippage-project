"""
Random Forest GridSearchCV on June-Aug 2024 training data, 6 FEATURES, MSE scoring.

Grid: 5x4x4x4x2 = 640 combinations x 3-fold TimeSeriesSplit = 1,920 fits.

Parameters:
  max_depth       : [5, 10, 20, 30, 40]
  n_estimators    : [50, 100, 200, 400]
  min_samples_leaf: [1, 5, 10, 20]
  max_features    : ['sqrt', 0.33, 0.5, 1.0]
  bootstrap       : [True, False]

Features (6):
  dollar_value, log_dollar_value, participation_rate, roll_spread_500,
  roll_vol_500, exchange_id

Scoring: neg_mean_squared_error

Output:
  - data/gridsearch_rf_mse.csv
  - aapl_gridsearch_rf_mse.png
  - Console: in-sample / OOS metrics for AAPL and COIN
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import warnings
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

# -- helpers ------------------------------------------------------------------
def r2(ytrue, ypred):
    ss_res = ((ytrue - ypred) ** 2).sum()
    ss_tot = ((ytrue - ytrue.mean()) ** 2).sum()
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

def mae_fn(ytrue, ypred):
    return np.mean(np.abs(ytrue - ypred))

def rmse_fn(ytrue, ypred):
    return np.sqrt(np.mean((ytrue - ypred) ** 2))

# -- Load AAPL data ----------------------------------------------------------
df_tr = pd.read_parquet("data/lit_buy_features_v2.parquet")
df_te = pd.read_parquet("data/lit_buy_features_v2_sep.parquet")
df_tr["abs_impact"] = df_tr["impact_vwap_bps"].abs()
df_te["abs_impact"] = df_te["impact_vwap_bps"].abs()
df_tr = df_tr.sort_values("date").reset_index(drop=True)

print(f"AAPL Train (CV only): {len(df_tr):,} trades  |  {df_tr['date'].nunique()} days  "
      f"({df_tr['date'].min()} .. {df_tr['date'].max()})")
print(f"AAPL Holdout (Sep):   {len(df_te):,} trades  |  {df_te['date'].nunique()} days  "
      f"({df_te['date'].min()} .. {df_te['date'].max()})")

FEATURES = [
    "dollar_value", "log_dollar_value", "participation_rate",
    "roll_spread_500", "roll_vol_500", "exchange_id",
]
X_tr = df_tr[FEATURES].to_numpy(dtype=np.float64)
y_tr = df_tr["abs_impact"].to_numpy(dtype=np.float64)
X_te = df_te[FEATURES].to_numpy(dtype=np.float64)
y_te = df_te["abs_impact"].to_numpy(dtype=np.float64)

print(f"Features ({len(FEATURES)}): {FEATURES}")
print(f"Train shape: {X_tr.shape}  |  Target mean: {y_tr.mean():.4f} bps")

# -- Grid definition ----------------------------------------------------------
PARAM_GRID = {
    "max_depth":        [5, 10, 20, 30, 40],
    "n_estimators":     [50, 100, 200, 400],
    "min_samples_leaf": [1, 5, 10, 20],
    "max_features":     ["sqrt", 0.33, 0.5, 1.0],
    "bootstrap":        [True, False],
}

n_combos = 1
for v in PARAM_GRID.values():
    n_combos *= len(v)
print(f"\nGrid: {n_combos:,} combinations x 3 folds = {n_combos * 3:,} fits")
print(f"Scoring: neg_mean_squared_error")

# -- GridSearchCV -------------------------------------------------------------
base = RandomForestRegressor(random_state=42, n_jobs=1)
tscv = TimeSeriesSplit(n_splits=3)

gs = GridSearchCV(
    base, PARAM_GRID,
    cv=tscv,
    scoring="neg_mean_squared_error",
    refit=True,
    n_jobs=-1,
    verbose=1,
)

print("\nStarting GridSearchCV (on Jun-Aug AAPL data only)...", flush=True)
t0 = time.time()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    gs.fit(X_tr, y_tr)
elapsed = time.time() - t0
print(f"Done in {elapsed:.1f}s.\n")

# -- Results table ------------------------------------------------------------
res = pd.DataFrame(gs.cv_results_)
results = pd.DataFrame({
    "rank":             res["rank_test_score"],
    "max_depth":        res["param_max_depth"],
    "n_estimators":     res["param_n_estimators"],
    "min_samples_leaf": res["param_min_samples_leaf"],
    "max_features":     res["param_max_features"],
    "bootstrap":        res["param_bootstrap"],
    "mean_MSE":         -res["mean_test_score"],
    "std_MSE":          res["std_test_score"],
})
results = results.sort_values("rank").reset_index(drop=True)

results.to_csv("data/gridsearch_rf_mse.csv", index=False)
print(f"Saved full table -> data/gridsearch_rf_mse.csv  ({len(results):,} rows)\n")

pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 140)
pd.set_option("display.float_format", "{:.6f}".format)

print("=" * 120)
print("  TOP 20 CONFIGURATIONS (lowest mean inner-CV MSE)")
print("=" * 120)
print(results.head(20).to_string(index=False))

print(f"\n{'=' * 120}")
print("  BOTTOM 5 CONFIGURATIONS (highest mean inner-CV MSE)")
print("=" * 120)
print(results.tail(5).to_string(index=False))

best = gs.best_params_
best_rmse = np.sqrt(-gs.best_score_)
print(f"\n  Best: {best}")
print(f"  Best mean MSE: {-gs.best_score_:.6f}  (RMSE: {best_rmse:.6f} bps)")

# -- Evaluate best RF on AAPL train + Sep holdout ----------------------------
best_model = gs.best_estimator_
pred_tr_aapl = np.maximum(best_model.predict(X_tr), 0.0)
pred_te_aapl = np.maximum(best_model.predict(X_te), 0.0)

print(f"\n{'=' * 120}")
print("  AAPL RESULTS (best RF params)")
print("=" * 120)
print(f"  In-sample:  R2={r2(y_tr, pred_tr_aapl):+.4f}  MAE={mae_fn(y_tr, pred_tr_aapl):.4f}  RMSE={rmse_fn(y_tr, pred_tr_aapl):.4f} bps")
print(f"  OOS (Sep):  R2={r2(y_te, pred_te_aapl):+.4f}  MAE={mae_fn(y_te, pred_te_aapl):.4f}  RMSE={rmse_fn(y_te, pred_te_aapl):.4f} bps")

imp = best_model.feature_importances_
print(f"\n  Feature importances (RF, best params):")
for feat, val in sorted(zip(FEATURES, imp), key=lambda x: -x[1]):
    bar = "#" * int(val * 100)
    print(f"    {feat:<22} {val:.4f}  {bar}")

# -- Evaluate on COIN --------------------------------------------------------
df_tr_coin = pd.read_parquet("data/coin_lit_buy_features_train.parquet")
df_te_coin = pd.read_parquet("data/coin_lit_buy_features_test.parquet")
df_tr_coin["abs_impact"] = df_tr_coin["impact_vwap_bps"].abs()
df_te_coin["abs_impact"] = df_te_coin["impact_vwap_bps"].abs()
df_tr_coin = df_tr_coin.sort_values("date").reset_index(drop=True)

X_tr_coin = df_tr_coin[FEATURES].to_numpy(dtype=np.float64)
y_tr_coin = df_tr_coin["abs_impact"].to_numpy(dtype=np.float64)
X_te_coin = df_te_coin[FEATURES].to_numpy(dtype=np.float64)
y_te_coin = df_te_coin["abs_impact"].to_numpy(dtype=np.float64)

# Retrain RF with same best params on COIN training data
rf_coin = RandomForestRegressor(random_state=42, n_jobs=-1, **best)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    rf_coin.fit(X_tr_coin, y_tr_coin)

pred_tr_coin = np.maximum(rf_coin.predict(X_tr_coin), 0.0)
pred_te_coin = np.maximum(rf_coin.predict(X_te_coin), 0.0)

print(f"\n{'=' * 120}")
print(f"  COIN RESULTS (same best RF params, retrained on COIN train)")
print("=" * 120)
print(f"  COIN Train: {len(df_tr_coin):,} trades  |  Test: {len(df_te_coin):,} trades")
print(f"  In-sample:  R2={r2(y_tr_coin, pred_tr_coin):+.4f}  MAE={mae_fn(y_tr_coin, pred_tr_coin):.4f}  RMSE={rmse_fn(y_tr_coin, pred_tr_coin):.4f} bps")
print(f"  OOS (Sep):  R2={r2(y_te_coin, pred_te_coin):+.4f}  MAE={mae_fn(y_te_coin, pred_te_coin):.4f}  RMSE={rmse_fn(y_te_coin, pred_te_coin):.4f} bps")

# -- PLOTS (AAPL grid search) ------------------------------------------------
fig = plt.figure(figsize=(24, 12))
gs_fig = gridspec.GridSpec(2, 3, figure=fig, wspace=0.32, hspace=0.45)

METRIC_COL = "mean_MSE"
METRIC_LABEL = "MSE"

# Panel 1: Heatmap max_depth vs n_estimators
ax1 = fig.add_subplot(gs_fig[0, 0])
depths = sorted(results["max_depth"].unique())
n_ests = sorted(results["n_estimators"].unique())
heat1 = np.zeros((len(depths), len(n_ests)))
depth_labels = [str(d) for d in depths]
for i, d in enumerate(depths):
    for j, ne in enumerate(n_ests):
        mask = (results["max_depth"] == d) & (results["n_estimators"] == ne)
        heat1[i, j] = results.loc[mask, METRIC_COL].mean()

im1 = ax1.imshow(heat1, cmap="RdYlGn_r", aspect="auto")
ax1.set_xticks(range(len(n_ests)))
ax1.set_xticklabels([str(n) for n in n_ests], fontsize=9)
ax1.set_yticks(range(len(depths)))
ax1.set_yticklabels(depth_labels, fontsize=9)
ax1.set_xlabel("n_estimators", fontsize=10)
ax1.set_ylabel("max_depth", fontsize=10)
ax1.set_title(f"AAPL Mean {METRIC_LABEL}: max_depth vs n_estimators\n(averaged over other params)",
              fontsize=11, fontweight="bold")
for i in range(len(depths)):
    for j in range(len(n_ests)):
        ax1.text(j, i, f"{heat1[i, j]:.2f}", ha="center", va="center",
                 fontsize=8, fontweight="bold",
                 color="white" if heat1[i, j] > heat1.mean() else "black")
fig.colorbar(im1, ax=ax1, shrink=0.8, pad=0.02).set_label(METRIC_LABEL, fontsize=9)

# Panel 2: Heatmap max_depth vs min_samples_leaf
ax2 = fig.add_subplot(gs_fig[0, 1])
leaves = sorted(results["min_samples_leaf"].unique())
heat2 = np.zeros((len(depths), len(leaves)))
for i, d in enumerate(depths):
    for j, lf in enumerate(leaves):
        mask = (results["max_depth"] == d) & (results["min_samples_leaf"] == lf)
        heat2[i, j] = results.loc[mask, METRIC_COL].mean()

im2 = ax2.imshow(heat2, cmap="RdYlGn_r", aspect="auto")
ax2.set_xticks(range(len(leaves)))
ax2.set_xticklabels([str(l) for l in leaves], fontsize=9)
ax2.set_yticks(range(len(depths)))
ax2.set_yticklabels(depth_labels, fontsize=9)
ax2.set_xlabel("min_samples_leaf", fontsize=10)
ax2.set_ylabel("max_depth", fontsize=10)
ax2.set_title(f"AAPL Mean {METRIC_LABEL}: max_depth vs min_samples_leaf\n(averaged over other params)",
              fontsize=11, fontweight="bold")
for i in range(len(depths)):
    for j in range(len(leaves)):
        ax2.text(j, i, f"{heat2[i, j]:.2f}", ha="center", va="center",
                 fontsize=8, fontweight="bold",
                 color="white" if heat2[i, j] > heat2.mean() else "black")
fig.colorbar(im2, ax=ax2, shrink=0.8, pad=0.02).set_label(METRIC_LABEL, fontsize=9)

# Panel 3: MSE sensitivity per hyperparameter
ax3 = fig.add_subplot(gs_fig[0, 2])
param_names = ["max_depth", "n_estimators", "min_samples_leaf", "max_features", "bootstrap"]
param_colors = ["#2563eb", "#dc2626", "#16a34a", "#7c3aed", "#f59e0b"]

for pi, (pname, color) in enumerate(zip(param_names, param_colors)):
    vals = sorted(results[pname].unique(), key=lambda x: (isinstance(x, str), x))
    metric_at_vals = []
    for v in vals:
        mask = results[pname] == v
        metric_at_vals.append(results.loc[mask, METRIC_COL].mean())

    x_norm = np.linspace(0, 1, len(vals))
    ax3.plot(x_norm, metric_at_vals, marker="o", color=color, linewidth=1.8,
             markersize=6, label=pname, alpha=0.85)
    for xi, (xn, v) in enumerate(zip(x_norm, vals)):
        label_str = str(v) if v is not None else "None"
        ax3.annotate(label_str, (xn, metric_at_vals[xi]),
                     textcoords="offset points", xytext=(0, -13 if pi % 2 == 0 else 11),
                     fontsize=6.5, color=color, ha="center", fontweight="bold")

ax3.set_xlabel("Parameter value (normalized to [0,1])", fontsize=10)
ax3.set_ylabel(f"Mean inner-CV {METRIC_LABEL}", fontsize=10)
ax3.set_title(f"AAPL {METRIC_LABEL} sensitivity per hyperparameter\n(marginal averages)",
              fontsize=11, fontweight="bold")
ax3.legend(fontsize=8, loc="best", ncol=2)
ax3.grid(True, alpha=0.2)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)

# Panel 4: Feature importances
ax4 = fig.add_subplot(gs_fig[1, 0])
sort_idx = np.argsort(imp)[::-1]
feat_sorted = [FEATURES[i] for i in sort_idx]
imp_sorted = imp[sort_idx]
feat_colors = plt.cm.Set2(np.linspace(0, 1, len(FEATURES)))

ax4.barh(range(len(FEATURES)), imp_sorted[::-1],
         color=feat_colors[::-1], edgecolor="white", linewidth=0.6, height=0.6)
ax4.set_yticks(range(len(FEATURES)))
ax4.set_yticklabels(feat_sorted[::-1], fontsize=9)
for i, v in enumerate(imp_sorted[::-1]):
    ax4.text(v + 0.003, i, f"{v:.4f}", va="center", fontsize=8.5)
ax4.set_xlabel("Feature importance (Mean Decrease in Impurity)", fontsize=10)
ax4.set_title("AAPL RF Feature Importances",
              fontsize=11, fontweight="bold")
ax4.grid(axis="x", alpha=0.2)
ax4.spines["top"].set_visible(False)
ax4.spines["right"].set_visible(False)

# Panel 5: Predicted vs Actual (OOS)
ax5 = fig.add_subplot(gs_fig[1, 1])
rng = np.random.default_rng(42)
clip_v = np.percentile(y_te, 98)
samp = rng.choice(len(y_te), size=min(5000, len(y_te)), replace=False)
ax5.scatter(y_te[samp], np.clip(pred_te_aapl[samp], 0, clip_v),
            alpha=0.08, s=8, color="#2563eb", linewidths=0, rasterized=True)
ax5.axline((0, 0), slope=1, color="black", lw=1.2, ls="--", alpha=0.6, label="Perfect prediction")
ax5.set_xlim(0, clip_v)
ax5.set_ylim(0, clip_v)
ax5.set_xlabel("Actual |impact_vwap_bps|", fontsize=10)
ax5.set_ylabel("Predicted", fontsize=10)
ax5.set_title(f"AAPL Predicted vs Actual (Sep 2024)\n"
              f"R2={r2(y_te, pred_te_aapl):+.4f}  RMSE={rmse_fn(y_te, pred_te_aapl):.4f}",
              fontsize=10, fontweight="bold")
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.18)
ax5.spines["top"].set_visible(False)
ax5.spines["right"].set_visible(False)

# Panel 6: Summary text
ax6 = fig.add_subplot(gs_fig[1, 2])
ax6.axis("off")

text = (
    f"HOLDOUT RESULTS (6 features, RF MSE)\n"
    f"{'=' * 46}\n\n"
    f"Best hyperparameters (from AAPL grid search):\n"
    f"  max_depth        = {best.get('max_depth')}\n"
    f"  n_estimators     = {best.get('n_estimators')}\n"
    f"  min_samples_leaf = {best.get('min_samples_leaf')}\n"
    f"  max_features     = {best.get('max_features')}\n"
    f"  bootstrap        = {best.get('bootstrap')}\n\n"
    f"Best inner-CV MSE:  {-gs.best_score_:.4f}\n"
    f"Best inner-CV RMSE: {best_rmse:.4f} bps\n\n"
    f"AAPL (train {len(df_tr):,} / test {len(df_te):,}):\n"
    f"  In-sample  R2={r2(y_tr, pred_tr_aapl):+.4f}  MAE={mae_fn(y_tr, pred_tr_aapl):.4f}\n"
    f"  OOS (Sep)  R2={r2(y_te, pred_te_aapl):+.4f}  MAE={mae_fn(y_te, pred_te_aapl):.4f}\n\n"
    f"COIN (train {len(df_tr_coin):,} / test {len(df_te_coin):,}):\n"
    f"  In-sample  R2={r2(y_tr_coin, pred_tr_coin):+.4f}  MAE={mae_fn(y_tr_coin, pred_tr_coin):.4f}\n"
    f"  OOS (Sep)  R2={r2(y_te_coin, pred_te_coin):+.4f}  MAE={mae_fn(y_te_coin, pred_te_coin):.4f}\n"
)
ax6.text(0.05, 0.95, text, transform=ax6.transAxes,
         fontsize=10, fontfamily="monospace", verticalalignment="top",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8fafc", edgecolor="#cbd5e1"))

fig.suptitle(
    f"AAPL lit buy blocks  Random Forest GridSearchCV  (6 features, MSE scoring)\n"
    f"{n_combos:,} combos x 3-fold TimeSeriesSplit (Jun-Aug only)  |  "
    f"Best inner-CV RMSE: {best_rmse:.4f} bps",
    fontsize=13, fontweight="bold", y=1.02,
)

plt.savefig("aapl_gridsearch_rf_mse.png", dpi=150, bbox_inches="tight")
print(f"\nSaved -> aapl_gridsearch_rf_mse.png")
