"""
XGB GridSearchCV on June-Aug 2024 training data, 6 FEATURES, LAD (MAE) scoring.

objective: reg:absoluteerror
scoring: neg_mean_absolute_error

Grid: 4x4x4x3x3x3 = 1,728 combinations x 3-fold TimeSeriesSplit = 5,184 fits.

Output:
  - data/gridsearch_xgb_lad.csv
  - aapl_gridsearch_xgb_lad.png
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

from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

def r2(ytrue, ypred):
    ss_res = ((ytrue - ypred)**2).sum()
    ss_tot = ((ytrue - ytrue.mean())**2).sum()
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

def mae_fn(ytrue, ypred):
    return np.mean(np.abs(ytrue - ypred))

def rmse_fn(ytrue, ypred):
    return np.sqrt(np.mean((ytrue - ypred)**2))

# -- Load AAPL data ----------------------------------------------------------
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

print(f"AAPL Train: {len(df_tr):,}  |  Test: {len(df_te):,}")

# -- Grid definition ----------------------------------------------------------
PARAM_GRID = {
    "max_depth":        [1, 2, 3, 4],
    "n_estimators":     [50, 80, 120, 200],
    "learning_rate":    [0.01, 0.04, 0.07, 0.1],
    "min_child_weight": [1, 5, 10],
    "reg_alpha":        [0, 1.0, 10.0],
    "reg_lambda":       [0.1, 1.0, 10.0],
}

n_combos = 1
for v in PARAM_GRID.values():
    n_combos *= len(v)
print(f"Grid: {n_combos:,} combinations x 3 folds = {n_combos * 3:,} fits")

# -- GridSearchCV -------------------------------------------------------------
base = XGBRegressor(
    objective="reg:absoluteerror",
    tree_method="hist",
    verbosity=0,
    random_state=42,
    n_jobs=1,
)

tscv = TimeSeriesSplit(n_splits=3)

gs = GridSearchCV(
    base, PARAM_GRID,
    cv=tscv,
    scoring="neg_mean_absolute_error",
    refit=True,
    n_jobs=1,
    verbose=1,
)

print("Starting GridSearchCV...", flush=True)
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
    "learning_rate":    res["param_learning_rate"],
    "min_child_weight": res["param_min_child_weight"],
    "reg_alpha":        res["param_reg_alpha"],
    "reg_lambda":       res["param_reg_lambda"],
    "mean_MAE":         -res["mean_test_score"],
    "std_MAE":          res["std_test_score"],
})
results = results.sort_values("rank").reset_index(drop=True)
results.to_csv("data/gridsearch_xgb_lad.csv", index=False)
print(f"Saved -> data/gridsearch_xgb_lad.csv  ({len(results):,} rows)\n")

pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 140)
pd.set_option("display.float_format", "{:.6f}".format)

print("=" * 120)
print("  TOP 20 CONFIGURATIONS (lowest mean inner-CV MAE)")
print("=" * 120)
print(results.head(20).to_string(index=False))

best = gs.best_params_
best_mae = -gs.best_score_
print(f"\n  Best: {best}")
print(f"  Best mean CV MAE: {best_mae:.6f} bps")

# -- Evaluate on AAPL ---------------------------------------------------------
best_model = gs.best_estimator_
pred_tr_aapl = np.maximum(best_model.predict(X_tr), 0.0)
pred_te_aapl = np.maximum(best_model.predict(X_te), 0.0)

print(f"\n{'=' * 120}")
print("  AAPL RESULTS (XGBoost LAD, best params)")
print("=" * 120)
print(f"  In-sample:  R2={r2(y_tr, pred_tr_aapl):+.4f}  MAE={mae_fn(y_tr, pred_tr_aapl):.4f}  RMSE={rmse_fn(y_tr, pred_tr_aapl):.4f}")
print(f"  OOS (Sep):  R2={r2(y_te, pred_te_aapl):+.4f}  MAE={mae_fn(y_te, pred_te_aapl):.4f}  RMSE={rmse_fn(y_te, pred_te_aapl):.4f}")

imp = best_model.feature_importances_
print(f"\n  Feature importances:")
for feat, val in sorted(zip(FEATURES, imp), key=lambda x: -x[1]):
    print(f"    {feat:<22} {val:.4f}")

# -- Evaluate on COIN ---------------------------------------------------------
df_tr_coin = pd.read_parquet("data/coin_lit_buy_features_train.parquet")
df_te_coin = pd.read_parquet("data/coin_lit_buy_features_test.parquet")
df_tr_coin["abs_impact"] = df_tr_coin["impact_vwap_bps"].abs()
df_te_coin["abs_impact"] = df_te_coin["impact_vwap_bps"].abs()
df_tr_coin = df_tr_coin.sort_values("date").reset_index(drop=True)
X_tr_coin = df_tr_coin[FEATURES].to_numpy(dtype=np.float64)
y_tr_coin = df_tr_coin["abs_impact"].to_numpy(dtype=np.float64)
X_te_coin = df_te_coin[FEATURES].to_numpy(dtype=np.float64)
y_te_coin = df_te_coin["abs_impact"].to_numpy(dtype=np.float64)

xgb_coin = XGBRegressor(objective="reg:absoluteerror", tree_method="hist",
                         verbosity=0, random_state=42, n_jobs=1, **best)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    xgb_coin.fit(X_tr_coin, y_tr_coin)
pred_tr_coin = np.maximum(xgb_coin.predict(X_tr_coin), 0.0)
pred_te_coin = np.maximum(xgb_coin.predict(X_te_coin), 0.0)

print(f"\n{'=' * 120}")
print("  COIN RESULTS (XGBoost LAD, same best params, retrained)")
print("=" * 120)
print(f"  COIN Train: {len(df_tr_coin):,}  |  Test: {len(df_te_coin):,}")
print(f"  In-sample:  R2={r2(y_tr_coin, pred_tr_coin):+.4f}  MAE={mae_fn(y_tr_coin, pred_tr_coin):.4f}  RMSE={rmse_fn(y_tr_coin, pred_tr_coin):.4f}")
print(f"  OOS (Sep):  R2={r2(y_te_coin, pred_te_coin):+.4f}  MAE={mae_fn(y_te_coin, pred_te_coin):.4f}  RMSE={rmse_fn(y_te_coin, pred_te_coin):.4f}")

# -- PLOTS --------------------------------------------------------------------
fig = plt.figure(figsize=(24, 12))
gs_fig = gridspec.GridSpec(2, 3, figure=fig, wspace=0.32, hspace=0.45)
METRIC_COL = "mean_MAE"
METRIC_LABEL = "MAE"

# Panel 1: Heatmap max_depth vs learning_rate
ax1 = fig.add_subplot(gs_fig[0, 0])
depths = sorted(results["max_depth"].unique())
lrs = sorted(results["learning_rate"].unique())
heat1 = np.zeros((len(depths), len(lrs)))
for i, d in enumerate(depths):
    for j, lr in enumerate(lrs):
        mask = (results["max_depth"] == d) & (results["learning_rate"] == lr)
        heat1[i, j] = results.loc[mask, METRIC_COL].mean()
im1 = ax1.imshow(heat1, cmap="RdYlGn_r", aspect="auto")
ax1.set_xticks(range(len(lrs)))
ax1.set_xticklabels([f"{lr}" for lr in lrs], fontsize=9)
ax1.set_yticks(range(len(depths)))
ax1.set_yticklabels([f"{d}" for d in depths], fontsize=9)
ax1.set_xlabel("learning_rate", fontsize=10)
ax1.set_ylabel("max_depth", fontsize=10)
ax1.set_title(f"AAPL Mean {METRIC_LABEL}: max_depth vs learning_rate\n(averaged over other params)",
              fontsize=11, fontweight="bold")
for i in range(len(depths)):
    for j in range(len(lrs)):
        ax1.text(j, i, f"{heat1[i, j]:.3f}", ha="center", va="center",
                 fontsize=8, fontweight="bold",
                 color="white" if heat1[i, j] > heat1.mean() else "black")
fig.colorbar(im1, ax=ax1, shrink=0.8, pad=0.02).set_label(METRIC_LABEL, fontsize=9)

# Panel 2: Heatmap reg_alpha vs reg_lambda
ax2 = fig.add_subplot(gs_fig[0, 1])
alphas = sorted(results["reg_alpha"].unique())
lambdas = sorted(results["reg_lambda"].unique())
heat2 = np.zeros((len(alphas), len(lambdas)))
for i, a in enumerate(alphas):
    for j, l in enumerate(lambdas):
        mask = (results["reg_alpha"] == a) & (results["reg_lambda"] == l)
        heat2[i, j] = results.loc[mask, METRIC_COL].mean()
im2 = ax2.imshow(heat2, cmap="RdYlGn_r", aspect="auto")
ax2.set_xticks(range(len(lambdas)))
ax2.set_xticklabels([f"{l}" for l in lambdas], fontsize=9)
ax2.set_yticks(range(len(alphas)))
ax2.set_yticklabels([f"{a}" for a in alphas], fontsize=9)
ax2.set_xlabel("reg_lambda (L2)", fontsize=10)
ax2.set_ylabel("reg_alpha (L1)", fontsize=10)
ax2.set_title(f"AAPL Mean {METRIC_LABEL}: reg_alpha vs reg_lambda\n(averaged over other params)",
              fontsize=11, fontweight="bold")
for i in range(len(alphas)):
    for j in range(len(lambdas)):
        ax2.text(j, i, f"{heat2[i, j]:.3f}", ha="center", va="center",
                 fontsize=8, fontweight="bold",
                 color="white" if heat2[i, j] > heat2.mean() else "black")
fig.colorbar(im2, ax=ax2, shrink=0.8, pad=0.02).set_label(METRIC_LABEL, fontsize=9)

# Panel 3: MAE sensitivity per hyperparameter
ax3 = fig.add_subplot(gs_fig[0, 2])
param_names = ["max_depth", "n_estimators", "learning_rate",
               "min_child_weight", "reg_alpha", "reg_lambda"]
param_colors = ["#2563eb", "#dc2626", "#16a34a", "#7c3aed", "#f59e0b", "#06b6d4"]
for pi, (pname, color) in enumerate(zip(param_names, param_colors)):
    vals = sorted(results[pname].unique())
    metric_at_vals = []
    for v in vals:
        mask = pd.Series(True, index=results.index)
        for other in param_names:
            if other != pname:
                mask &= (results[other] == best[other])
        mask &= (results[pname] == v)
        if mask.sum() > 0:
            metric_at_vals.append(results.loc[mask, METRIC_COL].values[0])
        else:
            metric_at_vals.append(np.nan)
    x_norm = np.linspace(0, 1, len(vals))
    ax3.plot(x_norm, metric_at_vals, marker="o", color=color, linewidth=1.8,
             markersize=6, label=pname, alpha=0.85)
    for xi, (xn, v) in enumerate(zip(x_norm, vals)):
        ax3.annotate(f"{v}", (xn, metric_at_vals[xi]),
                     textcoords="offset points", xytext=(0, -13 if pi % 2 == 0 else 11),
                     fontsize=6.5, color=color, ha="center", fontweight="bold")
ax3.set_xlabel("Parameter value (normalized to [0,1])", fontsize=10)
ax3.set_ylabel(f"Mean inner-CV {METRIC_LABEL}", fontsize=10)
ax3.set_title(f"AAPL {METRIC_LABEL} sensitivity per hyperparameter\n(others fixed at best values)",
              fontsize=11, fontweight="bold")
ax3.legend(fontsize=7, loc="best", ncol=2)
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
ax4.set_xlabel("Feature importance (gain)", fontsize=10)
ax4.set_title("AAPL XGBoost LAD Feature Importances", fontsize=11, fontweight="bold")
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
              f"R2={r2(y_te, pred_te_aapl):+.4f}  MAE={mae_fn(y_te, pred_te_aapl):.4f}",
              fontsize=10, fontweight="bold")
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.18)
ax5.spines["top"].set_visible(False)
ax5.spines["right"].set_visible(False)

# Panel 6: Summary text
ax6 = fig.add_subplot(gs_fig[1, 2])
ax6.axis("off")
sep = "=" * 46
text = (
    f"RESULTS (6 features, XGBoost LAD)\n"
    f"{sep}\n\n"
    f"Best hyperparameters:\n"
    f"  max_depth       = {best['max_depth']}\n"
    f"  n_estimators    = {best['n_estimators']}\n"
    f"  learning_rate   = {best['learning_rate']}\n"
    f"  min_child_weight= {best['min_child_weight']}\n"
    f"  reg_alpha       = {best['reg_alpha']}\n"
    f"  reg_lambda      = {best['reg_lambda']}\n\n"
    f"Best inner-CV MAE: {best_mae:.4f} bps\n\n"
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
    f"AAPL lit buy blocks  XGBoost LAD GridSearchCV  (6 features, MAE scoring)\n"
    f"{n_combos:,} combos x 3-fold TimeSeriesSplit  |  Best inner-CV MAE: {best_mae:.4f} bps",
    fontsize=13, fontweight="bold", y=1.02,
)

plt.savefig("aapl_gridsearch_xgb_lad.png", dpi=150, bbox_inches="tight")
print(f"\nSaved -> aapl_gridsearch_xgb_lad.png")
