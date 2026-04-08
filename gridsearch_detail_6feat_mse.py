"""
XGB GridSearchCV on June-Aug 2024 training data — 6 FEATURES, MSE scoring.

Same setup as gridsearch_detail_6feat.py but scoring=neg_mean_squared_error
instead of neg_mean_absolute_error.

Grid: 4x4x4x3x3x3 = 1,728 combinations x 3-fold TimeSeriesSplit = 5,184 fits.

Parameters:
  max_depth       : [1, 2, 3, 4]
  n_estimators    : [50, 80, 120, 200]
  learning_rate   : [0.01, 0.04, 0.07, 0.1]
  min_child_weight: [1, 5, 10]
  reg_alpha       : [0, 1.0, 10.0]
  reg_lambda      : [0.1, 1.0, 10.0]

Features (6):
  dollar_value, log_dollar_value, participation_rate, roll_spread_500,
  roll_vol_500, exchange_id

Scoring: neg_mean_squared_error
GridSearchCV objective: reg:squarederror

Output:
  - data/gridsearch_details_6feat_mse.csv
  - aapl_gridsearch_detail_6feat_mse.png
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

# ── Load training and test data ──────────────────────────────────────────────
df_tr = pd.read_parquet("data/lit_buy_features_v2.parquet")
df_te = pd.read_parquet("data/lit_buy_features_v2_sep.parquet")

df_tr["abs_impact"] = df_tr["impact_vwap_bps"].abs()
df_te["abs_impact"] = df_te["impact_vwap_bps"].abs()

df_tr = df_tr.sort_values("date").reset_index(drop=True)

print(f"Train (CV only): {len(df_tr):,} trades  |  {df_tr['date'].nunique()} days  "
      f"({df_tr['date'].min()} .. {df_tr['date'].max()})")
print(f"Holdout (Sep):   {len(df_te):,} trades  |  {df_te['date'].nunique()} days  "
      f"({df_te['date'].min()} .. {df_te['date'].max()})")
print("  ** September data is NOT used in the 3-fold CV — holdout only **")

# ── Feature / target arrays ─────────────────────────────────────────────────
FEATURES = [
    "dollar_value",
    "log_dollar_value",
    "participation_rate",
    "roll_spread_500",
    "roll_vol_500",
    "exchange_id",
]
X_tr = df_tr[FEATURES].to_numpy(dtype=np.float64)
y_tr = df_tr["abs_impact"].to_numpy(dtype=np.float64)
X_te = df_te[FEATURES].to_numpy(dtype=np.float64)
y_te = df_te["abs_impact"].to_numpy(dtype=np.float64)

print(f"Features ({len(FEATURES)}): {FEATURES}")
print(f"Train shape: {X_tr.shape}  |  Target mean: {y_tr.mean():.4f} bps")

# ── Grid definition ─────────────────────────────────────────────────────────
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
print(f"\nGrid: {n_combos:,} combinations x 3 folds = {n_combos * 3:,} fits")
print(f"Scoring: neg_mean_squared_error")
print(f"GridSearchCV objective: reg:squarederror")

# ── GridSearchCV — runs ONLY on June-Aug training data ──────────────────────
base = XGBRegressor(
    objective="reg:squarederror",
    tree_method="hist",
    verbosity=0,
    random_state=42,
    n_jobs=1,
)

tscv = TimeSeriesSplit(n_splits=3)

gs = GridSearchCV(
    base,
    PARAM_GRID,
    cv=tscv,
    scoring="neg_mean_squared_error",
    refit=True,
    n_jobs=1,
    verbose=1,
)

print("\nStarting GridSearchCV (on Jun-Aug data only)...", flush=True)
t0 = time.time()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    gs.fit(X_tr, y_tr)

elapsed = time.time() - t0
print(f"Done in {elapsed:.1f}s.\n")

# ── Build results table ─────────────────────────────────────────────────────
res = pd.DataFrame(gs.cv_results_)

results = pd.DataFrame({
    "rank":             res["rank_test_score"],
    "max_depth":        res["param_max_depth"],
    "n_estimators":     res["param_n_estimators"],
    "learning_rate":    res["param_learning_rate"],
    "min_child_weight": res["param_min_child_weight"],
    "reg_alpha":        res["param_reg_alpha"],
    "reg_lambda":       res["param_reg_lambda"],
    "mean_MSE":         -res["mean_test_score"],
    "std_MSE":          res["std_test_score"],
})
results = results.sort_values("rank").reset_index(drop=True)

# ── Save full table ─────────────────────────────────────────────────────────
results.to_csv("data/gridsearch_details_6feat_mse.csv", index=False)
print(f"Saved full table -> data/gridsearch_details_6feat_mse.csv  ({len(results):,} rows)\n")

# ── Print top 20 and bottom 5 ──────────────────────────────────────────────
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

best_mse = np.sqrt(-gs.best_score_)
print(f"\n  Best: {gs.best_params_}")
print(f"  Best mean MSE: {-gs.best_score_:.6f}  (RMSE: {best_mse:.6f} bps)")
print(f"  MSE range: {results['mean_MSE'].min():.6f} .. {results['mean_MSE'].max():.6f}")

# ── Retrain best config with BOTH objectives, evaluate on Sep holdout ───────
best = gs.best_params_

def r2(ytrue, ypred):
    ss_res = ((ytrue - ypred) ** 2).sum()
    ss_tot = ((ytrue - ytrue.mean()) ** 2).sum()
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

def mae_fn(ytrue, ypred):
    return np.mean(np.abs(ytrue - ypred))

def rmse_fn(ytrue, ypred):
    return np.sqrt(np.mean((ytrue - ypred) ** 2))

print(f"\n{'=' * 120}")
print("  HOLDOUT COMPARISON: best hyperparams with both objectives (6 features)")
print(f"  Params: {best}")
print("=" * 120)

holdout_results = {}
for obj_name, obj in [("reg:squarederror", "reg:squarederror"),
                       ("reg:absoluteerror", "reg:absoluteerror")]:
    model = XGBRegressor(
        objective=obj,
        tree_method="hist",
        verbosity=0,
        random_state=42,
        n_jobs=1,
        max_depth=best["max_depth"],
        n_estimators=best["n_estimators"],
        learning_rate=best["learning_rate"],
        min_child_weight=best["min_child_weight"],
        reg_alpha=best["reg_alpha"],
        reg_lambda=best["reg_lambda"],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_tr, y_tr)
    pred = np.maximum(model.predict(X_te), 0.0)
    r2_val = r2(y_te, pred)
    mae_val = mae_fn(y_te, pred)
    rmse_val = rmse_fn(y_te, pred)
    holdout_results[obj_name] = {"r2": r2_val, "mae": mae_val, "rmse": rmse_val, "model": model}
    print(f"  {obj_name:<25}  OOS R2={r2_val:+.4f}  OOS MAE={mae_val:.4f}  OOS RMSE={rmse_val:.4f} bps")

# Feature importances from squarederror model
print(f"\n  Feature importances (reg:squarederror, best params):")
imp_se = holdout_results["reg:squarederror"]["model"].feature_importances_
for feat, val in sorted(zip(FEATURES, imp_se), key=lambda x: -x[1]):
    bar = "#" * int(val * 150)
    print(f"    {feat:<22} {val:.4f}  {bar}")

print("=" * 120)

# ── PLOTS ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(24, 12))
gs_fig = gridspec.GridSpec(2, 3, figure=fig, wspace=0.32, hspace=0.45)

METRIC_COL = "mean_MSE"
METRIC_LABEL = "MSE"

# ── Panel 1: Heatmap of max_depth vs learning_rate ──────────────────────────
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
ax1.set_title(f"Mean {METRIC_LABEL}: max_depth vs learning_rate\n(averaged over other params)",
              fontsize=11, fontweight="bold")

for i in range(len(depths)):
    for j in range(len(lrs)):
        ax1.text(j, i, f"{heat1[i, j]:.2f}", ha="center", va="center",
                 fontsize=8, fontweight="bold",
                 color="white" if heat1[i, j] > heat1.mean() else "black")

cb1 = fig.colorbar(im1, ax=ax1, shrink=0.8, pad=0.02)
cb1.set_label(f"{METRIC_LABEL}", fontsize=9)

# ── Panel 2: Heatmap of reg_alpha vs reg_lambda ────────────────────────────
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
ax2.set_title(f"Mean {METRIC_LABEL}: reg_alpha vs reg_lambda\n(averaged over other params)",
              fontsize=11, fontweight="bold")

for i in range(len(alphas)):
    for j in range(len(lambdas)):
        ax2.text(j, i, f"{heat2[i, j]:.2f}", ha="center", va="center",
                 fontsize=8, fontweight="bold",
                 color="white" if heat2[i, j] > heat2.mean() else "black")

cb2 = fig.colorbar(im2, ax=ax2, shrink=0.8, pad=0.02)
cb2.set_label(f"{METRIC_LABEL}", fontsize=9)

# ── Panel 3: Line plot — MSE vs each param at best values ──────────────────
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
ax3.set_title(f"{METRIC_LABEL} sensitivity per hyperparameter\n(others fixed at best values)",
              fontsize=11, fontweight="bold")
ax3.legend(fontsize=8, loc="upper right", ncol=2)
ax3.grid(True, alpha=0.2)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)

# ── Panel 4: Feature importances bar chart ──────────────────────────────────
ax4 = fig.add_subplot(gs_fig[1, 0])

sort_idx = np.argsort(imp_se)[::-1]
feat_sorted = [FEATURES[i] for i in sort_idx]
imp_sorted = imp_se[sort_idx]
feat_colors = plt.cm.Set2(np.linspace(0, 1, len(FEATURES)))

ax4.barh(range(len(FEATURES)), imp_sorted[::-1],
         color=feat_colors[::-1], edgecolor="white", linewidth=0.6, height=0.6)
ax4.set_yticks(range(len(FEATURES)))
ax4.set_yticklabels(feat_sorted[::-1], fontsize=9)
for i, v in enumerate(imp_sorted[::-1]):
    ax4.text(v + 0.003, i, f"{v:.4f}", va="center", fontsize=8.5)
ax4.set_xlabel("Feature importance (gain)", fontsize=10)
ax4.set_title("Feature importances\n(reg:squarederror, best params, 6 features)",
              fontsize=11, fontweight="bold")
ax4.grid(axis="x", alpha=0.2)
ax4.spines["top"].set_visible(False)
ax4.spines["right"].set_visible(False)

# ── Panel 5: Heatmap of max_depth vs min_child_weight ──────────────────────
ax5 = fig.add_subplot(gs_fig[1, 1])

mcws = sorted(results["min_child_weight"].unique())
heat5 = np.zeros((len(depths), len(mcws)))
for i, d in enumerate(depths):
    for j, mcw in enumerate(mcws):
        mask = (results["max_depth"] == d) & (results["min_child_weight"] == mcw)
        heat5[i, j] = results.loc[mask, METRIC_COL].mean()

im5 = ax5.imshow(heat5, cmap="RdYlGn_r", aspect="auto")
ax5.set_xticks(range(len(mcws)))
ax5.set_xticklabels([f"{m}" for m in mcws], fontsize=9)
ax5.set_yticks(range(len(depths)))
ax5.set_yticklabels([f"{d}" for d in depths], fontsize=9)
ax5.set_xlabel("min_child_weight", fontsize=10)
ax5.set_ylabel("max_depth", fontsize=10)
ax5.set_title(f"Mean {METRIC_LABEL}: max_depth vs min_child_weight\n(averaged over other params)",
              fontsize=11, fontweight="bold")

for i in range(len(depths)):
    for j in range(len(mcws)):
        ax5.text(j, i, f"{heat5[i, j]:.2f}", ha="center", va="center",
                 fontsize=8, fontweight="bold",
                 color="white" if heat5[i, j] > heat5.mean() else "black")

cb5 = fig.colorbar(im5, ax=ax5, shrink=0.8, pad=0.02)
cb5.set_label(f"{METRIC_LABEL}", fontsize=9)

# ── Panel 6: Holdout summary ───────────────────────────────────────────────
ax6 = fig.add_subplot(gs_fig[1, 2])
ax6.axis("off")

se = holdout_results["reg:squarederror"]
ae = holdout_results["reg:absoluteerror"]

text = (
    f"HOLDOUT RESULTS (Sep 2024, 6 features)\n"
    f"CV scoring: MSE\n"
    f"{'─' * 44}\n\n"
    f"Best hyperparameters:\n"
    f"  max_depth       = {best['max_depth']}\n"
    f"  n_estimators    = {best['n_estimators']}\n"
    f"  learning_rate   = {best['learning_rate']}\n"
    f"  min_child_weight= {best['min_child_weight']}\n"
    f"  reg_alpha       = {best['reg_alpha']}\n"
    f"  reg_lambda      = {best['reg_lambda']}\n\n"
    f"Best inner-CV MSE:  {-gs.best_score_:.4f}\n"
    f"Best inner-CV RMSE: {best_mse:.4f} bps\n\n"
    f"Objective          OOS R2   OOS MAE  OOS RMSE\n"
    f"{'─' * 48}\n"
    f"squarederror     {se['r2']:+.4f}  {se['mae']:.4f}  {se['rmse']:.4f}\n"
    f"absoluteerror    {ae['r2']:+.4f}  {ae['mae']:.4f}  {ae['rmse']:.4f}\n"
)
ax6.text(0.05, 0.95, text, transform=ax6.transAxes,
         fontsize=10, fontfamily="monospace", verticalalignment="top",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8fafc", edgecolor="#cbd5e1"))

fig.suptitle(
    "AAPL lit buy blocks — XGB GridSearchCV detail  (6 features, MSE scoring)\n"
    f"{n_combos:,} combos x 3-fold TimeSeriesSplit (Jun-Aug only)  |  "
    f"Best inner-CV RMSE: {best_mse:.4f} bps",
    fontsize=13, fontweight="bold", y=1.02,
)

plt.savefig("aapl_gridsearch_detail_6feat_mse.png", dpi=150, bbox_inches="tight")
print(f"\nSaved -> aapl_gridsearch_detail_6feat_mse.png")
