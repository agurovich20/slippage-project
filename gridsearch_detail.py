"""
XGB GridSearchCV on June-Aug 2024 training data.

Search uses reg:squarederror (fast analytic gradient) to find best hyperparams,
then retrains the best config with BOTH objectives (squarederror & absoluteerror)
and evaluates on the September 2024 holdout.

Grid: 4x3x3x3x3x3 = 972 combinations x 5-fold TimeSeriesSplit = 4,860 fits.

Parameters:
  max_depth       : [1, 2, 3, 4]
  n_estimators    : [50, 100, 200]
  learning_rate   : [0.01, 0.05, 0.1]
  min_child_weight: [1, 5, 10]
  reg_alpha       : [0, 1.0, 10.0]
  reg_lambda      : [0.1, 1.0, 10.0]

Scoring: neg_mean_absolute_error
GridSearchCV objective: reg:squarederror

Output:
  - Console: top 20 and bottom 5 rows, holdout comparison
  - data/gridsearch_details.csv  (full results table)
  - aapl_gridsearch_detail.png   (3-panel figure)
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

# Sort by date for proper time-series splitting
df_tr = df_tr.sort_values("date").reset_index(drop=True)

print(f"Train: {len(df_tr):,} trades  |  {df_tr['date'].nunique()} days  "
      f"({df_tr['date'].min()} .. {df_tr['date'].max()})")
print(f"Test : {len(df_te):,} trades  |  {df_te['date'].nunique()} days  "
      f"({df_te['date'].min()} .. {df_te['date'].max()})")

# ── Feature / target arrays ─────────────────────────────────────────────────
FEATURES = ["roll_spread_500", "roll_vol_500", "participation_rate"]
X_tr = df_tr[FEATURES].to_numpy(dtype=np.float64)
y_tr = df_tr["abs_impact"].to_numpy(dtype=np.float64)
X_te = df_te[FEATURES].to_numpy(dtype=np.float64)
y_te = df_te["abs_impact"].to_numpy(dtype=np.float64)

print(f"Features shape: {X_tr.shape}  |  Target mean: {y_tr.mean():.4f} bps")

# ── Grid definition ─────────────────────────────────────────────────────────
PARAM_GRID = {
    "max_depth":        [1, 2, 3, 4],
    "n_estimators":     [50, 100, 200],
    "learning_rate":    [0.01, 0.05, 0.1],
    "min_child_weight": [1, 5, 10],
    "reg_alpha":        [0, 1.0, 10.0],
    "reg_lambda":       [0.1, 1.0, 10.0],
}

n_combos = 1
for v in PARAM_GRID.values():
    n_combos *= len(v)
print(f"\nGrid: {n_combos:,} combinations x 5 folds = {n_combos * 5:,} fits")
print(f"GridSearchCV objective: reg:squarederror (fast)")

# ── GridSearchCV with reg:squarederror ──────────────────────────────────────
base = XGBRegressor(
    objective="reg:squarederror",
    tree_method="hist",
    verbosity=0,
    random_state=42,
    n_jobs=1,
)

tscv = TimeSeriesSplit(n_splits=5)

gs = GridSearchCV(
    base,
    PARAM_GRID,
    cv=tscv,
    scoring="neg_mean_absolute_error",
    refit=True,
    n_jobs=1,
    verbose=1,
)

print("\nStarting GridSearchCV...", flush=True)
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
    "mean_MAE":         -res["mean_test_score"],   # flip sign
    "std_MAE":          res["std_test_score"],
})
results = results.sort_values("rank").reset_index(drop=True)

# ── Save full table ─────────────────────────────────────────────────────────
results.to_csv("data/gridsearch_details.csv", index=False)
print(f"Saved full table -> data/gridsearch_details.csv  ({len(results):,} rows)\n")

# ── Print top 20 and bottom 5 ──────────────────────────────────────────────
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 140)
pd.set_option("display.float_format", "{:.6f}".format)

print("=" * 120)
print("  TOP 20 CONFIGURATIONS (lowest mean inner-CV MAE)")
print("=" * 120)
print(results.head(20).to_string(index=False))

print(f"\n{'=' * 120}")
print("  BOTTOM 5 CONFIGURATIONS (highest mean inner-CV MAE)")
print("=" * 120)
print(results.tail(5).to_string(index=False))

print(f"\n  Best: {gs.best_params_}")
print(f"  Best mean MAE: {-gs.best_score_:.6f} bps")
print(f"  MAE range: {results['mean_MAE'].min():.6f} .. {results['mean_MAE'].max():.6f} bps")

# ── Retrain best config with BOTH objectives, evaluate on Sep holdout ───────
best = gs.best_params_

def r2(ytrue, ypred):
    ss_res = ((ytrue - ypred) ** 2).sum()
    ss_tot = ((ytrue - ytrue.mean()) ** 2).sum()
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

def mae_fn(ytrue, ypred):
    return np.mean(np.abs(ytrue - ypred))

print(f"\n{'=' * 120}")
print("  HOLDOUT COMPARISON: best hyperparams with both objectives")
print(f"  Params: {best}")
print("=" * 120)

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
    print(f"  {obj_name:<25}  OOS R2={r2_val:+.4f}  OOS MAE={mae_val:.4f} bps")

print("=" * 120)

# ── PLOTS ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 6.5))
gs_fig = gridspec.GridSpec(1, 3, figure=fig, wspace=0.32)

# ── Panel 1: Heatmap of max_depth vs learning_rate ──────────────────────────
ax1 = fig.add_subplot(gs_fig[0, 0])

depths = sorted(results["max_depth"].unique())
lrs = sorted(results["learning_rate"].unique())
heat1 = np.zeros((len(depths), len(lrs)))
for i, d in enumerate(depths):
    for j, lr in enumerate(lrs):
        mask = (results["max_depth"] == d) & (results["learning_rate"] == lr)
        heat1[i, j] = results.loc[mask, "mean_MAE"].mean()

im1 = ax1.imshow(heat1, cmap="RdYlGn_r", aspect="auto")
ax1.set_xticks(range(len(lrs)))
ax1.set_xticklabels([f"{lr}" for lr in lrs], fontsize=9)
ax1.set_yticks(range(len(depths)))
ax1.set_yticklabels([f"{d}" for d in depths], fontsize=9)
ax1.set_xlabel("learning_rate", fontsize=10)
ax1.set_ylabel("max_depth", fontsize=10)
ax1.set_title("Mean MAE: max_depth vs learning_rate\n(averaged over other params)",
              fontsize=11, fontweight="bold")

for i in range(len(depths)):
    for j in range(len(lrs)):
        ax1.text(j, i, f"{heat1[i, j]:.4f}", ha="center", va="center",
                 fontsize=8, fontweight="bold",
                 color="white" if heat1[i, j] > heat1.mean() else "black")

cb1 = fig.colorbar(im1, ax=ax1, shrink=0.8, pad=0.02)
cb1.set_label("MAE (bps)", fontsize=9)

# ── Panel 2: Heatmap of reg_alpha vs reg_lambda ────────────────────────────
ax2 = fig.add_subplot(gs_fig[0, 1])

alphas = sorted(results["reg_alpha"].unique())
lambdas = sorted(results["reg_lambda"].unique())
heat2 = np.zeros((len(alphas), len(lambdas)))
for i, a in enumerate(alphas):
    for j, l in enumerate(lambdas):
        mask = (results["reg_alpha"] == a) & (results["reg_lambda"] == l)
        heat2[i, j] = results.loc[mask, "mean_MAE"].mean()

im2 = ax2.imshow(heat2, cmap="RdYlGn_r", aspect="auto")
ax2.set_xticks(range(len(lambdas)))
ax2.set_xticklabels([f"{l}" for l in lambdas], fontsize=9)
ax2.set_yticks(range(len(alphas)))
ax2.set_yticklabels([f"{a}" for a in alphas], fontsize=9)
ax2.set_xlabel("reg_lambda (L2)", fontsize=10)
ax2.set_ylabel("reg_alpha (L1)", fontsize=10)
ax2.set_title("Mean MAE: reg_alpha vs reg_lambda\n(averaged over other params)",
              fontsize=11, fontweight="bold")

for i in range(len(alphas)):
    for j in range(len(lambdas)):
        ax2.text(j, i, f"{heat2[i, j]:.4f}", ha="center", va="center",
                 fontsize=8, fontweight="bold",
                 color="white" if heat2[i, j] > heat2.mean() else "black")

cb2 = fig.colorbar(im2, ax=ax2, shrink=0.8, pad=0.02)
cb2.set_label("MAE (bps)", fontsize=9)

# ── Panel 3: Line plot — MAE vs each param at best values ──────────────────
ax3 = fig.add_subplot(gs_fig[0, 2])

param_names = ["max_depth", "n_estimators", "learning_rate",
               "min_child_weight", "reg_alpha", "reg_lambda"]
param_colors = ["#2563eb", "#dc2626", "#16a34a", "#7c3aed", "#f59e0b", "#06b6d4"]

for pi, (pname, color) in enumerate(zip(param_names, param_colors)):
    vals = sorted(results[pname].unique())
    mae_at_vals = []
    for v in vals:
        # Fix all other params at best, vary this one
        mask = pd.Series(True, index=results.index)
        for other in param_names:
            if other != pname:
                mask &= (results[other] == best[other])
        mask &= (results[pname] == v)
        if mask.sum() > 0:
            mae_at_vals.append(results.loc[mask, "mean_MAE"].values[0])
        else:
            mae_at_vals.append(np.nan)

    # Normalize x to [0, 1] for overlaying different scales
    x_norm = np.linspace(0, 1, len(vals))
    ax3.plot(x_norm, mae_at_vals, marker="o", color=color, linewidth=1.8,
             markersize=6, label=pname, alpha=0.85)

    # Annotate tick values below/above
    for xi, (xn, v) in enumerate(zip(x_norm, vals)):
        ax3.annotate(f"{v}", (xn, mae_at_vals[xi]),
                     textcoords="offset points", xytext=(0, -13 if pi % 2 == 0 else 11),
                     fontsize=6.5, color=color, ha="center", fontweight="bold")

ax3.set_xlabel("Parameter value (normalized to [0,1])", fontsize=10)
ax3.set_ylabel("Mean inner-CV MAE (bps)", fontsize=10)
ax3.set_title("MAE sensitivity per hyperparameter\n(others fixed at best values)",
              fontsize=11, fontweight="bold")
ax3.legend(fontsize=8, loc="upper right", ncol=2)
ax3.grid(True, alpha=0.2)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)

fig.suptitle(
    "AAPL lit buy blocks — XGB GridSearchCV detail  (search obj: squarederror)\n"
    f"972 combos x 5-fold TimeSeriesSplit  |  "
    f"Best inner-CV MAE: {-gs.best_score_:.4f} bps",
    fontsize=13, fontweight="bold", y=1.03,
)

plt.savefig("aapl_gridsearch_detail.png", dpi=150, bbox_inches="tight")
print(f"\nSaved -> aapl_gridsearch_detail.png")
