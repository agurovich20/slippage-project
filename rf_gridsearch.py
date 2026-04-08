"""
Random Forest GridSearchCV on June-Aug 2024 training data.

Grid search uses criterion='squared_error' only.
n_jobs=-1 on the RF estimator (parallelise tree building), n_jobs=1 on GridSearchCV.

Grid: 3x4x3x2x3 = 216 combinations x 5-fold TimeSeriesSplit = 1,080 fits.
Parameters:
  n_estimators    : [100, 200, 500]
  max_depth       : [3, 5, 8, None]
  min_samples_leaf: [5, 10, 50]
  min_samples_split: [5, 10]
  max_features    : ['sqrt', 0.5, 1.0]

After grid search, retrain best config with both criterion='squared_error'
and criterion='absolute_error', evaluate on September holdout.

Scoring: neg_mean_absolute_error
Output:
  - Console: top 20 and bottom 5 rows, holdout comparison
  - data/rf_gridsearch_details.csv  (full results table)
  - aapl_rf_gridsearch.png          (3-panel figure)
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

# ── Load training and test data ──────────────────────────────────────────────
df_tr = pd.read_parquet("data/lit_buy_features_v2.parquet")
df_te = pd.read_parquet("data/lit_buy_features_v2_sep.parquet")

df_tr["abs_impact"] = df_tr["impact_vwap_bps"].abs()
df_te["abs_impact"] = df_te["impact_vwap_bps"].abs()

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
    "n_estimators":     [100, 200, 500],
    "max_depth":        [3, 5, 8, None],
    "min_samples_leaf": [5, 10, 50],
    "min_samples_split":[5, 10],
    "max_features":     ["sqrt", 0.5, 1.0],
}

n_combos = 1
for v in PARAM_GRID.values():
    n_combos *= len(v)
print(f"\nGrid: {n_combos:,} combinations x 5 folds = {n_combos * 5:,} fits")

# ── Metric helpers ──────────────────────────────────────────────────────────
def r2(ytrue, ypred):
    ss_res = ((ytrue - ypred) ** 2).sum()
    ss_tot = ((ytrue - ytrue.mean()) ** 2).sum()
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

def mae_fn(ytrue, ypred):
    return np.mean(np.abs(ytrue - ypred))

# ── Run GridSearchCV with criterion='squared_error' ─────────────────────────
tscv = TimeSeriesSplit(n_splits=5)

print(f"\n{'='*80}")
print(f"  GridSearchCV with criterion='squared_error'")
print(f"  n_jobs=-1 on RF estimator, n_jobs=1 on GridSearchCV")
print(f"{'='*80}")

base = RandomForestRegressor(
    criterion="squared_error",
    random_state=42,
    n_jobs=-1,          # parallelise tree building within each RF
)

gs = GridSearchCV(
    base,
    PARAM_GRID,
    cv=tscv,
    scoring="neg_mean_absolute_error",
    refit=True,
    n_jobs=1,           # sequential fold/param iteration
    verbose=1,
)

t0 = time.time()
print("Starting GridSearchCV...", flush=True)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    gs.fit(X_tr, y_tr)
elapsed = time.time() - t0
print(f"Done in {elapsed:.1f}s.\n")

# Build results table
res = pd.DataFrame(gs.cv_results_)
df_res = pd.DataFrame({
    "rank":             res["rank_test_score"],
    "n_estimators":     res["param_n_estimators"],
    "max_depth":        res["param_max_depth"].fillna("None").astype(str),
    "min_samples_leaf": res["param_min_samples_leaf"],
    "min_samples_split":res["param_min_samples_split"],
    "max_features":     res["param_max_features"].astype(str),
    "mean_MAE":         -res["mean_test_score"],
    "std_MAE":          res["std_test_score"],
})
df_res = df_res.sort_values("rank").reset_index(drop=True)

# Print top 20 and bottom 5
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 160)
pd.set_option("display.float_format", "{:.6f}".format)

print(f"\n{'='*130}")
print(f"  TOP 20 — criterion='squared_error' (lowest mean inner-CV MAE)")
print(f"{'='*130}")
print(df_res.head(20).to_string(index=False))

print(f"\n{'='*130}")
print(f"  BOTTOM 5 — criterion='squared_error' (highest mean inner-CV MAE)")
print(f"{'='*130}")
print(df_res.tail(5).to_string(index=False))

best_params = gs.best_params_
print(f"\n  Best: {best_params}")
print(f"  Best mean MAE: {-gs.best_score_:.6f} bps")
print(f"  MAE range: {df_res['mean_MAE'].min():.6f} .. {df_res['mean_MAE'].max():.6f} bps")

# ── Save results table ─────────────────────────────────────────────────────
df_res.to_csv("data/rf_gridsearch_details.csv", index=False)
print(f"\nSaved full table -> data/rf_gridsearch_details.csv  ({len(df_res):,} rows)")

# ── Retrain best config with both criteria, evaluate on September holdout ──
CRITERIA = ["squared_error", "absolute_error"]
best_per_criterion = {}

print(f"\n{'='*100}")
print("  SEPTEMBER HOLDOUT — retrain best params with both criteria")
print(f"{'='*100}")

for crit in CRITERIA:
    rf = RandomForestRegressor(
        criterion=crit,
        random_state=42,
        n_jobs=-1,
        **best_params,
    )
    rf.fit(X_tr, y_tr)
    pred = np.maximum(rf.predict(X_te), 0.0)
    rv = r2(y_te, pred)
    mv = mae_fn(y_te, pred)

    best_per_criterion[crit] = {
        "estimator": rf,
        "pred": pred,
        "oos_r2": rv,
        "oos_mae": mv,
        "importances": rf.feature_importances_,
    }
    print(f"  RF criterion='{crit}'  |  params={best_params}")
    print(f"    OOS R2={rv:+.4f}  OOS MAE={mv:.4f} bps")
    print(f"    Feature importances: {dict(zip(FEATURES, rf.feature_importances_.round(4)))}\n")

print(f"{'='*100}")

# ── PLOTS ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 6.5))
gs_fig = gridspec.GridSpec(1, 3, figure=fig, wspace=0.32)

# ── Panel 1: Heatmap of max_depth vs n_estimators ───────────────────────────
ax1 = fig.add_subplot(gs_fig[0, 0])

depths = sorted(df_res["max_depth"].unique(), key=lambda x: (x == "None", x))
n_ests = sorted(df_res["n_estimators"].unique())
heat1 = np.zeros((len(depths), len(n_ests)))
for i, d in enumerate(depths):
    for j, ne in enumerate(n_ests):
        mask = (df_res["max_depth"] == d) & (df_res["n_estimators"] == ne)
        heat1[i, j] = df_res.loc[mask, "mean_MAE"].mean()

im1 = ax1.imshow(heat1, cmap="RdYlGn_r", aspect="auto")
ax1.set_xticks(range(len(n_ests)))
ax1.set_xticklabels([f"{ne}" for ne in n_ests], fontsize=9)
ax1.set_yticks(range(len(depths)))
ax1.set_yticklabels([f"{d}" for d in depths], fontsize=9)
ax1.set_xlabel("n_estimators", fontsize=10)
ax1.set_ylabel("max_depth", fontsize=10)
ax1.set_title("Mean MAE: max_depth vs n_estimators\n(averaged over other params)",
              fontsize=11, fontweight="bold")

for i in range(len(depths)):
    for j in range(len(n_ests)):
        ax1.text(j, i, f"{heat1[i, j]:.4f}", ha="center", va="center",
                 fontsize=8, fontweight="bold",
                 color="white" if heat1[i, j] > heat1.mean() else "black")

cb1 = fig.colorbar(im1, ax=ax1, shrink=0.8, pad=0.02)
cb1.set_label("MAE (bps)", fontsize=9)

# ── Panel 2: Heatmap of min_samples_leaf vs max_features ────────────────────
ax2 = fig.add_subplot(gs_fig[0, 1])

leaves = sorted(df_res["min_samples_leaf"].unique())
feats = sorted(df_res["max_features"].unique(), key=lambda x: (x == "sqrt", x))
heat2 = np.zeros((len(leaves), len(feats)))
for i, lf in enumerate(leaves):
    for j, mf in enumerate(feats):
        mask = (df_res["min_samples_leaf"] == lf) & (df_res["max_features"] == mf)
        heat2[i, j] = df_res.loc[mask, "mean_MAE"].mean()

im2 = ax2.imshow(heat2, cmap="RdYlGn_r", aspect="auto")
ax2.set_xticks(range(len(feats)))
ax2.set_xticklabels([f"{f}" for f in feats], fontsize=9)
ax2.set_yticks(range(len(leaves)))
ax2.set_yticklabels([f"{lf}" for lf in leaves], fontsize=9)
ax2.set_xlabel("max_features", fontsize=10)
ax2.set_ylabel("min_samples_leaf", fontsize=10)
ax2.set_title("Mean MAE: min_samples_leaf vs max_features\n(averaged over other params)",
              fontsize=11, fontweight="bold")

for i in range(len(leaves)):
    for j in range(len(feats)):
        ax2.text(j, i, f"{heat2[i, j]:.4f}", ha="center", va="center",
                 fontsize=8, fontweight="bold",
                 color="white" if heat2[i, j] > heat2.mean() else "black")

cb2 = fig.colorbar(im2, ax=ax2, shrink=0.8, pad=0.02)
cb2.set_label("MAE (bps)", fontsize=9)

# ── Panel 3: Line plot — MAE vs each param at best values ──────────────────
ax3 = fig.add_subplot(gs_fig[0, 2])

param_names = ["n_estimators", "max_depth", "min_samples_leaf",
               "min_samples_split", "max_features"]
param_colors = ["#2563eb", "#dc2626", "#16a34a", "#7c3aed", "#f59e0b"]

for pi, (pname, color) in enumerate(zip(param_names, param_colors)):
    vals = sorted(df_res[pname].unique(), key=lambda x: (x == "None", x == "sqrt", str(x)))
    mae_at_vals = []
    for v in vals:
        mask = pd.Series(True, index=df_res.index)
        for other in param_names:
            if other != pname:
                bval = "None" if best_params[other] is None else str(best_params[other])
                mask &= (df_res[other] == bval)
        mask &= (df_res[pname] == v)
        if mask.sum() > 0:
            mae_at_vals.append(df_res.loc[mask, "mean_MAE"].values[0])
        else:
            mae_at_vals.append(np.nan)

    x_norm = np.linspace(0, 1, len(vals))
    ax3.plot(x_norm, mae_at_vals, marker="o", color=color, linewidth=1.8,
             markersize=6, label=pname, alpha=0.85)

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

# Suptitle with holdout results
se_mae = best_per_criterion["squared_error"]["oos_mae"]
ae_mae = best_per_criterion["absolute_error"]["oos_mae"]
se_r2  = best_per_criterion["squared_error"]["oos_r2"]
ae_r2  = best_per_criterion["absolute_error"]["oos_r2"]
fig.suptitle(
    "AAPL lit buy blocks — Random Forest GridSearchCV detail\n"
    f"216 combos x 5-fold TimeSeriesSplit = 1,080 fits  |  "
    f"OOS: SE R2={se_r2:+.4f} MAE={se_mae:.4f}  "
    f"AE R2={ae_r2:+.4f} MAE={ae_mae:.4f} bps",
    fontsize=13, fontweight="bold", y=1.03,
)

plt.savefig("aapl_rf_gridsearch.png", dpi=150, bbox_inches="tight")
print(f"\nSaved -> aapl_rf_gridsearch.png")
