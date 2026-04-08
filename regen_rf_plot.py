"""Regenerate RF grid search plot from saved CSV (no grid search rerun)."""
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
from sklearn.ensemble import RandomForestRegressor

def r2(ytrue, ypred):
    ss_res = ((ytrue - ypred)**2).sum()
    ss_tot = ((ytrue - ytrue.mean())**2).sum()
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

def mae_fn(ytrue, ypred):
    return np.mean(np.abs(ytrue - ypred))

def rmse_fn(ytrue, ypred):
    return np.sqrt(np.mean((ytrue - ypred)**2))

FEATURES = [
    "dollar_value", "log_dollar_value", "participation_rate",
    "roll_spread_500", "roll_vol_500", "exchange_id",
]
best = dict(max_depth=30, n_estimators=50, min_samples_leaf=20,
            max_features=0.33, bootstrap=False)
best_mse = 20.148001
best_rmse = np.sqrt(best_mse)

results = pd.read_csv("data/gridsearch_rf_mse.csv")
n_combos = len(results)

# Load data and retrain
df_tr = pd.read_parquet("data/lit_buy_features_v2.parquet")
df_te = pd.read_parquet("data/lit_buy_features_v2_sep.parquet")
df_tr["abs_impact"] = df_tr["impact_vwap_bps"].abs()
df_te["abs_impact"] = df_te["impact_vwap_bps"].abs()
df_tr = df_tr.sort_values("date").reset_index(drop=True)
X_tr = df_tr[FEATURES].to_numpy(dtype=np.float64)
y_tr = df_tr["abs_impact"].to_numpy(dtype=np.float64)
X_te = df_te[FEATURES].to_numpy(dtype=np.float64)
y_te = df_te["abs_impact"].to_numpy(dtype=np.float64)

model = RandomForestRegressor(random_state=42, n_jobs=-1, **best)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model.fit(X_tr, y_tr)
pred_tr_aapl = np.maximum(model.predict(X_tr), 0.0)
pred_te_aapl = np.maximum(model.predict(X_te), 0.0)
imp = model.feature_importances_

# COIN
df_tr_coin = pd.read_parquet("data/coin_lit_buy_features_train.parquet")
df_te_coin = pd.read_parquet("data/coin_lit_buy_features_test.parquet")
df_tr_coin["abs_impact"] = df_tr_coin["impact_vwap_bps"].abs()
df_te_coin["abs_impact"] = df_te_coin["impact_vwap_bps"].abs()
df_tr_coin = df_tr_coin.sort_values("date").reset_index(drop=True)
X_tr_coin = df_tr_coin[FEATURES].to_numpy(dtype=np.float64)
y_tr_coin = df_tr_coin["abs_impact"].to_numpy(dtype=np.float64)
X_te_coin = df_te_coin[FEATURES].to_numpy(dtype=np.float64)
y_te_coin = df_te_coin["abs_impact"].to_numpy(dtype=np.float64)
rf_coin = RandomForestRegressor(random_state=42, n_jobs=-1, **best)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    rf_coin.fit(X_tr_coin, y_tr_coin)
pred_tr_coin = np.maximum(rf_coin.predict(X_tr_coin), 0.0)
pred_te_coin = np.maximum(rf_coin.predict(X_te_coin), 0.0)

# -- PLOT --
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
    metric_at_vals = [results.loc[results[pname] == v, METRIC_COL].mean() for v in vals]
    x_norm = np.linspace(0, 1, len(vals))
    ax3.plot(x_norm, metric_at_vals, marker="o", color=color, linewidth=1.8,
             markersize=6, label=pname, alpha=0.85)
    for xi, (xn, v) in enumerate(zip(x_norm, vals)):
        ax3.annotate(str(v), (xn, metric_at_vals[xi]),
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
ax4.set_title("AAPL RF Feature Importances", fontsize=11, fontweight="bold")
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
sep = "=" * 46
text = (
    f"HOLDOUT RESULTS (6 features, RF MSE)\n"
    f"{sep}\n\n"
    f"Best hyperparameters (from AAPL grid search):\n"
    f"  max_depth        = {best['max_depth']}\n"
    f"  n_estimators     = {best['n_estimators']}\n"
    f"  min_samples_leaf = {best['min_samples_leaf']}\n"
    f"  max_features     = {best['max_features']}\n"
    f"  bootstrap        = {best['bootstrap']}\n\n"
    f"Best inner-CV MSE:  {best_mse:.4f}\n"
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
print("Saved -> aapl_gridsearch_rf_mse.png")
