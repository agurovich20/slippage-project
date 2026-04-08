"""
Side-by-side comparison of XGBoost and Random Forest fits (LAD and MSE variants).

Trains all 4 models on AAPL lit buy data and produces:
  1. xgb_rf_pred_vs_actual.png   - 2x2 predicted vs actual scatter
  2. xgb_rf_residuals.png        - 2x2 residual distributions + QQ
  3. xgb_rf_partial_dep.png      - Partial dependence for top 3 features
  4. xgb_rf_model_summary.png    - Bar chart comparison of metrics + importance heatmap
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
from scipy import stats
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import partial_dependence

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
FEAT_NICE = {
    "dollar_value": "Dollar Value",
    "log_dollar_value": "Log Dollar Value",
    "participation_rate": "Participation Rate",
    "roll_spread_500": "Roll Spread",
    "roll_vol_500": "Roll Volatility",
    "exchange_id": "Exchange ID",
}

X_tr = df_tr[FEATURES].to_numpy(dtype=np.float64)
y_tr = df_tr["abs_impact"].to_numpy(dtype=np.float64)
X_te = df_te[FEATURES].to_numpy(dtype=np.float64)
y_te = df_te["abs_impact"].to_numpy(dtype=np.float64)

print(f"Train: {len(df_tr):,}  |  Test: {len(df_te):,}")

# -- Metrics helper -----------------------------------------------------------
def r2(y, yh):
    ss = ((y - yh)**2).sum(); st = ((y - y.mean())**2).sum()
    return 1 - ss/st if st > 0 else np.nan

def mae(y, yh):
    return np.mean(np.abs(y - yh))

def rmse(y, yh):
    return np.sqrt(np.mean((y - yh)**2))

def median_ae(y, yh):
    return np.median(np.abs(y - yh))

# -- Train all 4 models -------------------------------------------------------
models = {}

# XGBoost LAD
xgb_lad = XGBRegressor(
    objective="reg:absoluteerror", tree_method="hist",
    max_depth=3, n_estimators=200, learning_rate=0.1,
    min_child_weight=1, reg_alpha=0, reg_lambda=1.0,
    verbosity=0, random_state=42, n_jobs=1,
)
# XGBoost MSE
xgb_mse = XGBRegressor(
    objective="reg:squarederror", tree_method="hist",
    max_depth=1, n_estimators=200, learning_rate=0.1,
    min_child_weight=1, reg_alpha=10.0, reg_lambda=1.0,
    verbosity=0, random_state=42, n_jobs=1,
)
# RF LAD
rf_lad = RandomForestRegressor(
    criterion="absolute_error", max_depth=20, n_estimators=200,
    min_samples_leaf=20, max_features="sqrt", bootstrap=True,
    random_state=42, n_jobs=-1,
)
# RF MSE
rf_mse = RandomForestRegressor(
    max_depth=30, n_estimators=50, min_samples_leaf=20,
    max_features=0.33, bootstrap=False,
    random_state=42, n_jobs=-1,
)

specs = [
    ("XGBoost LAD", xgb_lad, "#2563eb"),
    ("XGBoost MSE", xgb_mse, "#7c3aed"),
    ("RF LAD",      rf_lad,   "#dc2626"),
    ("RF MSE",      rf_mse,   "#16a34a"),
]

for name, model, color in specs:
    print(f"Training {name}...", flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_tr, y_tr)
    pred_tr = np.maximum(model.predict(X_tr), 0.0)
    pred_te = np.maximum(model.predict(X_te), 0.0)
    models[name] = {
        "model": model, "color": color,
        "pred_tr": pred_tr, "pred_te": pred_te,
        "r2_tr": r2(y_tr, pred_tr), "r2_te": r2(y_te, pred_te),
        "mae_tr": mae(y_tr, pred_tr), "mae_te": mae(y_te, pred_te),
        "rmse_tr": rmse(y_tr, pred_tr), "rmse_te": rmse(y_te, pred_te),
        "medae_te": median_ae(y_te, pred_te),
        "resid_te": y_te - pred_te,
    }
    print(f"  {name}: OOS R2={models[name]['r2_te']:+.4f}  "
          f"MAE={models[name]['mae_te']:.4f}  RMSE={models[name]['rmse_te']:.4f}")

rng = np.random.default_rng(42)

# =============================================================================
# Figure 1: Predicted vs Actual (2x2)
# =============================================================================
fig1, axes1 = plt.subplots(2, 2, figsize=(16, 14))
clip_v = np.percentile(y_te, 98)
samp = rng.choice(len(y_te), size=min(5000, len(y_te)), replace=False)

for ax, (name, info) in zip(axes1.flat, models.items()):
    pred = info["pred_te"]
    color = info["color"]

    ax.scatter(y_te[samp], np.clip(pred[samp], 0, clip_v),
               alpha=0.10, s=8, color=color, linewidths=0,
               edgecolors="none", rasterized=True)
    ax.axline((0, 0), slope=1, color="black", lw=1.3, ls="--", alpha=0.6)

    # Binned mean trend
    order = np.argsort(y_te)
    chunks = np.array_split(order, 25)
    bin_x = [y_te[c].mean() for c in chunks]
    bin_y = [pred[c].mean() for c in chunks]
    ax.plot(bin_x, bin_y, color="black", lw=2, marker="o", markersize=3,
            zorder=5, label="Binned mean")

    ax.set_xlim(0, clip_v)
    ax.set_ylim(0, clip_v)
    ax.set_aspect("equal")
    ax.set_xlabel("Actual |impact| (bps)", fontsize=10)
    ax.set_ylabel("Predicted (bps)", fontsize=10)

    box = (f"R$^2$ = {info['r2_te']:+.4f}\n"
           f"MAE  = {info['mae_te']:.4f}\n"
           f"RMSE = {info['rmse_te']:.4f}")
    ax.text(0.03, 0.97, box, transform=ax.transAxes, fontsize=9.5,
            family="monospace", va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#94a3b8", alpha=0.92))

    ax.set_title(f"{name}", fontsize=12, fontweight="bold", color=color)
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(True, alpha=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig1.suptitle(
    "AAPL Predicted vs Actual |slippage|: XGBoost vs Random Forest\n"
    f"6 features, Train: Jun-Aug ({len(df_tr):,}), Test: Sep ({len(df_te):,})",
    fontsize=14, fontweight="bold", y=1.01,
)
fig1.tight_layout()
fig1.savefig("xgb_rf_pred_vs_actual.png", dpi=150, bbox_inches="tight")
print("\nSaved -> xgb_rf_pred_vs_actual.png")

# =============================================================================
# Figure 2: Residual Analysis (2x2 grid: histogram + QQ per model pair)
# =============================================================================
fig2 = plt.figure(figsize=(20, 16))
gs2 = gridspec.GridSpec(2, 2, figure=fig2, hspace=0.38, wspace=0.30)

for idx, (name, info) in enumerate(models.items()):
    row, col = divmod(idx, 2)
    ax = fig2.add_subplot(gs2[row, col])
    resid = info["resid_te"]
    color = info["color"]
    clip_r = np.percentile(np.abs(resid), 98)

    # Histogram
    ax.hist(np.clip(resid, -clip_r, clip_r), bins=80, color=color,
            alpha=0.6, edgecolor="white", linewidth=0.3, density=True, zorder=2)

    # Laplace + Normal fits
    loc_l, b_l = stats.laplace.fit(resid)
    mu_n, sig_n = stats.norm.fit(resid)
    xg = np.linspace(-clip_r, clip_r, 300)
    ax.plot(xg, stats.laplace.pdf(xg, loc_l, b_l),
            color="#f59e0b", lw=2, ls=":", zorder=4,
            label=f"Laplace (b={b_l:.3f})")
    ax.plot(xg, stats.norm.pdf(xg, mu_n, sig_n),
            color="black", lw=1.5, ls="--", zorder=4,
            label=f"Normal (sig={sig_n:.3f})")

    ax.axvline(0, color="gray", lw=1, ls="--", alpha=0.5)
    ax.axvline(np.mean(resid), color="#dc2626", lw=1.5, ls="-", alpha=0.8,
               label=f"Mean: {np.mean(resid):.3f}")
    ax.axvline(np.median(resid), color="#16a34a", lw=1.5, ls="--", alpha=0.8,
               label=f"Median: {np.median(resid):.3f}")

    box = (f"Std:  {np.std(resid):.4f}\n"
           f"Skew: {pd.Series(resid).skew():.3f}\n"
           f"Kurt: {pd.Series(resid).kurtosis():.1f}")
    ax.text(0.97, 0.97, box, transform=ax.transAxes, fontsize=9,
            family="monospace", va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#94a3b8", alpha=0.92))

    ax.set_xlabel("Residual (actual - predicted, bps)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(f"{name} Residuals (Sep OOS)", fontsize=12, fontweight="bold", color=color)
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(True, alpha=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig2.suptitle(
    "AAPL Residual Distributions: XGBoost vs Random Forest (Sep 2024 holdout)\n"
    "Laplace and Normal density overlays",
    fontsize=14, fontweight="bold", y=1.01,
)
fig2.savefig("xgb_rf_residuals.png", dpi=150, bbox_inches="tight")
print("Saved -> xgb_rf_residuals.png")

# =============================================================================
# Figure 3: Partial Dependence Comparison (top 3 features, all 4 models)
# =============================================================================
top_features = ["roll_spread_500", "participation_rate", "log_dollar_value"]
top_idx = [FEATURES.index(f) for f in top_features]

fig3, axes3 = plt.subplots(1, 3, figsize=(22, 6.5))

# Wrap X_tr in DataFrame for sklearn partial_dependence
X_tr_df = pd.DataFrame(X_tr, columns=FEATURES)

for ax, feat, fi in zip(axes3, top_features, top_idx):
    for name, info in models.items():
        model = info["model"]
        color = info["color"]

        pdp = partial_dependence(
            model, X_tr_df, features=[feat],
            kind="average", grid_resolution=80,
        )
        grid_vals = pdp["grid_values"][0]
        avg_pred = pdp["average"][0]

        ax.plot(grid_vals, avg_pred, lw=2.2, color=color, label=name, alpha=0.85)

    # Feature distribution rug (light)
    feat_vals = X_tr[:, fi]
    p2, p98 = np.percentile(feat_vals, [2, 98])
    rug_mask = (feat_vals >= p2) & (feat_vals <= p98)
    rug_sample = rng.choice(np.where(rug_mask)[0], size=min(300, rug_mask.sum()), replace=False)
    ax.scatter(feat_vals[rug_sample], np.full(len(rug_sample), ax.get_ylim()[0]),
               marker="|", color="gray", alpha=0.3, s=15, zorder=1)

    ax.set_xlabel(FEAT_NICE.get(feat, feat), fontsize=11)
    ax.set_ylabel("Partial Dependence (bps)", fontsize=11)
    ax.set_title(f"PDP: {FEAT_NICE.get(feat, feat)}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig3.suptitle(
    "AAPL Partial Dependence Comparison: XGBoost vs Random Forest\n"
    "How each model responds to the top 3 features (holding others at training mean)",
    fontsize=13, fontweight="bold", y=1.04,
)
fig3.tight_layout()
fig3.savefig("xgb_rf_partial_dep.png", dpi=150, bbox_inches="tight")
print("Saved -> xgb_rf_partial_dep.png")

# =============================================================================
# Figure 4: Model Summary Dashboard
# =============================================================================
fig4 = plt.figure(figsize=(22, 14))
gs4 = gridspec.GridSpec(2, 3, figure=fig4, hspace=0.40, wspace=0.35)

model_names = list(models.keys())
model_colors = [models[n]["color"] for n in model_names]

# -- Panel 1: OOS MAE comparison bar chart ------------------------------------
ax_mae = fig4.add_subplot(gs4[0, 0])
mae_vals = [models[n]["mae_te"] for n in model_names]
bars = ax_mae.bar(range(4), mae_vals, color=model_colors,
                  edgecolor="white", linewidth=0.8, width=0.6)
for i, v in enumerate(mae_vals):
    ax_mae.text(i, v + 0.003, f"{v:.4f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
ax_mae.set_xticks(range(4))
ax_mae.set_xticklabels(model_names, fontsize=9, rotation=15)
ax_mae.set_ylabel("MAE (bps)", fontsize=11)
ax_mae.set_title("OOS MAE Comparison", fontsize=12, fontweight="bold")
ax_mae.grid(axis="y", alpha=0.2)
ax_mae.spines["top"].set_visible(False)
ax_mae.spines["right"].set_visible(False)

# -- Panel 2: OOS RMSE comparison bar chart -----------------------------------
ax_rmse = fig4.add_subplot(gs4[0, 1])
rmse_vals = [models[n]["rmse_te"] for n in model_names]
bars = ax_rmse.bar(range(4), rmse_vals, color=model_colors,
                   edgecolor="white", linewidth=0.8, width=0.6)
for i, v in enumerate(rmse_vals):
    ax_rmse.text(i, v + 0.003, f"{v:.4f}", ha="center", va="bottom",
                 fontsize=10, fontweight="bold")
ax_rmse.set_xticks(range(4))
ax_rmse.set_xticklabels(model_names, fontsize=9, rotation=15)
ax_rmse.set_ylabel("RMSE (bps)", fontsize=11)
ax_rmse.set_title("OOS RMSE Comparison", fontsize=12, fontweight="bold")
ax_rmse.grid(axis="y", alpha=0.2)
ax_rmse.spines["top"].set_visible(False)
ax_rmse.spines["right"].set_visible(False)

# -- Panel 3: OOS R2 comparison bar chart -------------------------------------
ax_r2 = fig4.add_subplot(gs4[0, 2])
r2_vals = [models[n]["r2_te"] for n in model_names]
bars = ax_r2.bar(range(4), r2_vals, color=model_colors,
                 edgecolor="white", linewidth=0.8, width=0.6)
for i, v in enumerate(r2_vals):
    ax_r2.text(i, v + 0.001, f"{v:+.4f}", ha="center", va="bottom",
               fontsize=10, fontweight="bold")
ax_r2.set_xticks(range(4))
ax_r2.set_xticklabels(model_names, fontsize=9, rotation=15)
ax_r2.set_ylabel("R$^2$", fontsize=11)
ax_r2.set_title("OOS R$^2$ Comparison", fontsize=12, fontweight="bold")
ax_r2.grid(axis="y", alpha=0.2)
ax_r2.spines["top"].set_visible(False)
ax_r2.spines["right"].set_visible(False)

# -- Panel 4: Residual QQ plot overlay ----------------------------------------
ax_qq = fig4.add_subplot(gs4[1, 0])
for name, info in models.items():
    resid = info["resid_te"]
    osm = np.sort(resid)
    n = len(osm)
    theoretical = stats.norm.ppf(np.linspace(0.001, 0.999, n))
    # Subsample for speed
    step = max(1, n // 2000)
    ax_qq.scatter(theoretical[::step], osm[::step], s=6, alpha=0.4,
                  color=info["color"], edgecolors="none", label=name)

lim = max(abs(ax_qq.get_xlim()[0]), abs(ax_qq.get_xlim()[1]))
ax_qq.plot([-lim, lim], [-lim, lim], color="black", lw=1.2, ls="--", alpha=0.5)
ax_qq.set_xlabel("Theoretical Quantiles (Normal)", fontsize=10)
ax_qq.set_ylabel("Sample Quantiles (bps)", fontsize=10)
ax_qq.set_title("QQ Plot: Residuals vs Normal", fontsize=12, fontweight="bold")
ax_qq.legend(fontsize=8.5)
ax_qq.grid(True, alpha=0.18)
ax_qq.spines["top"].set_visible(False)
ax_qq.spines["right"].set_visible(False)

# -- Panel 5: MAE by actual-value quintile (calibration check) ----------------
ax_cal = fig4.add_subplot(gs4[1, 1])
n_bins = 8
bin_edges = np.percentile(y_te, np.linspace(0, 100, n_bins + 1))
bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(n_bins)]
bar_width = 0.18

for j, (name, info) in enumerate(models.items()):
    pred = info["pred_te"]
    bin_mae = []
    for i in range(n_bins):
        mask = (y_te >= bin_edges[i]) & (y_te < bin_edges[i+1])
        if i == n_bins - 1:
            mask = (y_te >= bin_edges[i]) & (y_te <= bin_edges[i+1])
        bin_mae.append(mae(y_te[mask], pred[mask]) if mask.sum() > 0 else 0)
    x_pos = np.arange(n_bins) + j * bar_width
    ax_cal.bar(x_pos, bin_mae, width=bar_width, color=info["color"],
               edgecolor="white", linewidth=0.4, label=name, alpha=0.85)

ax_cal.set_xticks(np.arange(n_bins) + 1.5 * bar_width)
ax_cal.set_xticklabels([f"{c:.2f}" for c in bin_centers], fontsize=7.5, rotation=25)
ax_cal.set_xlabel("Actual |impact| bin center (bps)", fontsize=10)
ax_cal.set_ylabel("MAE within bin (bps)", fontsize=10)
ax_cal.set_title("MAE by Actual Impact Octile\n(where does each model struggle?)",
                 fontsize=11, fontweight="bold")
ax_cal.legend(fontsize=8)
ax_cal.grid(axis="y", alpha=0.2)
ax_cal.spines["top"].set_visible(False)
ax_cal.spines["right"].set_visible(False)

# -- Panel 6: Feature importance heatmap (sklearn-style) ----------------------
ax_imp = fig4.add_subplot(gs4[1, 2])

imp_matrix = np.zeros((len(model_names), len(FEATURES)))
for i, (name, info) in enumerate(models.items()):
    model = info["model"]
    if hasattr(model, "feature_importances_"):
        imp_matrix[i] = model.feature_importances_
    else:
        # XGBoost: use gain importance normalized
        booster = model.get_booster()
        scores = booster.get_score(importance_type="gain")
        total = sum(scores.values()) if scores else 1
        for j in range(len(FEATURES)):
            imp_matrix[i, j] = scores.get(f"f{j}", 0) / total

im = ax_imp.imshow(imp_matrix, aspect="auto", cmap="YlOrRd")
ax_imp.set_xticks(range(len(FEATURES)))
ax_imp.set_xticklabels([FEAT_NICE.get(f, f) for f in FEATURES],
                       fontsize=8.5, rotation=35, ha="right")
ax_imp.set_yticks(range(len(model_names)))
ax_imp.set_yticklabels(model_names, fontsize=9.5)

for i in range(len(model_names)):
    for j in range(len(FEATURES)):
        val = imp_matrix[i, j]
        text_color = "white" if val > imp_matrix.max() * 0.6 else "black"
        ax_imp.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=8.5, color=text_color, fontweight="bold")

cbar = plt.colorbar(im, ax=ax_imp, fraction=0.03, pad=0.04)
cbar.set_label("Feature Importance", fontsize=9)
ax_imp.set_title("Feature Importance Heatmap\n(gain for XGB, impurity for RF)",
                 fontsize=11, fontweight="bold")

fig4.suptitle(
    "AAPL Model Comparison Dashboard: XGBoost vs Random Forest\n"
    f"6 features, Train: Jun-Aug ({len(df_tr):,}), Test: Sep ({len(df_te):,})",
    fontsize=14, fontweight="bold", y=1.01,
)
fig4.savefig("xgb_rf_model_summary.png", dpi=150, bbox_inches="tight")
print("Saved -> xgb_rf_model_summary.png")

print("\nAll visualizations complete!")
