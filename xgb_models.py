"""
XGBoost slippage models, SHAP analysis, partial dependence, ICE curves, and a
side-by-side comparison with Random Forest. LAD and MSE variants treated separately
since the feature importance structure differs noticeably between them.
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
import shap
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import partial_dependence


# =
def run_xgb_lad_analysis():
    """
    SHAP, feature importance, and residual analytics for XGBoost LAD model.

    Best params from LAD-scored grid search (6 features, 3-fold CV):
      max_depth=3, n_estimators=200, learning_rate=0.1,
      min_child_weight=1, reg_alpha=0, reg_lambda=1.0

    Output: aapl_xgb_lad_analysis.png
    """
    # Load data
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

    # Train model
    BEST = dict(max_depth=3, n_estimators=200, learning_rate=0.1,
                min_child_weight=1, reg_alpha=0, reg_lambda=1.0)

    model = XGBRegressor(
        objective="reg:absoluteerror", tree_method="hist",
        verbosity=0, random_state=42, n_jobs=1, **BEST,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_tr, y_tr)

    pred_tr = np.maximum(model.predict(X_tr), 0.0)
    pred_te = np.maximum(model.predict(X_te), 0.0)

    # Metrics
    def r2(y, yh):
        ss = ((y - yh)**2).sum(); st = ((y - y.mean())**2).sum()
        return 1 - ss/st if st > 0 else np.nan

    for label, y, pred in [("In-sample", y_tr, pred_tr), ("OOS (Sep)", y_te, pred_te)]:
        print(f"  {label}: R2={r2(y, pred):+.4f}  MAE={np.mean(np.abs(y - pred)):.4f}  "
              f"RMSE={np.sqrt(np.mean((y - pred)**2)):.4f}")

    # SHAP on test set
    rng = np.random.default_rng(42)
    explainer = shap.TreeExplainer(model, X_tr)

    n_shap = min(3000, len(X_te))
    shap_idx = rng.choice(len(X_te), size=n_shap, replace=False)
    X_shap = X_te[shap_idx]

    print("\nComputing SHAP values...", flush=True)
    shap_vals = explainer.shap_values(X_shap)
    base_val = explainer.expected_value

    print(f"SHAP base value: {base_val:.4f}")
    print("Mean |SHAP| per feature:")
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    for feat, ms in sorted(zip(FEATURES, mean_abs_shap), key=lambda x: -x[1]):
        print(f"  {feat:<22} {ms:.5f}")

    # XGBoost native importances
    imp_gain = model.get_booster().get_score(importance_type="gain")
    imp_weight = model.get_booster().get_score(importance_type="weight")
    imp_cover = model.get_booster().get_score(importance_type="cover")

    print("\nXGBoost native importances:")
    print(f"  {'Feature':<22} {'Gain':>10} {'Weight':>10} {'Cover':>10}")
    for i, feat in enumerate(FEATURES):
        key = f"f{i}"
        g = imp_gain.get(key, 0)
        w = imp_weight.get(key, 0)
        c = imp_cover.get(key, 0)
        print(f"  {feat:<22} {g:>10.2f} {w:>10.0f} {c:>10.1f}")

    # PLOTS
    fig = plt.figure(figsize=(26, 18))
    gs = gridspec.GridSpec(3, 3, figure=fig, wspace=0.35, hspace=0.45)

    # Panel 1: SHAP beeswarm
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
    ax1.set_title("AAPL SHAP Beeswarm, XGBoost LAD (Sep 2024 test trades)\n"
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

    # Panel 2: Mean |SHAP| bar
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

    # Panel 3-5: SHAP dependence for top 3
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

    # Panel 6: Predicted vs Actual
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

    # Panel 7: Residual distribution
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

    # Panel 8: MAE by decile
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
        "AAPL lit buy blocks  XGBoost LAD analytics  (6 features, depth=3)\n"
        f"Train: Jun-Aug ({len(df_tr):,})  Test: Sep ({len(df_te):,})  |  "
        f"OOS R2={r2(y_te, pred_te):+.4f}  MAE={np.mean(np.abs(y_te - pred_te)):.4f} bps",
        fontsize=13, fontweight="bold", y=1.01,
    )

    plt.savefig("aapl_xgb_lad_analysis.png", dpi=150, bbox_inches="tight")
    print("\nSaved -> aapl_xgb_lad_analysis.png")


# =
def run_xgb_mse_analysis():
    """
    SHAP, feature importance, and residual analytics for XGBoost MSE model.

    Best params from MSE-scored grid search (6 features, 3-fold CV):
      max_depth=1, n_estimators=200, learning_rate=0.1,
      min_child_weight=1, reg_alpha=10.0, reg_lambda=1.0

    Output: aapl_xgb_mse_analysis.png
    """
    # Load data
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

    # Train model
    BEST = dict(max_depth=1, n_estimators=200, learning_rate=0.1,
                min_child_weight=1, reg_alpha=10.0, reg_lambda=1.0)

    model = XGBRegressor(
        objective="reg:squarederror", tree_method="hist",
        verbosity=0, random_state=42, n_jobs=1, **BEST,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_tr, y_tr)

    pred_tr = np.maximum(model.predict(X_tr), 0.0)
    pred_te = np.maximum(model.predict(X_te), 0.0)

    # Metrics
    def r2(y, yh):
        ss = ((y - yh)**2).sum(); st = ((y - y.mean())**2).sum()
        return 1 - ss/st if st > 0 else np.nan

    for label, y, pred in [("In-sample", y_tr, pred_tr), ("OOS (Sep)", y_te, pred_te)]:
        print(f"  {label}: R2={r2(y,pred):+.4f}  MAE={np.mean(np.abs(y-pred)):.4f}  "
              f"RMSE={np.sqrt(np.mean((y-pred)**2)):.4f}")

    # SHAP on test set
    explainer = shap.TreeExplainer(model, X_tr)

    # Use a subsample for SHAP on test (faster, still representative)
    rng = np.random.default_rng(42)
    n_shap = min(3000, len(X_te))
    shap_idx = rng.choice(len(X_te), size=n_shap, replace=False)
    X_shap = X_te[shap_idx]
    y_shap = y_te[shap_idx]

    shap_vals = explainer.shap_values(X_shap)
    base_val = explainer.expected_value

    print(f"\nSHAP base value: {base_val:.4f}")
    print("Mean |SHAP| per feature:")
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    for feat, ms in sorted(zip(FEATURES, mean_abs_shap), key=lambda x: -x[1]):
        print(f"  {feat:<22} {ms:.5f}")

    # XGBoost native importances
    imp_gain = model.get_booster().get_score(importance_type="gain")
    imp_weight = model.get_booster().get_score(importance_type="weight")
    imp_cover = model.get_booster().get_score(importance_type="cover")

    print("\nXGBoost native importances:")
    print(f"  {'Feature':<22} {'Gain':>10} {'Weight':>10} {'Cover':>10}")
    for i, feat in enumerate(FEATURES):
        key = f"f{i}"
        g = imp_gain.get(key, 0)
        w = imp_weight.get(key, 0)
        c = imp_cover.get(key, 0)
        print(f"  {feat:<22} {g:>10.2f} {w:>10.0f} {c:>10.1f}")

    # PLOTS
    fig = plt.figure(figsize=(26, 18))
    gs = gridspec.GridSpec(3, 3, figure=fig, wspace=0.35, hspace=0.45)

    # Panel 1: SHAP beeswarm
    ax1 = fig.add_subplot(gs[0, :2])
    cmap = cm.coolwarm

    order = np.argsort(mean_abs_shap)  # ascending -> top = most important
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
    ax1.set_title("AAPL SHAP Beeswarm, XGBoost MSE (Sep 2024 test trades)\n"
                  "color = feature value (blue=low, red=high)",
                  fontsize=11, fontweight="bold")
    ax1.grid(axis="x", alpha=0.2)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    sm = cm.ScalarMappable(cmap=cmap, norm=Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1, fraction=0.02, pad=0.02)
    cbar.set_label("Feature value (low -> high)", fontsize=8)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["low", "mid", "high"], fontsize=7.5)

    # Panel 2: Mean |SHAP| bar
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

    # Panel 3-5: SHAP dependence plots for top 3 features
    top3 = bar_order[:3]
    dep_colors = ["#2563eb", "#16a34a", "#dc2626"]

    for k, fi in enumerate(top3):
        ax = fig.add_subplot(gs[1, k])
        # Color by the next most important feature for interaction
        interact_fi = bar_order[1] if fi == bar_order[0] else bar_order[0]
        interact_vals = X_shap[:, interact_fi]
        iv_norm = (interact_vals - interact_vals.min()) / ((interact_vals.max() - interact_vals.min()) + 1e-12)

        sc = ax.scatter(X_shap[:, fi], shap_vals[:, fi],
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

    # Panel 6: Predicted vs Actual scatter
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
                  f"R2={r2(y_te, pred_te):+.4f}  RMSE={np.sqrt(np.mean((y_te-pred_te)**2)):.4f}",
                  fontsize=10, fontweight="bold")
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.18)
    ax6.spines["top"].set_visible(False)
    ax6.spines["right"].set_visible(False)

    # Panel 7: Residual distribution
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

    # Panel 8: MAE by decile of predicted value
    ax8 = fig.add_subplot(gs[2, 2])
    decile_labels = pd.qcut(pred_te, q=10, labels=False, duplicates="drop")
    n_deciles = len(np.unique(decile_labels))
    dec_mae = []
    dec_centers = []
    dec_counts = []
    for d in range(n_deciles):
        mask = decile_labels == d
        dec_mae.append(np.mean(np.abs(y_te[mask] - pred_te[mask])))
        dec_centers.append(np.mean(pred_te[mask]))
        dec_counts.append(mask.sum())

    bars = ax8.bar(range(n_deciles), dec_mae, color="#f59e0b",
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
        "AAPL lit buy blocks  - XGBoost MSE analytics  (6 features, depth=1)\n"
        f"Train: Jun-Aug ({len(df_tr):,})  Test: Sep ({len(df_te):,})  |  "
        f"OOS R2={r2(y_te,pred_te):+.4f}  RMSE={np.sqrt(np.mean((y_te-pred_te)**2)):.4f} bps",
        fontsize=13, fontweight="bold", y=1.01,
    )

    plt.savefig("aapl_xgb_mse_analysis.png", dpi=150, bbox_inches="tight")
    print("\nSaved -> aapl_xgb_mse_analysis.png")


# =
def run_xgb_mse_nonlinear():
    """
    Non-linearity analysis for XGBoost MSE model (6 features, depth=1).

    Plots:
      Row 0: Partial dependence plots (PDP) for all 5 active features
      Row 1: ICE plots (Individual Conditional Expectation) for top 3 features
      Row 2: SHAP interaction values heatmap, residual vs each feature, non-linearity test

    Output: aapl_xgb_mse_nonlinear.png
    """
    # Load & train
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

    BEST = dict(max_depth=1, n_estimators=200, learning_rate=0.1,
                min_child_weight=1, reg_alpha=10.0, reg_lambda=1.0)

    model = XGBRegressor(objective="reg:squarederror", tree_method="hist",
                         verbosity=0, random_state=42, n_jobs=1, **BEST)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_tr, y_tr)

    pred_te = np.maximum(model.predict(X_te), 0.0)
    resid = y_te - pred_te

    # SHAP interaction values (on subsample)
    rng = np.random.default_rng(42)
    n_shap = 1500
    shap_idx = rng.choice(len(X_te), size=n_shap, replace=False)
    X_shap = X_te[shap_idx]

    explainer = shap.TreeExplainer(model)
    print("Computing SHAP interaction values...", flush=True)
    shap_interact = explainer.shap_interaction_values(X_shap)  # (n, 6, 6)
    print(f"  shape: {shap_interact.shape}")

    # Mean absolute interaction matrix
    mean_interact = np.abs(shap_interact).mean(axis=0)  # (6, 6)

    print("\nMean |SHAP interaction| matrix:")
    header = "                     " + "  ".join(f"{f[:8]:>8}" for f in FEATURES)
    print(header)
    for i, feat in enumerate(FEATURES):
        row = "  ".join(f"{mean_interact[i, j]:>8.4f}" for j in range(len(FEATURES)))
        print(f"  {feat:<20} {row}")

    # Active features (skip log_dollar_value which has 0 importance)
    ACTIVE = [0, 2, 3, 4, 5]  # indices of active features
    ACTIVE_NAMES = [FEATURES[i] for i in ACTIVE]

    # PDP helper
    def compute_pdp(model, X_background, feat_idx, grid_n=200):
        """1D partial dependence: average prediction over background while sweeping feat_idx."""
        feat_vals = X_background[:, feat_idx]
        grid = np.linspace(np.percentile(feat_vals, 1), np.percentile(feat_vals, 99), grid_n)
        pdp_vals = np.zeros(grid_n)
        # Use subsample of background for speed
        bg_idx = rng.choice(len(X_background), size=min(2000, len(X_background)), replace=False)
        X_bg = X_background[bg_idx].copy()
        for gi, gval in enumerate(grid):
            X_mod = X_bg.copy()
            X_mod[:, feat_idx] = gval
            pdp_vals[gi] = np.maximum(model.predict(X_mod), 0.0).mean()
        return grid, pdp_vals

    # ICE helper
    def compute_ice(model, X_instances, feat_idx, grid_n=200):
        """ICE curves for a set of instances."""
        feat_vals = X_instances[:, feat_idx]
        grid = np.linspace(np.percentile(feat_vals, 1), np.percentile(feat_vals, 99), grid_n)
        ice = np.zeros((len(X_instances), grid_n))
        for gi, gval in enumerate(grid):
            X_mod = X_instances.copy()
            X_mod[:, feat_idx] = gval
            ice[:, gi] = np.maximum(model.predict(X_mod), 0.0)
        return grid, ice

    # OLS residual test (check for non-linear residual patterns)
    print("\nNon-linearity test: OLS residual vs feature, then fit quadratic")
    print(f"  {'Feature':<22} {'Lin coef':>10} {'Quad coef':>10} {'Quad R2 gain':>13}")
    for fi in ACTIVE:
        feat = X_te[:, fi]
        # Linear fit of residual ~ feature
        c_lin = np.polyfit(feat, resid, 1)
        pred_lin = np.polyval(c_lin, feat)
        ss_lin = ((resid - pred_lin)**2).sum()
        # Quadratic fit
        c_quad = np.polyfit(feat, resid, 2)
        pred_quad = np.polyval(c_quad, feat)
        ss_quad = ((resid - pred_quad)**2).sum()
        ss_tot = ((resid - resid.mean())**2).sum()
        r2_lin = 1 - ss_lin / ss_tot
        r2_quad = 1 - ss_quad / ss_tot
        print(f"  {FEATURES[fi]:<22} {c_lin[0]:>+10.5f} {c_quad[0]:>+10.5f} {r2_quad - r2_lin:>+13.6f}")

    # PLOT
    fig = plt.figure(figsize=(28, 20))
    gs_fig = gridspec.GridSpec(3, 5, figure=fig, wspace=0.38, hspace=0.45)

    # Row 0: PDP for 5 active features
    pdp_colors = ["#2563eb", "#16a34a", "#dc2626", "#7c3aed", "#f59e0b"]
    for k, fi in enumerate(ACTIVE):
        ax = fig.add_subplot(gs_fig[0, k])
        grid, pdp_vals = compute_pdp(model, X_tr, fi)

        # Also scatter test data lightly
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
        # Clip x
        ax.set_xlim(np.percentile(X_te[:, fi], 1), np.percentile(X_te[:, fi], 99))

    # Row 1: ICE for top 3 features + interaction heatmap + residual Q-Q
    top3_fi = [4, 2, 0]  # vol, prate, dollar_value (by SHAP importance)
    ice_colors = ["#7c3aed", "#16a34a", "#2563eb"]

    for k, fi in enumerate(top3_fi):
        ax = fig.add_subplot(gs_fig[1, k])
        # Pick 100 random test instances
        ice_idx = rng.choice(len(X_te), size=100, replace=False)
        grid, ice = compute_ice(model, X_te[ice_idx], fi)

        for row in range(ice.shape[0]):
            ax.plot(grid, ice[row], color=ice_colors[k], alpha=0.08, lw=0.8)
        # PDP overlay
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

    # Panel: SHAP interaction heatmap
    ax_heat = fig.add_subplot(gs_fig[1, 3:])

    # Only show active features in heatmap
    active_interact = mean_interact[np.ix_(ACTIVE, ACTIVE)]
    active_names = [FEATURES[i] for i in ACTIVE]

    im = ax_heat.imshow(active_interact, cmap="YlOrRd", aspect="auto")
    ax_heat.set_xticks(range(len(ACTIVE)))
    ax_heat.set_xticklabels(active_names, fontsize=8, rotation=30, ha="right")
    ax_heat.set_yticks(range(len(ACTIVE)))
    ax_heat.set_yticklabels(active_names, fontsize=8)

    for i in range(len(ACTIVE)):
        for j in range(len(ACTIVE)):
            val = active_interact[i, j]
            color = "white" if val > active_interact.max() * 0.6 else "black"
            ax_heat.text(j, i, f"{val:.4f}", ha="center", va="center",
                         fontsize=8, fontweight="bold", color=color)

    cb = fig.colorbar(im, ax=ax_heat, shrink=0.8, pad=0.03)
    cb.set_label("Mean |SHAP interaction|", fontsize=9)
    ax_heat.set_title("SHAP interaction matrix\n(off-diagonal = pairwise interactions)",
                      fontsize=10, fontweight="bold")

    # Row 2: Residual vs each active feature
    for k, fi in enumerate(ACTIVE):
        ax = fig.add_subplot(gs_fig[2, k])
        samp = rng.choice(len(X_te), size=min(3000, len(X_te)), replace=False)
        feat_samp = X_te[samp, fi]
        resid_samp = resid[samp]

        ax.scatter(feat_samp, resid_samp, alpha=0.06, s=6, color="#64748b",
                  linewidths=0, rasterized=True)
        ax.axhline(0, color="black", lw=1, ls="--", alpha=0.5)

        # LOWESS-like: binned mean residual
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
        "AAPL XGBoost MSE — Non-linearity analysis\n"
        "PDPs, ICE curves, SHAP interactions, residual patterns  |  "
        f"depth=1, 200 trees, 6 features",
        fontsize=14, fontweight="bold", y=1.01,
    )

    plt.savefig("aapl_xgb_mse_nonlinear.png", dpi=150, bbox_inches="tight")
    print("\nSaved -> aapl_xgb_mse_nonlinear.png")


# =
def run_xgb_rf_comparison():
    """
    Side-by-side comparison of XGBoost and Random Forest fits (LAD and MSE variants).

    Trains all 4 models on AAPL lit buy data and produces:
      1. xgb_rf_pred_vs_actual.png   - 2x2 predicted vs actual scatter
      2. xgb_rf_residuals.png        - 2x2 residual distributions + QQ
      3. xgb_rf_partial_dep.png      - Partial dependence for top 3 features
      4. xgb_rf_model_summary.png    - Bar chart comparison of metrics + importance heatmap
    """
    # Load data
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

    # Metrics helper
    def r2(y, yh):
        ss = ((y - yh)**2).sum(); st = ((y - y.mean())**2).sum()
        return 1 - ss/st if st > 0 else np.nan

    def mae(y, yh):
        return np.mean(np.abs(y - yh))

    def rmse(y, yh):
        return np.sqrt(np.mean((y - yh)**2))

    def median_ae(y, yh):
        return np.median(np.abs(y - yh))

    # Train all 4 models
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

    # Figure 1: Predicted vs Actual (2x2)
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

    # Figure 2: Residual Analysis (2x2 grid: histogram + QQ per model pair)
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

    # Figure 3: Partial Dependence Comparison (top 3 features, all 4 models)
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

    # Figure 4: Model Summary Dashboard
    fig4 = plt.figure(figsize=(22, 14))
    gs4 = gridspec.GridSpec(2, 3, figure=fig4, hspace=0.40, wspace=0.35)

    model_names = list(models.keys())
    model_colors = [models[n]["color"] for n in model_names]

    # Panel 1: OOS MAE comparison bar chart
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

    # Panel 2: OOS RMSE comparison bar chart
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

    # Panel 3: OOS R2 comparison bar chart
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

    # Panel 4: Residual QQ plot overlay
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

    # Panel 5: MAE by actual-value quintile (calibration check)
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

    # Panel 6: Feature importance heatmap (sklearn-style)
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


# =
if __name__ == "__main__":
    run_xgb_lad_analysis()
    run_xgb_mse_analysis()
    run_xgb_mse_nonlinear()
    run_xgb_rf_comparison()
