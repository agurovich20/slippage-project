"""
Random Forest models consolidated.

Functions:
  run_rf_holdout()        - RF-MSE and RF-MAE holdout evaluation vs OLS/XGB baselines
  run_rf_lad_analysis()   - SHAP, feature importance, residuals for RF LAD model
  run_rf_mse_analysis()   - SHAP, feature importance, residuals for RF MSE model
  run_rf_mse_nonlinear()  - PDP, ICE, SHAP interactions, non-linearity analysis
  run_regen_rf_plot()     - Regenerate RF grid search plot from saved CSV
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
from numpy.polynomial import polynomial as P
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.inspection import permutation_importance


# =============================================================================
def run_rf_holdout():
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


# =============================================================================
def run_rf_lad_analysis():
    """
    SHAP, feature importance, and residual analytics for Random Forest LAD model.

    Best params from LAD-scored grid search (6 features, 3-fold CV):
      max_depth=20, n_estimators=200, min_samples_leaf=20,
      max_features='sqrt', bootstrap=True, criterion='absolute_error'

    Output: aapl_rf_lad_analysis.png
    """
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
    FEAT_SHORT = ["dollar_val", "log_dollar", "prate", "spread", "vol", "exch_id"]

    X_tr = df_tr[FEATURES].to_numpy(dtype=np.float64)
    y_tr = df_tr["abs_impact"].to_numpy(dtype=np.float64)
    X_te = df_te[FEATURES].to_numpy(dtype=np.float64)
    y_te = df_te["abs_impact"].to_numpy(dtype=np.float64)

    print(f"Train: {len(df_tr):,}  |  Test: {len(df_te):,}")

    # -- Train model --------------------------------------------------------------
    BEST = dict(max_depth=20, n_estimators=200, min_samples_leaf=20,
                max_features="sqrt", bootstrap=True)

    model = RandomForestRegressor(
        criterion="absolute_error", random_state=42, n_jobs=-1, **BEST,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_tr, y_tr)

    pred_tr = np.maximum(model.predict(X_tr), 0.0)
    pred_te = np.maximum(model.predict(X_te), 0.0)

    # -- Metrics ------------------------------------------------------------------
    def r2(y, yh):
        ss = ((y - yh)**2).sum(); st = ((y - y.mean())**2).sum()
        return 1 - ss/st if st > 0 else np.nan

    for label, y, pred in [("In-sample", y_tr, pred_tr), ("OOS (Sep)", y_te, pred_te)]:
        print(f"  {label}: R2={r2(y, pred):+.4f}  MAE={np.mean(np.abs(y - pred)):.4f}  "
              f"RMSE={np.sqrt(np.mean((y - pred)**2)):.4f}")

    # -- SHAP on test set ---------------------------------------------------------
    rng = np.random.default_rng(42)

    n_bg = min(500, len(X_tr))
    bg_idx = rng.choice(len(X_tr), size=n_bg, replace=False)

    print("\nComputing SHAP values (TreeExplainer)...", flush=True)
    explainer = shap.TreeExplainer(model, X_tr[bg_idx])

    n_shap = min(3000, len(X_te))
    shap_idx = rng.choice(len(X_te), size=n_shap, replace=False)
    X_shap = X_te[shap_idx]

    shap_vals = explainer.shap_values(X_shap, check_additivity=False)
    base_val = explainer.expected_value

    print(f"SHAP base value: {base_val:.4f}")
    print("Mean |SHAP| per feature:")
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    for feat, ms in sorted(zip(FEATURES, mean_abs_shap), key=lambda x: -x[1]):
        print(f"  {feat:<22} {ms:.5f}")

    # -- RF native importances ----------------------------------------------------
    imp = model.feature_importances_
    print("\nRF Feature Importances (Mean Decrease in Impurity):")
    for feat, val in sorted(zip(FEATURES, imp), key=lambda x: -x[1]):
        print(f"  {feat:<22} {val:.4f}")

    # -- Permutation importance ----------------------------------------------------
    print("\nComputing permutation importance on test set...", flush=True)
    perm_result = permutation_importance(model, X_te, y_te, n_repeats=10,
                                          random_state=42, scoring="neg_mean_absolute_error",
                                          n_jobs=-1)
    print("Permutation importance (decrease in neg_MAE):")
    for feat, mean_imp, std_imp in sorted(
        zip(FEATURES, perm_result.importances_mean, perm_result.importances_std),
        key=lambda x: -x[1]
    ):
        print(f"  {feat:<22} {mean_imp:.5f} (+/- {std_imp:.5f})")

    # -- PLOTS --------------------------------------------------------------------
    fig = plt.figure(figsize=(26, 18))
    gs = gridspec.GridSpec(3, 3, figure=fig, wspace=0.35, hspace=0.45)

    # -- Panel 1: SHAP beeswarm ---------------------------------------------------
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
    ax1.set_title("AAPL SHAP Beeswarm, Random Forest LAD (Sep 2024 test trades)\n"
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

    # -- Panel 2: Mean |SHAP| bar -------------------------------------------------
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

    # -- Panel 3-5: SHAP dependence for top 3 ------------------------------------
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

    # -- Panel 6: Predicted vs Actual ---------------------------------------------
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

    # -- Panel 7: Residual distribution -------------------------------------------
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

    # -- Panel 8: MAE by decile --------------------------------------------------
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
        "AAPL lit buy blocks  Random Forest LAD analytics  (6 features, depth=20)\n"
        f"Train: Jun-Aug ({len(df_tr):,})  Test: Sep ({len(df_te):,})  |  "
        f"OOS R2={r2(y_te, pred_te):+.4f}  MAE={np.mean(np.abs(y_te - pred_te)):.4f} bps",
        fontsize=13, fontweight="bold", y=1.01,
    )

    plt.savefig("aapl_rf_lad_analysis.png", dpi=150, bbox_inches="tight")
    print("\nSaved -> aapl_rf_lad_analysis.png")


# =============================================================================
def run_rf_mse_analysis():
    """
    SHAP, feature importance, and residual analytics for Random Forest MSE model.

    Best params from MSE-scored grid search (6 features, 3-fold CV):
      max_depth=30, n_estimators=50, min_samples_leaf=20,
      max_features=0.33, bootstrap=False

    Output: aapl_rf_mse_analysis.png
    """
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
    FEAT_SHORT = ["dollar_val", "log_dollar", "prate", "spread", "vol", "exch_id"]

    X_tr = df_tr[FEATURES].to_numpy(dtype=np.float64)
    y_tr = df_tr["abs_impact"].to_numpy(dtype=np.float64)
    X_te = df_te[FEATURES].to_numpy(dtype=np.float64)
    y_te = df_te["abs_impact"].to_numpy(dtype=np.float64)

    print(f"Train: {len(df_tr):,}  |  Test: {len(df_te):,}")

    # -- Train model --------------------------------------------------------------
    BEST = dict(max_depth=30, n_estimators=50, min_samples_leaf=20,
                max_features=0.33, bootstrap=False)

    model = RandomForestRegressor(random_state=42, n_jobs=-1, **BEST)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_tr, y_tr)

    pred_tr = np.maximum(model.predict(X_tr), 0.0)
    pred_te = np.maximum(model.predict(X_te), 0.0)

    # -- Metrics ------------------------------------------------------------------
    def r2(y, yh):
        ss = ((y - yh)**2).sum(); st = ((y - y.mean())**2).sum()
        return 1 - ss/st if st > 0 else np.nan

    for label, y, pred in [("In-sample", y_tr, pred_tr), ("OOS (Sep)", y_te, pred_te)]:
        print(f"  {label}: R2={r2(y, pred):+.4f}  MAE={np.mean(np.abs(y - pred)):.4f}  "
              f"RMSE={np.sqrt(np.mean((y - pred)**2)):.4f}")

    # -- SHAP on test set ---------------------------------------------------------
    rng = np.random.default_rng(42)

    # Use background subsample for TreeExplainer
    n_bg = min(500, len(X_tr))
    bg_idx = rng.choice(len(X_tr), size=n_bg, replace=False)
    X_bg = X_tr[bg_idx]

    print("\nComputing SHAP values (TreeExplainer)...", flush=True)
    explainer = shap.TreeExplainer(model, X_bg)

    n_shap = min(3000, len(X_te))
    shap_idx = rng.choice(len(X_te), size=n_shap, replace=False)
    X_shap = X_te[shap_idx]
    y_shap = y_te[shap_idx]

    shap_vals = explainer.shap_values(X_shap, check_additivity=False)
    base_val = explainer.expected_value

    print(f"SHAP base value: {base_val:.4f}")
    print("Mean |SHAP| per feature:")
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    for feat, ms in sorted(zip(FEATURES, mean_abs_shap), key=lambda x: -x[1]):
        print(f"  {feat:<22} {ms:.5f}")

    # -- RF native importances ----------------------------------------------------
    imp = model.feature_importances_
    print("\nRF Feature Importances (Mean Decrease in Impurity):")
    for feat, val in sorted(zip(FEATURES, imp), key=lambda x: -x[1]):
        print(f"  {feat:<22} {val:.4f}")

    # -- Permutation importance on test set ----------------------------------------
    print("\nComputing permutation importance on test set...", flush=True)
    perm_result = permutation_importance(model, X_te, y_te, n_repeats=10,
                                          random_state=42, scoring="neg_mean_squared_error",
                                          n_jobs=-1)
    print("Permutation importance (decrease in neg_MSE):")
    for feat, mean_imp, std_imp in sorted(
        zip(FEATURES, perm_result.importances_mean, perm_result.importances_std),
        key=lambda x: -x[1]
    ):
        print(f"  {feat:<22} {mean_imp:.5f} (+/- {std_imp:.5f})")

    # -- PLOTS --------------------------------------------------------------------
    fig = plt.figure(figsize=(26, 18))
    gs = gridspec.GridSpec(3, 3, figure=fig, wspace=0.35, hspace=0.45)

    # -- Panel 1: SHAP beeswarm ---------------------------------------------------
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
    ax1.set_title("AAPL SHAP Beeswarm, Random Forest MSE (Sep 2024 test trades)\n"
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

    # -- Panel 2: Mean |SHAP| bar -------------------------------------------------
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

    # -- Panel 3-5: SHAP dependence plots for top 3 features ----------------------
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

    # -- Panel 6: Predicted vs Actual scatter --------------------------------------
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

    # -- Panel 7: Residual distribution -------------------------------------------
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

    # -- Panel 8: MAE by decile of predicted value --------------------------------
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
        "AAPL lit buy blocks  Random Forest MSE analytics  (6 features, depth=30)\n"
        f"Train: Jun-Aug ({len(df_tr):,})  Test: Sep ({len(df_te):,})  |  "
        f"OOS R2={r2(y_te, pred_te):+.4f}  RMSE={np.sqrt(np.mean((y_te - pred_te)**2)):.4f} bps",
        fontsize=13, fontweight="bold", y=1.01,
    )

    plt.savefig("aapl_rf_mse_analysis.png", dpi=150, bbox_inches="tight")
    print("\nSaved -> aapl_rf_mse_analysis.png")


# =============================================================================
def run_rf_mse_nonlinear():
    """
    Non-linearity analysis for Random Forest MSE model (6 features, depth=30).

    Plots:
      Row 0: Partial dependence plots (PDP) for all 6 features
      Row 1: ICE plots (Individual Conditional Expectation) for top 3 features,
             SHAP interaction heatmap (2 cols)
      Row 2: Residual vs each feature (5 most important) + permutation importance bar

    Output: aapl_rf_mse_nonlinear.png
    """
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


# =============================================================================
def run_regen_rf_plot():
    """Regenerate RF grid search plot from saved CSV (no grid search rerun).
    Output: aapl_gridsearch_rf_mse.png
    """
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


# =============================================================================
if __name__ == "__main__":
    run_rf_holdout()
    run_rf_lad_analysis()
    run_rf_mse_analysis()
    run_rf_mse_nonlinear()
    run_regen_rf_plot()
