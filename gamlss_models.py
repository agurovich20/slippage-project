"""
Consolidated GAMLSS models for block trade impact prediction.

Functions:
  run_gamlss_laplace()       - Linear LAD location + OLS scale, Laplace intervals
  run_gamlss_xgb()           - XGBoost GAMLSS with RS iterations
  run_gamlss_xgb_gengauss()  - Generalized Gaussian extension of XGB GAMLSS
  run_gamlss_full()          - Full GAMLSS with P-splines and RS algorithm
  run_rerun_gamlss()         - XGB GAMLSS validated across 6 stocks
  run_pooled_gamlss()        - Pooled XGB GAMLSS (median-normalized, 6 stocks)
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize_scalar
from scipy.special import gamma as gammafn
from scipy.stats import gennorm, chi2
from scipy.interpolate import BSpline
from scipy.sparse import issparse
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit


# =============================================================================
# run_gamlss_laplace
# =============================================================================

def run_gamlss_laplace():
    """
    GAMLSS-style Laplace distributional regression for block trade impact.

    For each trade i:  impact ~ Laplace(mu_i, b_i)
      - mu_i = location (conditional median), fitted via LAD (QuantReg tau=0.5)
      - b_i  = scale (dispersion), fitted via OLS on |residuals| from Stage 1

    Stages:
      1. Fit location model (LAD) on training data
      2. Fit scale model (OLS on |residuals|) on training data
      3. Compute prediction intervals on test set, check coverage
      4. Repeat for COIN
      5. Print results
      6. Create 2x4 figure

    Output:
      - aapl_coin_gamlss.png
      - data/gamlss_results.csv
    """

    FEATURES = [
        "dollar_value", "log_dollar_value", "participation_rate",
        "roll_spread_500", "roll_vol_500", "exchange_id",
    ]

    # -- Helpers ------------------------------------------------------------------

    def fit_location_model(X_tr, y_tr, X_te):
        """LAD regression (QuantReg tau=0.5) for conditional median."""
        X_tr_c = sm.add_constant(X_tr)
        X_te_c = sm.add_constant(X_te)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            qr = sm.QuantReg(y_tr, X_tr_c).fit(q=0.5, max_iter=5000)
        mu_hat_tr = qr.predict(X_tr_c)
        mu_hat_te = qr.predict(X_te_c)
        return qr, mu_hat_tr, mu_hat_te


    def fit_scale_model(X_tr, abs_resid_tr, X_te):
        """OLS on |residuals| to predict Laplace scale b."""
        X_tr_c = np.column_stack([np.ones(len(X_tr)), X_tr])
        X_te_c = np.column_stack([np.ones(len(X_te)), X_te])
        gamma, _, _, _ = np.linalg.lstsq(X_tr_c, abs_resid_tr, rcond=None)
        b_hat_tr = np.clip(X_tr_c @ gamma, 0.1, None)
        b_hat_te = np.clip(X_te_c @ gamma, 0.1, None)
        return gamma, b_hat_tr, b_hat_te


    def compute_coverage(y, mu_hat, b_hat, level):
        """Fraction of actuals within the Laplace prediction interval at given level."""
        # For Laplace, quantile at p is mu + b * sign(p-0.5) * ln(1 / (1 - 2*|p-0.5|))
        # Symmetric interval at level alpha: mu +/- b * ln(1 / (1 - alpha))
        z = np.log(1.0 / (1.0 - level))
        lo = np.maximum(mu_hat - z * b_hat, 0.0)
        hi = mu_hat + z * b_hat
        covered = ((y >= lo) & (y <= hi)).mean()
        width = (hi - lo).mean()
        return covered, width, lo, hi


    # -- Process both tickers ----------------------------------------------------

    TICKERS = [
        ("AAPL", "data/lit_buy_features_v2.parquet", "data/lit_buy_features_v2_sep.parquet"),
        ("COIN", "data/coin_lit_buy_features_train.parquet", "data/coin_lit_buy_features_test.parquet"),
    ]

    all_results = {}
    csv_rows = []

    for ticker, tr_file, te_file in TICKERS:
        print(f"\n{'=' * 80}")
        print(f"  {ticker}")
        print(f"{'=' * 80}")

        df_tr = pd.read_parquet(tr_file)
        df_te = pd.read_parquet(te_file)
        df_tr["abs_impact"] = df_tr["impact_vwap_bps"].abs()
        df_te["abs_impact"] = df_te["impact_vwap_bps"].abs()
        df_tr = df_tr.sort_values("date").reset_index(drop=True)

        X_tr = df_tr[FEATURES].to_numpy(dtype=np.float64)
        y_tr = df_tr["abs_impact"].to_numpy(dtype=np.float64)
        X_te = df_te[FEATURES].to_numpy(dtype=np.float64)
        y_te = df_te["abs_impact"].to_numpy(dtype=np.float64)

        print(f"  Train: {len(df_tr):,}  |  Test: {len(df_te):,}")

        # Stage 1: Location model
        qr_model, mu_hat_tr, mu_hat_te = fit_location_model(X_tr, y_tr, X_te)

        print(f"\n  STAGE 1: Location model (LAD, tau=0.5)")
        print(f"  {'Feature':<22} {'Coef':>12}")
        coef_names = ["intercept"] + FEATURES
        for name, coef in zip(coef_names, qr_model.params):
            print(f"  {name:<22} {coef:>+12.6f}")

        mae_tr = np.mean(np.abs(y_tr - mu_hat_tr))
        mae_te = np.mean(np.abs(y_te - mu_hat_te))
        print(f"  Train MAE: {mae_tr:.4f}  |  Test MAE: {mae_te:.4f}")

        # Stage 2: Scale model
        abs_resid_tr = np.abs(y_tr - mu_hat_tr)
        gamma, b_hat_tr, b_hat_te = fit_scale_model(X_tr, abs_resid_tr, X_te)

        print(f"\n  STAGE 2: Scale model (OLS on |residuals|)")
        print(f"  {'Feature':<22} {'Coef':>12}")
        for name, coef in zip(coef_names, gamma):
            print(f"  {name:<22} {coef:>+12.6f}")

        print(f"  Train b_hat: mean={b_hat_tr.mean():.4f}, median={np.median(b_hat_tr):.4f}")
        print(f"  Test  b_hat: mean={b_hat_te.mean():.4f}, median={np.median(b_hat_te):.4f}")

        # Stage 3: Prediction intervals
        print(f"\n  STAGE 3: Coverage on test set")
        print(f"  {'Level':>8} {'Coverage':>10} {'Mean Width':>12}")
        coverage_data = {}
        for level in [0.50, 0.80, 0.90]:
            cov, width, lo, hi = compute_coverage(y_te, mu_hat_te, b_hat_te, level)
            coverage_data[level] = (cov, width)
            print(f"  {level:>8.0%} {cov:>10.4f} {width:>12.4f}")
            csv_rows.append({
                "ticker": ticker,
                "level": level,
                "nominal_coverage": level,
                "actual_coverage": round(cov, 6),
                "mean_interval_width": round(width, 4),
                "test_mae": round(mae_te, 4),
                "n_train": len(df_tr),
                "n_test": len(df_te),
            })

        # Full calibration curve (for plotting)
        cal_levels = np.linspace(0.05, 0.99, 50)
        cal_actual = []
        for lv in cal_levels:
            c, _, _, _ = compute_coverage(y_te, mu_hat_te, b_hat_te, lv)
            cal_actual.append(c)
        cal_actual = np.array(cal_actual)

        # Store for plotting
        all_results[ticker] = {
            "y_te": y_te, "mu_hat_te": mu_hat_te, "b_hat_te": b_hat_te,
            "X_te": X_te, "abs_resid_te": np.abs(y_te - mu_hat_te),
            "cal_levels": cal_levels, "cal_actual": cal_actual,
            "coverage_data": coverage_data,
            "mae_te": mae_te,
        }

    # -- Save CSV -----------------------------------------------------------------
    csv_df = pd.DataFrame(csv_rows)
    csv_df.to_csv("data/gamlss_results.csv", index=False)
    print(f"\nSaved -> data/gamlss_results.csv")

    # -- PLOTS (2 rows x 4 cols) -------------------------------------------------
    fig = plt.figure(figsize=(28, 12))
    gs_fig = gridspec.GridSpec(2, 4, figure=fig, wspace=0.30, hspace=0.40)

    rng = np.random.default_rng(42)

    for row_idx, (ticker, _, _) in enumerate(TICKERS):
        r = all_results[ticker]
        y_te = r["y_te"]
        mu_hat = r["mu_hat_te"]
        b_hat = r["b_hat_te"]
        X_te = r["X_te"]
        abs_resid = r["abs_resid_te"]

        # -- Col 1: Prediction intervals for 200 random trades --------------------
        ax1 = fig.add_subplot(gs_fig[row_idx, 0])

        n_show = min(200, len(y_te))
        show_idx = rng.choice(len(y_te), size=n_show, replace=False)
        # Sort by mu_hat
        sort_order = np.argsort(mu_hat[show_idx])
        show_idx_sorted = show_idx[sort_order]

        x_pos = np.arange(n_show)
        mu_s = mu_hat[show_idx_sorted]
        b_s = b_hat[show_idx_sorted]
        y_s = y_te[show_idx_sorted]

        z90 = np.log(1.0 / (1.0 - 0.90))
        lo90 = np.maximum(mu_s - z90 * b_s, 0.0)
        hi90 = mu_s + z90 * b_s

        ax1.fill_between(x_pos, lo90, hi90, alpha=0.25, color="#2563eb", label="90% interval")
        ax1.plot(x_pos, mu_s, color="#dc2626", lw=1.5, label="Predicted median", zorder=3)
        ax1.scatter(x_pos, y_s, s=12, color="black", alpha=0.5, zorder=4, label="Actual impact")
        ax1.set_xlabel("Trade index (sorted by predicted median)", fontsize=9)
        ax1.set_ylabel("|impact_vwap_bps|", fontsize=9)
        ax1.set_title(f"{ticker} Prediction Intervals (90%)\n200 random test trades",
                      fontsize=10, fontweight="bold")
        ax1.legend(fontsize=7.5, loc="upper left")
        ax1.grid(True, alpha=0.15)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

        # -- Col 2: Calibration plot -----------------------------------------------
        ax2 = fig.add_subplot(gs_fig[row_idx, 1])

        ax2.plot(r["cal_levels"], r["cal_actual"], color="#2563eb", lw=2.2,
                 marker="o", markersize=3, label="Laplace model")
        ax2.plot([0, 1], [0, 1], color="black", lw=1.2, ls="--", alpha=0.6,
                 label="Perfect calibration")

        # Annotate key levels
        for lv in [0.50, 0.80, 0.90]:
            cov = r["coverage_data"][lv][0]
            ax2.annotate(f"{lv:.0%}: {cov:.1%}", xy=(lv, cov),
                         textcoords="offset points", xytext=(10, -10),
                         fontsize=7.5, color="#dc2626", fontweight="bold",
                         arrowprops=dict(arrowstyle="->", color="#dc2626", lw=0.8))

        ax2.set_xlabel("Nominal coverage level", fontsize=9)
        ax2.set_ylabel("Actual coverage", fontsize=9)
        ax2.set_title(f"{ticker} Calibration Plot\n(Laplace distributional model)",
                      fontsize=10, fontweight="bold")
        ax2.legend(fontsize=8)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.set_aspect("equal")
        ax2.grid(True, alpha=0.15)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        # -- Col 3: Binned scale validation ----------------------------------------
        ax3 = fig.add_subplot(gs_fig[row_idx, 2])

        n_bins = 20
        bin_labels = pd.qcut(b_hat, q=n_bins, labels=False, duplicates="drop")
        n_actual_bins = len(np.unique(bin_labels))
        bin_pred_mean = []
        bin_actual_mean = []
        for bi in range(n_actual_bins):
            mask = bin_labels == bi
            if mask.sum() > 0:
                bin_pred_mean.append(b_hat[mask].mean())
                bin_actual_mean.append(abs_resid[mask].mean())

        ax3.scatter(bin_pred_mean, bin_actual_mean, s=50, color="#16a34a",
                    edgecolors="white", linewidth=0.8, zorder=3)
        lim_max = max(max(bin_pred_mean), max(bin_actual_mean)) * 1.1
        ax3.plot([0, lim_max], [0, lim_max], color="black", lw=1.2, ls="--", alpha=0.6,
                 label="Perfect scale calibration")
        ax3.set_xlabel("Mean predicted scale (b_hat)", fontsize=9)
        ax3.set_ylabel("Mean actual |residual|", fontsize=9)
        ax3.set_title(f"{ticker} Scale Calibration\n(20 quantile bins by b_hat)",
                      fontsize=10, fontweight="bold")
        ax3.legend(fontsize=8)
        ax3.set_xlim(0, lim_max)
        ax3.set_ylim(0, lim_max)
        ax3.set_aspect("equal")
        ax3.grid(True, alpha=0.15)
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)

        # -- Col 4: Interval width vs roll_spread_500 -----------------------------
        ax4 = fig.add_subplot(gs_fig[row_idx, 3])

        spread_idx = FEATURES.index("roll_spread_500")
        spread_vals = X_te[:, spread_idx]
        width_90 = 2 * z90 * b_hat  # total interval width

        samp = rng.choice(len(spread_vals), size=min(3000, len(spread_vals)), replace=False)
        ax4.scatter(spread_vals[samp], width_90[samp], s=8, alpha=0.15,
                    color="#7c3aed", linewidths=0, rasterized=True)

        # Binned mean overlay
        n_bins_sp = 25
        sp_bins = np.linspace(np.percentile(spread_vals, 2), np.percentile(spread_vals, 98), n_bins_sp + 1)
        bin_centers_sp = []
        bin_width_sp = []
        for bi in range(n_bins_sp):
            mask = (spread_vals >= sp_bins[bi]) & (spread_vals < sp_bins[bi + 1])
            if mask.sum() > 5:
                bin_centers_sp.append((sp_bins[bi] + sp_bins[bi + 1]) / 2)
                bin_width_sp.append(width_90[mask].mean())

        ax4.plot(bin_centers_sp, bin_width_sp, color="#dc2626", lw=2.2, marker="o",
                 markersize=4, label="Binned mean width", zorder=5)
        ax4.set_xlabel("roll_spread_500 (bps)", fontsize=9)
        ax4.set_ylabel("90% interval width (bps)", fontsize=9)
        ax4.set_title(f"{ticker} Interval Width vs Spread\n(heteroscedasticity captured)",
                      fontsize=10, fontweight="bold")
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.15)
        ax4.spines["top"].set_visible(False)
        ax4.spines["right"].set_visible(False)
        ax4.set_xlim(sp_bins[0], sp_bins[-1])

    fig.suptitle(
        "GAMLSS Laplace Distributional Regression: Prediction Intervals for Block Trade Impact\n"
        "Location = LAD (QuantReg median)  |  Scale = OLS on |residuals|  |  6 features",
        fontsize=14, fontweight="bold", y=1.02,
    )

    plt.savefig("aapl_coin_gamlss.png", dpi=150, bbox_inches="tight")
    print("Saved -> aapl_coin_gamlss.png")


# =============================================================================
# run_gamlss_xgb
# =============================================================================

def run_gamlss_xgb():
    """
    GAMLSS-style XGBoost distributional regression for block trade impact.

    For each trade i:  impact ~ Laplace(mu_i, b_i)
      - mu_i = XGBoost LAD location model (conditional median)
      - b_i  = XGBoost MSE scale model on |residuals| from Stage 1

    Stages:
      1. Fit XGB location model (LAD) on training data
      2. Fit XGB scale model (grid-searched) on |residuals|
      3. Compute prediction intervals on test set, check coverage
      4. Repeat for COIN
      5. Comparison table (with linear GAMLSS side by side)
      6. Create 2x4 figure

    Output:
      - aapl_coin_gamlss_xgb.png
      - data/gamlss_xgb_results.csv
    """

    FEATURES = [
        "dollar_value", "log_dollar_value", "participation_rate",
        "roll_spread_500", "roll_vol_500", "exchange_id",
    ]

    LOC_PARAMS = dict(
        max_depth=3, n_estimators=200, learning_rate=0.07,
        min_child_weight=5, reg_alpha=10, reg_lambda=10,
    )

    SCALE_GRID = {
        "max_depth": [2, 3, 5],
        "n_estimators": [50, 100],
        "min_child_weight": [10, 20],
    }

    # -- Helpers ------------------------------------------------------------------

    def fit_location_model(X_tr, y_tr, X_te):
        """XGBoost LAD for conditional median."""
        model = XGBRegressor(
            objective="reg:absoluteerror", tree_method="hist",
            verbosity=0, random_state=42, n_jobs=1, **LOC_PARAMS,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_tr, y_tr)
        mu_hat_tr = np.maximum(model.predict(X_tr), 0.0)
        mu_hat_te = np.maximum(model.predict(X_te), 0.0)
        return model, mu_hat_tr, mu_hat_te


    def fit_scale_model(X_tr, abs_resid_tr, X_te):
        """Grid-searched XGBoost MSE for Laplace scale b."""
        base = XGBRegressor(
            objective="reg:squarederror", tree_method="hist",
            verbosity=0, random_state=42, n_jobs=1,
            learning_rate=0.1, reg_alpha=1, reg_lambda=1,
        )
        tscv = TimeSeriesSplit(n_splits=3)
        gs = GridSearchCV(
            base, SCALE_GRID, cv=tscv,
            scoring="neg_mean_absolute_error",
            refit=True, n_jobs=1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gs.fit(X_tr, abs_resid_tr)

        best = gs.best_estimator_
        b_hat_tr = np.clip(best.predict(X_tr), 0.1, None)
        b_hat_te = np.clip(best.predict(X_te), 0.1, None)
        return best, gs.best_params_, b_hat_tr, b_hat_te


    def compute_coverage(y, mu_hat, b_hat, level):
        """Fraction of actuals within the Laplace prediction interval at given level."""
        z = np.log(1.0 / (1.0 - level))
        lo = np.maximum(mu_hat - z * b_hat, 0.0)
        hi = mu_hat + z * b_hat
        covered = ((y >= lo) & (y <= hi)).mean()
        width = (hi - lo).mean()
        return covered, width, lo, hi


    # -- Process both tickers ----------------------------------------------------

    TICKERS = [
        ("AAPL", "data/lit_buy_features_v2.parquet", "data/lit_buy_features_v2_sep.parquet"),
        ("COIN", "data/coin_lit_buy_features_train.parquet", "data/coin_lit_buy_features_test.parquet"),
    ]

    all_results = {}
    csv_rows = []

    for ticker, tr_file, te_file in TICKERS:
        print(f"\n{'=' * 80}")
        print(f"  {ticker}")
        print(f"{'=' * 80}")

        df_tr = pd.read_parquet(tr_file)
        df_te = pd.read_parquet(te_file)
        df_tr["abs_impact"] = df_tr["impact_vwap_bps"].abs()
        df_te["abs_impact"] = df_te["impact_vwap_bps"].abs()
        df_tr = df_tr.sort_values("date").reset_index(drop=True)

        X_tr = df_tr[FEATURES].to_numpy(dtype=np.float64)
        y_tr = df_tr["abs_impact"].to_numpy(dtype=np.float64)
        X_te = df_te[FEATURES].to_numpy(dtype=np.float64)
        y_te = df_te["abs_impact"].to_numpy(dtype=np.float64)

        print(f"  Train: {len(df_tr):,}  |  Test: {len(df_te):,}")

        # Stage 1: Location model
        loc_model, mu_hat_tr, mu_hat_te = fit_location_model(X_tr, y_tr, X_te)

        mae_tr = np.mean(np.abs(y_tr - mu_hat_tr))
        mae_te = np.mean(np.abs(y_te - mu_hat_te))
        print(f"\n  STAGE 1: XGB Location (LAD)")
        print(f"  Train MAE: {mae_tr:.4f}  |  Test MAE: {mae_te:.4f}")

        # Feature importance (gain)
        imp = loc_model.get_booster().get_score(importance_type="gain")
        print(f"  Location feature importance (gain):")
        for i, feat in enumerate(FEATURES):
            key = f"f{i}"
            print(f"    {feat:<22} {imp.get(key, 0):.2f}")

        # Stage 2: Scale model
        abs_resid_tr = np.abs(y_tr - mu_hat_tr)
        scale_model, best_scale_params, b_hat_tr, b_hat_te = fit_scale_model(
            X_tr, abs_resid_tr, X_te)

        print(f"\n  STAGE 2: XGB Scale (grid-searched)")
        print(f"  Best params: {best_scale_params}")
        print(f"  Train b_hat: mean={b_hat_tr.mean():.4f}, median={np.median(b_hat_tr):.4f}")
        print(f"  Test  b_hat: mean={b_hat_te.mean():.4f}, median={np.median(b_hat_te):.4f}")

        # Scale feature importance
        imp_s = scale_model.get_booster().get_score(importance_type="gain")
        print(f"  Scale feature importance (gain):")
        for i, feat in enumerate(FEATURES):
            key = f"f{i}"
            print(f"    {feat:<22} {imp_s.get(key, 0):.2f}")

        # -- Save two-stage (iteration 0) results for comparison -------------------
        mae_te_iter0 = mae_te
        mu_hat_te_iter0 = mu_hat_te.copy()
        b_hat_te_iter0 = b_hat_te.copy()

        def laplace_loglik(y, mu, b):
            """Total Laplace log-likelihood: sum(-log(2*b) - |y - mu| / b)."""
            return np.sum(-np.log(2.0 * b) - np.abs(y - mu) / b)

        ll_init = laplace_loglik(y_tr, mu_hat_tr, b_hat_tr)
        ll_history = [ll_init]
        mae_te_history = [mae_te]
        print(f"\n  RS ITERATION (up to 5 iterations)")
        print(f"  Iter 0 (two-stage): log-lik = {ll_init:,.2f}")

        prev_ll = ll_init
        for rs_iter in range(1, 6):
            # (1) Compute weights w_i = 1 / b_hat_i
            w_i = 1.0 / b_hat_tr

            # (2) Refit location model with sample_weight = w_i
            loc_rs = XGBRegressor(
                objective="reg:absoluteerror", tree_method="hist",
                verbosity=0, random_state=42, n_jobs=1, **LOC_PARAMS,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                loc_rs.fit(X_tr, y_tr, sample_weight=w_i)
            mu_hat_tr = np.maximum(loc_rs.predict(X_tr), 0.0)
            mu_hat_te = np.maximum(loc_rs.predict(X_te), 0.0)

            # (3) Recompute absolute residuals
            abs_resid_tr = np.abs(y_tr - mu_hat_tr)

            # (4) Refit scale model on new residuals
            scale_model, best_scale_params, b_hat_tr, b_hat_te = fit_scale_model(
                X_tr, abs_resid_tr, X_te)

            # (5) Compute Laplace log-likelihood on training data
            ll = laplace_loglik(y_tr, mu_hat_tr, b_hat_tr)
            ll_history.append(ll)
            mae_te_history.append(np.mean(np.abs(y_te - mu_hat_te)))
            pct_change = abs(ll - prev_ll) / abs(prev_ll) * 100
            print(f"  Iter {rs_iter}: log-lik = {ll:,.2f}  (delta = {pct_change:.4f}%)")

            if pct_change < 0.01:
                print(f"  Converged at iteration {rs_iter} (delta < 0.01%)")
                break
            prev_ll = ll

        mae_te_rs = np.mean(np.abs(y_te - mu_hat_te))
        print(f"\n  RS converged: Test MAE = {mae_te_rs:.4f}")

        # -- Save RS (iteration 0) test metrics for comparison table ---------------
        iter0_coverage = {}
        for level in [0.50, 0.80, 0.90]:
            cov0, width0, _, _ = compute_coverage(y_te, mu_hat_te_iter0, b_hat_te_iter0, level)
            iter0_coverage[level] = (cov0, width0)

        rs_coverage = {}
        for level in [0.50, 0.80, 0.90]:
            cov_rs, width_rs, _, _ = compute_coverage(y_te, mu_hat_te, b_hat_te, level)
            rs_coverage[level] = (cov_rs, width_rs)

        all_results[ticker + "_iter0"] = {
            "mae_te": mae_te_iter0,
            "coverage_data": iter0_coverage,
        }
        all_results[ticker + "_rs"] = {
            "mae_te": mae_te_rs,
            "coverage_data": rs_coverage,
            "ll_history": ll_history,
            "mae_te_history": mae_te_history,
        }

        # Stage 3: Prediction intervals (using RS-converged model)
        print(f"\n  STAGE 3: Coverage on test set (RS-converged)")
        print(f"  {'Level':>8} {'Coverage':>10} {'Mean Width':>12}")
        coverage_data = rs_coverage
        for level in [0.50, 0.80, 0.90]:
            cov, width = coverage_data[level]
            print(f"  {level:>8.0%} {cov:>10.4f} {width:>12.4f}")
            csv_rows.append({
                "ticker": ticker,
                "model": "XGB_GAMLSS_RS",
                "level": level,
                "nominal_coverage": level,
                "actual_coverage": round(cov, 6),
                "mean_interval_width": round(width, 4),
                "test_mae": round(mae_te_rs, 4),
                "n_train": len(df_tr),
                "n_test": len(df_te),
            })

        # Full calibration curve (RS-converged)
        cal_levels = np.linspace(0.05, 0.99, 50)
        cal_actual = []
        for lv in cal_levels:
            c, _, _, _ = compute_coverage(y_te, mu_hat_te, b_hat_te, lv)
            cal_actual.append(c)
        cal_actual = np.array(cal_actual)

        all_results[ticker] = {
            "y_te": y_te, "mu_hat_te": mu_hat_te, "b_hat_te": b_hat_te,
            "X_te": X_te, "abs_resid_te": np.abs(y_te - mu_hat_te),
            "cal_levels": cal_levels, "cal_actual": cal_actual,
            "coverage_data": coverage_data,
            "mae_te": mae_te_rs,
        }

    # -- Save CSV -----------------------------------------------------------------
    csv_df = pd.DataFrame(csv_rows)
    csv_df.to_csv("data/gamlss_xgb_results.csv", index=False)
    print(f"\nSaved -> data/gamlss_xgb_results.csv")

    # -- RS Comparison table: Two-Stage vs Converged RS ----------------------------
    print(f"\n{'=' * 90}")
    print("  COMPARISON: Two-Stage (Iter 0) vs RS-Converged (Final Iter)")
    print(f"{'=' * 90}")
    print(f"  {'Model':<30} {'MAE':>8} {'90% Cov':>8} {'80% Cov':>8} {'50% Cov':>8} {'90% Width':>10}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")

    for ticker in ["AAPL", "COIN"]:
        r0 = all_results[ticker + "_iter0"]
        cd0 = r0["coverage_data"]
        print(f"  {ticker + ' Two-Stage':<30} {r0['mae_te']:>8.4f} {cd0[0.90][0]:>8.4f} "
              f"{cd0[0.80][0]:>8.4f} {cd0[0.50][0]:>8.4f} {cd0[0.90][1]:>10.4f}")

        rrs = all_results[ticker + "_rs"]
        cdrs = rrs["coverage_data"]
        print(f"  {ticker + ' RS-Converged':<30} {rrs['mae_te']:>8.4f} {cdrs[0.90][0]:>8.4f} "
              f"{cdrs[0.80][0]:>8.4f} {cdrs[0.50][0]:>8.4f} {cdrs[0.90][1]:>10.4f}")

    print(f"\n  Nominal targets:  90% -> 0.9000   80% -> 0.8000   50% -> 0.5000")

    # -- Stage 5: Comparison table with linear GAMLSS ----------------------------
    print(f"\n{'=' * 80}")
    print("  COMPARISON: XGB GAMLSS vs Linear GAMLSS")
    print(f"{'=' * 80}")

    # Load linear GAMLSS results
    try:
        lin_df = pd.read_csv("data/gamlss_results.csv")
        has_linear = True
    except FileNotFoundError:
        has_linear = False

    print(f"\n  {'Model':<28} {'90% Cov':>8} {'80% Cov':>8} {'50% Cov':>8} "
          f"{'MAE':>8} {'90% Width':>10}")
    print(f"  {'-'*28} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")

    for ticker in ["AAPL", "COIN"]:
        r = all_results[ticker]
        cd = r["coverage_data"]
        print(f"  {ticker + ' XGB GAMLSS':<28} {cd[0.90][0]:>8.4f} {cd[0.80][0]:>8.4f} "
              f"{cd[0.50][0]:>8.4f} {r['mae_te']:>8.4f} {cd[0.90][1]:>10.4f}")

        if has_linear:
            lr = lin_df[(lin_df["ticker"] == ticker) & (lin_df["level"] == 0.90)]
            lr80 = lin_df[(lin_df["ticker"] == ticker) & (lin_df["level"] == 0.80)]
            lr50 = lin_df[(lin_df["ticker"] == ticker) & (lin_df["level"] == 0.50)]
            if len(lr) > 0:
                print(f"  {ticker + ' Linear GAMLSS':<28} {lr['actual_coverage'].values[0]:>8.4f} "
                      f"{lr80['actual_coverage'].values[0]:>8.4f} "
                      f"{lr50['actual_coverage'].values[0]:>8.4f} "
                      f"{lr['test_mae'].values[0]:>8.4f} "
                      f"{lr['mean_interval_width'].values[0]:>10.4f}")

    # -- Stage 6: Plots (2 rows x 4 cols) ----------------------------------------
    fig = plt.figure(figsize=(28, 12))
    gs_fig = gridspec.GridSpec(2, 4, figure=fig, wspace=0.30, hspace=0.40)

    rng = np.random.default_rng(42)

    for row_idx, (ticker, _, _) in enumerate(TICKERS):
        r = all_results[ticker]
        y_te = r["y_te"]
        mu_hat = r["mu_hat_te"]
        b_hat = r["b_hat_te"]
        X_te = r["X_te"]
        abs_resid = r["abs_resid_te"]

        # -- Col 1: Prediction intervals for 200 random trades --------------------
        ax1 = fig.add_subplot(gs_fig[row_idx, 0])

        n_show = min(200, len(y_te))
        show_idx = rng.choice(len(y_te), size=n_show, replace=False)
        sort_order = np.argsort(mu_hat[show_idx])
        show_idx_sorted = show_idx[sort_order]

        x_pos = np.arange(n_show)
        mu_s = mu_hat[show_idx_sorted]
        b_s = b_hat[show_idx_sorted]
        y_s = y_te[show_idx_sorted]

        z90 = np.log(1.0 / (1.0 - 0.90))
        lo90 = np.maximum(mu_s - z90 * b_s, 0.0)
        hi90 = mu_s + z90 * b_s

        ax1.fill_between(x_pos, lo90, hi90, alpha=0.25, color="#2563eb", label="90% interval")
        ax1.plot(x_pos, mu_s, color="#dc2626", lw=1.5, label="Predicted median", zorder=3)
        ax1.scatter(x_pos, y_s, s=12, color="black", alpha=0.5, zorder=4, label="Actual impact")
        ax1.set_xlabel("Trade index (sorted by predicted median)", fontsize=9)
        ax1.set_ylabel("|impact_vwap_bps|", fontsize=9)
        ax1.set_title(f"{ticker} XGB GAMLSS Prediction Intervals (90%)\n200 random test trades",
                      fontsize=10, fontweight="bold")
        ax1.legend(fontsize=7.5, loc="upper left")
        ax1.grid(True, alpha=0.15)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

        # -- Col 2: Calibration plot -----------------------------------------------
        ax2 = fig.add_subplot(gs_fig[row_idx, 1])

        ax2.plot(r["cal_levels"], r["cal_actual"], color="#2563eb", lw=2.2,
                 marker="o", markersize=3, label="XGB GAMLSS")
        ax2.plot([0, 1], [0, 1], color="black", lw=1.2, ls="--", alpha=0.6,
                 label="Perfect calibration")

        for lv in [0.50, 0.80, 0.90]:
            cov = r["coverage_data"][lv][0]
            ax2.annotate(f"{lv:.0%}: {cov:.1%}", xy=(lv, cov),
                         textcoords="offset points", xytext=(10, -10),
                         fontsize=7.5, color="#dc2626", fontweight="bold",
                         arrowprops=dict(arrowstyle="->", color="#dc2626", lw=0.8))

        ax2.set_xlabel("Nominal coverage level", fontsize=9)
        ax2.set_ylabel("Actual coverage", fontsize=9)
        ax2.set_title(f"{ticker} Calibration Plot\n(XGB GAMLSS distributional model)",
                      fontsize=10, fontweight="bold")
        ax2.legend(fontsize=8)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.set_aspect("equal")
        ax2.grid(True, alpha=0.15)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        # -- Col 3: Binned scale validation ----------------------------------------
        ax3 = fig.add_subplot(gs_fig[row_idx, 2])

        n_bins = 20
        bin_labels = pd.qcut(b_hat, q=n_bins, labels=False, duplicates="drop")
        n_actual_bins = len(np.unique(bin_labels))
        bin_pred_mean = []
        bin_actual_mean = []
        for bi in range(n_actual_bins):
            mask = bin_labels == bi
            if mask.sum() > 0:
                bin_pred_mean.append(b_hat[mask].mean())
                bin_actual_mean.append(abs_resid[mask].mean())

        ax3.scatter(bin_pred_mean, bin_actual_mean, s=50, color="#16a34a",
                    edgecolors="white", linewidth=0.8, zorder=3)
        lim_max = max(max(bin_pred_mean), max(bin_actual_mean)) * 1.1
        ax3.plot([0, lim_max], [0, lim_max], color="black", lw=1.2, ls="--", alpha=0.6,
                 label="Perfect scale calibration")
        ax3.set_xlabel("Mean predicted scale (b_hat)", fontsize=9)
        ax3.set_ylabel("Mean actual |residual|", fontsize=9)
        ax3.set_title(f"{ticker} Scale Calibration\n(20 quantile bins by b_hat)",
                      fontsize=10, fontweight="bold")
        ax3.legend(fontsize=8)
        ax3.set_xlim(0, lim_max)
        ax3.set_ylim(0, lim_max)
        ax3.set_aspect("equal")
        ax3.grid(True, alpha=0.15)
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)

        # -- Col 4: Interval width vs roll_spread_500 -----------------------------
        ax4 = fig.add_subplot(gs_fig[row_idx, 3])

        spread_idx = FEATURES.index("roll_spread_500")
        spread_vals = X_te[:, spread_idx]
        width_90 = 2 * z90 * b_hat

        samp = rng.choice(len(spread_vals), size=min(3000, len(spread_vals)), replace=False)
        ax4.scatter(spread_vals[samp], width_90[samp], s=8, alpha=0.15,
                    color="#7c3aed", linewidths=0, rasterized=True)

        # Binned mean overlay
        n_bins_sp = 25
        sp_bins = np.linspace(np.percentile(spread_vals, 2), np.percentile(spread_vals, 98), n_bins_sp + 1)
        bin_centers_sp = []
        bin_width_sp = []
        for bi in range(n_bins_sp):
            mask = (spread_vals >= sp_bins[bi]) & (spread_vals < sp_bins[bi + 1])
            if mask.sum() > 5:
                bin_centers_sp.append((sp_bins[bi] + sp_bins[bi + 1]) / 2)
                bin_width_sp.append(width_90[mask].mean())

        ax4.plot(bin_centers_sp, bin_width_sp, color="#dc2626", lw=2.2, marker="o",
                 markersize=4, label="Binned mean width", zorder=5)
        ax4.set_xlabel("roll_spread_500 (bps)", fontsize=9)
        ax4.set_ylabel("90% interval width (bps)", fontsize=9)
        ax4.set_title(f"{ticker} Interval Width vs Spread\n(heteroscedasticity captured by XGB)",
                      fontsize=10, fontweight="bold")
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.15)
        ax4.spines["top"].set_visible(False)
        ax4.spines["right"].set_visible(False)
        ax4.set_xlim(sp_bins[0], sp_bins[-1])

    fig.suptitle(
        "XGBoost GAMLSS Laplace Distributional Regression: Prediction Intervals for Block Trade Impact\n"
        "Location = XGB LAD  |  Scale = XGB MSE on |residuals| (grid-searched)  |  6 features",
        fontsize=14, fontweight="bold", y=1.02,
    )

    plt.savefig("aapl_coin_gamlss_xgb.png", dpi=150, bbox_inches="tight")
    print("Saved -> aapl_coin_gamlss_xgb.png")
    plt.close(fig)

    # -- RS Convergence Plot -------------------------------------------------------
    AAPL_COLOR = "#2563eb"
    COIN_COLOR = "#dc2626"

    fig_rs, axes_rs = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel 1: Log-likelihood convergence
    ax_ll = axes_rs[0]
    for ticker, color, marker in [("AAPL", AAPL_COLOR, "o"), ("COIN", COIN_COLOR, "s")]:
        ll_h = all_results[ticker + "_rs"]["ll_history"]
        iters = list(range(len(ll_h)))
        ax_ll.plot(iters, ll_h, color=color, lw=2.2, marker=marker, markersize=7,
                   label=ticker, zorder=3)
        for i, v in enumerate(ll_h):
            ax_ll.annotate(f"{v:,.0f}", xy=(i, v), textcoords="offset points",
                           xytext=(0, 10 if ticker == "AAPL" else -14),
                           fontsize=7, color=color, ha="center")
    ax_ll.set_xlabel("RS Iteration", fontsize=11)
    ax_ll.set_ylabel("Laplace Log-Likelihood (train)", fontsize=11)
    ax_ll.set_title("Log-Likelihood Convergence", fontsize=12, fontweight="bold")
    ax_ll.legend(fontsize=10)
    ax_ll.grid(True, alpha=0.2)
    ax_ll.spines["top"].set_visible(False)
    ax_ll.spines["right"].set_visible(False)
    ax_ll.set_xticks(range(max(len(all_results["AAPL_rs"]["ll_history"]),
                               len(all_results["COIN_rs"]["ll_history"]))))

    # Panel 2: Test MAE across iterations
    ax_mae = axes_rs[1]
    for ticker, color, marker in [("AAPL", AAPL_COLOR, "o"), ("COIN", COIN_COLOR, "s")]:
        mae_h = all_results[ticker + "_rs"]["mae_te_history"]
        iters = list(range(len(mae_h)))
        ax_mae.plot(iters, mae_h, color=color, lw=2.2, marker=marker, markersize=7,
                    label=ticker, zorder=3)
        for i, v in enumerate(mae_h):
            ax_mae.annotate(f"{v:.4f}", xy=(i, v), textcoords="offset points",
                            xytext=(0, 10 if ticker == "AAPL" else -14),
                            fontsize=7, color=color, ha="center")
    ax_mae.set_xlabel("RS Iteration", fontsize=11)
    ax_mae.set_ylabel("Test MAE (bps)", fontsize=11)
    ax_mae.set_title("Test MAE Across Iterations", fontsize=12, fontweight="bold")
    ax_mae.legend(fontsize=10)
    ax_mae.grid(True, alpha=0.2)
    ax_mae.spines["top"].set_visible(False)
    ax_mae.spines["right"].set_visible(False)
    ax_mae.set_xticks(range(max(len(all_results["AAPL_rs"]["mae_te_history"]),
                                 len(all_results["COIN_rs"]["mae_te_history"]))))

    # Panel 3: Coverage comparison (Two-Stage vs RS)
    ax_cov = axes_rs[2]
    levels = [0.50, 0.80, 0.90]
    x = np.arange(len(levels))
    bw = 0.18

    for j, (ticker, color) in enumerate([("AAPL", AAPL_COLOR), ("COIN", COIN_COLOR)]):
        cd0 = all_results[ticker + "_iter0"]["coverage_data"]
        cdrs = all_results[ticker + "_rs"]["coverage_data"]
        vals_0 = [cd0[lv][0] for lv in levels]
        vals_rs = [cdrs[lv][0] for lv in levels]
        offset = (j - 0.5) * bw * 2.5
        ax_cov.bar(x + offset - bw/2, vals_0, width=bw, color=color, alpha=0.4,
                   edgecolor=color, linewidth=1.2, label=f"{ticker} Two-Stage" if j == 0 else f"{ticker} Two-Stage")
        ax_cov.bar(x + offset + bw/2, vals_rs, width=bw, color=color, alpha=0.85,
                   edgecolor="white", linewidth=0.5, label=f"{ticker} RS" if j == 0 else f"{ticker} RS")

    for lv_val, xi in zip(levels, x):
        ax_cov.axhline(lv_val, color="gray", lw=0.8, ls=":", alpha=0.5, xmin=0, xmax=1)

    ax_cov.set_xticks(x)
    ax_cov.set_xticklabels(["50%", "80%", "90%"], fontsize=11)
    ax_cov.set_xlabel("Nominal Coverage Level", fontsize=11)
    ax_cov.set_ylabel("Actual Coverage", fontsize=11)
    ax_cov.set_title("Coverage: Two-Stage vs RS\n(dotted = nominal)", fontsize=12, fontweight="bold")
    ax_cov.legend(fontsize=8, ncol=2)
    ax_cov.set_ylim(0.4, 1.0)
    ax_cov.grid(axis="y", alpha=0.15)
    ax_cov.spines["top"].set_visible(False)
    ax_cov.spines["right"].set_visible(False)

    fig_rs.suptitle("RS Iteration Convergence: Two-Stage XGBoost vs RS-Converged\n"
                    "AAPL and COIN | Laplace GAMLSS | Up to 5 iterations",
                    fontsize=13, fontweight="bold", y=1.04)
    plt.tight_layout()
    fig_rs.savefig("rs_convergence.png", dpi=150, bbox_inches="tight")
    print("Saved -> rs_convergence.png")
    plt.close(fig_rs)


# =============================================================================
# run_gamlss_xgb_gengauss
# =============================================================================

def run_gamlss_xgb_gengauss():
    """
    Generalized Gaussian extension of XGB GAMLSS.

    Estimates shape parameter p of the generalized Gaussian from standardized
    residuals, then recalibrates prediction intervals vs the Laplace (p=1) baseline.

    Uses the same location/scale XGBoost models and hyperparameters as gamlss_xgb.py.

    Output:
      - aapl_coin_gengauss.png
    """

    # -- Same config as gamlss_xgb.py --------------------------------------------
    FEATURES = [
        "dollar_value", "log_dollar_value", "participation_rate",
        "roll_spread_500", "roll_vol_500", "exchange_id",
    ]

    LOC_PARAMS = dict(
        max_depth=3, n_estimators=200, learning_rate=0.07,
        min_child_weight=5, reg_alpha=10, reg_lambda=10,
    )

    SCALE_GRID = {
        "max_depth": [2, 3, 5],
        "n_estimators": [50, 100],
        "min_child_weight": [10, 20],
    }


    def fit_location_model(X_tr, y_tr, X_te):
        model = XGBRegressor(
            objective="reg:absoluteerror", tree_method="hist",
            verbosity=0, random_state=42, n_jobs=1, **LOC_PARAMS,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_tr, y_tr)
        mu_hat_tr = np.maximum(model.predict(X_tr), 0.0)
        mu_hat_te = np.maximum(model.predict(X_te), 0.0)
        return model, mu_hat_tr, mu_hat_te


    def fit_scale_model(X_tr, abs_resid_tr, X_te):
        base = XGBRegressor(
            objective="reg:squarederror", tree_method="hist",
            verbosity=0, random_state=42, n_jobs=1,
            learning_rate=0.1, reg_alpha=1, reg_lambda=1,
        )
        tscv = TimeSeriesSplit(n_splits=3)
        gs = GridSearchCV(
            base, SCALE_GRID, cv=tscv,
            scoring="neg_mean_absolute_error",
            refit=True, n_jobs=-1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gs.fit(X_tr, abs_resid_tr)
        best = gs.best_estimator_
        b_hat_tr = np.clip(best.predict(X_tr), 0.1, None)
        b_hat_te = np.clip(best.predict(X_te), 0.1, None)
        return best, gs.best_params_, b_hat_tr, b_hat_te


    # -- Generalized Gaussian helpers ---------------------------------------------

    def gengauss_loglik(p, z):
        """
        Log-likelihood for shape p given standardized residuals z = |y - mu| / b.
        Under generalized Gaussian:
          ll(p) = n*log(p) - n*log(2) - n*log(Gamma(1/p)) - sum(z^p)
        """
        n = len(z)
        return n * np.log(p) - n * np.log(2) - n * np.log(gammafn(1.0 / p)) - np.sum(z ** p)


    def estimate_shape(z, bounds=(0.3, 3.0)):
        """Find p that maximizes gengauss_loglik."""
        result = minimize_scalar(
            lambda p: -gengauss_loglik(p, z),
            bounds=bounds, method="bounded",
        )
        return result.x


    def b_hat_to_gennorm_scale(b_hat, p):
        """
        Convert b_hat (mean absolute deviation) to scipy gennorm scale parameter s.

        For gennorm(beta=p, scale=s): E[|X - loc|] = s * Gamma(2/p) / Gamma(1/p).
        We want this to equal b_hat, so s = b_hat * Gamma(1/p) / Gamma(2/p).
        """
        return b_hat * gammafn(1.0 / p) / gammafn(2.0 / p)


    def compute_gennorm_coverage(y, mu_hat, b_hat, p, level):
        """Coverage and width using generalized Gaussian intervals."""
        s = b_hat_to_gennorm_scale(b_hat, p)
        alpha = 1.0 - level
        lo = np.maximum(gennorm.ppf(alpha / 2, beta=p, loc=mu_hat, scale=s), 0.0)
        hi = gennorm.ppf(1 - alpha / 2, beta=p, loc=mu_hat, scale=s)
        covered = ((y >= lo) & (y <= hi)).mean()
        width = (hi - lo).mean()
        return covered, width


    def compute_laplace_coverage(y, mu_hat, b_hat, level):
        """Coverage and width using Laplace (p=1) intervals."""
        z = np.log(1.0 / (1.0 - level))
        lo = np.maximum(mu_hat - z * b_hat, 0.0)
        hi = mu_hat + z * b_hat
        covered = ((y >= lo) & (y <= hi)).mean()
        width = (hi - lo).mean()
        return covered, width


    # -- Process both tickers -----------------------------------------------------

    TICKERS = [
        ("AAPL", "data/lit_buy_features_v2.parquet", "data/lit_buy_features_v2_sep.parquet"),
        ("COIN", "data/coin_lit_buy_features_train.parquet", "data/coin_lit_buy_features_test.parquet"),
    ]

    all_results = {}

    for ticker, tr_file, te_file in TICKERS:
        print(f"\n{'=' * 80}")
        print(f"  {ticker}")
        print(f"{'=' * 80}")

        df_tr = pd.read_parquet(tr_file)
        df_te = pd.read_parquet(te_file)
        df_tr["abs_impact"] = df_tr["impact_vwap_bps"].abs()
        df_te["abs_impact"] = df_te["impact_vwap_bps"].abs()
        df_tr = df_tr.sort_values("date").reset_index(drop=True)

        X_tr = df_tr[FEATURES].to_numpy(dtype=np.float64)
        y_tr = df_tr["abs_impact"].to_numpy(dtype=np.float64)
        X_te = df_te[FEATURES].to_numpy(dtype=np.float64)
        y_te = df_te["abs_impact"].to_numpy(dtype=np.float64)

        print(f"  Train: {len(df_tr):,}  |  Test: {len(df_te):,}")

        # Stage 1: Location model (same as gamlss_xgb.py)
        loc_model, mu_hat_tr, mu_hat_te = fit_location_model(X_tr, y_tr, X_te)
        print(f"  Location MAE:  Train={np.mean(np.abs(y_tr - mu_hat_tr)):.4f}  "
              f"Test={np.mean(np.abs(y_te - mu_hat_te)):.4f}")

        # Stage 2: Scale model (same as gamlss_xgb.py)
        abs_resid_tr = np.abs(y_tr - mu_hat_tr)
        scale_model, best_params, b_hat_tr, b_hat_te = fit_scale_model(
            X_tr, abs_resid_tr, X_te)
        print(f"  Scale model best params: {best_params}")
        print(f"  b_hat test: mean={b_hat_te.mean():.4f}  median={np.median(b_hat_te):.4f}")

        # Stage 3: Estimate shape parameter p from training standardized residuals
        z_tr = np.abs(y_tr - mu_hat_tr) / b_hat_tr
        p_hat = estimate_shape(z_tr)

        ll_p1 = gengauss_loglik(1.0, z_tr)
        ll_phat = gengauss_loglik(p_hat, z_tr)
        lr_stat = 2 * (ll_phat - ll_p1)
        lr_pval = 1.0 - chi2.cdf(lr_stat, df=1)

        print(f"\n  SHAPE PARAMETER ESTIMATION")
        print(f"  Estimated p = {p_hat:.4f}")
        if p_hat < 1.0:
            print(f"  Interpretation: p < 1 -> sharper peak than Laplace (super-Laplacian)")
        elif abs(p_hat - 1.0) < 0.01:
            print(f"  Interpretation: p ~ 1 -> approximately Laplace")
        else:
            print(f"  Interpretation: p > 1 -> closer to Gaussian (sub-Laplacian)")
        print(f"  Log-likelihood at p=1 (Laplace): {ll_p1:.2f}")
        print(f"  Log-likelihood at p={p_hat:.4f}:      {ll_phat:.2f}")
        print(f"  LR test statistic: {lr_stat:.2f}  (p-value: {lr_pval:.2e})")

        # Stage 4: Coverage comparison
        print(f"\n  COVERAGE COMPARISON (test set)")
        header = (f"  {'Model':<24} {'90% Cov':>8} {'80% Cov':>8} {'50% Cov':>8} "
                  f"{'90% Width':>10} {'50% Width':>10}")
        print(header)
        print(f"  {'-' * 24} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 10} {'-' * 10}")

        laplace_covs = {}
        gg_covs = {}
        for level in [0.90, 0.80, 0.50]:
            cov_l, w_l = compute_laplace_coverage(y_te, mu_hat_te, b_hat_te, level)
            cov_g, w_g = compute_gennorm_coverage(y_te, mu_hat_te, b_hat_te, p_hat, level)
            laplace_covs[level] = (cov_l, w_l)
            gg_covs[level] = (cov_g, w_g)

        print(f"  {'Laplace (p=1)':<24} "
              f"{laplace_covs[0.90][0]:>8.4f} {laplace_covs[0.80][0]:>8.4f} "
              f"{laplace_covs[0.50][0]:>8.4f} {laplace_covs[0.90][1]:>10.4f} "
              f"{laplace_covs[0.50][1]:>10.4f}")
        print(f"  {f'GenGaussian (p={p_hat:.3f})':<24} "
              f"{gg_covs[0.90][0]:>8.4f} {gg_covs[0.80][0]:>8.4f} "
              f"{gg_covs[0.50][0]:>8.4f} {gg_covs[0.90][1]:>10.4f} "
              f"{gg_covs[0.50][1]:>10.4f}")

        # Full calibration curves
        cal_levels = np.linspace(0.05, 0.99, 50)
        cal_laplace = []
        cal_gengauss = []
        for lv in cal_levels:
            c_l, _ = compute_laplace_coverage(y_te, mu_hat_te, b_hat_te, lv)
            c_g, _ = compute_gennorm_coverage(y_te, mu_hat_te, b_hat_te, p_hat, lv)
            cal_laplace.append(c_l)
            cal_gengauss.append(c_g)

        # Standardized residuals on test set for plotting
        z_te = np.abs(y_te - mu_hat_te) / b_hat_te

        all_results[ticker] = {
            "y_te": y_te, "mu_hat_te": mu_hat_te, "b_hat_te": b_hat_te,
            "z_tr": z_tr, "z_te": z_te,
            "p_hat": p_hat,
            "ll_p1": ll_p1, "ll_phat": ll_phat,
            "lr_stat": lr_stat, "lr_pval": lr_pval,
            "laplace_covs": laplace_covs, "gg_covs": gg_covs,
            "cal_levels": cal_levels,
            "cal_laplace": np.array(cal_laplace),
            "cal_gengauss": np.array(cal_gengauss),
        }

    # -- Summary print ------------------------------------------------------------
    print(f"\n\n{'=' * 80}")
    print("  SUMMARY")
    print(f"{'=' * 80}")
    for ticker in ["AAPL", "COIN"]:
        r = all_results[ticker]
        print(f"\n  {ticker}: p_hat = {r['p_hat']:.4f}  "
              f"(LR stat = {r['lr_stat']:.2f}, p-value = {r['lr_pval']:.2e})")
        print(f"    50% coverage:  Laplace={r['laplace_covs'][0.50][0]:.4f}  "
              f"GenGauss={r['gg_covs'][0.50][0]:.4f}  (nominal=0.50)")
        print(f"    80% coverage:  Laplace={r['laplace_covs'][0.80][0]:.4f}  "
              f"GenGauss={r['gg_covs'][0.80][0]:.4f}  (nominal=0.80)")
        print(f"    90% coverage:  Laplace={r['laplace_covs'][0.90][0]:.4f}  "
              f"GenGauss={r['gg_covs'][0.90][0]:.4f}  (nominal=0.90)")

    # =============================================================================
    # FIGURE: 2 rows (AAPL, COIN) x 3 columns
    # =============================================================================
    fig = plt.figure(figsize=(22, 13))
    gs_fig = gridspec.GridSpec(2, 3, figure=fig, wspace=0.30, hspace=0.42)

    rng = np.random.default_rng(42)

    for row_idx, (ticker, _, _) in enumerate(TICKERS):
        r = all_results[ticker]
        p_hat = r["p_hat"]
        z_te = r["z_te"]

        # -- Col 1: Calibration curves (Laplace vs GenGaussian) -------------------
        ax1 = fig.add_subplot(gs_fig[row_idx, 0])

        ax1.plot([0, 1], [0, 1], color="black", lw=1.2, ls="--", alpha=0.5,
                 label="Perfect calibration")
        ax1.plot(r["cal_levels"], r["cal_laplace"], color="#2563eb", lw=2.2,
                 marker="o", markersize=3, label="Laplace (p=1)", alpha=0.85)
        ax1.plot(r["cal_levels"], r["cal_gengauss"], color="#dc2626", lw=2.2,
                 marker="s", markersize=3, label=f"GenGaussian (p={p_hat:.3f})", alpha=0.85)

        # Annotate key levels
        for lv in [0.50, 0.80, 0.90]:
            cov_l = r["laplace_covs"][lv][0]
            cov_g = r["gg_covs"][lv][0]
            ax1.plot(lv, cov_l, "o", color="#2563eb", markersize=7, zorder=5)
            ax1.plot(lv, cov_g, "s", color="#dc2626", markersize=7, zorder=5)
            offset_y = 12 if cov_g > cov_l else -14
            ax1.annotate(f"{lv:.0%}: L={cov_l:.1%}, G={cov_g:.1%}",
                         xy=(lv, (cov_l + cov_g) / 2),
                         textcoords="offset points", xytext=(8, offset_y),
                         fontsize=7, fontweight="bold",
                         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#94a3b8", alpha=0.9))

        ax1.set_xlabel("Nominal coverage level", fontsize=10)
        ax1.set_ylabel("Actual coverage", fontsize=10)
        ax1.set_title(f"{ticker} Calibration: Laplace vs GenGaussian\n"
                      f"(p={p_hat:.3f}, LR p-value={r['lr_pval']:.1e})",
                      fontsize=11, fontweight="bold")
        ax1.legend(fontsize=8.5, loc="upper left")
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.set_aspect("equal")
        ax1.grid(True, alpha=0.15)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

        # -- Col 2: Histogram of |z| with theoretical densities -------------------
        ax2 = fig.add_subplot(gs_fig[row_idx, 1])

        clip_z = np.percentile(z_te, 98)
        bins = np.linspace(0, clip_z, 70)
        ax2.hist(z_te[z_te <= clip_z], bins=bins, density=True, color="#7c3aed",
                 alpha=0.55, edgecolor="white", linewidth=0.3, label="Observed |z|")

        # Theoretical densities for |z|: pdf of |z| = 2 * pdf_gengauss(z) for z >= 0
        z_grid = np.linspace(0.001, clip_z, 300)

        # Laplace: |z| ~ Exponential(rate=1) => pdf = exp(-z)
        laplace_pdf = np.exp(-z_grid)
        ax2.plot(z_grid, laplace_pdf, color="#2563eb", lw=2.2, ls="--",
                 label="Laplace (p=1): Exp(1)")

        # GenGaussian |z|: folded density = 2 * gennorm.pdf(z, beta=p, scale=s_unit)
        # where s_unit corresponds to unit mean-absolute-deviation
        s_unit = gammafn(1.0 / p_hat) / gammafn(2.0 / p_hat)
        gg_pdf = 2 * gennorm.pdf(z_grid, beta=p_hat, loc=0, scale=s_unit)
        ax2.plot(z_grid, gg_pdf, color="#dc2626", lw=2.2,
                 label=f"GenGaussian (p={p_hat:.3f})")

        ax2.set_xlabel("Standardized residual |z| = |y - mu| / b", fontsize=10)
        ax2.set_ylabel("Density", fontsize=10)
        ax2.set_title(f"{ticker} Standardized Residual Distribution\n"
                      f"(test set, n={len(z_te):,})",
                      fontsize=11, fontweight="bold")
        ax2.legend(fontsize=8.5)
        ax2.grid(True, alpha=0.15)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        # -- Col 3: QQ plot of z against GenGaussian quantiles --------------------
        ax3 = fig.add_subplot(gs_fig[row_idx, 2])

        # Use signed standardized residuals for QQ
        signed_z = (r["y_te"] - r["mu_hat_te"]) / r["b_hat_te"]
        signed_z_sorted = np.sort(signed_z)
        n = len(signed_z_sorted)

        # Theoretical quantiles from fitted generalized Gaussian
        probs = (np.arange(1, n + 1) - 0.5) / n
        s_unit = gammafn(1.0 / p_hat) / gammafn(2.0 / p_hat)
        theo_quantiles = gennorm.ppf(probs, beta=p_hat, loc=0, scale=s_unit)

        # Subsample for plotting
        step = max(1, n // 3000)
        ax3.scatter(theo_quantiles[::step], signed_z_sorted[::step],
                    s=8, alpha=0.4, color="#dc2626", edgecolors="none",
                    label=f"GenGauss QQ (p={p_hat:.3f})", zorder=3)

        # Also show Laplace QQ for comparison
        theo_laplace = gennorm.ppf(probs, beta=1.0, loc=0, scale=1.0)
        ax3.scatter(theo_laplace[::step], signed_z_sorted[::step],
                    s=6, alpha=0.25, color="#2563eb", edgecolors="none",
                    label="Laplace QQ (p=1)", zorder=2)

        lim = max(abs(signed_z_sorted[int(0.001 * n)]),
                  abs(signed_z_sorted[int(0.999 * n)])) * 1.1
        ax3.plot([-lim, lim], [-lim, lim], color="black", lw=1.2, ls="--", alpha=0.5)
        ax3.set_xlabel("Theoretical quantiles", fontsize=10)
        ax3.set_ylabel("Sample quantiles (z)", fontsize=10)
        ax3.set_title(f"{ticker} QQ Plot: Standardized Residuals\n"
                      f"GenGaussian (p={p_hat:.3f}) vs Laplace (p=1)",
                      fontsize=11, fontweight="bold")
        ax3.legend(fontsize=8.5, loc="upper left")
        ax3.grid(True, alpha=0.15)
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)

    fig.suptitle(
        "Generalized Gaussian Extension of XGB GAMLSS\n"
        "Estimating shape p from standardized residuals to improve interval calibration",
        fontsize=14, fontweight="bold", y=1.02,
    )

    plt.savefig("aapl_coin_gengauss.png", dpi=150, bbox_inches="tight")
    print("\nSaved -> aapl_coin_gengauss.png")
    print("\nDone!")


# =============================================================================
# run_gamlss_full
# =============================================================================

def run_gamlss_full():
    """
    Full GAMLSS (Rigby & Stasinopoulos 2005) for Laplace distribution.

    For each trade i:  |impact_i| ~ Laplace(mu_i, b_i)

    Additive predictors with P-splines:
      mu_i = a0 + sum_j f_j(x_ij)         [identity link]
      log(b_i) = g0 + sum_j g_j(x_ij)     [log link]

    RS algorithm:
      1. Initialize mu, b
      2. Update mu: PIRLS with Laplace Fisher scoring for location
      3. Update b: PIRLS with Laplace Fisher scoring for scale
      4. Monitor global deviance, converge when relative change < tol

    Output:
      - gamlss_full_results.png
      - Console comparison table vs Two-Stage XGBoost
    """

    FEATURES = [
        "dollar_value", "log_dollar_value", "participation_rate",
        "roll_spread_500", "roll_vol_500", "exchange_id",
    ]


    # ── P-Spline Design Matrix ───────────────────────────────────────────────────

    class GAMLSSDesign:
        """Design matrix: intercept + P-spline basis for each feature."""

        def __init__(self, n_basis=12, degree=3):
            self.n_basis = n_basis
            self.degree = degree
            self.knot_sequences = []
            self.n_basis_per_feat = []

        def fit_transform(self, X):
            n, p = X.shape
            blocks = [np.ones((n, 1))]
            self.knot_sequences = []
            self.n_basis_per_feat = []

            for j in range(p):
                xj = X[:, j]
                n_internal = self.n_basis - self.degree + 1
                internal = np.percentile(xj, np.linspace(0, 100, n_internal + 2)[1:-1])
                internal = np.unique(internal)
                xmin, xmax = xj.min() - 1e-6, xj.max() + 1e-6
                knots = np.concatenate([
                    np.repeat(xmin, self.degree + 1),
                    internal,
                    np.repeat(xmax, self.degree + 1),
                ])
                self.knot_sequences.append(knots)
                B = BSpline.design_matrix(xj, knots, self.degree)
                if issparse(B):
                    B = B.toarray()
                self.n_basis_per_feat.append(B.shape[1])
                blocks.append(B)

            return np.hstack(blocks)

        def transform(self, X):
            n, p = X.shape
            blocks = [np.ones((n, 1))]
            for j in range(p):
                knots = self.knot_sequences[j]
                xj = np.clip(X[:, j], knots[0], knots[-1])
                B = BSpline.design_matrix(xj, knots, self.degree)
                if issparse(B):
                    B = B.toarray()
                blocks.append(B)
            return np.hstack(blocks)

        def penalty_matrix(self, order=2):
            total = 1 + sum(self.n_basis_per_feat)
            P = np.zeros((total, total))
            offset = 1  # skip intercept
            for nb in self.n_basis_per_feat:
                if nb > order:
                    D = np.diff(np.eye(nb), n=order, axis=0)
                    P[offset:offset + nb, offset:offset + nb] = D.T @ D
                offset += nb
            return P


    # ── GAMLSS fitting ───────────────────────────────────────────────────────────

    def laplace_deviance(y, mu, b):
        """Global deviance = -2 * log-likelihood."""
        return 2.0 * np.sum(np.log(2.0 * b) + np.abs(y - mu) / b)


    def pwls_step(B, z, w, P, lam):
        """One penalized weighted least squares solve."""
        WB = w[:, None] * B
        A = B.T @ WB + lam * P + 1e-8 * np.eye(B.shape[1])
        rhs = WB.T @ z
        return np.linalg.solve(A, rhs)


    def fit_gamlss_laplace(X_tr, y_tr, X_te, n_basis=12, lam_mu=10.0, lam_b=10.0,
                            max_iter=50, tol=1e-6, verbose=True):
        """
        Fit full GAMLSS for Laplace(mu, b) using RS algorithm with P-splines.
        """
        # Build design matrices (separate bases for mu and b)
        des_mu = GAMLSSDesign(n_basis=n_basis)
        des_b = GAMLSSDesign(n_basis=n_basis)

        B_mu_tr = des_mu.fit_transform(X_tr)
        B_mu_te = des_mu.transform(X_te)
        B_b_tr = des_b.fit_transform(X_tr)
        B_b_te = des_b.transform(X_te)

        P_mu = des_mu.penalty_matrix()
        P_b = des_b.penalty_matrix()

        n = len(y_tr)

        # Initialize: median for location, mean |residual| for scale
        mu = np.full(n, np.median(y_tr))
        b = np.full(n, np.mean(np.abs(y_tr - mu)))
        b = np.clip(b, 0.01, None)

        dev = laplace_deviance(y_tr, mu, b)
        dev_history = [dev]
        if verbose:
            print(f"    Init : deviance = {dev:,.2f}")

        beta_mu = None
        beta_b = None

        for it in range(1, max_iter + 1):
            # ── Update mu (identity link) ──
            # Laplace score for mu: sign(y - mu) / b
            # Fisher weight: 1 / b^2
            # Working variable: z_mu = mu + score / weight = mu + sign(y - mu) * b
            sign_r = np.sign(y_tr - mu)
            sign_r[sign_r == 0] = 0.001
            z_mu = mu + sign_r * b
            w_mu = 1.0 / (b ** 2)

            beta_mu_new = pwls_step(B_mu_tr, z_mu, w_mu, P_mu, lam_mu)
            mu_new = B_mu_tr @ beta_mu_new
            mu_new = np.maximum(mu_new, 0.0)

            # Step-halving for mu
            dev_after_mu = laplace_deviance(y_tr, mu_new, b)
            step = 1.0
            for _ in range(5):
                if dev_after_mu <= dev + 1.0:  # allow tiny increase
                    break
                step *= 0.5
                mu_cand = (1 - step) * mu + step * mu_new
                dev_after_mu = laplace_deviance(y_tr, mu_cand, b)
                mu_new = mu_cand

            mu = mu_new
            beta_mu = beta_mu_new

            # ── Update b (log link: eta_b = log(b)) ──
            # Laplace score for log(b): -1 + |y - mu| / b
            # Fisher weight: 1
            # Working variable: z_b = log(b) + (-1 + |y - mu| / b)
            abs_r = np.abs(y_tr - mu)
            eta_b = np.log(np.maximum(b, 1e-6))
            z_b = eta_b + (-1.0 + abs_r / b)
            w_b = np.ones(n)

            beta_b_new = pwls_step(B_b_tr, z_b, w_b, P_b, lam_b)
            eta_b_new = B_b_tr @ beta_b_new
            b_new = np.clip(np.exp(eta_b_new), 0.01, 500.0)

            # Step-halving for b
            dev_after_b = laplace_deviance(y_tr, mu, b_new)
            step = 1.0
            for _ in range(5):
                if dev_after_b <= dev + 1.0:
                    break
                step *= 0.5
                b_cand = np.exp((1 - step) * np.log(b) + step * np.log(b_new))
                b_cand = np.clip(b_cand, 0.01, 500.0)
                dev_after_b = laplace_deviance(y_tr, mu, b_cand)
                b_new = b_cand

            b = b_new
            beta_b = beta_b_new

            dev_new = laplace_deviance(y_tr, mu, b)
            dev_history.append(dev_new)
            rel_change = abs(dev_new - dev) / (abs(dev) + 1e-10)

            if verbose:
                print(f"    Iter {it:2d}: deviance = {dev_new:,.2f}  "
                      f"(rel delta = {rel_change:.8f})")

            if rel_change < tol:
                if verbose:
                    print(f"    Converged at iteration {it}")
                break
            dev = dev_new

        # Predict on test
        mu_te = np.maximum(B_mu_te @ beta_mu, 0.0)
        b_te = np.clip(np.exp(B_b_te @ beta_b), 0.01, 500.0)

        return mu, b, mu_te, b_te, dev_history


    def compute_coverage(y, mu, b, level):
        z = np.log(1.0 / (1.0 - level))
        lo = np.maximum(mu - z * b, 0.0)
        hi = mu + z * b
        cov = ((y >= lo) & (y <= hi)).mean()
        width = (hi - lo).mean()
        return cov, width


    # ── Main ─────────────────────────────────────────────────────────────────────

    TICKERS = [
        ("AAPL", "data/lit_buy_features_v2.parquet", "data/lit_buy_features_v2_sep.parquet"),
        ("COIN", "data/coin_lit_buy_features_train.parquet", "data/coin_lit_buy_features_test.parquet"),
    ]

    results = {}

    for ticker, tr_file, te_file in TICKERS:
        print(f"\n{'=' * 70}")
        print(f"  {ticker}: Full GAMLSS (RS algorithm + P-splines, Laplace)")
        print(f"{'=' * 70}")

        df_tr = pd.read_parquet(tr_file)
        df_te = pd.read_parquet(te_file)
        df_tr["abs_impact"] = df_tr["impact_vwap_bps"].abs()
        df_te["abs_impact"] = df_te["impact_vwap_bps"].abs()
        df_tr = df_tr.sort_values("date").reset_index(drop=True)

        X_tr = df_tr[FEATURES].to_numpy(dtype=np.float64)
        y_tr = df_tr["abs_impact"].to_numpy(dtype=np.float64)
        X_te = df_te[FEATURES].to_numpy(dtype=np.float64)
        y_te = df_te["abs_impact"].to_numpy(dtype=np.float64)

        print(f"  Train: {len(df_tr):,}  |  Test: {len(df_te):,}")

        mu_tr, b_tr, mu_te, b_te, dev_hist = fit_gamlss_laplace(
            X_tr, y_tr, X_te, n_basis=12, lam_mu=10.0, lam_b=10.0,
            max_iter=50, tol=1e-6, verbose=True)

        mae_te = np.mean(np.abs(y_te - mu_te))
        ll_train = -0.5 * laplace_deviance(y_tr, mu_tr, b_tr)
        print(f"\n  Test MAE: {mae_te:.4f}")
        print(f"  Final train log-lik: {ll_train:,.2f}")

        cov_data = {}
        print(f"  {'Level':>8} {'Coverage':>10} {'Width':>10}")
        for level in [0.50, 0.80, 0.90]:
            cov, width = compute_coverage(y_te, mu_te, b_te, level)
            cov_data[level] = (cov, width)
            print(f"  {level:>8.0%} {cov:>10.4f} {width:>10.4f}")

        results[ticker] = {
            "mae_te": mae_te,
            "coverage_data": cov_data,
            "dev_history": dev_hist,
            "y_te": y_te, "mu_te": mu_te, "b_te": b_te,
        }


    # ── Comparison table ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 90}")
    print("  COMPARISON: Full GAMLSS vs Two-Stage XGBoost (RS)")
    print(f"{'=' * 90}")
    print(f"  {'Model':<35} {'MAE':>8} {'90% Cov':>8} {'80% Cov':>8} "
          f"{'50% Cov':>8} {'90% Width':>10}")
    print(f"  {'-' * 35} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 10}")

    try:
        xgb_df = pd.read_csv("data/gamlss_xgb_results.csv")
        has_xgb = True
    except FileNotFoundError:
        has_xgb = False

    for ticker in ["AAPL", "COIN"]:
        r = results[ticker]
        cd = r["coverage_data"]
        print(f"  {ticker + ' Full GAMLSS':<35} {r['mae_te']:>8.4f} {cd[0.90][0]:>8.4f} "
              f"{cd[0.80][0]:>8.4f} {cd[0.50][0]:>8.4f} {cd[0.90][1]:>10.4f}")

        if has_xgb:
            rs_rows = xgb_df[xgb_df["model"] == "XGB_GAMLSS_RS"]
            if len(rs_rows) == 0:
                rs_rows = xgb_df
            xr90 = rs_rows[(rs_rows["ticker"] == ticker) & (rs_rows["level"] == 0.90)]
            xr80 = rs_rows[(rs_rows["ticker"] == ticker) & (rs_rows["level"] == 0.80)]
            xr50 = rs_rows[(rs_rows["ticker"] == ticker) & (rs_rows["level"] == 0.50)]
            if len(xr90) > 0:
                print(f"  {ticker + ' Two-Stage XGBoost (RS)':<35} "
                      f"{xr90['test_mae'].values[0]:>8.4f} "
                      f"{xr90['actual_coverage'].values[0]:>8.4f} "
                      f"{xr80['actual_coverage'].values[0]:>8.4f} "
                      f"{xr50['actual_coverage'].values[0]:>8.4f} "
                      f"{xr90['mean_interval_width'].values[0]:>10.4f}")

    print(f"\n  Nominal targets:  90% -> 0.9000   80% -> 0.8000   50% -> 0.5000")


    # ── Fit Two-Stage XGB inline for calibration curve overlay ───────────────────
    LOC_PARAMS = dict(max_depth=3, n_estimators=200, learning_rate=0.07,
                      min_child_weight=5, reg_alpha=10, reg_lambda=10)

    xgb_results = {}
    for ticker, tr_file, te_file in TICKERS:
        df_tr = pd.read_parquet(tr_file)
        df_te = pd.read_parquet(te_file)
        df_tr["abs_impact"] = df_tr["impact_vwap_bps"].abs()
        df_te["abs_impact"] = df_te["impact_vwap_bps"].abs()
        df_tr = df_tr.sort_values("date").reset_index(drop=True)
        X_tr = df_tr[FEATURES].to_numpy(dtype=np.float64)
        y_tr = df_tr["abs_impact"].to_numpy(dtype=np.float64)
        X_te = df_te[FEATURES].to_numpy(dtype=np.float64)
        y_te = df_te["abs_impact"].to_numpy(dtype=np.float64)

        loc = XGBRegressor(objective="reg:absoluteerror", tree_method="hist",
                           verbosity=0, random_state=42, n_jobs=1, **LOC_PARAMS)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loc.fit(X_tr, y_tr)
        mu_tr_x = np.maximum(loc.predict(X_tr), 0.0)
        mu_te_x = np.maximum(loc.predict(X_te), 0.0)
        sc = XGBRegressor(objective="reg:squarederror", tree_method="hist", verbosity=0,
                          random_state=42, n_jobs=1, max_depth=5, n_estimators=50,
                          min_child_weight=20, learning_rate=0.1, reg_alpha=1, reg_lambda=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sc.fit(X_tr, np.abs(y_tr - mu_tr_x))
        b_te_x = np.clip(sc.predict(X_te), 0.1, None)
        xgb_results[ticker] = {"mu_te": mu_te_x, "b_te": b_te_x, "y_te": y_te}
    print("Two-Stage XGB fitted for overlay comparison.")


    # ── Plot ─────────────────────────────────────────────────────────────────────
    AAPL_COLOR = "#2563eb"
    COIN_COLOR = "#dc2626"
    cal_levels = np.linspace(0.05, 0.99, 50)

    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(2, 2, wspace=0.30, hspace=0.35)

    # ── Panel 1 (top-left): Deviance convergence ─────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    for ticker, color, marker in [("AAPL", AAPL_COLOR, "o"), ("COIN", COIN_COLOR, "s")]:
        dh = results[ticker]["dev_history"]
        # Normalize: show % reduction from init
        dh_pct = [(dh[0] - d) / dh[0] * 100 for d in dh]
        ax1.plot(range(len(dh)), dh_pct, color=color, lw=2.2, marker=marker,
                 markersize=5, markevery=2, label=ticker)

    ax1.set_xlabel("RS Iteration", fontsize=12)
    ax1.set_ylabel("Deviance Reduction from Init (%)", fontsize=12)
    ax1.set_title("RS Algorithm Convergence", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.2)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Inset: raw deviance (small)
    ax1_in = ax1.inset_axes([0.50, 0.25, 0.45, 0.40])
    for ticker, color, marker in [("AAPL", AAPL_COLOR, "o"), ("COIN", COIN_COLOR, "s")]:
        dh = results[ticker]["dev_history"]
        ax1_in.plot(range(len(dh)), dh, color=color, lw=1.5, marker=marker, markersize=3)
    ax1_in.set_ylabel("Deviance", fontsize=7)
    ax1_in.set_xlabel("Iter", fontsize=7)
    ax1_in.tick_params(labelsize=7)
    ax1_in.grid(True, alpha=0.15)
    ax1_in.set_title("Raw deviance", fontsize=7)

    # ── Panel 2 (top-right): Calibration curves — GAMLSS vs XGB ─────────────────
    ax2 = fig.add_subplot(gs[0, 1])

    ax2.fill_between([0, 1], [0, 1], [0.05, 1.05], color="gray", alpha=0.06)
    ax2.fill_between([0, 1], [-0.05, 0.95], [0, 1], color="gray", alpha=0.06)

    for ticker, color, marker in [("AAPL", AAPL_COLOR, "o"), ("COIN", COIN_COLOR, "s")]:
        # GAMLSS (solid)
        r = results[ticker]
        cal_g = [compute_coverage(r["y_te"], r["mu_te"], r["b_te"], lv)[0] for lv in cal_levels]
        ax2.plot(cal_levels, cal_g, color=color, lw=2.5, marker=marker, markersize=4,
                 markevery=3, label=f"{ticker} GAMLSS", zorder=3)

        # XGB (dashed)
        rx = xgb_results[ticker]
        cal_x = [compute_coverage(rx["y_te"], rx["mu_te"], rx["b_te"], lv)[0] for lv in cal_levels]
        ax2.plot(cal_levels, cal_x, color=color, lw=1.8, ls="--", alpha=0.6,
                 label=f"{ticker} Two-Stage XGB", zorder=2)

    ax2.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.4, label="Perfect")
    ax2.set_xlabel("Nominal Coverage", fontsize=12)
    ax2.set_ylabel("Actual Coverage", fontsize=12)
    ax2.set_title("Calibration: Full GAMLSS (solid) vs Two-Stage XGB (dashed)",
                  fontsize=13, fontweight="bold")
    ax2.legend(fontsize=9, loc="upper left")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.15)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # ── Panel 3 (bottom-left): Coverage deviation ────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])

    ax3.axhline(0, color="black", lw=1.2, ls="--", alpha=0.5, zorder=1)
    ax3.fill_between(cal_levels, -0.02, 0.02, color="green", alpha=0.08, label="\u00b12% band")
    ax3.fill_between(cal_levels, -0.05, 0.05, color="orange", alpha=0.05, label="\u00b15% band")

    for ticker, color, marker in [("AAPL", AAPL_COLOR, "o"), ("COIN", COIN_COLOR, "s")]:
        # GAMLSS
        r = results[ticker]
        cal_g = np.array([compute_coverage(r["y_te"], r["mu_te"], r["b_te"], lv)[0]
                          for lv in cal_levels])
        ax3.plot(cal_levels, cal_g - cal_levels, color=color, lw=2.2, marker=marker,
                 markersize=4, markevery=3, label=f"{ticker} GAMLSS", zorder=3)

        # XGB
        rx = xgb_results[ticker]
        cal_x = np.array([compute_coverage(rx["y_te"], rx["mu_te"], rx["b_te"], lv)[0]
                          for lv in cal_levels])
        ax3.plot(cal_levels, cal_x - cal_levels, color=color, lw=1.8, ls="--", alpha=0.6,
                 label=f"{ticker} XGB", zorder=2)

    ax3.set_xlabel("Nominal Coverage Level", fontsize=12)
    ax3.set_ylabel("Actual \u2212 Nominal", fontsize=12)
    ax3.set_title("Coverage Deviation: GAMLSS (solid) vs XGB (dashed)",
                  fontsize=13, fontweight="bold")
    ax3.legend(fontsize=8, loc="upper left", ncol=2)
    ax3.set_xlim(0.05, 0.95)
    ax3.set_ylim(-0.12, 0.15)
    ax3.grid(True, alpha=0.2)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    # ── Panel 4 (bottom-right): Summary comparison table as chart ────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")

    # Build table data
    col_labels = ["MAE", "50% Cov", "80% Cov", "90% Cov", "90% Width"]
    row_labels = []
    cell_data = []
    cell_colors = []

    for ticker in ["AAPL", "COIN"]:
        base_color = AAPL_COLOR if ticker == "AAPL" else COIN_COLOR

        # GAMLSS row
        r = results[ticker]
        cd = r["coverage_data"]
        row_labels.append(f"{ticker} Full GAMLSS")
        cell_data.append([f"{r['mae_te']:.4f}", f"{cd[0.50][0]:.4f}", f"{cd[0.80][0]:.4f}",
                          f"{cd[0.90][0]:.4f}", f"{cd[0.90][1]:.2f}"])
        cell_colors.append([base_color + "30"] * 5)

        # XGB row
        rx = xgb_results[ticker]
        mae_x = np.mean(np.abs(rx["y_te"] - rx["mu_te"]))
        cov90_x, w90_x = compute_coverage(rx["y_te"], rx["mu_te"], rx["b_te"], 0.90)
        cov80_x, _ = compute_coverage(rx["y_te"], rx["mu_te"], rx["b_te"], 0.80)
        cov50_x, _ = compute_coverage(rx["y_te"], rx["mu_te"], rx["b_te"], 0.50)
        row_labels.append(f"{ticker} Two-Stage XGB")
        cell_data.append([f"{mae_x:.4f}", f"{cov50_x:.4f}", f"{cov80_x:.4f}",
                          f"{cov90_x:.4f}", f"{w90_x:.2f}"])
        cell_colors.append([base_color + "15"] * 5)

        # Nominal row
        row_labels.append("Nominal target")
        cell_data.append(["", "0.5000", "0.8000", "0.9000", ""])
        cell_colors.append(["#f0f0f0"] * 5)

    table = ax4.table(cellText=cell_data, rowLabels=row_labels, colLabels=col_labels,
                      cellColours=cell_colors, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)

    # Bold header
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight="bold")
        if col == -1:
            cell.set_text_props(fontweight="bold", fontsize=9)
        cell.set_edgecolor("#cccccc")

    ax4.set_title("Summary Comparison", fontsize=13, fontweight="bold", pad=20)

    fig.suptitle("Full GAMLSS (Rigby & Stasinopoulos 2005) vs Two-Stage XGBoost\n"
                 "Laplace Distribution | P-Spline Smooth Terms | RS Algorithm | 6 Features",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig("gamlss_full_results.png", dpi=150, bbox_inches="tight")
    print("\nSaved -> gamlss_full_results.png")
    plt.close(fig)


    # ── Fan Plot ─────────────────────────────────────────────────────────────────
    print("Plotting fan charts...")

    rng = np.random.default_rng(42)
    n_show = 300
    SMOOTH_WIN = 15  # rolling window for smoothing fan edges

    BAND_COLORS_AAPL = {"90%": "#93c5fd", "80%": "#60a5fa", "50%": "#3b82f6"}
    BAND_COLORS_COIN = {"90%": "#fca5a5", "80%": "#f87171", "50%": "#ef4444"}


    def smooth(arr, win=SMOOTH_WIN):
        """Centered rolling mean, preserving array length."""
        kernel = np.ones(win) / win
        padded = np.pad(arr, win // 2, mode="edge")
        return np.convolve(padded, kernel, mode="valid")[:len(arr)]


    fig_fan, axes_fan = plt.subplots(2, 2, figsize=(20, 14))

    for row, (ticker, color, bands) in enumerate([
        ("AAPL", AAPL_COLOR, BAND_COLORS_AAPL),
        ("COIN", COIN_COLOR, BAND_COLORS_COIN),
    ]):
        r = results[ticker]
        y_te = r["y_te"]
        mu_te = r["mu_te"]
        b_te = r["b_te"]

        rx = xgb_results[ticker]
        mu_te_x = rx["mu_te"]
        b_te_x = rx["b_te"]

        n_trades = min(n_show, len(y_te))
        idx = rng.choice(len(y_te), size=n_trades, replace=False)

        # ── Left column: Full GAMLSS ──
        ax_g = axes_fan[row, 0]
        sort_g = np.argsort(mu_te[idx])
        idx_g = idx[sort_g]
        x_pos = np.arange(n_trades)

        mu_s = mu_te[idx_g]
        b_s = b_te[idx_g]
        y_s = y_te[idx_g]

        # Smooth mu and b for the fan edges, keep raw for coverage calc
        mu_sm = smooth(mu_s)
        b_sm = smooth(b_s)

        for level, band_color, label in [
            (0.90, bands["90%"], "90% interval"),
            (0.80, bands["80%"], "80% interval"),
            (0.50, bands["50%"], "50% interval"),
        ]:
            z = np.log(1.0 / (1.0 - level))
            lo = np.maximum(mu_sm - z * b_sm, 0.0)
            hi = mu_sm + z * b_sm
            ax_g.fill_between(x_pos, lo, hi, alpha=0.7, color=band_color, label=label)

        ax_g.plot(x_pos, mu_sm, color="white", lw=1.8, label="Predicted median", zorder=3)
        ax_g.plot(x_pos, mu_sm, color=color, lw=1.2, zorder=3)
        ax_g.scatter(x_pos, y_s, s=10, color="black", alpha=0.5, zorder=4, label="Actual |impact|")

        # Coverage computed on raw (unsmoothed) predictions
        cov90, _ = compute_coverage(y_s, mu_s, b_s, 0.90)
        cov80, _ = compute_coverage(y_s, mu_s, b_s, 0.80)
        cov50, _ = compute_coverage(y_s, mu_s, b_s, 0.50)

        ax_g.set_xlabel("Trade index (sorted by predicted median)", fontsize=10)
        ax_g.set_ylabel("|impact| (bps)", fontsize=10)
        ax_g.set_title(f"{ticker} Full GAMLSS\n"
                       f"({n_trades} random test trades)",
                       fontsize=11, fontweight="bold")
        ax_g.legend(fontsize=8, loc="upper left", ncol=2)
        ax_g.grid(True, alpha=0.12)
        ax_g.spines["top"].set_visible(False)
        ax_g.spines["right"].set_visible(False)

        # ── Right column: Two-Stage XGBoost ──
        ax_x = axes_fan[row, 1]
        sort_x = np.argsort(mu_te_x[idx])
        idx_x = idx[sort_x]

        mu_sx = mu_te_x[idx_x]
        b_sx = b_te_x[idx_x]
        y_sx = y_te[idx_x]

        mu_sx_sm = smooth(mu_sx)
        b_sx_sm = smooth(b_sx)

        for level, band_color, label in [
            (0.90, bands["90%"], "90% interval"),
            (0.80, bands["80%"], "80% interval"),
            (0.50, bands["50%"], "50% interval"),
        ]:
            z = np.log(1.0 / (1.0 - level))
            lo = np.maximum(mu_sx_sm - z * b_sx_sm, 0.0)
            hi = mu_sx_sm + z * b_sx_sm
            ax_x.fill_between(x_pos, lo, hi, alpha=0.7, color=band_color, label=label)

        ax_x.plot(x_pos, mu_sx_sm, color="white", lw=1.8, zorder=3)
        ax_x.plot(x_pos, mu_sx_sm, color=color, lw=1.2, label="Predicted median", zorder=3)
        ax_x.scatter(x_pos, y_sx, s=10, color="black", alpha=0.5, zorder=4, label="Actual |impact|")

        # Coverage on raw predictions
        cov90x, _ = compute_coverage(y_sx, mu_sx, b_sx, 0.90)
        cov80x, _ = compute_coverage(y_sx, mu_sx, b_sx, 0.80)
        cov50x, _ = compute_coverage(y_sx, mu_sx, b_sx, 0.50)

        ax_x.set_xlabel("Trade index (sorted by predicted median)", fontsize=10)
        ax_x.set_ylabel("|impact| (bps)", fontsize=10)
        ax_x.set_title(f"{ticker} Two-Stage XGBoost\n"
                       f"({n_trades} random test trades)",
                       fontsize=11, fontweight="bold")
        ax_x.legend(fontsize=8, loc="upper left", ncol=2)
        ax_x.grid(True, alpha=0.12)
        ax_x.spines["top"].set_visible(False)
        ax_x.spines["right"].set_visible(False)

        # Match y-axis limits across the pair
        ymax = max(ax_g.get_ylim()[1], ax_x.get_ylim()[1])
        ax_g.set_ylim(0, ymax)
        ax_x.set_ylim(0, ymax)

    fig_fan.suptitle(
        "Prediction Interval Fan Charts: Full GAMLSS vs Two-Stage XGBoost\n"
        "50% / 80% / 90% Laplace Intervals | 300 Random Test Trades",
        fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig_fan.savefig("gamlss_fan_chart.png", dpi=150, bbox_inches="tight")
    print("Saved -> gamlss_fan_chart.png")
    plt.close(fig_fan)


# =============================================================================
# run_rerun_gamlss
# =============================================================================

def run_rerun_gamlss():
    """
    Rerun GAMLSS + visualization for all 6 stocks using saved feature parquets.
    Filters NVDA outliers (stock split on 2024-06-10) before fitting.
    """

    FEATURES_3 = ["roll_spread_500", "roll_vol_500", "participation_rate"]
    IMPACT_CAP_BPS = 500

    DATASETS = {
        "AAPL": ("data/lit_buy_features_v2.parquet", "data/lit_buy_features_v2_sep.parquet"),
        "COIN": ("data/coin_lit_buy_features_train.parquet", "data/coin_lit_buy_features_test.parquet"),
        "NVDA": ("data/nvda_lit_buy_features_train.parquet", "data/nvda_lit_buy_features_test.parquet"),
        "AMD":  ("data/amd_lit_buy_features_train.parquet", "data/amd_lit_buy_features_test.parquet"),
        "AMZN": ("data/amzn_lit_buy_features_train.parquet", "data/amzn_lit_buy_features_test.parquet"),
        "TSLA": ("data/tsla_lit_buy_features_train.parquet", "data/tsla_lit_buy_features_test.parquet"),
    }


    def run_gamlss(train_df, test_df, ticker):
        X_tr = train_df[FEATURES_3].to_numpy(dtype=np.float64)
        y_tr = train_df["impact_vwap_bps"].abs().to_numpy(dtype=np.float64)
        X_te = test_df[FEATURES_3].to_numpy(dtype=np.float64)
        y_te = test_df["impact_vwap_bps"].abs().to_numpy(dtype=np.float64)

        loc_model = XGBRegressor(
            objective="reg:absoluteerror", tree_method="hist",
            max_depth=3, n_estimators=200, learning_rate=0.07,
            min_child_weight=5, reg_alpha=10, reg_lambda=10,
            verbosity=0, random_state=42, n_jobs=1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loc_model.fit(X_tr, y_tr)

        mu_hat_tr = np.maximum(loc_model.predict(X_tr), 0.0)
        mu_hat_te = np.maximum(loc_model.predict(X_te), 0.0)
        loc_mae = np.mean(np.abs(y_te - mu_hat_te))

        abs_resid_tr = np.abs(y_tr - mu_hat_tr)
        scale_model = XGBRegressor(
            objective="reg:squarederror", tree_method="hist",
            max_depth=3, n_estimators=50, learning_rate=0.1,
            min_child_weight=20, reg_alpha=1, reg_lambda=1,
            verbosity=0, random_state=42, n_jobs=1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scale_model.fit(X_tr, abs_resid_tr)

        b_hat_te = np.clip(scale_model.predict(X_te), 0.1, None)

        coverages = {}
        for level in [0.50, 0.80, 0.90]:
            z = np.log(1.0 / (1.0 - level))
            lo = np.maximum(mu_hat_te - z * b_hat_te, 0.0)
            hi = mu_hat_te + z * b_hat_te
            cov = ((y_te >= lo) & (y_te <= hi)).mean()
            width = (hi - lo).mean()
            coverages[level] = (cov, width)

        return {
            "ticker": ticker,
            "n_train": len(train_df),
            "n_test": len(test_df),
            "mean_abs_impact": y_te.mean(),
            "loc_mae": loc_mae,
            "cov_90": coverages[0.90][0],
            "cov_80": coverages[0.80][0],
            "cov_50": coverages[0.50][0],
            "width_90": coverages[0.90][1],
        }


    all_results = []
    for ticker, (tr_path, te_path) in DATASETS.items():
        train_df = pd.read_parquet(tr_path)
        test_df = pd.read_parquet(te_path)

        # Filter extreme outliers
        n0_tr, n0_te = len(train_df), len(test_df)
        train_df = train_df[train_df["impact_vwap_bps"].abs() <= IMPACT_CAP_BPS]
        test_df = test_df[test_df["impact_vwap_bps"].abs() <= IMPACT_CAP_BPS]
        removed = (n0_tr - len(train_df)) + (n0_te - len(test_df))
        if removed > 0:
            print(f"  {ticker}: removed {removed} rows with |impact| > {IMPACT_CAP_BPS} bps")

        r = run_gamlss(train_df, test_df, ticker)
        all_results.append(r)
        print(f"  {ticker}: train={r['n_train']:,}, test={r['n_test']:,}, "
              f"MAE={r['loc_mae']:.4f}, 90% cov={r['cov_90']:.4f}")

    # Summary table
    print(f"\n{'=' * 100}")
    print(f"  CROSS-STOCK XGB GAMLSS VALIDATION (3 features: spread, vol, participation)")
    print(f"  Location: XGB LAD (depth=3, n=200) | Scale: XGB MSE (depth=3, n=50)")
    print(f"  Train: Jun-Aug 2024 | Test: Sep 2024 | Laplace intervals")
    print(f"{'=' * 100}")
    print(f"  {'Ticker':<8} {'n_train':>8} {'n_test':>7} {'Mean|imp|':>10} "
          f"{'Loc MAE':>8} {'90% Cov':>8} {'80% Cov':>8} {'50% Cov':>8} "
          f"{'90% Width':>10}")
    print(f"  {'-' * 8} {'-' * 8} {'-' * 7} {'-' * 10} {'-' * 8} "
          f"{'-' * 8} {'-' * 8} {'-' * 8} {'-' * 10}")

    for r in all_results:
        print(f"  {r['ticker']:<8} {r['n_train']:>8,} {r['n_test']:>7,} "
              f"{r['mean_abs_impact']:>10.4f} {r['loc_mae']:>8.4f} "
              f"{r['cov_90']:>8.4f} {r['cov_80']:>8.4f} {r['cov_50']:>8.4f} "
              f"{r['width_90']:>10.4f}")

    print(f"\n  Nominal targets:  90% -> 0.9000   80% -> 0.8000   50% -> 0.5000")

    # Visualization
    results_df = pd.DataFrame(all_results)
    n_stocks = len(results_df)
    tickers_ordered = results_df["ticker"].tolist()
    colors = ["#2563eb", "#dc2626", "#16a34a", "#f59e0b", "#7c3aed", "#ec4899"][:n_stocks]

    fig = plt.figure(figsize=(24, 14))
    gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.32, hspace=0.42)

    x = np.arange(n_stocks)
    bar_w = 0.55

    # Panel 1: Coverage
    ax1 = fig.add_subplot(gs[0, 0])
    bw = 0.22
    for j, (level, label) in enumerate([(0.90, "90%"), (0.80, "80%"), (0.50, "50%")]):
        col_name = f"cov_{int(level*100):d}"
        vals = results_df[col_name].values
        offset = (j - 1) * bw
        ax1.bar(x + offset, vals, width=bw, label=f"{label} actual",
                alpha=0.85, edgecolor="white", linewidth=0.5)
        ax1.axhline(level, color="gray", lw=0.8, ls=":", alpha=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(tickers_ordered, fontsize=10, fontweight="bold")
    ax1.set_ylabel("Actual Coverage", fontsize=11)
    ax1.set_title("Coverage by Stock\n(dashed = nominal)", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.2)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.set_ylim(0, 1.05)

    # Panel 2: Location MAE
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(x, results_df["loc_mae"].values, color=colors, width=bar_w,
            edgecolor="white", linewidth=0.8)
    for i, v in enumerate(results_df["loc_mae"].values):
        ax2.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(tickers_ordered, fontsize=10, fontweight="bold")
    ax2.set_ylabel("MAE (bps)", fontsize=11)
    ax2.set_title("Location Model MAE (OOS)", fontsize=12, fontweight="bold")
    ax2.grid(axis="y", alpha=0.2)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Panel 3: Mean absolute impact
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.bar(x, results_df["mean_abs_impact"].values, color=colors, width=bar_w,
            edgecolor="white", linewidth=0.8)
    for i, v in enumerate(results_df["mean_abs_impact"].values):
        ax3.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(tickers_ordered, fontsize=10, fontweight="bold")
    ax3.set_ylabel("|impact| (bps)", fontsize=11)
    ax3.set_title("Mean |impact| on Test Set", fontsize=12, fontweight="bold")
    ax3.grid(axis="y", alpha=0.2)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    # Panel 4: 90% interval width
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.bar(x, results_df["width_90"].values, color=colors, width=bar_w,
            edgecolor="white", linewidth=0.8)
    for i, v in enumerate(results_df["width_90"].values):
        ax4.text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")
    ax4.set_xticks(x)
    ax4.set_xticklabels(tickers_ordered, fontsize=10, fontweight="bold")
    ax4.set_ylabel("Width (bps)", fontsize=11)
    ax4.set_title("Mean 90% Interval Width", fontsize=12, fontweight="bold")
    ax4.grid(axis="y", alpha=0.2)
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)

    # Panel 5: Coverage deviation heatmap
    ax5 = fig.add_subplot(gs[1, 1])
    cov_matrix = np.array([
        [r["cov_90"] - 0.90, r["cov_80"] - 0.80, r["cov_50"] - 0.50]
        for r in all_results
    ])
    lim = max(abs(cov_matrix.min()), abs(cov_matrix.max())) * 1.1
    im = ax5.imshow(cov_matrix, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax5.set_xticks([0, 1, 2])
    ax5.set_xticklabels(["90%", "80%", "50%"], fontsize=10)
    ax5.set_yticks(range(n_stocks))
    ax5.set_yticklabels(tickers_ordered, fontsize=10, fontweight="bold")
    for i in range(n_stocks):
        for j in range(3):
            v = cov_matrix[i, j]
            text_color = "white" if abs(v) > lim * 0.5 else "black"
            ax5.text(j, i, f"{v:+.3f}", ha="center", va="center",
                     fontsize=9.5, fontweight="bold", color=text_color)
    cbar = plt.colorbar(im, ax=ax5, fraction=0.04, pad=0.04)
    cbar.set_label("Coverage - Nominal", fontsize=9)
    ax5.set_title("Coverage Deviation from Nominal\n(blue=under, red=over)",
                  fontsize=12, fontweight="bold")

    # Panel 6: Sample size
    ax6 = fig.add_subplot(gs[1, 2])
    train_sizes = results_df["n_train"].values
    test_sizes = results_df["n_test"].values
    ax6.bar(x - 0.15, train_sizes, width=0.3, color="#64748b", label="Train",
            edgecolor="white", linewidth=0.5)
    ax6.bar(x + 0.15, test_sizes, width=0.3, color="#f59e0b", label="Test",
            edgecolor="white", linewidth=0.5)
    for i, (tr, te) in enumerate(zip(train_sizes, test_sizes)):
        ax6.text(i - 0.15, tr + 50, f"{tr:,}", ha="center", fontsize=7.5,
                 fontweight="bold", rotation=45)
        ax6.text(i + 0.15, te + 50, f"{te:,}", ha="center", fontsize=7.5,
                 fontweight="bold", rotation=45)
    ax6.set_xticks(x)
    ax6.set_xticklabels(tickers_ordered, fontsize=10, fontweight="bold")
    ax6.set_ylabel("Number of trades", fontsize=11)
    ax6.set_title("Dataset Size (Train vs Test)", fontsize=12, fontweight="bold")
    ax6.legend(fontsize=9)
    ax6.grid(axis="y", alpha=0.2)
    ax6.spines["top"].set_visible(False)
    ax6.spines["right"].set_visible(False)

    fig.suptitle(
        "Cross-Stock XGB GAMLSS Validation: Block Trade Impact Prediction\n"
        "3 features (spread, volatility, participation rate) | "
        "Fixed AAPL hyperparameters | Train: Jun-Aug, Test: Sep 2024",
        fontsize=14, fontweight="bold", y=1.01,
    )

    fig.savefig("cross_stock_validation.png", dpi=150, bbox_inches="tight")
    print("\nSaved -> cross_stock_validation.png")

    results_df.to_csv("data/cross_stock_results.csv", index=False)
    print("Saved -> data/cross_stock_results.csv")


# =============================================================================
# run_pooled_gamlss
# =============================================================================

def run_pooled_gamlss():
    """
    Pooled XGB GAMLSS: single model trained on normalized, pooled data from 6 stocks.
    Tests on each stock's September holdout individually.

    Key idea: normalize each stock's features by its training-set median so that
    "median spread for COIN" and "median spread for AAPL" both map to 1.0.
    The target is also normalized by median |impact|.

    Output:
      - pooled_vs_perstock.png
      - Comparison table: per-stock vs pooled
    """

    FEATURES_3 = ["roll_spread_500", "roll_vol_500", "participation_rate"]
    IMPACT_CAP_BPS = 500

    DATASETS = {
        "AAPL": ("data/lit_buy_features_v2.parquet", "data/lit_buy_features_v2_sep.parquet"),
        "COIN": ("data/coin_lit_buy_features_train.parquet", "data/coin_lit_buy_features_test.parquet"),
        "NVDA": ("data/nvda_lit_buy_features_train.parquet", "data/nvda_lit_buy_features_test.parquet"),
        "AMD":  ("data/amd_lit_buy_features_train.parquet", "data/amd_lit_buy_features_test.parquet"),
        "AMZN": ("data/amzn_lit_buy_features_train.parquet", "data/amzn_lit_buy_features_test.parquet"),
        "TSLA": ("data/tsla_lit_buy_features_train.parquet", "data/tsla_lit_buy_features_test.parquet"),
    }

    LOC_PARAMS = dict(max_depth=3, n_estimators=200, learning_rate=0.07,
                      min_child_weight=5, reg_alpha=10, reg_lambda=10)
    SCALE_PARAMS = dict(max_depth=3, n_estimators=50, learning_rate=0.1,
                        min_child_weight=20, reg_alpha=1, reg_lambda=1)


    # ═════════════════════════════════════════════════════════════════════════════
    # STEP 1: Load and pool training data
    # ═════════════════════════════════════════════════════════════════════════════
    print("STEP 1: Loading and normalizing training data...")

    stock_data = {}
    pooled_X = []
    pooled_y = []

    for ticker, (tr_f, te_f) in DATASETS.items():
        df_tr = pd.read_parquet(tr_f)
        df_te = pd.read_parquet(te_f)
        df_tr = df_tr[df_tr["impact_vwap_bps"].abs() <= IMPACT_CAP_BPS]
        df_te = df_te[df_te["impact_vwap_bps"].abs() <= IMPACT_CAP_BPS]
        df_tr["abs_impact"] = df_tr["impact_vwap_bps"].abs()
        df_te["abs_impact"] = df_te["impact_vwap_bps"].abs()

        # Training-set medians for normalization
        medians = {}
        for feat in FEATURES_3:
            medians[feat] = df_tr[feat].median()
            # Guard against zero median
            if medians[feat] == 0 or np.isnan(medians[feat]):
                medians[feat] = df_tr[feat].mean()
                if medians[feat] == 0:
                    medians[feat] = 1.0

        median_impact = df_tr["abs_impact"].median()
        if median_impact == 0 or np.isnan(median_impact):
            median_impact = df_tr["abs_impact"].mean()

        # Normalize training features and target
        X_tr_norm = np.column_stack([
            df_tr[feat].values / medians[feat] for feat in FEATURES_3
        ])
        y_tr_norm = df_tr["abs_impact"].values / median_impact

        # Normalize test features (same medians from training)
        X_te_norm = np.column_stack([
            df_te[feat].values / medians[feat] for feat in FEATURES_3
        ])
        y_te_raw = df_te["abs_impact"].values

        stock_data[ticker] = {
            "medians": medians,
            "median_impact": median_impact,
            "X_tr_norm": X_tr_norm,
            "y_tr_norm": y_tr_norm,
            "X_te_norm": X_te_norm,
            "y_te_raw": y_te_raw,
            "n_train": len(df_tr),
            "n_test": len(df_te),
        }

        pooled_X.append(X_tr_norm)
        pooled_y.append(y_tr_norm)

    pooled_X = np.vstack(pooled_X)
    pooled_y = np.concatenate(pooled_y)

    print(f"\n  {'Ticker':<8} {'n_train':>8} {'med_impact':>11} {'med_spread':>11} "
          f"{'med_vol':>10} {'med_prate':>10}")
    print(f"  {'-'*8} {'-'*8} {'-'*11} {'-'*11} {'-'*10} {'-'*10}")
    for ticker in DATASETS:
        d = stock_data[ticker]
        m = d["medians"]
        print(f"  {ticker:<8} {d['n_train']:>8,} {d['median_impact']:>11.4f} "
              f"{m['roll_spread_500']:>11.4f} {m['roll_vol_500']:>10.4f} "
              f"{m['participation_rate']:>10.6f}")
    print(f"\n  Pooled training set: {len(pooled_y):,} trades")


    # ═════════════════════════════════════════════════════════════════════════════
    # STEP 2: Fit pooled location model
    # ═════════════════════════════════════════════════════════════════════════════
    print("\nSTEP 2: Fitting pooled location model (XGB LAD)...")

    loc_model = XGBRegressor(
        objective="reg:absoluteerror", tree_method="hist",
        verbosity=0, random_state=42, n_jobs=1, **LOC_PARAMS,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loc_model.fit(pooled_X, pooled_y)

    mu_pooled_tr = np.maximum(loc_model.predict(pooled_X), 0.0)
    mae_tr_norm = np.mean(np.abs(pooled_y - mu_pooled_tr))
    print(f"  Pooled train MAE (normalized): {mae_tr_norm:.4f}")


    # ═════════════════════════════════════════════════════════════════════════════
    # STEP 3: Fit pooled scale model
    # ═════════════════════════════════════════════════════════════════════════════
    print("\nSTEP 3: Fitting pooled scale model (XGB MSE on |residuals|)...")

    abs_resid_tr = np.abs(pooled_y - mu_pooled_tr)
    scale_model = XGBRegressor(
        objective="reg:squarederror", tree_method="hist",
        verbosity=0, random_state=42, n_jobs=1, **SCALE_PARAMS,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scale_model.fit(pooled_X, abs_resid_tr)

    print(f"  Scale model fitted on {len(abs_resid_tr):,} residuals")


    # ═════════════════════════════════════════════════════════════════════════════
    # STEP 4: Evaluate per stock
    # ═════════════════════════════════════════════════════════════════════════════
    print("\nSTEP 4: Evaluating pooled model on each stock's September holdout...\n")

    pooled_results = {}

    for ticker in DATASETS:
        d = stock_data[ticker]
        X_te_norm = d["X_te_norm"]
        y_te = d["y_te_raw"]
        scale = d["median_impact"]

        # Predict in normalized space, then un-normalize
        mu_te_norm = np.maximum(loc_model.predict(X_te_norm), 0.0)
        b_te_norm = np.clip(scale_model.predict(X_te_norm), 0.01, None)

        mu_te = mu_te_norm * scale
        b_te = np.clip(b_te_norm * scale, 0.1, None)

        mae_te = np.mean(np.abs(y_te - mu_te))

        cov_data = {}
        for level in [0.50, 0.80, 0.90]:
            z = np.log(1.0 / (1.0 - level))
            lo = np.maximum(mu_te - z * b_te, 0.0)
            hi = mu_te + z * b_te
            cov = ((y_te >= lo) & (y_te <= hi)).mean()
            width = (hi - lo).mean()
            cov_data[level] = (cov, width)

        pooled_results[ticker] = {
            "mae_te": mae_te,
            "coverage_data": cov_data,
            "n_test": len(y_te),
        }

        print(f"  {ticker}: MAE={mae_te:.4f}, "
              f"90% cov={cov_data[0.90][0]:.4f}, "
              f"80% cov={cov_data[0.80][0]:.4f}, "
              f"50% cov={cov_data[0.50][0]:.4f}, "
              f"90% width={cov_data[0.90][1]:.2f}")


    # ═════════════════════════════════════════════════════════════════════════════
    # STEP 5: Fit per-stock models for comparison
    # ═════════════════════════════════════════════════════════════════════════════
    print("\nFitting per-stock models for comparison...")

    perstock_results = {}

    for ticker, (tr_f, te_f) in DATASETS.items():
        df_tr = pd.read_parquet(tr_f)
        df_te = pd.read_parquet(te_f)
        df_tr = df_tr[df_tr["impact_vwap_bps"].abs() <= IMPACT_CAP_BPS]
        df_te = df_te[df_te["impact_vwap_bps"].abs() <= IMPACT_CAP_BPS]
        df_tr["abs_impact"] = df_tr["impact_vwap_bps"].abs()
        df_te["abs_impact"] = df_te["impact_vwap_bps"].abs()

        X_tr = df_tr[FEATURES_3].to_numpy(dtype=np.float64)
        y_tr = df_tr["abs_impact"].to_numpy(dtype=np.float64)
        X_te = df_te[FEATURES_3].to_numpy(dtype=np.float64)
        y_te = df_te["abs_impact"].to_numpy(dtype=np.float64)

        loc_ps = XGBRegressor(objective="reg:absoluteerror", tree_method="hist",
                              verbosity=0, random_state=42, n_jobs=1, **LOC_PARAMS)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loc_ps.fit(X_tr, y_tr)
        mu_tr_ps = np.maximum(loc_ps.predict(X_tr), 0.0)
        mu_te_ps = np.maximum(loc_ps.predict(X_te), 0.0)

        abs_r_ps = np.abs(y_tr - mu_tr_ps)
        sc_ps = XGBRegressor(objective="reg:squarederror", tree_method="hist",
                             verbosity=0, random_state=42, n_jobs=1, **SCALE_PARAMS)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sc_ps.fit(X_tr, abs_r_ps)
        b_te_ps = np.clip(sc_ps.predict(X_te), 0.1, None)

        mae_ps = np.mean(np.abs(y_te - mu_te_ps))

        cov_data_ps = {}
        for level in [0.50, 0.80, 0.90]:
            z = np.log(1.0 / (1.0 - level))
            lo = np.maximum(mu_te_ps - z * b_te_ps, 0.0)
            hi = mu_te_ps + z * b_te_ps
            cov = ((y_te >= lo) & (y_te <= hi)).mean()
            width = (hi - lo).mean()
            cov_data_ps[level] = (cov, width)

        perstock_results[ticker] = {
            "mae_te": mae_ps,
            "coverage_data": cov_data_ps,
        }


    # ═════════════════════════════════════════════════════════════════════════════
    # STEP 5: Comparison table
    # ═════════════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 95}")
    print("  COMPARISON: Per-Stock XGB GAMLSS vs Pooled XGB GAMLSS")
    print(f"{'=' * 95}")
    print(f"  {'Ticker':<8} {'n_test':>7} "
          f"{'PS 90%':>7} {'Pool 90%':>8} "
          f"{'PS MAE':>7} {'Pool MAE':>9} "
          f"{'PS Width':>9} {'Pool Width':>11}")
    print(f"  {'-'*8} {'-'*7} "
          f"{'-'*7} {'-'*8} "
          f"{'-'*7} {'-'*9} "
          f"{'-'*9} {'-'*11}")

    for ticker in DATASETS:
        ps = perstock_results[ticker]
        pl = pooled_results[ticker]
        ps_cd = ps["coverage_data"]
        pl_cd = pl["coverage_data"]
        print(f"  {ticker:<8} {pl['n_test']:>7,} "
              f"{ps_cd[0.90][0]:>7.4f} {pl_cd[0.90][0]:>8.4f} "
              f"{ps['mae_te']:>7.4f} {pl['mae_te']:>9.4f} "
              f"{ps_cd[0.90][1]:>9.2f} {pl_cd[0.90][1]:>11.2f}")

    print(f"\n  Nominal 90% target: 0.9000")


    # ═════════════════════════════════════════════════════════════════════════════
    # STEP 6: Figure
    # ═════════════════════════════════════════════════════════════════════════════
    print("\nPlotting comparison...")

    tickers_ordered = list(DATASETS.keys())
    x = np.arange(len(tickers_ordered))
    bw = 0.35

    COLORS_PS = "#2563eb"
    COLORS_PL = "#dc2626"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))

    # Left panel: 90% coverage
    cov_ps = [perstock_results[t]["coverage_data"][0.90][0] for t in tickers_ordered]
    cov_pl = [pooled_results[t]["coverage_data"][0.90][0] for t in tickers_ordered]

    bars1 = ax1.bar(x - bw/2, cov_ps, width=bw, color=COLORS_PS, alpha=0.8,
                    edgecolor="white", linewidth=0.8, label="Per-Stock")
    bars2 = ax1.bar(x + bw/2, cov_pl, width=bw, color=COLORS_PL, alpha=0.8,
                    edgecolor="white", linewidth=0.8, label="Pooled")

    ax1.axhline(0.90, color="black", lw=1.2, ls="--", alpha=0.5, label="Nominal 90%")

    for i, (v1, v2) in enumerate(zip(cov_ps, cov_pl)):
        ax1.text(i - bw/2, v1 + 0.005, f"{v1:.3f}", ha="center", fontsize=7.5, color=COLORS_PS)
        ax1.text(i + bw/2, v2 + 0.005, f"{v2:.3f}", ha="center", fontsize=7.5, color=COLORS_PL)

    ax1.set_xticks(x)
    ax1.set_xticklabels(tickers_ordered, fontsize=10, fontweight="bold")
    ax1.set_ylabel("Actual Coverage", fontsize=11)
    ax1.set_title("90% Coverage: Per-Stock vs Pooled", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9, loc="lower right")
    ax1.set_ylim(0.75, 1.0)
    ax1.grid(axis="y", alpha=0.2)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Right panel: MAE
    mae_ps = [perstock_results[t]["mae_te"] for t in tickers_ordered]
    mae_pl = [pooled_results[t]["mae_te"] for t in tickers_ordered]

    bars3 = ax2.bar(x - bw/2, mae_ps, width=bw, color=COLORS_PS, alpha=0.8,
                    edgecolor="white", linewidth=0.8, label="Per-Stock")
    bars4 = ax2.bar(x + bw/2, mae_pl, width=bw, color=COLORS_PL, alpha=0.8,
                    edgecolor="white", linewidth=0.8, label="Pooled")

    for i, (v1, v2) in enumerate(zip(mae_ps, mae_pl)):
        ax2.text(i - bw/2, v1 + 0.03, f"{v1:.3f}", ha="center", fontsize=7.5, color=COLORS_PS)
        ax2.text(i + bw/2, v2 + 0.03, f"{v2:.3f}", ha="center", fontsize=7.5, color=COLORS_PL)

    ax2.set_xticks(x)
    ax2.set_xticklabels(tickers_ordered, fontsize=10, fontweight="bold")
    ax2.set_ylabel("MAE (bps)", fontsize=11)
    ax2.set_title("Location MAE: Per-Stock vs Pooled", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.2)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle("Pooled XGB GAMLSS (median-normalized, 6 stocks) vs Per-Stock Models\n"
                 "3 Features | Fixed AAPL Hyperparameters | Train: Jun-Aug 2024 | Test: Sep 2024",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig("pooled_vs_perstock.png", dpi=150, bbox_inches="tight")
    print("Saved -> pooled_vs_perstock.png")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    run_gamlss_laplace()
    run_gamlss_xgb()
    run_gamlss_xgb_gengauss()
    run_gamlss_full()
    run_rerun_gamlss()
    run_pooled_gamlss()
