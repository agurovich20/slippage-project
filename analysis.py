"""
Coverage tables, Huber losses, model comparison, prediction interval plots, and SHAP analysis
for the AAPL/COIN slippage models. Most of the heavy lifting is in run_model_comparison_v2
(5-model walk-forward CV) and run_prediction_intervals (the two-stage GAMLSS fan charts).
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"]      = "1"

import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
from xgboost import XGBRegressor
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import shap
from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def run_compute_tables():
    FEATURES = [
        "dollar_value", "log_dollar_value", "participation_rate",
        "roll_spread_500", "roll_vol_500", "exchange_id",
    ]

    datasets = {
        "AAPL": ("data/lit_buy_features_v2.parquet", "data/lit_buy_features_v2_sep.parquet"),
        "COIN": ("data/coin_lit_buy_features_train.parquet", "data/coin_lit_buy_features_test.parquet"),
    }

    def fit_linear_gamlss(X_tr, y_tr, X_te):
        X_tr_c = sm.add_constant(X_tr)
        X_te_c = sm.add_constant(X_te)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            qr = sm.QuantReg(y_tr, X_tr_c).fit(q=0.5, max_iter=5000)
        mu_tr = qr.predict(X_tr_c)
        mu_te = qr.predict(X_te_c)
        abs_r = np.abs(y_tr - mu_tr)
        Xg_tr = np.column_stack([np.ones(len(X_tr)), X_tr])
        Xg_te = np.column_stack([np.ones(len(X_te)), X_te])
        gamma, _, _, _ = np.linalg.lstsq(Xg_tr, abs_r, rcond=None)
        b_te = np.clip(Xg_te @ gamma, 0.1, None)
        return mu_te, b_te

    def fit_xgb_gamlss(X_tr, y_tr, X_te):
        loc = XGBRegressor(objective="reg:absoluteerror", tree_method="hist", verbosity=0,
                           random_state=42, n_jobs=1, max_depth=3, n_estimators=200,
                           learning_rate=0.07, min_child_weight=5, reg_alpha=10, reg_lambda=10)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loc.fit(X_tr, y_tr)
        mu_tr = np.maximum(loc.predict(X_tr), 0.0)
        mu_te = np.maximum(loc.predict(X_te), 0.0)
        abs_r = np.abs(y_tr - mu_tr)
        sc = XGBRegressor(objective="reg:squarederror", tree_method="hist", verbosity=0,
                          random_state=42, n_jobs=1, max_depth=5, n_estimators=50,
                          min_child_weight=20, learning_rate=0.1, reg_alpha=1, reg_lambda=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sc.fit(X_tr, abs_r)
        b_te = np.clip(sc.predict(X_te), 0.1, None)
        return mu_te, b_te

    def compute_coverage(y, mu, b, level):
        z = np.log(1.0 / (1.0 - level))
        lo = np.maximum(mu - z * b, 0.0)
        hi = mu + z * b
        cov = ((y >= lo) & (y <= hi)).mean()
        width = (hi - lo).mean()
        return cov, width

    # load and score
    results = []
    for ticker, (tr_f, te_f) in datasets.items():
        df_tr = pd.read_parquet(tr_f)
        df_te = pd.read_parquet(te_f)
        df_tr["abs_impact"] = df_tr["impact_vwap_bps"].abs()
        df_te["abs_impact"] = df_te["impact_vwap_bps"].abs()
        df_tr = df_tr.sort_values("date").reset_index(drop=True)

        X_tr = df_tr[FEATURES].to_numpy(dtype=np.float64)
        y_tr = df_tr["abs_impact"].to_numpy(dtype=np.float64)
        X_te = df_te[FEATURES].to_numpy(dtype=np.float64)
        y_te = df_te["abs_impact"].to_numpy(dtype=np.float64)

        print(f"{ticker}: train={len(y_tr)}, test={len(y_te)}")

        # linear
        mu_lin, b_lin = fit_linear_gamlss(X_tr, y_tr, X_te)
        mae_lin = np.mean(np.abs(y_te - mu_lin))
        cov90_lin, w90_lin = compute_coverage(y_te, mu_lin, b_lin, 0.90)

        # XGBoost
        mu_xgb, b_xgb = fit_xgb_gamlss(X_tr, y_tr, X_te)
        mae_xgb = np.mean(np.abs(y_te - mu_xgb))
        cov90_xgb, w90_xgb = compute_coverage(y_te, mu_xgb, b_xgb, 0.90)

        results.append({
            "ticker": ticker,
            "lin_mae": mae_lin, "lin_w90": w90_lin, "lin_cov90": cov90_lin,
            "xgb_mae": mae_xgb, "xgb_w90": w90_xgb, "xgb_cov90": cov90_xgb,
        })

    # print comparison tables
    print("\n" + "=" * 70)
    print("  Two-Stage Linear Regression")
    print("=" * 70)
    print(f"  {'Ticker':<8} {'Location MAE':>14} {'90% Width':>12} {'90% Coverage':>14}")
    print(f"  {'------':<8} {'------------':>14} {'---------':>12} {'------------':>14}")
    for r in results:
        print(f"  {r['ticker']:<8} {r['lin_mae']:>14.4f} {r['lin_w90']:>12.4f} {r['lin_cov90']:>14.4f}")

    print("\n" + "=" * 70)
    print("  Two-Stage XGBoost")
    print("=" * 70)
    print(f"  {'Ticker':<8} {'Location MAE':>14} {'90% Width':>12} {'90% Coverage':>14}")
    print(f"  {'------':<8} {'------------':>14} {'---------':>12} {'------------':>14}")
    for r in results:
        print(f"  {r['ticker']:<8} {r['xgb_mae']:>14.4f} {r['xgb_w90']:>12.4f} {r['xgb_cov90']:>14.4f}")

    print("\n" + "=" * 70)
    print("  Combined Comparison")
    print("=" * 70)
    print(f"  {'Ticker':<8} {'Model':<28} {'Location MAE':>14} {'90% Width':>12} {'90% Coverage':>14}")
    print(f"  {'------':<8} {'-----':<28} {'------------':>14} {'---------':>12} {'------------':>14}")
    for r in results:
        print(f"  {r['ticker']:<8} {'Two-Stage Linear Regression':<28} {r['lin_mae']:>14.4f} {r['lin_w90']:>12.4f} {r['lin_cov90']:>14.4f}")
        print(f"  {'':<8} {'Two-Stage XGBoost':<28} {r['xgb_mae']:>14.4f} {r['xgb_w90']:>12.4f} {r['xgb_cov90']:>14.4f}")


def run_compute_huber_delta2():
    """
    Compute Huber loss (delta=1 and delta=2) for all 6 models.

    Replicates the exact setup used for the article table (medium_article_v2.md):
      - Train: Jun-Aug 2024  (data/lit_buy_features_v2.parquet,     35,020 trades)
      - Test:  Sep 2024      (data/lit_buy_features_v2_sep.parquet,   9,152 trades)
      - Target: abs(impact_vwap_bps)
      - Features (6): dollar_value, log_dollar_value, participation_rate,
                      roll_spread_500, roll_vol_500, exchange_id

    Best hyperparameters loaded directly from the saved gridsearch CSV files:
      - XGB-MSE : data/gridsearch_details_6feat.csv
      - XGB-LAD : data/gridsearch_xgb_lad.csv
      - RF-MSE  : data/gridsearch_rf_mse.csv
      - RF-LAD  : data/gridsearch_rf_lad.csv

    OLS and OLS-LAD have no hyperparameters to tune.

    Huber loss definition (same as sklearn mean_tweedie_deviance / standard formula):
      h_delta(e) = 0.5*e^2            if |e| <= delta
                   delta*(|e| - 0.5*delta)   otherwise
      Huber(delta) = mean(h_delta(y - y_hat))
    """

    # data
    FEATURES = [
        "dollar_value", "log_dollar_value", "participation_rate",
        "roll_spread_500", "roll_vol_500", "exchange_id",
    ]

    df_tr = pd.read_parquet("data/lit_buy_features_v2.parquet")
    df_te = pd.read_parquet("data/lit_buy_features_v2_sep.parquet")
    df_tr["abs_impact"] = df_tr["impact_vwap_bps"].abs()
    df_te["abs_impact"] = df_te["impact_vwap_bps"].abs()
    df_tr = df_tr.sort_values("date").reset_index(drop=True)

    X_tr = df_tr[FEATURES].to_numpy(dtype=np.float64)
    y_tr = df_tr["abs_impact"].to_numpy(dtype=np.float64)
    X_te = df_te[FEATURES].to_numpy(dtype=np.float64)
    y_te = df_te["abs_impact"].to_numpy(dtype=np.float64)

    print(f"Train: {len(df_tr):,} trades  |  Test: {len(df_te):,} trades")
    print(f"Test median |impact|: {np.median(y_te):.4f} bps")

    # metric helpers
    def r2(ytrue, ypred):
        ss_res = ((ytrue - ypred) ** 2).sum()
        ss_tot = ((ytrue - ytrue.mean()) ** 2).sum()
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    def mae(ytrue, ypred):
        return np.mean(np.abs(ytrue - ypred))

    def medae(ytrue, ypred):
        return np.median(np.abs(ytrue - ypred))

    def huber(ytrue, ypred, delta):
        e = ytrue - ypred
        abs_e = np.abs(e)
        return np.mean(np.where(abs_e <= delta, 0.5 * e**2, delta * (abs_e - 0.5 * delta)))

    # load best hyperparameters from gridsearch CSVs
    def best_params(csv_path, param_cols):
        df = pd.read_csv(csv_path)
        row = df.sort_values("rank").iloc[0]
        return {c: row[c] for c in param_cols}

    xgb_cols = ["max_depth", "n_estimators", "learning_rate", "min_child_weight",
                "reg_alpha", "reg_lambda"]
    rf_cols  = ["max_depth", "n_estimators", "min_samples_leaf", "max_features", "bootstrap"]

    best_xgb_mse = best_params("data/gridsearch_details_6feat.csv", xgb_cols)
    best_xgb_lad = best_params("data/gridsearch_xgb_lad.csv",       xgb_cols)
    best_rf_mse  = best_params("data/gridsearch_rf_mse.csv",         rf_cols)
    best_rf_lad  = best_params("data/gridsearch_rf_lad.csv",         rf_cols)

    # Cast int params
    for d in (best_xgb_mse, best_xgb_lad):
        d["max_depth"]        = int(d["max_depth"])
        d["n_estimators"]     = int(d["n_estimators"])
        d["min_child_weight"] = int(d["min_child_weight"])
    for d in (best_rf_mse, best_rf_lad):
        d["n_estimators"]     = int(d["n_estimators"])
        d["min_samples_leaf"] = int(d["min_samples_leaf"])
        d["bootstrap"]        = bool(d["bootstrap"])
        # max_depth: keep as int unless None
        if not pd.isna(d["max_depth"]):
            d["max_depth"] = int(d["max_depth"])
        else:
            d["max_depth"] = None
        # max_features: convert numeric string to float if needed
        try:
            d["max_features"] = float(d["max_features"])
        except (ValueError, TypeError):
            pass  # leave as string ('sqrt', 'log2', etc.)

    print("\nBest hyperparameters loaded from CSVs:")
    print(f"  XGB-MSE : {best_xgb_mse}")
    print(f"  XGB-LAD : {best_xgb_lad}")
    print(f"  RF-MSE  : {best_rf_mse}")
    print(f"  RF-LAD  : {best_rf_lad}")

    # train all 6 models
    print("\nTraining models...")

    # (1) OLS
    beta_ols, *_ = np.linalg.lstsq(
        np.column_stack([X_tr, np.ones(len(X_tr))]), y_tr, rcond=None
    )
    pred_ols = np.maximum(np.column_stack([X_te, np.ones(len(X_te))]) @ beta_ols, 0.0)
    print("  OLS done")

    # (2) OLS-LAD
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lad_res = QuantReg(y_tr, np.column_stack([X_tr, np.ones(len(X_tr))])).fit(
            q=0.5, max_iter=5000, p_tol=1e-6
        )
    pred_lad = np.maximum(
        np.column_stack([X_te, np.ones(len(X_te))]) @ lad_res.params, 0.0
    )
    print("  OLS-LAD done")

    # (3) XGB-MSE
    xgb_mse = XGBRegressor(
        objective="reg:squarederror", tree_method="hist",
        verbosity=0, random_state=42, n_jobs=1, **best_xgb_mse
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xgb_mse.fit(X_tr, y_tr)
    pred_xgb_mse = np.maximum(xgb_mse.predict(X_te), 0.0)
    print("  XGB-MSE done")

    # (4) XGB-LAD
    xgb_lad = XGBRegressor(
        objective="reg:absoluteerror", tree_method="hist",
        verbosity=0, random_state=42, n_jobs=1, **best_xgb_lad
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xgb_lad.fit(X_tr, y_tr)
    pred_xgb_lad = np.maximum(xgb_lad.predict(X_te), 0.0)
    print("  XGB-LAD done")

    # (5) RF-MSE
    rf_mse = RandomForestRegressor(
        criterion="squared_error", random_state=42, n_jobs=1, **best_rf_mse
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rf_mse.fit(X_tr, y_tr)
    pred_rf_mse = np.maximum(rf_mse.predict(X_te), 0.0)
    print("  RF-MSE done")

    # (6) RF-LAD
    rf_lad = RandomForestRegressor(
        criterion="absolute_error", random_state=42, n_jobs=1, **best_rf_lad
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rf_lad.fit(X_tr, y_tr)
    pred_rf_lad = np.maximum(rf_lad.predict(X_te), 0.0)
    print("  RF-LAD done")

    # compute metrics
    MODELS = [
        ("XGB-LAD", pred_xgb_lad),
        ("RF-LAD",  pred_rf_lad),
        ("OLS-LAD", pred_lad),
        ("RF-MSE",  pred_rf_mse),
        ("XGB-MSE", pred_xgb_mse),
        ("OLS",     pred_ols),
    ]

    print(f"\n{'='*80}")
    print("  AAPL Sep 2024 holdout  |  target: abs(impact_vwap_bps)")
    print(f"{'='*80}")
    print(f"  {'Model':<10}  {'R2':>8}  {'MAE':>8}  {'MedAE':>8}  {'Huber(d=1)':>12}  {'Huber(d=2)':>12}")
    print(f"  {'-'*72}")

    rows = []
    for name, pred in MODELS:
        rv      = r2(y_te, pred)
        mv      = mae(y_te, pred)
        mev     = medae(y_te, pred)
        h1      = huber(y_te, pred, delta=1)
        h2      = huber(y_te, pred, delta=2)
        rows.append(dict(model=name, r2=rv, mae=mv, medae=mev, huber1=h1, huber2=h2))
        print(f"  {name:<10}  {rv:>+8.4f}  {mv:>8.4f}  {mev:>8.4f}  {h1:>12.4f}  {h2:>12.4f}")

    print(f"{'='*80}")

    # Article reference (delta=1 only)
    print("""
  Article table (medium_article_v2.md) for reference — Huber(d=1):
    XGB-LAD   R2=+0.066  MAE=1.463  MedAE=0.718  Huber=1.084
    RF-LAD    R2=+0.089  MAE=1.493  MedAE=0.774  Huber=1.106
    OLS-LAD   R2=-0.229  MAE=1.701  MedAE=0.707  Huber=1.157
    RF-MSE    R2=+0.086  MAE=1.739  MedAE=1.012  Huber=1.322
    XGB-MSE   R2=+0.083  MAE=1.781  MedAE=1.109  Huber=1.356
    OLS       R2=+0.033  MAE=1.761  MedAE=1.314  Huber=1.317
""")


def run_ice_roll_vol():
    """ICE curves for roll_vol_500 from RF-MSE. Saves ice_roll_vol.png."""

    FEATURES = [
        "dollar_value", "log_dollar_value", "participation_rate",
        "roll_spread_500", "roll_vol_500", "exchange_id",
    ]
    VOL_IDX = FEATURES.index("roll_vol_500")

    # load and train
    df_tr = pd.read_parquet("data/lit_buy_features_v2.parquet")
    df_te = pd.read_parquet("data/lit_buy_features_v2_sep.parquet")
    df_tr["abs_impact"] = df_tr["impact_vwap_bps"].abs()
    df_te["abs_impact"] = df_te["impact_vwap_bps"].abs()
    df_tr = df_tr.sort_values("date").reset_index(drop=True)

    X_tr = df_tr[FEATURES].to_numpy(dtype=np.float64)
    y_tr = df_tr["abs_impact"].to_numpy(dtype=np.float64)
    X_te = df_te[FEATURES].to_numpy(dtype=np.float64)

    print("Training RF...", flush=True)
    model = RandomForestRegressor(
        max_depth=30, n_estimators=50, min_samples_leaf=20,
        max_features=0.33, bootstrap=False, random_state=42, n_jobs=-1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_tr, y_tr)

    # compute ICE curves
    print("Computing ICE curves...", flush=True)
    rng = np.random.default_rng(42)
    n_ice = 100
    ice_idx = rng.choice(len(X_te), size=n_ice, replace=False)
    X_ice = X_te[ice_idx]

    vol_vals = X_te[:, VOL_IDX]
    grid = np.linspace(np.percentile(vol_vals, 1), np.percentile(vol_vals, 99), 200)

    ice = np.zeros((n_ice, len(grid)))
    for gi, gval in enumerate(grid):
        X_mod = X_ice.copy()
        X_mod[:, VOL_IDX] = gval
        ice[:, gi] = np.maximum(model.predict(X_mod), 0.0)

    pdp_mean = ice.mean(axis=0)

    # plot
    print("Plotting...", flush=True)
    fig, ax = plt.subplots(figsize=(10, 7))

    for row in range(n_ice):
        ax.plot(grid, ice[row], color="#7c3aed", alpha=0.08, lw=0.8)

    ax.plot(grid, pdp_mean, color="black", lw=2.5, label="PDP (mean)", zorder=5)

    ax.set_xlabel("500 Trade Rolling Volatility (bps)", fontsize=12)
    ax.set_ylabel("|slippage| (bps)", fontsize=12)
    ax.set_title("AAPL Random Forest ICE: Rolling Volatility",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(grid[0], grid[-1])

    plt.tight_layout()
    plt.savefig("ice_roll_vol.png", dpi=150, bbox_inches="tight")
    print("Saved -> ice_roll_vol.png")


def run_model_comparison_v2():
    """
    Five-model comparison with 500-trade rolling features, run on both signed and
    absolute slippage targets. Same five models as v1 (OLS, Almgren, Lasso, RF, XGB),
    5-fold time-series walk-forward CV. Saves _signed.png and _abs.png.
    """

    # load data
    df = pd.read_parquet("data/lit_buy_features_v2.parquet")
    df = df.sort_values("date").reset_index(drop=True)

    FEATURES = [
        "dollar_value",
        "log_dollar_value",
        "participation_rate",
        "roll_spread_500",
        "roll_vol_500",
        "time_of_day",
        "exchange_id",
        "day_of_week",
    ]

    IDX_SPREAD = FEATURES.index("roll_spread_500")
    IDX_VOL    = FEATURES.index("roll_vol_500")
    IDX_PRATE  = FEATURES.index("participation_rate")

    X_all = df[FEATURES].to_numpy(dtype=np.float64)
    dates  = df["date"].to_numpy()

    print(f"Dataset: {len(df):,} rows, {df['date'].nunique()} unique dates")
    print(f"Date range: {dates[0]} to {dates[-1]}")

    # time-series 5-fold walk-forward CV
    n_folds = 5
    unique_dates = np.array(sorted(df["date"].unique()))
    n_days = len(unique_dates)

    date_fold = np.digitize(
        np.arange(n_days),
        bins=np.linspace(0, n_days, n_folds + 1)[1:-1],
    )
    date_to_fold = dict(zip(unique_dates, date_fold))
    row_fold = np.array([date_to_fold[d] for d in dates])

    print(f"\nFold sizes (rows):")
    for f in range(n_folds):
        mask = row_fold == f
        days_in = np.unique(dates[mask])
        print(f"  fold {f}: {mask.sum():>6,} rows  |  {len(days_in):2d} days  "
              f"({days_in[0]} .. {days_in[-1]})")

    splits = []
    for k in range(1, n_folds):
        tr_mask = row_fold < k
        te_mask = row_fold == k
        splits.append((np.where(tr_mask)[0], np.where(te_mask)[0]))

    print(f"\n{len(splits)} walk-forward splits\n")

    # helpers
    def metrics(y_true, y_pred):
        r2   = r2_score(y_true, y_pred)
        mae  = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        return r2, mae, rmse

    def almgren_pred(X, c1, c2, c3):
        return (c1 * X[:, IDX_SPREAD]
                + c2 * X[:, IDX_VOL] * np.sqrt(X[:, IDX_PRATE])
                + c3)

    def fit_almgren(X_tr, y_tr, X_te):
        try:
            p0     = [0.1, 0.5, 0.0]
            bounds = ([-50, -200, -50], [50, 200, 50])
            popt, _ = curve_fit(
                lambda X, c1, c2, c3: almgren_pred(X, c1, c2, c3),
                X_tr, y_tr, p0=p0, bounds=bounds, maxfev=20_000,
            )
            return almgren_pred(X_te, *popt), popt
        except Exception as e:
            print(f"    [Almgren fit failed: {e}]")
            intercept = y_tr.mean()
            return np.full(len(X_te), intercept), (0.0, 0.0, intercept)

    def run_cv(y_all, label):
        """Run full 5-model CV for a given target array y_all."""
        print(f"\n{'='*70}")
        print(f"TARGET: {label}")
        print(f"{'='*70}")

        results = {name: {"r2": [], "mae": [], "rmse": []}
                   for name in ["OLS", "Almgren-Chr", "Lasso", "RandomForest", "XGBoost"]}
        xgb_imp_accum = np.zeros(len(FEATURES))

        for fold_idx, (tr_idx, te_idx) in enumerate(splits):
            X_tr, X_te = X_all[tr_idx], X_all[te_idx]
            y_tr, y_te = y_all[tr_idx], y_all[te_idx]
            print(f"\n--- Fold {fold_idx+1}/{len(splits)}  "
                  f"(train={len(tr_idx):,}  test={len(te_idx):,}) ---")

            # Model 1: OLS
            ols = LinearRegression()
            ols.fit(X_tr, y_tr)
            y_hat = ols.predict(X_te)
            r2, mae, rmse = metrics(y_te, y_hat)
            results["OLS"]["r2"].append(r2); results["OLS"]["mae"].append(mae); results["OLS"]["rmse"].append(rmse)
            print(f"  OLS:          R2={r2:+.4f}  MAE={mae:.4f}  RMSE={rmse:.4f}")

            # Model 2: Almgren-Chr
            y_hat, popt = fit_almgren(X_tr, y_tr, X_te)
            r2, mae, rmse = metrics(y_te, y_hat)
            results["Almgren-Chr"]["r2"].append(r2); results["Almgren-Chr"]["mae"].append(mae); results["Almgren-Chr"]["rmse"].append(rmse)
            print(f"  Almgren-Chr:  R2={r2:+.4f}  MAE={mae:.4f}  RMSE={rmse:.4f}  "
                  f"c=({popt[0]:.3f},{popt[1]:.3f},{popt[2]:.3f})")

            # Model 3: Lasso
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)
            lasso_cv = LassoCV(
                alphas=np.logspace(-4, 2, 60), cv=5,
                max_iter=10_000, random_state=42, n_jobs=1,
            )
            lasso_cv.fit(X_tr_s, y_tr)
            y_hat = lasso_cv.predict(X_te_s)
            r2, mae, rmse = metrics(y_te, y_hat)
            results["Lasso"]["r2"].append(r2); results["Lasso"]["mae"].append(mae); results["Lasso"]["rmse"].append(rmse)
            print(f"  Lasso:        R2={r2:+.4f}  MAE={mae:.4f}  RMSE={rmse:.4f}  "
                  f"alpha={lasso_cv.alpha_:.5f}")

            # Model 4: Random Forest
            rf = RandomForestRegressor(
                n_estimators=300, max_depth=6, min_samples_leaf=50,
                max_features=0.5, n_jobs=1, random_state=42,
            )
            rf.fit(X_tr, y_tr)
            y_hat = rf.predict(X_te)
            r2, mae, rmse = metrics(y_te, y_hat)
            results["RandomForest"]["r2"].append(r2); results["RandomForest"]["mae"].append(mae); results["RandomForest"]["rmse"].append(rmse)
            print(f"  RandomForest: R2={r2:+.4f}  MAE={mae:.4f}  RMSE={rmse:.4f}")

            # Model 5: XGBoost
            xgb_model = xgb.XGBRegressor(
                n_estimators=500, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=50,
                reg_alpha=0.1, reg_lambda=1.0, tree_method="hist",
                random_state=42, n_jobs=1, verbosity=0,
            )
            xgb_model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
            y_hat = xgb_model.predict(X_te)
            r2, mae, rmse = metrics(y_te, y_hat)
            results["XGBoost"]["r2"].append(r2); results["XGBoost"]["mae"].append(mae); results["XGBoost"]["rmse"].append(rmse)
            print(f"  XGBoost:      R2={r2:+.4f}  MAE={mae:.4f}  RMSE={rmse:.4f}")
            xgb_imp_accum += xgb_model.feature_importances_

        # Summary table
        print(f"\n{'='*68}")
        print(f"{'Model':<16}  {'Mean R2':>9}  {'Std R2':>8}  {'Mean MAE':>9}  {'Mean RMSE':>10}")
        print(f"{'='*68}")
        summary_rows = []
        for name, vals in results.items():
            r2_arr  = np.array(vals["r2"])
            mae_arr = np.array(vals["mae"])
            rmse_arr = np.array(vals["rmse"])
            r2_mean, r2_std = r2_arr.mean(), r2_arr.std()
            mae_mean = mae_arr.mean()
            rmse_mean = rmse_arr.mean()
            summary_rows.append({
                "model": name, "r2_mean": r2_mean, "r2_std": r2_std,
                "mae_mean": mae_mean, "rmse_mean": rmse_mean,
            })
            print(f"  {name:<14}  {r2_mean:>+9.4f}  {r2_std:>8.4f}  "
                  f"{mae_mean:>9.4f}  {rmse_mean:>10.4f}")
        print(f"{'='*68}")

        # XGB importances
        xgb_imp = xgb_imp_accum / len(splits)
        imp_df = (
            pd.DataFrame({"feature": FEATURES, "importance": xgb_imp})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        print("\n--- XGBoost feature importances (mean gain, averaged over folds) ---")
        for _, row in imp_df.iterrows():
            bar = "#" * int(row["importance"] * 200)
            print(f"  {row['feature']:<25}  {row['importance']:.4f}  {bar}")

        return pd.DataFrame(summary_rows).set_index("model"), results

    # signed target
    y_signed = df["impact_vwap_bps"].to_numpy(dtype=np.float64)
    summary_signed, results_signed = run_cv(y_signed, "impact_vwap_bps (signed)")

    # absolute target
    y_abs = np.abs(y_signed)
    summary_abs, results_abs = run_cv(y_abs, "|impact_vwap_bps| (abs)")

    # plot bar charts
    model_names = list(results_signed.keys())
    colors = ["#64748b", "#0ea5e9", "#f59e0b", "#10b981", "#ef4444"]

    def make_bar_chart(results, title_suffix, fname):
        r2_means = [np.array(results[m]["r2"]).mean() for m in model_names]
        r2_stds  = [np.array(results[m]["r2"]).std()  for m in model_names]

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(
            model_names, r2_means,
            yerr=r2_stds, capsize=5,
            color=colors, edgecolor="white", linewidth=0.8,
            error_kw=dict(elinewidth=1.5, capthick=1.5, ecolor="black"),
        )
        for bar, val, err in zip(bars, r2_means, r2_stds):
            y_pos = val + err + 0.001 if val >= 0 else val - err - 0.001
            va    = "bottom" if val >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                    f"{val:+.4f}", ha="center", va=va, fontsize=9.5, fontweight="bold")

        ax.axhline(0, color="black", lw=1, ls="--", alpha=0.6)
        ax.set_ylabel("Mean out-of-sample R2 (4-fold walk-forward)", fontsize=11)
        ax.set_title(
            f"AAPL lit buy block trades — model comparison v2\n"
            f"(trade-level features, 500-trade rolling window, target: {title_suffix})",
            fontsize=11, fontweight="bold",
        )
        ax.set_xticklabels(model_names, fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"saved -> {fname}")

    make_bar_chart(results_signed, "signed impact_vwap_bps",
                   "aapl_model_comparison_v2_signed.png")
    make_bar_chart(results_abs,    "|impact_vwap_bps|",
                   "aapl_model_comparison_v2_abs.png")

    # side-by-side summary
    print("\n\n" + "="*80)
    print("COMPARISON: signed vs abs target  (mean OOS R2)")
    print("="*80)
    print(f"{'Model':<16}  {'Signed R2':>11}  {'Abs R2':>11}  {'Delta':>9}")
    print("-"*55)
    for name in model_names:
        r2_s = np.array(results_signed[name]["r2"]).mean()
        r2_a = np.array(results_abs[name]["r2"]).mean()
        print(f"  {name:<14}  {r2_s:>+11.4f}  {r2_a:>+11.4f}  {r2_a - r2_s:>+9.4f}")
    print("="*80)


def run_prediction_intervals():
    """
    Prediction interval plots for the two-stage GAMLSS: fan charts sorted by predicted
    median, size-binned intervals, and a direct linear vs XGB comparison for AAPL/COIN.
    """

    FEATURES = [
        "dollar_value", "log_dollar_value", "participation_rate",
        "roll_spread_500", "roll_vol_500", "exchange_id",
    ]

    def fit_linear_gamlss(X_tr, y_tr, X_te):
        X_tr_c = sm.add_constant(X_tr)
        X_te_c = sm.add_constant(X_te)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            qr = sm.QuantReg(y_tr, X_tr_c).fit(q=0.5, max_iter=5000)
        mu_tr = qr.predict(X_tr_c)
        mu_te = qr.predict(X_te_c)
        abs_r = np.abs(y_tr - mu_tr)
        Xg_tr = np.column_stack([np.ones(len(X_tr)), X_tr])
        Xg_te = np.column_stack([np.ones(len(X_te)), X_te])
        gamma, _, _, _ = np.linalg.lstsq(Xg_tr, abs_r, rcond=None)
        b_te = np.clip(Xg_te @ gamma, 0.1, None)
        return mu_te, b_te

    def fit_xgb_gamlss(X_tr, y_tr, X_te):
        loc = XGBRegressor(objective="reg:absoluteerror", tree_method="hist", verbosity=0,
                           random_state=42, n_jobs=1, max_depth=3, n_estimators=200,
                           learning_rate=0.07, min_child_weight=5, reg_alpha=10, reg_lambda=10)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loc.fit(X_tr, y_tr)
        mu_tr = np.maximum(loc.predict(X_tr), 0.0)
        mu_te = np.maximum(loc.predict(X_te), 0.0)
        abs_r = np.abs(y_tr - mu_tr)
        sc = XGBRegressor(objective="reg:squarederror", tree_method="hist", verbosity=0,
                          random_state=42, n_jobs=1, max_depth=5, n_estimators=50,
                          min_child_weight=20, learning_rate=0.1, reg_alpha=1, reg_lambda=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sc.fit(X_tr, abs_r)
        b_te = np.clip(sc.predict(X_te), 0.1, None)
        return mu_te, b_te

    # load data
    print("Loading data...", flush=True)

    datasets = {
        "AAPL": ("data/lit_buy_features_v2.parquet", "data/lit_buy_features_v2_sep.parquet"),
        "COIN": ("data/coin_lit_buy_features_train.parquet", "data/coin_lit_buy_features_test.parquet"),
    }

    data = {}
    for ticker, (tr_f, te_f) in datasets.items():
        df_tr = pd.read_parquet(tr_f)
        df_te = pd.read_parquet(te_f)
        df_tr["abs_impact"] = df_tr["impact_vwap_bps"].abs()
        df_te["abs_impact"] = df_te["impact_vwap_bps"].abs()
        df_tr = df_tr.sort_values("date").reset_index(drop=True)
        data[ticker] = {
            "X_tr": df_tr[FEATURES].to_numpy(dtype=np.float64),
            "y_tr": df_tr["abs_impact"].to_numpy(dtype=np.float64),
            "X_te": df_te[FEATURES].to_numpy(dtype=np.float64),
            "y_te": df_te["abs_impact"].to_numpy(dtype=np.float64),
            "dollar_te": df_te["dollar_value"].to_numpy(dtype=np.float64),
        }

    # fit all models
    print("Fitting models...", flush=True)
    results = {}
    for ticker in ["AAPL", "COIN"]:
        d = data[ticker]
        print(f"  {ticker}...", flush=True)
        mu_lin, b_lin = fit_linear_gamlss(d["X_tr"], d["y_tr"], d["X_te"])
        mu_xgb, b_xgb = fit_xgb_gamlss(d["X_tr"], d["y_tr"], d["X_te"])
        results[ticker] = {
            "y_te": d["y_te"], "dollar_te": d["dollar_te"],
            "mu_lin": mu_lin, "b_lin": b_lin,
            "mu_xgb": mu_xgb, "b_xgb": b_xgb,
        }

    rng = np.random.default_rng(42)
    AAPL_COLOR = "#2563eb"
    COIN_COLOR = "#dc2626"
    BAND_COLORS = {"90%": "#bfdbfe", "80%": "#93c5fd", "50%": "#60a5fa"}
    BAND_COLORS_COIN = {"90%": "#fecaca", "80%": "#fca5a5", "50%": "#f87171"}

    # =============================================================================
    # Figure 1: Fan charts — 50/80/90% intervals sorted by predicted median
    # =============================================================================
    print("Plotting fan charts...", flush=True)

    fig1, axes1 = plt.subplots(2, 2, figsize=(20, 14))

    for col, model_key, model_name in [(0, "lin", "Two-Stage Linear Regression"), (1, "xgb", "Two-Stage XGBoost")]:
        for row, ticker in [(0, "AAPL"), (1, "COIN")]:
            ax = axes1[row, col]
            r = results[ticker]
            mu = r[f"mu_{model_key}"]
            b = r[f"b_{model_key}"]
            y = r["y_te"]

            n_show = min(300, len(y))
            idx = rng.choice(len(y), size=n_show, replace=False)
            order = np.argsort(mu[idx])
            idx_sorted = idx[order]

            x = np.arange(n_show)
            mu_s = mu[idx_sorted]
            b_s = b[idx_sorted]
            y_s = y[idx_sorted]

            colors = BAND_COLORS if ticker == "AAPL" else BAND_COLORS_COIN
            base_color = AAPL_COLOR if ticker == "AAPL" else COIN_COLOR

            for level, label in [(0.90, "90%"), (0.80, "80%"), (0.50, "50%")]:
                z = np.log(1.0 / (1.0 - level))
                lo = np.maximum(mu_s - z * b_s, 0.0)
                hi = mu_s + z * b_s
                ax.fill_between(x, lo, hi, alpha=0.7, color=colors[label], label=f"{label} interval")

            ax.plot(x, mu_s, color=base_color, lw=1.5, label="Predicted median", zorder=3)
            ax.scatter(x, y_s, s=10, color="black", alpha=0.45, zorder=4, label="Actual")

            ax.set_xlabel("Trade index (sorted by predicted median)", fontsize=10)
            ax.set_ylabel("|slippage| (bps)", fontsize=10)
            ax.set_title(f"{ticker} {model_name} (300 random test trades)",
                         fontsize=11, fontweight="bold")
            ax.legend(fontsize=8, loc="upper left", ncol=2)
            ax.grid(True, alpha=0.12)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    fig1.suptitle("Prediction Interval Fan Charts: 50% / 80% / 90% Laplace Intervals",
                  fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("pred_intervals_fan.png", dpi=150, bbox_inches="tight")
    print("Saved -> pred_intervals_fan.png")

    # =============================================================================
    # Figure 2: Intervals binned by dollar value (trade size)
    # =============================================================================
    print("Plotting intervals by trade size...", flush=True)

    fig2, axes2 = plt.subplots(2, 2, figsize=(20, 14))

    for col, model_key, model_name in [(0, "lin", "Two-Stage Linear Regression"), (1, "xgb", "Two-Stage XGBoost")]:
        for row, ticker in [(0, "AAPL"), (1, "COIN")]:
            ax = axes2[row, col]
            r = results[ticker]
            mu = r[f"mu_{model_key}"]
            b = r[f"b_{model_key}"]
            y = r["y_te"]
            dv = r["dollar_te"]

            base_color = AAPL_COLOR if ticker == "AAPL" else COIN_COLOR

            n_bins = 15
            bin_edges = np.percentile(dv, np.linspace(0, 100, n_bins + 1))
            bin_edges = np.unique(bin_edges)
            actual_bins = len(bin_edges) - 1

            bin_centers = []
            bin_mu = []
            bin_y_mean = []
            bin_y_med = []
            bin_lo90 = []
            bin_hi90 = []
            bin_lo80 = []
            bin_hi80 = []
            bin_lo50 = []
            bin_hi50 = []

            for i in range(actual_bins):
                mask = (dv >= bin_edges[i]) & (dv < bin_edges[i + 1])
                if i == actual_bins - 1:
                    mask = (dv >= bin_edges[i]) & (dv <= bin_edges[i + 1])
                if mask.sum() < 5:
                    continue
                bin_centers.append(dv[mask].mean())
                bin_mu.append(mu[mask].mean())
                bin_y_mean.append(y[mask].mean())
                bin_y_med.append(np.median(y[mask]))
                for level, lo_list, hi_list in [
                    (0.90, bin_lo90, bin_hi90),
                    (0.80, bin_lo80, bin_hi80),
                    (0.50, bin_lo50, bin_hi50),
                ]:
                    z = np.log(1.0 / (1.0 - level))
                    lo_list.append(np.maximum(mu[mask] - z * b[mask], 0).mean())
                    hi_list.append((mu[mask] + z * b[mask]).mean())

            bc = np.array(bin_centers)
            colors_set = BAND_COLORS if ticker == "AAPL" else BAND_COLORS_COIN

            ax.fill_between(bc, bin_lo90, bin_hi90, alpha=0.6, color=colors_set["90%"], label="90% interval")
            ax.fill_between(bc, bin_lo80, bin_hi80, alpha=0.7, color=colors_set["80%"], label="80% interval")
            ax.fill_between(bc, bin_lo50, bin_hi50, alpha=0.8, color=colors_set["50%"], label="50% interval")
            ax.plot(bc, bin_mu, color=base_color, lw=2.2, marker="o", markersize=5,
                    label="Predicted median", zorder=3)
            ax.scatter(bc, bin_y_med, s=40, color="black", marker="x", linewidths=1.5,
                       zorder=5, label="Actual median")

            ax.set_xlabel("Dollar value (trade size)", fontsize=10)
            ax.set_ylabel("|slippage| (bps)", fontsize=10)
            ax.set_title(f"{ticker} {model_name}\nPrediction Intervals by Trade Size (15 bins)",
                         fontsize=11, fontweight="bold")
            ax.legend(fontsize=8, loc="upper right")
            ax.grid(True, alpha=0.12)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

    fig2.suptitle("Prediction Intervals by Trade Size: How Uncertainty Scales with Dollar Value",
                  fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("pred_intervals_by_size.png", dpi=150, bbox_inches="tight")
    print("Saved -> pred_intervals_by_size.png")

    # =============================================================================
    # Figure 3: Linear vs XGB side by side (direct comparison)
    # =============================================================================
    print("Plotting Linear vs XGB comparison...", flush=True)

    fig3, axes3 = plt.subplots(1, 2, figsize=(20, 8))

    for ax, ticker in zip(axes3, ["AAPL", "COIN"]):
        r = results[ticker]
        y = r["y_te"]
        base_color = AAPL_COLOR if ticker == "AAPL" else COIN_COLOR

        n_show = 250
        idx = rng.choice(len(y), size=min(n_show, len(y)), replace=False)
        # Sort by XGB predicted median for consistent ordering
        order = np.argsort(r["mu_xgb"][idx])
        idx_sorted = idx[order]
        x = np.arange(len(idx_sorted))

        z90 = np.log(10.0)

        # Linear intervals (background, gray)
        mu_lin_s = r["mu_lin"][idx_sorted]
        b_lin_s = r["b_lin"][idx_sorted]
        lo_lin = np.maximum(mu_lin_s - z90 * b_lin_s, 0.0)
        hi_lin = mu_lin_s + z90 * b_lin_s

        # XGB intervals (foreground, colored)
        mu_xgb_s = r["mu_xgb"][idx_sorted]
        b_xgb_s = r["b_xgb"][idx_sorted]
        lo_xgb = np.maximum(mu_xgb_s - z90 * b_xgb_s, 0.0)
        hi_xgb = mu_xgb_s + z90 * b_xgb_s

        y_s = y[idx_sorted]

        ax.fill_between(x, lo_lin, hi_lin, alpha=0.2, color="#9ca3af",
                        label=f"Linear 90% (mean w={hi_lin.mean()-lo_lin.mean():.1f})")
        ax.fill_between(x, lo_xgb, hi_xgb, alpha=0.3, color=base_color,
                        label=f"XGB 90% (mean w={hi_xgb.mean()-lo_xgb.mean():.1f})")
        ax.plot(x, mu_lin_s, color="#6b7280", lw=1.0, ls="--", alpha=0.7, label="Linear median")
        ax.plot(x, mu_xgb_s, color=base_color, lw=1.5, label="XGB median", zorder=3)
        ax.scatter(x, y_s, s=10, color="black", alpha=0.4, zorder=4, label="Actual")

        ax.set_xlabel("Trade index (sorted by XGB predicted median)", fontsize=10)
        ax.set_ylabel("|slippage| (bps)", fontsize=10)
        ax.set_title(f"{ticker} Linear vs Two-Stage XGBoost (90% intervals, 250 random test trades)",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=8.5, loc="upper left")
        ax.grid(True, alpha=0.12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig3.suptitle("Linear vs Two-Stage XGBoost: 90% Prediction Interval Comparison",
                  fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("pred_intervals_linear_vs_xgb.png", dpi=150, bbox_inches="tight")
    print("Saved -> pred_intervals_linear_vs_xgb.png")

    print("\nDone!")


def run_shap_analysis():
    """
    SHAP beeswarm and partial dependence plots using the best XGBoost fold (fold 1,
    train 2024-06-03..2024-06-20). Reproduces the exact fold so SHAP values match
    what's discussed in the article. Saves aapl_shap.png.
    """

    # reproduce fold 1 exactly
    df = pd.read_parquet("data/lit_buy_features_v2.parquet")
    df["abs_impact"] = df["impact_vwap_bps"].abs()
    df = df.sort_values("date").reset_index(drop=True)

    COLS = ["date", "roll_spread_500", "roll_vol_500", "participation_rate", "abs_impact"]
    df   = df[COLS].dropna()

    N_BINS  = 50
    N_FOLDS = 5
    FEAT_NAMES = ["mean_spread", "mean_vol", "mean_prate"]
    FEAT_LABELS = [
        "mean_spread\n(Roll spread, bps)",
        "mean_vol\n(realized vol, bps)",
        "mean_prate\n(participation rate)",
    ]

    unique_dates = np.array(sorted(df["date"].unique()))
    n_days = len(unique_dates)
    date_fold = np.digitize(
        np.arange(n_days),
        bins=np.linspace(0, n_days, N_FOLDS + 1)[1:-1],
    )
    dates    = df["date"].to_numpy()
    date_to_fold = dict(zip(unique_dates, date_fold))
    row_fold = np.array([date_to_fold[d] for d in dates])

    def make_bins(data_idx):
        sub = df.iloc[data_idx][
            ["roll_spread_500", "roll_vol_500", "participation_rate", "abs_impact"]
        ].copy()
        sub["bin"] = pd.qcut(sub["roll_spread_500"], q=N_BINS, labels=False,
                              duplicates="drop")
        return (
            sub.groupby("bin", observed=True)
            .agg(
                mean_spread=("roll_spread_500",   "mean"),
                mean_vol   =("roll_vol_500",      "mean"),
                mean_prate =("participation_rate", "mean"),
                mean_abs   =("abs_impact",         "mean"),
                count      =("abs_impact",         "count"),
            )
            .reset_index(drop=True)
        )

    # Fold 1: train = block 0, test = block 1
    tr_idx = np.where(row_fold < 1)[0]
    te_idx = np.where(row_fold == 1)[0]

    tr_bins = make_bins(tr_idx)
    te_bins = make_bins(te_idx)

    X_tr = tr_bins[FEAT_NAMES].to_numpy(dtype=np.float64)
    y_tr = tr_bins["mean_abs"].to_numpy(dtype=np.float64)
    X_te = te_bins[FEAT_NAMES].to_numpy(dtype=np.float64)
    y_te = te_bins["mean_abs"].to_numpy(dtype=np.float64)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # Train with best-fold hyperparameters
    BEST_PARAMS = dict(
        learning_rate=0.05, max_depth=2, min_child_weight=5, n_estimators=50,
        tree_method="hist", random_state=42, n_jobs=1, verbosity=0,
    )
    model = xgb.XGBRegressor(**BEST_PARAMS)
    model.fit(X_tr_s, y_tr)

    y_hat = model.predict(X_te_s)
    ss_res = ((y_te - y_hat) ** 2).sum()
    ss_tot = ((y_te - y_te.mean()) ** 2).sum()
    oos_r2 = 1.0 - ss_res / ss_tot
    print(f"Reproduced fold 1  OOS R²={oos_r2:.4f}  "
          f"(train bins={len(tr_bins)}, test bins={len(te_bins)})")

    # SHAP values via TreeExplainer on scaled test bins
    explainer  = shap.TreeExplainer(model, X_tr_s)
    shap_vals  = explainer.shap_values(X_te_s)   # (50, 3)
    base_value = explainer.expected_value
    print(f"SHAP base value (E[f(x)]): {base_value:.4f}")
    print(f"SHAP array shape: {shap_vals.shape}")
    print(f"Mean |SHAP| per feature:")
    for name, mv in zip(FEAT_NAMES, np.abs(shap_vals).mean(axis=0)):
        print(f"  {name:<15} {mv:.5f}")

    # figure layout
    fig = plt.figure(figsize=(15, 10))
    gs  = gridspec.GridSpec(
        2, 3,
        figure=fig,
        height_ratios=[1.05, 1],
        hspace=0.50,
        wspace=0.38,
    )

    ax_bee  = fig.add_subplot(gs[0, :2])   # beeswarm spans cols 0-1
    ax_bar  = fig.add_subplot(gs[0, 2])    # mean |SHAP| bar
    ax_pdp  = [fig.add_subplot(gs[1, k]) for k in range(3)]

    # Panel A: SHAP beeswarm (manual)
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)        # (3,)
    order = np.argsort(mean_abs_shap)                      # ascending → top = most important

    rng = np.random.default_rng(0)

    # Colormap: blue (low feature value) → red (high)
    cmap = cm.coolwarm

    for plot_row, feat_idx in enumerate(order):
        sv  = shap_vals[:, feat_idx]                       # SHAP values for this feature
        fv  = X_te[:, feat_idx]                            # unscaled feature values (for color)
        fv_norm = (fv - fv.min()) / ((fv.max() - fv.min()) + 1e-12)  # 0..1 for colormap

        # Beeswarm: deterministic y-jitter proportional to density
        y_jitter = rng.uniform(-0.28, 0.28, size=len(sv))

        colors = cmap(fv_norm)
        ax_bee.scatter(
            sv, plot_row + y_jitter,
            c=colors, s=45, alpha=0.88,
            edgecolors="none", zorder=3,
        )

    ax_bee.set_yticks(range(len(order)))
    ax_bee.set_yticklabels(
        [FEAT_LABELS[i] for i in order], fontsize=9.5
    )
    ax_bee.axvline(0, color="black", lw=1.0, ls="--", alpha=0.5)
    ax_bee.set_xlabel("SHAP value  (impact on |impact_vwap_bps| prediction)", fontsize=9.5)
    ax_bee.set_title(
        f"SHAP beeswarm — best XGBoost fold (OOS R²={oos_r2:.4f})\n"
        f"50 test bins  |  color = feature value (blue=low, red=high)",
        fontsize=10, fontweight="bold",
    )
    ax_bee.grid(axis="x", alpha=0.2)
    ax_bee.spines["top"].set_visible(False)
    ax_bee.spines["right"].set_visible(False)

    # Colorbar for beeswarm
    sm = cm.ScalarMappable(cmap=cmap, norm=Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax_bee, fraction=0.025, pad=0.02)
    cbar.set_label("Feature value\n(low → high)", fontsize=8)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["low", "mid", "high"], fontsize=7.5)

    # Panel B: mean |SHAP| bar chart
    bar_order = np.argsort(mean_abs_shap)[::-1]           # descending for bar
    bar_colors = ["#2563eb", "#16a34a", "#dc2626"]
    bar_labels_short = ["mean_spread", "mean_vol", "mean_prate"]

    ax_bar.barh(
        range(3),
        mean_abs_shap[bar_order],
        color=[bar_colors[i] for i in bar_order],
        edgecolor="white", linewidth=0.6, height=0.55,
    )
    ax_bar.set_yticks(range(3))
    ax_bar.set_yticklabels(
        [bar_labels_short[i] for i in bar_order], fontsize=9.5
    )
    for j, (val, feat_idx) in enumerate(zip(mean_abs_shap[bar_order], bar_order)):
        ax_bar.text(val + 0.0002, j, f"{val:.4f}", va="center", fontsize=8.5)
    ax_bar.set_xlabel("Mean |SHAP value|", fontsize=9.5)
    ax_bar.set_title("SHAP feature importance\n(mean |SHAP|)", fontsize=10, fontweight="bold")
    ax_bar.grid(axis="x", alpha=0.2)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    # Panel C: partial dependence plots
    # For each feature: sweep across its test-set range, hold other two at test mean.
    # Both sweep and held values pass through the scaler before prediction.

    te_mean_raw = X_te.mean(axis=0)   # shape (3,) — mean of each unscaled feature
    GRID_N = 200
    PDP_COLORS = ["#2563eb", "#16a34a", "#dc2626"]

    for k, (ax, feat_idx) in enumerate(zip(ax_pdp, range(3))):
        feat_min = X_te[:, feat_idx].min()
        feat_max = X_te[:, feat_idx].max()
        grid_raw = np.linspace(feat_min, feat_max, GRID_N)

        # Build raw feature matrix: sweep feature k, hold others at test mean
        X_sweep_raw = np.tile(te_mean_raw, (GRID_N, 1))   # (GRID_N, 3)
        X_sweep_raw[:, feat_idx] = grid_raw

        X_sweep_s = scaler.transform(X_sweep_raw)
        y_pdp     = model.predict(X_sweep_s)

        # PDP line
        ax.plot(grid_raw, y_pdp, color=PDP_COLORS[k], lw=2.2, zorder=4,
                label="PDP (other features at test mean)")

        # Actual test bin scatter
        ax.scatter(
            X_te[:, feat_idx], y_te,
            color="#1e293b", s=35, alpha=0.65, zorder=5,
            edgecolors="white", linewidths=0.5,
            label="Test bin means",
        )

        # SHAP-based local slopes: annotate mean SHAP
        mean_shap_k = np.abs(shap_vals[:, feat_idx]).mean()
        ax.set_xlabel(FEAT_LABELS[k], fontsize=9.5)
        ax.set_ylabel("|impact_vwap_bps| (bps)", fontsize=9)
        ax.set_title(
            f"PDP: {FEAT_NAMES[k]}\nmean |SHAP| = {mean_shap_k:.4f}",
            fontsize=9.5, fontweight="bold",
        )
        ax.legend(fontsize=7.5, loc="upper left")
        ax.grid(True, alpha=0.18)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # supertitle
    fig.suptitle(
        "AAPL lit buy block trades — SHAP analysis on best XGBoost fold\n"
        "50-bin aggregated data  |  features: mean_spread, mean_vol, mean_prate  "
        "|  target: |impact_vwap_bps|",
        fontsize=11, fontweight="bold", y=1.01,
    )

    plt.savefig("aapl_shap.png", dpi=150, bbox_inches="tight")
    print("\nsaved -> aapl_shap.png")


if __name__ == "__main__":
    run_compute_tables()
    run_compute_huber_delta2()
    run_ice_roll_vol()
    run_model_comparison_v2()
    run_prediction_intervals()
    run_shap_analysis()
