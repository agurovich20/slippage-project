"""
LAD and XGB-MAE on signed impact_vwap_bps for AAPL and COIN.

Models:
  1. OLS       — baseline (already computed)
  2. LAD-Uni   — quantile regression at median, univariate (spread only)
  3. LAD-Multi — quantile regression at median, 3 features
  4. XGB-MAE   — XGBRegressor with reg:absoluteerror, 3 features

Train/test: same as before (AAPL Jun-Aug/Sep, COIN Jan-Aug/Sep).
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import warnings
import numpy as np
import pandas as pd
from statsmodels.regression.quantile_regression import QuantReg
from xgboost import XGBRegressor

# ══════════════════════════════════════════════════════════════════════════════
# Load data
# ══════════════════════════════════════════════════════════════════════════════
aapl_tr = pd.read_parquet("data/lit_buy_features_v2.parquet")
aapl_te = pd.read_parquet("data/lit_buy_features_v2_sep.parquet")
coin_tr = pd.read_parquet("data/coin_lit_buy_features_train.parquet")
coin_te = pd.read_parquet("data/coin_lit_buy_features_test.parquet")

TARGET = "impact_vwap_bps"
FEAT_UNI = ["roll_spread_500"]
FEAT_MULTI = ["roll_spread_500", "roll_vol_500", "participation_rate"]


def r2_fn(y, yhat):
    ss_res = ((y - yhat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def mae_fn(y, yhat):
    return np.mean(np.abs(y - yhat))


def median_ae_fn(y, yhat):
    return np.median(np.abs(y - yhat))


# ══════════════════════════════════════════════════════════════════════════════
# Run models for each ticker
# ══════════════════════════════════════════════════════════════════════════════
for ticker, df_tr, df_te in [("AAPL", aapl_tr, aapl_te), ("COIN", coin_tr, coin_te)]:
    y_tr = df_tr[TARGET].values.astype(np.float64)
    y_te = df_te[TARGET].values.astype(np.float64)

    print(f"\n{'='*72}")
    print(f"  {ticker}   train={len(df_tr):,}  test={len(df_te):,}")
    print(f"{'='*72}")

    results = []

    # ── OLS-Uni ───────────────────────────────────────────────────────────
    X_tr_u = np.column_stack([df_tr[FEAT_UNI].values, np.ones(len(df_tr))])
    X_te_u = np.column_stack([df_te[FEAT_UNI].values, np.ones(len(df_te))])
    beta_ols, *_ = np.linalg.lstsq(X_tr_u, y_tr, rcond=None)
    pred_ols = X_te_u @ beta_ols
    results.append(("OLS-Uni", pred_ols, beta_ols))

    # ── OLS-Multi ─────────────────────────────────────────────────────────
    X_tr_m = np.column_stack([df_tr[FEAT_MULTI].values, np.ones(len(df_tr))])
    X_te_m = np.column_stack([df_te[FEAT_MULTI].values, np.ones(len(df_te))])
    beta_ols_m, *_ = np.linalg.lstsq(X_tr_m, y_tr, rcond=None)
    pred_ols_m = X_te_m @ beta_ols_m
    results.append(("OLS-Multi", pred_ols_m, beta_ols_m))

    # ── LAD-Uni ───────────────────────────────────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lad_u = QuantReg(y_tr, X_tr_u).fit(q=0.5, max_iter=5000, p_tol=1e-6)
    pred_lad_u = X_te_u @ lad_u.params
    results.append(("LAD-Uni", pred_lad_u, lad_u.params))

    # ── LAD-Multi ─────────────────────────────────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lad_m = QuantReg(y_tr, X_tr_m).fit(q=0.5, max_iter=5000, p_tol=1e-6)
    pred_lad_m = X_te_m @ lad_m.params
    results.append(("LAD-Multi", pred_lad_m, lad_m.params))

    # ── XGB-MAE (3 features) ─────────────────────────────────────────────
    X3_tr = df_tr[FEAT_MULTI].values.astype(np.float64)
    X3_te = df_te[FEAT_MULTI].values.astype(np.float64)

    xgb_mae = XGBRegressor(
        objective="reg:absoluteerror",
        max_depth=3, n_estimators=50, learning_rate=0.1,
        min_child_weight=1, reg_alpha=10, reg_lambda=10,
        tree_method="hist", verbosity=0, random_state=42, n_jobs=1,
    )
    xgb_mae.fit(X3_tr, y_tr)
    pred_xgb = xgb_mae.predict(X3_te)
    results.append(("XGB-MAE", pred_xgb, None))

    # ── XGB-MAE tuned (more trees, lower LR) ─────────────────────────────
    xgb_mae2 = XGBRegressor(
        objective="reg:absoluteerror",
        max_depth=4, n_estimators=200, learning_rate=0.05,
        min_child_weight=30, reg_alpha=5, reg_lambda=5,
        subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", verbosity=0, random_state=42, n_jobs=1,
    )
    xgb_mae2.fit(X3_tr, y_tr)
    pred_xgb2 = xgb_mae2.predict(X3_te)
    results.append(("XGB-MAE-v2", pred_xgb2, None))

    # ── Print results ─────────────────────────────────────────────────────
    print(f"\n  {'Model':<14} {'OOS R2':>9} {'OOS MAE':>9} {'OOS MedAE':>10} {'Coefficients'}")
    print(f"  {'-'*70}")
    for name, pred, coef in results:
        r2 = r2_fn(y_te, pred)
        mae = mae_fn(y_te, pred)
        medae = median_ae_fn(y_te, pred)
        coef_str = ""
        if coef is not None:
            coef_str = "  ".join(f"{c:+.5f}" for c in coef)
        print(f"  {name:<14} {r2:>+9.4f} {mae:>9.4f} {medae:>10.4f}   {coef_str}")

    # ── Feature importances for XGB ───────────────────────────────────────
    imp = xgb_mae.feature_importances_
    print(f"\n  XGB-MAE feature importances:")
    for feat, val in sorted(zip(FEAT_MULTI, imp), key=lambda x: -x[1]):
        print(f"    {feat:<25} {val:.4f}")

    imp2 = xgb_mae2.feature_importances_
    print(f"  XGB-MAE-v2 feature importances:")
    for feat, val in sorted(zip(FEAT_MULTI, imp2), key=lambda x: -x[1]):
        print(f"    {feat:<25} {val:.4f}")

print(f"\n{'='*72}")
print("Done.")
