import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from xgboost import XGBRegressor

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

# Load data and compute metrics
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
    
    # Two-Stage Linear Regression
    mu_lin, b_lin = fit_linear_gamlss(X_tr, y_tr, X_te)
    mae_lin = np.mean(np.abs(y_te - mu_lin))
    cov90_lin, w90_lin = compute_coverage(y_te, mu_lin, b_lin, 0.90)
    
    # Two-Stage XGBoost
    mu_xgb, b_xgb = fit_xgb_gamlss(X_tr, y_tr, X_te)
    mae_xgb = np.mean(np.abs(y_te - mu_xgb))
    cov90_xgb, w90_xgb = compute_coverage(y_te, mu_xgb, b_xgb, 0.90)
    
    results.append({
        "ticker": ticker,
        "lin_mae": mae_lin, "lin_w90": w90_lin, "lin_cov90": cov90_lin,
        "xgb_mae": mae_xgb, "xgb_w90": w90_xgb, "xgb_cov90": cov90_xgb,
    })

# Print tables
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
