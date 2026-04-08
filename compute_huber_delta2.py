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

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"]      = "1"

import warnings
import numpy as np
import pandas as pd
from statsmodels.regression.quantile_regression import QuantReg
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# ── Data ───────────────────────────────────────────────────────────────────────
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

# ── Metrics ────────────────────────────────────────────────────────────────────
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

# ── Load best hyperparameters from CSVs ────────────────────────────────────────
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

# ── Train all 6 models ─────────────────────────────────────────────────────────
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

# ── Compute all metrics ────────────────────────────────────────────────────────
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
