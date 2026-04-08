"""
Five-model comparison for predicting impact_vwap_bps on AAPL lit buy block trades.

Models:
  1. OLS          — all 8 features, closed-form
  2. Almgren-Chr  — impact = c1*spread + c2*vol*prate^0.5 + c3  (scipy curve_fit)
  3. Lasso        — L1 regularized OLS (alpha via 1-SE rule on inner CV)
  4. Random Forest
  5. XGBoost

CV: time-series 5-fold — data sorted by date; each fold trains on earlier dates,
    tests on the next chronological block (no shuffling, no future leakage).

Output: out-of-sample R2 / MAE / RMSE per model + XGB importances
        aapl_model_comparison.png  — bar chart of R2 across models
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"]      = "1"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb

# ── 1. Load data ───────────────────────────────────────────────────────────────
df = pd.read_parquet("data/lit_buy_features.parquet")
df = df.sort_values("date").reset_index(drop=True)

FEATURES = [
    "dollar_value",
    "log_dollar_value",
    "participation_rate",
    "daily_roll_spread",
    "trail_1min_volatility",
    "time_of_day",
    "exchange_id",
    "day_of_week",
]
TARGET = "impact_vwap_bps"

X_all = df[FEATURES].to_numpy(dtype=np.float64)
y_all = df[TARGET].to_numpy(dtype=np.float64)
dates  = df["date"].to_numpy()

print(f"Dataset: {len(df):,} rows, {df['date'].nunique()} unique dates")
print(f"Date range: {dates[0]} to {dates[-1]}")

# ── 2. Time-series 5-fold CV split on date blocks ─────────────────────────────
# Assign each unique date to one of 5 consecutive bins, then derive row folds.
n_folds = 5
unique_dates = np.array(sorted(df["date"].unique()))
n_days = len(unique_dates)

# Assign fold index to each date  (fold 0 = earliest, fold 4 = latest)
date_fold = np.digitize(
    np.arange(n_days),
    bins=np.linspace(0, n_days, n_folds + 1)[1:-1],  # 4 interior cut-points
)  # shape (n_days,), values 0..4

date_to_fold = dict(zip(unique_dates, date_fold))
row_fold = np.array([date_to_fold[d] for d in dates])  # fold label per row

print(f"\nFold sizes (rows):")
for f in range(n_folds):
    mask = row_fold == f
    days_in = np.unique(dates[mask])
    print(f"  fold {f}: {mask.sum():>6,} rows  |  {len(days_in):2d} days  "
          f"({days_in[0]} .. {days_in[-1]})")

# Walk-forward splits: train on folds 0..k-1, test on fold k  (k = 1..4)
# Fold 0 is only ever training data (pure warm-up).
splits = []
for k in range(1, n_folds):
    tr_mask = row_fold < k
    te_mask = row_fold == k
    splits.append((np.where(tr_mask)[0], np.where(te_mask)[0]))

print(f"\n{len(splits)} walk-forward splits (train on folds 0..k-1, test on fold k)\n")

# ── 3. Metrics helper ─────────────────────────────────────────────────────────
def metrics(y_true, y_pred):
    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return r2, mae, rmse

# ── 4. Almgren-Chr hybrid formula ─────────────────────────────────────────────
# impact = c1 * daily_roll_spread + c2 * trail_1min_volatility * prate^0.5 + c3
# Feature indices:
IDX_SPREAD = FEATURES.index("daily_roll_spread")
IDX_VOL    = FEATURES.index("trail_1min_volatility")
IDX_PRATE  = FEATURES.index("participation_rate")

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
        print(f"    [Almgren fit failed: {e}] — falling back to intercept-only")
        intercept = y_tr.mean()
        return np.full(len(X_te), intercept), (0.0, 0.0, intercept)

# ── 5. Cross-validation loop ──────────────────────────────────────────────────
results = {name: {"r2": [], "mae": [], "rmse": []}
           for name in ["OLS", "Almgren-Chr", "Lasso", "RandomForest", "XGBoost"]}

xgb_importances_accum = np.zeros(len(FEATURES))

for fold_idx, (tr_idx, te_idx) in enumerate(splits):
    X_tr, X_te = X_all[tr_idx], X_all[te_idx]
    y_tr, y_te = y_all[tr_idx], y_all[te_idx]
    print(f"--- Fold {fold_idx + 1}/{len(splits)}  "
          f"(train={len(tr_idx):,}  test={len(te_idx):,}) ---")

    # ── Model 1: OLS ────────────────────────────────────────────────────────
    ols = LinearRegression()
    ols.fit(X_tr, y_tr)
    y_hat_ols = ols.predict(X_te)
    r2, mae, rmse = metrics(y_te, y_hat_ols)
    results["OLS"]["r2"].append(r2)
    results["OLS"]["mae"].append(mae)
    results["OLS"]["rmse"].append(rmse)
    print(f"  OLS:          R2={r2:+.4f}  MAE={mae:.4f}  RMSE={rmse:.4f}")

    # ── Model 2: Almgren-Chr hybrid ──────────────────────────────────────────
    y_hat_alm, popt = fit_almgren(X_tr, y_tr, X_te)
    r2, mae, rmse = metrics(y_te, y_hat_alm)
    results["Almgren-Chr"]["r2"].append(r2)
    results["Almgren-Chr"]["mae"].append(mae)
    results["Almgren-Chr"]["rmse"].append(rmse)
    print(f"  Almgren-Chr:  R2={r2:+.4f}  MAE={mae:.4f}  RMSE={rmse:.4f}  "
          f"c=({popt[0]:.3f},{popt[1]:.3f},{popt[2]:.3f})")

    # ── Model 3: Lasso (StandardScaler + LassoCV inner CV) ──────────────────
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    # LassoCV uses its own internal k-fold; we don't need a separate inner CV
    lasso_cv = LassoCV(
        alphas=np.logspace(-4, 2, 60),
        cv=5, max_iter=10_000, random_state=42, n_jobs=1,
    )
    lasso_cv.fit(X_tr_s, y_tr)
    y_hat_lasso = lasso_cv.predict(X_te_s)
    r2, mae, rmse = metrics(y_te, y_hat_lasso)
    results["Lasso"]["r2"].append(r2)
    results["Lasso"]["mae"].append(mae)
    results["Lasso"]["rmse"].append(rmse)
    print(f"  Lasso:        R2={r2:+.4f}  MAE={mae:.4f}  RMSE={rmse:.4f}  "
          f"alpha={lasso_cv.alpha_:.5f}")

    # ── Model 4: Random Forest ───────────────────────────────────────────────
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=50,
        max_features=0.5,
        n_jobs=1,
        random_state=42,
    )
    rf.fit(X_tr, y_tr)
    y_hat_rf = rf.predict(X_te)
    r2, mae, rmse = metrics(y_te, y_hat_rf)
    results["RandomForest"]["r2"].append(r2)
    results["RandomForest"]["mae"].append(mae)
    results["RandomForest"]["rmse"].append(rmse)
    print(f"  RandomForest: R2={r2:+.4f}  MAE={mae:.4f}  RMSE={rmse:.4f}")

    # ── Model 5: XGBoost ────────────────────────────────────────────────────
    xgb_model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=50,
        reg_alpha=0.1,
        reg_lambda=1.0,
        tree_method="hist",
        random_state=42,
        n_jobs=1,
        verbosity=0,
    )
    xgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_te, y_te)],
        verbose=False,
    )
    y_hat_xgb = xgb_model.predict(X_te)
    r2, mae, rmse = metrics(y_te, y_hat_xgb)
    results["XGBoost"]["r2"].append(r2)
    results["XGBoost"]["mae"].append(mae)
    results["XGBoost"]["rmse"].append(rmse)
    print(f"  XGBoost:      R2={r2:+.4f}  MAE={mae:.4f}  RMSE={rmse:.4f}")
    xgb_importances_accum += xgb_model.feature_importances_

print()

# ── 6. Aggregate results ───────────────────────────────────────────────────────
summary_rows = []
print("=" * 68)
print(f"{'Model':<16}  {'Mean R2':>9}  {'Std R2':>8}  {'Mean MAE':>9}  {'Mean RMSE':>10}")
print("=" * 68)
for name, vals in results.items():
    r2_arr   = np.array(vals["r2"])
    mae_arr  = np.array(vals["mae"])
    rmse_arr = np.array(vals["rmse"])
    r2_mean, r2_std = r2_arr.mean(), r2_arr.std()
    mae_mean        = mae_arr.mean()
    rmse_mean       = rmse_arr.mean()
    summary_rows.append({
        "model": name, "r2_mean": r2_mean, "r2_std": r2_std,
        "mae_mean": mae_mean, "rmse_mean": rmse_mean,
    })
    print(f"  {name:<14}  {r2_mean:>+9.4f}  {r2_std:>8.4f}  "
          f"{mae_mean:>9.4f}  {rmse_mean:>10.4f}")
print("=" * 68)

summary = pd.DataFrame(summary_rows).set_index("model")

# ── 7. XGBoost feature importances (averaged over folds) ─────────────────────
xgb_imp = xgb_importances_accum / len(splits)
imp_df = (
    pd.DataFrame({"feature": FEATURES, "importance": xgb_imp})
    .sort_values("importance", ascending=False)
    .reset_index(drop=True)
)
print("\n--- XGBoost feature importances (mean gain, averaged over folds) ---")
for _, row in imp_df.iterrows():
    bar = "#" * int(row["importance"] * 200)
    print(f"  {row['feature']:<25}  {row['importance']:.4f}  {bar}")

# ── 8. Bar chart: mean OOS R2 per model ──────────────────────────────────────
model_names = list(results.keys())
r2_means = [np.array(results[m]["r2"]).mean() for m in model_names]
r2_stds  = [np.array(results[m]["r2"]).std()  for m in model_names]
colors = ["#64748b", "#0ea5e9", "#f59e0b", "#10b981", "#ef4444"]

fig, ax = plt.subplots(figsize=(10, 5))

bars = ax.bar(
    model_names, r2_means,
    yerr=r2_stds, capsize=5,
    color=colors, edgecolor="white", linewidth=0.8,
    error_kw=dict(elinewidth=1.5, capthick=1.5, ecolor="black"),
)

# Value labels on bars
for bar, val, err in zip(bars, r2_means, r2_stds):
    y_pos = val + err + 0.001 if val >= 0 else val - err - 0.001
    va    = "bottom" if val >= 0 else "top"
    ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
            f"{val:+.4f}", ha="center", va=va, fontsize=9.5, fontweight="bold")

ax.axhline(0, color="black", lw=1, ls="--", alpha=0.6)
ax.set_ylabel("Mean out-of-sample R2 (4-fold walk-forward)", fontsize=11)
ax.set_title(
    "AAPL lit buy block trades — model comparison\n"
    "(time-series 5-fold CV, predict impact_vwap_bps)",
    fontsize=12, fontweight="bold",
)
ax.set_xticklabels(model_names, fontsize=10)
ax.grid(axis="y", alpha=0.25)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("aapl_model_comparison.png", dpi=150, bbox_inches="tight")
print("\nsaved -> aapl_model_comparison.png")
