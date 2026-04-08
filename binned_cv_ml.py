"""
Time-series 5-fold CV at the binned level — parametric + ML models.

Identical split/bin logic to binned_cv.py (50 bins, 5-block walk-forward).

Parametric models (same as before, features: spread only / spread+prate):
  [1] Linear:    c1*spread + c2
  [2] Quadratic: c1*spread^2 + c2*spread + c3
  [3] Vlad bond: c1*spread + c2*prate^0.4 + c3

ML models (features: mean_spread, mean_vol, mean_prate — all three bin means):
  [4] Random Forest  — inner 3-fold GridSearchCV on training bins
  [5] XGBoost        — inner 3-fold GridSearchCV on training bins

GridSearchCV grids:
  RF:  n_estimators=[50,100,200]  max_depth=[2,3,5]  min_samples_leaf=[2,3,5]
  XGB: n_estimators=[50,100,200]  max_depth=[2,3]
       learning_rate=[0.05,0.1,0.2]  min_child_weight=[2,3,5]

Note: training set = ~50 bin means per fold; inner CV folds have ~16 points each.
StandardScaler applied before each ML model (fit on train bins only).

Output: aapl_binned_cv_ml.png
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"]      = "1"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import xgboost as xgb

# ── Load ───────────────────────────────────────────────────────────────────────
df = pd.read_parquet("data/lit_buy_features_v2.parquet")
df["abs_impact"] = df["impact_vwap_bps"].abs()
df = df.sort_values("date").reset_index(drop=True)

COLS  = ["date", "roll_spread_500", "roll_vol_500", "participation_rate", "abs_impact"]
df    = df[COLS].dropna()
dates = df["date"].to_numpy()

print(f"Loaded {len(df):,} rows across {df['date'].nunique()} dates")

N_BINS  = 50
N_FOLDS = 5
INNER_CV = 3

# ── Date-block fold assignment ─────────────────────────────────────────────────
unique_dates = np.array(sorted(df["date"].unique()))
n_days = len(unique_dates)

date_fold = np.digitize(
    np.arange(n_days),
    bins=np.linspace(0, n_days, N_FOLDS + 1)[1:-1],
)
date_to_fold = dict(zip(unique_dates, date_fold))
row_fold = np.array([date_to_fold[d] for d in dates])

print(f"\nFold sizes:")
for f in range(N_FOLDS):
    mask = row_fold == f
    days_in = np.unique(dates[mask])
    print(f"  block {f}: {mask.sum():>6,} rows  |  {len(days_in):2d} days  "
          f"({days_in[0]} .. {days_in[-1]})")

splits = []
for k in range(1, N_FOLDS):
    splits.append((np.where(row_fold < k)[0], np.where(row_fold == k)[0]))

print(f"\n{len(splits)} walk-forward OOS splits\n")

# ── Bin helper — now includes mean_vol ────────────────────────────────────────
def make_bins(data_idx):
    sub = df.iloc[data_idx][
        ["roll_spread_500", "roll_vol_500", "participation_rate", "abs_impact"]
    ].copy()
    sub["bin"] = pd.qcut(sub["roll_spread_500"], q=N_BINS, labels=False,
                          duplicates="drop")
    agg = (
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
    return agg

# ── OLS helpers ────────────────────────────────────────────────────────────────
def ols_fit(Xd, y):
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    yh = Xd @ beta
    ss_res = float(((y - yh) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return beta, (1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan)

def r2(y_true, y_pred):
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

# ── GridSearch grids ──────────────────────────────────────────────────────────
RF_GRID = {
    "rf__n_estimators":    [50, 100, 200],
    "rf__max_depth":       [2, 3, 5],
    "rf__min_samples_leaf":[2, 3, 5],
}

XGB_GRID = {
    "xgb__n_estimators":    [50, 100, 200],
    "xgb__max_depth":       [2, 3],
    "xgb__learning_rate":   [0.05, 0.1, 0.2],
    "xgb__min_child_weight":[2, 3, 5],
}

# ── CV loop ────────────────────────────────────────────────────────────────────
model_names = ["Linear", "Quadratic", "Vlad bond", "RandomForest", "XGBoost"]
oos_r2  = {m: [] for m in model_names}
best_params_log = {"RandomForest": [], "XGBoost": []}

for fold_idx, (tr_idx, te_idx) in enumerate(splits):
    tr_bins = make_bins(tr_idx)
    te_bins = make_bins(te_idx)

    s_tr = tr_bins["mean_spread"].to_numpy()
    v_tr = tr_bins["mean_vol"].to_numpy()
    p_tr = tr_bins["mean_prate"].to_numpy()
    y_tr = tr_bins["mean_abs"].to_numpy()
    n_tr = len(tr_bins)

    s_te = te_bins["mean_spread"].to_numpy()
    v_te = te_bins["mean_vol"].to_numpy()
    p_te = te_bins["mean_prate"].to_numpy()
    y_te = te_bins["mean_abs"].to_numpy()

    print(f"--- Fold {fold_idx+1}/{len(splits)}  "
          f"(train bins={n_tr}, test bins={len(te_bins)}) ---")

    # ── [1] Linear ──────────────────────────────────────────────────────────
    b1, _ = ols_fit(np.column_stack([s_tr, np.ones(n_tr)]), y_tr)
    r2_1  = r2(y_te, b1[0]*s_te + b1[1])
    oos_r2["Linear"].append(r2_1)
    print(f"  [1] Linear:    OOS R2={r2_1:+.4f}  c=({b1[0]:+.4f},{b1[1]:+.4f})")

    # ── [2] Quadratic ───────────────────────────────────────────────────────
    b2, _ = ols_fit(np.column_stack([s_tr**2, s_tr, np.ones(n_tr)]), y_tr)
    r2_2  = r2(y_te, b2[0]*s_te**2 + b2[1]*s_te + b2[2])
    oos_r2["Quadratic"].append(r2_2)
    print(f"  [2] Quadratic: OOS R2={r2_2:+.4f}  "
          f"c=({b2[0]:+.4f},{b2[1]:+.4f},{b2[2]:+.4f})")

    # ── [3] Vlad bond ───────────────────────────────────────────────────────
    b3, _ = ols_fit(np.column_stack([s_tr, p_tr**0.4, np.ones(n_tr)]), y_tr)
    r2_3  = r2(y_te, b3[0]*s_te + b3[1]*p_te**0.4 + b3[2])
    oos_r2["Vlad bond"].append(r2_3)
    print(f"  [3] Vlad bond: OOS R2={r2_3:+.4f}  "
          f"c=({b3[0]:+.4f},{b3[1]:+.4f},{b3[2]:+.4f})")

    # ML feature matrices (spread, vol, prate)
    X_tr_ml = np.column_stack([s_tr, v_tr, p_tr])
    X_te_ml = np.column_stack([s_te, v_te, p_te])

    # ── [4] Random Forest via GridSearchCV ──────────────────────────────────
    rf_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestRegressor(random_state=42, n_jobs=1)),
    ])
    rf_gs = GridSearchCV(
        rf_pipe, RF_GRID,
        cv=INNER_CV, scoring="r2",
        refit=True, n_jobs=1,
    )
    rf_gs.fit(X_tr_ml, y_tr)
    y_hat_rf = rf_gs.predict(X_te_ml)
    r2_4 = r2(y_te, y_hat_rf)
    oos_r2["RandomForest"].append(r2_4)
    best_rf = {k.replace("rf__",""): v
               for k, v in rf_gs.best_params_.items()}
    best_params_log["RandomForest"].append(best_rf)
    print(f"  [4] RF:        OOS R2={r2_4:+.4f}  "
          f"best={best_rf}  inner_R2={rf_gs.best_score_:.4f}")

    # ── [5] XGBoost via GridSearchCV ────────────────────────────────────────
    xgb_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("xgb", xgb.XGBRegressor(
            tree_method="hist", random_state=42,
            n_jobs=1, verbosity=0,
        )),
    ])
    xgb_gs = GridSearchCV(
        xgb_pipe, XGB_GRID,
        cv=INNER_CV, scoring="r2",
        refit=True, n_jobs=1,
    )
    xgb_gs.fit(X_tr_ml, y_tr)
    y_hat_xgb = xgb_gs.predict(X_te_ml)
    r2_5 = r2(y_te, y_hat_xgb)
    oos_r2["XGBoost"].append(r2_5)
    best_xgb = {k.replace("xgb__",""): v
                for k, v in xgb_gs.best_params_.items()}
    best_params_log["XGBoost"].append(best_xgb)
    print(f"  [5] XGBoost:   OOS R2={r2_5:+.4f}  "
          f"best={best_xgb}  inner_R2={xgb_gs.best_score_:.4f}")

# ── Summary table ──────────────────────────────────────────────────────────────
print("\n" + "="*75)
print(f"{'Model':<14}  {'Mean OOS R2':>12}  {'Std OOS R2':>11}  Fold R2s")
print("="*75)
for name in model_names:
    arr = np.array(oos_r2[name])
    fold_str = "  ".join(f"{v:+.4f}" for v in arr)
    print(f"  {name:<12}  {arr.mean():>+12.4f}  {arr.std():>11.4f}  [{fold_str}]")
print("="*75)

print("\nBest hyperparameters per fold:")
for name in ["RandomForest", "XGBoost"]:
    print(f"  {name}:")
    for i, p in enumerate(best_params_log[name]):
        print(f"    fold {i+1}: {p}")

# ── Bar chart ──────────────────────────────────────────────────────────────────
means  = [np.array(oos_r2[m]).mean() for m in model_names]
stds   = [np.array(oos_r2[m]).std()  for m in model_names]
colors = ["#2563eb", "#16a34a", "#dc2626", "#7c3aed", "#ea580c"]
labels = [
    "[1] Linear\nc1·s+c2",
    "[2] Quadratic\nc1·s²+c2·s+c3",
    "[3] Vlad bond\nc1·s+c2·p⁰·⁴+c3",
    "[4] Random\nForest",
    "[5] XGBoost",
]

fig, ax = plt.subplots(figsize=(12, 6))

x_pos = np.arange(len(model_names))
bars  = ax.bar(
    x_pos, means,
    yerr=stds, capsize=6,
    color=colors, edgecolor="white", linewidth=0.8, width=0.52,
    error_kw=dict(elinewidth=1.8, capthick=1.8, ecolor="black"),
)

# Individual fold dots
for i, name in enumerate(model_names):
    fold_vals = oos_r2[name]
    jitter = np.linspace(-0.13, 0.13, len(fold_vals))
    ax.scatter(
        x_pos[i] + jitter, fold_vals,
        color="white", edgecolors=colors[i],
        s=60, zorder=5, linewidths=2.0,
    )

# Value labels — clip annotation y to a sane range so quadratic label doesn't
# disappear off-canvas
y_min_ax = min(means) - max(stds) - 0.15
for i, (bar, val, err) in enumerate(zip(bars, means, stds)):
    ann_y = max(val + err + 0.02, y_min_ax + 0.05) if val >= 0 else val - err - 0.02
    va    = "bottom" if val >= 0 else "top"
    ax.text(
        bar.get_x() + bar.get_width() / 2, ann_y,
        f"{val:+.4f}\n(±{err:.4f})",
        ha="center", va=va, fontsize=8.5, fontweight="bold",
    )

ax.axhline(0, color="black", lw=1.2, ls="--", alpha=0.5)
ax.set_xticks(x_pos)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("OOS R² on 50 test-bin means", fontsize=11)
ax.set_title(
    "AAPL lit buy block trades — binned CV: parametric vs ML  (target: |impact_vwap_bps|)\n"
    "5-block walk-forward · 50 quantile bins by roll_spread_500 · "
    "ML features: mean_spread, mean_vol, mean_prate\n"
    "(dots = individual fold OOS R²;  bar = mean;  error bar = std)",
    fontsize=10.5, fontweight="bold",
)
ax.grid(axis="y", alpha=0.25)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("aapl_binned_cv_ml.png", dpi=150, bbox_inches="tight")
print("\nsaved -> aapl_binned_cv_ml.png")
