"""
Summary visualizations for the trexquant market impact project.

Produces three figures:
  1. model_comparison.png — OOS metrics for all 6 point-prediction models (AAPL + COIN)
  2. gamlss_calibration_overlay.png — Linear vs XGB vs Bayesian calibration overlaid
  3. eda_overview.png — Data EDA: target distributions, feature correlations, feature vs impact
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

FEATURES = [
    "dollar_value", "log_dollar_value", "participation_rate",
    "roll_spread_500", "roll_vol_500", "exchange_id",
]
FEAT_SHORT = ["dollar_val", "log_dollar", "prate", "spread", "vol", "exch_id"]

# =============================================================================
# FIGURE 1: Model Comparison Dashboard
# =============================================================================
print("Building Figure 1: Model Comparison Dashboard...", flush=True)

ols = pd.read_csv("data/ols_results.csv")
oos = ols[ols["sample"] == "out_of_sample"].copy()

MODEL_ORDER = ["OLS", "OLS_LAD", "XGBoost_MSE", "XGBoost_LAD", "RF_MSE", "RF_LAD"]
MODEL_LABELS = ["OLS\n(MSE)", "OLS\n(LAD)", "XGB\n(MSE)", "XGB\n(LAD)", "RF\n(MSE)", "RF\n(LAD)"]
COLORS_AAPL = "#2563eb"
COLORS_COIN = "#dc2626"

fig1 = plt.figure(figsize=(22, 14))
gs1 = gridspec.GridSpec(2, 3, figure=fig1, wspace=0.30, hspace=0.45)

metrics = [
    ("MAE", "MAE (bps) — lower is better"),
    ("MedAE", "Median AE (bps) — lower is better"),
    ("RMSE", "RMSE (bps) — lower is better"),
    ("R2", "R² — higher is better"),
]

for idx, (col, ylabel) in enumerate(metrics):
    row, c = divmod(idx, 3) if idx < 3 else (1, idx - 3)
    if idx == 3:
        row, c = 1, 0
    elif idx < 3:
        row, c = 0, idx
    ax = fig1.add_subplot(gs1[row, c])

    x = np.arange(len(MODEL_ORDER))
    w = 0.35

    vals_aapl = []
    vals_coin = []
    for m in MODEL_ORDER:
        v_a = oos[(oos["ticker"] == "AAPL") & (oos["model"] == m)][col].values
        v_c = oos[(oos["ticker"] == "COIN") & (oos["model"] == m)][col].values
        vals_aapl.append(v_a[0] if len(v_a) > 0 else np.nan)
        vals_coin.append(v_c[0] if len(v_c) > 0 else np.nan)

    bars_a = ax.bar(x - w/2, vals_aapl, w, color=COLORS_AAPL, alpha=0.8, label="AAPL",
                    edgecolor="white", linewidth=0.5)
    bars_c = ax.bar(x + w/2, vals_coin, w, color=COLORS_COIN, alpha=0.8, label="COIN",
                    edgecolor="white", linewidth=0.5)

    # Value labels
    for bar, val in zip(bars_a, vals_aapl):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 * abs(bar.get_height()),
                    f"{val:.3f}" if abs(val) < 1 else f"{val:.2f}",
                    ha="center", va="bottom", fontsize=7, fontweight="bold", color=COLORS_AAPL)
    for bar, val in zip(bars_c, vals_coin):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 * abs(bar.get_height()),
                    f"{val:.3f}" if abs(val) < 1 else f"{val:.2f}",
                    ha="center", va="bottom", fontsize=7, fontweight="bold", color=COLORS_COIN)

    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_LABELS, fontsize=8.5)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(f"Out-of-Sample {col}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Highlight best
    if col == "R2":
        best_a = max(vals_aapl)
        best_c = max(vals_coin)
    else:
        best_a = min(v for v in vals_aapl if not np.isnan(v))
        best_c = min(v for v in vals_coin if not np.isnan(v))
    for i, (va, vc) in enumerate(zip(vals_aapl, vals_coin)):
        if va == best_a:
            bars_a[i].set_edgecolor("#000000")
            bars_a[i].set_linewidth(2)
        if vc == best_c:
            bars_c[i].set_edgecolor("#000000")
            bars_c[i].set_linewidth(2)

# Panel 5: In-sample vs OOS MAE (overfitting check)
ax5 = fig1.add_subplot(gs1[1, 1])
ins = ols[ols["sample"] == "in_sample"]

for i, m in enumerate(MODEL_ORDER):
    mae_in_a = ins[(ins["ticker"] == "AAPL") & (ins["model"] == m)]["MAE"].values
    mae_oos_a = oos[(oos["ticker"] == "AAPL") & (oos["model"] == m)]["MAE"].values
    if len(mae_in_a) > 0 and len(mae_oos_a) > 0:
        ax5.scatter(mae_in_a[0], mae_oos_a[0], s=100, color=COLORS_AAPL, zorder=3,
                    edgecolors="white", linewidth=1)
        ax5.annotate(MODEL_ORDER[i].replace("_", "\n"), xy=(mae_in_a[0], mae_oos_a[0]),
                     textcoords="offset points", xytext=(8, -4), fontsize=6.5, color=COLORS_AAPL)

    mae_in_c = ins[(ins["ticker"] == "COIN") & (ins["model"] == m)]["MAE"].values
    mae_oos_c = oos[(oos["ticker"] == "COIN") & (oos["model"] == m)]["MAE"].values
    if len(mae_in_c) > 0 and len(mae_oos_c) > 0:
        ax5.scatter(mae_in_c[0], mae_oos_c[0], s=100, color=COLORS_COIN, zorder=3,
                    edgecolors="white", linewidth=1, marker="s")
        ax5.annotate(MODEL_ORDER[i].replace("_", "\n"), xy=(mae_in_c[0], mae_oos_c[0]),
                     textcoords="offset points", xytext=(8, -4), fontsize=6.5, color=COLORS_COIN)

ax5.plot([0, 10], [0, 10], color="black", lw=1, ls="--", alpha=0.4, label="No overfitting line")
ax5.set_xlabel("In-sample MAE (bps)", fontsize=10)
ax5.set_ylabel("Out-of-sample MAE (bps)", fontsize=10)
ax5.set_title("Overfitting Check: In-sample vs OOS MAE\n(above diagonal = overfitting)",
              fontsize=11, fontweight="bold")
ax5.legend(fontsize=8, handles=[
    Patch(facecolor=COLORS_AAPL, label="AAPL"),
    Patch(facecolor=COLORS_COIN, label="COIN"),
    plt.Line2D([0], [0], color="black", ls="--", lw=1, label="No overfit"),
])
ax5.grid(True, alpha=0.15)
ax5.spines["top"].set_visible(False)
ax5.spines["right"].set_visible(False)

# Panel 6: MSE vs LAD objective comparison
ax6 = fig1.add_subplot(gs1[1, 2])

pairs = [("OLS", "OLS_LAD"), ("XGBoost_MSE", "XGBoost_LAD"), ("RF_MSE", "RF_LAD")]
pair_labels = ["OLS", "XGBoost", "RF"]
pair_colors = ["#7c3aed", "#f59e0b", "#16a34a"]

for j, (mse_m, lad_m) in enumerate(pairs):
    for ticker, marker in [("AAPL", "o"), ("COIN", "s")]:
        mse_mae = oos[(oos["ticker"] == ticker) & (oos["model"] == mse_m)]["MAE"].values
        lad_mae = oos[(oos["ticker"] == ticker) & (oos["model"] == lad_m)]["MAE"].values
        if len(mse_mae) > 0 and len(lad_mae) > 0:
            ax6.scatter(mse_mae[0], lad_mae[0], s=120, color=pair_colors[j], marker=marker,
                        edgecolors="white", linewidth=1, zorder=3)
            ax6.annotate(f"{pair_labels[j]}\n{ticker}", xy=(mse_mae[0], lad_mae[0]),
                         textcoords="offset points", xytext=(8, -4), fontsize=6.5,
                         color=pair_colors[j])

lim = [1.2, 4.5]
ax6.plot(lim, lim, color="black", lw=1, ls="--", alpha=0.4, label="Equal MAE")
ax6.set_xlabel("MSE objective — OOS MAE (bps)", fontsize=10)
ax6.set_ylabel("LAD objective — OOS MAE (bps)", fontsize=10)
ax6.set_title("MSE vs LAD Objective: OOS MAE\n(below diagonal = LAD wins)",
              fontsize=11, fontweight="bold")
ax6.legend(fontsize=8, handles=[
    Patch(facecolor=pair_colors[0], label="OLS"),
    Patch(facecolor=pair_colors[1], label="XGBoost"),
    Patch(facecolor=pair_colors[2], label="RF"),
    plt.Line2D([0], [0], color="black", ls="--", lw=1, label="Equal"),
])
ax6.grid(True, alpha=0.15)
ax6.spines["top"].set_visible(False)
ax6.spines["right"].set_visible(False)

fig1.suptitle(
    "Model Comparison Dashboard: Out-of-Sample Performance on |impact_vwap_bps|\n"
    "AAPL (35K train, 9.2K test)  |  COIN (15.7K train, 959 test)  |  6 features  |  "
    "Bold border = best per ticker",
    fontsize=14, fontweight="bold", y=1.01,
)
plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
print("Saved -> model_comparison.png")

# =============================================================================
# FIGURE 2: GAMLSS Calibration Overlay
# =============================================================================
print("Building Figure 2: GAMLSS Calibration Overlay...", flush=True)

# Recompute calibration curves from the raw data
df_tr_a = pd.read_parquet("data/lit_buy_features_v2.parquet")
df_te_a = pd.read_parquet("data/lit_buy_features_v2_sep.parquet")
df_tr_c = pd.read_parquet("data/coin_lit_buy_features_train.parquet")
df_te_c = pd.read_parquet("data/coin_lit_buy_features_test.parquet")

for df in [df_tr_a, df_te_a, df_tr_c, df_te_c]:
    df["abs_impact"] = df["impact_vwap_bps"].abs()
df_tr_a = df_tr_a.sort_values("date").reset_index(drop=True)
df_tr_c = df_tr_c.sort_values("date").reset_index(drop=True)

import warnings
import statsmodels.api as sm
from xgboost import XGBRegressor

cal_levels = np.linspace(0.05, 0.99, 50)


def laplace_coverage(y, mu, b, level):
    z = np.log(1.0 / (1.0 - level))
    lo = np.maximum(mu - z * b, 0.0)
    hi = mu + z * b
    return ((y >= lo) & (y <= hi)).mean()


def fit_linear_gamlss(X_tr, y_tr, X_te, y_te):
    """LAD location + OLS scale (same as gamlss_laplace.py)."""
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


def fit_xgb_gamlss(X_tr, y_tr, X_te, y_te):
    """XGB LAD location + XGB MSE scale (same as gamlss_xgb.py)."""
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


fig2, axes2 = plt.subplots(1, 2, figsize=(16, 7))

for ax, (ticker, df_tr, df_te) in zip(axes2, [
    ("AAPL", df_tr_a, df_te_a),
    ("COIN", df_tr_c, df_te_c),
]):
    X_tr = df_tr[FEATURES].to_numpy(dtype=np.float64)
    y_tr = df_tr["abs_impact"].to_numpy(dtype=np.float64)
    X_te = df_te[FEATURES].to_numpy(dtype=np.float64)
    y_te = df_te["abs_impact"].to_numpy(dtype=np.float64)

    # Linear GAMLSS
    mu_lin, b_lin = fit_linear_gamlss(X_tr, y_tr, X_te, y_te)
    cal_lin = [laplace_coverage(y_te, mu_lin, b_lin, lv) for lv in cal_levels]

    # XGB GAMLSS
    mu_xgb, b_xgb = fit_xgb_gamlss(X_tr, y_tr, X_te, y_te)
    cal_xgb = [laplace_coverage(y_te, mu_xgb, b_xgb, lv) for lv in cal_levels]

    ax.plot(cal_levels, cal_lin, color="#2563eb", lw=2.5, marker="o", markersize=3,
            label="Linear GAMLSS (QuantReg + OLS scale)", alpha=0.9)
    ax.plot(cal_levels, cal_xgb, color="#dc2626", lw=2.5, marker="s", markersize=3,
            label="XGB GAMLSS (XGB location + XGB scale)", alpha=0.9)
    ax.plot([0, 1], [0, 1], color="black", lw=1.2, ls="--", alpha=0.5,
            label="Perfect calibration")

    # Annotate key levels
    for lv in [0.50, 0.90]:
        cv_l = laplace_coverage(y_te, mu_lin, b_lin, lv)
        cv_x = laplace_coverage(y_te, mu_xgb, b_xgb, lv)
        ax.annotate(f"Linear {cv_l:.1%}", xy=(lv, cv_l), textcoords="offset points",
                    xytext=(-50, 10), fontsize=7.5, color="#2563eb", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#2563eb", lw=0.7))
        ax.annotate(f"XGB {cv_x:.1%}", xy=(lv, cv_x), textcoords="offset points",
                    xytext=(10, -15), fontsize=7.5, color="#dc2626", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="#dc2626", lw=0.7))

    ax.set_xlabel("Nominal coverage level", fontsize=11)
    ax.set_ylabel("Actual coverage", fontsize=11)
    ax.set_title(f"{ticker}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig2.suptitle(
    "GAMLSS Calibration Comparison: Linear (QuantReg location + OLS scale) vs XGBoost (both stages)\n"
    "Closer to diagonal = better calibrated  |  Laplace(mu, b) prediction intervals",
    fontsize=14, fontweight="bold", y=1.02,
)
plt.tight_layout()
plt.savefig("gamlss_calibration_overlay.png", dpi=150, bbox_inches="tight")
print("Saved -> gamlss_calibration_overlay.png")

# =============================================================================
# FIGURE 3: EDA Overview
# =============================================================================
print("Building Figure 3: EDA Overview...", flush=True)

fig3 = plt.figure(figsize=(26, 16))
gs3 = gridspec.GridSpec(3, 4, figure=fig3, wspace=0.30, hspace=0.45)

rng = np.random.default_rng(42)

# -- Row 1, Col 1-2: Target distribution (AAPL and COIN) ---------------------
for col_idx, (ticker, df_tr, df_te) in enumerate([
    ("AAPL", df_tr_a, df_te_a), ("COIN", df_tr_c, df_te_c)
]):
    ax = fig3.add_subplot(gs3[0, col_idx])
    y_all = pd.concat([df_tr["abs_impact"], df_te["abs_impact"]])
    clip = np.percentile(y_all, 98)
    ax.hist(df_tr["abs_impact"].clip(upper=clip), bins=80, density=True, alpha=0.6,
            color="#2563eb", edgecolor="white", linewidth=0.3, label="Train")
    ax.hist(df_te["abs_impact"].clip(upper=clip), bins=80, density=True, alpha=0.5,
            color="#dc2626", edgecolor="white", linewidth=0.3, label="Test")
    ax.axvline(df_tr["abs_impact"].median(), color="#2563eb", lw=1.5, ls="--",
               label=f"Train median: {df_tr['abs_impact'].median():.2f}")
    ax.axvline(df_te["abs_impact"].median(), color="#dc2626", lw=1.5, ls="--",
               label=f"Test median: {df_te['abs_impact'].median():.2f}")
    ax.set_xlabel("|impact_vwap_bps|", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(f"{ticker} Target Distribution\n(clipped at p98={clip:.1f} bps)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=7.5)
    ax.grid(True, alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# -- Row 1, Col 3: Target stats table ----------------------------------------
ax_table = fig3.add_subplot(gs3[0, 2])
ax_table.axis("off")

stats_data = []
for ticker, df_tr, df_te in [("AAPL", df_tr_a, df_te_a), ("COIN", df_tr_c, df_te_c)]:
    for label, df in [("Train", df_tr), ("Test", df_te)]:
        y = df["abs_impact"]
        stats_data.append([
            f"{ticker} {label}", f"{len(df):,}", f"{y.mean():.2f}", f"{y.median():.2f}",
            f"{y.std():.2f}", f"{y.skew():.1f}", f"{y.kurtosis():.0f}",
        ])

col_labels = ["Dataset", "N", "Mean", "Median", "Std", "Skew", "Kurt"]
table = ax_table.table(cellText=stats_data, colLabels=col_labels,
                       cellLoc="center", loc="center")
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.0, 1.6)
for (r, c), cell in table.get_celld().items():
    if r == 0:
        cell.set_facecolor("#e0e7ff")
        cell.set_text_props(fontweight="bold")
    cell.set_edgecolor("#d1d5db")
ax_table.set_title("Target Summary Statistics\n|impact_vwap_bps|",
                    fontsize=11, fontweight="bold", pad=20)

# -- Row 1, Col 4: Train/test time split -------------------------------------
ax_ts = fig3.add_subplot(gs3[0, 3])
for ticker, df_tr, df_te, color in [
    ("AAPL", df_tr_a, df_te_a, "#2563eb"),
    ("COIN", df_tr_c, df_te_c, "#dc2626"),
]:
    tr_dates = pd.to_datetime(df_tr["date"])
    te_dates = pd.to_datetime(df_te["date"])
    tr_daily = tr_dates.dt.date.value_counts().sort_index()
    te_daily = te_dates.dt.date.value_counts().sort_index()
    ax_ts.bar(tr_daily.index, tr_daily.values, color=color, alpha=0.4, width=1, label=f"{ticker} train")
    ax_ts.bar(te_daily.index, te_daily.values, color=color, alpha=0.8, width=1, label=f"{ticker} test")
ax_ts.set_xlabel("Date", fontsize=10)
ax_ts.set_ylabel("Trades per day", fontsize=10)
ax_ts.set_title("Temporal Train/Test Split\n(walk-forward holdout)", fontsize=11, fontweight="bold")
ax_ts.legend(fontsize=7.5)
ax_ts.tick_params(axis="x", rotation=30, labelsize=7)
ax_ts.grid(axis="y", alpha=0.2)
ax_ts.spines["top"].set_visible(False)
ax_ts.spines["right"].set_visible(False)

# -- Row 2: Feature distributions (AAPL train) --------------------------------
for i, (feat, short) in enumerate(zip(FEATURES, FEAT_SHORT)):
    ax = fig3.add_subplot(gs3[1, i]) if i < 4 else None
    if ax is None:
        break
    vals_a = df_tr_a[feat].values
    vals_c = df_tr_c[feat].values
    clip_lo = np.percentile(np.concatenate([vals_a, vals_c]), 1)
    clip_hi = np.percentile(np.concatenate([vals_a, vals_c]), 99)
    ax.hist(np.clip(vals_a, clip_lo, clip_hi), bins=60, density=True, alpha=0.6,
            color="#2563eb", edgecolor="white", linewidth=0.3, label="AAPL")
    ax.hist(np.clip(vals_c, clip_lo, clip_hi), bins=60, density=True, alpha=0.5,
            color="#dc2626", edgecolor="white", linewidth=0.3, label="COIN")
    ax.set_xlabel(feat, fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.set_title(f"{feat}", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# -- Row 3, Col 1: Feature correlation heatmap (AAPL) -------------------------
ax_corr = fig3.add_subplot(gs3[2, 0])
corr = df_tr_a[FEATURES + ["abs_impact"]].corr()
im = ax_corr.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
labels_corr = FEAT_SHORT + ["impact"]
ax_corr.set_xticks(range(len(labels_corr)))
ax_corr.set_xticklabels(labels_corr, fontsize=7.5, rotation=45, ha="right")
ax_corr.set_yticks(range(len(labels_corr)))
ax_corr.set_yticklabels(labels_corr, fontsize=7.5)
for i2 in range(len(labels_corr)):
    for j2 in range(len(labels_corr)):
        ax_corr.text(j2, i2, f"{corr.iloc[i2, j2]:.2f}", ha="center", va="center",
                     fontsize=6.5, color="white" if abs(corr.iloc[i2, j2]) > 0.5 else "black")
plt.colorbar(im, ax=ax_corr, fraction=0.046, pad=0.04)
ax_corr.set_title("AAPL Feature Correlations\n(Pearson)", fontsize=10, fontweight="bold")

# -- Row 3, Col 2: Feature correlation heatmap (COIN) -------------------------
ax_corr2 = fig3.add_subplot(gs3[2, 1])
corr2 = df_tr_c[FEATURES + ["abs_impact"]].corr()
im2 = ax_corr2.imshow(corr2, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax_corr2.set_xticks(range(len(labels_corr)))
ax_corr2.set_xticklabels(labels_corr, fontsize=7.5, rotation=45, ha="right")
ax_corr2.set_yticks(range(len(labels_corr)))
ax_corr2.set_yticklabels(labels_corr, fontsize=7.5)
for i2 in range(len(labels_corr)):
    for j2 in range(len(labels_corr)):
        ax_corr2.text(j2, i2, f"{corr2.iloc[i2, j2]:.2f}", ha="center", va="center",
                      fontsize=6.5, color="white" if abs(corr2.iloc[i2, j2]) > 0.5 else "black")
plt.colorbar(im2, ax=ax_corr2, fraction=0.046, pad=0.04)
ax_corr2.set_title("COIN Feature Correlations\n(Pearson)", fontsize=10, fontweight="bold")

# -- Row 3, Col 3: Scatter: participation_rate vs impact ----------------------
ax_sc1 = fig3.add_subplot(gs3[2, 2])
samp_a = rng.choice(len(df_tr_a), size=min(3000, len(df_tr_a)), replace=False)
samp_c = rng.choice(len(df_tr_c), size=min(3000, len(df_tr_c)), replace=False)
ax_sc1.scatter(df_tr_a["participation_rate"].iloc[samp_a],
               df_tr_a["abs_impact"].iloc[samp_a], s=6, alpha=0.1,
               color="#2563eb", linewidths=0, rasterized=True, label="AAPL")
ax_sc1.scatter(df_tr_c["participation_rate"].iloc[samp_c],
               df_tr_c["abs_impact"].iloc[samp_c], s=6, alpha=0.1,
               color="#dc2626", linewidths=0, rasterized=True, label="COIN")
clip_y = max(np.percentile(df_tr_a["abs_impact"], 98), np.percentile(df_tr_c["abs_impact"], 98))
ax_sc1.set_ylim(0, clip_y)
ax_sc1.set_xlabel("participation_rate", fontsize=10)
ax_sc1.set_ylabel("|impact_vwap_bps|", fontsize=10)
ax_sc1.set_title("Participation Rate vs Impact\n(3K random trades each)",
                 fontsize=10, fontweight="bold")
ax_sc1.legend(fontsize=8, markerscale=5)
ax_sc1.grid(True, alpha=0.15)
ax_sc1.spines["top"].set_visible(False)
ax_sc1.spines["right"].set_visible(False)

# -- Row 3, Col 4: Scatter: roll_vol_500 vs impact ----------------------------
ax_sc2 = fig3.add_subplot(gs3[2, 3])
ax_sc2.scatter(df_tr_a["roll_vol_500"].iloc[samp_a],
               df_tr_a["abs_impact"].iloc[samp_a], s=6, alpha=0.1,
               color="#2563eb", linewidths=0, rasterized=True, label="AAPL")
ax_sc2.scatter(df_tr_c["roll_vol_500"].iloc[samp_c],
               df_tr_c["abs_impact"].iloc[samp_c], s=6, alpha=0.1,
               color="#dc2626", linewidths=0, rasterized=True, label="COIN")
ax_sc2.set_ylim(0, clip_y)
ax_sc2.set_xlabel("roll_vol_500", fontsize=10)
ax_sc2.set_ylabel("|impact_vwap_bps|", fontsize=10)
ax_sc2.set_title("Rolling Volatility vs Impact\n(3K random trades each)",
                 fontsize=10, fontweight="bold")
ax_sc2.legend(fontsize=8, markerscale=5)
ax_sc2.grid(True, alpha=0.15)
ax_sc2.spines["top"].set_visible(False)
ax_sc2.spines["right"].set_visible(False)

fig3.suptitle(
    "Data Exploration: AAPL and COIN Lit Buy Block Trades\n"
    "Target = |impact_vwap_bps|  |  6 features  |  Jun–Sep 2024",
    fontsize=14, fontweight="bold", y=1.01,
)
plt.savefig("eda_overview.png", dpi=150, bbox_inches="tight")
print("Saved -> eda_overview.png")

print("\nAll visualizations complete!")
