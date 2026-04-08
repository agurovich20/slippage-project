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
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

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
