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
