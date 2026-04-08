"""
Generalized Gaussian extension of XGB GAMLSS.

Estimates shape parameter p of the generalized Gaussian from standardized
residuals, then recalibrates prediction intervals vs the Laplace (p=1) baseline.

Uses the same location/scale XGBoost models and hyperparameters as gamlss_xgb.py.

Output:
  - aapl_coin_gengauss.png
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
from scipy.optimize import minimize_scalar
from scipy.special import gamma as gammafn
from scipy.stats import gennorm, chi2
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

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
