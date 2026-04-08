"""
XGB-MAE residual distribution analysis for AAPL and COIN.
Fits Normal and Laplace to residuals, reports AIC/KS/kurtosis.
QQ-plot of residuals against Laplace quantiles.
Output: aapl_residual_distribution.png
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from scipy import stats
from xgboost import XGBRegressor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FEATURES = ["roll_spread_500", "roll_vol_500", "participation_rate"]


def fit_and_report(ticker, X_tr, y_tr):
    xgb = XGBRegressor(
        objective="reg:absoluteerror",
        max_depth=3, n_estimators=50, learning_rate=0.1,
        min_child_weight=1, reg_alpha=10, reg_lambda=10,
        tree_method="hist", verbosity=0, random_state=42, n_jobs=1,
    )
    xgb.fit(X_tr, y_tr)
    pred = np.maximum(xgb.predict(X_tr), 0.0)
    resid = y_tr - pred

    kurt = stats.kurtosis(resid, fisher=True)

    mu_n, sig_n = stats.norm.fit(resid)
    ll_n = stats.norm.logpdf(resid, mu_n, sig_n).sum()
    aic_n = 2 * 2 - 2 * ll_n

    loc_l, b_l = stats.laplace.fit(resid)
    ll_l = stats.laplace.logpdf(resid, loc_l, b_l).sum()
    aic_l = 2 * 2 - 2 * ll_l

    rs = np.sort(resid)
    n = len(rs)
    ecdf = np.arange(1, n + 1) / n
    ecdf_m = np.arange(0, n) / n

    cdf_n = stats.norm.cdf(rs, mu_n, sig_n)
    D_n = max(np.max(ecdf - cdf_n), np.max(cdf_n - ecdf_m))
    p_n = stats.kstwo.sf(D_n, n)

    cdf_l = stats.laplace.cdf(rs, loc_l, b_l)
    D_l = max(np.max(ecdf - cdf_l), np.max(cdf_l - ecdf_m))
    p_l = stats.kstwo.sf(D_l, n)

    print(f"\n  {ticker} XGB-MAE residuals (n={n:,})")
    print(f"  mean={resid.mean():.4f}  std={resid.std():.4f}  kurtosis={kurt:.4f}")
    print(f"  (Normal kurtosis=0, Laplace kurtosis=3)")
    print(f"\n  {'Distribution':<12} {'AIC':>12} {'KS stat':>10} {'p-value':>12}")
    print(f"  {'-'*48}")
    best_aic = min(aic_n, aic_l)
    print(f"  {'Normal':<12} {aic_n:>12.2f} {D_n:>10.6f} {p_n:>12.2e}"
          f"{' <-- best' if aic_n == best_aic else ''}")
    print(f"  {'Laplace':<12} {aic_l:>12.2f} {D_l:>10.6f} {p_l:>12.2e}"
          f"{' <-- best' if aic_l == best_aic else ''}")

    return resid, loc_l, b_l


# Load data
aapl_tr = pd.read_parquet("data/lit_buy_features_v2.parquet")
coin_tr = pd.read_parquet("data/coin_lit_buy_features_train.parquet")
aapl_tr["abs_impact"] = aapl_tr["impact_vwap_bps"].abs()
coin_tr["abs_impact"] = coin_tr["impact_vwap_bps"].abs()

print("=" * 60)
resid_a, loc_a, b_a = fit_and_report(
    "AAPL",
    aapl_tr[FEATURES].to_numpy(dtype=np.float64),
    aapl_tr["abs_impact"].to_numpy(dtype=np.float64),
)

resid_c, loc_c, b_c = fit_and_report(
    "COIN",
    coin_tr[FEATURES].to_numpy(dtype=np.float64),
    coin_tr["abs_impact"].to_numpy(dtype=np.float64),
)
print(f"\n{'='*60}")

# Plot: 2 rows (AAPL, COIN) x 2 cols (histogram + QQ)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for row, (ticker, resid, loc_l, b_l, color) in enumerate([
    ("AAPL", resid_a, loc_a, b_a, "#2563eb"),
    ("COIN", resid_c, loc_c, b_c, "#dc2626"),
]):
    kurt = stats.kurtosis(resid, fisher=True)
    n = len(resid)

    # Left: histogram + fitted PDFs
    ax = axes[row, 0]
    lo = np.percentile(resid, 0.5)
    hi = np.percentile(resid, 99.5)
    x_grid = np.linspace(lo, hi, 600)

    ax.hist(resid[(resid >= lo) & (resid <= hi)], bins=100, density=True,
            color="#e2e8f0", alpha=0.7, edgecolor="none", zorder=1)

    mu_n, sig_n = stats.norm.fit(resid)
    ax.plot(x_grid, stats.norm.pdf(x_grid, mu_n, sig_n),
            color="#94a3b8", lw=2, ls="--", label="Normal", zorder=3)
    ax.plot(x_grid, stats.laplace.pdf(x_grid, loc_l, b_l),
            color=color, lw=2.5, label="Laplace", zorder=3)

    ax.set_xlim(lo, hi)
    ax.set_xlabel("Residual (bps)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(f"{ticker}: XGB-MAE residual distribution (n={n:,}, kurtosis={kurt:.2f})",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Right: QQ plot against Laplace
    ax2 = axes[row, 1]
    rs = np.sort(resid)
    theoretical_q = stats.laplace.ppf(np.linspace(1 / (n + 1), n / (n + 1), n), loc_l, b_l)

    ax2.scatter(theoretical_q, rs, s=3, alpha=0.3, color=color, edgecolors="none")
    q_min = min(theoretical_q.min(), rs.min())
    q_max = max(theoretical_q.max(), rs.max())
    ax2.plot([q_min, q_max], [q_min, q_max], "k--", lw=1.5, alpha=0.7, label="y = x")
    ax2.set_xlabel("Laplace theoretical quantiles (bps)", fontsize=10)
    ax2.set_ylabel("Sample quantiles (bps)", fontsize=10)
    ax2.set_title(f"{ticker}: QQ plot vs Laplace", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.2)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.set_aspect("equal", adjustable="datalim")

fig.suptitle("XGB-MAE residual analysis: Normal vs Laplace fit + QQ plot",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("aapl_residual_distribution.png", dpi=150, bbox_inches="tight")
print(f"\nSaved -> aapl_residual_distribution.png")
