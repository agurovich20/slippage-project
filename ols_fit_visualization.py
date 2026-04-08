"""
OLS fit visualizations for AAPL and COIN lit buy block trades.
Target: signed impact_vwap_bps (slippage).

Four-panel figure:
  Top-left:     AAPL scatter + OLS line (signed impact vs roll_spread_500)
  Top-right:    COIN scatter + OLS line (signed impact vs roll_spread_500)
  Bottom-left:  AAPL residual diagnostics (residuals vs fitted)
  Bottom-right: COIN residual diagnostics (residuals vs fitted)

Plus a second figure: predicted vs actual + residual histograms.

Output:
  aapl_coin_ols_fit.png
  aapl_coin_ols_predicted_vs_actual.png
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
from scipy import stats

# ══════════════════════════════════════════════════════════════════════════════
# Load data
# ══════════════════════════════════════════════════════════════════════════════
aapl_tr = pd.read_parquet("data/lit_buy_features_v2.parquet")
aapl_te = pd.read_parquet("data/lit_buy_features_v2_sep.parquet")
coin_tr = pd.read_parquet("data/coin_lit_buy_features_train.parquet")
coin_te = pd.read_parquet("data/coin_lit_buy_features_test.parquet")

print(f"AAPL train: {len(aapl_tr):,}  test: {len(aapl_te):,}")
print(f"COIN train: {len(coin_tr):,}  test: {len(coin_te):,}")


# ══════════════════════════════════════════════════════════════════════════════
# OLS helper — univariate: impact_vwap_bps = c1 * spread + c2
# ══════════════════════════════════════════════════════════════════════════════
def ols_univariate(x_train, y_train, x_test, y_test):
    """Fit OLS on train, evaluate on both train and test."""
    X_tr = np.column_stack([x_train, np.ones(len(x_train))])
    X_te = np.column_stack([x_test, np.ones(len(x_test))])
    beta, *_ = np.linalg.lstsq(X_tr, y_train, rcond=None)

    yhat_tr = X_tr @ beta
    yhat_te = X_te @ beta

    def r2(y, yhat):
        ss_res = ((y - yhat) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "beta": beta,
        "yhat_tr": yhat_tr,
        "yhat_te": yhat_te,
        "resid_tr": y_train - yhat_tr,
        "resid_te": y_test - yhat_te,
        "r2_tr": r2(y_train, yhat_tr),
        "r2_te": r2(y_test, yhat_te),
        "mae_tr": np.mean(np.abs(y_train - yhat_tr)),
        "mae_te": np.mean(np.abs(y_test - yhat_te)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Fit OLS for both tickers — signed impact
# ══════════════════════════════════════════════════════════════════════════════
aapl_fit = ols_univariate(
    aapl_tr["roll_spread_500"].values, aapl_tr["impact_vwap_bps"].values,
    aapl_te["roll_spread_500"].values, aapl_te["impact_vwap_bps"].values,
)
coin_fit = ols_univariate(
    coin_tr["roll_spread_500"].values, coin_tr["impact_vwap_bps"].values,
    coin_te["roll_spread_500"].values, coin_te["impact_vwap_bps"].values,
)

for name, fit in [("AAPL", aapl_fit), ("COIN", coin_fit)]:
    b = fit["beta"]
    print(f"\n{name} OLS: impact = {b[0]:+.5f} * spread {b[1]:+.5f}")
    print(f"  Train R2={fit['r2_tr']:.4f}  MAE={fit['mae_tr']:.4f}")
    print(f"  Test  R2={fit['r2_te']:+.4f}  MAE={fit['mae_te']:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Scatter + OLS line + residuals (2x2 grid)
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 14))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.28)

datasets = [
    ("AAPL", aapl_tr, aapl_te, aapl_fit, "#2563eb"),
    ("COIN", coin_tr, coin_te, coin_fit, "#dc2626"),
]

for col, (ticker, df_tr, df_te, fit, color) in enumerate(datasets):
    x_tr = df_tr["roll_spread_500"].values
    y_tr = df_tr["impact_vwap_bps"].values
    x_te = df_te["roll_spread_500"].values
    y_te = df_te["impact_vwap_bps"].values
    b = fit["beta"]

    # ── Top panel: scatter + OLS line ─────────────────────────────────────
    ax = fig.add_subplot(gs[0, col])

    # Subsample train for visibility (plot at most 4000 points)
    rng = np.random.default_rng(42)
    n_plot = min(4000, len(x_tr))
    idx_plot = rng.choice(len(x_tr), size=n_plot, replace=False)

    ax.scatter(x_tr[idx_plot], y_tr[idx_plot], s=6, alpha=0.15,
               color=color, edgecolors="none", zorder=2, label="Train")
    ax.scatter(x_te, y_te, s=10, alpha=0.25, color="#f59e0b",
               edgecolors="none", zorder=2, label="Test (Sep)")

    # OLS line
    xg = np.linspace(
        min(x_tr.min(), x_te.min()) * 0.95,
        max(x_tr.max(), x_te.max()) * 1.05,
        300,
    )
    yg = b[0] * xg + b[1]
    ax.plot(xg, yg, color="black", lw=2.2, zorder=4,
            label=f"OLS: {b[0]:+.4f}x {b[1]:+.4f}")

    # Zero line
    ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.5, zorder=1)

    # Annotation box
    box_text = (
        f"Train: R$^2$={fit['r2_tr']:.4f}, MAE={fit['mae_tr']:.3f}\n"
        f"Test:  R$^2$={fit['r2_te']:+.4f}, MAE={fit['mae_te']:.3f}\n"
        f"Train n={len(df_tr):,}  Test n={len(df_te):,}"
    )
    ax.text(0.97, 0.97, box_text, transform=ax.transAxes, fontsize=9,
            family="monospace", va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#94a3b8", alpha=0.92))

    ax.set_xlabel("roll_spread_500 (bps)", fontsize=11)
    ax.set_ylabel("impact_vwap_bps (signed, bps)", fontsize=11)
    ax.set_title(f"{ticker} — OLS fit: signed impact vs spread", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8.5, loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Cap y-axis at 1st/99th percentile for readability
    all_y = np.concatenate([y_tr, y_te])
    y_lo, y_hi = np.percentile(all_y, [1, 99])
    margin = (y_hi - y_lo) * 0.1
    ax.set_ylim(y_lo - margin, y_hi + margin)

    # ── Bottom panel: residual analysis ───────────────────────────────────
    ax_bot = fig.add_subplot(gs[1, col])

    # Residuals vs fitted (train)
    resid = fit["resid_tr"]
    fitted = fit["yhat_tr"]

    n_resid_plot = min(4000, len(resid))
    idx_r = rng.choice(len(resid), size=n_resid_plot, replace=False)

    ax_bot.scatter(fitted[idx_r], resid[idx_r], s=5, alpha=0.12,
                   color=color, edgecolors="none", zorder=2)
    ax_bot.axhline(0, color="black", lw=1.2, ls="--", alpha=0.7, zorder=3)

    # Binned mean of residuals
    n_rbins = 30
    r_order = np.argsort(fitted)
    chunks = np.array_split(r_order, n_rbins)
    bin_x = [fitted[c].mean() for c in chunks]
    bin_y = [resid[c].mean() for c in chunks]
    ax_bot.plot(bin_x, bin_y, color="#16a34a", lw=2.2, zorder=5,
                label="Binned mean residual")

    # Stats
    resid_mean = resid.mean()
    resid_std = resid.std()
    ax_bot.text(0.97, 0.97,
                f"Residual mean: {resid_mean:.4f}\n"
                f"Residual std:  {resid_std:.4f}\n"
                f"Skewness: {stats.skew(resid):.3f}\n"
                f"Kurtosis: {stats.kurtosis(resid):.3f}",
                transform=ax_bot.transAxes, fontsize=9, family="monospace",
                va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#94a3b8", alpha=0.92))

    ax_bot.set_xlabel("Fitted value (bps)", fontsize=11)
    ax_bot.set_ylabel("Residual (bps)", fontsize=11)
    ax_bot.set_title(f"{ticker} — OLS residuals vs fitted (train)", fontsize=13, fontweight="bold")
    ax_bot.legend(fontsize=9, loc="upper left")
    ax_bot.grid(True, alpha=0.18)
    ax_bot.spines["top"].set_visible(False)
    ax_bot.spines["right"].set_visible(False)

    # Cap residual y-axis
    r_cap = np.percentile(np.abs(resid), 99)
    ax_bot.set_ylim(-r_cap * 1.2, r_cap * 1.2)

fig.suptitle(
    "OLS Regression: impact_vwap_bps (signed) ~ roll_spread_500\n"
    "AAPL vs COIN — lit buy block trades",
    fontsize=15, fontweight="bold", y=1.01,
)

plt.savefig("aapl_coin_ols_fit.png", dpi=150, bbox_inches="tight")
print("\nSaved -> aapl_coin_ols_fit.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Predicted vs actual + residual histogram (2x2)
# ══════════════════════════════════════════════════════════════════════════════
fig2 = plt.figure(figsize=(18, 14))
gs2 = gridspec.GridSpec(2, 2, figure=fig2, hspace=0.35, wspace=0.28)

for col, (ticker, df_tr, df_te, fit, color) in enumerate(datasets):
    y_tr = df_tr["impact_vwap_bps"].values
    y_te = df_te["impact_vwap_bps"].values

    # ── Top panel: predicted vs actual (test set) ─────────────────────────
    ax = fig2.add_subplot(gs2[0, col])
    ax.scatter(y_te, fit["yhat_te"], s=12, alpha=0.3, color=color,
               edgecolors="none", zorder=2)

    # 45-degree line
    all_vals = np.concatenate([y_te, fit["yhat_te"]])
    lim_lo, lim_hi = np.percentile(all_vals, [1, 99])
    margin = (lim_hi - lim_lo) * 0.1
    lim_lo -= margin
    lim_hi += margin
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color="black", lw=1.5,
            ls="--", alpha=0.7, zorder=3, label="Perfect prediction")

    # Binned means for trend
    n_bins_pva = 25
    order = np.argsort(y_te)
    chunks = np.array_split(order, n_bins_pva)
    bin_actual = [y_te[c].mean() for c in chunks]
    bin_pred = [fit["yhat_te"][c].mean() for c in chunks]
    ax.plot(bin_actual, bin_pred, color="#16a34a", lw=2.5, zorder=5,
            marker="o", markersize=4, label="Binned mean")

    ax.text(0.03, 0.97,
            f"OOS R$^2$ = {fit['r2_te']:+.4f}\n"
            f"OOS MAE = {fit['mae_te']:.4f}",
            transform=ax.transAxes, fontsize=10, family="monospace",
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#94a3b8", alpha=0.92))

    ax.set_xlabel("Actual impact (signed, bps)", fontsize=11)
    ax.set_ylabel("Predicted impact (signed, bps)", fontsize=11)
    ax.set_title(f"{ticker} — Predicted vs Actual (Sep holdout)", fontsize=13, fontweight="bold")
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_aspect("equal")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Bottom panel: residual histogram (train + test overlay) ───────────
    ax_h = fig2.add_subplot(gs2[1, col])

    resid_tr = fit["resid_tr"]
    resid_te = fit["resid_te"]

    # Clip for readability
    clip = np.percentile(np.abs(resid_tr), 99.5)
    bins_h = np.linspace(-clip, clip, 80)

    ax_h.hist(resid_tr, bins=bins_h, density=True, alpha=0.5,
              color=color, edgecolor="none", label="Train residuals", zorder=2)
    ax_h.hist(resid_te, bins=bins_h, density=True, alpha=0.6,
              color="#f59e0b", edgecolor="none", label="Test residuals", zorder=3)

    # Fit normal to train residuals for reference
    mu_r, sig_r = stats.norm.fit(resid_tr)
    xg_r = np.linspace(-clip, clip, 300)
    ax_h.plot(xg_r, stats.norm.pdf(xg_r, mu_r, sig_r),
              color="black", lw=1.8, ls="--", alpha=0.7, zorder=4,
              label=f"Normal fit ($\\mu$={mu_r:.2f}, $\\sigma$={sig_r:.2f})")

    # Fit Laplace for comparison
    loc_l, b_l = stats.laplace.fit(resid_tr)
    ax_h.plot(xg_r, stats.laplace.pdf(xg_r, loc_l, b_l),
              color="#16a34a", lw=1.8, ls=":", alpha=0.8, zorder=4,
              label=f"Laplace fit (loc={loc_l:.2f}, b={b_l:.2f})")

    ax_h.axvline(0, color="gray", lw=1, ls="--", alpha=0.6)

    ax_h.text(0.97, 0.97,
              f"Train: mean={resid_tr.mean():.3f}, std={resid_tr.std():.3f}\n"
              f"Test:  mean={resid_te.mean():.3f}, std={resid_te.std():.3f}",
              transform=ax_h.transAxes, fontsize=9, family="monospace",
              va="top", ha="right",
              bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#94a3b8", alpha=0.92))

    ax_h.set_xlabel("Residual (bps)", fontsize=11)
    ax_h.set_ylabel("Density", fontsize=11)
    ax_h.set_title(f"{ticker} — Residual distribution (OLS, signed target)", fontsize=13, fontweight="bold")
    ax_h.legend(fontsize=8, loc="upper left")
    ax_h.grid(True, alpha=0.18)
    ax_h.spines["top"].set_visible(False)
    ax_h.spines["right"].set_visible(False)

fig2.suptitle(
    "OLS Diagnostics: Predicted vs Actual + Residual Distributions\n"
    "AAPL vs COIN — signed impact_vwap_bps (Sep holdout)",
    fontsize=15, fontweight="bold", y=1.01,
)

plt.savefig("aapl_coin_ols_predicted_vs_actual.png", dpi=150, bbox_inches="tight")
print("Saved -> aapl_coin_ols_predicted_vs_actual.png")
