"""
OLS baseline models consolidated.

Functions:
  run_ols_fit_visualization()    - OLS fit + residual diagnostics (signed impact, AAPL & COIN)
  run_ols_fitted_line()          - Scatter of |slippage| vs spread with OLS line
  run_ols_unsigned_by_feature()  - |slippage| scatter panels rotating over features
  run_ols_unsigned_diagnostics() - QQ plots, partial dependence, error by regime
  run_ols_unsigned_visualization() - OLS fit + residuals + predicted-vs-actual (6 features)
  run_coin_signed_ols_plot()     - COIN signed OLS scatter (tick-test filtered)
  run_coin_no_ticktest_ols()     - COIN signed OLS scatter (no tick-test filter)
  run_temporal_holdout()         - True temporal holdout: OLS/LAD/XGB/Semipar comparison
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
from scipy import stats
from statsmodels.regression.quantile_regression import QuantReg
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV


# =============================================================================
def run_ols_fit_visualization():
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


# =============================================================================
def run_ols_fitted_line():
    """
    Scatter of |slippage| vs Roll Estimated Spread with OLS line.

    Output: ols_fitted_line.png
    """
    datasets = {
        "AAPL": ("data/lit_buy_features_v2.parquet", "data/lit_buy_features_v2_sep.parquet"),
        "COIN": ("data/coin_lit_buy_features_train.parquet", "data/coin_lit_buy_features_test.parquet"),
    }

    COLORS = {"AAPL": "#2563eb", "COIN": "#dc2626"}
    rng = np.random.default_rng(42)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for ax, (ticker, (tr_f, te_f)) in zip(axes, datasets.items()):
        df_tr = pd.read_parquet(tr_f)
        df_te = pd.read_parquet(te_f)
        df_tr["abs_impact"] = df_tr["impact_vwap_bps"].abs()
        df_te["abs_impact"] = df_te["impact_vwap_bps"].abs()

        x_tr = df_tr["roll_spread_500"].values
        y_tr = df_tr["abs_impact"].values
        x_te = df_te["roll_spread_500"].values
        y_te = df_te["abs_impact"].values

        # Univariate OLS
        X_tr = np.column_stack([x_tr, np.ones(len(x_tr))])
        beta, _, _, _ = np.linalg.lstsq(X_tr, y_tr, rcond=None)

        color = COLORS[ticker]

        n_plot = min(5000, len(x_tr))
        idx = rng.choice(len(x_tr), size=n_plot, replace=False)
        ax.scatter(x_tr[idx], y_tr[idx], s=8, alpha=0.1, color=color,
                   edgecolors="none", label="Train", rasterized=True)
        ax.scatter(x_te, y_te, s=12, alpha=0.2, color="#f59e0b",
                   edgecolors="none", label="Test", rasterized=True)

        x_line = np.linspace(0, np.percentile(np.concatenate([x_tr, x_te]), 99), 300)
        y_line = beta[0] * x_line + beta[1]
        ax.plot(x_line, y_line, color="black", lw=3, zorder=5,
                label=f"OLS: {beta[0]:.2f}x + {beta[1]:.2f}")

        ax.set_xlabel("Roll Estimated Spread (bps)", fontsize=12)
        ax.set_ylabel("|slippage| (bps)", fontsize=12)
        ax.set_title(f"{ticker} OLS: |slippage| vs Roll Estimated Spread",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True, alpha=0.15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        y_cap = np.percentile(np.concatenate([y_tr, y_te]), 98)
        ax.set_ylim(0, y_cap * 1.1)
        ax.set_xlim(0, x_line[-1])

    plt.tight_layout()
    plt.savefig("ols_fitted_line.png", dpi=150, bbox_inches="tight")
    print("Saved -> ols_fitted_line.png")


# =============================================================================
def run_ols_unsigned_by_feature():
    """
    OLS |slippage| scatter plots rotating over features as x-axis.

    One figure per ticker, each with a panel per feature showing:
      - Train scatter (subsampled)
      - Test scatter
      - Binned mean trend line

    Output:
      - ols_unsigned_aapl_by_feature.png
      - ols_unsigned_coin_by_feature.png
    """
    FEATURES = [
        "dollar_value", "log_dollar_value", "participation_rate",
        "roll_spread_500", "roll_vol_500", "exchange_id",
    ]

    FEATURE_LABELS = {
        "dollar_value": "Dollar Value",
        "log_dollar_value": "Log Dollar Value",
        "participation_rate": "Participation Rate",
        "roll_spread_500": "Roll Spread 500 (bps)",
        "roll_vol_500": "Roll Volatility 500 (bps)",
        "exchange_id": "Exchange ID",
    }

    COLORS = {"AAPL": "#2563eb", "COIN": "#dc2626"}

    datasets = {
        "AAPL": ("data/lit_buy_features_v2.parquet", "data/lit_buy_features_v2_sep.parquet"),
        "COIN": ("data/coin_lit_buy_features_train.parquet", "data/coin_lit_buy_features_test.parquet"),
    }

    rng = np.random.default_rng(42)

    for ticker, (tr_f, te_f) in datasets.items():
        print(f"Processing {ticker}...", flush=True)
        df_tr = pd.read_parquet(tr_f)
        df_te = pd.read_parquet(te_f)
        df_tr["abs_impact"] = df_tr["impact_vwap_bps"].abs()
        df_te["abs_impact"] = df_te["impact_vwap_bps"].abs()
        df_tr = df_tr.sort_values("date").reset_index(drop=True)

        y_tr = df_tr["abs_impact"].values
        y_te = df_te["abs_impact"].values
        color = COLORS[ticker]

        fig, axes = plt.subplots(2, 3, figsize=(22, 13))
        axes = axes.ravel()

        for i, feat in enumerate(FEATURES):
            ax = axes[i]
            x_tr = df_tr[feat].values
            x_te = df_te[feat].values

            # Subsample train
            n_plot = min(4000, len(x_tr))
            idx = rng.choice(len(x_tr), size=n_plot, replace=False)

            ax.scatter(x_tr[idx], y_tr[idx], s=6, alpha=0.12,
                       color=color, edgecolors="none", zorder=2, label=f"Train ({len(x_tr):,})")
            ax.scatter(x_te, y_te, s=10, alpha=0.25, color="#f59e0b",
                       edgecolors="none", zorder=2, label=f"Test ({len(x_te):,})")

            # Binned mean trend (train)
            n_bins = 25
            lo_pct, hi_pct = np.percentile(x_tr, [2, 98])
            edges = np.linspace(lo_pct, hi_pct, n_bins + 1)
            bx, by = [], []
            for j in range(n_bins):
                mask = (x_tr >= edges[j]) & (x_tr < edges[j + 1])
                if j == n_bins - 1:
                    mask = (x_tr >= edges[j]) & (x_tr <= edges[j + 1])
                if mask.sum() > 10:
                    bx.append((edges[j] + edges[j + 1]) / 2)
                    by.append(y_tr[mask].mean())
            ax.plot(bx, by, color="black", lw=2.2, zorder=4, marker="o", markersize=3,
                    label="Binned mean")

            # Correlation
            valid = np.isfinite(x_tr) & np.isfinite(y_tr)
            corr = np.corrcoef(x_tr[valid], y_tr[valid])[0, 1]
            ax.text(0.97, 0.97, f"r = {corr:.3f}",
                    transform=ax.transAxes, fontsize=10, family="monospace",
                    va="top", ha="right",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#94a3b8", alpha=0.92))

            ax.set_xlabel(FEATURE_LABELS[feat], fontsize=11)
            ax.set_ylabel("|slippage| (bps)", fontsize=11)
            ax.set_title(f"|slippage| vs {FEATURE_LABELS[feat]}",
                         fontsize=11, fontweight="bold")
            ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9)
            ax.grid(True, alpha=0.18)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # Cap y-axis
            y_hi = np.percentile(np.concatenate([y_tr, y_te]), 98)
            ax.set_ylim(0, y_hi * 1.15)
            ax.set_xlim(np.percentile(x_tr, 1), np.percentile(x_tr, 99))

        fig.suptitle(f"{ticker} OLS Feature Exploration: |slippage| vs Each Feature\n"
                     f"Train: {len(df_tr):,} trades (Jun-Aug)  |  Test: {len(df_te):,} trades (Sep)",
                     fontsize=14, fontweight="bold", y=1.01)
        plt.tight_layout()
        fname = f"ols_unsigned_{ticker.lower()}_by_feature.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"Saved -> {fname}")

    print("\nDone!")


# =============================================================================
def run_ols_unsigned_diagnostics():
    """
    Additional unsigned OLS diagnostics and visualizations.

    Output:
      - ols_unsigned_qq_residuals.png       QQ plots: Normal vs Laplace fit
      - ols_unsigned_partial_dependence.png Partial dependence (marginal effects)
      - ols_unsigned_error_by_regime.png    MAE broken down by trade size regime
    """
    FEATURES = [
        "dollar_value", "log_dollar_value", "participation_rate",
        "roll_spread_500", "roll_vol_500", "exchange_id",
    ]
    FEATURE_LABELS = {
        "dollar_value": "Dollar Value",
        "log_dollar_value": "Log Dollar Value",
        "participation_rate": "Participation Rate",
        "roll_spread_500": "Roll Spread (bps)",
        "roll_vol_500": "Roll Volatility (bps)",
        "exchange_id": "Exchange ID",
    }
    COLORS = {"AAPL": "#2563eb", "COIN": "#dc2626"}

    # -- Load and fit --------------------------------------------------------------
    datasets = {
        "AAPL": ("data/lit_buy_features_v2.parquet", "data/lit_buy_features_v2_sep.parquet"),
        "COIN": ("data/coin_lit_buy_features_train.parquet", "data/coin_lit_buy_features_test.parquet"),
    }

    fitted = {}
    for ticker, (tr_f, te_f) in datasets.items():
        df_tr = pd.read_parquet(tr_f)
        df_te = pd.read_parquet(te_f)
        df_tr["abs_impact"] = df_tr["impact_vwap_bps"].abs()
        df_te["abs_impact"] = df_te["impact_vwap_bps"].abs()
        df_tr = df_tr.sort_values("date").reset_index(drop=True)

        X_tr = np.column_stack([np.ones(len(df_tr)), df_tr[FEATURES].to_numpy(dtype=np.float64)])
        X_te = np.column_stack([np.ones(len(df_te)), df_te[FEATURES].to_numpy(dtype=np.float64)])
        y_tr = df_tr["abs_impact"].to_numpy(dtype=np.float64)
        y_te = df_te["abs_impact"].to_numpy(dtype=np.float64)

        beta, _, _, _ = np.linalg.lstsq(X_tr, y_tr, rcond=None)
        yhat_te = X_te @ beta
        resid_te = y_te - yhat_te

        fitted[ticker] = {
            "df_tr": df_tr, "df_te": df_te,
            "X_tr": X_tr, "X_te": X_te,
            "y_tr": y_tr, "y_te": y_te,
            "beta": beta, "yhat_te": yhat_te, "resid_te": resid_te,
        }

    rng = np.random.default_rng(42)

    # =============================================================================
    # Figure 1: QQ plots — Normal vs Laplace (test residuals)
    # =============================================================================
    print("Plotting QQ residuals...", flush=True)

    fig1, axes1 = plt.subplots(2, 2, figsize=(16, 14))

    for col, ticker in enumerate(["AAPL", "COIN"]):
        resid = fitted[ticker]["resid_te"]
        color = COLORS[ticker]

        # Sort residuals for QQ
        resid_sorted = np.sort(resid)
        n = len(resid_sorted)
        probs = (np.arange(1, n + 1) - 0.5) / n

        # -- Normal QQ --
        ax = axes1[0, col]
        norm_quant = stats.norm.ppf(probs, loc=resid.mean(), scale=resid.std())
        ax.scatter(norm_quant, resid_sorted, s=4, alpha=0.3, color=color, edgecolors="none")
        lim = max(abs(norm_quant.min()), abs(norm_quant.max()), abs(resid_sorted.min()), abs(resid_sorted.max()))
        lim *= 0.5  # zoom in to core
        ax.plot([-lim, lim], [-lim, lim], color="black", lw=1.5, ls="--", alpha=0.7)
        ax.set_xlabel("Normal theoretical quantiles", fontsize=11)
        ax.set_ylabel("Sample quantiles (residuals)", fontsize=11)
        ax.set_title(f"{ticker} Normal QQ Plot (test residuals)", fontsize=12, fontweight="bold")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # -- Laplace QQ --
        ax2 = axes1[1, col]
        loc_l, b_l = stats.laplace.fit(resid)
        lap_quant = stats.laplace.ppf(probs, loc=loc_l, scale=b_l)
        ax2.scatter(lap_quant, resid_sorted, s=4, alpha=0.3, color=color, edgecolors="none")
        ax2.plot([-lim, lim], [-lim, lim], color="black", lw=1.5, ls="--", alpha=0.7)
        ax2.set_xlabel("Laplace theoretical quantiles", fontsize=11)
        ax2.set_ylabel("Sample quantiles (residuals)", fontsize=11)
        ax2.set_title(f"{ticker} Laplace QQ Plot (test residuals)", fontsize=12, fontweight="bold")
        ax2.set_xlim(-lim, lim)
        ax2.set_ylim(-lim, lim)
        ax2.set_aspect("equal")
        ax2.grid(True, alpha=0.15)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

    fig1.suptitle("OLS Residual QQ Plots: Normal vs Laplace\n"
                  "Laplace provides a better tail fit, motivating GAMLSS",
                  fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("ols_unsigned_qq_residuals.png", dpi=150, bbox_inches="tight")
    print("Saved -> ols_unsigned_qq_residuals.png")

    # =============================================================================
    # Figure 2: Partial dependence — marginal effect of each feature
    # =============================================================================
    print("Plotting partial dependence...", flush=True)

    fig2, axes2 = plt.subplots(2, 3, figsize=(22, 13))

    # Use AAPL and COIN side by side within each subplot
    for i, feat in enumerate(FEATURES):
        ax = axes2.ravel()[i]
        feat_idx = i + 1  # +1 because column 0 is intercept

        for ticker in ["AAPL", "COIN"]:
            f = fitted[ticker]
            X_te = f["X_te"]
            beta = f["beta"]
            color = COLORS[ticker]

            feat_vals = X_te[:, feat_idx]
            # Grid of feature values
            lo, hi = np.percentile(feat_vals, [2, 98])
            grid = np.linspace(lo, hi, 100)

            # Partial dependence: mean prediction when feature is set to grid value
            X_mean = X_te.mean(axis=0)
            pd_vals = []
            for gv in grid:
                X_temp = X_mean.copy()
                X_temp[feat_idx] = gv
                pd_vals.append(X_temp @ beta)
            pd_vals = np.array(pd_vals)

            ax.plot(grid, pd_vals, color=color, lw=2.5, label=ticker)

            # Rug plot (subsampled)
            n_rug = min(500, len(feat_vals))
            rug_idx = rng.choice(len(feat_vals), size=n_rug, replace=False)
            ax.scatter(feat_vals[rug_idx], np.full(n_rug, ax.get_ylim()[0] if i > 0 else 0),
                       s=3, alpha=0.15, color=color, marker="|")

        ax.set_xlabel(FEATURE_LABELS[feat], fontsize=11)
        ax.set_ylabel("Predicted |slippage| (bps)", fontsize=11)
        ax.set_title(f"Partial Dependence: {FEATURE_LABELS[feat]}",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig2.suptitle("OLS Partial Dependence Plots: AAPL and COIN\n"
                  "Marginal effect of each feature on predicted |slippage|",
                  fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("ols_unsigned_partial_dependence.png", dpi=150, bbox_inches="tight")
    print("Saved -> ols_unsigned_partial_dependence.png")

    # =============================================================================
    # Figure 3: Error by trade size regime + temporal MAE
    # =============================================================================
    print("Plotting error by regime...", flush=True)

    fig3, axes3 = plt.subplots(2, 2, figsize=(18, 14))

    for col, ticker in enumerate(["AAPL", "COIN"]):
        f = fitted[ticker]
        y_te = f["y_te"]
        yhat_te = f["yhat_te"]
        ae_te = np.abs(y_te - yhat_te)
        df_te = f["df_te"]
        color = COLORS[ticker]

        # -- Top: MAE by trade size tercile ----------------------------------------
        ax = axes3[0, col]
        dv = df_te["dollar_value"].values
        terciles = pd.qcut(dv, q=3, labels=["Small", "Medium", "Large"])
        regime_mae = []
        regime_labels = ["Small", "Medium", "Large"]
        regime_counts = []
        for label in regime_labels:
            mask = terciles == label
            regime_mae.append(ae_te[mask].mean())
            regime_counts.append(mask.sum())

        bars = ax.bar(regime_labels, regime_mae, color=[color] * 3, alpha=0.8,
                      edgecolor="white", linewidth=1.5)
        for bar, mae_val, cnt in zip(bars, regime_mae, regime_counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    f"MAE={mae_val:.2f}\nn={cnt:,}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

        ax.set_ylabel("MAE (bps)", fontsize=11)
        ax.set_title(f"{ticker} MAE by Trade Size Tercile",
                     fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.15, axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # -- Bottom: MAE by volatility regime --------------------------------------
        ax2 = axes3[1, col]
        vol = df_te["roll_vol_500"].values
        vol_terciles = pd.qcut(vol, q=3, labels=["Low Vol", "Med Vol", "High Vol"])
        vol_labels = ["Low Vol", "Med Vol", "High Vol"]
        vol_mae = []
        vol_counts = []
        vol_med_slip = []
        for label in vol_labels:
            mask = vol_terciles == label
            vol_mae.append(ae_te[mask].mean())
            vol_counts.append(mask.sum())
            vol_med_slip.append(np.median(y_te[mask]))

        bars2 = ax2.bar(vol_labels, vol_mae, color=[color] * 3, alpha=0.8,
                        edgecolor="white", linewidth=1.5)
        for bar, mae_val, cnt, med in zip(bars2, vol_mae, vol_counts, vol_med_slip):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                     f"MAE={mae_val:.2f}\nn={cnt:,}\nmed slip={med:.1f}",
                     ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax2.set_ylabel("MAE (bps)", fontsize=11)
        ax2.set_title(f"{ticker} MAE by Volatility Regime",
                      fontsize=12, fontweight="bold")
        ax2.grid(True, alpha=0.15, axis="y")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

    fig3.suptitle("OLS Error Decomposition by Regime\n"
                  "Where does the linear model struggle?",
                  fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("ols_unsigned_error_by_regime.png", dpi=150, bbox_inches="tight")
    print("Saved -> ols_unsigned_error_by_regime.png")

    print("\nDone!")


# =============================================================================
def run_ols_unsigned_visualization():
    """
    OLS regression visualizations for unsigned slippage: abs(impact_vwap_bps).

    Uses all 6 features. Temporal holdout: Jun-Aug train, Sep test.

    Output:
      - ols_unsigned_fit.png              (2x2: scatter + residuals per ticker)
      - ols_unsigned_predicted_vs_actual.png  (2x2: pred vs actual + residual hist)
    """
    FEATURES = [
        "dollar_value", "log_dollar_value", "participation_rate",
        "roll_spread_500", "roll_vol_500", "exchange_id",
    ]

    # -- Load data ----------------------------------------------------------------
    datasets = {
        "AAPL": ("data/lit_buy_features_v2.parquet", "data/lit_buy_features_v2_sep.parquet"),
        "COIN": ("data/coin_lit_buy_features_train.parquet", "data/coin_lit_buy_features_test.parquet"),
    }

    tickers = []
    for ticker, (tr_f, te_f) in datasets.items():
        df_tr = pd.read_parquet(tr_f)
        df_te = pd.read_parquet(te_f)
        df_tr["abs_impact"] = df_tr["impact_vwap_bps"].abs()
        df_te["abs_impact"] = df_te["impact_vwap_bps"].abs()
        df_tr = df_tr.sort_values("date").reset_index(drop=True)

        X_tr = np.column_stack([np.ones(len(df_tr)), df_tr[FEATURES].to_numpy(dtype=np.float64)])
        X_te = np.column_stack([np.ones(len(df_te)), df_te[FEATURES].to_numpy(dtype=np.float64)])
        y_tr = df_tr["abs_impact"].to_numpy(dtype=np.float64)
        y_te = df_te["abs_impact"].to_numpy(dtype=np.float64)

        beta, _, _, _ = np.linalg.lstsq(X_tr, y_tr, rcond=None)
        yhat_tr = X_tr @ beta
        yhat_te = X_te @ beta

        def r2(y, yhat):
            ss_res = ((y - yhat) ** 2).sum()
            ss_tot = ((y - y.mean()) ** 2).sum()
            return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        tickers.append({
            "name": ticker,
            "df_tr": df_tr, "df_te": df_te,
            "X_tr": X_tr, "X_te": X_te,
            "y_tr": y_tr, "y_te": y_te,
            "beta": beta,
            "yhat_tr": yhat_tr, "yhat_te": yhat_te,
            "resid_tr": y_tr - yhat_tr, "resid_te": y_te - yhat_te,
            "r2_tr": r2(y_tr, yhat_tr), "r2_te": r2(y_te, yhat_te),
            "mae_tr": np.mean(np.abs(y_tr - yhat_tr)),
            "mae_te": np.mean(np.abs(y_te - yhat_te)),
        })
        print(f"{ticker}: Train R2={r2(y_tr, yhat_tr):.4f} MAE={np.mean(np.abs(y_tr - yhat_tr)):.4f} | "
              f"Test R2={r2(y_te, yhat_te):.4f} MAE={np.mean(np.abs(y_te - yhat_te)):.4f}")

    COLORS = {"AAPL": "#2563eb", "COIN": "#dc2626"}
    rng = np.random.default_rng(42)

    # =============================================================================
    # Figure 1: Scatter (slippage vs roll_spread) + Residuals vs Fitted (2x2)
    # =============================================================================
    fig1 = plt.figure(figsize=(18, 14))
    gs1 = gridspec.GridSpec(2, 2, figure=fig1, hspace=0.35, wspace=0.28)

    for col, t in enumerate(tickers):
        ticker = t["name"]
        color = COLORS[ticker]
        spread_tr = t["df_tr"]["roll_spread_500"].values
        spread_te = t["df_te"]["roll_spread_500"].values

        # -- Top: scatter + OLS trend line (slippage vs spread) --------------------
        ax = fig1.add_subplot(gs1[0, col])

        n_plot = min(4000, len(spread_tr))
        idx = rng.choice(len(spread_tr), size=n_plot, replace=False)

        ax.scatter(spread_tr[idx], t["y_tr"][idx], s=6, alpha=0.15,
                   color=color, edgecolors="none", zorder=2, label=f"Train ({len(t['y_tr']):,})")
        ax.scatter(spread_te, t["y_te"], s=10, alpha=0.25, color="#f59e0b",
                   edgecolors="none", zorder=2, label=f"Test ({len(t['y_te']):,})")

        # Binned mean trend
        all_spread = np.concatenate([spread_tr, spread_te])
        all_y = np.concatenate([t["y_tr"], t["y_te"]])
        n_trend = 30
        edges = np.linspace(np.percentile(all_spread, 2), np.percentile(all_spread, 98), n_trend + 1)
        bx, by = [], []
        for i in range(n_trend):
            mask = (spread_tr >= edges[i]) & (spread_tr < edges[i + 1])
            if mask.sum() > 10:
                bx.append((edges[i] + edges[i + 1]) / 2)
                by.append(t["y_tr"][mask].mean())
        ax.plot(bx, by, color="black", lw=2.2, zorder=4, marker="o", markersize=3,
                label="Binned mean (train)")

        box_text = (
            f"Train: R$^2$={t['r2_tr']:.4f}, MAE={t['mae_tr']:.3f}\n"
            f"Test:  R$^2$={t['r2_te']:.4f}, MAE={t['mae_te']:.3f}"
        )
        ax.text(0.97, 0.97, box_text, transform=ax.transAxes, fontsize=9,
                family="monospace", va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#94a3b8", alpha=0.92))

        ax.set_xlabel("roll_spread_500 (bps)", fontsize=11)
        ax.set_ylabel("|slippage| (bps)", fontsize=11)
        ax.set_title(f"{ticker} OLS: |slippage| vs Spread (6 features)",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=8.5, loc="upper left", framealpha=0.9)
        ax.grid(True, alpha=0.18)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        y_lo, y_hi = np.percentile(all_y, [0, 98])
        ax.set_ylim(max(y_lo - 0.5, 0), y_hi * 1.1)

        # -- Bottom: residuals vs fitted (train) -----------------------------------
        ax_bot = fig1.add_subplot(gs1[1, col])

        resid = t["resid_tr"]
        fitted = t["yhat_tr"]
        n_r = min(4000, len(resid))
        idx_r = rng.choice(len(resid), size=n_r, replace=False)

        ax_bot.scatter(fitted[idx_r], resid[idx_r], s=5, alpha=0.12,
                       color=color, edgecolors="none", zorder=2)
        ax_bot.axhline(0, color="black", lw=1.2, ls="--", alpha=0.7, zorder=3)

        n_rbins = 30
        r_order = np.argsort(fitted)
        chunks = np.array_split(r_order, n_rbins)
        bin_x = [fitted[c].mean() for c in chunks]
        bin_y = [resid[c].mean() for c in chunks]
        ax_bot.plot(bin_x, bin_y, color="#16a34a", lw=2.2, zorder=5,
                    label="Binned mean residual")

        ax_bot.text(0.97, 0.97,
                    f"Residual mean: {resid.mean():.4f}\n"
                    f"Residual std:  {resid.std():.4f}\n"
                    f"Skewness: {stats.skew(resid):.3f}\n"
                    f"Kurtosis: {stats.kurtosis(resid):.3f}",
                    transform=ax_bot.transAxes, fontsize=9, family="monospace",
                    va="top", ha="right",
                    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#94a3b8", alpha=0.92))

        ax_bot.set_xlabel("Fitted value (bps)", fontsize=11)
        ax_bot.set_ylabel("Residual (bps)", fontsize=11)
        ax_bot.set_title(f"{ticker} OLS Residuals vs Fitted (train)",
                         fontsize=13, fontweight="bold")
        ax_bot.legend(fontsize=9, loc="upper left")
        ax_bot.grid(True, alpha=0.18)
        ax_bot.spines["top"].set_visible(False)
        ax_bot.spines["right"].set_visible(False)
        r_cap = np.percentile(np.abs(resid), 99)
        ax_bot.set_ylim(-r_cap * 1.2, r_cap * 1.2)

    fig1.suptitle(
        "OLS Regression on |slippage|: AAPL and COIN\n"
        "6 features, temporal holdout (Jun-Aug train, Sep test)",
        fontsize=15, fontweight="bold", y=1.01,
    )
    plt.savefig("ols_unsigned_fit.png", dpi=150, bbox_inches="tight")
    print("Saved -> ols_unsigned_fit.png")

    # =============================================================================
    # Figure 2: Predicted vs Actual + Residual Histograms (2x2)
    # =============================================================================
    fig2 = plt.figure(figsize=(18, 14))
    gs2 = gridspec.GridSpec(2, 2, figure=fig2, hspace=0.35, wspace=0.28)

    for col, t in enumerate(tickers):
        ticker = t["name"]
        color = COLORS[ticker]
        y_te = t["y_te"]
        yhat_te = t["yhat_te"]

        # -- Top: predicted vs actual (test) ---------------------------------------
        ax = fig2.add_subplot(gs2[0, col])
        ax.scatter(y_te, yhat_te, s=12, alpha=0.3, color=color,
                   edgecolors="none", zorder=2)

        all_vals = np.concatenate([y_te, yhat_te])
        lim_lo = 0
        lim_hi = np.percentile(all_vals, 99) * 1.1
        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color="black", lw=1.5,
                ls="--", alpha=0.7, zorder=3, label="Perfect prediction")

        n_bins_pva = 25
        order = np.argsort(y_te)
        chunks = np.array_split(order, n_bins_pva)
        bin_actual = [y_te[c].mean() for c in chunks]
        bin_pred = [yhat_te[c].mean() for c in chunks]
        ax.plot(bin_actual, bin_pred, color="#16a34a", lw=2.5, zorder=5,
                marker="o", markersize=4, label="Binned mean")

        ax.text(0.03, 0.97,
                f"OOS R$^2$ = {t['r2_te']:.4f}\n"
                f"OOS MAE = {t['mae_te']:.4f}",
                transform=ax.transAxes, fontsize=10, family="monospace",
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#94a3b8", alpha=0.92))

        ax.set_xlabel("Actual |slippage| (bps)", fontsize=11)
        ax.set_ylabel("Predicted |slippage| (bps)", fontsize=11)
        ax.set_title(f"{ticker} Predicted vs Actual (Sep holdout)",
                     fontsize=13, fontweight="bold")
        ax.set_xlim(lim_lo, lim_hi)
        ax.set_ylim(lim_lo, lim_hi)
        ax.set_aspect("equal")
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, alpha=0.18)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # -- Bottom: residual histogram (train + test overlay) ---------------------
        ax_h = fig2.add_subplot(gs2[1, col])
        resid_tr = t["resid_tr"]
        resid_te = t["resid_te"]

        clip = np.percentile(np.abs(resid_tr), 99.5)
        bins_h = np.linspace(-clip, clip, 80)

        ax_h.hist(resid_tr, bins=bins_h, density=True, alpha=0.5,
                  color=color, edgecolor="none", label="Train residuals", zorder=2)
        ax_h.hist(resid_te, bins=bins_h, density=True, alpha=0.6,
                  color="#f59e0b", edgecolor="none", label="Test residuals", zorder=3)

        loc_l, b_l = stats.laplace.fit(resid_tr)
        xg_r = np.linspace(-clip, clip, 300)
        ax_h.plot(xg_r, stats.laplace.pdf(xg_r, loc_l, b_l),
                  color="#16a34a", lw=1.8, ls=":", alpha=0.8, zorder=4,
                  label=f"Laplace fit (loc={loc_l:.2f}, b={b_l:.2f})")

        mu_r, sig_r = stats.norm.fit(resid_tr)
        ax_h.plot(xg_r, stats.norm.pdf(xg_r, mu_r, sig_r),
                  color="black", lw=1.8, ls="--", alpha=0.7, zorder=4,
                  label=f"Normal fit (mu={mu_r:.2f}, sig={sig_r:.2f})")

        ax_h.axvline(0, color="gray", lw=1, ls="--", alpha=0.6)

        ax_h.text(0.97, 0.97,
                  f"Train: mean={resid_tr.mean():.3f}, std={resid_tr.std():.3f}\n"
                  f"Test:  mean={resid_te.mean():.3f}, std={resid_te.std():.3f}",
                  transform=ax_h.transAxes, fontsize=9, family="monospace",
                  va="top", ha="right",
                  bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#94a3b8", alpha=0.92))

        ax_h.set_xlabel("Residual (bps)", fontsize=11)
        ax_h.set_ylabel("Density", fontsize=11)
        ax_h.set_title(f"{ticker} Residual Distribution (OLS, unsigned target)",
                       fontsize=13, fontweight="bold")
        ax_h.legend(fontsize=8, loc="upper left")
        ax_h.grid(True, alpha=0.18)
        ax_h.spines["top"].set_visible(False)
        ax_h.spines["right"].set_visible(False)

    fig2.suptitle(
        "OLS Diagnostics on |slippage|: Predicted vs Actual + Residuals\n"
        "AAPL and COIN, 6 features, Sep holdout",
        fontsize=15, fontweight="bold", y=1.01,
    )
    plt.savefig("ols_unsigned_predicted_vs_actual.png", dpi=150, bbox_inches="tight")
    print("Saved -> ols_unsigned_predicted_vs_actual.png")

    print("\nDone!")


# =============================================================================
def run_coin_signed_ols_plot():
    """COIN signed OLS scatter (tick-test filtered). Output: coin_signed_ols.png"""
    # Load data
    coin_tr = pd.read_parquet("data/coin_lit_buy_features_train.parquet")
    coin_te = pd.read_parquet("data/coin_lit_buy_features_test.parquet")

    x_tr = coin_tr["roll_spread_500"].values.astype(np.float64)
    y_tr = coin_tr["impact_vwap_bps"].values.astype(np.float64)
    x_te = coin_te["roll_spread_500"].values.astype(np.float64)
    y_te = coin_te["impact_vwap_bps"].values.astype(np.float64)

    # OLS fit
    X_tr = np.column_stack([x_tr, np.ones(len(x_tr))])
    beta, *_ = np.linalg.lstsq(X_tr, y_tr, rcond=None)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6.5))

    rng = np.random.default_rng(42)
    mask_tr = x_tr <= 20
    mask_te = x_te <= 20
    idx_tr = np.where(mask_tr)[0]
    n_plot = min(4000, len(idx_tr))
    idx_plot = rng.choice(idx_tr, size=n_plot, replace=False)

    ax.scatter(x_tr[idx_plot], y_tr[idx_plot], s=10, alpha=0.4,
               color="#dc2626", edgecolors="none", zorder=2, label="Train")
    ax.scatter(x_te[mask_te], y_te[mask_te], s=14, alpha=0.55,
               color="#d97706", edgecolors="none", zorder=3, label="Test (Sep)")

    xg = np.linspace(0, 20, 300)
    yg = beta[0] * xg + beta[1]
    ax.plot(xg, yg, color="black", lw=2.2, zorder=4,
            label=f"OLS: {beta[0]:+.4f}x {beta[1]:+.4f}")

    ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.5, zorder=1)

    ax.set_xlim(0, 20)
    ax.set_ylim(-25, 30)

    ax.set_xlabel("Roll Spread Estimate (bps)", fontsize=12)
    ax.set_ylabel("Slippage (bps)", fontsize=12)
    ax.set_title("COIN OLS Fit", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig("coin_signed_ols.png", dpi=150, bbox_inches="tight")
    print("Saved -> coin_signed_ols.png")


# =============================================================================
def run_coin_no_ticktest_ols():
    """
    Same as coin_signed_ols_plot.py but WITHOUT tick-test filtering.
    Uses ALL COIN lit block trades (buy + sell + unknown), raw impact_vwap_bps.
    Output: coin_no_ticktest_ols.png
    """
    # ── Config (same as build_coin_features.py) ──────────────────────────────────
    DARK_IDS   = {4, 6, 16, 62, 201, 202, 203}
    COIN_MID   = 222.0
    WINDOW     = 500
    MIN_TICKS  = 10

    def roll_spread_500(px_window):
        dp = np.diff(px_window.astype(np.float64))
        if len(dp) < 2:
            return np.nan
        cov_mat = np.cov(dp[1:], dp[:-1], ddof=1)
        cov1 = cov_mat[0, 1]
        if cov1 < 0:
            return 2.0 * np.sqrt(-cov1)
        return np.nan

    # ── 1. Load ALL COIN lit block trades (no tick-test filter) ──────────────────
    bt = pd.read_parquet(
        "data/block_trades.parquet",
        columns=["ticker", "date", "timestamp_ns", "price", "size",
                 "dollar_value", "exchange", "impact_vwap_bps"],
        filters=[("ticker", "==", "COIN")],
    )

    # Lit only, must have impact
    trades = bt[
        (~bt["exchange"].isin(DARK_IDS)) &
        bt["impact_vwap_bps"].notna()
    ].copy()
    trades = trades.sort_values(["date", "timestamp_ns"]).reset_index(drop=True)
    print(f"COIN lit block trades (all sides): {len(trades):,}")

    # ── 2. Compute roll_spread_500 for each trade ───────────────────────────────
    roll_spread_arr = np.full(len(trades), np.nan)

    days_processed = 0
    for date, grp in trades.groupby("date"):
        tick_path = f"data/COIN/{date}.parquet"
        if not os.path.exists(tick_path):
            continue

        ticks = pd.read_parquet(tick_path, columns=["sip_timestamp", "price"])
        ticks = ticks.sort_values("sip_timestamp").reset_index(drop=True)

        ts = ticks["sip_timestamp"].to_numpy(dtype=np.int64)
        px = ticks["price"].to_numpy(dtype=np.float64)

        block_ts = grp["timestamp_ns"].to_numpy(dtype=np.int64)
        hi_idx = np.searchsorted(ts, block_ts, side="left")

        for i, (idx, hi) in enumerate(zip(grp.index, hi_idx)):
            lo = max(0, hi - WINDOW)
            if hi - lo < MIN_TICKS:
                continue
            spread_raw = roll_spread_500(px[lo:hi])
            if not np.isnan(spread_raw):
                roll_spread_arr[idx] = spread_raw / COIN_MID * 1e4

        days_processed += 1
        if days_processed % 50 == 0:
            print(f"  Processed {days_processed} days...")

    print(f"  Processed {days_processed} days total")

    trades["roll_spread_500"] = roll_spread_arr

    # ── 3. Drop NaN, split train/test ───────────────────────────────────────────
    feat = trades.dropna(subset=["roll_spread_500", "impact_vwap_bps"]).copy()
    feat = feat.reset_index(drop=True)
    print(f"Rows after dropping NaN: {len(feat):,}")

    train = feat[feat["date"] < "2024-09-01"].reset_index(drop=True)
    test  = feat[feat["date"] >= "2024-09-01"].reset_index(drop=True)
    print(f"Train: {len(train):,}  Test: {len(test):,}")

    # ── 4. OLS fit ───────────────────────────────────────────────────────────────
    x_tr = train["roll_spread_500"].values.astype(np.float64)
    y_tr = train["impact_vwap_bps"].values.astype(np.float64)
    x_te = test["roll_spread_500"].values.astype(np.float64)
    y_te = test["impact_vwap_bps"].values.astype(np.float64)

    X_tr = np.column_stack([x_tr, np.ones(len(x_tr))])
    beta, *_ = np.linalg.lstsq(X_tr, y_tr, rcond=None)
    print(f"OLS: impact = {beta[0]:+.4f} * spread {beta[1]:+.4f}")

    # ── 5. Plot (same style/zoom as coin_signed_ols_plot.py) ─────────────────────
    fig, ax = plt.subplots(figsize=(10, 6.5))

    rng = np.random.default_rng(42)
    mask_tr = x_tr <= 20
    mask_te = x_te <= 20
    idx_tr = np.where(mask_tr)[0]
    n_plot = min(4000, len(idx_tr))
    idx_plot = rng.choice(idx_tr, size=n_plot, replace=False)

    ax.scatter(x_tr[idx_plot], y_tr[idx_plot], s=10, alpha=0.4,
               color="#dc2626", edgecolors="none", zorder=2, label="Train")
    ax.scatter(x_te[mask_te], y_te[mask_te], s=14, alpha=0.55,
               color="#d97706", edgecolors="none", zorder=3, label="Test (Sep)")

    xg = np.linspace(0, 20, 300)
    yg = beta[0] * xg + beta[1]
    ax.plot(xg, yg, color="black", lw=2.2, zorder=4,
            label=f"OLS: {beta[0]:+.4f}x {beta[1]:+.4f}")

    ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.5, zorder=1)

    ax.set_xlim(0, 20)
    ax.set_ylim(-25, 30)

    ax.set_xlabel("Roll Spread Estimate (bps)", fontsize=12)
    ax.set_ylabel("Slippage (bps)", fontsize=12)
    ax.set_title("Signed COIN OLS Fit", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig("coin_no_ticktest_ols.png", dpi=150, bbox_inches="tight")
    print("Saved -> coin_no_ticktest_ols.png")


# =============================================================================
def run_temporal_holdout():
    """
    True temporal holdout evaluation.

    Train: June–August 2024  (data/lit_buy_features_v2.parquet,  35,020 trades, 63 days)
    Test : September 2024    (data/lit_buy_features_v2_sep.parquet, 9,152 trades, 20 days)

    Models evaluated
    ----------------
      (1) OLS-Uni      : OLS of abs(impact) ~ spread + 1
      (2) LAD-Uni      : LAD of abs(impact) ~ spread + 1  (QuantReg tau=0.5)
      (3) XGB-MSE      : XGBoost (reg:squarederror)  ~ [spread, vol, prate]
      (4) XGB-MAE      : XGBoost (reg:absoluteerror) ~ [spread, vol, prate]
      (5) Semipar-MAE  : LAD(spread) + XGBoost-MAE(residuals ~ vol, prate)

    All XGBoost models use 3-fold inner CV on the training data to tune:
      max_depth=[2,3], n_estimators=[50,100,200], learning_rate=[0.05,0.1],
      min_child_weight=[3,5].

    OOS metrics (individual September trades): R² and MAE.

    Output: aapl_temporal_holdout.png
    """
    # ── Load training and test sets ────────────────────────────────────────────────
    df_tr = pd.read_parquet("data/lit_buy_features_v2.parquet")
    df_te = pd.read_parquet("data/lit_buy_features_v2_sep.parquet")

    df_tr["abs_impact"] = df_tr["impact_vwap_bps"].abs()
    df_te["abs_impact"] = df_te["impact_vwap_bps"].abs()

    print(f"Train: {len(df_tr):,} trades  |  {df_tr['date'].nunique()} days  "
          f"({df_tr['date'].min()} .. {df_tr['date'].max()})")
    print(f"Test : {len(df_te):,} trades  |  {df_te['date'].nunique()} days  "
          f"({df_te['date'].min()} .. {df_te['date'].max()})")

    # ── Feature arrays ─────────────────────────────────────────────────────────────
    def make_arrays(df):
        spread = df["roll_spread_500"].to_numpy(dtype=np.float64)
        vol    = df["roll_vol_500"].to_numpy(dtype=np.float64)
        prate  = df["participation_rate"].to_numpy(dtype=np.float64)
        y      = df["abs_impact"].to_numpy(dtype=np.float64)
        X_lad  = np.column_stack([spread, np.ones(len(df))])          # [spread, 1]
        X_full = np.column_stack([spread, vol, prate])                 # all 3
        X_res  = np.column_stack([vol, prate])                         # vol & prate only
        return spread, vol, prate, y, X_lad, X_full, X_res

    sp_tr, vo_tr, pr_tr, y_tr, Xlad_tr, Xfull_tr, Xres_tr = make_arrays(df_tr)
    sp_te, vo_te, pr_te, y_te, Xlad_te, Xfull_te, Xres_te = make_arrays(df_te)

    # ── Metric helpers ─────────────────────────────────────────────────────────────
    def r2(ytrue, ypred):
        ss_res = ((ytrue - ypred) ** 2).sum()
        ss_tot = ((ytrue - ytrue.mean()) ** 2).sum()
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    def mae(ytrue, ypred):
        return np.mean(np.abs(ytrue - ypred))

    # ── XGBoost grid ──────────────────────────────────────────────────────────────
    PARAM_GRID = {
        "max_depth":        [2, 3],
        "n_estimators":     [50, 100, 200],
        "learning_rate":    [0.05, 0.1],
        "min_child_weight": [3, 5],
    }

    def fit_xgb_gs(X_tr, y_tr, objective, inner_cv=3):
        base = XGBRegressor(
            objective=objective,
            tree_method="hist", verbosity=0,
            random_state=42, n_jobs=1,
        )
        scoring = ("neg_mean_absolute_error" if "absolute" in objective
                   else "neg_mean_squared_error")
        gs = GridSearchCV(base, PARAM_GRID, cv=inner_cv,
                          scoring=scoring, refit=True, n_jobs=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gs.fit(X_tr, y_tr)
        return gs.best_estimator_, gs.best_params_

    # ── (1) OLS-Uni ───────────────────────────────────────────────────────────────
    print("\n--- Fitting OLS-Uni ---")
    beta_ols, *_ = np.linalg.lstsq(Xlad_tr, y_tr, rcond=None)
    pred_ols     = np.maximum(Xlad_te @ beta_ols, 0.0)
    r2_ols, mae_ols = r2(y_te, pred_ols), mae(y_te, pred_ols)
    print(f"  c1={beta_ols[0]:+.5f}  c2={beta_ols[1]:+.5f}  "
          f"OOS R2={r2_ols:+.4f}  MAE={mae_ols:.4f}")

    # ── (2) LAD-Uni ───────────────────────────────────────────────────────────────
    print("--- Fitting LAD-Uni ---")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lad_res  = QuantReg(y_tr, Xlad_tr).fit(q=0.5, max_iter=2000, p_tol=1e-6)
    beta_lad     = lad_res.params
    lad_pred_tr  = Xlad_tr @ beta_lad
    pred_lad     = np.maximum(Xlad_te @ beta_lad, 0.0)
    r2_lad, mae_lad = r2(y_te, pred_lad), mae(y_te, pred_lad)
    print(f"  c1={beta_lad[0]:+.5f}  c2={beta_lad[1]:+.5f}  "
          f"OOS R2={r2_lad:+.4f}  MAE={mae_lad:.4f}")

    # ── (3) XGB-MSE ───────────────────────────────────────────────────────────────
    print("--- Fitting XGB-MSE (GridSearchCV 3-fold) ---")
    xgb_mse, params_mse = fit_xgb_gs(Xfull_tr, y_tr, "reg:squarederror")
    pred_mse = np.maximum(xgb_mse.predict(Xfull_te), 0.0)
    r2_mse, mae_mse = r2(y_te, pred_mse), mae(y_te, pred_mse)
    print(f"  best={params_mse}  OOS R2={r2_mse:+.4f}  MAE={mae_mse:.4f}")

    # ── (4) XGB-MAE ───────────────────────────────────────────────────────────────
    print("--- Fitting XGB-MAE (GridSearchCV 3-fold) ---")
    xgb_mae, params_mae = fit_xgb_gs(Xfull_tr, y_tr, "reg:absoluteerror")
    pred_mae = np.maximum(xgb_mae.predict(Xfull_te), 0.0)
    r2_mae, mae_mae = r2(y_te, pred_mae), mae(y_te, pred_mae)
    print(f"  best={params_mae}  OOS R2={r2_mae:+.4f}  MAE={mae_mae:.4f}")

    # ── (5) Semipar-MAE ──────────────────────────────────────────────────────────
    print("--- Fitting Semipar-MAE ---")
    resid_tr = y_tr - lad_pred_tr
    xgb_res, params_res = fit_xgb_gs(Xres_tr, resid_tr, "reg:absoluteerror")
    xgb_res_pred_te = xgb_res.predict(Xres_te)
    pred_semi = np.maximum(pred_lad + xgb_res_pred_te, 0.0)
    r2_semi, mae_semi = r2(y_te, pred_semi), mae(y_te, pred_semi)
    print(f"  best={params_res}  OOS R2={r2_semi:+.4f}  MAE={mae_semi:.4f}")

    # ── Summary ───────────────────────────────────────────────────────────────────
    MODEL_NAMES  = ["OLS-Uni", "LAD-Uni", "XGB-MSE", "XGB-MAE", "Semipar-MAE"]
    OOS_R2       = [r2_ols, r2_lad, r2_mse, r2_mae, r2_semi]
    OOS_MAE      = [mae_ols, mae_lad, mae_mse, mae_mae, mae_semi]
    ALL_PREDS    = [pred_ols, pred_lad, pred_mse, pred_mae, pred_semi]

    print(f"\n{'='*64}")
    print(f"  TEMPORAL HOLDOUT: Train Jun-Aug 2024  ->  Test Sep 2024")
    print(f"{'='*64}")
    print(f"  {'Model':<22}  {'OOS R2':>9}  {'OOS MAE (bps)':>14}")
    print(f"  {'-'*50}")
    for name, rv, mv in zip(MODEL_NAMES, OOS_R2, OOS_MAE):
        print(f"  {name:<22}  {rv:>+9.4f}  {mv:>14.4f}")
    print(f"{'='*64}")

    # Reference: test set naive mean predictor
    naive_mae  = mae(y_te, np.full(len(y_te), y_te.mean()))
    naive_r2   = 0.0
    print(f"\n  Naive mean baseline (predict train mean={y_tr.mean():.3f}): "
          f"MAE={mae(y_te, np.full(len(y_te), y_tr.mean())):.4f}")
    print(f"  Test abs_impact: mean={y_te.mean():.4f}  std={y_te.std():.4f}  "
          f"median={np.median(y_te):.4f}")

    # ── Plot ───────────────────────────────────────────────────────────────────────
    COLORS = ["#2563eb", "#16a34a", "#7c3aed", "#dc2626", "#f59e0b"]

    fig = plt.figure(figsize=(18, 10))
    gs_fig = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.36)
    ax1 = fig.add_subplot(gs_fig[0, 0])   # OOS R² bar
    ax2 = fig.add_subplot(gs_fig[0, 1])   # OOS MAE bar
    ax3 = fig.add_subplot(gs_fig[0, 2])   # feature distribution train vs test
    ax4 = fig.add_subplot(gs_fig[1, 0])   # predictions vs actual scatter (best model)
    ax5 = fig.add_subplot(gs_fig[1, 1])   # absolute error distribution violin
    ax6 = fig.add_subplot(gs_fig[1, 2])   # MAE by September date

    xpos = np.arange(5)

    # ── Panel 1: OOS R² bars ──────────────────────────────────────────────────────
    bars1 = ax1.bar(xpos, OOS_R2, color=COLORS, width=0.55,
                    edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars1, OOS_R2):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 v + (0.003 if v >= 0 else -0.006),
                 f"{v:+.4f}", ha="center", va="bottom" if v >= 0 else "top",
                 fontsize=8.5, fontweight="bold")
    ax1.axhline(0, color="black", lw=0.9, ls="--", alpha=0.5)
    ax1.set_xticks(xpos); ax1.set_xticklabels(MODEL_NAMES, fontsize=8.5, rotation=10)
    ax1.set_ylabel("OOS R²  (Sep 2024)", fontsize=10)
    ax1.set_title("OOS R²\n(Sep 2024 holdout)", fontsize=11, fontweight="bold")
    ax1.grid(True, axis="y", alpha=0.2)
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

    # ── Panel 2: OOS MAE bars ─────────────────────────────────────────────────────
    bars2 = ax2.bar(xpos, OOS_MAE, color=COLORS, width=0.55,
                    edgecolor="white", linewidth=0.8)
    for bar, v in zip(bars2, OOS_MAE):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                 f"{v:.4f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax2.set_xticks(xpos); ax2.set_xticklabels(MODEL_NAMES, fontsize=8.5, rotation=10)
    ax2.set_ylabel("OOS MAE  (bps)", fontsize=10)
    ax2.set_title("OOS MAE\n(Sep 2024 holdout)", fontsize=11, fontweight="bold")
    ax2.grid(True, axis="y", alpha=0.2)
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

    # ── Panel 3: spread distribution train vs test ────────────────────────────────
    clip_sp = np.percentile(np.concatenate([sp_tr, sp_te]), 99)
    ax3.hist(np.clip(sp_tr, 0, clip_sp), bins=60, density=True,
             color="#2563eb", alpha=0.5, label=f"Train Jun-Aug (n={len(sp_tr):,})")
    ax3.hist(np.clip(sp_te, 0, clip_sp), bins=60, density=True,
             color="#dc2626", alpha=0.5, label=f"Test Sep (n={len(sp_te):,})")
    ax3.set_xlabel("roll_spread_500 (bps)", fontsize=10)
    ax3.set_ylabel("Density", fontsize=10)
    ax3.set_title("Feature distribution: roll_spread_500\nTrain vs Test (regime check)",
                  fontsize=11, fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.18)
    ax3.spines["top"].set_visible(False); ax3.spines["right"].set_visible(False)

    # ── Panel 4: Predicted vs actual for best MAE model ──────────────────────────
    best_name = MODEL_NAMES[np.argmin(OOS_MAE)]
    best_pred = ALL_PREDS[np.argmin(OOS_MAE)]
    best_color = COLORS[np.argmin(OOS_MAE)]
    clip_v = np.percentile(y_te, 98)
    rng = np.random.default_rng(42)
    samp = rng.choice(len(y_te), size=min(5000, len(y_te)), replace=False)
    ax4.scatter(y_te[samp], np.clip(best_pred[samp], 0, clip_v),
                alpha=0.07, s=5, color=best_color, linewidths=0)
    ax4.axline((0, 0), slope=1, color="black", lw=1.2, ls="--", alpha=0.6,
               label="Perfect prediction")
    ax4.set_xlim(0, clip_v); ax4.set_ylim(0, clip_v)
    ax4.set_xlabel("Actual |impact_vwap_bps|  (Sep 2024)", fontsize=10)
    ax4.set_ylabel(f"Predicted  [{best_name}]", fontsize=10)
    ax4.set_title(f"Predicted vs actual — {best_name}\n"
                  f"R²={r2(y_te, best_pred):+.4f}  MAE={mae(y_te, best_pred):.4f} bps",
                  fontsize=11, fontweight="bold")
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.18)
    ax4.spines["top"].set_visible(False); ax4.spines["right"].set_visible(False)

    # ── Panel 5: Absolute error distributions (violin) ────────────────────────────
    abs_errs = [np.abs(y_te - p) for p in ALL_PREDS]
    clip_e   = np.percentile(abs_errs[0], 97)
    vdata    = [np.clip(e, 0, clip_e) for e in abs_errs]
    vp = ax5.violinplot(vdata, positions=xpos, widths=0.5,
                        showmedians=True, showextrema=False)
    for patch, color in zip(vp["bodies"], COLORS):
        patch.set_facecolor(color); patch.set_alpha(0.45)
    vp["cmedians"].set_color("black"); vp["cmedians"].set_linewidth(1.8)
    for i, (name, e) in enumerate(zip(MODEL_NAMES, abs_errs)):
        med = np.median(e)
        ax5.text(i, med + clip_e * 0.02, f"med={med:.3f}", ha="center",
                 va="bottom", fontsize=7.5, fontweight="bold")
    ax5.set_xticks(xpos); ax5.set_xticklabels(MODEL_NAMES, fontsize=8.5, rotation=10)
    ax5.set_ylabel(f"|error|  (bps, clipped at {clip_e:.1f})", fontsize=10)
    ax5.set_title("Absolute error distribution\n(Sep 2024 holdout, individual trades)",
                  fontsize=11, fontweight="bold")
    ax5.grid(True, axis="y", alpha=0.2)
    ax5.spines["top"].set_visible(False); ax5.spines["right"].set_visible(False)

    # ── Panel 6: MAE by September date ────────────────────────────────────────────
    dates_te  = df_te["date"].to_numpy()
    sep_dates = sorted(set(dates_te))
    w = 0.14
    d_pos = np.arange(len(sep_dates))

    for ci, (name, preds, color) in enumerate(zip(MODEL_NAMES, ALL_PREDS, COLORS)):
        day_mae = [mae(y_te[dates_te == d], preds[dates_te == d]) for d in sep_dates]
        offset  = (ci - 2) * w
        ax6.bar(d_pos + offset, day_mae, width=w, color=color,
                alpha=0.8, label=name, edgecolor="white", linewidth=0.3)

    ax6.set_xticks(d_pos)
    ax6.set_xticklabels([d[5:] for d in sep_dates], fontsize=7, rotation=45)
    ax6.set_ylabel("Daily MAE  (bps)", fontsize=10)
    ax6.set_title("Per-day OOS MAE across September 2024\n(spike days reveal regime stress)",
                  fontsize=11, fontweight="bold")
    ax6.legend(fontsize=7.5, loc="upper right", ncol=2)
    ax6.grid(True, axis="y", alpha=0.2)
    ax6.spines["top"].set_visible(False); ax6.spines["right"].set_visible(False)

    fig.suptitle(
        "AAPL lit buy block trades — True temporal holdout evaluation\n"
        f"Train: Jun–Aug 2024 ({len(df_tr):,} trades, 63 days)  "
        f"->  Test: Sep 2024 ({len(df_te):,} trades, 20 days)",
        fontsize=12, fontweight="bold", y=1.01,
    )

    plt.savefig("aapl_temporal_holdout.png", dpi=150, bbox_inches="tight")
    print("\nsaved -> aapl_temporal_holdout.png")


# =============================================================================
if __name__ == "__main__":
    run_ols_fit_visualization()
    run_ols_fitted_line()
    run_ols_unsigned_by_feature()
    run_ols_unsigned_diagnostics()
    run_ols_unsigned_visualization()
    run_coin_signed_ols_plot()
    run_coin_no_ticktest_ols()
    run_temporal_holdout()
