"""
OLS regression visualizations for unsigned slippage: abs(impact_vwap_bps).

Uses all 6 features. Temporal holdout: Jun-Aug train, Sep test.

Output:
  - ols_unsigned_fit.png              (2x2: scatter + residuals per ticker)
  - ols_unsigned_predicted_vs_actual.png  (2x2: pred vs actual + residual hist)
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
