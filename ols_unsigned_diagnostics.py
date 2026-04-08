"""
Additional unsigned OLS diagnostics and visualizations.

Output:
  - ols_unsigned_qq_residuals.png       QQ plots: Normal vs Laplace fit
  - ols_unsigned_partial_dependence.png Partial dependence (marginal effects)
  - ols_unsigned_error_by_regime.png    MAE broken down by trade size regime
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

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
