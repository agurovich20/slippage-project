"""
Signed OLS visualisation for AAPL and COIN, with LAD and XGB-MAE overlaid.

Figure 1 (aapl_coin_signed_ols_fit.png)  — 2×2
  Top:    scatter + OLS / LAD / XGB-MAE fit lines
  Bottom: residuals vs fitted (OLS)

Figure 2 (aapl_coin_signed_ols_diag.png) — 2×2
  Top:    predicted vs actual (test, all 3 models)
  Bottom: residual histograms (OLS vs LAD overlay)
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

# ══════════════════════════════════════════════════════════════════════════════
# Load data
# ══════════════════════════════════════════════════════════════════════════════
aapl_tr = pd.read_parquet("data/lit_buy_features_v2.parquet")
aapl_te = pd.read_parquet("data/lit_buy_features_v2_sep.parquet")
coin_tr = pd.read_parquet("data/coin_lit_buy_features_train.parquet")
coin_te = pd.read_parquet("data/coin_lit_buy_features_test.parquet")

TARGET = "impact_vwap_bps"
FEAT_UNI = ["roll_spread_500"]
FEAT_MULTI = ["roll_spread_500", "roll_vol_500", "participation_rate"]


def r2_fn(y, yh):
    ss_r = ((y - yh) ** 2).sum()
    ss_t = ((y - y.mean()) ** 2).sum()
    return 1.0 - ss_r / ss_t if ss_t > 0 else np.nan


def mae_fn(y, yh):
    return np.mean(np.abs(y - yh))


# ══════════════════════════════════════════════════════════════════════════════
# Fit models per ticker
# ══════════════════════════════════════════════════════════════════════════════
fits = {}

for ticker, df_tr, df_te in [("AAPL", aapl_tr, aapl_te),
                               ("COIN", coin_tr, coin_te)]:
    y_tr = df_tr[TARGET].values.astype(np.float64)
    y_te = df_te[TARGET].values.astype(np.float64)
    x_tr = df_tr["roll_spread_500"].values.astype(np.float64)
    x_te = df_te["roll_spread_500"].values.astype(np.float64)

    X_tr_u = np.column_stack([x_tr, np.ones(len(x_tr))])
    X_te_u = np.column_stack([x_te, np.ones(len(x_te))])

    # OLS-Uni
    beta_ols, *_ = np.linalg.lstsq(X_tr_u, y_tr, rcond=None)
    pred_ols_tr = X_tr_u @ beta_ols
    pred_ols_te = X_te_u @ beta_ols

    # LAD-Uni
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lad = QuantReg(y_tr, X_tr_u).fit(q=0.5, max_iter=5000, p_tol=1e-6)
    pred_lad_tr = X_tr_u @ lad.params
    pred_lad_te = X_te_u @ lad.params

    # XGB-MAE (3 features)
    X3_tr = df_tr[FEAT_MULTI].values.astype(np.float64)
    X3_te = df_te[FEAT_MULTI].values.astype(np.float64)
    xgb = XGBRegressor(
        objective="reg:absoluteerror",
        max_depth=3, n_estimators=50, learning_rate=0.1,
        min_child_weight=1, reg_alpha=10, reg_lambda=10,
        tree_method="hist", verbosity=0, random_state=42, n_jobs=1,
    )
    xgb.fit(X3_tr, y_tr)
    pred_xgb_tr = xgb.predict(X3_tr)
    pred_xgb_te = xgb.predict(X3_te)

    fits[ticker] = dict(
        df_tr=df_tr, df_te=df_te,
        y_tr=y_tr, y_te=y_te, x_tr=x_tr, x_te=x_te,
        beta_ols=beta_ols, beta_lad=lad.params,
        pred_ols_tr=pred_ols_tr, pred_ols_te=pred_ols_te,
        pred_lad_tr=pred_lad_tr, pred_lad_te=pred_lad_te,
        pred_xgb_tr=pred_xgb_tr, pred_xgb_te=pred_xgb_te,
    )

    # quick summary
    for nm, p in [("OLS", pred_ols_te), ("LAD", pred_lad_te), ("XGB-MAE", pred_xgb_te)]:
        print(f"  {ticker} {nm:<8}  R2={r2_fn(y_te,p):+.4f}  MAE={mae_fn(y_te,p):.4f}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — scatter + fit lines  /  residuals vs fitted
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 14))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.28)

TICKER_COLOR = {"AAPL": "#2563eb", "COIN": "#dc2626"}
rng = np.random.default_rng(42)

for col, ticker in enumerate(["AAPL", "COIN"]):
    f = fits[ticker]
    color = TICKER_COLOR[ticker]

    # ── top: scatter + 3 fit lines ────────────────────────────────────────
    ax = fig.add_subplot(gs[0, col])

    n_plot = min(4000, len(f["x_tr"]))
    idx = rng.choice(len(f["x_tr"]), size=n_plot, replace=False)

    ax.scatter(f["x_tr"][idx], f["y_tr"][idx], s=6, alpha=0.12,
               color=color, edgecolors="none", zorder=2, label="Train")
    ax.scatter(f["x_te"], f["y_te"], s=10, alpha=0.20,
               color="#f59e0b", edgecolors="none", zorder=2, label="Test (Sep)")

    xg = np.linspace(0, np.percentile(np.concatenate([f["x_tr"], f["x_te"]]), 99.5), 300)

    # OLS line
    b = f["beta_ols"]
    ax.plot(xg, b[0] * xg + b[1], color="black", lw=2.2, zorder=5,
            label=f"OLS: {b[0]:+.4f}x {b[1]:+.4f}")

    # LAD line
    bl = f["beta_lad"]
    ax.plot(xg, bl[0] * xg + bl[1], color="#16a34a", lw=2.2, ls="--", zorder=5,
            label=f"LAD: {bl[0]:+.4f}x {bl[1]:+.4f}")

    # XGB — plot binned prediction vs spread (not a line)
    spread_all = np.concatenate([f["x_tr"], f["x_te"]])
    pred_xgb_all = np.concatenate([f["pred_xgb_tr"], f["pred_xgb_te"]])
    order = np.argsort(spread_all)
    n_xgb_bins = 40
    chunks = np.array_split(order, n_xgb_bins)
    bx = [spread_all[c].mean() for c in chunks]
    by = [pred_xgb_all[c].mean() for c in chunks]
    ax.plot(bx, by, color="#7c3aed", lw=2.2, ls=":", marker=".", markersize=5,
            zorder=5, label="XGB-MAE (binned pred)")

    ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.5, zorder=1)

    # stats box
    box = (
        f"{'':>10} {'R²':>7} {'MAE':>7}\n"
        f"{'OLS':>10} {r2_fn(f['y_te'], f['pred_ols_te']):>+7.4f} {mae_fn(f['y_te'], f['pred_ols_te']):>7.3f}\n"
        f"{'LAD':>10} {r2_fn(f['y_te'], f['pred_lad_te']):>+7.4f} {mae_fn(f['y_te'], f['pred_lad_te']):>7.3f}\n"
        f"{'XGB-MAE':>10} {r2_fn(f['y_te'], f['pred_xgb_te']):>+7.4f} {mae_fn(f['y_te'], f['pred_xgb_te']):>7.3f}"
    )
    ax.text(0.97, 0.97, box, transform=ax.transAxes, fontsize=8.5,
            family="monospace", va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#94a3b8", alpha=0.92))

    all_y = np.concatenate([f["y_tr"], f["y_te"]])
    y_lo, y_hi = np.percentile(all_y, [1, 99])
    margin = (y_hi - y_lo) * 0.1
    ax.set_ylim(y_lo - margin, y_hi + margin)

    ax.set_xlabel("roll_spread_500 (bps)", fontsize=11)
    ax.set_ylabel("impact_vwap_bps (signed)", fontsize=11)
    ax.set_title(f"{ticker} — signed impact vs spread\nOLS / LAD / XGB-MAE",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── bottom: residuals vs fitted (OLS, train) ─────────────────────────
    ax2 = fig.add_subplot(gs[1, col])

    resid = f["y_tr"] - f["pred_ols_tr"]
    fitted = f["pred_ols_tr"]

    n_rp = min(4000, len(resid))
    idx_r = rng.choice(len(resid), size=n_rp, replace=False)

    ax2.scatter(fitted[idx_r], resid[idx_r], s=5, alpha=0.12,
                color=color, edgecolors="none", zorder=2)
    ax2.axhline(0, color="black", lw=1.2, ls="--", alpha=0.7, zorder=3)

    # binned mean residual
    r_order = np.argsort(fitted)
    chunks = np.array_split(r_order, 30)
    bx_r = [fitted[c].mean() for c in chunks]
    by_r = [resid[c].mean() for c in chunks]
    ax2.plot(bx_r, by_r, color="#16a34a", lw=2.2, zorder=5,
             label="Binned mean residual")

    ax2.text(0.97, 0.97,
             f"Residual mean: {resid.mean():.4f}\n"
             f"Residual std:  {resid.std():.4f}\n"
             f"Skewness: {stats.skew(resid):.3f}\n"
             f"Kurtosis: {stats.kurtosis(resid):.3f}",
             transform=ax2.transAxes, fontsize=9, family="monospace",
             va="top", ha="right",
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#94a3b8", alpha=0.92))

    r_cap = np.percentile(np.abs(resid), 99)
    ax2.set_ylim(-r_cap * 1.2, r_cap * 1.2)
    ax2.set_xlabel("Fitted value (bps)", fontsize=11)
    ax2.set_ylabel("Residual (bps)", fontsize=11)
    ax2.set_title(f"{ticker} — OLS residuals vs fitted (train, signed)",
                  fontsize=13, fontweight="bold")
    ax2.legend(fontsize=9, loc="upper left")
    ax2.grid(True, alpha=0.18)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

fig.suptitle(
    "Signed impact_vwap_bps ~ roll_spread_500  —  OLS / LAD / XGB-MAE\n"
    "AAPL vs COIN lit buy block trades",
    fontsize=15, fontweight="bold", y=1.01,
)
plt.savefig("aapl_coin_signed_ols_fit.png", dpi=150, bbox_inches="tight")
print("Saved -> aapl_coin_signed_ols_fit.png")
plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — predicted vs actual  /  residual histograms
# ══════════════════════════════════════════════════════════════════════════════
fig2 = plt.figure(figsize=(18, 14))
gs2 = gridspec.GridSpec(2, 2, figure=fig2, hspace=0.35, wspace=0.28)

MODEL_STYLES = {
    "OLS":     dict(color="black",   alpha=0.25, marker="o", ms=8,  lbl_c="black"),
    "LAD":     dict(color="#16a34a",  alpha=0.25, marker="s", ms=8,  lbl_c="#16a34a"),
    "XGB-MAE": dict(color="#7c3aed", alpha=0.25, marker="D", ms=7,  lbl_c="#7c3aed"),
}

for col, ticker in enumerate(["AAPL", "COIN"]):
    f = fits[ticker]
    color = TICKER_COLOR[ticker]

    # ── top: predicted vs actual (test) ───────────────────────────────────
    ax = fig2.add_subplot(gs2[0, col])
    y_te = f["y_te"]

    preds = [
        ("OLS",     f["pred_ols_te"]),
        ("LAD",     f["pred_lad_te"]),
        ("XGB-MAE", f["pred_xgb_te"]),
    ]

    all_vals = np.concatenate([y_te] + [p for _, p in preds])
    lo, hi = np.percentile(all_vals, [1, 99])
    margin = (hi - lo) * 0.1
    lo -= margin; hi += margin

    ax.plot([lo, hi], [lo, hi], color="gray", lw=1.2, ls="--", alpha=0.6,
            zorder=1, label="Perfect prediction")

    for nm, pred in preds:
        st = MODEL_STYLES[nm]
        # binned means
        order = np.argsort(y_te)
        chunks = np.array_split(order, 25)
        ba = [y_te[c].mean() for c in chunks]
        bp = [pred[c].mean() for c in chunks]
        r2 = r2_fn(y_te, pred)
        mae = mae_fn(y_te, pred)
        ax.plot(ba, bp, color=st["lbl_c"], lw=2.2, marker=st["marker"],
                markersize=st["ms"], alpha=0.8, zorder=4,
                label=f"{nm}  R²={r2:+.4f}  MAE={mae:.3f}")

    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Actual impact (signed, bps)", fontsize=11)
    ax.set_ylabel("Predicted impact (signed, bps)", fontsize=11)
    ax.set_title(f"{ticker} — Predicted vs Actual (Sep holdout, binned means)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=8.5, loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── bottom: residual histograms (OLS vs LAD, test) ────────────────────
    ax_h = fig2.add_subplot(gs2[1, col])

    resid_ols = y_te - f["pred_ols_te"]
    resid_lad = y_te - f["pred_lad_te"]
    resid_xgb = y_te - f["pred_xgb_te"]

    clip = np.percentile(np.abs(resid_ols), 99.5)
    bins_h = np.linspace(-clip, clip, 70)

    ax_h.hist(resid_ols, bins=bins_h, density=True, alpha=0.45,
              color=color, edgecolor="none", label="OLS residuals", zorder=2)
    ax_h.hist(resid_lad, bins=bins_h, density=True, alpha=0.40,
              color="#16a34a", edgecolor="none", label="LAD residuals", zorder=3)
    ax_h.hist(resid_xgb, bins=bins_h, density=True, alpha=0.35,
              color="#7c3aed", edgecolor="none", label="XGB-MAE residuals", zorder=3)

    # Laplace fit on OLS residuals
    loc_l, b_l = stats.laplace.fit(resid_ols)
    xg_r = np.linspace(-clip, clip, 300)
    ax_h.plot(xg_r, stats.laplace.pdf(xg_r, loc_l, b_l),
              color="black", lw=1.8, ls="--", alpha=0.7, zorder=4,
              label=f"Laplace (loc={loc_l:.2f}, b={b_l:.2f})")

    ax_h.axvline(0, color="gray", lw=1, ls="--", alpha=0.5)

    ax_h.text(0.97, 0.97,
              f"OLS:     mean={resid_ols.mean():+.3f}  std={resid_ols.std():.3f}\n"
              f"LAD:     mean={resid_lad.mean():+.3f}  std={resid_lad.std():.3f}\n"
              f"XGB-MAE: mean={resid_xgb.mean():+.3f}  std={resid_xgb.std():.3f}",
              transform=ax_h.transAxes, fontsize=9, family="monospace",
              va="top", ha="right",
              bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#94a3b8", alpha=0.92))

    ax_h.set_xlabel("Residual (bps)", fontsize=11)
    ax_h.set_ylabel("Density", fontsize=11)
    ax_h.set_title(f"{ticker} — Test residual distributions (signed target)",
                   fontsize=13, fontweight="bold")
    ax_h.legend(fontsize=8, loc="upper left")
    ax_h.grid(True, alpha=0.18)
    ax_h.spines["top"].set_visible(False)
    ax_h.spines["right"].set_visible(False)

fig2.suptitle(
    "OLS / LAD / XGB-MAE Diagnostics — signed impact_vwap_bps\n"
    "AAPL vs COIN lit buy block trades (Sep holdout)",
    fontsize=15, fontweight="bold", y=1.01,
)
plt.savefig("aapl_coin_signed_ols_diag.png", dpi=150, bbox_inches="tight")
print("Saved -> aapl_coin_signed_ols_diag.png")
plt.close(fig2)
