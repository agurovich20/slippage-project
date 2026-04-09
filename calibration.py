"""
Calibration analysis for the XGB GAMLSS slippage model.

Consolidates:
  - calibration_curves.py     : Laplace coverage curves across 6 stocks
  - calibration_naive.py      : Naive Gaussian (global mean/std) baseline
  - calibration_naive_ols.py  : OLS Gaussian baseline
  - calibration_new_stocks.py : Calibration for NVDA, AMD, AMZN, TSLA
  - calibration_overlays.py   : Linear vs XGB overlay: AAPL vs COIN
  - calibration_pooled.py     : Pooled XGB GAMLSS calibration
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xgboost import XGBRegressor


# ══════════════════════════════════════════════════════════════════════════════
# calibration_curves.py
# ══════════════════════════════════════════════════════════════════════════════
def run_calibration_curves():
    """Calibration curves for XGB GAMLSS Laplace prediction intervals across 6 stocks.
    Plots nominal coverage vs actual coverage at many levels.
    """
    FEATURES_3 = ["roll_spread_500", "roll_vol_500", "participation_rate"]
    IMPACT_CAP_BPS = 500

    DATASETS = {
        "AAPL": ("data/lit_buy_features_v2.parquet", "data/lit_buy_features_v2_sep.parquet"),
        "COIN": ("data/coin_lit_buy_features_train.parquet", "data/coin_lit_buy_features_test.parquet"),
        "NVDA": ("data/nvda_lit_buy_features_train.parquet", "data/nvda_lit_buy_features_test.parquet"),
        "AMD":  ("data/amd_lit_buy_features_train.parquet", "data/amd_lit_buy_features_test.parquet"),
        "AMZN": ("data/amzn_lit_buy_features_train.parquet", "data/amzn_lit_buy_features_test.parquet"),
        "TSLA": ("data/tsla_lit_buy_features_train.parquet", "data/tsla_lit_buy_features_test.parquet"),
    }

    COLORS = {
        "AAPL": "#2563eb", "COIN": "#dc2626", "NVDA": "#16a34a",
        "AMD":  "#f59e0b", "AMZN": "#7c3aed", "TSLA": "#ec4899",
    }

    NOMINAL_LEVELS = np.arange(0.05, 1.00, 0.05)  # 5%, 10%, ..., 95%

    def fit_gamlss(train_df, test_df):
        X_tr = train_df[FEATURES_3].to_numpy(dtype=np.float64)
        y_tr = train_df["impact_vwap_bps"].abs().to_numpy(dtype=np.float64)
        X_te = test_df[FEATURES_3].to_numpy(dtype=np.float64)
        y_te = test_df["impact_vwap_bps"].abs().to_numpy(dtype=np.float64)

        loc_model = XGBRegressor(
            objective="reg:absoluteerror", tree_method="hist",
            max_depth=3, n_estimators=200, learning_rate=0.07,
            min_child_weight=5, reg_alpha=10, reg_lambda=10,
            verbosity=0, random_state=42, n_jobs=1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loc_model.fit(X_tr, y_tr)

        mu_hat_tr = np.maximum(loc_model.predict(X_tr), 0.0)
        mu_hat_te = np.maximum(loc_model.predict(X_te), 0.0)

        abs_resid_tr = np.abs(y_tr - mu_hat_tr)
        scale_model = XGBRegressor(
            objective="reg:squarederror", tree_method="hist",
            max_depth=3, n_estimators=50, learning_rate=0.1,
            min_child_weight=20, reg_alpha=1, reg_lambda=1,
            verbosity=0, random_state=42, n_jobs=1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scale_model.fit(X_tr, abs_resid_tr)

        b_hat_te = np.clip(scale_model.predict(X_te), 0.1, None)

        return y_te, mu_hat_te, b_hat_te

    def compute_calibration(y_te, mu_hat_te, b_hat_te, levels):
        actual_coverages = []
        for level in levels:
            z = np.log(1.0 / (1.0 - level))
            lo = np.maximum(mu_hat_te - z * b_hat_te, 0.0)
            hi = mu_hat_te + z * b_hat_te
            cov = ((y_te >= lo) & (y_te <= hi)).mean()
            actual_coverages.append(cov)
        return np.array(actual_coverages)

    # Fit models and compute calibration for each stock
    results = {}
    for ticker, (tr_path, te_path) in DATASETS.items():
        train_df = pd.read_parquet(tr_path)
        test_df = pd.read_parquet(te_path)
        train_df = train_df[train_df["impact_vwap_bps"].abs() <= IMPACT_CAP_BPS]
        test_df = test_df[test_df["impact_vwap_bps"].abs() <= IMPACT_CAP_BPS]

        y_te, mu_hat_te, b_hat_te = fit_gamlss(train_df, test_df)
        actual = compute_calibration(y_te, mu_hat_te, b_hat_te, NOMINAL_LEVELS)
        results[ticker] = actual
        print(f"  {ticker}: done (n_test={len(y_te):,})")

    # ─── Figure ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    # Panel 1: Calibration curves (all stocks overlaid)
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color="black", lw=1.5, ls="--", alpha=0.6,
            label="Perfect calibration", zorder=1)
    ax.fill_between([0, 1], [0, 1], [0.05, 1.05], color="gray", alpha=0.06)
    ax.fill_between([0, 1], [-0.05, 0.95], [0, 1], color="gray", alpha=0.06)

    for ticker in DATASETS:
        ax.plot(NOMINAL_LEVELS, results[ticker], color=COLORS[ticker],
                lw=2.5, marker="o", markersize=5, label=ticker, zorder=3)

    ax.set_xlabel("Nominal Coverage Level", fontsize=13)
    ax.set_ylabel("Actual Coverage", fontsize=13)
    ax.set_title("Calibration Curves: XGB GAMLSS with Laplace Intervals\n"
                 "6 Stocks, 3 Features, Fixed AAPL Hyperparameters",
                 fontsize=14, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(fontsize=10, loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel 2: Deviation from nominal (actual - nominal)
    ax2 = axes[1]
    ax2.axhline(0, color="black", lw=1.5, ls="--", alpha=0.6, zorder=1)
    ax2.fill_between(NOMINAL_LEVELS, -0.02, 0.02, color="green", alpha=0.08,
                     label="\u00b12% band")
    ax2.fill_between(NOMINAL_LEVELS, -0.05, 0.05, color="orange", alpha=0.05,
                     label="\u00b15% band")

    for ticker in DATASETS:
        deviation = results[ticker] - NOMINAL_LEVELS
        ax2.plot(NOMINAL_LEVELS, deviation, color=COLORS[ticker],
                 lw=2.5, marker="o", markersize=5, label=ticker, zorder=3)

    ax2.set_xlabel("Nominal Coverage Level", fontsize=13)
    ax2.set_ylabel("Actual \u2212 Nominal Coverage", fontsize=13)
    ax2.set_title("Coverage Deviation from Nominal\n"
                  "Positive = over-coverage (conservative), Negative = under-coverage",
                  fontsize=14, fontweight="bold")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(-0.15, 0.15)
    ax2.legend(fontsize=9, loc="upper left", framealpha=0.9, ncol=2)
    ax2.grid(True, alpha=0.2)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle(
        "XGB GAMLSS Calibration: Block Trade Impact Prediction Intervals\n"
        "Train: Jun\u2013Aug 2024 | Test: Sep 2024 | Laplace distribution",
        fontsize=15, fontweight="bold", y=1.03,
    )

    plt.tight_layout()
    fig.savefig("calibration_curves.png", dpi=150, bbox_inches="tight")
    print("\nSaved -> calibration_curves.png")


# ══════════════════════════════════════════════════════════════════════════════
# calibration_naive.py
# ══════════════════════════════════════════════════════════════════════════════
def run_calibration_naive():
    """Naive baseline calibration: global mean + global std, Gaussian intervals.
    mu = training mean of |impact|, sigma = training std of |impact|.
    Same interval for every trade.

    Output: calibration_naive.png
    """
    IMPACT_CAP_BPS = 500
    cal_levels = np.linspace(0.05, 0.99, 50)

    DATASETS = {
        "AAPL": ("data/lit_buy_features_v2.parquet", "data/lit_buy_features_v2_sep.parquet"),
        "COIN": ("data/coin_lit_buy_features_train.parquet", "data/coin_lit_buy_features_test.parquet"),
        "NVDA": ("data/nvda_lit_buy_features_train.parquet", "data/nvda_lit_buy_features_test.parquet"),
        "AMD":  ("data/amd_lit_buy_features_train.parquet", "data/amd_lit_buy_features_test.parquet"),
        "AMZN": ("data/amzn_lit_buy_features_train.parquet", "data/amzn_lit_buy_features_test.parquet"),
        "TSLA": ("data/tsla_lit_buy_features_train.parquet", "data/tsla_lit_buy_features_test.parquet"),
    }

    COLORS = {
        "AAPL": "#2563eb", "COIN": "#dc2626",
        "NVDA": "#16a34a", "AMD": "#f59e0b", "AMZN": "#7c3aed", "TSLA": "#ec4899",
    }
    MARKERS = {
        "AAPL": "o", "COIN": "X",
        "NVDA": "D", "AMD": "s", "AMZN": "^", "TSLA": "v",
    }

    def normal_coverage(y, mu, sigma, level):
        """Coverage of a symmetric Gaussian interval [mu - z*sigma, mu + z*sigma], clipped at 0."""
        z = stats.norm.ppf(0.5 + level / 2.0)
        lo = np.maximum(mu - z * sigma, 0.0)
        hi = mu + z * sigma
        return ((y >= lo) & (y <= hi)).mean()

    results = {}
    for ticker, (tr_f, te_f) in DATASETS.items():
        df_tr = pd.read_parquet(tr_f)
        df_te = pd.read_parquet(te_f)
        df_tr = df_tr[df_tr["impact_vwap_bps"].abs() <= IMPACT_CAP_BPS]
        df_te = df_te[df_te["impact_vwap_bps"].abs() <= IMPACT_CAP_BPS]

        y_tr = df_tr["impact_vwap_bps"].abs().to_numpy(dtype=np.float64)
        y_te = df_te["impact_vwap_bps"].abs().to_numpy(dtype=np.float64)

        mu = y_tr.mean()
        sigma = y_tr.std()

        cal = np.array([normal_coverage(y_te, mu, sigma, lv) for lv in cal_levels])

        results[ticker] = {
            "cal": cal, "mu": mu, "sigma": sigma,
            "y_te": y_te, "n_test": len(y_te),
        }
        print(f"{ticker}: mu={mu:.2f}  sigma={sigma:.2f}  n_test={len(y_te):,}")

    # -- Calibration curve ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 9))

    for ticker in DATASETS:
        r = results[ticker]
        ax.plot(cal_levels, r["cal"], color=COLORS[ticker], lw=2.5,
                marker=MARKERS[ticker], markersize=4, markevery=3,
                label=f"{ticker} (n={r['n_test']:,})")

    ax.plot([0, 1], [0, 1], color="black", lw=1.2, ls="--", alpha=0.5,
            label="Perfect calibration")

    ax.set_xlabel("Nominal coverage level", fontsize=12)
    ax.set_ylabel("Actual coverage", fontsize=12)
    ax.set_title("Naive Gaussian Calibration",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig("calibration_naive.png", dpi=150, bbox_inches="tight")
    print("Saved -> calibration_naive.png")

    # -- Coverage deviation --------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    ax2.axhline(0, color="black", lw=1.5, ls="--", alpha=0.6, zorder=1)
    ax2.fill_between(cal_levels, -0.05, 0.05, color="orange", alpha=0.05, label="\u00b15% band")

    for ticker in DATASETS:
        r = results[ticker]
        deviation = r["cal"] - cal_levels
        ax2.plot(cal_levels, deviation, color=COLORS[ticker], lw=2.5,
                 marker=MARKERS[ticker], markersize=5, markevery=3,
                 label=f"{ticker} (n={r['n_test']:,})", zorder=3)

    ax2.set_xlabel("Nominal Coverage Level", fontsize=13)
    ax2.set_ylabel("Actual \u2212 Nominal Coverage", fontsize=13)
    ax2.set_title("Coverage Deviation: Naive Gaussian",
                  fontsize=13, fontweight="bold")
    ax2.set_xlim(0.05, 0.95)
    ax2.set_ylim(-0.12, 0.15)
    ax2.legend(fontsize=10, loc="upper left", framealpha=0.9)
    ax2.grid(True, alpha=0.2)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    fig2.savefig("coverage_deviation_naive.png", dpi=150, bbox_inches="tight")
    print("Saved -> coverage_deviation_naive.png")

    print("\nDone!")


# ══════════════════════════════════════════════════════════════════════════════
# calibration_naive_ols.py
# ══════════════════════════════════════════════════════════════════════════════
def run_calibration_naive_ols():
    """Naive Gaussian + OLS baseline calibration.
    mu_i = OLS(spread) per trade, sigma = constant training RMSE.
    Gaussian intervals: [mu_i - z*sigma, mu_i + z*sigma].

    Output: calibration_naive_ols.png, coverage_deviation_naive_ols.png
    """
    IMPACT_CAP_BPS = 500
    cal_levels = np.linspace(0.05, 0.99, 50)

    DATASETS = {
        "AAPL": ("data/lit_buy_features_v2.parquet", "data/lit_buy_features_v2_sep.parquet"),
        "COIN": ("data/coin_lit_buy_features_train.parquet", "data/coin_lit_buy_features_test.parquet"),
        "NVDA": ("data/nvda_lit_buy_features_train.parquet", "data/nvda_lit_buy_features_test.parquet"),
        "AMD":  ("data/amd_lit_buy_features_train.parquet", "data/amd_lit_buy_features_test.parquet"),
        "AMZN": ("data/amzn_lit_buy_features_train.parquet", "data/amzn_lit_buy_features_test.parquet"),
        "TSLA": ("data/tsla_lit_buy_features_train.parquet", "data/tsla_lit_buy_features_test.parquet"),
    }

    COLORS = {
        "AAPL": "#2563eb", "COIN": "#dc2626",
        "NVDA": "#16a34a", "AMD": "#f59e0b", "AMZN": "#7c3aed", "TSLA": "#ec4899",
    }
    MARKERS = {
        "AAPL": "o", "COIN": "X",
        "NVDA": "D", "AMD": "s", "AMZN": "^", "TSLA": "v",
    }

    def normal_coverage(y, mu, sigma, level):
        z = stats.norm.ppf(0.5 + level / 2.0)
        lo = np.maximum(mu - z * sigma, 0.0)
        hi = mu + z * sigma
        return ((y >= lo) & (y <= hi)).mean()

    results = {}
    for ticker, (tr_f, te_f) in DATASETS.items():
        df_tr = pd.read_parquet(tr_f)
        df_te = pd.read_parquet(te_f)
        df_tr = df_tr[df_tr["impact_vwap_bps"].abs() <= IMPACT_CAP_BPS]
        df_te = df_te[df_te["impact_vwap_bps"].abs() <= IMPACT_CAP_BPS]

        y_tr = df_tr["impact_vwap_bps"].abs().to_numpy(dtype=np.float64)
        y_te = df_te["impact_vwap_bps"].abs().to_numpy(dtype=np.float64)
        x_tr = df_tr["roll_spread_500"].to_numpy(dtype=np.float64)
        x_te = df_te["roll_spread_500"].to_numpy(dtype=np.float64)

        # OLS: y = c1*spread + c0
        X_tr = np.column_stack([x_tr, np.ones(len(x_tr))])
        X_te = np.column_stack([x_te, np.ones(len(x_te))])
        beta, *_ = np.linalg.lstsq(X_tr, y_tr, rcond=None)

        mu_tr = np.maximum(X_tr @ beta, 0.0)
        mu_te = np.maximum(X_te @ beta, 0.0)

        # Constant sigma = training RMSE
        sigma = np.sqrt(np.mean((y_tr - mu_tr) ** 2))

        cal = np.array([normal_coverage(y_te, mu_te, sigma, lv) for lv in cal_levels])

        results[ticker] = {
            "cal": cal, "mu_te": mu_te, "sigma": sigma,
            "beta": beta, "y_te": y_te, "n_test": len(y_te),
        }
        print(f"{ticker}: c1={beta[0]:+.4f}  c0={beta[1]:+.4f}  sigma={sigma:.2f}  n_test={len(y_te):,}")

    # -- Calibration curve ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 9))

    for ticker in DATASETS:
        r = results[ticker]
        ax.plot(cal_levels, r["cal"], color=COLORS[ticker], lw=2.5,
                marker=MARKERS[ticker], markersize=4, markevery=3,
                label=f"{ticker} (n={r['n_test']:,})")

    ax.plot([0, 1], [0, 1], color="black", lw=1.2, ls="--", alpha=0.5,
            label="Perfect calibration")

    ax.set_xlabel("Nominal coverage level", fontsize=12)
    ax.set_ylabel("Actual coverage", fontsize=12)
    ax.set_title("OLS Gaussian Calibration",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig("calibration_naive_ols.png", dpi=150, bbox_inches="tight")
    print("Saved -> calibration_naive_ols.png")

    # -- Coverage deviation --------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    ax2.axhline(0, color="black", lw=1.5, ls="--", alpha=0.6, zorder=1)
    ax2.fill_between(cal_levels, -0.05, 0.05, color="orange", alpha=0.05, label="\u00b15% band")

    for ticker in DATASETS:
        r = results[ticker]
        deviation = r["cal"] - cal_levels
        ax2.plot(cal_levels, deviation, color=COLORS[ticker], lw=2.5,
                 marker=MARKERS[ticker], markersize=5, markevery=3,
                 label=f"{ticker} (n={r['n_test']:,})", zorder=3)

    ax2.set_xlabel("Nominal Coverage Level", fontsize=13)
    ax2.set_ylabel("Actual \u2212 Nominal Coverage", fontsize=13)
    ax2.set_title("Coverage Deviation: OLS Gaussian",
                  fontsize=13, fontweight="bold")
    ax2.set_xlim(0.05, 0.95)
    ax2.set_ylim(-0.12, 0.15)
    ax2.legend(fontsize=10, loc="upper left", framealpha=0.9)
    ax2.grid(True, alpha=0.2)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    fig2.savefig("coverage_deviation_naive_ols.png", dpi=150, bbox_inches="tight")
    print("Saved -> coverage_deviation_naive_ols.png")

    print("\nDone!")


# ══════════════════════════════════════════════════════════════════════════════
# calibration_new_stocks.py
# ══════════════════════════════════════════════════════════════════════════════
def run_calibration_new_stocks():
    """Calibration curves and coverage deviation for NVDA, AMD, AMZN, TSLA.
    Two-Stage XGBoost with Laplace intervals, 3 features, fixed AAPL hyperparameters.

    Output:
      - calibration_xgb_nvda_amd.png
      - calibration_xgb_amzn_tsla.png
      - coverage_deviation_4stocks.png
    """
    FEATURES_3 = ["roll_spread_500", "roll_vol_500", "participation_rate"]
    IMPACT_CAP_BPS = 500
    cal_levels = np.linspace(0.05, 0.99, 50)

    DATASETS = {
        "NVDA": ("data/nvda_lit_buy_features_train.parquet", "data/nvda_lit_buy_features_test.parquet"),
        "AMD":  ("data/amd_lit_buy_features_train.parquet", "data/amd_lit_buy_features_test.parquet"),
        "AMZN": ("data/amzn_lit_buy_features_train.parquet", "data/amzn_lit_buy_features_test.parquet"),
        "TSLA": ("data/tsla_lit_buy_features_train.parquet", "data/tsla_lit_buy_features_test.parquet"),
    }

    COLORS = {
        "NVDA": "#16a34a", "AMD": "#f59e0b", "AMZN": "#7c3aed", "TSLA": "#ec4899",
    }
    MARKERS = {
        "NVDA": "o", "AMD": "s", "AMZN": "D", "TSLA": "^",
    }

    def laplace_coverage(y, mu, b, level):
        z = np.log(1.0 / (1.0 - level))
        lo = np.maximum(mu - z * b, 0.0)
        hi = mu + z * b
        return ((y >= lo) & (y <= hi)).mean()

    def fit_xgb_gamlss(X_tr, y_tr, X_te):
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
                          random_state=42, n_jobs=1, max_depth=3, n_estimators=50,
                          min_child_weight=20, learning_rate=0.1, reg_alpha=1, reg_lambda=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sc.fit(X_tr, abs_r)
        b_te = np.clip(sc.predict(X_te), 0.1, None)
        return mu_te, b_te

    # -- Load and fit all 4 stocks ------------------------------------------------
    results = {}
    for ticker, (tr_f, te_f) in DATASETS.items():
        print(f"Fitting {ticker}...", flush=True)
        df_tr = pd.read_parquet(tr_f)
        df_te = pd.read_parquet(te_f)
        df_tr = df_tr[df_tr["impact_vwap_bps"].abs() <= IMPACT_CAP_BPS]
        df_te = df_te[df_te["impact_vwap_bps"].abs() <= IMPACT_CAP_BPS]
        df_tr["abs_impact"] = df_tr["impact_vwap_bps"].abs()
        df_te["abs_impact"] = df_te["impact_vwap_bps"].abs()

        X_tr = df_tr[FEATURES_3].to_numpy(dtype=np.float64)
        y_tr = df_tr["abs_impact"].to_numpy(dtype=np.float64)
        X_te = df_te[FEATURES_3].to_numpy(dtype=np.float64)
        y_te = df_te["abs_impact"].to_numpy(dtype=np.float64)

        mu_te, b_te = fit_xgb_gamlss(X_tr, y_tr, X_te)
        cal = np.array([laplace_coverage(y_te, mu_te, b_te, lv) for lv in cal_levels])

        results[ticker] = {
            "cal": cal, "mu_te": mu_te, "b_te": b_te,
            "y_te": y_te, "n_test": len(y_te),
        }
        print(f"  {ticker}: n_test={len(y_te):,}")

    def plot_pair(ticker1, ticker2, filename):
        """Calibration curve for a pair of stocks, same style as calibration_xgb_overlay.png."""
        fig, ax = plt.subplots(figsize=(8, 8))

        c1, c2 = COLORS[ticker1], COLORS[ticker2]
        r1, r2 = results[ticker1], results[ticker2]

        ax.plot(cal_levels, r1["cal"], color=c1, lw=2.5,
                marker=MARKERS[ticker1], markersize=4,
                label=f"{ticker1} ({r1['n_test']:,} test trades)")
        ax.plot(cal_levels, r2["cal"], color=c2, lw=2.5,
                marker=MARKERS[ticker2], markersize=4,
                label=f"{ticker2} ({r2['n_test']:,} test trades)")
        ax.plot([0, 1], [0, 1], color="black", lw=1.2, ls="--", alpha=0.5,
                label="Perfect calibration")

        for ticker, color, yoff in [(ticker1, c1, 12), (ticker2, c2, -16)]:
            r = results[ticker]
            for lv in [0.50, 0.80, 0.90]:
                cov = laplace_coverage(r["y_te"], r["mu_te"], r["b_te"], lv)
                z = np.log(1.0 / (1.0 - lv))
                w = (2 * z * r["b_te"]).mean()
                ax.annotate(f"{lv:.0%}: {cov:.1%} (w={w:.1f})", xy=(lv, cov),
                            textcoords="offset points", xytext=(12, yoff),
                            fontsize=8, color=color, fontweight="bold",
                            arrowprops=dict(arrowstyle="->", color=color, lw=0.8))

        ax.set_xlabel("Nominal coverage level", fontsize=12)
        ax.set_ylabel("Actual coverage", fontsize=12)
        ax.set_title(f"Two-Stage XGBoost Calibration: {ticker1} and {ticker2}",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=10, loc="upper left")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        fig.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"Saved -> {filename}")
        plt.close(fig)

    # -- Figure 1: All 4 stocks on one calibration curve --------------------------
    fig, ax = plt.subplots(figsize=(9, 9))

    for ticker in ["NVDA", "AMD", "AMZN", "TSLA"]:
        r = results[ticker]
        ax.plot(cal_levels, r["cal"], color=COLORS[ticker], lw=2.5,
                marker=MARKERS[ticker], markersize=4, markevery=3,
                label=f"{ticker} ({r['n_test']:,} test trades)")

    ax.plot([0, 1], [0, 1], color="black", lw=1.2, ls="--", alpha=0.5,
            label="Perfect calibration")

    # Vertical reference lines at 50%, 80%, 90% with label on the diagonal
    for lv in [0.50, 0.80, 0.90]:
        ax.axvline(lv, color="gray", lw=0.8, ls=":", alpha=0.4, zorder=1)
        ax.annotate(f"{lv:.0%}", xy=(lv, lv), textcoords="offset points",
                    xytext=(-14, -10), fontsize=9, color="black", fontweight="bold",
                    alpha=0.6)
        # Small tick marks on each curve at these levels
        for ticker in ["NVDA", "AMD", "AMZN", "TSLA"]:
            r = results[ticker]
            cov = laplace_coverage(r["y_te"], r["mu_te"], r["b_te"], lv)
            ax.plot(lv, cov, marker="_", markersize=10, markeredgewidth=2.5,
                    color=COLORS[ticker], zorder=5)

    ax.set_xlabel("Nominal coverage level", fontsize=12)
    ax.set_ylabel("Actual coverage", fontsize=12)
    ax.set_title("Two-Stage XGBoost Calibration: NVDA, AMD, AMZN, TSLA",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig("calibration_xgb_4stocks.png", dpi=150, bbox_inches="tight")
    print("Saved -> calibration_xgb_4stocks.png")
    plt.close(fig)

    # -- Figure 3: Coverage deviation for all 4 stocks ----------------------------
    fig3, ax3 = plt.subplots(figsize=(10, 6))

    ax3.axhline(0, color="black", lw=1.5, ls="--", alpha=0.6, zorder=1)
    ax3.fill_between(cal_levels, -0.05, 0.05, color="orange", alpha=0.05, label="\u00b15% band")

    for ticker in ["NVDA", "AMD", "AMZN", "TSLA"]:
        r = results[ticker]
        deviation = r["cal"] - cal_levels
        ax3.plot(cal_levels, deviation, color=COLORS[ticker], lw=2.5,
                 marker=MARKERS[ticker], markersize=5, markevery=3,
                 label=f"{ticker} (n={r['n_test']:,})", zorder=3)

    ax3.set_xlabel("Nominal Coverage Level", fontsize=13)
    ax3.set_ylabel("Actual \u2212 Nominal Coverage", fontsize=13)
    ax3.set_title("Coverage Deviation: Two-Stage XGBoost (NVDA, AMD, AMZN, TSLA)\n"
                  "Positive = over-coverage (conservative), Negative = under-coverage",
                  fontsize=13, fontweight="bold")
    ax3.set_xlim(0.05, 0.95)
    ax3.set_ylim(-0.12, 0.15)
    ax3.legend(fontsize=10, loc="upper left", framealpha=0.9)
    ax3.grid(True, alpha=0.2)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    plt.tight_layout()
    fig3.savefig("coverage_deviation_4stocks.png", dpi=150, bbox_inches="tight")
    print("Saved -> coverage_deviation_4stocks.png")

    print("\nDone!")


# ══════════════════════════════════════════════════════════════════════════════
# calibration_overlays.py
# ══════════════════════════════════════════════════════════════════════════════
def run_calibration_overlays():
    """Two calibration overlay plots:
      1. Linear GAMLSS: AAPL vs COIN on same axes
      2. XGB GAMLSS: AAPL vs COIN on same axes

    Output:
      - calibration_linear_overlay.png
      - calibration_xgb_overlay.png
    """
    FEATURES = [
        "dollar_value", "log_dollar_value", "participation_rate",
        "roll_spread_500", "roll_vol_500", "exchange_id",
    ]

    cal_levels = np.linspace(0.05, 0.99, 50)

    def laplace_coverage(y, mu, b, level):
        z = np.log(1.0 / (1.0 - level))
        lo = np.maximum(mu - z * b, 0.0)
        hi = mu + z * b
        return ((y >= lo) & (y <= hi)).mean()

    def fit_linear_gamlss(X_tr, y_tr, X_te):
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

    def fit_xgb_gamlss(X_tr, y_tr, X_te):
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

    # -- Load data ----------------------------------------------------------------
    print("Loading data...", flush=True)

    datasets = {
        "AAPL": ("data/lit_buy_features_v2.parquet", "data/lit_buy_features_v2_sep.parquet"),
        "COIN": ("data/coin_lit_buy_features_train.parquet", "data/coin_lit_buy_features_test.parquet"),
    }

    data = {}
    for ticker, (tr_f, te_f) in datasets.items():
        df_tr = pd.read_parquet(tr_f)
        df_te = pd.read_parquet(te_f)
        df_tr["abs_impact"] = df_tr["impact_vwap_bps"].abs()
        df_te["abs_impact"] = df_te["impact_vwap_bps"].abs()
        df_tr = df_tr.sort_values("date").reset_index(drop=True)
        data[ticker] = {
            "X_tr": df_tr[FEATURES].to_numpy(dtype=np.float64),
            "y_tr": df_tr["abs_impact"].to_numpy(dtype=np.float64),
            "X_te": df_te[FEATURES].to_numpy(dtype=np.float64),
            "y_te": df_te["abs_impact"].to_numpy(dtype=np.float64),
        }

    # -- Fit models and compute calibration curves --------------------------------
    results = {}

    for ticker in ["AAPL", "COIN"]:
        d = data[ticker]
        print(f"Fitting {ticker}...", flush=True)

        mu_lin, b_lin = fit_linear_gamlss(d["X_tr"], d["y_tr"], d["X_te"])
        mu_xgb, b_xgb = fit_xgb_gamlss(d["X_tr"], d["y_tr"], d["X_te"])

        cal_lin = [laplace_coverage(d["y_te"], mu_lin, b_lin, lv) for lv in cal_levels]
        cal_xgb = [laplace_coverage(d["y_te"], mu_xgb, b_xgb, lv) for lv in cal_levels]

        results[ticker] = {
            "cal_lin": np.array(cal_lin),
            "cal_xgb": np.array(cal_xgb),
            "mu_lin": mu_lin, "b_lin": b_lin,
            "mu_xgb": mu_xgb, "b_xgb": b_xgb,
            "y_te": d["y_te"],
        }

    AAPL_COLOR = "#2563eb"
    COIN_COLOR = "#dc2626"

    # -- Figure 1: Linear GAMLSS — AAPL vs COIN ----------------------------------
    print("Plotting Linear GAMLSS overlay...", flush=True)

    fig1, ax1 = plt.subplots(figsize=(8, 8))

    ax1.plot(cal_levels, results["AAPL"]["cal_lin"], color=AAPL_COLOR, lw=2.5,
             marker="o", markersize=4, label="AAPL (9,152 test trades)")
    ax1.plot(cal_levels, results["COIN"]["cal_lin"], color=COIN_COLOR, lw=2.5,
             marker="s", markersize=4, label="COIN (959 test trades)")
    ax1.plot([0, 1], [0, 1], color="black", lw=1.2, ls="--", alpha=0.5,
             label="Perfect calibration")

    for ticker, color, yoff in [("AAPL", AAPL_COLOR, 12), ("COIN", COIN_COLOR, -16)]:
        r = results[ticker]
        for lv in [0.50, 0.80, 0.90]:
            cov = laplace_coverage(r["y_te"], r["mu_lin"], r["b_lin"], lv)
            _, width = lv, None  # just need cov
            z = np.log(1.0 / (1.0 - lv))
            w = (2 * z * r["b_lin"]).mean()
            ax1.annotate(f"{lv:.0%}: {cov:.1%} (w={w:.1f})", xy=(lv, cov),
                         textcoords="offset points", xytext=(12, yoff),
                         fontsize=8, color=color, fontweight="bold",
                         arrowprops=dict(arrowstyle="->", color=color, lw=0.8))

    ax1.set_xlabel("Nominal coverage level", fontsize=12)
    ax1.set_ylabel("Actual coverage", fontsize=12)
    ax1.set_title("Two-Stage Linear Regression Calibration: AAPL and COIN",
                  fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10, loc="upper left")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.15)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig("calibration_linear_overlay.png", dpi=150, bbox_inches="tight")
    print("Saved -> calibration_linear_overlay.png")

    # -- Figure 2: XGB GAMLSS — AAPL vs COIN -------------------------------------
    print("Plotting XGB GAMLSS overlay...", flush=True)

    fig2, ax2 = plt.subplots(figsize=(8, 8))

    ax2.plot(cal_levels, results["AAPL"]["cal_xgb"], color=AAPL_COLOR, lw=2.5,
             marker="o", markersize=4, label="AAPL (9,152 test trades)")
    ax2.plot(cal_levels, results["COIN"]["cal_xgb"], color=COIN_COLOR, lw=2.5,
             marker="s", markersize=4, label="COIN (959 test trades)")
    ax2.plot([0, 1], [0, 1], color="black", lw=1.2, ls="--", alpha=0.5,
             label="Perfect calibration")

    for ticker, color, yoff in [("AAPL", AAPL_COLOR, 12), ("COIN", COIN_COLOR, -16)]:
        r = results[ticker]
        for lv in [0.50, 0.80, 0.90]:
            cov = laplace_coverage(r["y_te"], r["mu_xgb"], r["b_xgb"], lv)
            z = np.log(1.0 / (1.0 - lv))
            w = (2 * z * r["b_xgb"]).mean()
            ax2.annotate(f"{lv:.0%}: {cov:.1%} (w={w:.1f})", xy=(lv, cov),
                         textcoords="offset points", xytext=(12, yoff),
                         fontsize=8, color=color, fontweight="bold",
                         arrowprops=dict(arrowstyle="->", color=color, lw=0.8))

    ax2.set_xlabel("Nominal coverage level", fontsize=12)
    ax2.set_ylabel("Actual coverage", fontsize=12)
    ax2.set_title("Two-Stage XGBoost Calibration: AAPL and COIN",
                  fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10, loc="upper left")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.15)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig("calibration_xgb_overlay.png", dpi=150, bbox_inches="tight")
    print("Saved -> calibration_xgb_overlay.png")

    print("\nDone!")


# ══════════════════════════════════════════════════════════════════════════════
# calibration_pooled.py
# ══════════════════════════════════════════════════════════════════════════════
def run_calibration_pooled():
    """Calibration and coverage plots for the pooled XGB GAMLSS (6 stocks).

    Output:
      - calibration_pooled_6stocks.png      All 6 stocks on one calibration curve
      - coverage_deviation_pooled.png       Coverage deviation (actual - nominal)
      - calibration_pooled_vs_perstock.png  Side-by-side: per-stock vs pooled calibration
    """
    FEATURES_3 = ["roll_spread_500", "roll_vol_500", "participation_rate"]
    IMPACT_CAP_BPS = 500
    cal_levels = np.linspace(0.05, 0.99, 50)

    DATASETS = {
        "AAPL": ("data/lit_buy_features_v2.parquet", "data/lit_buy_features_v2_sep.parquet"),
        "COIN": ("data/coin_lit_buy_features_train.parquet", "data/coin_lit_buy_features_test.parquet"),
        "NVDA": ("data/nvda_lit_buy_features_train.parquet", "data/nvda_lit_buy_features_test.parquet"),
        "AMD":  ("data/amd_lit_buy_features_train.parquet", "data/amd_lit_buy_features_test.parquet"),
        "AMZN": ("data/amzn_lit_buy_features_train.parquet", "data/amzn_lit_buy_features_test.parquet"),
        "TSLA": ("data/tsla_lit_buy_features_train.parquet", "data/tsla_lit_buy_features_test.parquet"),
    }

    COLORS = {
        "AAPL": "#2563eb", "COIN": "#dc2626", "NVDA": "#16a34a",
        "AMD": "#f59e0b", "AMZN": "#7c3aed", "TSLA": "#ec4899",
    }
    MARKERS = {
        "AAPL": "o", "COIN": "s", "NVDA": "D", "AMD": "^", "AMZN": "v", "TSLA": "P",
    }

    LOC_PARAMS = dict(max_depth=3, n_estimators=200, learning_rate=0.07,
                      min_child_weight=5, reg_alpha=10, reg_lambda=10)
    SCALE_PARAMS = dict(max_depth=3, n_estimators=50, learning_rate=0.1,
                        min_child_weight=20, reg_alpha=1, reg_lambda=1)

    def laplace_coverage(y, mu, b, level):
        z = np.log(1.0 / (1.0 - level))
        lo = np.maximum(mu - z * b, 0.0)
        hi = mu + z * b
        return ((y >= lo) & (y <= hi)).mean()

    # ═════════════════════════════════════════════════════════════════════════════
    # Load data and build pooled training set
    # ═════════════════════════════════════════════════════════════════════════════
    print("Loading data and normalizing...", flush=True)

    stock_data = {}
    pooled_X, pooled_y = [], []

    for ticker, (tr_f, te_f) in DATASETS.items():
        df_tr = pd.read_parquet(tr_f)
        df_te = pd.read_parquet(te_f)
        df_tr = df_tr[df_tr["impact_vwap_bps"].abs() <= IMPACT_CAP_BPS]
        df_te = df_te[df_te["impact_vwap_bps"].abs() <= IMPACT_CAP_BPS]
        df_tr["abs_impact"] = df_tr["impact_vwap_bps"].abs()
        df_te["abs_impact"] = df_te["impact_vwap_bps"].abs()

        medians = {}
        for feat in FEATURES_3:
            medians[feat] = df_tr[feat].median()
            if medians[feat] == 0 or np.isnan(medians[feat]):
                medians[feat] = df_tr[feat].mean()
                if medians[feat] == 0:
                    medians[feat] = 1.0

        median_impact = df_tr["abs_impact"].median()
        if median_impact == 0 or np.isnan(median_impact):
            median_impact = df_tr["abs_impact"].mean()

        X_tr_norm = np.column_stack([df_tr[feat].values / medians[feat] for feat in FEATURES_3])
        y_tr_norm = df_tr["abs_impact"].values / median_impact
        X_te_norm = np.column_stack([df_te[feat].values / medians[feat] for feat in FEATURES_3])

        stock_data[ticker] = {
            "medians": medians, "median_impact": median_impact,
            "X_tr_raw": df_tr[FEATURES_3].to_numpy(dtype=np.float64),
            "y_tr_raw": df_tr["abs_impact"].to_numpy(dtype=np.float64),
            "X_te_raw": df_te[FEATURES_3].to_numpy(dtype=np.float64),
            "y_te_raw": df_te["abs_impact"].to_numpy(dtype=np.float64),
            "X_tr_norm": X_tr_norm, "y_tr_norm": y_tr_norm,
            "X_te_norm": X_te_norm,
            "n_test": len(df_te),
        }
        pooled_X.append(X_tr_norm)
        pooled_y.append(y_tr_norm)

    pooled_X = np.vstack(pooled_X)
    pooled_y = np.concatenate(pooled_y)
    print(f"  Pooled training: {len(pooled_y):,} trades")

    # ═════════════════════════════════════════════════════════════════════════════
    # Fit pooled model
    # ═════════════════════════════════════════════════════════════════════════════
    print("Fitting pooled location + scale models...", flush=True)

    loc_model = XGBRegressor(objective="reg:absoluteerror", tree_method="hist",
                             verbosity=0, random_state=42, n_jobs=1, **LOC_PARAMS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loc_model.fit(pooled_X, pooled_y)

    mu_pooled_tr = np.maximum(loc_model.predict(pooled_X), 0.0)
    abs_resid = np.abs(pooled_y - mu_pooled_tr)

    scale_model = XGBRegressor(objective="reg:squarederror", tree_method="hist",
                               verbosity=0, random_state=42, n_jobs=1, **SCALE_PARAMS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scale_model.fit(pooled_X, abs_resid)

    # ═════════════════════════════════════════════════════════════════════════════
    # Fit per-stock models and compute calibration for both
    # ═════════════════════════════════════════════════════════════════════════════
    print("Computing calibration curves...", flush=True)

    results_pooled = {}
    results_perstock = {}

    for ticker in DATASETS:
        d = stock_data[ticker]
        y_te = d["y_te_raw"]
        scale = d["median_impact"]

        # --- Pooled predictions ---
        mu_pool_norm = np.maximum(loc_model.predict(d["X_te_norm"]), 0.0)
        b_pool_norm = np.clip(scale_model.predict(d["X_te_norm"]), 0.01, None)
        mu_pool = mu_pool_norm * scale
        b_pool = np.clip(b_pool_norm * scale, 0.1, None)

        cal_pool = np.array([laplace_coverage(y_te, mu_pool, b_pool, lv) for lv in cal_levels])
        results_pooled[ticker] = {
            "cal": cal_pool, "mu_te": mu_pool, "b_te": b_pool,
            "y_te": y_te, "n_test": len(y_te),
        }

        # --- Per-stock predictions ---
        loc_ps = XGBRegressor(objective="reg:absoluteerror", tree_method="hist",
                              verbosity=0, random_state=42, n_jobs=1, **LOC_PARAMS)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loc_ps.fit(d["X_tr_raw"], d["y_tr_raw"])
        mu_tr_ps = np.maximum(loc_ps.predict(d["X_tr_raw"]), 0.0)
        mu_te_ps = np.maximum(loc_ps.predict(d["X_te_raw"]), 0.0)

        abs_r_ps = np.abs(d["y_tr_raw"] - mu_tr_ps)
        sc_ps = XGBRegressor(objective="reg:squarederror", tree_method="hist",
                             verbosity=0, random_state=42, n_jobs=1, **SCALE_PARAMS)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sc_ps.fit(d["X_tr_raw"], abs_r_ps)
        b_te_ps = np.clip(sc_ps.predict(d["X_te_raw"]), 0.1, None)

        cal_ps = np.array([laplace_coverage(y_te, mu_te_ps, b_te_ps, lv) for lv in cal_levels])
        results_perstock[ticker] = {
            "cal": cal_ps, "mu_te": mu_te_ps, "b_te": b_te_ps,
            "y_te": y_te, "n_test": len(y_te),
        }

        print(f"  {ticker}: 90% pooled={cal_pool[cal_levels >= 0.89][0]:.3f}, "
              f"per-stock={cal_ps[cal_levels >= 0.89][0]:.3f}")

    TICKERS = list(DATASETS.keys())

    # ═════════════════════════════════════════════════════════════════════════════
    # Figure 1: Pooled calibration — all 6 stocks
    # ═════════════════════════════════════════════════════════════════════════════
    print("\nPlotting pooled calibration curve...", flush=True)

    fig1, ax1 = plt.subplots(figsize=(9, 9))

    for ticker in TICKERS:
        r = results_pooled[ticker]
        ax1.plot(cal_levels, r["cal"], color=COLORS[ticker], lw=2.5,
                 marker=MARKERS[ticker], markersize=4, markevery=3,
                 label=f"{ticker} ({r['n_test']:,} test trades)")

    ax1.plot([0, 1], [0, 1], color="black", lw=1.2, ls="--", alpha=0.5,
             label="Perfect calibration")

    for lv in [0.50, 0.80, 0.90]:
        ax1.axvline(lv, color="gray", lw=0.8, ls=":", alpha=0.4, zorder=1)
        ax1.annotate(f"{lv:.0%}", xy=(lv, lv), textcoords="offset points",
                     xytext=(-14, -10), fontsize=9, color="black", fontweight="bold",
                     alpha=0.6)
        for ticker in TICKERS:
            r = results_pooled[ticker]
            cov = laplace_coverage(r["y_te"], r["mu_te"], r["b_te"], lv)
            ax1.plot(lv, cov, marker="_", markersize=10, markeredgewidth=2.5,
                     color=COLORS[ticker], zorder=5)

    ax1.set_xlabel("Nominal coverage level", fontsize=12)
    ax1.set_ylabel("Actual coverage", fontsize=12)
    ax1.set_title("Pooled XGB GAMLSS Calibration (6 stocks)",
                  fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9, loc="upper left")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.15)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    plt.tight_layout()
    fig1.savefig("calibration_pooled_6stocks.png", dpi=150, bbox_inches="tight")
    print("Saved -> calibration_pooled_6stocks.png")
    plt.close(fig1)

    # ═════════════════════════════════════════════════════════════════════════════
    # Figure 2: Coverage deviation — pooled model
    # ═════════════════════════════════════════════════════════════════════════════
    print("Plotting coverage deviation...", flush=True)

    fig2, ax2 = plt.subplots(figsize=(10, 6))

    ax2.axhline(0, color="black", lw=1.5, ls="--", alpha=0.6, zorder=1)
    ax2.fill_between(cal_levels, -0.05, 0.05, color="orange", alpha=0.05, label="\u00b15% band")

    for ticker in TICKERS:
        r = results_pooled[ticker]
        deviation = r["cal"] - cal_levels
        ax2.plot(cal_levels, deviation, color=COLORS[ticker], lw=2.5,
                 marker=MARKERS[ticker], markersize=5, markevery=3,
                 label=f"{ticker} (n={r['n_test']:,})", zorder=3)

    ax2.set_xlabel("Nominal Coverage Level", fontsize=13)
    ax2.set_ylabel("Actual \u2212 Nominal Coverage", fontsize=13)
    ax2.set_title("Coverage Deviation: Pooled XGB GAMLSS (6 stocks)\n"
                  "Positive = over-coverage (conservative), Negative = under-coverage",
                  fontsize=13, fontweight="bold")
    ax2.set_xlim(0.05, 0.95)
    ax2.set_ylim(-0.12, 0.15)
    ax2.legend(fontsize=9, loc="upper left", framealpha=0.9)
    ax2.grid(True, alpha=0.2)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    fig2.savefig("coverage_deviation_pooled.png", dpi=150, bbox_inches="tight")
    print("Saved -> coverage_deviation_pooled.png")
    plt.close(fig2)

    # ═════════════════════════════════════════════════════════════════════════════
    # Figure 3: Per-stock vs Pooled calibration — side by side
    # ═════════════════════════════════════════════════════════════════════════════
    print("Plotting per-stock vs pooled comparison...", flush=True)

    fig3, (ax_ps, ax_pl) = plt.subplots(1, 2, figsize=(18, 8.5))

    for ax, res, title in [
        (ax_ps, results_perstock, "Per-Stock XGB GAMLSS"),
        (ax_pl, results_pooled, "Pooled XGB GAMLSS (6 stocks)"),
    ]:

        for ticker in TICKERS:
            r = res[ticker]
            ax.plot(cal_levels, r["cal"], color=COLORS[ticker], lw=2.5,
                    marker=MARKERS[ticker], markersize=4, markevery=3,
                    label=f"{ticker} ({r['n_test']:,})")

        ax.plot([0, 1], [0, 1], color="black", lw=1.2, ls="--", alpha=0.5,
                label="Perfect")

        for lv in [0.50, 0.80, 0.90]:
            ax.axvline(lv, color="gray", lw=0.8, ls=":", alpha=0.4, zorder=1)
            for ticker in TICKERS:
                r = res[ticker]
                cov = laplace_coverage(r["y_te"], r["mu_te"], r["b_te"], lv)
                ax.plot(lv, cov, marker="_", markersize=8, markeredgewidth=2,
                        color=COLORS[ticker], zorder=5)

        ax.set_xlabel("Nominal coverage level", fontsize=11)
        ax.set_ylabel("Actual coverage", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(fontsize=8, loc="upper left", ncol=2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig3.suptitle("Calibration Comparison: Per-Stock vs Pooled Model",
                  fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig3.savefig("calibration_pooled_vs_perstock.png", dpi=150, bbox_inches="tight")
    print("Saved -> calibration_pooled_vs_perstock.png")
    plt.close(fig3)

    print("\nDone!")


if __name__ == "__main__":
    print("=" * 60)
    print("Calibration curves (XGB GAMLSS, 6 stocks)")
    print("=" * 60)
    run_calibration_curves()

    print("\n" + "=" * 60)
    print("Naive Gaussian baseline calibration")
    print("=" * 60)
    run_calibration_naive()

    print("\n" + "=" * 60)
    print("OLS Gaussian baseline calibration")
    print("=" * 60)
    run_calibration_naive_ols()

    print("\n" + "=" * 60)
    print("New stocks calibration (NVDA, AMD, AMZN, TSLA)")
    print("=" * 60)
    run_calibration_new_stocks()

    print("\n" + "=" * 60)
    print("Calibration overlays (Linear vs XGB, AAPL vs COIN)")
    print("=" * 60)
    run_calibration_overlays()

    print("\n" + "=" * 60)
    print("Pooled XGB GAMLSS calibration")
    print("=" * 60)
    run_calibration_pooled()
