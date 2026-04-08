"""
Full GAMLSS (Rigby & Stasinopoulos 2005) for Laplace distribution.

For each trade i:  |impact_i| ~ Laplace(mu_i, b_i)

Additive predictors with P-splines:
  mu_i = a0 + sum_j f_j(x_ij)         [identity link]
  log(b_i) = g0 + sum_j g_j(x_ij)     [log link]

RS algorithm:
  1. Initialize mu, b
  2. Update mu: PIRLS with Laplace Fisher scoring for location
  3. Update b: PIRLS with Laplace Fisher scoring for scale
  4. Monitor global deviance, converge when relative change < tol

Output:
  - gamlss_full_results.png
  - Console comparison table vs Two-Stage XGBoost
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
from scipy.sparse import issparse

FEATURES = [
    "dollar_value", "log_dollar_value", "participation_rate",
    "roll_spread_500", "roll_vol_500", "exchange_id",
]


# ── P-Spline Design Matrix ───────────────────────────────────────────────────

class GAMLSSDesign:
    """Design matrix: intercept + P-spline basis for each feature."""

    def __init__(self, n_basis=12, degree=3):
        self.n_basis = n_basis
        self.degree = degree
        self.knot_sequences = []
        self.n_basis_per_feat = []

    def fit_transform(self, X):
        n, p = X.shape
        blocks = [np.ones((n, 1))]
        self.knot_sequences = []
        self.n_basis_per_feat = []

        for j in range(p):
            xj = X[:, j]
            n_internal = self.n_basis - self.degree + 1
            internal = np.percentile(xj, np.linspace(0, 100, n_internal + 2)[1:-1])
            internal = np.unique(internal)
            xmin, xmax = xj.min() - 1e-6, xj.max() + 1e-6
            knots = np.concatenate([
                np.repeat(xmin, self.degree + 1),
                internal,
                np.repeat(xmax, self.degree + 1),
            ])
            self.knot_sequences.append(knots)
            B = BSpline.design_matrix(xj, knots, self.degree)
            if issparse(B):
                B = B.toarray()
            self.n_basis_per_feat.append(B.shape[1])
            blocks.append(B)

        return np.hstack(blocks)

    def transform(self, X):
        n, p = X.shape
        blocks = [np.ones((n, 1))]
        for j in range(p):
            knots = self.knot_sequences[j]
            xj = np.clip(X[:, j], knots[0], knots[-1])
            B = BSpline.design_matrix(xj, knots, self.degree)
            if issparse(B):
                B = B.toarray()
            blocks.append(B)
        return np.hstack(blocks)

    def penalty_matrix(self, order=2):
        total = 1 + sum(self.n_basis_per_feat)
        P = np.zeros((total, total))
        offset = 1  # skip intercept
        for nb in self.n_basis_per_feat:
            if nb > order:
                D = np.diff(np.eye(nb), n=order, axis=0)
                P[offset:offset + nb, offset:offset + nb] = D.T @ D
            offset += nb
        return P


# ── GAMLSS fitting ───────────────────────────────────────────────────────────

def laplace_deviance(y, mu, b):
    """Global deviance = -2 * log-likelihood."""
    return 2.0 * np.sum(np.log(2.0 * b) + np.abs(y - mu) / b)


def pwls_step(B, z, w, P, lam):
    """One penalized weighted least squares solve."""
    WB = w[:, None] * B
    A = B.T @ WB + lam * P + 1e-8 * np.eye(B.shape[1])
    rhs = WB.T @ z
    return np.linalg.solve(A, rhs)


def fit_gamlss_laplace(X_tr, y_tr, X_te, n_basis=12, lam_mu=10.0, lam_b=10.0,
                        max_iter=50, tol=1e-6, verbose=True):
    """
    Fit full GAMLSS for Laplace(mu, b) using RS algorithm with P-splines.
    """
    # Build design matrices (separate bases for mu and b)
    des_mu = GAMLSSDesign(n_basis=n_basis)
    des_b = GAMLSSDesign(n_basis=n_basis)

    B_mu_tr = des_mu.fit_transform(X_tr)
    B_mu_te = des_mu.transform(X_te)
    B_b_tr = des_b.fit_transform(X_tr)
    B_b_te = des_b.transform(X_te)

    P_mu = des_mu.penalty_matrix()
    P_b = des_b.penalty_matrix()

    n = len(y_tr)

    # Initialize: median for location, mean |residual| for scale
    mu = np.full(n, np.median(y_tr))
    b = np.full(n, np.mean(np.abs(y_tr - mu)))
    b = np.clip(b, 0.01, None)

    dev = laplace_deviance(y_tr, mu, b)
    dev_history = [dev]
    if verbose:
        print(f"    Init : deviance = {dev:,.2f}")

    beta_mu = None
    beta_b = None

    for it in range(1, max_iter + 1):
        # ── Update mu (identity link) ──
        # Laplace score for mu: sign(y - mu) / b
        # Fisher weight: 1 / b^2
        # Working variable: z_mu = mu + score / weight = mu + sign(y - mu) * b
        sign_r = np.sign(y_tr - mu)
        sign_r[sign_r == 0] = 0.001
        z_mu = mu + sign_r * b
        w_mu = 1.0 / (b ** 2)

        beta_mu_new = pwls_step(B_mu_tr, z_mu, w_mu, P_mu, lam_mu)
        mu_new = B_mu_tr @ beta_mu_new
        mu_new = np.maximum(mu_new, 0.0)

        # Step-halving for mu
        dev_after_mu = laplace_deviance(y_tr, mu_new, b)
        step = 1.0
        for _ in range(5):
            if dev_after_mu <= dev + 1.0:  # allow tiny increase
                break
            step *= 0.5
            mu_cand = (1 - step) * mu + step * mu_new
            dev_after_mu = laplace_deviance(y_tr, mu_cand, b)
            mu_new = mu_cand

        mu = mu_new
        beta_mu = beta_mu_new

        # ── Update b (log link: eta_b = log(b)) ──
        # Laplace score for log(b): -1 + |y - mu| / b
        # Fisher weight: 1
        # Working variable: z_b = log(b) + (-1 + |y - mu| / b)
        abs_r = np.abs(y_tr - mu)
        eta_b = np.log(np.maximum(b, 1e-6))
        z_b = eta_b + (-1.0 + abs_r / b)
        w_b = np.ones(n)

        beta_b_new = pwls_step(B_b_tr, z_b, w_b, P_b, lam_b)
        eta_b_new = B_b_tr @ beta_b_new
        b_new = np.clip(np.exp(eta_b_new), 0.01, 500.0)

        # Step-halving for b
        dev_after_b = laplace_deviance(y_tr, mu, b_new)
        step = 1.0
        for _ in range(5):
            if dev_after_b <= dev + 1.0:
                break
            step *= 0.5
            b_cand = np.exp((1 - step) * np.log(b) + step * np.log(b_new))
            b_cand = np.clip(b_cand, 0.01, 500.0)
            dev_after_b = laplace_deviance(y_tr, mu, b_cand)
            b_new = b_cand

        b = b_new
        beta_b = beta_b_new

        dev_new = laplace_deviance(y_tr, mu, b)
        dev_history.append(dev_new)
        rel_change = abs(dev_new - dev) / (abs(dev) + 1e-10)

        if verbose:
            print(f"    Iter {it:2d}: deviance = {dev_new:,.2f}  "
                  f"(rel delta = {rel_change:.8f})")

        if rel_change < tol:
            if verbose:
                print(f"    Converged at iteration {it}")
            break
        dev = dev_new

    # Predict on test
    mu_te = np.maximum(B_mu_te @ beta_mu, 0.0)
    b_te = np.clip(np.exp(B_b_te @ beta_b), 0.01, 500.0)

    return mu, b, mu_te, b_te, dev_history


def compute_coverage(y, mu, b, level):
    z = np.log(1.0 / (1.0 - level))
    lo = np.maximum(mu - z * b, 0.0)
    hi = mu + z * b
    cov = ((y >= lo) & (y <= hi)).mean()
    width = (hi - lo).mean()
    return cov, width


# ── Main ─────────────────────────────────────────────────────────────────────

TICKERS = [
    ("AAPL", "data/lit_buy_features_v2.parquet", "data/lit_buy_features_v2_sep.parquet"),
    ("COIN", "data/coin_lit_buy_features_train.parquet", "data/coin_lit_buy_features_test.parquet"),
]

results = {}

for ticker, tr_file, te_file in TICKERS:
    print(f"\n{'=' * 70}")
    print(f"  {ticker}: Full GAMLSS (RS algorithm + P-splines, Laplace)")
    print(f"{'=' * 70}")

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

    mu_tr, b_tr, mu_te, b_te, dev_hist = fit_gamlss_laplace(
        X_tr, y_tr, X_te, n_basis=12, lam_mu=10.0, lam_b=10.0,
        max_iter=50, tol=1e-6, verbose=True)

    mae_te = np.mean(np.abs(y_te - mu_te))
    ll_train = -0.5 * laplace_deviance(y_tr, mu_tr, b_tr)
    print(f"\n  Test MAE: {mae_te:.4f}")
    print(f"  Final train log-lik: {ll_train:,.2f}")

    cov_data = {}
    print(f"  {'Level':>8} {'Coverage':>10} {'Width':>10}")
    for level in [0.50, 0.80, 0.90]:
        cov, width = compute_coverage(y_te, mu_te, b_te, level)
        cov_data[level] = (cov, width)
        print(f"  {level:>8.0%} {cov:>10.4f} {width:>10.4f}")

    results[ticker] = {
        "mae_te": mae_te,
        "coverage_data": cov_data,
        "dev_history": dev_hist,
        "y_te": y_te, "mu_te": mu_te, "b_te": b_te,
    }


# ── Comparison table ─────────────────────────────────────────────────────────
print(f"\n{'=' * 90}")
print("  COMPARISON: Full GAMLSS vs Two-Stage XGBoost (RS)")
print(f"{'=' * 90}")
print(f"  {'Model':<35} {'MAE':>8} {'90% Cov':>8} {'80% Cov':>8} "
      f"{'50% Cov':>8} {'90% Width':>10}")
print(f"  {'-' * 35} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 10}")

try:
    xgb_df = pd.read_csv("data/gamlss_xgb_results.csv")
    has_xgb = True
except FileNotFoundError:
    has_xgb = False

for ticker in ["AAPL", "COIN"]:
    r = results[ticker]
    cd = r["coverage_data"]
    print(f"  {ticker + ' Full GAMLSS':<35} {r['mae_te']:>8.4f} {cd[0.90][0]:>8.4f} "
          f"{cd[0.80][0]:>8.4f} {cd[0.50][0]:>8.4f} {cd[0.90][1]:>10.4f}")

    if has_xgb:
        rs_rows = xgb_df[xgb_df["model"] == "XGB_GAMLSS_RS"]
        if len(rs_rows) == 0:
            rs_rows = xgb_df
        xr90 = rs_rows[(rs_rows["ticker"] == ticker) & (rs_rows["level"] == 0.90)]
        xr80 = rs_rows[(rs_rows["ticker"] == ticker) & (rs_rows["level"] == 0.80)]
        xr50 = rs_rows[(rs_rows["ticker"] == ticker) & (rs_rows["level"] == 0.50)]
        if len(xr90) > 0:
            print(f"  {ticker + ' Two-Stage XGBoost (RS)':<35} "
                  f"{xr90['test_mae'].values[0]:>8.4f} "
                  f"{xr90['actual_coverage'].values[0]:>8.4f} "
                  f"{xr80['actual_coverage'].values[0]:>8.4f} "
                  f"{xr50['actual_coverage'].values[0]:>8.4f} "
                  f"{xr90['mean_interval_width'].values[0]:>10.4f}")

print(f"\n  Nominal targets:  90% -> 0.9000   80% -> 0.8000   50% -> 0.5000")


# ── Fit Two-Stage XGB inline for calibration curve overlay ───────────────────
import warnings
from xgboost import XGBRegressor

LOC_PARAMS = dict(max_depth=3, n_estimators=200, learning_rate=0.07,
                  min_child_weight=5, reg_alpha=10, reg_lambda=10)

xgb_results = {}
for ticker, tr_file, te_file in TICKERS:
    df_tr = pd.read_parquet(tr_file)
    df_te = pd.read_parquet(te_file)
    df_tr["abs_impact"] = df_tr["impact_vwap_bps"].abs()
    df_te["abs_impact"] = df_te["impact_vwap_bps"].abs()
    df_tr = df_tr.sort_values("date").reset_index(drop=True)
    X_tr = df_tr[FEATURES].to_numpy(dtype=np.float64)
    y_tr = df_tr["abs_impact"].to_numpy(dtype=np.float64)
    X_te = df_te[FEATURES].to_numpy(dtype=np.float64)
    y_te = df_te["abs_impact"].to_numpy(dtype=np.float64)

    loc = XGBRegressor(objective="reg:absoluteerror", tree_method="hist",
                       verbosity=0, random_state=42, n_jobs=1, **LOC_PARAMS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loc.fit(X_tr, y_tr)
    mu_tr_x = np.maximum(loc.predict(X_tr), 0.0)
    mu_te_x = np.maximum(loc.predict(X_te), 0.0)
    sc = XGBRegressor(objective="reg:squarederror", tree_method="hist", verbosity=0,
                      random_state=42, n_jobs=1, max_depth=5, n_estimators=50,
                      min_child_weight=20, learning_rate=0.1, reg_alpha=1, reg_lambda=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc.fit(X_tr, np.abs(y_tr - mu_tr_x))
    b_te_x = np.clip(sc.predict(X_te), 0.1, None)
    xgb_results[ticker] = {"mu_te": mu_te_x, "b_te": b_te_x, "y_te": y_te}
print("Two-Stage XGB fitted for overlay comparison.")


# ── Plot ─────────────────────────────────────────────────────────────────────
AAPL_COLOR = "#2563eb"
COIN_COLOR = "#dc2626"
cal_levels = np.linspace(0.05, 0.99, 50)

fig = plt.figure(figsize=(20, 16))
gs = fig.add_gridspec(2, 2, wspace=0.30, hspace=0.35)

# ── Panel 1 (top-left): Deviance convergence ─────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
for ticker, color, marker in [("AAPL", AAPL_COLOR, "o"), ("COIN", COIN_COLOR, "s")]:
    dh = results[ticker]["dev_history"]
    # Normalize: show % reduction from init
    dh_pct = [(dh[0] - d) / dh[0] * 100 for d in dh]
    ax1.plot(range(len(dh)), dh_pct, color=color, lw=2.2, marker=marker,
             markersize=5, markevery=2, label=ticker)

ax1.set_xlabel("RS Iteration", fontsize=12)
ax1.set_ylabel("Deviance Reduction from Init (%)", fontsize=12)
ax1.set_title("RS Algorithm Convergence", fontsize=13, fontweight="bold")
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.2)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# Inset: raw deviance (small)
ax1_in = ax1.inset_axes([0.50, 0.25, 0.45, 0.40])
for ticker, color, marker in [("AAPL", AAPL_COLOR, "o"), ("COIN", COIN_COLOR, "s")]:
    dh = results[ticker]["dev_history"]
    ax1_in.plot(range(len(dh)), dh, color=color, lw=1.5, marker=marker, markersize=3)
ax1_in.set_ylabel("Deviance", fontsize=7)
ax1_in.set_xlabel("Iter", fontsize=7)
ax1_in.tick_params(labelsize=7)
ax1_in.grid(True, alpha=0.15)
ax1_in.set_title("Raw deviance", fontsize=7)

# ── Panel 2 (top-right): Calibration curves — GAMLSS vs XGB ─────────────────
ax2 = fig.add_subplot(gs[0, 1])

ax2.fill_between([0, 1], [0, 1], [0.05, 1.05], color="gray", alpha=0.06)
ax2.fill_between([0, 1], [-0.05, 0.95], [0, 1], color="gray", alpha=0.06)

for ticker, color, marker in [("AAPL", AAPL_COLOR, "o"), ("COIN", COIN_COLOR, "s")]:
    # GAMLSS (solid)
    r = results[ticker]
    cal_g = [compute_coverage(r["y_te"], r["mu_te"], r["b_te"], lv)[0] for lv in cal_levels]
    ax2.plot(cal_levels, cal_g, color=color, lw=2.5, marker=marker, markersize=4,
             markevery=3, label=f"{ticker} GAMLSS", zorder=3)

    # XGB (dashed)
    rx = xgb_results[ticker]
    cal_x = [compute_coverage(rx["y_te"], rx["mu_te"], rx["b_te"], lv)[0] for lv in cal_levels]
    ax2.plot(cal_levels, cal_x, color=color, lw=1.8, ls="--", alpha=0.6,
             label=f"{ticker} Two-Stage XGB", zorder=2)

ax2.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.4, label="Perfect")
ax2.set_xlabel("Nominal Coverage", fontsize=12)
ax2.set_ylabel("Actual Coverage", fontsize=12)
ax2.set_title("Calibration: Full GAMLSS (solid) vs Two-Stage XGB (dashed)",
              fontsize=13, fontweight="bold")
ax2.legend(fontsize=9, loc="upper left")
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_aspect("equal")
ax2.grid(True, alpha=0.15)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

# ── Panel 3 (bottom-left): Coverage deviation ────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])

ax3.axhline(0, color="black", lw=1.2, ls="--", alpha=0.5, zorder=1)
ax3.fill_between(cal_levels, -0.02, 0.02, color="green", alpha=0.08, label="\u00b12% band")
ax3.fill_between(cal_levels, -0.05, 0.05, color="orange", alpha=0.05, label="\u00b15% band")

for ticker, color, marker in [("AAPL", AAPL_COLOR, "o"), ("COIN", COIN_COLOR, "s")]:
    # GAMLSS
    r = results[ticker]
    cal_g = np.array([compute_coverage(r["y_te"], r["mu_te"], r["b_te"], lv)[0]
                      for lv in cal_levels])
    ax3.plot(cal_levels, cal_g - cal_levels, color=color, lw=2.2, marker=marker,
             markersize=4, markevery=3, label=f"{ticker} GAMLSS", zorder=3)

    # XGB
    rx = xgb_results[ticker]
    cal_x = np.array([compute_coverage(rx["y_te"], rx["mu_te"], rx["b_te"], lv)[0]
                      for lv in cal_levels])
    ax3.plot(cal_levels, cal_x - cal_levels, color=color, lw=1.8, ls="--", alpha=0.6,
             label=f"{ticker} XGB", zorder=2)

ax3.set_xlabel("Nominal Coverage Level", fontsize=12)
ax3.set_ylabel("Actual \u2212 Nominal", fontsize=12)
ax3.set_title("Coverage Deviation: GAMLSS (solid) vs XGB (dashed)",
              fontsize=13, fontweight="bold")
ax3.legend(fontsize=8, loc="upper left", ncol=2)
ax3.set_xlim(0.05, 0.95)
ax3.set_ylim(-0.12, 0.15)
ax3.grid(True, alpha=0.2)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)

# ── Panel 4 (bottom-right): Summary comparison table as chart ────────────────
ax4 = fig.add_subplot(gs[1, 1])
ax4.axis("off")

# Build table data
col_labels = ["MAE", "50% Cov", "80% Cov", "90% Cov", "90% Width"]
row_labels = []
cell_data = []
cell_colors = []

for ticker in ["AAPL", "COIN"]:
    base_color = AAPL_COLOR if ticker == "AAPL" else COIN_COLOR

    # GAMLSS row
    r = results[ticker]
    cd = r["coverage_data"]
    row_labels.append(f"{ticker} Full GAMLSS")
    cell_data.append([f"{r['mae_te']:.4f}", f"{cd[0.50][0]:.4f}", f"{cd[0.80][0]:.4f}",
                      f"{cd[0.90][0]:.4f}", f"{cd[0.90][1]:.2f}"])
    cell_colors.append([base_color + "30"] * 5)

    # XGB row
    rx = xgb_results[ticker]
    mae_x = np.mean(np.abs(rx["y_te"] - rx["mu_te"]))
    cov90_x, w90_x = compute_coverage(rx["y_te"], rx["mu_te"], rx["b_te"], 0.90)
    cov80_x, _ = compute_coverage(rx["y_te"], rx["mu_te"], rx["b_te"], 0.80)
    cov50_x, _ = compute_coverage(rx["y_te"], rx["mu_te"], rx["b_te"], 0.50)
    row_labels.append(f"{ticker} Two-Stage XGB")
    cell_data.append([f"{mae_x:.4f}", f"{cov50_x:.4f}", f"{cov80_x:.4f}",
                      f"{cov90_x:.4f}", f"{w90_x:.2f}"])
    cell_colors.append([base_color + "15"] * 5)

    # Nominal row
    row_labels.append("Nominal target")
    cell_data.append(["", "0.5000", "0.8000", "0.9000", ""])
    cell_colors.append(["#f0f0f0"] * 5)

table = ax4.table(cellText=cell_data, rowLabels=row_labels, colLabels=col_labels,
                  cellColours=cell_colors, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.0, 1.8)

# Bold header
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(fontweight="bold")
    if col == -1:
        cell.set_text_props(fontweight="bold", fontsize=9)
    cell.set_edgecolor("#cccccc")

ax4.set_title("Summary Comparison", fontsize=13, fontweight="bold", pad=20)

fig.suptitle("Full GAMLSS (Rigby & Stasinopoulos 2005) vs Two-Stage XGBoost\n"
             "Laplace Distribution | P-Spline Smooth Terms | RS Algorithm | 6 Features",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
fig.savefig("gamlss_full_results.png", dpi=150, bbox_inches="tight")
print("\nSaved -> gamlss_full_results.png")
plt.close(fig)


# ── Fan Plot ─────────────────────────────────────────────────────────────────
print("Plotting fan charts...")

rng = np.random.default_rng(42)
n_show = 300
SMOOTH_WIN = 15  # rolling window for smoothing fan edges

BAND_COLORS_AAPL = {"90%": "#93c5fd", "80%": "#60a5fa", "50%": "#3b82f6"}
BAND_COLORS_COIN = {"90%": "#fca5a5", "80%": "#f87171", "50%": "#ef4444"}


def smooth(arr, win=SMOOTH_WIN):
    """Centered rolling mean, preserving array length."""
    kernel = np.ones(win) / win
    padded = np.pad(arr, win // 2, mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(arr)]


fig_fan, axes_fan = plt.subplots(2, 2, figsize=(20, 14))

for row, (ticker, color, bands) in enumerate([
    ("AAPL", AAPL_COLOR, BAND_COLORS_AAPL),
    ("COIN", COIN_COLOR, BAND_COLORS_COIN),
]):
    r = results[ticker]
    y_te = r["y_te"]
    mu_te = r["mu_te"]
    b_te = r["b_te"]

    rx = xgb_results[ticker]
    mu_te_x = rx["mu_te"]
    b_te_x = rx["b_te"]

    n_trades = min(n_show, len(y_te))
    idx = rng.choice(len(y_te), size=n_trades, replace=False)

    # ── Left column: Full GAMLSS ──
    ax_g = axes_fan[row, 0]
    sort_g = np.argsort(mu_te[idx])
    idx_g = idx[sort_g]
    x_pos = np.arange(n_trades)

    mu_s = mu_te[idx_g]
    b_s = b_te[idx_g]
    y_s = y_te[idx_g]

    # Smooth mu and b for the fan edges, keep raw for coverage calc
    mu_sm = smooth(mu_s)
    b_sm = smooth(b_s)

    for level, band_color, label in [
        (0.90, bands["90%"], "90% interval"),
        (0.80, bands["80%"], "80% interval"),
        (0.50, bands["50%"], "50% interval"),
    ]:
        z = np.log(1.0 / (1.0 - level))
        lo = np.maximum(mu_sm - z * b_sm, 0.0)
        hi = mu_sm + z * b_sm
        ax_g.fill_between(x_pos, lo, hi, alpha=0.7, color=band_color, label=label)

    ax_g.plot(x_pos, mu_sm, color="white", lw=1.8, label="Predicted median", zorder=3)
    ax_g.plot(x_pos, mu_sm, color=color, lw=1.2, zorder=3)
    ax_g.scatter(x_pos, y_s, s=10, color="black", alpha=0.5, zorder=4, label="Actual |impact|")

    # Coverage computed on raw (unsmoothed) predictions
    cov90, _ = compute_coverage(y_s, mu_s, b_s, 0.90)
    cov80, _ = compute_coverage(y_s, mu_s, b_s, 0.80)
    cov50, _ = compute_coverage(y_s, mu_s, b_s, 0.50)

    ax_g.set_xlabel("Trade index (sorted by predicted median)", fontsize=10)
    ax_g.set_ylabel("|impact| (bps)", fontsize=10)
    ax_g.set_title(f"{ticker} Full GAMLSS\n"
                   f"({n_trades} random test trades)",
                   fontsize=11, fontweight="bold")
    ax_g.legend(fontsize=8, loc="upper left", ncol=2)
    ax_g.grid(True, alpha=0.12)
    ax_g.spines["top"].set_visible(False)
    ax_g.spines["right"].set_visible(False)

    # ── Right column: Two-Stage XGBoost ──
    ax_x = axes_fan[row, 1]
    sort_x = np.argsort(mu_te_x[idx])
    idx_x = idx[sort_x]

    mu_sx = mu_te_x[idx_x]
    b_sx = b_te_x[idx_x]
    y_sx = y_te[idx_x]

    mu_sx_sm = smooth(mu_sx)
    b_sx_sm = smooth(b_sx)

    for level, band_color, label in [
        (0.90, bands["90%"], "90% interval"),
        (0.80, bands["80%"], "80% interval"),
        (0.50, bands["50%"], "50% interval"),
    ]:
        z = np.log(1.0 / (1.0 - level))
        lo = np.maximum(mu_sx_sm - z * b_sx_sm, 0.0)
        hi = mu_sx_sm + z * b_sx_sm
        ax_x.fill_between(x_pos, lo, hi, alpha=0.7, color=band_color, label=label)

    ax_x.plot(x_pos, mu_sx_sm, color="white", lw=1.8, zorder=3)
    ax_x.plot(x_pos, mu_sx_sm, color=color, lw=1.2, label="Predicted median", zorder=3)
    ax_x.scatter(x_pos, y_sx, s=10, color="black", alpha=0.5, zorder=4, label="Actual |impact|")

    # Coverage on raw predictions
    cov90x, _ = compute_coverage(y_sx, mu_sx, b_sx, 0.90)
    cov80x, _ = compute_coverage(y_sx, mu_sx, b_sx, 0.80)
    cov50x, _ = compute_coverage(y_sx, mu_sx, b_sx, 0.50)

    ax_x.set_xlabel("Trade index (sorted by predicted median)", fontsize=10)
    ax_x.set_ylabel("|impact| (bps)", fontsize=10)
    ax_x.set_title(f"{ticker} Two-Stage XGBoost\n"
                   f"({n_trades} random test trades)",
                   fontsize=11, fontweight="bold")
    ax_x.legend(fontsize=8, loc="upper left", ncol=2)
    ax_x.grid(True, alpha=0.12)
    ax_x.spines["top"].set_visible(False)
    ax_x.spines["right"].set_visible(False)

    # Match y-axis limits across the pair
    ymax = max(ax_g.get_ylim()[1], ax_x.get_ylim()[1])
    ax_g.set_ylim(0, ymax)
    ax_x.set_ylim(0, ymax)

fig_fan.suptitle(
    "Prediction Interval Fan Charts: Full GAMLSS vs Two-Stage XGBoost\n"
    "50% / 80% / 90% Laplace Intervals | 300 Random Test Trades",
    fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
fig_fan.savefig("gamlss_fan_chart.png", dpi=150, bbox_inches="tight")
print("Saved -> gamlss_fan_chart.png")
plt.close(fig_fan)
