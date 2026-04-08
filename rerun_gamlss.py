"""
Rerun GAMLSS + visualization for all 6 stocks using saved feature parquets.
Filters NVDA outliers (stock split on 2024-06-10) before fitting.
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
from xgboost import XGBRegressor

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


def run_gamlss(train_df, test_df, ticker):
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
    loc_mae = np.mean(np.abs(y_te - mu_hat_te))

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

    coverages = {}
    for level in [0.50, 0.80, 0.90]:
        z = np.log(1.0 / (1.0 - level))
        lo = np.maximum(mu_hat_te - z * b_hat_te, 0.0)
        hi = mu_hat_te + z * b_hat_te
        cov = ((y_te >= lo) & (y_te <= hi)).mean()
        width = (hi - lo).mean()
        coverages[level] = (cov, width)

    return {
        "ticker": ticker,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "mean_abs_impact": y_te.mean(),
        "loc_mae": loc_mae,
        "cov_90": coverages[0.90][0],
        "cov_80": coverages[0.80][0],
        "cov_50": coverages[0.50][0],
        "width_90": coverages[0.90][1],
    }


all_results = []
for ticker, (tr_path, te_path) in DATASETS.items():
    train_df = pd.read_parquet(tr_path)
    test_df = pd.read_parquet(te_path)

    # Filter extreme outliers
    n0_tr, n0_te = len(train_df), len(test_df)
    train_df = train_df[train_df["impact_vwap_bps"].abs() <= IMPACT_CAP_BPS]
    test_df = test_df[test_df["impact_vwap_bps"].abs() <= IMPACT_CAP_BPS]
    removed = (n0_tr - len(train_df)) + (n0_te - len(test_df))
    if removed > 0:
        print(f"  {ticker}: removed {removed} rows with |impact| > {IMPACT_CAP_BPS} bps")

    r = run_gamlss(train_df, test_df, ticker)
    all_results.append(r)
    print(f"  {ticker}: train={r['n_train']:,}, test={r['n_test']:,}, "
          f"MAE={r['loc_mae']:.4f}, 90% cov={r['cov_90']:.4f}")

# Summary table
print(f"\n{'=' * 100}")
print(f"  CROSS-STOCK XGB GAMLSS VALIDATION (3 features: spread, vol, participation)")
print(f"  Location: XGB LAD (depth=3, n=200) | Scale: XGB MSE (depth=3, n=50)")
print(f"  Train: Jun-Aug 2024 | Test: Sep 2024 | Laplace intervals")
print(f"{'=' * 100}")
print(f"  {'Ticker':<8} {'n_train':>8} {'n_test':>7} {'Mean|imp|':>10} "
      f"{'Loc MAE':>8} {'90% Cov':>8} {'80% Cov':>8} {'50% Cov':>8} "
      f"{'90% Width':>10}")
print(f"  {'-' * 8} {'-' * 8} {'-' * 7} {'-' * 10} {'-' * 8} "
      f"{'-' * 8} {'-' * 8} {'-' * 8} {'-' * 10}")

for r in all_results:
    print(f"  {r['ticker']:<8} {r['n_train']:>8,} {r['n_test']:>7,} "
          f"{r['mean_abs_impact']:>10.4f} {r['loc_mae']:>8.4f} "
          f"{r['cov_90']:>8.4f} {r['cov_80']:>8.4f} {r['cov_50']:>8.4f} "
          f"{r['width_90']:>10.4f}")

print(f"\n  Nominal targets:  90% -> 0.9000   80% -> 0.8000   50% -> 0.5000")

# Visualization
results_df = pd.DataFrame(all_results)
n_stocks = len(results_df)
tickers_ordered = results_df["ticker"].tolist()
colors = ["#2563eb", "#dc2626", "#16a34a", "#f59e0b", "#7c3aed", "#ec4899"][:n_stocks]

fig = plt.figure(figsize=(24, 14))
gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.32, hspace=0.42)

x = np.arange(n_stocks)
bar_w = 0.55

# Panel 1: Coverage
ax1 = fig.add_subplot(gs[0, 0])
bw = 0.22
for j, (level, label) in enumerate([(0.90, "90%"), (0.80, "80%"), (0.50, "50%")]):
    col_name = f"cov_{int(level*100):d}"
    vals = results_df[col_name].values
    offset = (j - 1) * bw
    ax1.bar(x + offset, vals, width=bw, label=f"{label} actual",
            alpha=0.85, edgecolor="white", linewidth=0.5)
    ax1.axhline(level, color="gray", lw=0.8, ls=":", alpha=0.5)
ax1.set_xticks(x)
ax1.set_xticklabels(tickers_ordered, fontsize=10, fontweight="bold")
ax1.set_ylabel("Actual Coverage", fontsize=11)
ax1.set_title("Coverage by Stock\n(dashed = nominal)", fontsize=12, fontweight="bold")
ax1.legend(fontsize=8)
ax1.grid(axis="y", alpha=0.2)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.set_ylim(0, 1.05)

# Panel 2: Location MAE
ax2 = fig.add_subplot(gs[0, 1])
ax2.bar(x, results_df["loc_mae"].values, color=colors, width=bar_w,
        edgecolor="white", linewidth=0.8)
for i, v in enumerate(results_df["loc_mae"].values):
    ax2.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
ax2.set_xticks(x)
ax2.set_xticklabels(tickers_ordered, fontsize=10, fontweight="bold")
ax2.set_ylabel("MAE (bps)", fontsize=11)
ax2.set_title("Location Model MAE (OOS)", fontsize=12, fontweight="bold")
ax2.grid(axis="y", alpha=0.2)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

# Panel 3: Mean absolute impact
ax3 = fig.add_subplot(gs[0, 2])
ax3.bar(x, results_df["mean_abs_impact"].values, color=colors, width=bar_w,
        edgecolor="white", linewidth=0.8)
for i, v in enumerate(results_df["mean_abs_impact"].values):
    ax3.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
ax3.set_xticks(x)
ax3.set_xticklabels(tickers_ordered, fontsize=10, fontweight="bold")
ax3.set_ylabel("|impact| (bps)", fontsize=11)
ax3.set_title("Mean |impact| on Test Set", fontsize=12, fontweight="bold")
ax3.grid(axis="y", alpha=0.2)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)

# Panel 4: 90% interval width
ax4 = fig.add_subplot(gs[1, 0])
ax4.bar(x, results_df["width_90"].values, color=colors, width=bar_w,
        edgecolor="white", linewidth=0.8)
for i, v in enumerate(results_df["width_90"].values):
    ax4.text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")
ax4.set_xticks(x)
ax4.set_xticklabels(tickers_ordered, fontsize=10, fontweight="bold")
ax4.set_ylabel("Width (bps)", fontsize=11)
ax4.set_title("Mean 90% Interval Width", fontsize=12, fontweight="bold")
ax4.grid(axis="y", alpha=0.2)
ax4.spines["top"].set_visible(False)
ax4.spines["right"].set_visible(False)

# Panel 5: Coverage deviation heatmap
ax5 = fig.add_subplot(gs[1, 1])
cov_matrix = np.array([
    [r["cov_90"] - 0.90, r["cov_80"] - 0.80, r["cov_50"] - 0.50]
    for r in all_results
])
lim = max(abs(cov_matrix.min()), abs(cov_matrix.max())) * 1.1
im = ax5.imshow(cov_matrix, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
ax5.set_xticks([0, 1, 2])
ax5.set_xticklabels(["90%", "80%", "50%"], fontsize=10)
ax5.set_yticks(range(n_stocks))
ax5.set_yticklabels(tickers_ordered, fontsize=10, fontweight="bold")
for i in range(n_stocks):
    for j in range(3):
        v = cov_matrix[i, j]
        text_color = "white" if abs(v) > lim * 0.5 else "black"
        ax5.text(j, i, f"{v:+.3f}", ha="center", va="center",
                 fontsize=9.5, fontweight="bold", color=text_color)
cbar = plt.colorbar(im, ax=ax5, fraction=0.04, pad=0.04)
cbar.set_label("Coverage - Nominal", fontsize=9)
ax5.set_title("Coverage Deviation from Nominal\n(blue=under, red=over)",
              fontsize=12, fontweight="bold")

# Panel 6: Sample size
ax6 = fig.add_subplot(gs[1, 2])
train_sizes = results_df["n_train"].values
test_sizes = results_df["n_test"].values
ax6.bar(x - 0.15, train_sizes, width=0.3, color="#64748b", label="Train",
        edgecolor="white", linewidth=0.5)
ax6.bar(x + 0.15, test_sizes, width=0.3, color="#f59e0b", label="Test",
        edgecolor="white", linewidth=0.5)
for i, (tr, te) in enumerate(zip(train_sizes, test_sizes)):
    ax6.text(i - 0.15, tr + 50, f"{tr:,}", ha="center", fontsize=7.5,
             fontweight="bold", rotation=45)
    ax6.text(i + 0.15, te + 50, f"{te:,}", ha="center", fontsize=7.5,
             fontweight="bold", rotation=45)
ax6.set_xticks(x)
ax6.set_xticklabels(tickers_ordered, fontsize=10, fontweight="bold")
ax6.set_ylabel("Number of trades", fontsize=11)
ax6.set_title("Dataset Size (Train vs Test)", fontsize=12, fontweight="bold")
ax6.legend(fontsize=9)
ax6.grid(axis="y", alpha=0.2)
ax6.spines["top"].set_visible(False)
ax6.spines["right"].set_visible(False)

fig.suptitle(
    "Cross-Stock XGB GAMLSS Validation: Block Trade Impact Prediction\n"
    "3 features (spread, volatility, participation rate) | "
    "Fixed AAPL hyperparameters | Train: Jun-Aug, Test: Sep 2024",
    fontsize=14, fontweight="bold", y=1.01,
)

fig.savefig("cross_stock_validation.png", dpi=150, bbox_inches="tight")
print("\nSaved -> cross_stock_validation.png")

results_df.to_csv("data/cross_stock_results.csv", index=False)
print("Saved -> data/cross_stock_results.csv")
