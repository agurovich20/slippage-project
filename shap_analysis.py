"""
SHAP analysis on the best XGBoost fold (fold 1, OOS R²=+0.625).

Best fold: train on block 0 (2024-06-03..2024-06-20),
           test on block 1  (2024-06-21..2024-07-10).
Best params: learning_rate=0.05, max_depth=2, min_child_weight=5, n_estimators=50.

Plots (single figure, 2 rows):
  Row 0: SHAP beeswarm summary (all 50 test bins × 3 features)
  Row 1: Partial dependence plots for mean_spread, mean_vol, mean_prate
          — sweep each feature across its test-set range, other features
            held at their test-set mean; overlay actual test-bin scatter.

Output: aapl_shap.png
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"]      = "1"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import shap
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# ── Reproduce fold 1 exactly ──────────────────────────────────────────────────
df = pd.read_parquet("data/lit_buy_features_v2.parquet")
df["abs_impact"] = df["impact_vwap_bps"].abs()
df = df.sort_values("date").reset_index(drop=True)

COLS = ["date", "roll_spread_500", "roll_vol_500", "participation_rate", "abs_impact"]
df   = df[COLS].dropna()

N_BINS  = 50
N_FOLDS = 5
FEAT_NAMES = ["mean_spread", "mean_vol", "mean_prate"]
FEAT_LABELS = [
    "mean_spread\n(Roll spread, bps)",
    "mean_vol\n(realized vol, bps)",
    "mean_prate\n(participation rate)",
]

unique_dates = np.array(sorted(df["date"].unique()))
n_days = len(unique_dates)
date_fold = np.digitize(
    np.arange(n_days),
    bins=np.linspace(0, n_days, N_FOLDS + 1)[1:-1],
)
dates    = df["date"].to_numpy()
date_to_fold = dict(zip(unique_dates, date_fold))
row_fold = np.array([date_to_fold[d] for d in dates])

def make_bins(data_idx):
    sub = df.iloc[data_idx][
        ["roll_spread_500", "roll_vol_500", "participation_rate", "abs_impact"]
    ].copy()
    sub["bin"] = pd.qcut(sub["roll_spread_500"], q=N_BINS, labels=False,
                          duplicates="drop")
    return (
        sub.groupby("bin", observed=True)
        .agg(
            mean_spread=("roll_spread_500",   "mean"),
            mean_vol   =("roll_vol_500",      "mean"),
            mean_prate =("participation_rate", "mean"),
            mean_abs   =("abs_impact",         "mean"),
            count      =("abs_impact",         "count"),
        )
        .reset_index(drop=True)
    )

# Fold 1: train = block 0, test = block 1
tr_idx = np.where(row_fold < 1)[0]
te_idx = np.where(row_fold == 1)[0]

tr_bins = make_bins(tr_idx)
te_bins = make_bins(te_idx)

X_tr = tr_bins[FEAT_NAMES].to_numpy(dtype=np.float64)
y_tr = tr_bins["mean_abs"].to_numpy(dtype=np.float64)
X_te = te_bins[FEAT_NAMES].to_numpy(dtype=np.float64)
y_te = te_bins["mean_abs"].to_numpy(dtype=np.float64)

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)

# Train with best-fold hyperparameters
BEST_PARAMS = dict(
    learning_rate=0.05, max_depth=2, min_child_weight=5, n_estimators=50,
    tree_method="hist", random_state=42, n_jobs=1, verbosity=0,
)
model = xgb.XGBRegressor(**BEST_PARAMS)
model.fit(X_tr_s, y_tr)

y_hat = model.predict(X_te_s)
ss_res = ((y_te - y_hat) ** 2).sum()
ss_tot = ((y_te - y_te.mean()) ** 2).sum()
oos_r2 = 1.0 - ss_res / ss_tot
print(f"Reproduced fold 1  OOS R²={oos_r2:.4f}  "
      f"(train bins={len(tr_bins)}, test bins={len(te_bins)})")

# ── SHAP values (TreeExplainer on scaled test bins) ───────────────────────────
explainer  = shap.TreeExplainer(model, X_tr_s)
shap_vals  = explainer.shap_values(X_te_s)   # (50, 3)
base_value = explainer.expected_value
print(f"SHAP base value (E[f(x)]): {base_value:.4f}")
print(f"SHAP array shape: {shap_vals.shape}")
print(f"Mean |SHAP| per feature:")
for name, mv in zip(FEAT_NAMES, np.abs(shap_vals).mean(axis=0)):
    print(f"  {name:<15} {mv:.5f}")

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 10))
gs  = gridspec.GridSpec(
    2, 3,
    figure=fig,
    height_ratios=[1.05, 1],
    hspace=0.50,
    wspace=0.38,
)

ax_bee  = fig.add_subplot(gs[0, :2])   # beeswarm spans cols 0-1
ax_bar  = fig.add_subplot(gs[0, 2])    # mean |SHAP| bar
ax_pdp  = [fig.add_subplot(gs[1, k]) for k in range(3)]


# ── Panel A: SHAP beeswarm (manual) ──────────────────────────────────────────
mean_abs_shap = np.abs(shap_vals).mean(axis=0)        # (3,)
order = np.argsort(mean_abs_shap)                      # ascending → top = most important

rng = np.random.default_rng(0)

# Colormap: blue (low feature value) → red (high)
cmap = cm.coolwarm

for plot_row, feat_idx in enumerate(order):
    sv  = shap_vals[:, feat_idx]                       # SHAP values for this feature
    fv  = X_te[:, feat_idx]                            # unscaled feature values (for color)
    fv_norm = (fv - fv.min()) / ((fv.max() - fv.min()) + 1e-12)  # 0..1 for colormap

    # Beeswarm: deterministic y-jitter proportional to density
    y_jitter = rng.uniform(-0.28, 0.28, size=len(sv))

    colors = cmap(fv_norm)
    ax_bee.scatter(
        sv, plot_row + y_jitter,
        c=colors, s=45, alpha=0.88,
        edgecolors="none", zorder=3,
    )

ax_bee.set_yticks(range(len(order)))
ax_bee.set_yticklabels(
    [FEAT_LABELS[i] for i in order], fontsize=9.5
)
ax_bee.axvline(0, color="black", lw=1.0, ls="--", alpha=0.5)
ax_bee.set_xlabel("SHAP value  (impact on |impact_vwap_bps| prediction)", fontsize=9.5)
ax_bee.set_title(
    f"SHAP beeswarm — best XGBoost fold (OOS R²={oos_r2:.4f})\n"
    f"50 test bins  |  color = feature value (blue=low, red=high)",
    fontsize=10, fontweight="bold",
)
ax_bee.grid(axis="x", alpha=0.2)
ax_bee.spines["top"].set_visible(False)
ax_bee.spines["right"].set_visible(False)

# Colorbar for beeswarm
sm = cm.ScalarMappable(cmap=cmap, norm=Normalize(0, 1))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax_bee, fraction=0.025, pad=0.02)
cbar.set_label("Feature value\n(low → high)", fontsize=8)
cbar.set_ticks([0, 0.5, 1])
cbar.set_ticklabels(["low", "mid", "high"], fontsize=7.5)


# ── Panel B: mean |SHAP| bar chart ────────────────────────────────────────────
bar_order = np.argsort(mean_abs_shap)[::-1]           # descending for bar
bar_colors = ["#2563eb", "#16a34a", "#dc2626"]
bar_labels_short = ["mean_spread", "mean_vol", "mean_prate"]

ax_bar.barh(
    range(3),
    mean_abs_shap[bar_order],
    color=[bar_colors[i] for i in bar_order],
    edgecolor="white", linewidth=0.6, height=0.55,
)
ax_bar.set_yticks(range(3))
ax_bar.set_yticklabels(
    [bar_labels_short[i] for i in bar_order], fontsize=9.5
)
for j, (val, feat_idx) in enumerate(zip(mean_abs_shap[bar_order], bar_order)):
    ax_bar.text(val + 0.0002, j, f"{val:.4f}", va="center", fontsize=8.5)
ax_bar.set_xlabel("Mean |SHAP value|", fontsize=9.5)
ax_bar.set_title("SHAP feature importance\n(mean |SHAP|)", fontsize=10, fontweight="bold")
ax_bar.grid(axis="x", alpha=0.2)
ax_bar.spines["top"].set_visible(False)
ax_bar.spines["right"].set_visible(False)


# ── Panel C: Partial dependence plots ─────────────────────────────────────────
# For each feature: sweep across its test-set range, hold other two at test mean.
# Both sweep and held values pass through the scaler before prediction.

te_mean_raw = X_te.mean(axis=0)   # shape (3,) — mean of each unscaled feature
GRID_N = 200
PDP_COLORS = ["#2563eb", "#16a34a", "#dc2626"]

for k, (ax, feat_idx) in enumerate(zip(ax_pdp, range(3))):
    feat_min = X_te[:, feat_idx].min()
    feat_max = X_te[:, feat_idx].max()
    grid_raw = np.linspace(feat_min, feat_max, GRID_N)

    # Build raw feature matrix: sweep feature k, hold others at test mean
    X_sweep_raw = np.tile(te_mean_raw, (GRID_N, 1))   # (GRID_N, 3)
    X_sweep_raw[:, feat_idx] = grid_raw

    X_sweep_s = scaler.transform(X_sweep_raw)
    y_pdp     = model.predict(X_sweep_s)

    # PDP line
    ax.plot(grid_raw, y_pdp, color=PDP_COLORS[k], lw=2.2, zorder=4,
            label="PDP (other features at test mean)")

    # Actual test bin scatter
    ax.scatter(
        X_te[:, feat_idx], y_te,
        color="#1e293b", s=35, alpha=0.65, zorder=5,
        edgecolors="white", linewidths=0.5,
        label="Test bin means",
    )

    # SHAP-based local slopes: annotate mean SHAP
    mean_shap_k = np.abs(shap_vals[:, feat_idx]).mean()
    ax.set_xlabel(FEAT_LABELS[k], fontsize=9.5)
    ax.set_ylabel("|impact_vwap_bps| (bps)", fontsize=9)
    ax.set_title(
        f"PDP: {FEAT_NAMES[k]}\nmean |SHAP| = {mean_shap_k:.4f}",
        fontsize=9.5, fontweight="bold",
    )
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(True, alpha=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── Supertitle ────────────────────────────────────────────────────────────────
fig.suptitle(
    "AAPL lit buy block trades — SHAP analysis on best XGBoost fold\n"
    "50-bin aggregated data  |  features: mean_spread, mean_vol, mean_prate  "
    "|  target: |impact_vwap_bps|",
    fontsize=11, fontweight="bold", y=1.01,
)

plt.savefig("aapl_shap.png", dpi=150, bbox_inches="tight")
print("\nsaved -> aapl_shap.png")
