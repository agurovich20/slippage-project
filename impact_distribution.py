"""
Rashkovich & Iogansen (2022) half-normal / half-Laplace distribution fit
to AAPL lit buy block trade impact_vwap_bps.

Distribution:
  f(x | mu, sigma, rho) =
      C * exp( -(x-mu)^2 / (2*sigma^2) )   for x <= mu   [half-normal, left]
      C * exp( -(x-mu) / rho )              for x >  mu   [half-Laplace, right]

  where C = 1 / (sigma*sqrt(pi/2) + rho)  (normalisation constant).

Fit by MLE (Nelder-Mead on log-transformed sigma, rho for unconstrained opt).

Panels:
  (1) Overall histogram + fitted density
  (2) 5 conditional fitted densities by roll_spread_500 quintile
  (3) sigma and rho vs mean spread per quintile

Output: aapl_impact_distribution.png
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"]      = "1"

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Load ───────────────────────────────────────────────────────────────────────
df = pd.read_parquet("data/lit_buy_features_v2.parquet")
print(f"Loaded {len(df):,} trades")

x_all    = df["impact_vwap_bps"].to_numpy(dtype=np.float64)
spread   = df["roll_spread_500"].to_numpy(dtype=np.float64)

p1, p99 = np.percentile(x_all, [1, 99])
print(f"impact_vwap_bps: mean={x_all.mean():.4f}  median={np.median(x_all):.4f}  "
      f"std={x_all.std():.4f}  [p1={p1:.2f}, p99={p99:.2f}]")


# ── Distribution helpers ───────────────────────────────────────────────────────
SQRT_PI_OVER_2 = np.sqrt(np.pi / 2.0)

def norm_const(sigma, rho):
    return sigma * SQRT_PI_OVER_2 + rho

def pdf(x_grid, mu, sigma, rho):
    C    = 1.0 / norm_const(sigma, rho)
    left  = C * np.exp(-((x_grid - mu) ** 2) / (2.0 * sigma ** 2))
    right = C * np.exp(-(x_grid - mu) / rho)
    return np.where(x_grid <= mu, left, right)

def neg_log_lik(params, x):
    mu    = params[0]
    sigma = np.exp(params[1])   # log-transform keeps sigma > 0
    rho   = np.exp(params[2])   # log-transform keeps rho   > 0

    Z  = norm_const(sigma, rho)
    lm = x <= mu
    rm = ~lm

    ll  = -len(x) * np.log(Z)
    ll -= np.sum((x[lm] - mu) ** 2) / (2.0 * sigma ** 2)
    ll -= np.sum( x[rm] - mu)       / rho
    return -ll


def fit(x, label=""):
    """MLE fit with multiple starting points; returns (mu, sigma, rho, -nll)."""
    mu0    = np.median(x)
    left   = x[x <= mu0]
    right  = x[x >  mu0]
    sig0   = max(left.std()  if len(left)  > 1 else 1.0, 0.05)
    rho0   = max((right - mu0).mean() if len(right) > 1 else 1.0, 0.05)

    best = None
    # Sweep a grid of starting points to avoid local minima
    for mu_s in [mu0, mu0 - 0.5 * sig0, mu0 + 0.3 * rho0]:
        for ls in [np.log(sig0 * f) for f in (0.5, 1.0, 2.0)]:
            for lr in [np.log(rho0 * f) for f in (0.5, 1.0, 2.0)]:
                res = minimize(
                    neg_log_lik, x0=[mu_s, ls, lr], args=(x,),
                    method="Nelder-Mead",
                    options={"maxiter": 20_000, "xatol": 1e-7, "fatol": 1e-7},
                )
                if best is None or res.fun < best.fun:
                    best = res

    mu_fit    = best.x[0]
    sigma_fit = np.exp(best.x[1])
    rho_fit   = np.exp(best.x[2])
    ll_fit    = -best.fun / len(x)    # per-observation log-likelihood

    print(f"  {label:<22}  n={len(x):>6,}  "
          f"mu={mu_fit:+.4f}  sigma={sigma_fit:.4f}  rho={rho_fit:.4f}  "
          f"ll/n={ll_fit:.4f}")
    return mu_fit, sigma_fit, rho_fit, ll_fit


# ── Overall fit ────────────────────────────────────────────────────────────────
print("\n=== Fitting overall distribution ===")
mu_all, sig_all, rho_all, _ = fit(x_all, "overall")

# ── 5-bin fits by roll_spread_500 quintile ─────────────────────────────────────
print("\n=== Fitting per spread quintile ===")
df["q5"] = pd.qcut(df["roll_spread_500"], q=5, labels=False, duplicates="drop")

bin_results = []   # (mean_spread, mu, sigma, rho)
for q in range(5):
    mask  = df["q5"] == q
    x_bin = df.loc[mask, "impact_vwap_bps"].to_numpy(dtype=np.float64)
    s_mid = df.loc[mask, "roll_spread_500"].mean()
    label = f"Q{q+1} spread~{s_mid:.2f}bps"
    mu_q, sig_q, rho_q, _ = fit(x_bin, label)
    bin_results.append((s_mid, mu_q, sig_q, rho_q))

spread_mids = np.array([r[0] for r in bin_results])
mus         = np.array([r[1] for r in bin_results])
sigmas      = np.array([r[2] for r in bin_results])
rhos        = np.array([r[3] for r in bin_results])


# ── Figure ─────────────────────────────────────────────────────────────────────
QCOLORS = ["#1d4ed8", "#16a34a", "#ca8a04", "#dc2626", "#7c3aed"]
QLABELS = [f"Q{i+1}: spread~{spread_mids[i]:.2f} bps" for i in range(5)]

# Clip range for display: p0.5 to p99.5, rounded
x_lo = max(np.percentile(x_all, 0.5),  -20.0)
x_hi = min(np.percentile(x_all, 99.5),  30.0)
x_grid = np.linspace(x_lo, x_hi, 800)

fig = plt.figure(figsize=(18, 5.5))
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])


# ── Panel 1: Overall histogram + fitted density ───────────────────────────────
x_clipped = x_all[(x_all >= x_lo) & (x_all <= x_hi)]
ax1.hist(x_clipped, bins=120, density=True,
         color="#94a3b8", alpha=0.55, edgecolor="none",
         label=f"Empirical (n={len(x_all):,})")

y_fit = pdf(x_grid, mu_all, sig_all, rho_all)
ax1.plot(x_grid, y_fit, color="#dc2626", lw=2.2, zorder=5,
         label=(f"RI(2022) fit\n"
                f"$\\mu$={mu_all:+.3f}  "
                f"$\\sigma$={sig_all:.3f}  "
                f"$\\rho$={rho_all:.3f}"))

ax1.axvline(mu_all, color="#dc2626", lw=1.0, ls="--", alpha=0.6)

# Shade left (half-normal) and right (half-Laplace) regions
ax1.fill_between(x_grid[x_grid <= mu_all],
                 pdf(x_grid[x_grid <= mu_all], mu_all, sig_all, rho_all),
                 alpha=0.12, color="#2563eb", label="Half-normal (left)")
ax1.fill_between(x_grid[x_grid > mu_all],
                 pdf(x_grid[x_grid > mu_all], mu_all, sig_all, rho_all),
                 alpha=0.12, color="#f59e0b", label="Half-Laplace (right)")

ax1.set_xlabel("impact_vwap_bps", fontsize=10.5)
ax1.set_ylabel("Density", fontsize=10.5)
ax1.set_title(
    "Overall: half-normal / half-Laplace fit\n"
    "(Rashkovich & Iogansen 2022)",
    fontsize=10.5, fontweight="bold",
)
ax1.legend(fontsize=8, loc="upper right")
ax1.grid(True, alpha=0.18)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.set_xlim(x_lo, x_hi)


# ── Panel 2: 5 conditional fitted densities overlaid ─────────────────────────
for i, (color, label, (s_mid, mu_q, sig_q, rho_q)) in enumerate(
    zip(QCOLORS, QLABELS, bin_results)
):
    y_q = pdf(x_grid, mu_q, sig_q, rho_q)
    ax2.plot(x_grid, y_q, color=color, lw=2.0, alpha=0.9,
             label=f"{label}\n"
                   f"  $\\mu$={mu_q:+.3f}  $\\sigma$={sig_q:.3f}  $\\rho$={rho_q:.3f}")
    ax2.axvline(mu_q, color=color, lw=0.8, ls=":", alpha=0.6)

ax2.set_xlabel("impact_vwap_bps", fontsize=10.5)
ax2.set_ylabel("Density", fontsize=10.5)
ax2.set_title(
    "Conditional fit by roll_spread_500 quintile\n"
    "(higher spread → wider / heavier-tailed)",
    fontsize=10.5, fontweight="bold",
)
ax2.legend(fontsize=7, loc="upper right", framealpha=0.92)
ax2.grid(True, alpha=0.18)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.set_xlim(x_lo, x_hi)


# ── Panel 3: sigma and rho vs mean spread ─────────────────────────────────────
ax3.plot(spread_mids, sigmas, color="#2563eb", lw=2.2, marker="o", ms=7,
         label=f"$\\sigma$ (left half-normal width)")
ax3.plot(spread_mids, rhos,   color="#f59e0b", lw=2.2, marker="s", ms=7,
         label=f"$\\rho$ (right half-Laplace width)")

# Annotate with values
for i, (sm, sg, ro) in enumerate(zip(spread_mids, sigmas, rhos)):
    ax3.annotate(f"{sg:.3f}", (sm, sg), textcoords="offset points",
                 xytext=(-6, 6), fontsize=7.5, color="#2563eb")
    ax3.annotate(f"{ro:.3f}", (sm, ro), textcoords="offset points",
                 xytext=(-6, -12), fontsize=7.5, color="#ca8a04")

# Overall-fit reference lines
ax3.axhline(sig_all, color="#2563eb", lw=1.0, ls="--", alpha=0.5,
            label=f"Overall $\\sigma$={sig_all:.3f}")
ax3.axhline(rho_all, color="#f59e0b", lw=1.0, ls="--", alpha=0.5,
            label=f"Overall $\\rho$={rho_all:.3f}")

ax3.set_xlabel("Mean roll_spread_500 per quintile (bps)", fontsize=10.5)
ax3.set_ylabel("Distribution width (bps)", fontsize=10.5)
ax3.set_title(
    "$\\sigma$ (left width) and $\\rho$ (right width)\nvs spread level",
    fontsize=10.5, fontweight="bold",
)
ax3.legend(fontsize=8.5, loc="upper left")
ax3.grid(True, alpha=0.18)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)


fig.suptitle(
    "AAPL lit buy block trades — Rashkovich-Iogansen (2022) distribution fit  "
    "|  impact_vwap_bps\n"
    "Left: half-normal ($\\sigma$)  ·  Right: half-Laplace ($\\rho$)  ·  "
    "Conditioned on roll_spread_500 quintile",
    fontsize=11, fontweight="bold", y=1.02,
)

plt.savefig("aapl_impact_distribution.png", dpi=150, bbox_inches="tight")
print("\nsaved -> aapl_impact_distribution.png")
