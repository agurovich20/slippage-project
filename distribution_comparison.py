"""
Four-distribution comparison on AAPL lit buy impact_vwap_bps.

Models
------
  (1) Normal      k=2  — MLE: mu=mean, sigma=std
  (2) Laplace     k=2  — MLE: mu=median, b=mean|x-mu|
  (3) Skew-normal k=3  — MLE via scipy.stats.skewnorm.fit
  (4) RI(2022)    k=3  — half-normal left / half-Laplace right, Nelder-Mead MLE

For each: log-likelihood, AIC = 2k-2L, BIC = k*ln(n)-2L, KS statistic + p-value.

Fitted on: overall + 5 spread quintile subsets (6 groups total).

Output: aapl_distribution_comparison.png
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"]      = "1"

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Load ───────────────────────────────────────────────────────────────────────
df = pd.read_parquet("data/lit_buy_features_v2.parquet")
x_all = df["impact_vwap_bps"].to_numpy(dtype=np.float64)
df["q5"] = pd.qcut(df["roll_spread_500"], q=5, labels=False, duplicates="drop")

print(f"Loaded {len(x_all):,} trades")
print(f"impact: mean={x_all.mean():+.4f}  med={np.median(x_all):+.4f}  "
      f"std={x_all.std():.4f}  p1={np.percentile(x_all,1):.2f}  "
      f"p99={np.percentile(x_all,99):.2f}\n")


# ══════════════════════════════════════════════════════════════════════════════
# ── Distribution definitions ──────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

SQRT_PI2 = np.sqrt(np.pi / 2.0)   # sqrt(pi/2) ≈ 1.2533

# ── (4) RI helpers ────────────────────────────────────────────────────────────
def ri_neg_ll(params, x):
    mu, ls, lr = params
    s, r = np.exp(ls), np.exp(lr)
    Z  = s * SQRT_PI2 + r
    lm = x <= mu;  rm = ~lm
    return (len(x) * np.log(Z)
            + np.sum((x[lm] - mu)**2) / (2 * s**2)
            + np.sum(x[rm] - mu) / r)

def ri_pdf(x_grid, mu, s, r):
    C  = 1.0 / (s * SQRT_PI2 + r)
    return np.where(x_grid <= mu,
                    C * np.exp(-((x_grid - mu)**2) / (2 * s**2)),
                    C * np.exp(-(x_grid - mu) / r))

def ri_cdf(x_arr, mu, s, r):
    """Exact analytical CDF of the RI distribution."""
    x_arr = np.asarray(x_arr, dtype=float)
    Z     = s * SQRT_PI2 + r
    F_mu  = s * SQRT_PI2 / Z              # CDF at the mode
    out   = np.empty_like(x_arr)
    lm = x_arr <= mu;  rm = ~lm
    # Left: scaled half-normal CDF  (two-sided normal, so factor of sqrt(2pi))
    out[lm] = s * np.sqrt(2 * np.pi) * stats.norm.cdf(x_arr[lm], mu, s) / Z
    out[rm] = F_mu + (r * (1 - np.exp(-(x_arr[rm] - mu) / r))) / Z
    return out

def fit_ri(x):
    """Nelder-Mead MLE with grid of starting points. Returns (mu, s, r)."""
    mu0  = np.median(x)
    left = x[x <= mu0];  right = x[x > mu0]
    s0   = max(left.std()  if len(left)  > 1 else 1.0, 0.05)
    r0   = max((right - mu0).mean() if len(right) > 1 else 1.0, 0.05)
    best = None
    for mu_s in (mu0, mu0 - 0.5*s0, mu0 + 0.3*r0):
        for sf in (0.5, 1.0, 2.0):
            for rf in (0.5, 1.0, 2.0):
                res = minimize(ri_neg_ll,
                               x0=[mu_s, np.log(s0*sf), np.log(r0*rf)],
                               args=(x,), method="Nelder-Mead",
                               options={"maxiter": 20_000, "xatol":1e-7, "fatol":1e-7})
                if best is None or res.fun < best.fun:
                    best = res
    mu_f, s_f, r_f = best.x[0], np.exp(best.x[1]), np.exp(best.x[2])
    return mu_f, s_f, r_f


# ══════════════════════════════════════════════════════════════════════════════
# ── Unified fit-and-score routine ─────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def fit_all(x, label=""):
    n = len(x)
    rows = []

    # ── (1) Normal ───────────────────────────────────────────────────────────
    mu_n, sig_n = stats.norm.fit(x)
    ll_n  = stats.norm.logpdf(x, mu_n, sig_n).sum()
    ks_n  = stats.ks_1samp(x, lambda t: stats.norm.cdf(t, mu_n, sig_n))
    rows.append(dict(model="Normal", k=2, params=(mu_n, sig_n),
                     ll=ll_n, ks=ks_n.statistic, ksp=ks_n.pvalue,
                     pdf_fn=lambda g, p=(mu_n,sig_n): stats.norm.pdf(g, *p)))

    # ── (2) Laplace ──────────────────────────────────────────────────────────
    loc_l, b_l = stats.laplace.fit(x)
    ll_l  = stats.laplace.logpdf(x, loc_l, b_l).sum()
    ks_l  = stats.ks_1samp(x, lambda t: stats.laplace.cdf(t, loc_l, b_l))
    rows.append(dict(model="Laplace", k=2, params=(loc_l, b_l),
                     ll=ll_l, ks=ks_l.statistic, ksp=ks_l.pvalue,
                     pdf_fn=lambda g, p=(loc_l,b_l): stats.laplace.pdf(g, *p)))

    # ── (3) Skew-normal ──────────────────────────────────────────────────────
    a_s, xi_s, om_s = stats.skewnorm.fit(x)
    ll_s  = stats.skewnorm.logpdf(x, a_s, xi_s, om_s).sum()
    ks_s  = stats.ks_1samp(x, lambda t: stats.skewnorm.cdf(t, a_s, xi_s, om_s))
    rows.append(dict(model="Skew-normal", k=3, params=(a_s, xi_s, om_s),
                     ll=ll_s, ks=ks_s.statistic, ksp=ks_s.pvalue,
                     pdf_fn=lambda g, p=(a_s,xi_s,om_s): stats.skewnorm.pdf(g, *p)))

    # ── (4) RI (half-normal / half-Laplace) ─────────────────────────────────
    mu_r, s_r, r_r = fit_ri(x)
    ll_r  = -ri_neg_ll([mu_r, np.log(s_r), np.log(r_r)], x)
    ks_r  = stats.ks_1samp(x, lambda t: ri_cdf(np.atleast_1d(t), mu_r, s_r, r_r))
    rows.append(dict(model="RI(2022)", k=3, params=(mu_r, s_r, r_r),
                     ll=ll_r, ks=ks_r.statistic, ksp=ks_r.pvalue,
                     pdf_fn=lambda g, p=(mu_r,s_r,r_r): ri_pdf(g, *p)))

    # AIC / BIC
    for row in rows:
        row["aic"] = 2*row["k"] - 2*row["ll"]
        row["bic"] = row["k"]*np.log(n) - 2*row["ll"]

    if label:
        print(f"\n{'-'*72}")
        print(f"  {label}  (n={n:,})")
        print(f"  {'Model':<14} {'k':>2}  {'logL':>10}  {'AIC':>10}  "
              f"{'BIC':>10}  {'KS':>7}  {'KS p-val':>10}")
        print(f"  {'-'*68}")
        for row in rows:
            winner = " *" if row["aic"] == min(r["aic"] for r in rows) else "  "
            print(f"  {row['model']:<14} {row['k']:>2}  {row['ll']:>10.2f}  "
                  f"{row['aic']:>10.2f}  {row['bic']:>10.2f}  "
                  f"{row['ks']:>7.5f}  {row['ksp']:>10.2e}{winner}")

    return rows


# ══════════════════════════════════════════════════════════════════════════════
# ── Fit overall + 5 quintiles ─────────────────────────────────════════════════
# ══════════════════════════════════════════════════════════════════════════════

print("="*72)
print("DISTRIBUTION COMPARISON — AAPL lit buy  impact_vwap_bps")
print("="*72)

overall_rows = fit_all(x_all, label="OVERALL")

# Per-quintile
quintile_data = []   # list of (label, mean_spread, x_bin, rows)
for q in range(5):
    mask    = df["q5"] == q
    x_bin   = df.loc[mask, "impact_vwap_bps"].to_numpy(dtype=np.float64)
    s_mid   = df.loc[mask, "roll_spread_500"].mean()
    label   = f"Q{q+1}  spread~{s_mid:.2f} bps"
    rows_q  = fit_all(x_bin, label=label)
    quintile_data.append((label, s_mid, x_bin, rows_q))

# ── Ranking summary ───────────────────────────────────────────────────────────
print(f"\n\n{'='*72}")
print("AIC RANKING ACROSS REGIMES  (* = best per column)")
print(f"{'='*72}")
header = f"  {'Group':<26}  {'Normal':>9}  {'Laplace':>9}  {'SkewNorm':>9}  {'RI':>9}  Winner"
print(header)
print("  " + "-"*68)

all_groups = [("Overall", overall_rows)] + [(ql, qr) for ql, _, _, qr in quintile_data]
for grp_label, rows in all_groups:
    aics   = {r["model"]: r["aic"] for r in rows}
    best   = min(aics, key=aics.get)
    line   = f"  {grp_label:<26}"
    for m in ("Normal","Laplace","Skew-normal","RI(2022)"):
        v = aics[m]
        tag = " *" if m == best else "  "
        line += f"  {v:>7.0f}{tag}"
    line += f"  {best}"
    print(line)

print(f"\n\n{'='*72}")
print("BIC RANKING ACROSS REGIMES  (* = best per column)")
print(f"{'='*72}")
print(header)
print("  " + "-"*68)
for grp_label, rows in all_groups:
    bics   = {r["model"]: r["bic"] for r in rows}
    best   = min(bics, key=bics.get)
    line   = f"  {grp_label:<26}"
    for m in ("Normal","Laplace","Skew-normal","RI(2022)"):
        v = bics[m]
        tag = " *" if m == best else "  "
        line += f"  {v:>7.0f}{tag}"
    line += f"  {best}"
    print(line)

print(f"\n\n{'='*72}")
print("KS STATISTIC (lower = better fit)")
print(f"{'='*72}")
print(header)
print("  " + "-"*68)
for grp_label, rows in all_groups:
    kss    = {r["model"]: r["ks"] for r in rows}
    best   = min(kss, key=kss.get)
    line   = f"  {grp_label:<26}"
    for m in ("Normal","Laplace","Skew-normal","RI(2022)"):
        v = kss[m]
        tag = " *" if m == best else "  "
        line += f"  {v:>7.4f}{tag}"
    line += f"  {best}"
    print(line)


# ══════════════════════════════════════════════════════════════════════════════
# ── Plot: 6 panels (overall + 5 quintiles) ───────────────────────════════════
# ══════════════════════════════════════════════════════════════════════════════

MODEL_STYLES = {
    "Normal":      dict(color="#2563eb", ls="-",  lw=2.0, label="Normal"),
    "Laplace":     dict(color="#16a34a", ls="--", lw=2.0, label="Laplace"),
    "Skew-normal": dict(color="#f59e0b", ls="-.", lw=2.0, label="Skew-normal"),
    "RI(2022)":    dict(color="#dc2626", ls=":",  lw=2.4, label="RI(2022)"),
}

def draw_panel(ax, x, rows, title, n_hist_bins=100):
    lo = max(np.percentile(x, 0.5), -20.0)
    hi = min(np.percentile(x, 99.5),  35.0)
    x_grid = np.linspace(lo, hi, 600)

    # Histogram (density)
    ax.hist(x[(x >= lo) & (x <= hi)], bins=n_hist_bins, density=True,
            color="#cbd5e1", alpha=0.60, edgecolor="none", zorder=1)

    # 4 fitted densities
    for row in rows:
        st = MODEL_STYLES[row["model"]]
        ax.plot(x_grid, row["pdf_fn"](x_grid),
                color=st["color"], ls=st["ls"], lw=st["lw"],
                label=f"{st['label']}  AIC={row['aic']:.0f}", zorder=3)

    # Annotate best model
    best = min(rows, key=lambda r: r["aic"])
    ax.text(0.97, 0.97, f"Best AIC: {best['model']}",
            transform=ax.transAxes, fontsize=7.5, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#94a3b8", alpha=0.9))

    ax.set_xlim(lo, hi)
    ax.set_title(title, fontsize=9.5, fontweight="bold")
    ax.set_xlabel("impact_vwap_bps", fontsize=8.5)
    ax.set_ylabel("Density", fontsize=8.5)
    ax.grid(True, alpha=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7.5)


fig = plt.figure(figsize=(18, 11))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.35)

# Panel 0: overall
ax0 = fig.add_subplot(gs[0, 0])
draw_panel(ax0, x_all, overall_rows,
           f"Overall  (n={len(x_all):,})")
ax0.legend(fontsize=7, loc="upper right", framealpha=0.92)

# Panels 1-5: quintiles
axes_q = [fig.add_subplot(gs[r, c]) for r, c in
          [(0,1),(0,2),(1,0),(1,1),(1,2)]]
for i, (ax, (qlabel, s_mid, x_bin, rows_q)) in enumerate(
        zip(axes_q, quintile_data)):
    draw_panel(ax, x_bin, rows_q,
               f"Q{i+1}: spread ~ {s_mid:.2f} bps  (n={len(x_bin):,})")
    if i == 0:   # legend only on first quintile panel
        ax.legend(fontsize=7, loc="upper right", framealpha=0.92)

fig.suptitle(
    "AAPL lit buy block trades — distribution comparison on impact_vwap_bps\n"
    "Normal · Laplace · Skew-normal · RI(2022, half-normal/half-Laplace)\n"
    "MLE fit · ranked by AIC",
    fontsize=11, fontweight="bold", y=1.01,
)

plt.savefig("aapl_distribution_comparison.png", dpi=150, bbox_inches="tight")
print("\nsaved -> aapl_distribution_comparison.png")
