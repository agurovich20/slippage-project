"""
Exploratory analysis for AAPL and COIN block trade slippage. Covers buy/sell
classification, distribution comparisons, impact by trade size and regime, sweep cluster
analysis, and raw scatter plots.
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import gc
import logging
from pathlib import Path
from glob import glob

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
from scipy import stats
from scipy.optimize import minimize


def run_classify_sides():
    """
    Classify block trades as buy/sell using the tick test, add 'side' column,
    compute signed price impact, and show breakdown by side x dollar bucket.

    Tick test (Lee-Ready fallback, derived from block_trades.parquet directly):
      - pre_price is the last trade price within 1s before the block trade, which
        for liquid stocks like AAPL is always the immediately preceding trade.
      - Compare block price vs pre_price:
          price > pre_price  -> uptick   -> buy
          price < pre_price  -> downtick -> sell
          price == pre_price -> zero-tick -> forward-fill from prior block's direction

    signed_impact_bps = slippage_bps * side_sign  (+1 buy / -1 sell)
      Positive = price moved adversely for the initiator.
    """

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(__name__)

    DATA_DIR = Path("data")
    BLK_PATH = DATA_DIR / "block_trades.parquet"


    # Tick-test — runs entirely on pre_price already in block_trades.parquet

    def classify_blocks(df):
        """
        Classify each block trade using price vs pre_price (the immediately
        preceding trade price, already stored in block_trades.parquet).

        Zero-ticks are resolved by forward-filling the last non-zero direction
        within each (ticker, date) group — the zero-plus/zero-minus rule applied
        across the block-trade subsequence. Roughly 95% of blocks have a non-zero
        tick so this approximation is excellent.
        """
        raw = np.sign(df["price"].to_numpy() - df["pre_price"].to_numpy()).astype(float)
        raw[df["pre_price"].isna().to_numpy()] = np.nan   # no preceding trade -> unknown

        dirs = pd.Series(raw, index=df.index).replace(0.0, np.nan)
        dirs = (
            df.groupby(["ticker", "date"], sort=False, group_keys=False)
              .apply(lambda g: dirs.loc[g.index].ffill(), include_groups=False)
        )
        return dirs.fillna(0).astype(np.int8).to_numpy()


    # Main

    blocks = pq.read_table(BLK_PATH).to_pandas()
    log.info("Loaded %d block trades", len(blocks))

    blocks.sort_values(["ticker", "date", "timestamp_ns"], inplace=True, ignore_index=True)
    side_values = classify_blocks(blocks)

    for (ticker, date_str), grp in blocks.groupby(["ticker", "date"]):
        dirs = side_values[grp.index]
        log.info("  %s %s  buys=%d  sells=%d  unknown=%d",
                 ticker, date_str,
                 (dirs ==  1).sum(), (dirs == -1).sum(), (dirs == 0).sum())

    blocks["side"] = side_values
    blocks["side_label"] = pd.Categorical(
        np.where(side_values ==  1, "buy",
        np.where(side_values == -1, "sell", "unknown")),
        categories=["buy", "sell", "unknown"],
    )
    blocks["signed_impact_bps"] = blocks["slippage_bps"] * blocks["side"].replace(0, np.nan)

    pq.write_table(pa.Table.from_pandas(blocks, preserve_index=False), BLK_PATH)
    log.info("Saved updated block_trades.parquet with side + signed_impact_bps")

    # Summary stats
    classified = blocks[blocks["side_label"] != "unknown"].dropna(
        subset=["slippage_bps", "signed_impact_bps"]
    ).copy()

    bins   = [200_000, 500_000, 1_000_000, 5_000_000, np.inf]
    labels = ["$200k-500k", "$500k-1M", "$1M-5M", "$5M+"]
    classified["dv_bucket"] = pd.cut(
        classified["dollar_value"], bins=bins, labels=labels, right=False
    )

    def agg(g):
        si = g["signed_impact_bps"]
        return pd.Series({
            "n":               len(g),
            "med_slip_bps":    g["slippage_bps"].median(),
            "med_impact_bps":  si.median(),
            "mean_impact_bps": si.mean(),
            "p95_impact_bps":  si.quantile(0.95),
            "std_impact_bps":  si.std(),
        })

    def agg_dv(g):
        si = g["signed_impact_bps"]
        return pd.Series({
            "n":               len(g),
            "buy%":            100 * (g["side_label"] == "buy").mean(),
            "med_slip_bps":    g["slippage_bps"].median(),
            "med_impact_bps":  si.median(),
            "mean_impact_bps": si.mean(),
            "p95_impact_bps":  si.quantile(0.95),
            "std_impact_bps":  si.std(),
        })

    print(f"\n{'='*60}")
    print("SIDE DISTRIBUTION")
    print(f"{'='*60}")
    for label, n in blocks["side_label"].value_counts().items():
        print(f"  {label:<10}  {n:>7,}  ({100*n/len(blocks):.1f}%)")

    print(f"\n{'='*60}")
    print("SIGNED IMPACT BY SIDE  (bps)")
    print(f"{'='*60}")
    by_side = classified.groupby("side_label", observed=True).apply(agg, include_groups=False)
    print(by_side.to_string(float_format="{:.3f}".format))

    print(f"\n{'='*60}")
    print("SIGNED IMPACT BY DOLLAR BUCKET  (bps)")
    print(f"{'='*60}")
    by_dv = classified.groupby("dv_bucket", observed=True).apply(agg_dv, include_groups=False)
    print(by_dv.to_string(float_format="{:.3f}".format))

    print(f"\n{'='*60}")
    print("MEDIAN SIGNED IMPACT (bps) — bucket x side")
    print(f"{'='*60}")
    pivot = classified.pivot_table(
        values="signed_impact_bps", index="dv_bucket", columns="side_label",
        aggfunc="median", observed=True,
    )[["buy", "sell"]]
    print(pivot.to_string(float_format="{:.3f}".format))

    # Plot

    DV_ORDER = [l for l in labels if l in classified["dv_bucket"].cat.categories]
    COLORS   = {"buy": "#3a86ff", "sell": "#e63946"}
    CLIP     = 15

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("AAPL Block-Trade Signed Price Impact  (tick-test classification)",
                 fontsize=12, fontweight="bold")

    # Left: grouped box plot by dollar bucket x side
    ax = axes[0]
    width, gap, group_gap = 0.32, 0.06, 0.35
    xticks, xlabels = [], []

    for i, bucket in enumerate(DV_ORDER):
        base = i * (2 * width + gap + group_gap)
        for j, side in enumerate(["buy", "sell"]):
            x    = base + j * (width + gap)
            data = classified.loc[
                (classified["dv_bucket"] == bucket) & (classified["side_label"] == side),
                "signed_impact_bps",
            ].clip(-CLIP, CLIP).dropna()

            bp = ax.boxplot(
                data, positions=[x], widths=width, patch_artist=True,
                medianprops=dict(color="black", linewidth=1.8),
                whiskerprops=dict(linewidth=1.0),
                capprops=dict(linewidth=1.0),
                flierprops=dict(marker=".", markersize=1.5, alpha=0.2, linestyle="none"),
            )
            bp["boxes"][0].set_facecolor(COLORS[side])
            bp["boxes"][0].set_alpha(0.75)

        xticks.append(base + (width + gap) / 2)
        xlabels.append(bucket)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_ylabel(f"Signed impact (bps, clipped +/-{CLIP})", fontsize=10)
    ax.set_title("By dollar bucket", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(handles=[Patch(facecolor=c, label=s, alpha=0.75) for s, c in COLORS.items()],
              fontsize=9)

    # Right: heatmap of median signed impact
    ax2 = axes[1]
    heat = pivot[["buy", "sell"]].reindex(DV_ORDER).astype(float)
    lim  = max(abs(heat.values[~np.isnan(heat.values)]).max(), 0.01)
    im   = ax2.imshow(heat.values, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["Buy", "Sell"], fontsize=10)
    ax2.set_yticks(range(len(DV_ORDER)))
    ax2.set_yticklabels(DV_ORDER, fontsize=9)
    ax2.set_title("Median signed impact (bps)\nbucket x side", fontsize=11)
    for ri, row in enumerate(heat.values):
        for ci, val in enumerate(row):
            if not np.isnan(val):
                ax2.text(ci, ri, f"{val:.3f}", ha="center", va="center",
                         fontsize=11, fontweight="bold",
                         color="white" if abs(val) > lim * 0.5 else "black")
    plt.colorbar(im, ax=ax2, shrink=0.7, label="bps")

    plt.tight_layout()
    plt.savefig("signed_impact.png", dpi=150, bbox_inches="tight")
    log.info("Plot saved -> signed_impact.png")


def run_distribution_comparison():
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

    # Load
    df = pd.read_parquet("data/lit_buy_features_v2.parquet")
    x_all = df["impact_vwap_bps"].to_numpy(dtype=np.float64)
    df["q5"] = pd.qcut(df["roll_spread_500"], q=5, labels=False, duplicates="drop")

    print(f"Loaded {len(x_all):,} trades")
    print(f"impact: mean={x_all.mean():+.4f}  med={np.median(x_all):+.4f}  "
          f"std={x_all.std():.4f}  p1={np.percentile(x_all,1):.2f}  "
          f"p99={np.percentile(x_all,99):.2f}\n")


    # ── Distribution definitions ──────────────────────────────────────────────────

    SQRT_PI2 = np.sqrt(np.pi / 2.0)   # sqrt(pi/2) ≈ 1.2533

    # (4) RI helpers
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


    # ── Unified fit-and-score routine ─────────────────────────────────────────────

    def fit_all(x, label=""):
        n = len(x)
        rows = []

        # (1) Normal
        mu_n, sig_n = stats.norm.fit(x)
        ll_n  = stats.norm.logpdf(x, mu_n, sig_n).sum()
        ks_n  = stats.ks_1samp(x, lambda t: stats.norm.cdf(t, mu_n, sig_n))
        rows.append(dict(model="Normal", k=2, params=(mu_n, sig_n),
                         ll=ll_n, ks=ks_n.statistic, ksp=ks_n.pvalue,
                         pdf_fn=lambda g, p=(mu_n,sig_n): stats.norm.pdf(g, *p)))

        # (2) Laplace
        loc_l, b_l = stats.laplace.fit(x)
        ll_l  = stats.laplace.logpdf(x, loc_l, b_l).sum()
        ks_l  = stats.ks_1samp(x, lambda t: stats.laplace.cdf(t, loc_l, b_l))
        rows.append(dict(model="Laplace", k=2, params=(loc_l, b_l),
                         ll=ll_l, ks=ks_l.statistic, ksp=ks_l.pvalue,
                         pdf_fn=lambda g, p=(loc_l,b_l): stats.laplace.pdf(g, *p)))

        # (3) Skew-normal
        a_s, xi_s, om_s = stats.skewnorm.fit(x)
        ll_s  = stats.skewnorm.logpdf(x, a_s, xi_s, om_s).sum()
        ks_s  = stats.ks_1samp(x, lambda t: stats.skewnorm.cdf(t, a_s, xi_s, om_s))
        rows.append(dict(model="Skew-normal", k=3, params=(a_s, xi_s, om_s),
                         ll=ll_s, ks=ks_s.statistic, ksp=ks_s.pvalue,
                         pdf_fn=lambda g, p=(a_s,xi_s,om_s): stats.skewnorm.pdf(g, *p)))

        # (4) RI (half-normal / half-Laplace)
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


    # ── Fit overall + 5 quintiles ─────────────────────════════════════════════────

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

    # Ranking summary
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


    # ── Plot: 6 panels (overall + 5 quintiles) ───────────────════════════════════

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


def run_impact_by_size():
    """
    Bin AAPL block trades by dollar_value into 20 equal-count quantile buckets.
    For each bin, plot mean and median VWAP-bar signed impact (bps) for buys and
    sells separately. X-axis is log dollar value.

    Signed impact convention:
      buy  -> impact_vwap_bps as-is   (positive = price rose after block = adverse)
      sell -> -impact_vwap_bps         (positive = price fell after block = adverse)
    This puts both sides on the same "adverse impact" axis for easy comparison,
    and also plots them raw (unsigned) on a second panel for symmetry inspection.
    """

    # Load
    df = pq.read_table(
        "data/block_trades.parquet",
        columns=["dollar_value", "side_label", "impact_vwap_bps"],
    ).to_pandas()

    df = df[(df["side_label"] != "unknown") & df["impact_vwap_bps"].notna()].copy()

    # Signed impact: positive = price moved against initiator
    df["signed_impact"] = np.where(
        df["side_label"] == "buy",
         df["impact_vwap_bps"],
        -df["impact_vwap_bps"],
    )

    N_BINS = 20

    # Bin by dollar_value using equal-count quantiles (on the full population)
    df["bin"] = pd.qcut(df["dollar_value"], q=N_BINS, labels=False, duplicates="drop")

    def bin_stats(sub):
        """Aggregate mean, median, SEM, and count per bin per side."""
        rows = []
        for side in ("buy", "sell"):
            g = sub[sub["side_label"] == side]
            for bn, grp in g.groupby("bin", observed=True):
                si = grp["signed_impact"]
                rows.append({
                    "side":        side,
                    "bin":         bn,
                    "n":           len(grp),
                    "dv_mean":     grp["dollar_value"].mean(),
                    "dv_geo":      np.exp(np.log(grp["dollar_value"]).mean()),  # geometric mean
                    "mean":        si.mean(),
                    "median":      si.median(),
                    "sem":         si.sem(),
                    "q25":         si.quantile(0.25),
                    "q75":         si.quantile(0.75),
                })
        return pd.DataFrame(rows)

    bin_stats_df = bin_stats(df)
    buy  = bin_stats_df[bin_stats_df["side"] == "buy" ].sort_values("bin").reset_index(drop=True)
    sell = bin_stats_df[bin_stats_df["side"] == "sell"].sort_values("bin").reset_index(drop=True)

    # Print table
    print(f"\n{'='*78}")
    print(f"{'BIN':>4}  {'DV (geom mean $k)':>18}  "
          f"{'BUY n':>7} {'buy mean':>9} {'buy med':>8}  "
          f"{'SELL n':>7} {'sell mean':>9} {'sell med':>8}")
    print(f"{'='*78}")
    for _, br, sr in zip(range(N_BINS), buy.itertuples(), sell.itertuples()):
        print(f"{int(br.bin):>4}  ${br.dv_geo/1000:>16,.0f}k  "
              f"{br.n:>7,} {br.mean:>9.3f} {br.median:>8.3f}  "
              f"{sr.n:>7,} {sr.mean:>9.3f} {sr.median:>8.3f}")

    # Chart
    BUY_C  = "#2563eb"   # blue
    SELL_C = "#dc2626"   # red

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    fig.suptitle(
        "AAPL block-trade price impact vs. trade size\n"
        "20 equal-count quantile bins  |  VWAP-bar method",
        fontsize=13, fontweight="bold", y=1.01,
    )

    # Panel A: signed adverse impact (both sides on same axis)
    ax = axes[0]

    for side, sdf, color in [("Buy", buy, BUY_C), ("Sell", sell, SELL_C)]:
        x = sdf["dv_geo"].values

        # SEM band around mean
        ax.fill_between(x,
                        sdf["mean"] - 1.96 * sdf["sem"],
                        sdf["mean"] + 1.96 * sdf["sem"],
                        color=color, alpha=0.12)

        ax.plot(x, sdf["mean"],   color=color, lw=2.2, marker="o", ms=5,
                label=f"{side} — mean")
        ax.plot(x, sdf["median"], color=color, lw=1.4, marker="s", ms=4,
                ls="--", alpha=0.85, label=f"{side} — median")

    ax.axhline(0, color="black", lw=0.9, ls="--", alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("Trade size (dollar value, log scale)", fontsize=11)
    ax.set_ylabel("Signed adverse impact (bps)\n(positive = price moved against initiator)", fontsize=10)
    ax.set_title("Adverse impact by side", fontsize=11)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"${x/1e6:.1f}M" if x >= 1e6 else f"${x/1e3:.0f}k"
    ))
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.25)
    ax.grid(True, which="minor", alpha=0.1)

    # Panel B: raw (unsigned) impact — buy positive, sell negative
    ax2 = axes[1]

    buy_raw  = df[df["side_label"] == "buy" ].groupby("bin", observed=True)["impact_vwap_bps"]
    sell_raw = df[df["side_label"] == "sell"].groupby("bin", observed=True)["impact_vwap_bps"]

    # Merge onto bin x-positions from combined stats
    dv_x = (
        df.groupby("bin", observed=True)["dollar_value"]
          .apply(lambda s: np.exp(np.log(s).mean()))
          .sort_index()
    )

    for raw_grp, color, label_stem in [
        (buy_raw,  BUY_C,  "Buy"),
        (sell_raw, SELL_C, "Sell"),
    ]:
        means   = raw_grp.mean().reindex(dv_x.index)
        medians = raw_grp.median().reindex(dv_x.index)
        sems    = raw_grp.sem().reindex(dv_x.index)
        x       = dv_x.values

        ax2.fill_between(x, means - 1.96*sems, means + 1.96*sems,
                         color=color, alpha=0.12)
        ax2.plot(x, means,   color=color, lw=2.2, marker="o", ms=5,
                 label=f"{label_stem} — mean")
        ax2.plot(x, medians, color=color, lw=1.4, marker="s", ms=4,
                 ls="--", alpha=0.85, label=f"{label_stem} — median")

    ax2.axhline(0, color="black", lw=0.9, ls="--", alpha=0.5)
    ax2.set_xscale("log")
    ax2.set_xlabel("Trade size (dollar value, log scale)", fontsize=11)
    ax2.set_ylabel("Raw impact_vwap_bps\n(buy +, sell −, zero = no price move)", fontsize=10)
    ax2.set_title("Raw directional impact by side", fontsize=11)
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"${x/1e6:.1f}M" if x >= 1e6 else f"${x/1e3:.0f}k"
    ))
    ax2.legend(fontsize=9, ncol=2)
    ax2.grid(True, alpha=0.25)
    ax2.grid(True, which="minor", alpha=0.1)

    plt.tight_layout()
    plt.savefig("impact_by_size.png", dpi=160, bbox_inches="tight")
    print("\nChart saved -> impact_by_size.png")


def run_impact_distribution():
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

    # Load
    df = pd.read_parquet("data/lit_buy_features_v2.parquet")
    print(f"Loaded {len(df):,} trades")

    x_all    = df["impact_vwap_bps"].to_numpy(dtype=np.float64)
    spread   = df["roll_spread_500"].to_numpy(dtype=np.float64)

    p1, p99 = np.percentile(x_all, [1, 99])
    print(f"impact_vwap_bps: mean={x_all.mean():.4f}  median={np.median(x_all):.4f}  "
          f"std={x_all.std():.4f}  [p1={p1:.2f}, p99={p99:.2f}]")


    # Distribution helpers
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


    def fit_dist(x, label=""):
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


    # Overall fit
    print("\n=== Fitting overall distribution ===")
    mu_all, sig_all, rho_all, _ = fit_dist(x_all, "overall")

    # 5-bin fits by roll_spread_500 quintile
    print("\n=== Fitting per spread quintile ===")
    df["q5"] = pd.qcut(df["roll_spread_500"], q=5, labels=False, duplicates="drop")

    bin_results = []   # (mean_spread, mu, sigma, rho)
    for q in range(5):
        mask  = df["q5"] == q
        x_bin = df.loc[mask, "impact_vwap_bps"].to_numpy(dtype=np.float64)
        s_mid = df.loc[mask, "roll_spread_500"].mean()
        label = f"Q{q+1} spread~{s_mid:.2f}bps"
        mu_q, sig_q, rho_q, _ = fit_dist(x_bin, label)
        bin_results.append((s_mid, mu_q, sig_q, rho_q))

    spread_mids = np.array([r[0] for r in bin_results])
    mus         = np.array([r[1] for r in bin_results])
    sigmas      = np.array([r[2] for r in bin_results])
    rhos        = np.array([r[3] for r in bin_results])


    # Figure
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


    # Panel 1: Overall histogram + fitted density
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


    # Panel 2: 5 conditional fitted densities overlaid
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


    # Panel 3: sigma and rho vs mean spread
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


def run_impact_drivers():
    """
    For each block trade, compute trailing 1-minute volatility and volume
    from per-day tick files, then plot impact vs these drivers.
    Output: aapl_impact_drivers.png
    """

    # 1. Load buy block trades
    bt = pd.read_parquet("data/block_trades.parquet",
        columns=["date", "timestamp_ns", "side_label", "impact_vwap_bps"])
    buys = bt[(bt.side_label == "buy") & bt.impact_vwap_bps.notna()].copy()
    buys = buys.sort_values(["date", "timestamp_ns"]).reset_index(drop=True)
    print(f"{len(buys):,} AAPL buy block trades across {buys.date.nunique()} days")

    ONE_MIN_NS = 60 * 1_000_000_000

    # 2. Vectorized trailing features per day via prefix sums
    results = []

    for date, grp in buys.groupby("date"):
        tick_path = f"data/AAPL/{date}.parquet"
        if not os.path.exists(tick_path):
            continue
        ticks = pd.read_parquet(tick_path, columns=["sip_timestamp", "price", "size"])
        ticks = ticks.sort_values("sip_timestamp").reset_index(drop=True)

        ts  = ticks["sip_timestamp"].to_numpy(dtype=np.int64)
        px  = ticks["price"].to_numpy(dtype=np.float64)
        sz  = ticks["size"].to_numpy(dtype=np.int64)
        N   = len(ts)

        # prefix sums (length N+1, index 0 = 0)
        cum_sz  = np.empty(N + 1, dtype=np.int64);   cum_sz[0]  = 0; np.cumsum(sz, out=cum_sz[1:])
        diffs   = np.diff(px)                           # length N-1
        cum_d   = np.empty(N, dtype=np.float64);  cum_d[0]  = 0.0; np.cumsum(diffs, out=cum_d[1:])
        cum_d2  = np.empty(N, dtype=np.float64);  cum_d2[0] = 0.0; np.cumsum(diffs**2, out=cum_d2[1:])

        block_ts = grp["timestamp_ns"].to_numpy(dtype=np.int64)
        lo_idx   = np.searchsorted(ts, block_ts - ONE_MIN_NS, side="left")
        hi_idx   = np.searchsorted(ts, block_ts,              side="left")

        # volume: shares in [lo, hi)
        vol_shares = (cum_sz[hi_idx] - cum_sz[lo_idx]).astype(float)

        # volatility: std of price diffs in window [lo, hi)
        n_diffs  = (hi_idx - lo_idx - 1).astype(float)   # may be 0 or negative
        sum_d    = cum_d[hi_idx - 1]  - cum_d[lo_idx]
        sum_d2   = cum_d2[hi_idx - 1] - cum_d2[lo_idx]

        # guard: need >=2 ticks (>=1 diff) for std; n_diffs>=2 for ddof=1
        mask_ok  = (n_diffs >= 2) & (hi_idx > 0)
        variance = np.where(
            mask_ok,
            (sum_d2 - sum_d**2 / np.where(n_diffs > 0, n_diffs, 1)) /
            np.where(n_diffs > 1, n_diffs - 1, 1),
            np.nan
        )
        variance = np.where(variance < 0, 0.0, variance)   # float noise guard
        trail_std = np.where(mask_ok, np.sqrt(variance), np.nan)

        # set vol_shares to nan when window is empty
        vol_shares = np.where(hi_idx > lo_idx, vol_shares, np.nan)

        tmp = grp[["impact_vwap_bps"]].copy()
        tmp["trail_vol_px"]     = trail_std   # price-change std in $ per share
        tmp["trail_vol_shares"] = vol_shares
        results.append(tmp)
        print(f"  {date}: {len(grp):4d} buys, tick rows={N:,}")

    buys_feat = pd.concat(results)
    print(f"\nTotal: {len(buys_feat):,} trades")

    AAPL_MID = 190.0
    buys_feat["trail_vol_bps"] = buys_feat["trail_vol_px"] / AAPL_MID * 1e4

    valid = buys_feat.dropna(subset=["trail_vol_bps", "trail_vol_shares", "impact_vwap_bps"])
    print(f"Trades with full features: {len(valid):,}")
    print(f"  trail_vol_bps  — median={valid.trail_vol_bps.median():.2f}  p99={valid.trail_vol_bps.quantile(0.99):.2f}")
    print(f"  trail_vol_sh   — median={valid.trail_vol_shares.median():.0f}  p99={valid.trail_vol_shares.quantile(0.99):.0f}")

    # 3. Helper: binned mean
    CLIP = 15

    def make_bins(df, xcol, q=30):
        df = df.copy()
        df["impact_c"] = df["impact_vwap_bps"].clip(-CLIP, CLIP)
        df["bin"] = pd.qcut(df[xcol], q=q, duplicates="drop")
        binned = df.groupby("bin", observed=True).agg(
            x_mid=(xcol, "median"),
            mean_impact=("impact_vwap_bps", "mean"),
        ).reset_index()
        binned["mean_impact"] = binned["mean_impact"].clip(-CLIP, CLIP)
        return df, binned

    # 4. Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # — Panel A: impact vs trailing volatility —
    df_v, bin_v = make_bins(valid, "trail_vol_bps")
    sample_v = df_v.sample(n=min(10_000, len(df_v)), random_state=42)

    ax = axes[0]
    ax.scatter(sample_v["trail_vol_bps"], sample_v["impact_c"],
               alpha=0.10, s=7, color="#2563eb", linewidths=0,
               label=f"Trades (10k sample of {len(df_v):,})")
    ax.plot(bin_v["x_mid"], bin_v["mean_impact"],
            color="#f59e0b", lw=2.2, marker="o", ms=5, zorder=5,
            label="Binned mean (all trades)")
    ax.axhline(0, color="black", lw=1, ls="--", alpha=0.55)
    ax.set_xlabel("Trailing 1-min volatility (price-change std, bps)", fontsize=11)
    ax.set_ylabel("VWAP-bar impact (bps, clipped ±15)", fontsize=11)
    ax.set_title("Impact vs Trailing Volatility", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    # — Panel B: impact vs trailing volume —
    df_s, bin_s = make_bins(valid, "trail_vol_shares")
    sample_s = df_s.sample(n=min(10_000, len(df_s)), random_state=42)

    ax = axes[1]
    ax.scatter(sample_s["trail_vol_shares"], sample_s["impact_c"],
               alpha=0.10, s=7, color="#2563eb", linewidths=0,
               label=f"Trades (10k sample of {len(df_s):,})")
    ax.plot(bin_s["x_mid"], bin_s["mean_impact"],
            color="#f59e0b", lw=2.2, marker="o", ms=5, zorder=5,
            label="Binned mean (all trades)")
    ax.axhline(0, color="black", lw=1, ls="--", alpha=0.55)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v/1e6:.1f}M" if v >= 1e6 else f"{v/1e3:.0f}k"))
    ax.set_xlabel("Trailing 1-min volume (shares)", fontsize=11)
    ax.set_ylabel("VWAP-bar impact (bps, clipped ±15)", fontsize=11)
    ax.set_title("Impact vs Trailing Volume", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    fig.suptitle("AAPL buy block trades — impact drivers", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("aapl_impact_drivers.png", dpi=150, bbox_inches="tight")
    print("saved -> aapl_impact_drivers.png")


def run_lit_buy_scatter():
    """
    AAPL lit-only buy block trades — impact vs size scatter with OLS binned mean.
    Output: aapl_lit_buy_impact.png
    """

    # TRF / ORF exchange IDs — confirmed via client.get_exchanges()
    DARK_IDS = {4, 6, 16, 62, 201, 202, 203}

    df = pd.read_parquet("data/block_trades.parquet",
        columns=["side_label", "exchange", "dollar_value", "impact_vwap_bps"])

    lit  = df[~df["exchange"].isin(DARK_IDS)]
    buys = lit[(lit["side_label"] == "buy") & lit["impact_vwap_bps"].notna()].copy()

    n = len(buys)
    med = buys["impact_vwap_bps"].median()
    mn  = buys["impact_vwap_bps"].mean()
    std = buys["impact_vwap_bps"].std()

    print(f"Lit buy block trades : {n:,}")
    print(f"  Median impact (bps): {med:+.4f}")
    print(f"  Mean   impact (bps): {mn:+.4f}")
    print(f"  Std    impact (bps): {std:.4f}")

    CLIP = 15
    buys["impact_c"] = buys["impact_vwap_bps"].clip(-CLIP, CLIP)

    sample = buys.sample(n=min(10_000, n), random_state=42)

    buys["bin"] = pd.qcut(buys["dollar_value"], q=30, duplicates="drop")
    binned = buys.groupby("bin", observed=True).agg(
        dv_mid=("dollar_value", "median"),
        mean_impact=("impact_vwap_bps", "mean"),
    ).reset_index()
    binned["mean_impact"] = binned["mean_impact"].clip(-CLIP, CLIP)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(sample["dollar_value"], sample["impact_c"],
               alpha=0.10, s=7, color="#2563eb", linewidths=0,
               label=f"Trades (10k sample of {n:,})")

    ax.plot(binned["dv_mid"], binned["mean_impact"],
            color="#f59e0b", lw=2.2, marker="o", ms=5, zorder=5,
            label="Binned mean (all trades)")

    ax.axhline(0, color="black", lw=1, ls="--", alpha=0.55)
    ax.set_xscale("log")
    ax.set_xlabel("Trade size (dollar value)", fontsize=12)
    ax.set_ylabel("VWAP-bar impact (bps, clipped +-15)", fontsize=12)
    ax.set_title("AAPL lit-only buy block trades — impact vs size", fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"${v/1e6:.1f}M" if v >= 1e6 else f"${v/1e3:.0f}k"))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig("aapl_lit_buy_impact.png", dpi=150, bbox_inches="tight")
    print("saved -> aapl_lit_buy_impact.png")


def run_scatter_buys():
    """
    AAPL block-trade slippage scatter (all buy trades, not lit-filtered).
    Output: aapl_buy_slippage_scatter.png
    """

    df = pd.read_parquet("data/block_trades.parquet",
        columns=["side_label", "dollar_value", "impact_vwap_bps"])
    buys = df[(df.side_label == "buy") & df.impact_vwap_bps.notna()].copy()
    print(f"{len(buys):,} AAPL buy block trades")

    CLIP = 15
    buys["impact_c"] = buys["impact_vwap_bps"].clip(-CLIP, CLIP)

    # 10k random sample for scatter points
    sample = buys.sample(n=min(10_000, len(buys)), random_state=42)

    # Binned mean on full data (30 equal-count bins)
    buys["bin"] = pd.qcut(buys["dollar_value"], q=30, duplicates="drop")
    binned = buys.groupby("bin", observed=True).agg(
        dv_mid=("dollar_value", "median"),
        mean_impact=("impact_vwap_bps", "mean"),
        n=("impact_vwap_bps", "count"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(sample["dollar_value"], sample["impact_c"],
               alpha=0.10, s=7, color="#2563eb", linewidths=0,
               label=f"Trades (10k sample of {len(buys):,})")

    ax.plot(binned["dv_mid"], binned["mean_impact"].clip(-CLIP, CLIP),
            color="#f59e0b", lw=2.2, marker="o", ms=5, zorder=5,
            label="Binned mean (all trades)")

    ax.axhline(0, color="black", lw=1, ls="--", alpha=0.55)
    ax.set_xscale("log")
    ax.set_xlabel("Trade size (dollar value)", fontsize=12)
    ax.set_ylabel("Slippage — VWAP-bar impact (bps, clipped ±15)", fontsize=12)
    ax.set_title("AAPL block-trade slippage: buys", fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"${v/1e6:.1f}M" if v >= 1e6 else f"${v/1e3:.0f}k"))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig("aapl_buy_slippage_scatter.png", dpi=150, bbox_inches="tight")
    print("saved -> aapl_buy_slippage_scatter.png")


def run_adjacent_impact():
    """
    For each block trade, find the immediately adjacent trades in the full sequence
    (no time window) and compute:
        impact_bps = (post_price_adj - pre_price_adj) / pre_price_adj * 10000

    This measures how much the block actually moved the market, uncontaminated by
    the ±1-second window used in slippage_bps.

    Output: adds pre_adj, post_adj, impact_bps to block_trades.parquet
            saves adjacent_impact.png
    """

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(__name__)

    DATA_DIR = Path("data")
    BLK_PATH = DATA_DIR / "block_trades.parquet"


    # Adjacent-price lookup for one ticker-day file

    def adjacent_prices(trade_path, blk_ts, blk_px, blk_sz):
        """
        Load one trade file, sort by (participant_timestamp, sequence_number),
        find each block trade's position via searchsorted, return
        (pre_prices, post_prices) as float64 arrays (NaN where unavailable).
        """
        tbl = pq.read_table(
            trade_path,
            columns=["participant_timestamp", "price", "size", "sequence_number"],
        )
        ts_arr    = tbl.column("participant_timestamp").to_pylist()
        seq_arr   = tbl.column("sequence_number").to_pylist()
        price_col = tbl.column("price")
        size_col  = tbl.column("size")

        # Sort order by (timestamp, sequence_number)
        order = np.lexsort((
            np.array(seq_arr,  dtype=np.int64),
            np.array(ts_arr,   dtype=np.int64),
        ))

        ts_s    = np.array(ts_arr,            dtype=np.int64)[order]
        price_s = price_col.to_pylist()
        price_s = np.array(price_s,           dtype=np.float64)[order]
        size_s  = size_col.to_pylist()
        size_s  = np.array(size_s,            dtype=np.int64)[order]
        n       = len(ts_s)

        del tbl, ts_arr, seq_arr, price_col, size_col, order
        gc.collect()

        # Vectorised searchsorted for all block trades at once
        lo_arr = np.searchsorted(ts_s, blk_ts, side="left")
        hi_arr = np.searchsorted(ts_s, blk_ts, side="right")

        pos_arr = lo_arr.copy()  # will be refined for multi-trade timestamps

        # Resolve multi-trade timestamps (rare) by matching price + size
        for i in np.where(hi_arr - lo_arr != 1)[0]:
            lo, hi = lo_arr[i], hi_arr[i]
            if lo >= hi:
                continue
            hits = np.where(
                (price_s[lo:hi] == blk_px[i]) & (size_s[lo:hi] == blk_sz[i])
            )[0]
            if len(hits):
                pos_arr[i] = lo + hits[0]

        # Adjacent prices
        pre_pos  = pos_arr - 1
        post_pos = pos_arr + 1

        safe_pre  = np.clip(pre_pos,  0, n - 1)
        safe_post = np.clip(post_pos, 0, n - 1)

        pre_prices  = np.where(pre_pos  >= 0,     price_s[safe_pre],  np.nan)
        post_prices = np.where(post_pos <  n,     price_s[safe_post], np.nan)

        del ts_s, price_s, size_s
        gc.collect()

        return pre_prices, post_prices


    # Main

    blocks = pq.read_table(BLK_PATH).to_pandas()
    log.info("Loaded %d block trades", len(blocks))

    blocks.sort_values(["ticker", "date", "timestamp_ns"], inplace=True, ignore_index=True)

    pre_adj  = np.full(len(blocks), np.nan)
    post_adj = np.full(len(blocks), np.nan)

    for (ticker, date_str), grp in blocks.groupby(["ticker", "date"]):
        trade_path = DATA_DIR / ticker / f"{date_str}.parquet"
        if not trade_path.exists():
            log.warning("Missing %s", trade_path)
            continue

        idx      = grp.index.to_numpy()
        blk_ts   = grp["timestamp_ns"].to_numpy(np.int64)
        blk_px   = grp["price"].to_numpy(np.float64)
        blk_sz   = grp["size"].to_numpy(np.int64)

        pre, post = adjacent_prices(trade_path, blk_ts, blk_px, blk_sz)
        pre_adj[idx]  = pre
        post_adj[idx] = post

        n_valid = (~np.isnan(pre) & ~np.isnan(post)).sum()
        log.info("  %s %s  %d/%d with both neighbors", ticker, date_str, n_valid, len(grp))

    blocks["pre_adj"]    = pre_adj
    blocks["post_adj"]   = post_adj
    with np.errstate(invalid="ignore", divide="ignore"):
        blocks["impact_bps"] = (
            (blocks["post_adj"] - blocks["pre_adj"]) / blocks["pre_adj"] * 10_000
        )

    pq.write_table(pa.Table.from_pandas(blocks, preserve_index=False), BLK_PATH)
    log.info("Saved block_trades.parquet with pre_adj / post_adj / impact_bps")

    # Analytics
    ana = blocks[
        (blocks["side_label"] != "unknown") &
        blocks["impact_bps"].notna()
    ].copy()

    bins   = [200_000, 500_000, 1_000_000, 5_000_000, np.inf]
    dv_labels = ["$200k-500k", "$500k-1M", "$1M-5M", "$5M+"]
    ana["dv_bucket"] = pd.cut(ana["dollar_value"], bins=bins, labels=dv_labels, right=False)

    def tbl(g):
        s = g["impact_bps"]
        return pd.Series({
            "n":         len(g),
            "median":    s.median(),
            "mean":      s.mean(),
            "p25":       s.quantile(0.25),
            "p75":       s.quantile(0.75),
            "p95":       s.quantile(0.95),
            "std":       s.std(),
        })

    print(f"\n{'='*65}")
    print("ADJACENT-TRADE IMPACT BY SIDE  (bps)")
    print(f"{'='*65}")
    by_side = ana.groupby("side_label", observed=True).apply(tbl, include_groups=False)
    print(by_side.to_string(float_format="{:.3f}".format))

    print(f"\n{'='*65}")
    print("ADJACENT-TRADE IMPACT BY DOLLAR BUCKET  (bps, both sides)")
    print(f"{'='*65}")
    by_dv = ana.groupby("dv_bucket", observed=True).apply(tbl, include_groups=False)
    print(by_dv.to_string(float_format="{:.3f}".format))

    print(f"\n{'='*65}")
    print("MEDIAN IMPACT (bps) — bucket x side")
    print(f"{'='*65}")
    pivot = ana.pivot_table(
        values="impact_bps", index="dv_bucket", columns="side_label",
        aggfunc="median", observed=True,
    )[["buy", "sell"]]
    print(pivot.to_string(float_format="{:.4f}".format))

    # Histogram
    CLIP = 30   # bps — cuts <0.1% of extremes for readability

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("AAPL Block-Trade Adjacent-Trade Impact", fontsize=12, fontweight="bold")

    # Left: overlapping histograms, buy vs sell
    ax = axes[0]
    COLORS = {"buy": "#3a86ff", "sell": "#e63946"}
    for side, color in COLORS.items():
        data = ana.loc[ana["side_label"] == side, "impact_bps"].clip(-CLIP, CLIP)
        ax.hist(data, bins=120, range=(-CLIP, CLIP),
                color=color, alpha=0.55, label=side, density=True)

    ax.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.6)
    ax.set_xlabel("Impact (bps, clipped ±30)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Buy vs Sell", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # Right: by dollar bucket, stacked/overlapping on log-y
    ax2 = axes[1]
    BUCKET_COLORS = ["#4878d0", "#ee854a", "#6acc65", "#d65f5f"]
    for bucket, color in zip(dv_labels, BUCKET_COLORS):
        data = ana.loc[ana["dv_bucket"] == bucket, "impact_bps"].clip(-CLIP, CLIP)
        if len(data) == 0:
            continue
        ax2.hist(data, bins=80, range=(-CLIP, CLIP),
                 color=color, alpha=0.55, label=bucket, density=True)

    ax2.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.6)
    ax2.set_yscale("log")
    ax2.set_xlabel("Impact (bps, clipped ±30)", fontsize=11)
    ax2.set_ylabel("Density (log)", fontsize=11)
    ax2.set_title("By dollar bucket", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("adjacent_impact.png", dpi=150, bbox_inches="tight")
    log.info("Plot saved -> adjacent_impact.png")


def run_sweep_clusters():
    """
    Find lit buy sweep clusters per day (100ms gap, same exchange) and plot
    cluster impact vs cluster dollar value.
    Output: aapl_sweep_impact.png
    """

    DARK_IDS     = {4, 6, 16, 62, 201, 202, 203}
    GAP_NS       = 100_000_000        # 100 ms in nanoseconds
    MIN_DOLLAR   = 200_000

    # 1. Find sweep clusters per day
    all_clusters = []
    total_raw_lit_buys = 0

    tick_files = sorted(glob("data/AAPL/*.parquet"))

    for path in tick_files:
        date = os.path.basename(path).replace(".parquet", "")

        # Read with pyarrow, extract numpy arrays directly — no pandas DataFrame
        tbl = pq.read_table(path, columns=["sip_timestamp", "price", "size", "exchange"])
        ts_all = tbl.column("sip_timestamp").to_numpy(zero_copy_only=False)          # int64
        px_all = tbl.column("price").to_numpy(zero_copy_only=False).astype(np.float32)
        sz_all = tbl.column("size").to_numpy(zero_copy_only=False).astype(np.int32)
        ex_all = tbl.column("exchange").to_numpy(zero_copy_only=False).astype(np.int16)
        del tbl; gc.collect()

        # Sort by sip_timestamp with numpy argsort (avoids pandas temp index allocs)
        order  = np.argsort(ts_all, kind="stable")
        ts_all = ts_all[order]
        px_all = px_all[order]
        sz_all = sz_all[order]
        ex_all = ex_all[order]
        del order; gc.collect()

        # Filter to lit exchanges only
        lit_mask = ~np.isin(ex_all, list(DARK_IDS))
        ts  = ts_all[lit_mask]
        px  = px_all[lit_mask]
        sz  = sz_all[lit_mask]
        ex  = ex_all[lit_mask]
        del ts_all, px_all, sz_all, ex_all, lit_mask; gc.collect()

        if len(ts) < 2:
            continue

        # Tick test: classify each lit trade as buy / sell
        dp      = np.diff(px, prepend=px[0])       # first trade gets dp=0
        raw_dir = np.sign(dp).astype(float)
        raw_dir[raw_dir == 0] = np.nan             # zero-ticks: forward-fill
        # pandas ffill handles NaN streaks correctly
        direction = pd.Series(raw_dir).ffill().fillna(1.0).to_numpy()  # leading zeros → buy

        # Keep only uptick (buy) trades
        buy_mask = direction > 0
        b_ts = ts[buy_mask]
        b_px = px[buy_mask]
        b_sz = sz[buy_mask]
        b_ex = ex[buy_mask]
        n_buys = buy_mask.sum()
        total_raw_lit_buys += n_buys

        if n_buys < 2:
            continue

        # Cluster: new cluster when gap > 100 ms OR exchange changes
        gap_arr  = np.empty(n_buys, dtype=np.int64)
        gap_arr[0] = GAP_NS + 1                   # first trade always starts new cluster
        gap_arr[1:] = np.diff(b_ts)
        ex_chg   = np.concatenate([[True], b_ex[1:] != b_ex[:-1]])
        new_cls  = (gap_arr > GAP_NS) | ex_chg
        cls_id   = new_cls.cumsum()               # 1-indexed cluster labels

        # Aggregate per cluster
        unique_cls, first_pos = np.unique(cls_id, return_index=True)
        last_pos  = np.concatenate([first_pos[1:] - 1, [n_buys - 1]])

        dv_arr        = b_px * b_sz
        cum_dv        = np.concatenate([[0], np.cumsum(dv_arr)])
        cum_sz        = np.concatenate([[0], np.cumsum(b_sz)])
        cluster_dv    = cum_dv[last_pos + 1] - cum_dv[first_pos]
        cluster_sh    = cum_sz[last_pos + 1] - cum_sz[first_pos]
        cluster_n     = last_pos - first_pos + 1
        cluster_fp    = b_px[first_pos]
        cluster_lp    = b_px[last_pos]
        cluster_ts    = b_ts[first_pos]
        cluster_ex    = b_ex[first_pos]
        cluster_span  = (b_ts[last_pos] - b_ts[first_pos]) / 1_000_000   # ns → ms
        with np.errstate(invalid="ignore", divide="ignore"):
            cluster_impact = (cluster_lp - cluster_fp) / cluster_fp * 10_000

        df_day = pd.DataFrame({
            "date":          date,
            "cluster_ts":    cluster_ts,
            "exchange":      cluster_ex,
            "n_fills":       cluster_n,
            "total_shares":  cluster_sh,
            "dollar_value":  cluster_dv,
            "first_price":   cluster_fp,
            "last_price":    cluster_lp,
            "impact_bps":    cluster_impact,
            "time_span_ms":  cluster_span,
        })
        all_clusters.append(df_day)
        print(f"  {date}: {len(df_day):5,} clusters from {n_buys:6,} lit buy ticks")

    clusters = pd.concat(all_clusters, ignore_index=True)
    large    = clusters[clusters["dollar_value"] >= MIN_DOLLAR].copy()

    print(f"\n{'='*55}")
    print(f"Clusters found (dv >= $200k)  : {len(large):,}")
    print(f"Average fills per cluster     : {large['n_fills'].mean():.2f}")
    print(f"Median cluster dollar value   : ${large['dollar_value'].median():,.0f}")
    print(f"{'='*55}")
    print(f"  (all sizes before filter)   : {len(clusters):,}")
    print(f"  Total lit buy ticks used    : {total_raw_lit_buys:,}")
    print(f"  Multi-fill clusters (n>1)   : {(large['n_fills'] > 1).sum():,}  "
          f"({100*(large['n_fills']>1).mean():.1f}%)")
    print(f"  Median fills per cluster    : {large['n_fills'].median():.0f}")
    print(f"  Median time span (ms)       : {large['time_span_ms'].median():.2f}")
    print(f"  Median total shares         : {large['total_shares'].median():,.0f}")

    imp = large["impact_bps"]
    print(f"\nCluster impact_bps (dv >= $200k):")
    print(f"  Median : {imp.median():+.4f}")
    print(f"  Mean   : {imp.mean():+.4f}")
    print(f"  Std    : {imp.std():.4f}")

    # 2. Scatter + binned means
    CLIP    = 20
    N_BINS  = 30

    large["impact_c"] = large["impact_bps"].clip(-CLIP, CLIP)
    large["dv_bin"]   = pd.qcut(large["dollar_value"], q=N_BINS, duplicates="drop")

    binned = (
        large.groupby("dv_bin", observed=True)
        .agg(
            dv_mid     =("dollar_value", "median"),
            mean_impact=("impact_bps",   "mean"),
            n          =("impact_bps",   "count"),
        )
        .reset_index(drop=True)
    )
    binned["mean_impact_c"] = binned["mean_impact"].clip(-CLIP, CLIP)

    sample = large.sample(n=min(10_000, len(large)), random_state=42)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Raw scatter
    ax.scatter(
        sample["dollar_value"], sample["impact_c"],
        alpha=0.09, s=7, color="#2563eb", linewidths=0,
        label=f"Sweep clusters (10k sample of {len(large):,})",
    )

    # Binned mean line + dots
    ax.plot(
        binned["dv_mid"], binned["mean_impact_c"],
        color="#f59e0b", lw=2.2, zorder=5,
    )
    ax.scatter(
        binned["dv_mid"], binned["mean_impact_c"],
        color="#f59e0b", s=55, zorder=6, edgecolors="white", linewidths=0.8,
        label=f"Binned mean ({N_BINS} quantile bins)",
    )

    ax.axhline(0, color="black", lw=1, ls="--", alpha=0.55)
    ax.set_xscale("log")
    ax.set_xlabel("Cluster total dollar value  (log scale)", fontsize=12)
    ax.set_ylabel(
        "Cluster impact_bps  =  (last_px − first_px) / first_px × 10,000\n"
        "(clipped ±20 bps)",
        fontsize=11,
    )
    ax.set_title(
        "AAPL lit buy sweep clusters — cluster_impact_bps vs cluster_dollar_value\n"
        f"100-ms window, same exchange  |  {len(large):,} clusters with dv ≥ $200k",
        fontsize=12, fontweight="bold",
    )
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: (f"${v/1e6:.1f}M" if v >= 1e6 else f"${v/1e3:.0f}k")
    ))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig("aapl_sweep_impact.png", dpi=150, bbox_inches="tight")
    print("saved -> aapl_sweep_impact.png")


def run_sweep_forward():
    """
    Forward price impact of AAPL lit buy sweep clusters.

    For each cluster with dv >= $200k, find the first trade in the full tick stream
    (lit + dark) that occurs >= T seconds after the cluster's last fill timestamp.
    Forward impact = (forward_price - first_fill_price) / first_fill_price * 10,000 bps.

    Horizons: T = 1s and T = 5s.

    Uses the same clustering logic as sweep_clusters.py.  The forward price lookup
    keeps the full (lit + dark) sorted tick arrays alive so searchsorted can find
    the first post-horizon trade regardless of exchange.

    Output: aapl_sweep_forward.png  (two panels: 1s and 5s)
    """

    DARK_IDS   = {4, 6, 16, 62, 201, 202, 203}
    GAP_NS     = 100_000_000          # 100 ms cluster gap
    MIN_DOLLAR = 200_000
    NS_1S      = 1_000_000_000        # 1 second in ns
    NS_5S      = 5_000_000_000        # 5 seconds in ns

    all_clusters = []
    tick_files   = sorted(glob("data/AAPL/*.parquet"))

    for path in tick_files:
        date = os.path.basename(path).replace(".parquet", "")

        # Load ALL trades (lit + dark) for forward price lookup
        tbl    = pq.read_table(path, columns=["sip_timestamp", "price", "size", "exchange"])
        ts_raw = tbl.column("sip_timestamp").to_numpy(zero_copy_only=False)
        px_raw = tbl.column("price").to_numpy(zero_copy_only=False).astype(np.float32)
        sz_raw = tbl.column("size").to_numpy(zero_copy_only=False).astype(np.int32)
        ex_raw = tbl.column("exchange").to_numpy(zero_copy_only=False).astype(np.int16)
        del tbl; gc.collect()

        # Sort all trades once
        order  = np.argsort(ts_raw, kind="stable")
        ts_all = ts_raw[order]     # ALL trades, sorted — kept for forward lookup
        px_all = px_raw[order]
        sz_all = sz_raw[order]
        ex_all = ex_raw[order]
        del ts_raw, px_raw, sz_raw, ex_raw, order; gc.collect()

        N_ALL = len(ts_all)

        # Lit-only arrays for clustering
        lit_mask = ~np.isin(ex_all, list(DARK_IDS))
        ts = ts_all[lit_mask]
        px = px_all[lit_mask]
        sz = sz_all[lit_mask]
        ex = ex_all[lit_mask]

        if len(ts) < 2:
            del ts_all, px_all, sz_all, ex_all, lit_mask; gc.collect()
            continue

        # Tick test on lit trades
        dp        = np.diff(px, prepend=px[0])
        raw_dir   = np.sign(dp).astype(float)
        raw_dir[raw_dir == 0] = np.nan
        direction = pd.Series(raw_dir).ffill().fillna(1.0).to_numpy()

        buy_mask = direction > 0
        b_ts = ts[buy_mask];  b_px = px[buy_mask]
        b_sz = sz[buy_mask];  b_ex = ex[buy_mask]
        n_buys = buy_mask.sum()

        if n_buys < 2:
            del ts_all, px_all, sz_all, ex_all; gc.collect()
            continue

        # Cluster assignment
        gap_arr     = np.empty(n_buys, dtype=np.int64)
        gap_arr[0]  = GAP_NS + 1
        gap_arr[1:] = np.diff(b_ts)
        ex_chg      = np.concatenate([[True], b_ex[1:] != b_ex[:-1]])
        new_cls     = (gap_arr > GAP_NS) | ex_chg
        cls_id      = new_cls.cumsum()

        # Per-cluster aggregation
        unique_cls, first_pos = np.unique(cls_id, return_index=True)
        last_pos  = np.concatenate([first_pos[1:] - 1, [n_buys - 1]])

        dv_arr       = b_px.astype(np.float64) * b_sz
        cum_dv       = np.concatenate([[0.0], np.cumsum(dv_arr)])
        cum_sz       = np.concatenate([[0],   np.cumsum(b_sz)])

        cluster_dv   = cum_dv[last_pos + 1] - cum_dv[first_pos]
        cluster_sh   = cum_sz[last_pos + 1] - cum_sz[first_pos]
        cluster_n    = last_pos - first_pos + 1
        cluster_fp   = b_px[first_pos].astype(np.float64)
        cluster_lp   = b_px[last_pos].astype(np.float64)
        cluster_fts  = b_ts[first_pos]    # first-fill timestamp
        cluster_lts  = b_ts[last_pos]     # last-fill timestamp
        cluster_span = (cluster_lts - cluster_fts) / 1_000_000   # ms
        with np.errstate(invalid="ignore", divide="ignore"):
            cluster_impact = (cluster_lp - cluster_fp) / cluster_fp * 10_000

        # Forward price lookup (ALL trades, vectorised)
        def forward_impact(horizon_ns):
            """
            For each cluster, find the first trade in ts_all at
            >= cluster_lts + horizon_ns, then compute
            (that_price - first_fill_price) / first_fill_price * 10_000.
            Returns float64 array of length n_clusters; NaN where no trade found.
            """
            targets = cluster_lts + horizon_ns
            idx     = np.searchsorted(ts_all, targets, side="left")   # first trade >= target
            valid   = idx < N_ALL
            safe    = np.where(valid, idx, 0)                          # avoid OOB index
            fwd_px  = np.where(valid, px_all[safe].astype(np.float64), np.nan)
            with np.errstate(invalid="ignore", divide="ignore"):
                fwd_imp = np.where(
                    valid,
                    (fwd_px - cluster_fp) / cluster_fp * 10_000,
                    np.nan,
                )
            return fwd_imp

        fwd_1s = forward_impact(NS_1S)
        fwd_5s = forward_impact(NS_5S)

        del ts_all, px_all, sz_all, ex_all; gc.collect()

        df_day = pd.DataFrame({
            "date":           date,
            "n_fills":        cluster_n,
            "total_shares":   cluster_sh,
            "dollar_value":   cluster_dv,
            "first_price":    cluster_fp,
            "last_price":     cluster_lp,
            "impact_bps":     cluster_impact,
            "time_span_ms":   cluster_span,
            "fwd_impact_1s":  fwd_1s,
            "fwd_impact_5s":  fwd_5s,
        })
        all_clusters.append(df_day)
        n_valid_1s = np.sum(~np.isnan(fwd_1s))
        n_valid_5s = np.sum(~np.isnan(fwd_5s))
        print(f"  {date}: {len(df_day):5,} clusters  "
              f"fwd_1s={n_valid_1s:,}  fwd_5s={n_valid_5s:,}")

    # Combine and filter
    clusters = pd.concat(all_clusters, ignore_index=True)
    large    = clusters[clusters["dollar_value"] >= MIN_DOLLAR].copy()

    print(f"\n{'='*60}")
    print(f"Clusters (dv >= $200k)         : {len(large):,}")
    print(f"  with valid 1s forward price  : {large['fwd_impact_1s'].notna().sum():,}")
    print(f"  with valid 5s forward price  : {large['fwd_impact_5s'].notna().sum():,}")
    print(f"{'='*60}")

    for label, col in [("1s forward impact", "fwd_impact_1s"),
                       ("5s forward impact", "fwd_impact_5s"),
                       ("within-cluster impact", "impact_bps")]:
        s = large[col].dropna()
        print(f"\n{label} (bps):")
        print(f"  Mean   : {s.mean():+.4f}")
        print(f"  Median : {s.median():+.4f}")
        print(f"  Std    : {s.std():.4f}")


    # Plot
    CLIP   = 15
    N_BINS = 30

    def make_panel(ax, dv, impact, clip, n_bins, color_scatter, color_bin,
                   title, ylabel):
        valid = ~np.isnan(impact)
        dv_v  = dv[valid]
        imp_v = impact[valid]
        imp_c = np.clip(imp_v, -clip, clip)

        # quantile bins on dollar value
        dv_s  = pd.Series(dv_v)
        bins  = pd.qcut(dv_s, q=n_bins, duplicates="drop")
        binned = (
            pd.DataFrame({"dv": dv_v, "imp": imp_v, "bin": bins})
            .groupby("bin", observed=True)
            .agg(dv_mid=("dv", "median"), mean_imp=("imp", "mean"), n=("imp", "count"))
            .reset_index(drop=True)
        )
        binned["mean_imp_c"] = binned["mean_imp"].clip(-clip, clip)

        # sample for scatter (cap at 10k)
        rng   = np.random.default_rng(42)
        idx_s = rng.choice(len(dv_v), size=min(10_000, len(dv_v)), replace=False)

        ax.scatter(dv_v[idx_s], imp_c[idx_s],
                   alpha=0.08, s=6, color=color_scatter, linewidths=0,
                   label=f"Clusters ({len(dv_v):,} with valid forward price)")

        ax.plot(binned["dv_mid"], binned["mean_imp_c"],
                color=color_bin, lw=2.2, zorder=5)
        ax.scatter(binned["dv_mid"], binned["mean_imp_c"],
                   color=color_bin, s=55, zorder=6,
                   edgecolors="white", linewidths=0.8,
                   label=f"Binned mean ({n_bins} quantile bins)")

        ax.axhline(0, color="black", lw=1.0, ls="--", alpha=0.55)
        ax.set_xscale("log")
        ax.set_xlabel("Cluster dollar value  (log scale)", fontsize=10.5)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(
            lambda v, _: (f"${v/1e6:.1f}M" if v >= 1e6 else f"${v/1e3:.0f}k")))
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # annotate mean and median
        s = pd.Series(imp_v)
        ax.text(0.97, 0.97,
                f"mean  {s.mean():+.3f} bps\nmedian {s.median():+.3f} bps",
                transform=ax.transAxes, fontsize=8.5, family="monospace",
                ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          edgecolor="#94a3b8", alpha=0.9))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    make_panel(
        ax1,
        large["dollar_value"].to_numpy(),
        large["fwd_impact_1s"].to_numpy(),
        clip=CLIP, n_bins=N_BINS,
        color_scatter="#2563eb", color_bin="#f59e0b",
        title="1-second forward impact\n"
              "(next trade >= 1s after last fill vs first fill price)",
        ylabel="forward_impact_bps  (clipped ±15)",
    )

    make_panel(
        ax2,
        large["dollar_value"].to_numpy(),
        large["fwd_impact_5s"].to_numpy(),
        clip=CLIP, n_bins=N_BINS,
        color_scatter="#16a34a", color_bin="#dc2626",
        title="5-second forward impact\n"
              "(next trade >= 5s after last fill vs first fill price)",
        ylabel="forward_impact_bps  (clipped ±15)",
    )

    fig.suptitle(
        "AAPL lit buy sweep clusters — forward price impact vs cluster size\n"
        f"100-ms window, same-exchange clusters  |  dv >= $200k  |  {len(large):,} clusters",
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig("aapl_sweep_forward.png", dpi=150, bbox_inches="tight")
    print("\nsaved -> aapl_sweep_forward.png")


def run_coin_raw_ols_plot():
    """
    COIN block trades: raw slippage (pre-tick-test) vs trade size with OLS fit.
    slippage_bps = (post_price - pre_price) / pre_price * 10_000  — no sign flipping.
    Output: coin_raw_ols.png
    """

    # Load block trades, filter to COIN
    blocks = pd.read_parquet("data/block_trades.parquet")
    coin = blocks[blocks["ticker"] == "COIN"].dropna(subset=["slippage_bps"]).copy()
    print(f"COIN block trades with slippage: {len(coin):,}")

    x = coin["dollar_value"].values / 1e6  # trade size in $M for readability
    y = coin["slippage_bps"].values

    # OLS fit: slippage = beta0 + beta1 * dollar_value
    X = np.column_stack([x, np.ones(len(x))])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    print(f"OLS: slippage = {beta[0]:+.4f} * size_M {beta[1]:+.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6.5))

    ax.scatter(x, y, s=10, alpha=0.3, color="#dc2626", edgecolors="none", zorder=2,
               label=f"COIN blocks (n={len(coin):,})")

    xg = np.linspace(x.min(), x.max(), 300)
    yg = beta[0] * xg + beta[1]
    ax.plot(xg, yg, color="black", lw=2.2, zorder=4,
            label=f"OLS: {beta[0]:+.4f}x {beta[1]:+.4f}")

    ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.5, zorder=1)

    ax.set_xlabel("Trade Size ($M)", fontsize=12)
    ax.set_ylabel("Slippage (bps)", fontsize=12)
    ax.set_title("COIN OLS Fit", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Cap y-axis at 1st/99th percentile
    y_lo, y_hi = np.percentile(y, [1, 99])
    margin = (y_hi - y_lo) * 0.1
    ax.set_ylim(y_lo - margin, y_hi + margin)

    plt.tight_layout()
    plt.savefig("coin_raw_ols.png", dpi=150, bbox_inches="tight")
    print("Saved -> coin_raw_ols.png")


# Main

if __name__ == "__main__":
    run_classify_sides()
    run_distribution_comparison()
    run_impact_by_size()
    run_impact_distribution()
    run_impact_drivers()
    run_lit_buy_scatter()
    run_scatter_buys()
    run_adjacent_impact()
    run_sweep_clusters()
    run_sweep_forward()
    run_coin_raw_ols_plot()
