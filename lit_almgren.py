import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.optimize import curve_fit

# ── 1. Load lit buy block trades ──────────────────────────────────────────────
DARK_IDS = {4, 6, 16, 62, 201, 202, 203}
AAPL_MID = 190.0
ONE_MIN_NS = 60 * 1_000_000_000

bt = pd.read_parquet("data/block_trades.parquet",
    columns=["date", "timestamp_ns", "side_label", "exchange",
             "size", "impact_vwap_bps"])
buys = bt[
    (~bt["exchange"].isin(DARK_IDS)) &
    (bt["side_label"] == "buy") &
    bt["impact_vwap_bps"].notna()
].copy()
buys = buys.sort_values(["date", "timestamp_ns"]).reset_index(drop=True)
print(f"{len(buys):,} lit buy block trades across {buys.date.nunique()} days")

# ── 2. Vectorised trailing features via prefix sums (same as impact_drivers) ──
trail_vol_shares = np.full(len(buys), np.nan)
trail_vol_bps    = np.full(len(buys), np.nan)

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

    cum_sz = np.empty(N + 1, dtype=np.int64);  cum_sz[0] = 0
    np.cumsum(sz, out=cum_sz[1:])

    diffs   = np.diff(px)
    cum_d   = np.empty(N, dtype=np.float64);  cum_d[0]  = 0.0
    np.cumsum(diffs, out=cum_d[1:])
    cum_d2  = np.empty(N, dtype=np.float64);  cum_d2[0] = 0.0
    np.cumsum(diffs**2, out=cum_d2[1:])

    block_ts = grp["timestamp_ns"].to_numpy(dtype=np.int64)
    lo_idx   = np.searchsorted(ts, block_ts - ONE_MIN_NS, side="left")
    hi_idx   = np.searchsorted(ts, block_ts,              side="left")

    vol_sh   = (cum_sz[hi_idx] - cum_sz[lo_idx]).astype(float)
    n_diffs  = (hi_idx - lo_idx - 1).astype(float)
    sum_d    = cum_d[hi_idx - 1]  - cum_d[lo_idx]
    sum_d2   = cum_d2[hi_idx - 1] - cum_d2[lo_idx]

    mask_ok  = (n_diffs >= 2) & (hi_idx > 0)
    variance = np.where(
        mask_ok,
        (sum_d2 - sum_d**2 / np.where(n_diffs > 0, n_diffs, 1)) /
        np.where(n_diffs > 1, n_diffs - 1, 1),
        np.nan,
    )
    variance = np.where(variance < 0, 0.0, variance)
    px_std   = np.where(mask_ok, np.sqrt(variance), np.nan)

    trail_vol_shares[grp.index] = np.where(hi_idx > lo_idx, vol_sh, np.nan)
    trail_vol_bps[grp.index]    = px_std / AAPL_MID * 1e4

    print(f"  {date}: {len(grp):4d} lit buys")

buys["trail_vol_shares"] = trail_vol_shares
buys["trail_vol_bps"]    = trail_vol_bps

# ── 3. Participation rate ─────────────────────────────────────────────────────
buys["prate"] = buys["size"] / buys["trail_vol_shares"]

valid = buys.dropna(subset=["prate", "trail_vol_bps", "impact_vwap_bps"]).copy()
valid = valid[(valid["prate"] > 0) & (valid["prate"] <= 1.0) &
              (valid["trail_vol_bps"] > 0)].copy()
print(f"\n{len(valid):,} trades with valid prate and sigma")
print(f"  prate  — median={valid.prate.median():.4f}  p95={valid.prate.quantile(0.95):.4f}  max={valid.prate.max():.4f}")
print(f"  sigma  — median={valid.trail_vol_bps.median():.3f} bps")

# ── 4. Binned mean ────────────────────────────────────────────────────────────
CLIP = 15
valid["impact_c"] = valid["impact_vwap_bps"].clip(-CLIP, CLIP)
valid["bin"] = pd.qcut(valid["prate"], q=30, duplicates="drop")
binned = valid.groupby("bin", observed=True).agg(
    x_mid=("prate",           "median"),
    mean_impact=("impact_vwap_bps", "mean"),
    mean_sigma=("trail_vol_bps",    "mean"),
    n=("impact_vwap_bps",          "count"),
).reset_index()
binned["mean_impact_c"] = binned["mean_impact"].clip(-CLIP, CLIP)

# ── 5. Almgren model fit on binned means ─────────────────────────────────────
# impact = kappa * sigma * prate^beta
# Fitting on individual trades fails: per-trade noise (std ~4.7 bps) >> signal
# (~0.5 bps). Instead fit the binned means, which average out idiosyncratic noise.
# Weight each bin by sqrt(n) to down-weight sparse large-prate bins.

def almgren(X, kappa, beta):
    prate, sigma = X
    return kappa * sigma * np.power(prate, beta)

# Only use bins with positive mean impact and enough trades for stability
fit_bins = binned[(binned["mean_impact"] > 0) & (binned["n"] >= 20)].copy()

X_fit = (fit_bins["x_mid"].to_numpy(), fit_bins["mean_sigma"].to_numpy())
y_fit = fit_bins["mean_impact"].to_numpy()
w_fit = np.sqrt(fit_bins["n"].to_numpy())

p0     = [0.5, 0.5]
bounds = ([1e-4, 0.05], [50.0, 3.0])

try:
    popt, pcov = curve_fit(almgren, X_fit, y_fit, p0=p0, bounds=bounds,
                           sigma=1.0/w_fit, absolute_sigma=False, maxfev=20_000)
    kappa, beta = popt
    perr = np.sqrt(np.diag(pcov))
    print(f"\nAlmgren fit on binned means (n_bins={len(fit_bins)}):")
    print(f"  impact = kappa * sigma * prate^beta")
    print(f"  kappa = {kappa:.4f}  (SE={perr[0]:.4f})")
    print(f"  beta  = {beta:.4f}  (SE={perr[1]:.4f})")
    print(f"  (canonical Almgren 2005 finds beta ~ 0.6; sqrt model = 0.5)")
    fit_ok = True
except Exception as e:
    print(f"\ncurve_fit failed: {e}")
    fit_ok = False

# ── 6. Plot ───────────────────────────────────────────────────────────────────
sample = valid.sample(n=min(10_000, len(valid)), random_state=42)
sigma_med = valid["trail_vol_bps"].median()

fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(sample["prate"], sample["impact_c"],
           alpha=0.10, s=7, color="#2563eb", linewidths=0,
           label=f"Trades (10k sample of {len(valid):,})")

ax.plot(binned["x_mid"], binned["mean_impact_c"],
        color="#f59e0b", lw=2.2, marker="o", ms=5, zorder=5,
        label="Binned mean (all trades)")

if fit_ok:
    x_curve = np.linspace(valid["prate"].quantile(0.01),
                          valid["prate"].quantile(0.99), 300)
    y_curve = kappa * sigma_med * x_curve**beta
    ax.plot(x_curve, y_curve.clip(-CLIP, CLIP),
            color="#dc2626", lw=2.2, ls="--", zorder=6,
            label=f"Almgren fit @ median sigma\n"
                  f"kappa={kappa:.3f}, beta={beta:.3f}")

ax.axhline(0, color="black", lw=1, ls="--", alpha=0.55)
ax.set_xscale("log")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(
    lambda v, _: f"{v:.1%}" if v >= 0.01 else f"{v:.2%}"))
ax.set_xlabel("Participation rate (block size / trailing 1-min volume)", fontsize=11)
ax.set_ylabel("VWAP-bar impact (bps, clipped +-15)", fontsize=11)
ax.set_title("AAPL lit buy block trades — Almgren market impact model",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig("aapl_lit_almgren.png", dpi=150, bbox_inches="tight")
print("saved -> aapl_lit_almgren.png")
