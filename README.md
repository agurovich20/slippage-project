# Equity Trade Slippage Modeling

A semiparametric model for forecasting institutional equity trade slippage using tick-level data from Polygon.io. The model produces full predictive distributions — not just point estimates — using a two-stage location-scale framework with Laplace-distributed errors. Validated on pooled data from six stocks with no per-stock hyperparameter tuning.

---

## Overview

Trading costs are often the binding constraint on whether a strategy is viable. This project builds a model that, given trade and market features at execution time, produces:

- A point estimate of slippage (in basis points)
- Calibrated prediction intervals (e.g., the 90% interval contains the true slippage 90% of the time)
- The probability that slippage exceeds any threshold

The key finding is that **correcting the loss function matters more than model complexity**: switching from MSE to LAD (least absolute deviations) — the MLE under Laplace-distributed errors — outperforms every MSE-trained model on Huber loss, including models with significantly more parameters.

---

## Results

| Model | AAPL R² | AAPL MAE (bps) | AAPL Huber (δ=1) | COIN R² | COIN MAE (bps) |
|---|---|---|---|---|---|
| OLS | 3.3% | 1.76 | 1.32 | 5.8% | 4.02 |
| Linear-LAD | −22.9% | **1.70** | **1.16** | — | — |
| RF-MSE | 8.6% | 1.74 | 1.32 | 9.7% | 3.72 |
| RF-MAE | **8.9%** | 1.49 | 1.11 | — | — |
| XGB-MSE | 8.3% | 1.78 | 1.36 | **14.1%** | 3.69 |
| XGB-MAE | 6.6% | **1.46** | **1.08** | — | — |

**Calibration (90% nominal level):**

| Stock | Two-Stage Linear | Two-Stage XGBoost |
|---|---|---|
| AAPL | 90.9% | 91.5% |
| COIN | 92.6% | 91.3% |
| AMZN | — | 90.5% |
| AMD | — | 90.4% |
| NVDA | — | 92.0% |
| TSLA | — | 92.3% |

One set of hyperparameters, tuned on AAPL only, generalizes to all six stocks without retuning.

---

## Data

Sourced via the [Polygon.io](https://polygon.io) API ($79/month tier). Filters applied:

- Block trades only: notional value ≥ $200,000
- Lit exchanges only (dark pool/OTC venues excluded — no measurable slippage)
- Six stocks: AAPL, COIN, NVDA, AMD, AMZN, TSLA
- Train: January–August 2024 (or June–August for AAPL feature tuning)
- Test: September 2024

| Stock | Train | Test | Mean \|Slippage\| |
|---|---|---|---|
| AAPL | 35,020 | 9,152 | 2.1 bps |
| AMZN | 15,632 | 3,949 | 2.0 bps |
| AMD | 17,227 | 3,046 | 2.6 bps |
| NVDA | 125,835 | 33,435 | 2.6 bps |
| TSLA | 65,329 | 21,115 | 2.8 bps |
| COIN | 15,681 | 959 | 5.8 bps |

---

## Features

**Slippage target:** percentage change from the VWAP of the 1-second bar preceding the block trade to the execution price.

| Feature | Description |
|---|---|
| `dollar_value` | Block trade notional (price × size) |
| `log_dollar_value` | Log of above |
| `participation_rate` | Block DV / trailing 1-minute DV |
| `roll_spread_500` | Roll (1984) spread estimate from 500 prior ticks: `2√(−Cov(ΔP_t, ΔP_{t-1}))` |
| `roll_vol_500` | Realized volatility from 500 prior tick-to-tick log returns |
| `exchange_id` | Exchange identifier |
| `time_of_day` | Seconds since 9:30 ET |
| `day_of_week` | Day of week (0=Monday) |

---

## Model

**Why Laplace?** Fitting candidate distributions to signed slippage confirms that errors follow a Laplace distribution (AIC gap vs Normal: ~24,000 points on AAPL; generalized Gaussian shape parameter ≈ 0.98 vs 2.0 for Normal). Under Laplace errors, LAD is the MLE and is asymptotically twice as efficient as OLS.

**Two-stage GAMLSS:**

1. **Location model:** XGBoost trained with `reg:absoluteerror` estimates the conditional median `μ(x)`
2. **Scale model:** A second XGBoost trained with `reg:squarederror` on the absolute residuals `|y − μ̂|` estimates the conditional Laplace scale `b(x)`

From these, any prediction interval follows from the Laplace CDF:

```
[max(μ̂ − b̂·ln(1/α), 0),  μ̂ + b̂·ln(1/α)]
```

**Pooled model:** All six stocks are normalized by each stock's in-sample feature and target medians, then combined into a single training set. A single model fit on this pooled dataset maintains per-stock MAE within 0.06 bps and 90% coverage between 89–93% across all tickers.

---

## Repository Structure

```
.
├── pipeline.py          # Data fetching (Polygon.io), block trade identification,
│                        # cross-stock validation pipeline, summary figures
├── build_features.py    # Feature construction — 500-trade rolling Roll spread and vol
├── eda.py               # Exploratory analysis: distributions, impact by size/regime,
│                        # sweep clusters, buy/sell classification
├── ols_baseline.py      # OLS baselines, residual diagnostics, temporal holdout
├── gridsearch.py        # Hyperparameter search for XGBoost and RF (MSE and LAD)
├── xgb_models.py        # XGBoost SHAP analysis, PDPs, ICE curves, RF comparison
├── rf_models.py         # Random Forest SHAP analysis, ICE curves (overfitting evidence)
├── gamlss_models.py     # Two-stage location-scale models, RS iterations, pooled GAMLSS
├── analysis.py          # Coverage tables, Huber loss comparison, model comparison CV,
│                        # prediction interval figures, SHAP beeswarm
├── coin_analysis.py     # COIN distribution fits, bootstrap CIs, slippage breakdown
├── calibration.py       # Calibration curves for all models and baselines
├── data/                # Parquet files (tick data, features, gridsearch results)
└── medium_article_v2.md # Write-up
```

---

## Setup

**Python 3.10+** required.

```bash
pip install numpy pandas pyarrow xgboost scikit-learn statsmodels matplotlib scipy shap polygon-api-client
```

**API key:** Replace `API_KEY` in `pipeline.py::run_fetch_trades()` and `run_get_exchanges()` with your own Polygon.io key before fetching new data. The processed feature files in `data/` are already included.

---

## Running

Each file is independently executable. Functions are prefixed `run_` and called from a `__main__` block:

```bash
# Rebuild features (requires raw tick data in data/AAPL/, data/COIN/, etc.)
python build_features.py

# Run all analysis in order
python eda.py
python ols_baseline.py
python gridsearch.py
python xgb_models.py
python rf_models.py
python gamlss_models.py
python analysis.py
python coin_analysis.py
python calibration.py

# Full cross-stock pipeline (fetches data, builds features, fits model for all 6 stocks)
python pipeline.py
```

Individual functions can also be called directly:

```python
from analysis import run_model_comparison_v2
run_model_comparison_v2()
```

---

## Key References

- Roll, R. (1984). A Simple Implicit Measure of the Effective Bid-Ask Spread. *Journal of Finance*, 39(4), 1127–1139.
- Lee, C.M. & Ready, M.J. (1991). Inferring Trade Direction from Intraday Data. *Journal of Finance*, 46(2), 733–746.
- Rashkovich, V. & Iogansen, A. (2022). Occam's Razor for Bond Trade Costs. *Journal of Fixed Income*, 31(3).
- Isichenko, M. (2021). *Quantitative Portfolio Management*. Wiley.

---

## Author

Ari Gurovich — [github.com/agurovich20](https://github.com/agurovich20)

Full write-up: [medium_article_v2.md](medium_article_v2.md)
