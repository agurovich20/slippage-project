# Equity Trade Slippage Modeling
### Ari Gurovich

In endeavoring to develop a robust trading strategy, one must eventually contend with the simple fact that trading is not free. Trading costs eat into and can exceed trading alpha, making it of great practical importance to filter out trades whose expected costs exceed expected alpha. These trading costs decompose into explicit fees and implicit costs. Explicit fees, such as brokerage commissions, are known in advance and are typically small for institutional traders. Implicit costs, or slippage, are the focus of this project and are defined as the price displacement between order placement and execution. Slippage further bifurcates into a temporary price displacement component and a permanent market impact component. The temporary price displacement arises from high-frequency traders anticipating incoming volume, insufficient market liquidity to absorb the trade, or simple market drift and volatility during execution latency. Permanent market impact, the non-reverting component of slippage, represents the market learning from the trade.

In this project, I developed a semiparametric GAMLSS-style model to forecast trade slippage in equity markets. The model makes an empirically validated assumption that errors are Laplace distributed, and uses classical machine learning methods to estimate parameters conditional on trade and market features. Given these features, the model produces a full slippage distribution, including a point estimate, calibrated prediction intervals, and the probability of exceeding any slippage threshold. This can be used to systematically filter out trades where costs exceed alpha. Validated on pooled data from six stocks with no per-stock tuning, the model is very well calibrated (within 3 percentage points of accuracy) in the most operationally useful thresholds for trade filtering (≤45% and ≥85%).

A detailed write-up of this project is available on [Medium](https://medium.com/@arigurovich/predicting-slippage-in-equity-markets-using-probabilistic-machine-learning-2797ab3fb9c0)

---

## Repository Structure

```
.
├── pipeline.py          # Data fetching (Polygon.io), block trade identification,
│                        # cross-stock validation pipeline, summary figures
├── build_features.py    # Feature construction: 500-trade rolling Roll spread and vol, etc.
├── eda.py               # Exploratory analysis: distributions, slippage by size/regime,
│                        # sweep clusters, etc.
├── ols_baseline.py      # OLS baselines, residual diagnostics, temporal holdout
├── gridsearch.py        # Hyperparameter search for XGBoost and RF (MSE and LAD)
├── xgb_models.py        # XGBoost SHAP analysis, PDPs, ICE curves, RF comparison
├── rf_models.py         # Random Forest SHAP analysis, ICE curves (overfitting evidence)
├── gamlss_models.py     # Two-stage location-scale models, RS iterations, pooled GAMLSS
├── analysis.py          # Coverage tables, Huber loss comparison, model comparison CV,
│                        # prediction interval figures, SHAP beeswarm
├── coin_analysis.py     # COIN distribution fits, bootstrap CIs, slippage breakdown
├── calibration.py       # Calibration curves for all models and baselines
└── data/                # Parquet files (tick data, features, gridsearch results)
```

---

## Setup

**Python 3.10+** required.

```bash
pip install numpy pandas pyarrow xgboost scikit-learn statsmodels matplotlib scipy shap polygon-api-client
```

**API key:** Replace `API_KEY` in `pipeline.py` (`run_fetch_trades` and `run_get_exchanges`) with your own Polygon.io key before fetching new data. The processed feature files in `data/` are already included so the analysis scripts can be run without re-fetching.

---

## Running

Each file is independently executable. Functions are prefixed `run_` and called from a `__main__` block at the bottom of each file:

```bash
# Rebuild features (requires raw tick data in data/AAPL/, data/COIN/, etc.)
python build_features.py

# Run analysis scripts in order
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

## Data

Sourced via the [Polygon.io](https://polygon.io) API ($79/month tier). 

| Stock | Train | Test | Mean \|Slippage\| | Notes |
|---|---|---|---|---|
| AAPL | 35,020 | 9,152 | 2.1 bps | Primary; all hyperparameters tuned here |
| AMZN | 15,632 | 3,949 | 2.0 bps | Similar profile to AAPL |
| AMD | 17,227 | 3,046 | 2.6 bps | Mid-volatility |
| NVDA | 125,835 | 33,435 | 2.6 bps | Largest dataset |
| TSLA | 65,329 | 21,115 | 2.8 bps | High volatility |
| COIN | 15,681 | 959 | 5.8 bps | High volatility, high spread |

Train period: June–August 2024 (January–August for COIN). Test period: September 2024.

---

## Features

**Slippage target:** percentage change from the VWAP of the 1-second bar immediately preceding the block trade to the trade's execution price.

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

## Model and Results

Point estimation:
| Model | AAPL R² | AAPL MAE (bps) | AAPL Huber (δ=1) | COIN R² | COIN MAE (bps) |
|---|---|---|---|---|---|
| OLS | 3.3% | 1.76 | 1.32 | 5.8% | 4.02 |
| Linear-LAD | −22.9% | **1.70** | **1.16** | — | — |
| RF-MSE | 8.6% | 1.74 | 1.32 | 9.7% | 3.72 |
| RF-MAE | **8.9%** | 1.49 | 1.11 | — | — |
| XGB-MSE | 8.3% | 1.78 | 1.36 | **14.1%** | 3.69 |
| XGB-MAE | 6.6% | **1.46** | **1.08** | — | — |

**Two-stage approaches:**

1. **Location model:** XGBoost trained with `reg:absoluteerror` estimates the conditional median `μ(x)`
2. **Scale model:** A second XGBoost trained with `reg:squarederror` on the absolute residuals `|y − μ̂|` estimates the conditional Laplace scale `b(x)`

The two stages are separable because the Laplace MLE gradient with respect to location depends only on the signs of residuals, while the gradient with respect to scale depends only on their magnitudes. From the fitted `μ` and `b`, any prediction interval or exceedance probability follows directly from the Laplace CDF:

```
[max(μ̂ − b̂·ln(1/α), 0),  μ̂ + b̂·ln(1/α)]
```

**Calibration (90% nominal level):**

| Stock | Two-Stage Linear | Two-Stage XGBoost |
|---|---|---|
| AAPL | 90.9% | 91.5% |
| COIN | 92.6% | 91.3% |
| AMZN | — | 90.5% |
| AMD | — | 90.4% |
| NVDA | — | 92.0% |
| TSLA | — | 92.3% |

**Pooled model:** Features and targets are normalized by each stock's in-sample medians, then all six stocks are combined into a single training set. One set of hyperparameters, tuned on AAPL only, generalizes to all six stocks without retuning — maintaining per-stock MAE within 0.06 bps and 90% coverage between 89–93% across all tickers.

---


Ari Gurovich — [github.com/agurovich20](https://github.com/agurovich20)
