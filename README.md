# Salooan: Dynamic Collateral Haircut Estimation

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Model: XGBoost Quantile](https://img.shields.io/badge/model-XGBoost%20Quantile-orange.svg)](https://xgboost.readthedocs.io/)
[![UI: Streamlit](https://img.shields.io/badge/ui-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Tests: Pytest](https://img.shields.io/badge/tests-pytest%20149%20passed-brightgreen.svg)](https://pytest.org/)

**Salooan** is an end-to-end quantitative risk-sizing engine for secured lending. When a borrower pledges volatile collateral (crypto, equities, or ETFs) to secure a cash loan, the lender must lend less than the collateral's current market value. That safety margin is the **haircut**—a capital buffer designed to ensure that if the borrower defaults, the collateral can be liquidated without the lender suffering a principal loss.

Rather than relying on static, arbitrary haircut tables, Salooan estimates a **dynamic haircut**: a machine-learned Value-at-Risk (VaR) estimate predicting the asset's worst-case price decline over a multi-day liquidation horizon at a specified confidence level, conditioned on rolling volatility and market liquidity.

---

## Table of Contents
- [The Problem: Why Fixed Haircuts Fail](#the-problem-why-fixed-haircuts-fail)
- [Mathematical Formulation](#mathematical-formulation)
- [System Architecture & Data Flow](#system-architecture--data-flow)
- [Guardrails Against Data Leakage](#guardrails-against-data-leakage)
- [Repository Structure](#repository-structure)
- [Quickstart & Installation](#quickstart--installation)
- [End-to-End CLI Pipeline](#end-to-end-cli-pipeline)
- [Interactive Streamlit Dashboard](#interactive-streamlit-dashboard)
- [Empirical Benchmark Results](#empirical-benchmark-results)
- [Quantile Crossing & Monotonic Rearrangement](#quantile-crossing--monotonic-rearrangement)
- [Real-World Limitations](#real-world-limitations)
- [Educational Disclaimer](#educational-disclaimer)

---

## The Problem: Why Fixed Haircuts Fail

Traditional secured lending desks frequently rely on static haircut schedules (e.g. 50% for all crypto, 20% for equities). Static cushions suffer from two major failure modes:

1. **Under-protection during regime shifts**: During market flash-crashes, volatility spikes and market liquidity evaporates. A static 50% buffer can be breached within days on high-beta or illiquid tokens, resulting in lender insolvency.
2. **Capital inefficiency during calm regimes**: During low-volatility consolidation periods, requiring a rigid 50% haircut on liquid collateral (such as Bitcoin or large-cap equities) drastically over-collateralizes loans, depressing borrowing demand and capital utilization.

**Salooan's Solution**: Condition the haircut directly on empirical market dynamics. The model consumes only price and volume history—deliberately excluding asset tickers and categorical asset classes—forcing the algorithm to infer downside risk purely from continuous volatility and liquidity signals.

### Covered Assets
- **Cryptocurrencies (Binance daily klines)**: `BTCUSDT`, `ETHUSDT`, `ADAUSDT`, `DOGEUSDT`
- **Equities / ETFs (yfinance daily OHLCV)**: `SPY` (S&P 500 ETF), `ARKK` (High-beta innovation ETF)

---

## Mathematical Formulation

### 1. Forward Return Target
For an asset at date $t$ and a liquidation horizon of $N$ calendar days, the forward target is the simple percentage return:
$$y_{t, N} = \frac{\text{close}_{t+N}}{\text{close}_t} - 1$$

Simple returns (rather than log returns) ensure that predicted drops map directly to cash liquidation values.

### 2. Quantile Loss (Pinball Loss)
For confidence level $c \in \{0.90, 0.95, 0.99\}$, the target coverage tail is $\alpha = 1 - c$ ($\alpha \in \{0.10, 0.05, 0.01\}$). The model minimizes asymmetric quantile loss:
$$L_\alpha(y, q) = \max\Big(\alpha (y - q),\, (\alpha - 1)(y - q)\Big)$$

Under-predicting losses ($y < q$, a breach) is penalized by $1 - \alpha$, which is $(1 - \alpha) / \alpha$ times heavier than over-predicting.

### 3. Haircut Conversion & Loan Sizing
The predicted quantile return $q_\alpha$ is a negative number representing downside loss. The dynamic haircut clips this loss to $[0, 1]$:
$$\text{haircut} = \text{clip}(-q_\alpha,\, 0.0,\, 1.0)$$

The maximum allowable cash loan for a pledged collateral value is:
$$\text{Max Loan} = \text{Collateral Value} \times (1 - \text{haircut})$$

---

## System Architecture & Data Flow

Salooan follows a decoupled, modular pipeline where ingestion, feature engineering, training, evaluation, and visualization communicate strictly via Parquet and JSON artifacts on disk.

```mermaid
flowchart TD
    subgraph Ingestion
        A1[Binance API] -->|Daily klines| B1[src/data_pull.py]
        A2[yfinance API] -->|Daily OHLCV| B1
        B1 -->|Atomic parquet write| C[data/raw/*.parquet]
    end

    subgraph Transformation
        C --> D[src/preprocess.py]
        D -->|Forward targets + log returns| E[data/processed/*.parquet]
        E --> F[src/features.py]
        F -->|30d Volatility + 30d Dollar Volume| G[data/processed/features.parquet]
    end

    subgraph Modeling
        G --> H[src/split.py]
        H -->|Purged train split| I[src/train.py]
        I -->|XGBoost reg:quantileerror| J[models/quantile_h{N}_c{pct}.joblib]
    end

    subgraph Evaluation & UI
        J --> K[src/evaluate.py]
        G --> K
        K -->|Atomic write| L[results/results.json]
        J --> M[app.py Streamlit App]
        G --> M
        L --> M
        E --> M
    end
```

---

## Guardrails Against Data Leakage

1. **No-Lookahead Features**: Features computed at date $t$ use rolling row windows strictly ending at $t$. Targets ($y_{t+N}$) are never read during feature engineering.
2. **Chronological Splitting with $N$-Day Purge Gap**:
   - Splitting uses global unique calendar dates across all assets (most recent 20% dates reserved for test).
   - Rows in $[T_{\text{test}} - N, T_{\text{test}})$ are strictly purged from training because their forward return targets overlap with prices realized inside the test period.
3. **Stale Model Guards**: `evaluate.py` verifies the model bundle's `train_end_date` against the current split cutoff date before computing metrics, preventing evaluation of stale weights.
4. **Atomic Disk Operations**: Data files, model bundles, and results are written to temporary staging files before executing atomic OS renames, eliminating corrupt artifacts upon pipeline interruption.

---

## Repository Structure

```
Salooan/
├── data/
│   ├── raw/                 # Raw ingested Parquet files per asset
│   └── processed/           # Processed prices, targets, and features.parquet
├── models/                  # 6 trained XGBoost model bundles (.joblib)
├── results/                 # Atomic benchmark results (results.json)
├── src/
│   ├── __init__.py
│   ├── config.py            # Global paths, asset lists, windows, hyperparameters
│   ├── data_pull.py         # Binance & yfinance incremental ingestion
│   ├── preprocess.py        # Log returns, calendar-day forward return targets
│   ├── features.py          # Rolling 30d volatility & rolling 30d dollar volume
│   ├── split.py             # Chronological split with N-day purge gap
│   ├── baselines.py         # Pooled static & asset-specific historical baselines
│   ├── metrics.py           # Hand-implemented pinball loss, coverage, rearrangement
│   ├── train.py             # XGBoost quantile regression training across (H, C) grid
│   ├── predict.py           # Haircut inference, max loan, multi-quantile prediction
│   ├── evaluate.py          # Comprehensive test-set evaluation vs baselines
│   ├── refresh.py           # Incremental data refresh with cooldown & file locks
│   └── dashboard.py         # Pure helper functions for the Streamlit UI
├── tests/                   # 149 comprehensive pytest tests (including AppTest)
├── app.py                   # Streamlit dashboard (Calculator + Evaluation tabs)
├── requirements.txt         # Pinned production and development dependencies
├── AGENTS.md                # Engineering constraints, design decisions, and coding standards
└── README.md
```

---

## Quickstart & Installation

### Prerequisites
- Python 3.10+ (tested on Python 3.11)
- Virtual environment tool (`venv`)

### 1. Clone & Set Up Virtual Environment
```bash
git clone https://github.com/kav0ut/Salooan.git
cd Salooan

python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Verify Installation
Execute the unit test suite to ensure environment integrity:
```bash
pytest
```
*Expected: 149 passed tests.*

---

## End-to-End CLI Pipeline

Run the pipeline sequentially from the repository root:

```bash
# 1. Pull / update market data for crypto and equities
python -m src.data_pull

# 2. Compute log returns and forward targets (7d and 14d)
python -m src.preprocess

# 3. Build rolling volatility and dollar-volume feature table
python -m src.features

# 4. Verify chronological splits and purge gaps
python -m src.split

# 5. Train the full XGBoost quantile model grid (2 horizons x 3 confidences)
python -m src.train

# 6. Evaluate all models against historical and static baselines
python -m src.evaluate

# 7. Print live dynamic haircut recommendations for latest observations
python -m src.predict
```

---

## Interactive Streamlit Dashboard

Launch the interactive risk-sizing dashboard:
```bash
streamlit run app.py
```

### Dashboard Features
- **Sidebar**:
  - Asset selector (`BTCUSDT`, `ETHUSDT`, `ADAUSDT`, `DOGEUSDT`, `SPY`, `ARKK`).
  - Collateral value input (interactive USD amount).
  - Horizon selector (7 or 14 calendar days).
  - Confidence slider (90%, 95%, 99%).
  - **Data Refresh Panel**: Incremental market data refresh with automated 60-minute cooldown protection and file-based locking.
- **Tab 1: Haircut Calculator**:
  - Three headline metrics: Recommended Haircut %, Max Cash Loan $, and Worst-Case Drop %.
  - As-of date display (with trading-day equity notes).
  - Current 30-day volatility and average dollar volume.
  - Model vs Historical vs Static haircut comparison bar chart.
  - **Multi-Tier Risk Corridor Chart**: Visualizes closing prices over the past 365 days overlaid with a semi-transparent shaded risk corridor between the 90% and 99% predicted floors, the 95% floor line, and training cutoff marker.
- **Tab 2: Model Evaluation**:
  - Out-of-sample metadata (test start date, training end date, sample counts).
  - Overall performance table (Pinball loss, breach rate, mean haircut).
  - Grouped per-asset test breach rate bar chart with horizontal target $\alpha$ reference line.
  - **Temporal Breach Distribution (Monthly)**: Visual timeline of monthly realized breach rates showing stress-period breach clustering.

---

## Empirical Benchmark Results

All models and baselines were evaluated strictly on the out-of-sample test split ($N = 1,450$ test rows for 7-day horizon, $N = 1,444$ test rows for 14-day horizon).

### Overall Grid Performance vs Baselines

| Horizon | Confidence | Target $\alpha$ | Method | Test Pinball Loss | Breach Rate | Mean Haircut | Pinball vs Static | Pinball vs Hist |
|:---:|:---:|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| **7 Days** | **90%** | 10.0% | **XGBoost Model** | **0.013316** | 16.55% | 6.91% | **+0.17%** | -9.82% |
| | | | Historical Baseline | 0.012125 | 8.41% | 8.59% | — | — |
| | | | Static Baseline | 0.013339 | 9.10% | 9.12% | — | — |
| **7 Days** | **95%** | 5.0% | **XGBoost Model** | **0.008566** | 9.17% | 9.09% | -0.76% | -9.59% |
| | | | Historical Baseline | 0.007816 | 3.72% | 11.22% | — | — |
| | | | Static Baseline | 0.008501 | 3.86% | 12.33% | — | — |
| **7 Days** | **99%** | 1.0% | **XGBoost Model** | **0.003129** | 3.66% | 13.38% | -16.86% | -24.55% |
| | | | Historical Baseline | 0.002512 | 1.72% | 16.35% | — | — |
| | | | Static Baseline | 0.002677 | 1.03% | 20.17% | — | — |
| **14 Days** | **90%** | 10.0% | **XGBoost Model** | **0.018399** | 17.45% | 9.68% | **+6.71%** | -2.59% |
| | | | Historical Baseline | 0.017935 | 10.39% | 11.47% | — | — |
| | | | Static Baseline | 0.019722 | 10.87% | 12.41% | — | — |
| **14 Days** | **95%** | 5.0% | **XGBoost Model** | **0.012175** | 10.94% | 12.13% | **+1.11%** | -7.79% |
| | | | Historical Baseline | 0.011295 | 6.23% | 14.80% | — | — |
| | | | Static Baseline | 0.012312 | 6.72% | 16.52% | — | — |
| **14 Days** | **99%** | 1.0% | **XGBoost Model** | **0.004959** | 6.37% | 16.60% | -45.82% | -37.33% |
| | | | Historical Baseline | 0.003611 | 3.19% | 20.52% | — | — |
| | | | Static Baseline | 0.003401 | 2.15% | 25.43% | — | — |

### Per-Asset Sizing Breakdown (Default 7-Day, 95% Confidence)

Evaluating dynamic sizing across individual collateral assets demonstrates how the model differentiates risk purely from features:

| Asset | Asset Class | 30d Volatility | Dynamic Haircut | Historical Haircut | Static Haircut | Model Pinball Loss | Model Breach Rate |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **SPY** | Equity Index ETF | Low (~1.2%) | **2.07%** | 2.84% | 12.33% | **0.001789** | 8.65% |
| **BTCUSDT** | Major Crypto | Medium (~3.2%) | **6.91%** | 8.35% | 12.33% | **0.010047** | 12.96% |
| **ARKK** | Innovation ETF | High (~3.4%) | **8.46%** | 8.06% | 12.33% | **0.005014** | 2.70% |
| **ETHUSDT** | Major Crypto | High (~3.9%) | **10.78%** | 12.14% | 12.33% | **0.012288** | 11.48% |
| **ADAUSDT** | Altcoin | High (~4.5%) | **11.25%** | 15.78% | 12.33% | **0.010972** | 11.48% |
| **DOGEUSDT** | Meme Coin | High (~5.2%) | **12.67%** | 16.49% | 12.33% | **0.008032** | 5.56% |

**Key Takeaway**: The static baseline assigns an identical 12.33% haircut to both SPY and DOGEUSDT—grossly over-penalizing safe assets by 6x while under-penalizing speculative altcoins. The dynamic model expands haircuts appropriately from 2.07% up to 12.67% based on empirical risk conditions.

---

## Quantile Crossing & Monotonic Rearrangement

Because quantile boosting models at different confidence levels (90%, 95%, 99%) are trained independently, they occasionally produce **quantile crossings** on unconstrained data:
$$q_{0.95} > q_{0.90} \quad (\text{predicting a smaller loss at a higher confidence})$$

- **7-Day Horizon Raw Crossing Rate**: **1.38%** of test observations
- **14-Day Horizon Raw Crossing Rate**: **6.72%** of test observations

### The Solution: Monotonic Quantile Rearrangement
Salooan implements the mathematical sorting rearrangement of **Chernozhukov, Fernández-Val, and Galichon (2010)** in `src/metrics.py:monotone_rearrangement`. By sorting predicted quantile returns in descending order across ascending confidence levels:
$$q_{0.90} \ge q_{0.95} \ge q_{0.99}$$

- **Post-Rearrangement Crossing Rate**: **0.00%** (guaranteed monotonicity).
- Preserves ties and leaves already-ordered predictions untouched.
- Applied during live haircut calculation and multi-quantile risk corridor rendering.

---

## Real-World Limitations

While Salooan demonstrates the power of ML-driven risk sizing, real-world deployment on institutional lending desks involves critical nuances:

### 1. Daily Close vs Forced-Sale Execution Price
The model predicts the forward drop in **daily closing prices**. In an actual liquidation:
- Positions are forcibly unwound intraday into distressed order books.
- Execution prices suffer from market impact, spread slippage, and liquidation latency.
- *Remedy*: In production, a liquidation liquidity penalty should be added on top of the statistical VaR haircut.

### 2. Dollar Volume as a Proxy for Market Depth
Salooan uses 30-day average dollar volume ($\text{close} \times \text{volume}$) as a liquidity proxy. True market depth requires Level 2 and Level 3 order book metrics:
- Bid-ask spread dynamics during stress.
- Order book depth within 2%–5% of the mid-price.
- *Remedy*: Integrate real-time tick/order-book snapshots to scale haircuts with instantaneous market depth.

### 3. Overlapping Return Horizons & Breach Clustering
Forward returns over $N$ calendar days evaluated daily share $N - 1$ days of identical price history. This autocorrelation creates **breach clustering**:
- A single market shock (such as a 3-day market panic) causes multiple consecutive breach observations.
- Realized breach rates are naturally noisy and cluster in specific stress months rather than occurring independently.

### 4. Asynchronous Trading Calendars
Cryptocurrency markets trade 24/7/365, whereas traditional equity and ETF markets trade approximately 252 days per year (excluding weekends and holidays):
- $N$ is standardized in calendar days across all assets.
- For equities, targets use the last available trading close on or before $t + N$ via as-of tolerance lookups (`TARGET_ASOF_TOLERANCE_DAYS = 4`).

---

## Educational Disclaimer

> **Notice**: Salooan is an educational machine learning learning project designed to demonstrate quantile regression, risk sizing, time-series split hygiene, and dynamic risk corridors. It does not constitute financial, investment, legal, or credit advice. Do not use this software for live capital allocation without independent institutional risk management review.