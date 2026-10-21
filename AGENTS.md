# AGENTS.md

This file provides guidance to AI coding agents (Antigravity, Claude Code, and similar) when working with code in this repository.

## Project Overview

**Salooan** (dynamic collateral haircut estimation): a risk-sizing tool for secured lenders. When a borrower pledges an asset (crypto, stock or ETF) as collateral for a cash loan, the lender lends less than the asset's market value. That gap is the **haircut**, a safety cushion against the price falling before the collateral can be sold.

Instead of a fixed haircut per asset, this project predicts a **dynamic haircut**: the worst-case price drop over a liquidation horizon (e.g. 7 days) at a chosen confidence level (e.g. 95%), estimated from the asset's recent volatility and liquidity. This is a Value-at-Risk (VaR) style estimate.

- `haircut = -(predicted low-quantile forward return)`. Example: predicted 5th-percentile 7-day return of -22% gives a 22% haircut, so max loan = collateral value x (1 - 0.22).
- No real loan or borrower data exists. Only price/volume history is used. The loan scenario is a framing for the risk model; borrower credit risk is out of scope.
- This is a solo ML learning project. The goal is to understand the concepts, so hand-implement the core logic (metrics, feature engineering, split, haircut conversion) and use libraries only for infrastructure and the boosting model itself.

## Development Commands

- **Create env**: `python -m venv .venv` then `source .venv/bin/activate` (Windows: `.venv\Scripts\activate`)
- **Install**: `pip install -r requirements.txt`
- **Pull / refresh data**: `python -m src.data_pull`
- **Run tests**: `pytest`
- **Run one test file**: `pytest tests/test_features.py -v`
- **Run the app**: `streamlit run app.py`

Use the project's virtual environment for every command. The agent's terminal may not have it activated, so on Windows call it explicitly, e.g. `.venv\Scripts\python -m pytest` or `.venv\Scripts\python -m src.preprocess`.

All commands are run from the project root. Scripts must work from any working directory by building paths from `src/config.py`, never from relative strings.

## Development Workflow

These rules apply to every coding session in this repo, not just this milestone:

- **Plan before major work.** Before anything non-trivial (a new module, a refactor, a cross-cutting change), write a short implementation plan (what will change, which files are affected, any risks) and share it before writing code.
- **Do only what the current prompt asks.** Don't build ahead into later milestones.
- **Track milestones in `MILESTONES/`.** Each milestone has its own file (e.g. `MILESTONE_3.md`) marked with a status (`Completed`, `In Progress`, `Planned`), listing finished and remaining work. Keep these up to date whenever scope or status changes. `MILESTONES/` is listed in `.gitignore`, so these files are local-only.
- **The user handles all git operations.** Never run `git init`, `git add`, `git commit`, `git push`, or anything that modifies git history. After each meaningful, working change, instead propose a commit message in the format below and stop.
- **One commit per milestone.** When a milestone is complete and `pytest` passes, propose a single commit message from the milestone table below, using conventional prefixes (`feat:`, `fix:`, `test:`, `docs:`, `chore:`, `refactor:`) in the imperative mood. Bugs found later get their own `fix:` commit.
- **Test before proposing a commit.** Run `pytest` for the affected modules and report the result. Don't propose a commit on a known-broken build.
- **Add tests for new logic.** Any new function with real logic gets a `pytest` test in `tests/`. Use small synthetic DataFrames, never live API calls.
- **Guard against data leakage.** Features for a given date may only use data up to and including that date. Any feature or target change must be checked against this, and the no-lookahead test must keep passing.
- **Surface inconsistencies immediately.** If something doesn't line up with this file, the milestone files, or existing code (conflicting requirements, stale docs, design drift), stop and flag it with a suggested fix instead of silently working around it.
- **Write for a learner.** Clear, readable code. No docstrings on functions or methods. Use short inline # comments only where the reason behind a step isn't obvious. Small single-purpose functions with type hints and descriptive names. No clever one-liners that hide the logic.
- **Settings live in `src/config.py`.** Asset lists, paths, horizons, confidence levels, window sizes and the start date are never hardcoded elsewhere.

## Code Architecture

### Data flow
```
APIs -> data_pull.py -> data/raw -> preprocess.py -> data/processed
     -> features.py -> feature table -> split.py -> train.py -> models/
     -> evaluate.py -> results/ -> app.py (Streamlit)
```

The Streamlit app reads only saved models, processed data and results. It never calls the APIs directly, except through the refresh button (data refresh only; retraining is a separate action).

### Modules (`src/`)
- **`config.py`** - paths (built from the project root), `CRYPTO_ASSETS`, `EQUITY_ASSETS`, `START_DATE`, `HORIZONS_DAYS`, `CONFIDENCE_LEVELS`, `VOLATILITY_WINDOW`, `VOLUME_WINDOW`, `RANDOM_SEED`
- **`data_pull.py`** - ingestion. Binance public klines (daily, paginated 1000 per request) for crypto; `yfinance` for equities/ETFs. Drops today's unfinished candle, supports incremental refresh (fetch only after the last saved candle), writes atomically (temp file then rename). Output: `data/raw/<asset>.parquet`
- **`preprocess.py`** - cleans raw data, enforces dtypes, computes log returns, builds the forward-return target for horizon N. Output: `data/processed/`
- **`features.py`** - rolling volatility (std of log returns) and rolling average dollar volume per asset; merges all assets into one feature table. Stretch: max drawdown, correlation with BTC
- **`split.py`** - chronological train/test split with an N-day gap. Never shuffle
- **`baselines.py`** - static haircut and per-asset historical-quantile baselines
- **`metrics.py`** - hand-implemented pinball loss and coverage (breach rate) in numpy
- **`train.py`** - gradient boosting (XGBoost or LightGBM) with quantile objective; saves models with joblib to `models/`
- **`predict.py`** - latest-day features -> predicted quantile return -> haircut -> max loan
- **`evaluate.py`** - scores models and baselines on the test set, per asset and overall, writes `results/results.json`
- **`app.py`** (project root) - Streamlit dashboard

### Data schemas
- **Raw** (one row per asset per day): `date, open, high, low, close, volume, n_trades, taker_buy_volume, asset, asset_class`. Equities have NaN for `n_trades` and `taker_buy_volume`
- **Processed** adds: `log_return` and one `target_{N}d` column per horizon in `HORIZONS_DAYS` (simple forward return). Rows with no target (the latest N calendar days) are kept with NaN, because the app needs them for prediction
- **Feature table**: `asset, asset_class, date, volatility_30d, avg_dollar_volume_30d, target_{N}d`. Rows with incomplete rolling windows are dropped, never filled. Rows with no target are dropped only when building training sets; the latest row per asset is kept for prediction

### Key design decisions
- **Granularity**: daily candles, about 3 years of history, completed candles only. Not real-time
- **Horizon**: N is in calendar days for every asset. For crypto that is N rows; for equities the target uses the last available close on or before t+N (as-of lookup, tolerance `TARGET_ASOF_TOLERANCE_DAYS`)
- **Target**: simple forward return `close(t+N) / close(t) - 1`, so `haircut = -(predicted return)` exactly. Log returns are used only for the volatility feature
- **Storage**: Parquet for raw/processed data (no database), joblib for models, JSON for results
- **Model**: quantile / pinball loss objective at alpha = 1 - confidence (e.g. 0.05 for 95%), one model per (confidence, horizon) pair
- **Assets**: crypto BTCUSDT, ETHUSDT, ADAUSDT, DOGEUSDT (Binance); equities SPY, ARKK (yfinance). Chosen to give a spread of volatility and liquidity
- **Evaluation**: pinball loss and coverage, always compared against both baselines. Coverage should be close to alpha. Note that overlapping N-day returns make breaches cluster, so don't over-read small coverage differences
- **Refresh rule**: the app's refresh button is allowed only when a new daily candle could exist (last saved candle is older than yesterday, UTC). It refreshes data and features only; retraining is a separate, rarer action
- **Streamlit**: never retrain on a widget interaction. Load saved models with `@st.cache_resource` and data with `@st.cache_data`
- **Known limitations** (document in README): daily close is not a forced-sale price, no real bid-ask spread data, overlapping returns, crypto and equity calendars differ
- **Features**: two features only to start: rolling volatility and rolling average dollar volume. Anything else (drawdown, correlation, volume trend) is a stretch item in M10
- **volatility_30d**: sample standard deviation (ddof=1) of `log_return` over the last `VOLATILITY_WINDOW` rows, including today. It is the volatility of daily returns, not annualized
- **avg_dollar_volume_30d**: rolling mean of `close * volume` over the last `VOLUME_WINDOW` rows, including today. Dollar volume, not raw volume, is the liquidity proxy (crypto volume is in coins, equity volume is in shares)
- **Feature windows**: count rows, not calendar days (30 rows is 30 days for crypto, 30 trading days for equities). `min_periods = window`, so incomplete windows give NaN and are never filled
- **No lookahead**: the feature at date t uses only data up to and including t. Features never read `target_*` columns
- **Rows kept in feature table**: drops only rows with incomplete rolling windows. Rows with NaN targets are kept for app prediction; dropped only when building training sets
- **Feature table schema**: `asset, asset_class, date, volatility_<W>d, avg_dollar_volume_<W>d, target_7d, target_14d`, sorted by asset then date. Column names come from config (`VOLATILITY_COL = f"volatility_{VOLATILITY_WINDOW}d"`, `VOLUME_COL = f"avg_dollar_volume_{VOLUME_WINDOW}d"`)

## Technology Stack
- **Language**: Python 3.10+
- **Data**: pandas, numpy, pyarrow
- **Ingestion**: requests (Binance), yfinance (equities)
- **Modeling**: XGBoost or LightGBM (quantile objective), scikit-learn for utilities only, joblib. Metrics are hand-implemented, not imported
- **App / plots**: Streamlit, Plotly
- **Testing**: pytest

## Milestone Plan and Commits

One commit per milestone.

| # | Milestone | Commit message |
|---|---|---|
| 0 | Scaffold, `.gitignore`, `requirements.txt`, README skeleton | `chore: initialize project structure` |
| 1 | Data ingestion (Binance, yfinance, incremental refresh) | `feat: add data ingestion` |
| 2 | Preprocessing, log returns, forward-return target | `feat: add preprocessing and forward return target` |
| 3 | Rolling volatility and volume features, merged feature table, no-lookahead test | `feat: add rolling volatility and volume features` |
| 4 | Chronological split with gap, static and historical-quantile baselines | `feat: add time-based split and baselines` |
| 5 | Quantile boosting training and haircut prediction (minimal demo checkpoint) | `feat: add quantile training and haircut prediction` |
| 6 | Pinball loss, coverage, evaluation vs baselines | `feat: add evaluation against baselines` |
| 7 | Streamlit app (calculator and evaluation tab) | `feat: add Streamlit app` |
| 8 | Model grid (confidence x horizon) wired to the sidebar | `feat: add multi-confidence and horizon models` |
| 9 | Refresh button with cooldown | `feat: add data refresh with cooldown` |
| 10 | Stretch items | `feat: add <stretch item>` |
| 11 | README with workflow diagram, results, limitations | `docs: write README with results` |

The end of Milestone 5 is the minimum demoable checkpoint (the user will tag it `v0.1-minimal-demo`).

## Current Focus
Milestones 0, 1, 2, and 3 are complete. Milestone 3 is ready to commit (`feat: add rolling volatility and volume features`). Milestone 4 (chronological split with gap, baselines) is next; its spec will be in `MILESTONES/MILESTONE_4.md`.

`MILESTONES/` is the source of truth for milestone status and scope; update it there first and keep this section in step with it.

## Notes
- Standard layout: `src/` for modules, `tests/` for pytest, `data/raw` and `data/processed` for Parquet files, `models/` for joblib files, `results/` for JSON
- `data/`, `models/` and `results/` contents are gitignored; only their `.gitkeep` files are tracked
- Binance.com may be blocked in some regions; the fallback is `https://api.binance.us/api/v3/klines`, which has the same response format but fewer symbols
- yfinance is an unofficial library and can break or rate-limit, which is why all data is cached locally
