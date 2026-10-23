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
- **Write for a learner.** Clear, readable code. No docstrings anywhere (function, class or module), and no decorative divider or banner comments (such as # -----). Use short inline # comments only where the reason behind a step isn't obvious. Small single-purpose functions with type hints and descriptive names. No clever one-liners that hide the logic.
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
- **`split.py`** - prepares horizon modeling table (drops rows with NaN target), chronological train/test split by unique dates with an N-day purge gap. Never shuffle
- **`baselines.py`** - static (pooled training targets alpha-quantile) and historical (per-asset training targets alpha-quantile) baselines fit on training set only, predicting quantile returns aligned with test set
- **`metrics.py`** - hand-implemented pinball loss, coverage (breach rate), quantile crossing rate, and monotone quantile rearrangement in numpy with strict input validation
- **`train.py`** - trains `XGBRegressor` with `reg:quantileerror` on the purged train split for (horizon, confidence) pairs using `FEATURE_COLUMNS`; `train_grid` trains the full grid (`HORIZONS_DAYS x CONFIDENCE_LEVELS`), `config_key` helper; saves bundles with joblib to `models/`
- **`predict.py`** - loads model bundle, validates feature columns, predicts quantile return for latest feature row per asset, converts to haircut (`clip(-q, 0, 1)`), max loan, and multi-quantile prediction with optional rearrangement
- **`evaluate.py`** - pure core `evaluate_config(bundle, features)` with stale-model guard, `evaluate_grid` scores all grid models and baselines on test split, measures crossing rate per horizon, computes comparison block, writes `results/results.json` atomically
- **`refresh.py`** - data refresh with cooldown, status tracking, lock file, incremental ingestion, preprocessing and feature table rebuild; never trains or evaluates
- **`dashboard.py`** - pure helper functions for Streamlit UI (`load_results`, `calculator_result`, `baseline_haircuts`, `risk_band`, `latest_crossing`, `training_cutoff`, `newer_data_days`, `evaluation_tables`, `monthly_breach_rates`), no Streamlit dependency
- **`app.py`** (project root) - Streamlit dashboard UI (thin layout, calls `src/dashboard.py`, sidebar selects horizon/confidence and has Data section with refresh button/status, two tabs: haircut calculator with multi-tier risk corridor and evaluation with monthly breach timeline)

### Data schemas
- **Raw** (one row per asset per day): `date, open, high, low, close, volume, n_trades, taker_buy_volume, asset, asset_class`. Equities have NaN for `n_trades` and `taker_buy_volume`
- **Processed** adds: `log_return` and one `target_{N}d` column per horizon in `HORIZONS_DAYS` (simple forward return). Rows with no target (the latest N calendar days) are kept with NaN, because the app needs them for prediction
- **Feature table**: `asset, asset_class, date, volatility_30d, avg_dollar_volume_30d, target_{N}d`. Rows with incomplete rolling windows are dropped, never filled. Rows with no target are dropped only when building training sets; the latest row per asset is kept for prediction
- **results.json**: `{"generated_at": <UTC ISO string>, "configs": {"h{N}_c{pct}": {"horizon_days", "confidence", "alpha", "test_start", "train_end_date", "n_train_rows", "n_test_rows", "methods": {"model"|"static"|"historical": {"overall": {...}, "per_asset": {asset: {...}}}}, "comparison": {...}}}, "crossing": {"h{N}": {"n_rows", "crossing_rate", "confidences"}}}` with plain Python types only

### Key design decisions
- **Granularity**: daily candles, about 3 years of history, completed candles only. Not real-time
- **Horizon**: N is in calendar days for every asset. For crypto that is N rows; for equities the target uses the last available close on or before t+N (as-of lookup, tolerance `TARGET_ASOF_TOLERANCE_DAYS`)
- **Target**: simple forward return `close(t+N) / close(t) - 1`, so `haircut = -(predicted return)` exactly. Log returns are used only for the volatility feature
- **Storage**: Parquet for raw/processed data (no database), joblib for models, JSON for results
- **Model**: quantile / pinball loss objective at alpha = 1 - confidence (e.g. 0.05 for 95%), one model per (confidence, horizon) pair
- **Assets**: crypto BTCUSDT, ETHUSDT, ADAUSDT, DOGEUSDT (Binance); equities SPY, ARKK (yfinance). Chosen to give a spread of volatility and liquidity
- **Evaluation**: pinball loss and coverage, always compared against both baselines. Coverage should be close to alpha. Note that overlapping N-day returns make breaches cluster, so don't over-read small coverage differences
- **What the refresh button does**: it refreshes DATA only. It pulls new candles (incremental), re-runs preprocessing and feature building, clears the app's data caches and reruns the page. It NEVER trains or evaluates anything. Retraining stays a separate command-line action.
- **When the refresh button is allowed** (the refresh rule plus a retry limit):
  1. A new daily candle could exist: the latest saved crypto candle date is older than yesterday in UTC. The reference is the oldest of the four crypto assets' latest dates (any lagging asset makes a refresh worthwhile). Equities are ignored for this rule because they skip weekends and holidays, but they are refreshed together with the crypto assets.
  2. At least `REFRESH_MIN_INTERVAL_MINUTES` (60) have passed since the last refresh ATTEMPT, whether it succeeded or failed. This stops repeated clicks from hammering the APIs after a failure.
  When not allowed, the button is disabled and the sidebar says why ("Data is up to date as of ...", or "Next refresh possible at ... UTC").
- **Pure, injectable core** in `src/refresh.py`: the decision (`refresh_state`) takes the current time, the latest saved candle date and the status as arguments, and all time handling uses timezone-aware UTC datetimes (naive datetimes raise `ValueError`).
- **Status file** `REFRESH_STATUS_PATH` (inside `data/processed/`, which is gitignored), JSON: `last_attempt_utc`, `last_success_utc`, `last_error`, `rows_added` (per asset). Written atomically (temp file, then `os.replace`). A missing or corrupt file is treated as "no status", never as a crash.
- **Lock file** `REFRESH_LOCK_PATH`: created with exclusive-create semantics so two browser tabs cannot refresh at once. A lock older than `REFRESH_LOCK_TIMEOUT_MINUTES` (15) is considered stale and replaced. The lock is always released, including when the refresh raises.
- **Failure behavior**: existing files are never corrupted (writes are atomic). Each asset is ingested independently; one failing asset does not stop the others. Preprocessing and features are rebuilt only when at least one asset gained rows. Errors are stored in the status file and shown in the app with `st.error`; the app keeps working on the old data.
- **After a refresh the models are older than the data**, which is expected. To keep everything consistent without retraining:
  - **Baselines are fit on the same rows the models were evaluated against**: `baseline_haircuts` uses the modeling-table rows with `date <= bundle["train_end_date"]` (identical to the old split's train set before any refresh, and fixed afterwards).
  - **The chart marker** changes from the recomputed `test_start` to the bundle's `train_end_date`, labelled "model trained on data up to this date; everything after it is out-of-sample" (new data after a refresh is genuinely out-of-sample for the model).
  - **The Evaluation tab keeps reading `results.json`** and says so: evaluation reflects the last `python -m src.evaluate` run, with its own `test_start`.
  - `python -m src.evaluate` will raise its stale-model error after a refresh, by design. The message already says to retrain. The sidebar shows the retrain command when the newest feature date is more than `RETRAIN_SUGGEST_AFTER_DAYS` (60) after the bundle's `train_end_date`.
- **Caching**: after a successful refresh, `st.cache_data` is cleared (the bundle loader uses `st.cache_resource` and does not need clearing), the success message is kept across the rerun via `st.session_state`, and the app calls `st.rerun()`.
- **Time source**: only `app.py` calls `datetime.now(timezone.utc)`; everything else receives `now` as an argument.
- **Streamlit**: never retrain on a widget interaction. Load saved models with `@st.cache_resource` and data with `@st.cache_data`
- **Known limitations** (document in README): daily close is not a forced-sale price, no real bid-ask spread data, overlapping returns, crypto and equity calendars differ
- **Features**: two features only to start: rolling volatility and rolling average dollar volume. Anything else (drawdown, correlation, volume trend) is a stretch item in M10
- **volatility_30d**: sample standard deviation (ddof=1) of `log_return` over the last `VOLATILITY_WINDOW` rows, including today. It is the volatility of daily returns, not annualized
- **avg_dollar_volume_30d**: rolling mean of `close * volume` over the last `VOLUME_WINDOW` rows, including today. Dollar volume, not raw volume, is the liquidity proxy (crypto volume is in coins, equity volume is in shares)
- **Feature windows**: count rows, not calendar days (30 rows is 30 days for crypto, 30 trading days for equities). `min_periods = window`, so incomplete windows give NaN and are never filled
- **No lookahead**: the feature at date t uses only data up to and including t. Features never read `target_*` columns
- **Rows kept in feature table**: drops only rows with incomplete rolling windows. Rows with NaN targets are kept for app prediction; dropped only when building training sets
- **Feature table schema**: `asset, asset_class, date, volatility_<W>d, avg_dollar_volume_<W>d, target_7d, target_14d`, sorted by asset then date. Column names come from config (`VOLATILITY_COL = f"volatility_{VOLATILITY_WINDOW}d"`, `VOLUME_COL = f"avg_dollar_volume_{VOLUME_WINDOW}d"`)
- **One global date cutoff for all assets**: the test set is the most recent `TEST_FRACTION` (0.2) of unique dates in the modeling table, not rows. Crypto and equities have different calendars, and assets crash on the same days, so splitting each asset separately would leak market-wide information from the test period into training
- **Purge gap**: a training row only counts as clean if `date + N days < test_start`. Rows in the last N days before `test_start` are dropped from training (purged) because their targets are computed from prices inside the test period. The gap depends on horizon N
- **Test rows**: rows with `date >= test_start`. Their features look backwards into the training period, which is valid and expected
- **Split methodology**: no shuffling, no validation set in v1. Hyperparameters stay fixed
- **Rows with NaN target**: dropped when building the modeling table for a horizon (`prepare_horizon_table`). The latest feature row per asset is kept for prediction in M5
- **Baselines fit on train only**: `static` is the alpha-quantile of pooled training targets (alpha = 1 - confidence); `historical` is the alpha-quantile of each asset's own training targets
- **Baseline output**: predicted quantile return (negative number means loss), aligned with test rows. Haircut conversion (`-q`, floored at 0) happens in M5
- **High-confidence coverage limitation**: with a 20% test period and overlapping N-day returns, high confidence levels (99%) have very few expected breaches, making coverage at 99% noisy
- **Model**: `xgboost.XGBRegressor` with `objective="reg:quantileerror"` and `quantile_alpha = 1 - confidence` (pinball loss). This objective needs xgboost >= 2.0
- **Model features**: only `FEATURE_COLUMNS`. No asset id and no `asset_class` in v1, so the model must infer risk from volatility and liquidity alone, which is the point of a dynamic haircut
- **Parameterized, trained once for now**: `train_model` and the helpers take `horizon_days` and `confidence`; in M5 only the default pair (`DEFAULT_HORIZON_DAYS = 7`, `DEFAULT_CONFIDENCE = 0.95`) is trained. The full grid comes in M8
- **Trained on the TRAIN split only** (the purged split from M4), so the M6 evaluation on the test set is clean. Retraining on all data for deployment is a later stretch item
- **Fixed hyperparameters** from `MODEL_PARAMS` in config. No tuning and no early stopping in v1. Seed from `RANDOM_SEED`, so training is deterministic
- **Saved bundle** (joblib, one file per `(horizon, confidence)`): a dict with `model`, `horizon_days`, `confidence`, `feature_columns`, `train_end_date`, `n_train_rows`. Path: `models/quantile_h{N}_c{pct}.joblib` where `pct = round(confidence * 100)`. `predict` refuses to run if the bundle's `feature_columns` differ from the current `FEATURE_COLUMNS` (stale model guard)
- **Haircut conversion**: `haircut = clip(-q, 0, 1)` where `q` is the predicted quantile return. `max_loan = collateral_value * (1 - haircut)`
- **Prediction uses the latest feature row per asset**, including rows whose target is NaN. For equities the latest row is the last trading day, so always show the as-of date next to the prediction
- **Quantile crossing** is not an issue with a single model. Multi-quantile training is a stretch item in M10
- **Metrics are hand-implemented in numpy** in `src/metrics.py`; sklearn is allowed only inside tests as an independent reference (`pinball_loss(y_true, y_pred, alpha)` and `breach_rate(y_true, y_pred)`)
- **Evaluation scoring**: scores model and both baselines (static, historical) on the TEST split only, for one `(horizon, confidence)` pair, overall and per asset: `n_rows`, `pinball_loss`, `breach_rate`, `mean_haircut` (mean of `to_haircut(q)`)
- **Comparison block**: pinball improvement of the model vs each baseline in percent (`(baseline - model) / baseline * 100`, positive means the model is better), and average absolute distance of per-asset breach rates from alpha
- **Pure core, IO at the edges**: `evaluate_config(bundle, features)` takes bundle and feature table and returns a dict with no disk access; `main()` loads bundle, calls it, prints table and writes `results/results.json` atomically
- **Stale-model guard**: `evaluate_config` rebuilds the split with `split_by_date` and raises `ValueError` if `bundle["train_end_date"]` differs from the maximum training date of that split
- **Config-keyed results**: `results.json` is keyed by config (e.g. `h7_c95`) so M8 can add the grid without changing schema; written atomically (temp file then `os.replace`)
- **Overlapping return caveat**: overlapping N-day returns make breaches cluster, so breach rates are noisy and small differences from alpha are not evidence of a problem; with a 20% test fraction, 99% confidence has very few expected breaches
- **Thin UI, testable logic**: `app.py` only lays out widgets and calls functions in `src/dashboard.py`. All data preparation lives in `dashboard.py` as pure functions (no Streamlit imports), so it can be unit-tested
- **App is read-only**: loads saved model, feature table, processed price files and `results.json`. Never trains, never calls an API, never writes files. Refresh button comes in M9
- **Fixed configuration in M7**: only the default pair (`DEFAULT_HORIZON_DAYS`, `DEFAULT_CONFIDENCE`). Sidebar shows it as read-only text; grid sliders come in M8
- **Page layout**: `st.set_page_config(page_title="Salooan", layout="wide")` as the first Streamlit call, title "Salooan", one-line plain-language explanation of what a haircut is
- **Sidebar**: asset selector (six assets from config) and collateral value input (`DEMO_COLLATERAL_VALUE` default, minimum 0, step 100)
- **Two tabs**: "Haircut calculator" and "Evaluation"
- **Calculator tab**: three headline metrics (haircut %, max loan, predicted worst-case 7-day drop %); as-of date (last trading day for equities); current volatility and dollar-volume features; model vs baselines comparison bar chart; risk chart
- **Risk chart**: closing price over last 365 days plus predicted floor line (`close(t) * (1 + q(t))` plotted at `t + N` calendar days). Last N days of floor extend beyond last price date (live prediction). Vertical line marks `test_start` (start of out-of-sample period) with explanatory caption
- **Evaluation tab**: reads `results.json` for default config. Summary table (pinball loss, breach rate, mean haircut per method), grouped bar chart of per-asset breach rates with horizontal line at alpha, comparison block metrics, run facts (`test_start`, `train_end_date`, train and test row counts, `generated_at`), and overlapping return noise caption
- **Friendly failures**: if model file, feature table, or `results.json` is missing, show `st.error` or `st.warning` with the command to create it (`python -m src.train`, `python -m src.features`, `python -m src.evaluate`) and call `st.stop()`. Never show raw traceback
- **Caching**: `@st.cache_resource` for model bundle and `@st.cache_data` for DataFrames and computed tables
- **Footer**: educational project disclaimer, not financial advice
- **Plots**: Plotly
- **The grid is every pair** in `HORIZONS_DAYS x CONFIDENCE_LEVELS` from config, one independently trained model per pair, saved as `models/quantile_h{N}_c{pct}.joblib` (naming unchanged).
- **One source of truth for the config key.** A single helper `config_key(horizon_days, confidence)` returns strings like `h7_c95` and is used by training, evaluation, the app and `results.json`. It lives next to `model_path` (or in config), never re-implemented elsewhere.
- **CLI behavior changes**: `python -m src.train` now trains the full grid (it used to train only the default pair), and `python -m src.evaluate` now evaluates the full grid and writes all six configs to `results/results.json`. Each model still trains on the purged train split of its own horizon, and each evaluation still uses the stale-model guard.
- **Quantile crossing is measured, not corrected.** Models at different confidence levels are trained independently, so occasionally a higher-confidence model may predict a smaller loss than a lower-confidence one (the predicted return should be non-increasing as confidence rises). Correcting this is a stretch item (M10). In M8 we only measure and report it:
  - `crossing_rate(predictions_by_confidence)` in `metrics.py` (hand-implemented numpy): the share of rows where, with confidences sorted ascending, any `q[c_{i+1}] > q[c_i]` (ties are not violations). It needs at least two confidence levels and arrays of equal length.
  - `results.json` gets a top-level `crossing` block: `{"h7": {"n_rows": ..., "crossing_rate": ..., "confidences": [...]}, "h14": {...}}`, computed on the TEST rows of each horizon.
  - The calculator tab shows an `st.warning` when, for the selected asset's latest row, the predictions across the three confidence levels of the selected horizon violate the ordering.
- **Sidebar**: horizon as `st.sidebar.radio` (options from config, shown as "7 days" / "14 days"), confidence as `st.sidebar.select_slider` (options from config, shown as percentages). Remove the read-only "fixed configuration" caption from M7. Default selection is `DEFAULT_HORIZON_DAYS` and `DEFAULT_CONFIDENCE`.
- **Everything in the app follows the selection**: the loaded bundle, the headline metric labels (for example "Predicted worst-case 14-day drop (99% confidence)"), the baseline haircuts, the risk chart (floor shifted by the selected horizon), the test-period marker, and the Evaluation tab (which uses the selected config key).
- **Friendly failures** extend to the grid: a missing model file for the selected pair shows an error with the command `python -m src.train`; a missing config key in `results.json` shows a warning with the command `python -m src.evaluate`. Both then call `st.stop()`.
- **Caching**: the bundle loader (`st.cache_resource`) is keyed by horizon and confidence; DataFrame loaders stay `st.cache_data`.
- **Limitation to document in the README**: at 99% confidence the test window has very few expected breaches (about 1% of rows, and overlapping returns cluster them), so coverage at 99% is noisy; the 14-day horizon has fewer usable rows than the 7-day one.
- **Monotonic quantile rearrangement** (Chernozhukov, Fernández-Val, and Galichon 2010): models for different confidence levels are trained independently, which occasionally leads to crossing ($q_{0.95} > q_{0.90}$). Sorting predicted quantile returns descending across ascending confidence levels mathematically guarantees that $q_{0.90} \ge q_{0.95} \ge q_{0.99}$ (larger loss for higher confidence) on every single sample. Hand-implemented in `src/metrics.py:monotone_rearrangement`.
- **Multi-quantile risk corridor on the price chart**: instead of plotting a single dashed line for only the selected confidence, `risk_band` computes floor lines across confidences, and the risk chart renders a shaded corridor between 90% and 99% floors with the 95% floor as an interior line.
- **Temporal breach distribution (Breach-by-Month)**: in `src/dashboard.py`, `monthly_breach_rates(bundle, features)` evaluates out-of-sample test rows grouped by calendar month (`YYYY-MM`). The Evaluation tab renders a monthly bar chart with a horizontal line at target $\alpha$, visually demonstrating how breaches cluster during market stress due to overlapping multi-day returns.

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
| 10 | Monotonic risk band and monthly breach analysis | `feat: add monotonic risk band and monthly breach analysis` |
| 11 | README with workflow diagram, results, limitations | `docs: write README with results` |

The end of Milestone 5 is the minimum demoable checkpoint (the user will tag it `v0.1-minimal-demo`).

## Current Focus
Milestones 0-11 complete (Milestone 11 ready to commit: `docs: write README with results`). Spec in `MILESTONES/MILESTONE_11.md`.

`MILESTONES/` is the source of truth for milestone status and scope; update it there first and keep this section in step with it.

## Notes
- Standard layout: `src/` for modules, `tests/` for pytest, `data/raw` and `data/processed` for Parquet files, `models/` for joblib files, `results/` for JSON
- `data/`, `models/` and `results/` contents are gitignored; only their `.gitkeep` files are tracked
- Binance.com may be blocked in some regions; the fallback is `https://api.binance.us/api/v3/klines`, which has the same response format but fewer symbols
- yfinance is an unofficial library and can break or rate-limit, which is why all data is cached locally
