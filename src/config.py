import os

# Project root directory (two levels up from this file)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Directory paths
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FEATURES_PATH = os.path.join(PROCESSED_DIR, "features.parquet")

# Asset definitions
CRYPTO_ASSETS = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOGEUSDT"]
EQUITY_ASSETS = ["SPY", "ARKK"]

# Date and horizon settings
START_DATE = "2023-01-01"
HORIZONS_DAYS = [7, 14]
CONFIDENCE_LEVELS = [0.90, 0.95, 0.99]
TEST_FRACTION = 0.2
# Tolerance for as-of lookup when matching forward returns (in days)
TARGET_ASOF_TOLERANCE_DAYS = 5

# Window sizes for technical indicators
VOLATILITY_WINDOW = 30
VOLUME_WINDOW = 30
VOLATILITY_COL = f"volatility_{VOLATILITY_WINDOW}d"
VOLUME_COL = f"avg_dollar_volume_{VOLUME_WINDOW}d"
FEATURE_COLUMNS = [VOLATILITY_COL, VOLUME_COL]

# Random seed for reproducibility
RANDOM_SEED = 42

# Binance API
BINANCE_URL = "https://api.binance.com/api/v3/klines"  # Use https://api.binance.us for Binance.US
# Raw schema columns as per Binance kline response
COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
    "ignore"
]