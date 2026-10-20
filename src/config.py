"""
Configuration constants for the Salooan project.
"""
import os

# Project root directory (two levels up from this file)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Directory paths
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Asset definitions
CRYPTO_ASSETS = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOGEUSDT"]
EQUITY_ASSETS = ["SPY", "ARKK"]

# Date and horizon settings
START_DATE = "2023-01-01"
HORIZONS_DAYS = [7, 14]
CONFIDENCE_LEVELS = [0.90, 0.95, 0.99]

# Window sizes for technical indicators
VOLATILITY_WINDOW = 30
VOLUME_WINDOW = 30

# Random seed for reproducibility
RANDOM_SEED = 42