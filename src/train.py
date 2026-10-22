import os
import tempfile
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from src.config import (
    DEFAULT_CONFIDENCE,
    DEFAULT_HORIZON_DAYS,
    FEATURE_COLUMNS,
    FEATURES_PATH,
    MODEL_PARAMS,
    MODELS_DIR,
    RANDOM_SEED,
    TEST_FRACTION,
)
from src.split import prepare_horizon_table, split_by_date


def model_path(horizon_days: int, confidence: float) -> str:
    pct = round(confidence * 100)
    filename = f"quantile_h{horizon_days}_c{pct}.joblib"
    return os.path.join(MODELS_DIR, filename)


def train_model(train_table: pd.DataFrame, confidence: float) -> xgb.XGBRegressor:
    missing_features = [col for col in FEATURE_COLUMNS if col not in train_table.columns]
    if missing_features:
        raise ValueError(f"Missing required feature columns: {missing_features}")

    if "target" not in train_table.columns:
        raise ValueError("Missing required 'target' column in training table.")

    if train_table[FEATURE_COLUMNS].isna().any().any():
        raise ValueError("Feature columns contain NaN values; cannot train model.")

    alpha = 1.0 - confidence
    model = xgb.XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=alpha,
        random_state=RANDOM_SEED,
        **MODEL_PARAMS,
    )
    # Fit model on training features and target
    model.fit(train_table[FEATURE_COLUMNS], train_table["target"])
    return model


def build_bundle(
    model: xgb.XGBRegressor,
    horizon_days: int,
    confidence: float,
    train_table: pd.DataFrame,
) -> Dict[str, Any]:
    train_end_date = None
    if "date" in train_table.columns and len(train_table) > 0:
        train_end_date = str(pd.to_datetime(train_table["date"]).max().date())

    return {
        "model": model,
        "horizon_days": int(horizon_days),
        "confidence": float(confidence),
        "feature_columns": list(FEATURE_COLUMNS),
        "train_end_date": train_end_date,
        "n_train_rows": int(len(train_table)),
    }


def save_bundle(bundle: Dict[str, Any]) -> str:
    os.makedirs(MODELS_DIR, exist_ok=True)
    target_path = model_path(bundle["horizon_days"], bundle["confidence"])

    # Atomic write using a temporary file in the target directory
    fd, temp_path = tempfile.mkstemp(dir=MODELS_DIR, prefix="tmp_bundle_", suffix=".joblib")
    os.close(fd)
    try:
        joblib.dump(bundle, temp_path)
        os.replace(temp_path, target_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return target_path


def load_bundle(horizon_days: int, confidence: float) -> Dict[str, Any]:
    path = model_path(horizon_days, confidence)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model bundle not found at {path}")
    return joblib.load(path)


def main() -> None:
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"Feature table not found at {FEATURES_PATH}. Run 'python -m src.features' first.")

    features = pd.read_parquet(FEATURES_PATH)
    horizon_table = prepare_horizon_table(features, horizon_days=DEFAULT_HORIZON_DAYS)
    split = split_by_date(horizon_table, horizon_days=DEFAULT_HORIZON_DAYS, test_fraction=TEST_FRACTION)

    model = train_model(split.train, confidence=DEFAULT_CONFIDENCE)
    bundle = build_bundle(model, DEFAULT_HORIZON_DAYS, DEFAULT_CONFIDENCE, split.train)
    saved_path = save_bundle(bundle)

    train_dates = pd.to_datetime(split.train["date"])
    train_start_str = train_dates.min().strftime("%Y-%m-%d")
    train_end_str = train_dates.max().strftime("%Y-%m-%d")
    test_start_str = split.test_start.strftime("%Y-%m-%d")

    print("=" * 70)
    print("QUANTILE BOOSTING MODEL TRAINING SUMMARY")
    print("=" * 70)
    print(f"Horizon:            {DEFAULT_HORIZON_DAYS} days")
    print(f"Confidence:         {int(DEFAULT_CONFIDENCE * 100)}% (alpha = {1.0 - DEFAULT_CONFIDENCE:.2f})")
    print(f"Train rows used:    {len(split.train):,}")
    print(f"Train date range:   {train_start_str} to {train_end_str}")
    print(f"Test start date:    {test_start_str}")
    print(f"Saved model path:   {saved_path}")

    print("\nFeature Importances:")
    for feature_name, importance in zip(FEATURE_COLUMNS, model.feature_importances_):
        print(f"  {feature_name:<28}: {importance:.4f}")

    # Diagnostic breach rate on the test split
    test_preds = model.predict(split.test[FEATURE_COLUMNS])
    test_targets = split.test["target"].to_numpy()

    print("\nTest-Set Breach Rate Diagnostic (alpha = 0.05 target):")
    print(f"{'Asset':<10} {'Asset Class':<12} {'Breach Rate':<15} {'Test Rows':<10}")
    print("-" * 50)

    for asset in sorted(split.test["asset"].unique()):
        mask = (split.test["asset"] == asset).to_numpy()
        asset_class = split.test.loc[mask, "asset_class"].iloc[0]
        asset_breach = np.mean(test_targets[mask] < test_preds[mask])
        n_rows = int(mask.sum())
        print(f"{asset:<10} {asset_class:<12} {asset_breach:>13.2%} {n_rows:>10d}")

    overall_breach = np.mean(test_targets < test_preds)
    print("-" * 50)
    print(f"{'OVERALL':<10} {'all':<12} {overall_breach:>13.2%} {len(split.test):>10d}")
    print("=" * 70)


if __name__ == "__main__":
    main()
