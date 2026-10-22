import os
import tempfile
from typing import Any, Dict, List, Sequence

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from src.config import (
    CONFIDENCE_LEVELS,
    DEFAULT_CONFIDENCE,
    DEFAULT_HORIZON_DAYS,
    FEATURE_COLUMNS,
    FEATURES_PATH,
    HORIZONS_DAYS,
    MODEL_PARAMS,
    MODELS_DIR,
    PROJECT_ROOT,
    RANDOM_SEED,
    TEST_FRACTION,
)
from src.split import prepare_horizon_table, split_by_date


def config_key(horizon_days: int, confidence: float) -> str:
    pct = round(confidence * 100)
    return f"h{horizon_days}_c{pct}"


def model_path(
    horizon_days: int,
    confidence: float,
    models_dir: str | None = None,
) -> str:
    target_dir = models_dir if models_dir is not None else MODELS_DIR
    filename = f"quantile_{config_key(horizon_days, confidence)}.joblib"
    return os.path.join(target_dir, filename)


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


def save_bundle(bundle: Dict[str, Any], models_dir: str | None = None) -> str:
    target_dir = models_dir if models_dir is not None else MODELS_DIR
    os.makedirs(target_dir, exist_ok=True)
    target_path = model_path(bundle["horizon_days"], bundle["confidence"], models_dir=target_dir)

    # Atomic write using a temporary file in the target directory
    fd, temp_path = tempfile.mkstemp(dir=target_dir, prefix="tmp_bundle_", suffix=".joblib")
    os.close(fd)
    try:
        joblib.dump(bundle, temp_path)
        os.replace(temp_path, target_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return target_path


def load_bundle(
    horizon_days: int,
    confidence: float,
    models_dir: str | None = None,
) -> Dict[str, Any]:
    path = model_path(horizon_days, confidence, models_dir=models_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model bundle not found at {path}")
    return joblib.load(path)


def train_grid(
    features: pd.DataFrame,
    horizons: Sequence[int],
    confidences: Sequence[float],
    models_dir: str | None = None,
) -> List[Dict[str, Any]]:
    bundles: List[Dict[str, Any]] = []
    for horizon in horizons:
        horizon_table = prepare_horizon_table(features, horizon_days=horizon)
        split = split_by_date(horizon_table, horizon_days=horizon, test_fraction=TEST_FRACTION)
        for confidence in confidences:
            model = train_model(split.train, confidence=confidence)
            bundle = build_bundle(model, horizon, confidence, split.train)
            save_bundle(bundle, models_dir=models_dir)
            bundles.append(bundle)
    return bundles


def load_grid_bundles(
    horizon_days: int,
    confidences: Sequence[float],
    models_dir: str | None = None,
) -> Dict[float, Dict[str, Any]]:
    bundles: Dict[float, Dict[str, Any]] = {}
    for confidence in confidences:
        bundles[float(confidence)] = load_bundle(horizon_days, confidence, models_dir=models_dir)
    return bundles


def main() -> None:
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"Feature table not found at {FEATURES_PATH}. Run 'python -m src.features' first.")

    features = pd.read_parquet(FEATURES_PATH)
    bundles = train_grid(features, HORIZONS_DAYS, CONFIDENCE_LEVELS)
    bundles_by_pair = {(b["horizon_days"], b["confidence"]): b for b in bundles}

    rows = []
    for horizon in HORIZONS_DAYS:
        horizon_table = prepare_horizon_table(features, horizon_days=horizon)
        split = split_by_date(horizon_table, horizon_days=horizon, test_fraction=TEST_FRACTION)
        test_start_str = split.test_start.strftime("%Y-%m-%d")
        test_targets = split.test["target"].to_numpy()

        for conf in CONFIDENCE_LEVELS:
            bundle = bundles_by_pair[(horizon, conf)]
            saved_path = model_path(horizon, conf)
            test_preds = bundle["model"].predict(split.test[FEATURE_COLUMNS])
            overall_breach = float(np.mean(test_targets < test_preds))
            pair_label = f"h{horizon}_c{round(conf * 100)}"
            rel_path = os.path.relpath(saved_path, start=PROJECT_ROOT).replace("\\", "/")
            rows.append({
                "pair": pair_label,
                "train_rows": bundle["n_train_rows"],
                "train_end": bundle["train_end_date"],
                "test_start": test_start_str,
                "path": rel_path,
                "breach": overall_breach,
            })

    print("=" * 90)
    print("QUANTILE BOOSTING MODEL GRID TRAINING SUMMARY")
    print("=" * 90)
    print(f"{'Pair':<10} {'Train Rows':<12} {'Train End':<12} {'Test Start':<12} {'Saved Path':<32} {'Test Breach':<12}")
    print("-" * 90)
    for r in rows:
        print(f"{r['pair']:<10} {r['train_rows']:<12,d} {r['train_end']:<12} {r['test_start']:<12} {r['path']:<32} {r['breach']:>10.2%}")
    print("=" * 90)


if __name__ == "__main__":
    main()
