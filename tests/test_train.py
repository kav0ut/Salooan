import os
import numpy as np
import pandas as pd
import pytest

from src.config import (
    DEFAULT_CONFIDENCE,
    DEFAULT_HORIZON_DAYS,
    FEATURE_COLUMNS,
    VOLATILITY_COL,
    VOLUME_COL,
)
from src.train import (
    build_bundle,
    load_bundle,
    model_path,
    save_bundle,
    train_model,
)


def make_synthetic_training_data(n_rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    vol = rng.uniform(0.01, 0.05, size=n_rows)
    vol_dol = rng.uniform(1e6, 1e8, size=n_rows)
    target = rng.normal(0, vol, size=n_rows)
    start = pd.Timestamp("2023-01-01")
    dates = [start + pd.Timedelta(days=int(i % 400)) for i in range(n_rows)]
    assets = [f"ASSET_{i % 3}" for i in range(n_rows)]

    return pd.DataFrame({
        "asset": assets,
        "date": dates,
        VOLATILITY_COL: vol,
        VOLUME_COL: vol_dol,
        "target": target,
    })


def test_model_path_naming() -> None:
    path_7_95 = model_path(7, 0.95)
    assert os.path.basename(path_7_95) == "quantile_h7_c95.joblib"

    path_14_99 = model_path(14, 0.99)
    assert os.path.basename(path_14_99) == "quantile_h14_c99.joblib"

    path_7_90 = model_path(7, 0.90)
    assert os.path.basename(path_7_90) == "quantile_h7_c90.joblib"


def test_higher_volatility_gives_lower_quantile_prediction() -> None:
    train_df = make_synthetic_training_data(n_rows=5000, seed=42)
    model = train_model(train_df, confidence=DEFAULT_CONFIDENCE)

    # Low vs high volatility comparison with dollar volume kept identical
    eval_df = pd.DataFrame({
        VOLATILITY_COL: [0.01, 0.05],
        VOLUME_COL: [5e7, 5e7],
    })
    preds = model.predict(eval_df[FEATURE_COLUMNS])
    # Higher volatility should produce a more negative (lower) quantile return
    assert preds[1] < preds[0]


def test_fresh_synthetic_breach_rate_close_to_alpha() -> None:
    train_df = make_synthetic_training_data(n_rows=5000, seed=42)
    model = train_model(train_df, confidence=0.95)

    test_df = make_synthetic_training_data(n_rows=4000, seed=999)
    preds = model.predict(test_df[FEATURE_COLUMNS])
    breach_rate = np.mean(test_df["target"].to_numpy() < preds)

    # For alpha = 0.05, breach rate should be between 0.03 and 0.08
    assert 0.03 <= breach_rate <= 0.08


def test_higher_confidence_predicts_lower_values_on_average() -> None:
    train_df = make_synthetic_training_data(n_rows=5000, seed=42)
    model_95 = train_model(train_df, confidence=0.95)
    model_99 = train_model(train_df, confidence=0.99)

    test_df = make_synthetic_training_data(n_rows=2000, seed=123)
    preds_95 = model_95.predict(test_df[FEATURE_COLUMNS])
    preds_99 = model_99.predict(test_df[FEATURE_COLUMNS])

    # 99% confidence (alpha=0.01) represents a more severe loss quantile than 95% (alpha=0.05)
    assert preds_99.mean() < preds_95.mean()


def test_training_twice_with_same_seed_is_deterministic() -> None:
    train_df = make_synthetic_training_data(n_rows=3000, seed=42)
    test_df = make_synthetic_training_data(n_rows=1000, seed=888)

    model_a = train_model(train_df, confidence=DEFAULT_CONFIDENCE)
    model_b = train_model(train_df, confidence=DEFAULT_CONFIDENCE)

    preds_a = model_a.predict(test_df[FEATURE_COLUMNS])
    preds_b = model_b.predict(test_df[FEATURE_COLUMNS])
    np.testing.assert_array_equal(preds_a, preds_b)


def test_extra_columns_ignored_and_missing_or_nan_features_raise() -> None:
    train_df = make_synthetic_training_data(n_rows=1000, seed=42)

    # Extra columns should be gracefully ignored
    extra_df = train_df.copy()
    extra_df["irrelevant_metric"] = 999.0
    extra_df["random_category"] = "crypto"
    model = train_model(extra_df, confidence=DEFAULT_CONFIDENCE)
    assert model is not None

    # Missing a required feature column must raise ValueError
    missing_feature_df = train_df.drop(columns=[VOLATILITY_COL])
    with pytest.raises(ValueError, match="Missing required feature columns"):
        train_model(missing_feature_df, confidence=DEFAULT_CONFIDENCE)

    # Missing target column must raise ValueError
    missing_target_df = train_df.drop(columns=["target"])
    with pytest.raises(ValueError, match="Missing required 'target' column"):
        train_model(missing_target_df, confidence=DEFAULT_CONFIDENCE)

    # NaN in feature columns must raise ValueError
    nan_feature_df = train_df.copy()
    nan_feature_df.loc[5, VOLUME_COL] = np.nan
    with pytest.raises(ValueError, match="Feature columns contain NaN"):
        train_model(nan_feature_df, confidence=DEFAULT_CONFIDENCE)


def test_save_and_load_bundle_round_trip(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.train.MODELS_DIR", str(tmp_path))

    train_df = make_synthetic_training_data(n_rows=2000, seed=42)
    model = train_model(train_df, confidence=0.95)
    bundle = build_bundle(model, DEFAULT_HORIZON_DAYS, 0.95, train_df)

    saved_path = save_bundle(bundle)
    assert os.path.exists(saved_path)

    loaded_bundle = load_bundle(DEFAULT_HORIZON_DAYS, 0.95)
    assert loaded_bundle["horizon_days"] == DEFAULT_HORIZON_DAYS
    assert loaded_bundle["confidence"] == 0.95
    assert loaded_bundle["feature_columns"] == list(FEATURE_COLUMNS)
    assert loaded_bundle["n_train_rows"] == len(train_df)
    assert loaded_bundle["train_end_date"] == str(pd.to_datetime(train_df["date"]).max().date())

    test_df = make_synthetic_training_data(n_rows=500, seed=777)
    original_preds = model.predict(test_df[FEATURE_COLUMNS])
    loaded_preds = loaded_bundle["model"].predict(test_df[FEATURE_COLUMNS])
    np.testing.assert_array_equal(original_preds, loaded_preds)


def test_input_table_not_mutated() -> None:
    train_df = make_synthetic_training_data(n_rows=1000, seed=42)
    baseline_copy = train_df.copy(deep=True)

    model = train_model(train_df, confidence=DEFAULT_CONFIDENCE)
    pd.testing.assert_frame_equal(train_df, baseline_copy)

    _ = build_bundle(model, DEFAULT_HORIZON_DAYS, DEFAULT_CONFIDENCE, train_df)
    pd.testing.assert_frame_equal(train_df, baseline_copy)
