import os
import numpy as np
import pandas as pd
import pytest

from src.config import (
    DEFAULT_CONFIDENCE,
    DEFAULT_HORIZON_DAYS,
    FEATURE_COLUMNS,
    TEST_FRACTION,
    VOLATILITY_COL,
    VOLUME_COL,
)
from src.split import prepare_horizon_table, split_by_date
from src.train import (
    build_bundle,
    config_key,
    load_bundle,
    load_grid_bundles,
    model_path,
    save_bundle,
    train_grid,
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


def make_synthetic_grid_features(n_days: int = 150, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")

    rows = []
    for asset, asset_class in [("BTC", "crypto"), ("SPY", "equity")]:
        vol = rng.uniform(0.01, 0.04, size=n_days)
        vol_dol = rng.uniform(1e7, 1e8, size=n_days)
        t7 = rng.normal(0, vol, size=n_days)
        t14 = rng.normal(0, vol * 1.4, size=n_days)
        df_asset = pd.DataFrame({
            "asset": [asset] * n_days,
            "asset_class": [asset_class] * n_days,
            "date": dates,
            VOLATILITY_COL: vol,
            VOLUME_COL: vol_dol,
            "target_7d": t7,
            "target_14d": t14,
        })
        df_asset.loc[n_days - 7 :, "target_7d"] = np.nan
        df_asset.loc[n_days - 14 :, "target_14d"] = np.nan
        rows.append(df_asset)

    return pd.concat(rows, ignore_index=True).sort_values(by=["date", "asset"]).reset_index(drop=True)


def test_config_key_and_model_path_rounding() -> None:
    assert config_key(7, 0.95) == "h7_c95"
    assert config_key(14, 0.99) == "h14_c99"
    assert config_key(7, 0.90) == "h7_c90"
    path_7_95 = model_path(7, 0.95)
    assert os.path.basename(path_7_95) == "quantile_h7_c95.joblib"
    path_14_99 = model_path(14, 0.99)
    assert os.path.basename(path_14_99) == "quantile_h14_c99.joblib"


def test_train_grid_saves_files_and_returns_bundles_with_correct_metadata(tmp_path) -> None:
    features = make_synthetic_grid_features(n_days=150, seed=42)
    horizons = [7, 14]
    confidences = [0.90, 0.95, 0.99]
    models_dir = str(tmp_path)

    bundles = train_grid(features, horizons, confidences, models_dir=models_dir)
    assert len(bundles) == len(horizons) * len(confidences)

    expected_files = {
        f"quantile_h{h}_c{round(c * 100)}.joblib"
        for h in horizons
        for c in confidences
    }
    saved_files = set(os.listdir(models_dir))
    assert expected_files.issubset(saved_files)

    for h in horizons:
        h_table = prepare_horizon_table(features, horizon_days=h)
        split = split_by_date(h_table, horizon_days=h, test_fraction=TEST_FRACTION)
        expected_train_end = str(pd.to_datetime(split.train["date"]).max().date())
        expected_train_rows = len(split.train)

        for c in confidences:
            matching = [b for b in bundles if b["horizon_days"] == h and b["confidence"] == c]
            assert len(matching) == 1
            bundle = matching[0]
            assert bundle["horizon_days"] == h
            assert bundle["confidence"] == c
            assert bundle["feature_columns"] == list(FEATURE_COLUMNS)
            assert bundle["n_train_rows"] == expected_train_rows
            assert bundle["train_end_date"] == expected_train_end
            assert bundle["model"] is not None


def test_train_grid_higher_confidence_predicts_lower_values_on_average(tmp_path) -> None:
    features = make_synthetic_grid_features(n_days=150, seed=42)
    horizons = [7, 14]
    confidences = [0.90, 0.99]
    models_dir = str(tmp_path)

    bundles = train_grid(features, horizons, confidences, models_dir=models_dir)
    bundles_by_pair = {(b["horizon_days"], b["confidence"]): b for b in bundles}

    for h in horizons:
        h_table = prepare_horizon_table(features, horizon_days=h)
        split = split_by_date(h_table, horizon_days=h, test_fraction=TEST_FRACTION)
        preds_90 = bundles_by_pair[(h, 0.90)]["model"].predict(split.test[FEATURE_COLUMNS])
        preds_99 = bundles_by_pair[(h, 0.99)]["model"].predict(split.test[FEATURE_COLUMNS])
        assert preds_99.mean() < preds_90.mean()


def test_load_grid_bundles_and_missing_file_raises(tmp_path) -> None:
    features = make_synthetic_grid_features(n_days=120, seed=42)
    models_dir = str(tmp_path)
    train_grid(features, [7], [0.90, 0.95], models_dir=models_dir)

    bundles = load_grid_bundles(7, [0.90, 0.95], models_dir=models_dir)
    assert set(bundles.keys()) == {0.90, 0.95}
    assert bundles[0.90]["horizon_days"] == 7
    assert bundles[0.95]["horizon_days"] == 7

    # Missing confidence raises FileNotFoundError
    with pytest.raises(FileNotFoundError, match="Model bundle not found"):
        load_grid_bundles(7, [0.99], models_dir=models_dir)

    # Missing horizon raises FileNotFoundError
    with pytest.raises(FileNotFoundError, match="Model bundle not found"):
        load_grid_bundles(14, [0.90], models_dir=models_dir)


def test_train_grid_inputs_not_mutated(tmp_path) -> None:
    features = make_synthetic_grid_features(n_days=120, seed=42)
    features_copy = features.copy(deep=True)
    horizons = [7, 14]
    confidences = [0.90, 0.95]

    _ = train_grid(features, horizons, confidences, models_dir=str(tmp_path))

    pd.testing.assert_frame_equal(features, features_copy)
    assert horizons == [7, 14]
    assert confidences == [0.90, 0.95]

