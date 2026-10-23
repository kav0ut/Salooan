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
from src.metrics import crossing_rate
from src.predict import (
    latest_features,
    max_loan,
    predict_haircuts,
    predict_multiquantile,
    predict_quantile_return,
    to_haircut,
)
from src.train import build_bundle, train_model


def make_synthetic_feature_table() -> pd.DataFrame:
    return pd.DataFrame({
        "asset": ["BTCUSDT", "BTCUSDT", "SPY", "SPY", "SPY"],
        "asset_class": ["crypto", "crypto", "equity", "equity", "equity"],
        "date": [
            "2026-01-01",
            "2026-01-02",
            "2026-01-01",
            "2026-01-02",
            "2026-01-03",
        ],
        VOLATILITY_COL: [0.03, 0.035, 0.012, 0.013, 0.015],
        VOLUME_COL: [2e9, 2.1e9, 4e10, 3.9e10, 4.1e10],
        "target_7d": [-0.02, np.nan, 0.005, -0.01, np.nan],
    })


def make_trained_bundle() -> dict:
    rng = np.random.default_rng(42)
    n_rows = 3000
    vol = rng.uniform(0.01, 0.05, size=n_rows)
    vol_dol = rng.uniform(1e7, 1e9, size=n_rows)
    target = rng.normal(0, vol, size=n_rows)
    dates = [pd.Timestamp("2023-01-01") + pd.Timedelta(days=int(i % 300)) for i in range(n_rows)]

    train_df = pd.DataFrame({
        "asset": [f"ASSET_{i % 2}" for i in range(n_rows)],
        "date": dates,
        VOLATILITY_COL: vol,
        VOLUME_COL: vol_dol,
        "target": target,
    })
    model = train_model(train_df, confidence=DEFAULT_CONFIDENCE)
    return build_bundle(model, DEFAULT_HORIZON_DAYS, DEFAULT_CONFIDENCE, train_df)


def test_to_haircut_scalars_and_arrays() -> None:
    # Negative quantile returns yield positive haircuts
    assert to_haircut(-0.22) == pytest.approx(0.22)
    # Positive returns floor at zero haircut
    assert to_haircut(0.03) == pytest.approx(0.0)
    # Catastrophic losses clip at 1.0 (100% haircut)
    assert to_haircut(-1.4) == pytest.approx(1.0)

    # Arrays are element-wise clipped
    arr = np.array([-0.22, 0.03, -1.4])
    result = to_haircut(arr)
    np.testing.assert_allclose(result, [0.22, 0.0, 1.0])


def test_max_loan_calculation() -> None:
    assert max_loan(10000.0, 0.22) == pytest.approx(7800.0)

    loans = max_loan(10000.0, np.array([0.22, 0.0, 1.0]))
    np.testing.assert_allclose(loans, [7800.0, 10000.0, 0.0])


def test_latest_features_picks_latest_date_including_nan_target() -> None:
    features = make_synthetic_feature_table()
    latest = latest_features(features)

    # One row per unique asset
    assert len(latest) == 2
    assert sorted(latest["asset"].tolist()) == ["BTCUSDT", "SPY"]

    # Selected dates must be the latest observation, even when target is NaN
    btc_row = latest[latest["asset"] == "BTCUSDT"].iloc[0]
    spy_row = latest[latest["asset"] == "SPY"].iloc[0]

    assert pd.to_datetime(btc_row["date"]) == pd.Timestamp("2026-01-02")
    assert pd.to_datetime(spy_row["date"]) == pd.Timestamp("2026-01-03")


def test_predict_haircuts_bounds_and_structure() -> None:
    bundle = make_trained_bundle()
    features = make_synthetic_feature_table()
    collateral = 10000.0

    predictions = predict_haircuts(bundle, features, collateral_value=collateral)

    expected_cols = ["asset", "date"] + FEATURE_COLUMNS + ["predicted_return", "haircut", "max_loan"]
    assert list(predictions.columns) == expected_cols
    assert len(predictions) == 2

    # Haircut bounds [0, 1] and max loan bounds [0, collateral]
    for _, row in predictions.iterrows():
        assert 0.0 <= row["haircut"] <= 1.0
        assert 0.0 <= row["max_loan"] <= collateral


def test_mismatched_bundle_feature_columns_raises_value_error() -> None:
    bundle = make_trained_bundle()
    bundle["feature_columns"] = ["stale_feature_a", "stale_feature_b"]
    features = make_synthetic_feature_table()

    with pytest.raises(ValueError, match="Bundle feature columns"):
        predict_quantile_return(bundle, features)

    with pytest.raises(ValueError, match="Bundle feature columns"):
        predict_haircuts(bundle, features, collateral_value=10000.0)


def test_higher_volatility_asset_receives_larger_haircut() -> None:
    bundle = make_trained_bundle()

    # Two assets with equal volume but markedly different 30d volatilities
    features = pd.DataFrame({
        "asset": ["LOW_VOL_ASSET", "HIGH_VOL_ASSET"],
        "date": ["2026-01-01", "2026-01-01"],
        VOLATILITY_COL: [0.01, 0.06],
        VOLUME_COL: [5e8, 5e8],
    })

    result = predict_haircuts(bundle, features, collateral_value=10000.0)
    low_vol_hc = result.loc[result["asset"] == "LOW_VOL_ASSET", "haircut"].iloc[0]
    high_vol_hc = result.loc[result["asset"] == "HIGH_VOL_ASSET", "haircut"].iloc[0]

    assert high_vol_hc > low_vol_hc


def test_inputs_not_mutated() -> None:
    bundle = make_trained_bundle()
    features = make_synthetic_feature_table()
    features_copy = features.copy(deep=True)

    _ = latest_features(features)
    pd.testing.assert_frame_equal(features, features_copy)

    _ = predict_haircuts(bundle, features, collateral_value=10000.0)
    pd.testing.assert_frame_equal(features, features_copy)


def test_predict_multiquantile_monotone_rearrangement() -> None:
    features = make_synthetic_feature_table()

    class DummyModel:
        def __init__(self, val: float):
            self.val = val

        def predict(self, X):
            return np.full(len(X), self.val)

    # 0.95 predicts -0.02 (violating/crossing with 0.90 predicting -0.05)
    b_90 = {"model": DummyModel(-0.05), "feature_columns": list(FEATURE_COLUMNS), "confidence": 0.90, "horizon_days": 7}
    b_95 = {"model": DummyModel(-0.02), "feature_columns": list(FEATURE_COLUMNS), "confidence": 0.95, "horizon_days": 7}
    b_99 = {"model": DummyModel(-0.20), "feature_columns": list(FEATURE_COLUMNS), "confidence": 0.99, "horizon_days": 7}

    bundles = {0.90: b_90, 0.95: b_95, 0.99: b_99}

    raw_preds = predict_multiquantile(bundles, features, rearrange=False)
    assert crossing_rate(raw_preds) > 0.0

    rearranged_preds = predict_multiquantile(bundles, features, rearrange=True)
    assert crossing_rate(rearranged_preds) == pytest.approx(0.0)
    # 0.90 should get -0.02, 0.95 should get -0.05, 0.99 should get -0.20
    np.testing.assert_allclose(rearranged_preds[0.90], -0.02)
    np.testing.assert_allclose(rearranged_preds[0.95], -0.05)
    np.testing.assert_allclose(rearranged_preds[0.99], -0.20)
