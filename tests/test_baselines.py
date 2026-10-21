import numpy as np
import pandas as pd
import pytest

from src.baselines import (
    baseline_predictions,
    historical_quantiles,
    predict_historical,
    predict_static,
    static_quantile,
)


@pytest.fixture
def synthetic_train_test():
    rng = np.random.RandomState(42)
    n_train_a = 500
    n_train_b = 500

    # Asset A has lower volatility, Asset B has higher volatility
    train_a = pd.DataFrame(
        {
            "asset": ["A"] * n_train_a,
            "target": rng.normal(loc=0.0, scale=0.02, size=n_train_a),
        }
    )
    train_b = pd.DataFrame(
        {
            "asset": ["B"] * n_train_b,
            "target": rng.normal(loc=0.0, scale=0.08, size=n_train_b),
        }
    )
    train = pd.concat([train_a, train_b], ignore_index=True)

    n_test_a = 50
    n_test_b = 50
    test_a = pd.DataFrame(
        {
            "asset": ["A"] * n_test_a,
            "target": rng.normal(loc=0.0, scale=0.02, size=n_test_a),
        }
    )
    test_b = pd.DataFrame(
        {
            "asset": ["B"] * n_test_b,
            "target": rng.normal(loc=0.0, scale=0.08, size=n_test_b),
        }
    )
    # Interleave to verify exact row ordering preservation
    test = pd.concat([test_a, test_b], ignore_index=True).sample(frac=1.0, random_state=7).reset_index(drop=True)

    return train, test


def test_static_quantile_matches_numpy(synthetic_train_test):
    train, _ = synthetic_train_test
    conf = 0.95
    alpha = 0.05
    expected = float(np.quantile(train["target"].to_numpy(), alpha))
    actual = static_quantile(train, conf)
    assert actual == pytest.approx(expected, rel=1e-7)


def test_historical_quantiles_match_numpy_per_asset(synthetic_train_test):
    train, _ = synthetic_train_test
    conf = 0.90
    alpha = 0.10
    quantiles = historical_quantiles(train, conf)

    for asset in ["A", "B"]:
        asset_targets = train.loc[train["asset"] == asset, "target"].to_numpy()
        expected = float(np.quantile(asset_targets, alpha))
        assert quantiles[asset] == pytest.approx(expected, rel=1e-7)


def test_in_sample_breach_rate_close_to_alpha():
    # Large random sample to verify calibration on training set
    rng = np.random.RandomState(123)
    n = 10000
    train = pd.DataFrame(
        {
            "asset": ["X"] * (n // 2) + ["Y"] * (n // 2),
            "target": rng.normal(loc=0.0, scale=0.05, size=n),
        }
    )
    conf = 0.95
    alpha = 0.05

    # Static baseline in-sample breach rate
    q_stat = static_quantile(train, conf)
    static_breach = np.mean(train["target"].to_numpy() < q_stat)
    assert static_breach == pytest.approx(alpha, abs=0.015)

    # Historical baseline in-sample breach rate
    q_hist = historical_quantiles(train, conf)
    hist_preds = predict_historical(train, q_hist)
    hist_breach = np.mean(train["target"].to_numpy() < hist_preds)
    assert hist_breach == pytest.approx(alpha, abs=0.015)


def test_fit_on_train_only_test_targets_do_not_affect_fitted_values(synthetic_train_test):
    train, test = synthetic_train_test
    conf = 0.95

    # Baseline fit on train
    stat_orig = static_quantile(train, conf)
    hist_orig = historical_quantiles(train, conf)

    # Mutate test targets drastically
    test_mutated = test.copy(deep=True)
    test_mutated["target"] = -9999.0

    stat_after = static_quantile(train, conf)
    hist_after = historical_quantiles(train, conf)

    assert stat_orig == stat_after
    assert hist_orig == hist_after


def test_predictions_aligned_with_test_table_length_and_order(synthetic_train_test):
    train, test = synthetic_train_test
    conf = 0.95
    preds = baseline_predictions(train, test, conf)

    assert len(preds["static"]) == len(test)
    assert len(preds["historical"]) == len(test)

    q_hist = historical_quantiles(train, conf)
    for idx, row in test.iterrows():
        expected_hist = q_hist[row["asset"]]
        assert preds["historical"][idx] == pytest.approx(expected_hist)


def test_static_prediction_constant_and_historical_varies(synthetic_train_test):
    train, test = synthetic_train_test
    conf = 0.95
    preds = baseline_predictions(train, test, conf)

    # Static prediction is identical across all rows
    assert np.all(preds["static"] == preds["static"][0])

    # Historical prediction differs between volatile and stable assets
    assert not np.all(preds["historical"] == preds["historical"][0])
    mask_a = (test["asset"] == "A").to_numpy()
    mask_b = (test["asset"] == "B").to_numpy()
    # Volatile asset B should have a more negative quantile than asset A
    assert np.mean(preds["historical"][mask_b]) < np.mean(preds["historical"][mask_a])


def test_higher_confidence_gives_lower_quantile(synthetic_train_test):
    train, _ = synthetic_train_test
    # 99% VaR must be deeper in the tail (more negative return) than 90%
    q_90 = static_quantile(train, 0.90)
    q_95 = static_quantile(train, 0.95)
    q_99 = static_quantile(train, 0.99)

    assert q_99 < q_95 < q_90

    h_90 = historical_quantiles(train, 0.90)
    h_95 = historical_quantiles(train, 0.95)
    h_99 = historical_quantiles(train, 0.99)

    for asset in ["A", "B"]:
        assert h_99[asset] < h_95[asset] < h_90[asset]


def test_unseen_asset_in_test_raises_value_error(synthetic_train_test):
    train, _ = synthetic_train_test
    unseen_test = pd.DataFrame(
        {
            "asset": ["UNSEEN_ASSET"],
            "target": [-0.05],
        }
    )
    with pytest.raises(ValueError, match="missing from fitted historical quantiles"):
        baseline_predictions(train, unseen_test, confidence=0.95)


def test_inputs_not_mutated(synthetic_train_test):
    train, test = synthetic_train_test
    train_copy = train.copy(deep=True)
    test_copy = test.copy(deep=True)

    _ = baseline_predictions(train, test, confidence=0.95)

    pd.testing.assert_frame_equal(train, train_copy)
    pd.testing.assert_frame_equal(test, test_copy)
