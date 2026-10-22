import copy
import json
import os
import numpy as np
import pandas as pd
import pytest

from src.baselines import historical_quantiles, static_quantile
from src.config import (
    DEFAULT_CONFIDENCE,
    DEFAULT_HORIZON_DAYS,
    FEATURE_COLUMNS,
    TEST_FRACTION,
    VOLATILITY_COL,
    VOLUME_COL,
)
from src.dashboard import (
    baseline_haircuts,
    calculator_result,
    evaluation_tables,
    latest_crossing,
    load_results,
    risk_band,
    test_period_start,
)
from src.predict import to_haircut
from src.split import prepare_horizon_table, split_by_date
from src.train import build_bundle, train_model


def make_synthetic_features() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    assets = ["BTC_TEST", "ETH_TEST", "SPY_TEST"]
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    rows = []
    for d in dates:
        for a in assets:
            vol = float(rng.uniform(0.01, 0.05))
            dvol = float(rng.uniform(1e6, 1e8))
            target_7d = float(rng.normal(0.0, vol))
            rows.append({
                "asset": a,
                "asset_class": "crypto" if "TEST" in a and a != "SPY_TEST" else "equity",
                "date": d,
                VOLATILITY_COL: vol,
                VOLUME_COL: dvol,
                "target_7d": target_7d,
                "target_14d": target_7d,
            })
    return pd.DataFrame(rows)


def make_synthetic_bundle_and_features():
    features = make_synthetic_features()
    ht = prepare_horizon_table(features, DEFAULT_HORIZON_DAYS)
    split = split_by_date(ht, DEFAULT_HORIZON_DAYS, TEST_FRACTION)
    model = train_model(split.train, DEFAULT_CONFIDENCE)
    bundle = build_bundle(model, DEFAULT_HORIZON_DAYS, DEFAULT_CONFIDENCE, split.train)
    return bundle, features


def make_synthetic_results():
    return {
        "generated_at": "2026-10-22T12:00:00Z",
        "configs": {
            "h7_c95": {
                "horizon_days": 7,
                "confidence": 0.95,
                "alpha": 0.05,
                "test_start": "2023-03-20",
                "train_end_date": "2023-03-12",
                "n_train_rows": 210,
                "n_test_rows": 60,
                "methods": {
                    "model": {
                        "overall": {
                            "n_rows": 60,
                            "pinball_loss": 0.008,
                            "breach_rate": 0.05,
                            "mean_haircut": 0.09,
                        },
                        "per_asset": {
                            "BTC_TEST": {
                                "n_rows": 20,
                                "pinball_loss": 0.009,
                                "breach_rate": 0.05,
                                "mean_haircut": 0.10,
                            },
                            "ETH_TEST": {
                                "n_rows": 20,
                                "pinball_loss": 0.008,
                                "breach_rate": 0.05,
                                "mean_haircut": 0.09,
                            },
                            "SPY_TEST": {
                                "n_rows": 20,
                                "pinball_loss": 0.007,
                                "breach_rate": 0.05,
                                "mean_haircut": 0.08,
                            },
                        },
                    },
                    "historical": {
                        "overall": {
                            "n_rows": 60,
                            "pinball_loss": 0.010,
                            "breach_rate": 0.06,
                            "mean_haircut": 0.11,
                        },
                        "per_asset": {
                            "BTC_TEST": {
                                "n_rows": 20,
                                "pinball_loss": 0.011,
                                "breach_rate": 0.05,
                                "mean_haircut": 0.12,
                            },
                            "ETH_TEST": {
                                "n_rows": 20,
                                "pinball_loss": 0.010,
                                "breach_rate": 0.05,
                                "mean_haircut": 0.11,
                            },
                            "SPY_TEST": {
                                "n_rows": 20,
                                "pinball_loss": 0.009,
                                "breach_rate": 0.05,
                                "mean_haircut": 0.09,
                            },
                        },
                    },
                    "static": {
                        "overall": {
                            "n_rows": 60,
                            "pinball_loss": 0.012,
                            "breach_rate": 0.07,
                            "mean_haircut": 0.12,
                        },
                        "per_asset": {
                            "BTC_TEST": {
                                "n_rows": 20,
                                "pinball_loss": 0.012,
                                "breach_rate": 0.07,
                                "mean_haircut": 0.12,
                            },
                            "ETH_TEST": {
                                "n_rows": 20,
                                "pinball_loss": 0.012,
                                "breach_rate": 0.07,
                                "mean_haircut": 0.12,
                            },
                            "SPY_TEST": {
                                "n_rows": 20,
                                "pinball_loss": 0.012,
                                "breach_rate": 0.07,
                                "mean_haircut": 0.12,
                            },
                        },
                    },
                },
                "comparison": {
                    "pinball_improvement_vs_static_pct": 33.3,
                    "pinball_improvement_vs_historical_pct": 20.0,
                    "mean_abs_breach_distance_from_alpha": {
                        "model": 0.01,
                        "historical": 0.02,
                        "static": 0.03,
                    },
                },
            }
        },
    }


def test_load_results(tmp_path) -> None:
    missing_file = tmp_path / "missing.json"
    assert load_results(missing_file) is None

    valid_file = tmp_path / "valid.json"
    data = {"sample_key": "sample_val"}
    with open(valid_file, "w", encoding="utf-8") as f:
        json.dump(data, f)

    loaded = load_results(valid_file)
    assert loaded == data


def test_calculator_result() -> None:
    bundle, features = make_synthetic_bundle_and_features()
    collateral = 10000.0

    res = calculator_result(bundle, features, "BTC_TEST", collateral)

    assert res["asset"] == "BTC_TEST"
    assert res["haircut"] == to_haircut(res["predicted_return"])
    assert res["max_loan"] == pytest.approx(collateral * (1.0 - res["haircut"]))

    # Date must be the asset's latest feature date
    btc_dates = features[features["asset"] == "BTC_TEST"]["date"]
    expected_latest_date = pd.to_datetime(btc_dates.max())
    assert pd.to_datetime(res["date"]) == expected_latest_date

    # Unknown asset raises ValueError
    with pytest.raises(ValueError, match="Unknown asset"):
        calculator_result(bundle, features, "NONEXISTENT", collateral)

    # Negative collateral raises ValueError
    with pytest.raises(ValueError, match="negative"):
        calculator_result(bundle, features, "BTC_TEST", -500.0)


def test_baseline_haircuts() -> None:
    _, features = make_synthetic_bundle_and_features()

    ht = prepare_horizon_table(features, DEFAULT_HORIZON_DAYS)
    split = split_by_date(ht, DEFAULT_HORIZON_DAYS, TEST_FRACTION)

    expected_static = float(to_haircut(static_quantile(split.train, DEFAULT_CONFIDENCE)))
    hist_quantiles = historical_quantiles(split.train, DEFAULT_CONFIDENCE)
    expected_hist = float(to_haircut(hist_quantiles["BTC_TEST"]))

    res = baseline_haircuts(features, DEFAULT_HORIZON_DAYS, DEFAULT_CONFIDENCE, "BTC_TEST")

    assert res["static"] == pytest.approx(expected_static)
    assert res["historical"] == pytest.approx(expected_hist)

    with pytest.raises(ValueError, match="Asset 'NONEXISTENT' not found"):
        baseline_haircuts(features, DEFAULT_HORIZON_DAYS, DEFAULT_CONFIDENCE, "NONEXISTENT")


def test_risk_band() -> None:
    bundle, features = make_synthetic_bundle_and_features()

    # Generate synthetic prices with constant close
    prices = pd.DataFrame({
        "asset": features["asset"],
        "date": features["date"],
        "close": 100.0,
    })

    rb = risk_band(bundle, features, prices, "BTC_TEST", DEFAULT_HORIZON_DAYS, lookback_days=30)
    assert not rb.empty

    # floor_price equals close * (1 + q) computed independently
    btc_features = features[features["asset"] == "BTC_TEST"].copy()
    btc_features["date"] = pd.to_datetime(btc_features["date"])
    max_d = btc_features["date"].max()
    sub_features = btc_features[btc_features["date"] >= (max_d - pd.Timedelta(days=30))].sort_values("date")
    q_indep = bundle["model"].predict(sub_features[FEATURE_COLUMNS])
    expected_floors = 100.0 * (1.0 + q_indep)

    np.testing.assert_allclose(rb["floor_price"].to_numpy(), expected_floors, rtol=1e-5)

    # floor_date equals date plus the horizon
    expected_floor_dates = rb["date"] + pd.Timedelta(days=DEFAULT_HORIZON_DAYS)
    pd.testing.assert_series_equal(rb["floor_date"], expected_floor_dates, check_names=False)

    # Lookback window is respected
    assert rb["date"].min() >= max_d - pd.Timedelta(days=30)

    # Output is sorted chronologically by date
    assert rb["date"].is_monotonic_increasing


def test_risk_band_drops_missing_features() -> None:
    bundle, features = make_synthetic_bundle_and_features()
    # Introduce NaN into a feature row
    features_with_nan = features.copy()
    features_with_nan.loc[0, VOLATILITY_COL] = np.nan
    nan_date = pd.to_datetime(features_with_nan.loc[0, "date"])
    asset = features_with_nan.loc[0, "asset"]

    prices = pd.DataFrame({
        "asset": features_with_nan["asset"],
        "date": features_with_nan["date"],
        "close": 50.0,
    })

    rb = risk_band(bundle, features_with_nan, prices, asset, DEFAULT_HORIZON_DAYS, lookback_days=100)
    assert nan_date not in rb["date"].values


def test_test_period_start() -> None:
    _, features = make_synthetic_bundle_and_features()

    ht = prepare_horizon_table(features, DEFAULT_HORIZON_DAYS)
    split = split_by_date(ht, DEFAULT_HORIZON_DAYS, TEST_FRACTION)

    t_start = test_period_start(features, DEFAULT_HORIZON_DAYS)
    assert t_start == split.test_start


def test_evaluation_tables() -> None:
    results = make_synthetic_results()

    summary_df, per_asset_df, comparison_dict, info_dict = evaluation_tables(results, "h7_c95")

    # Summary has one row per method
    assert len(summary_df) == 3
    assert set(summary_df["method"]) == {"model", "historical", "static"}

    # Per-asset table has every asset
    expected_assets = {"BTC_TEST", "ETH_TEST", "SPY_TEST"}
    assert set(per_asset_df["asset"].unique()) == expected_assets

    # Comparison dict has expected keys
    assert "pinball_improvement_vs_static_pct" in comparison_dict
    assert "pinball_improvement_vs_historical_pct" in comparison_dict
    assert "mean_abs_breach_distance_from_alpha" in comparison_dict

    # Info dict contains facts about the run
    assert info_dict["horizon_days"] == 7
    assert info_dict["confidence"] == 0.95
    assert info_dict["test_start"] == "2023-03-20"
    assert info_dict["generated_at"] == "2026-10-22T12:00:00Z"

    # Missing config key raises KeyError
    with pytest.raises(KeyError, match="Config key 'h99_c99' not found"):
        evaluation_tables(results, "h99_c99")


def test_functions_do_not_mutate_inputs_or_write_files(tmp_path) -> None:
    bundle, features = make_synthetic_bundle_and_features()
    prices = pd.DataFrame({
        "asset": features["asset"],
        "date": features["date"],
        "close": 120.0,
    })
    results = make_synthetic_results()

    features_copy = features.copy(deep=True)
    prices_copy = prices.copy(deep=True)
    bundle_copy = copy.deepcopy(bundle)
    results_copy = copy.deepcopy(results)

    # Track directory contents to verify no files are written
    files_before = set(os.listdir(tmp_path))

    _ = calculator_result(bundle, features, "BTC_TEST", 1000.0)
    _ = baseline_haircuts(features, DEFAULT_HORIZON_DAYS, DEFAULT_CONFIDENCE, "BTC_TEST")
    _ = risk_band(bundle, features, prices, "BTC_TEST", DEFAULT_HORIZON_DAYS, lookback_days=30)
    _ = test_period_start(features, DEFAULT_HORIZON_DAYS)
    _ = evaluation_tables(results, "h7_c95")

    files_after = set(os.listdir(tmp_path))
    assert files_before == files_after

    pd.testing.assert_frame_equal(features, features_copy)
    pd.testing.assert_frame_equal(prices, prices_copy)
    assert bundle["train_end_date"] == bundle_copy["train_end_date"]
    assert bundle["feature_columns"] == bundle_copy["feature_columns"]
    assert results == results_copy


def test_latest_crossing() -> None:
    features = make_synthetic_features()

    class DummyModel:
        def __init__(self, val: float):
            self.val = val

        def predict(self, X):
            return np.full(len(X), self.val)

    b_90 = {"model": DummyModel(-0.05), "feature_columns": list(FEATURE_COLUMNS), "confidence": 0.90, "horizon_days": 7}
    b_95 = {"model": DummyModel(-0.10), "feature_columns": list(FEATURE_COLUMNS), "confidence": 0.95, "horizon_days": 7}
    b_99 = {"model": DummyModel(-0.20), "feature_columns": list(FEATURE_COLUMNS), "confidence": 0.99, "horizon_days": 7}

    ordered_bundles = {0.90: b_90, 0.95: b_95, 0.99: b_99}
    # Properly ordered predictions must return None
    assert latest_crossing(ordered_bundles, features, "BTC_TEST") is None

    # Crossing case: 95% predicts higher return than 90% (-0.02 > -0.05)
    b_cross_95 = {"model": DummyModel(-0.02), "feature_columns": list(FEATURE_COLUMNS), "confidence": 0.95, "horizon_days": 7}
    crossing_bundles = {0.90: b_90, 0.95: b_cross_95, 0.99: b_99}
    res = latest_crossing(crossing_bundles, features, "BTC_TEST")
    assert res is not None
    assert res[0.90] == pytest.approx(-0.05)
    assert res[0.95] == pytest.approx(-0.02)
    assert res[0.99] == pytest.approx(-0.20)

    # Unknown asset raises ValueError
    with pytest.raises(ValueError, match="Unknown asset"):
        latest_crossing(ordered_bundles, features, "UNKNOWN_ASSET")

    # Inputs are not mutated
    features_copy = features.copy(deep=True)
    _ = latest_crossing(crossing_bundles, features, "BTC_TEST")
    pd.testing.assert_frame_equal(features, features_copy)

