import copy
import json
import numpy as np
import pandas as pd
import pytest

from src.config import (
    CONFIDENCE_LEVELS,
    DEFAULT_CONFIDENCE,
    DEFAULT_HORIZON_DAYS,
    FEATURE_COLUMNS,
    HORIZONS_DAYS,
    TEST_FRACTION,
    VOLATILITY_COL,
    VOLUME_COL,
)
from src.evaluate import evaluate_config, evaluate_grid, score_method, write_results
from src.predict import to_haircut
from src.split import prepare_horizon_table, split_by_date
from src.train import build_bundle, config_key, train_model


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


def test_stale_bundle_raises_value_error() -> None:
    bundle, features = make_synthetic_bundle_and_features()
    # Artificially alter train_end_date to simulate a stale model
    bundle["train_end_date"] = "1999-01-01"

    with pytest.raises(ValueError, match="Stale model bundle"):
        evaluate_config(bundle, features)


def test_methods_structure_and_assets() -> None:
    bundle, features = make_synthetic_bundle_and_features()
    result = evaluate_config(bundle, features)

    expected_methods = {"model", "static", "historical"}
    assert set(result["methods"].keys()) == expected_methods

    expected_assets = {"BTC_TEST", "ETH_TEST", "SPY_TEST"}
    for method_name in expected_methods:
        method_dict = result["methods"][method_name]
        assert "overall" in method_dict
        assert "per_asset" in method_dict
        assert set(method_dict["per_asset"].keys()) == expected_assets


def test_row_counts_consistency() -> None:
    bundle, features = make_synthetic_bundle_and_features()
    result = evaluate_config(bundle, features)

    ht = prepare_horizon_table(features, DEFAULT_HORIZON_DAYS)
    split = split_by_date(ht, DEFAULT_HORIZON_DAYS, TEST_FRACTION)
    expected_test_rows = len(split.test)

    assert result["n_test_rows"] == expected_test_rows

    for method_name in ["model", "static", "historical"]:
        overall = result["methods"][method_name]["overall"]
        assert overall["n_rows"] == expected_test_rows

        per_asset = result["methods"][method_name]["per_asset"]
        sum_rows = sum(asset_data["n_rows"] for asset_data in per_asset.values())
        assert sum_rows == expected_test_rows


def test_overall_numbers_match_independent_recomputation() -> None:
    bundle, features = make_synthetic_bundle_and_features()
    result = evaluate_config(bundle, features)

    ht = prepare_horizon_table(features, DEFAULT_HORIZON_DAYS)
    split = split_by_date(ht, DEFAULT_HORIZON_DAYS, TEST_FRACTION)

    y_test = split.test["target"].to_numpy()
    preds = bundle["model"].predict(split.test[FEATURE_COLUMNS])
    alpha = result["alpha"]

    # Independent numpy calculation
    diff = y_test - preds
    expected_loss = float(np.mean(np.maximum(alpha * diff, (alpha - 1.0) * diff)))
    expected_breach = float(np.mean(y_test < preds))
    expected_haircut = float(np.mean(to_haircut(preds)))

    model_overall = result["methods"]["model"]["overall"]
    assert model_overall["pinball_loss"] == pytest.approx(expected_loss)
    assert model_overall["breach_rate"] == pytest.approx(expected_breach)
    assert model_overall["mean_haircut"] == pytest.approx(expected_haircut)


def test_comparison_percentages_and_distances() -> None:
    bundle, features = make_synthetic_bundle_and_features()
    result = evaluate_config(bundle, features)

    methods = result["methods"]
    comparison = result["comparison"]

    m_loss = methods["model"]["overall"]["pinball_loss"]
    s_loss = methods["static"]["overall"]["pinball_loss"]
    h_loss = methods["historical"]["overall"]["pinball_loss"]

    expected_vs_static = (s_loss - m_loss) / s_loss * 100.0
    expected_vs_hist = (h_loss - m_loss) / h_loss * 100.0

    assert comparison["pinball_improvement_vs_static_pct"] == pytest.approx(expected_vs_static)
    assert comparison["pinball_improvement_vs_historical_pct"] == pytest.approx(expected_vs_hist)

    alpha = result["alpha"]
    for method_name in ["model", "static", "historical"]:
        per_asset = methods[method_name]["per_asset"]
        expected_dist = float(np.mean([abs(d["breach_rate"] - alpha) for d in per_asset.values()]))
        assert comparison["mean_abs_breach_distance_from_alpha"][method_name] == pytest.approx(expected_dist)


def test_breach_rates_and_haircut_bounds() -> None:
    bundle, features = make_synthetic_bundle_and_features()
    result = evaluate_config(bundle, features)

    for method_name, method_data in result["methods"].items():
        assert 0.0 <= method_data["overall"]["breach_rate"] <= 1.0
        assert 0.0 <= method_data["overall"]["mean_haircut"] <= 1.0

        for asset, asset_data in method_data["per_asset"].items():
            assert 0.0 <= asset_data["breach_rate"] <= 1.0
            assert 0.0 <= asset_data["mean_haircut"] <= 1.0


def test_write_results_round_trip_and_types(tmp_path) -> None:
    bundle, features = make_synthetic_bundle_and_features()
    config_eval = evaluate_config(bundle, features)
    out_file = tmp_path / "results.json"

    payload = {
        "generated_at": "2026-10-22T17:00:00Z",
        "configs": {
            "h7_c95": config_eval,
        },
    }

    write_results(payload, path=str(out_file))
    assert out_file.exists()

    with open(out_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    assert "h7_c95" in loaded["configs"]
    cfg = loaded["configs"]["h7_c95"]
    assert cfg["horizon_days"] == 7
    assert cfg["confidence"] == 0.95

    # Check that loaded values are native Python primitives
    assert isinstance(cfg["comparison"]["pinball_improvement_vs_static_pct"], float)
    assert isinstance(cfg["n_train_rows"], int)
    assert isinstance(cfg["train_end_date"], str)


def test_evaluating_twice_is_deterministic() -> None:
    bundle, features = make_synthetic_bundle_and_features()

    eval_a = evaluate_config(bundle, features)
    eval_b = evaluate_config(bundle, features)

    assert eval_a == eval_b


def test_inputs_not_mutated() -> None:
    bundle, features = make_synthetic_bundle_and_features()
    bundle_copy = copy.deepcopy(bundle)
    features_copy = features.copy(deep=True)

    _ = evaluate_config(bundle, features)

    pd.testing.assert_frame_equal(features, features_copy)
    assert bundle["train_end_date"] == bundle_copy["train_end_date"]
    assert bundle["feature_columns"] == bundle_copy["feature_columns"]


def make_synthetic_grid_bundles_and_features():
    features = make_synthetic_features()
    horizons = [7, 14]
    confidences = [0.90, 0.95, 0.99]
    bundles = []
    for h in horizons:
        ht = prepare_horizon_table(features, h)
        split = split_by_date(ht, h, TEST_FRACTION)
        for c in confidences:
            model = train_model(split.train, c)
            bundle = build_bundle(model, h, c, split.train)
            bundles.append(bundle)
    return bundles, features


def test_evaluate_grid_keys_methods_and_crossing() -> None:
    bundles, features = make_synthetic_grid_bundles_and_features()
    payload = evaluate_grid(bundles, features)

    expected_keys = {
        config_key(h, c)
        for h in [7, 14]
        for c in [0.90, 0.95, 0.99]
    }
    assert set(payload["configs"].keys()) == expected_keys

    for key, cfg in payload["configs"].items():
        assert set(cfg["methods"].keys()) == {"model", "static", "historical"}

    assert set(payload["crossing"].keys()) == {"h7", "h14"}
    for h_key, cross_data in payload["crossing"].items():
        assert cross_data["n_rows"] > 0
        assert 0.0 <= cross_data["crossing_rate"] <= 1.0
        assert cross_data["confidences"] == [0.90, 0.95, 0.99]


def test_evaluate_grid_stale_bundle_raises() -> None:
    bundles, features = make_synthetic_grid_bundles_and_features()
    bundles[0]["train_end_date"] = "1999-01-01"

    with pytest.raises(ValueError, match="Stale model bundle"):
        evaluate_grid(bundles, features)


def test_evaluate_grid_json_roundtrip_and_types(tmp_path) -> None:
    bundles, features = make_synthetic_grid_bundles_and_features()
    payload = evaluate_grid(bundles, features)

    out_file = tmp_path / "results.json"
    write_results(payload, path=str(out_file))
    assert out_file.exists()

    with open(out_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    assert set(loaded.keys()) == {"generated_at", "configs", "crossing"}
    assert set(loaded["configs"].keys()) == set(payload["configs"].keys())
    assert set(loaded["crossing"].keys()) == {"h7", "h14"}

    for h_key in ["h7", "h14"]:
        assert isinstance(loaded["crossing"][h_key]["n_rows"], int)
        assert isinstance(loaded["crossing"][h_key]["crossing_rate"], float)
        assert isinstance(loaded["crossing"][h_key]["confidences"], list)


def test_evaluate_grid_evaluating_twice_is_identical_except_generated_at() -> None:
    bundles, features = make_synthetic_grid_bundles_and_features()

    run_1 = evaluate_grid(bundles, features)
    run_2 = evaluate_grid(bundles, features)

    run_1_copy = copy.deepcopy(run_1)
    run_2_copy = copy.deepcopy(run_2)

    del run_1_copy["generated_at"]
    del run_2_copy["generated_at"]

    assert run_1_copy == run_2_copy

