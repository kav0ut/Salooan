from datetime import datetime, timezone
import json
import os
import tempfile
from typing import Any, Dict, Sequence, Union

import numpy as np
import pandas as pd

from src.baselines import (
    historical_quantiles,
    predict_historical,
    predict_static,
    static_quantile,
)
from src.config import (
    DEFAULT_CONFIDENCE,
    DEFAULT_HORIZON_DAYS,
    FEATURE_COLUMNS,
    FEATURES_PATH,
    RESULTS_PATH,
    TEST_FRACTION,
)
from src.metrics import breach_rate, pinball_loss
from src.predict import to_haircut
from src.split import prepare_horizon_table, split_by_date
from src.train import load_bundle


def score_method(
    y_true: Union[np.ndarray, pd.Series, Sequence[float]],
    y_pred: Union[np.ndarray, pd.Series, Sequence[float]],
    alpha: float,
) -> Dict[str, Union[int, float]]:
    loss = pinball_loss(y_true, y_pred, alpha)
    rate = breach_rate(y_true, y_pred)
    haircuts = to_haircut(y_pred)
    return {
        "n_rows": int(len(y_true)),
        "pinball_loss": float(loss),
        "breach_rate": float(rate),
        "mean_haircut": float(np.mean(haircuts)),
    }


def evaluate_config(bundle: Dict[str, Any], features: pd.DataFrame) -> Dict[str, Any]:
    horizon_days = int(bundle["horizon_days"])
    confidence = float(bundle["confidence"])
    alpha = float(round(1.0 - confidence, 6))

    horizon_table = prepare_horizon_table(features, horizon_days=horizon_days)
    split = split_by_date(horizon_table, horizon_days=horizon_days, test_fraction=TEST_FRACTION)

    split_train_end_date = str(pd.to_datetime(split.train["date"]).max().date())
    bundle_train_end_date = str(bundle.get("train_end_date"))
    if bundle_train_end_date != split_train_end_date:
        raise ValueError(
            f"Stale model bundle: train_end_date in bundle ({bundle_train_end_date}) "
            f"differs from split max train date ({split_train_end_date}). "
            "Please retrain the model with 'python -m src.train'."
        )

    model = bundle["model"]
    test_X = split.test[FEATURE_COLUMNS]
    model_preds = model.predict(test_X)
    test_targets = split.test["target"].to_numpy()

    # Baselines calibrated strictly on the training split
    q_stat = static_quantile(split.train, confidence)
    q_hist = historical_quantiles(split.train, confidence)

    stat_preds = predict_static(split.test, q_stat)
    hist_preds = predict_historical(split.test, q_hist)

    methods = {
        "model": model_preds,
        "static": stat_preds,
        "historical": hist_preds,
    }

    assets = sorted(split.test["asset"].unique())
    method_scores: Dict[str, Dict[str, Any]] = {}

    for method_name, preds in methods.items():
        overall_score = score_method(test_targets, preds, alpha)
        per_asset_scores = {}
        for asset in assets:
            mask = (split.test["asset"] == asset).to_numpy()
            per_asset_scores[asset] = score_method(test_targets[mask], preds[mask], alpha)

        method_scores[method_name] = {
            "overall": overall_score,
            "per_asset": per_asset_scores,
        }

    # Comparison block: relative pinball improvement and distance from alpha
    stat_loss = method_scores["static"]["overall"]["pinball_loss"]
    hist_loss = method_scores["historical"]["overall"]["pinball_loss"]
    model_loss = method_scores["model"]["overall"]["pinball_loss"]

    improvement_vs_static = float((stat_loss - model_loss) / stat_loss * 100.0)
    improvement_vs_hist = float((hist_loss - model_loss) / hist_loss * 100.0)

    mean_abs_distances = {}
    for method_name in methods:
        distances = [
            abs(method_scores[method_name]["per_asset"][asset]["breach_rate"] - alpha)
            for asset in assets
        ]
        mean_abs_distances[method_name] = float(np.mean(distances))

    comparison = {
        "pinball_improvement_vs_static_pct": improvement_vs_static,
        "pinball_improvement_vs_historical_pct": improvement_vs_hist,
        "mean_abs_breach_distance_from_alpha": mean_abs_distances,
    }

    test_start_str = str(pd.to_datetime(split.test_start).date())

    return {
        "horizon_days": horizon_days,
        "confidence": confidence,
        "alpha": alpha,
        "test_start": test_start_str,
        "train_end_date": split_train_end_date,
        "n_train_rows": int(len(split.train)),
        "n_test_rows": int(len(split.test)),
        "methods": method_scores,
        "comparison": comparison,
    }


def write_results(results: Dict[str, Any], path: str = RESULTS_PATH) -> None:
    target_dir = os.path.dirname(path)
    os.makedirs(target_dir, exist_ok=True)

    # Atomic write to temporary file before replacing
    fd, temp_path = tempfile.mkstemp(dir=target_dir, prefix="tmp_results_", suffix=".json")
    os.close(fd)
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def main() -> None:
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"Feature table not found at {FEATURES_PATH}. Run 'python -m src.features' first.")

    features = pd.read_parquet(FEATURES_PATH)
    bundle = load_bundle(DEFAULT_HORIZON_DAYS, DEFAULT_CONFIDENCE)

    config_eval = evaluate_config(bundle, features)

    pct = round(DEFAULT_CONFIDENCE * 100)
    config_key = f"h{DEFAULT_HORIZON_DAYS}_c{pct}"

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "configs": {
            config_key: config_eval,
        },
    }

    write_results(payload, RESULTS_PATH)

    alpha = config_eval["alpha"]
    methods = config_eval["methods"]
    comparison = config_eval["comparison"]

    print("=" * 86)
    print("TEST-SET EVALUATION SUMMARY: MODEL VS BASELINES")
    print(f"Horizon: {config_eval['horizon_days']} Days | Confidence: {int(config_eval['confidence'] * 100)}% (alpha = {alpha:.2f})")
    print(f"Train Rows: {config_eval['n_train_rows']:,} (end: {config_eval['train_end_date']}) | Test Rows: {config_eval['n_test_rows']:,} (start: {config_eval['test_start']})")
    print("=" * 86)

    print("\nOVERALL PERFORMANCE BY METHOD:")
    print(f"{'Method':<16} {'Pinball Loss':<16} {'Breach Rate':<16} {'Mean Haircut':<16}")
    print("-" * 64)
    for method_name in ["model", "historical", "static"]:
        score = methods[method_name]["overall"]
        label = method_name.capitalize()
        print(f"{label:<16} {score['pinball_loss']:<16.6f} {score['breach_rate']:>10.2%}     {score['mean_haircut']:>10.2%}")

    print("\nPER-ASSET TEST BREACH RATES:")
    print(f"{'Asset':<10} {'Rows':<8} {'Model Breach':<16} {'Hist Breach':<16} {'Stat Breach':<16}")
    print("-" * 66)
    assets = sorted(methods["model"]["per_asset"].keys())
    for asset in assets:
        n_rows = methods["model"]["per_asset"][asset]["n_rows"]
        m_b = methods["model"]["per_asset"][asset]["breach_rate"]
        h_b = methods["historical"]["per_asset"][asset]["breach_rate"]
        s_b = methods["static"]["per_asset"][asset]["breach_rate"]
        print(f"{asset:<10} {n_rows:<8d} {m_b:>11.2%}      {h_b:>11.2%}      {s_b:>11.2%}")

    ov_m = methods["model"]["overall"]["breach_rate"]
    ov_h = methods["historical"]["overall"]["breach_rate"]
    ov_s = methods["static"]["overall"]["breach_rate"]
    print("-" * 66)
    print(f"{'OVERALL':<10} {config_eval['n_test_rows']:<8d} {ov_m:>11.2%}      {ov_h:>11.2%}      {ov_s:>11.2%}")

    print("\nCOMPARISON BLOCK:")
    p_vs_stat = comparison["pinball_improvement_vs_static_pct"]
    p_vs_hist = comparison["pinball_improvement_vs_historical_pct"]
    dist = comparison["mean_abs_breach_distance_from_alpha"]
    print(f"  Pinball improvement vs Static:     {p_vs_stat:+.2f}%")
    print(f"  Pinball improvement vs Historical: {p_vs_hist:+.2f}%")
    print("  Average absolute distance from alpha:")
    print(f"    Model:      {dist['model']:.4f}")
    print(f"    Historical: {dist['historical']:.4f}")
    print(f"    Static:     {dist['static']:.4f}")

    print(f"\nResults successfully written to: {RESULTS_PATH}")
    print("=" * 86)


if __name__ == "__main__":
    main()
