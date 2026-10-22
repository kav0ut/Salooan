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
    CONFIDENCE_LEVELS,
    DEFAULT_CONFIDENCE,
    DEFAULT_HORIZON_DAYS,
    FEATURE_COLUMNS,
    FEATURES_PATH,
    HORIZONS_DAYS,
    RESULTS_PATH,
    TEST_FRACTION,
)
from src.metrics import breach_rate, crossing_rate, pinball_loss
from src.predict import to_haircut
from src.split import prepare_horizon_table, split_by_date
from src.train import config_key, load_bundle, load_grid_bundles


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


def evaluate_grid(
    bundles: Union[Sequence[Dict[str, Any]], Dict[Any, Dict[str, Any]]],
    features: pd.DataFrame,
) -> Dict[str, Any]:
    if isinstance(bundles, dict):
        bundle_list = list(bundles.values())
    else:
        bundle_list = list(bundles)

    configs: Dict[str, Any] = {}
    for bundle in bundle_list:
        h = int(bundle["horizon_days"])
        c = float(bundle["confidence"])
        key = config_key(h, c)
        configs[key] = evaluate_config(bundle, features)

    # Build crossing block per horizon from the TEST rows
    horizons = sorted({int(b["horizon_days"]) for b in bundle_list})
    crossing: Dict[str, Any] = {}

    for h in horizons:
        h_table = prepare_horizon_table(features, horizon_days=h)
        split = split_by_date(h_table, horizon_days=h, test_fraction=TEST_FRACTION)
        test_X = split.test[FEATURE_COLUMNS]

        # Gather bundles for this horizon sorted by confidence ascending
        h_bundles = sorted(
            [b for b in bundle_list if int(b["horizon_days"]) == h],
            key=lambda b: float(b["confidence"]),
        )
        preds_by_conf = {
            float(b["confidence"]): b["model"].predict(test_X)
            for b in h_bundles
        }
        rate = crossing_rate(preds_by_conf)
        crossing[f"h{h}"] = {
            "n_rows": int(len(split.test)),
            "crossing_rate": float(rate),
            "confidences": [float(b["confidence"]) for b in h_bundles],
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "configs": configs,
        "crossing": crossing,
    }


def main() -> None:
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"Feature table not found at {FEATURES_PATH}. Run 'python -m src.features' first.")

    features = pd.read_parquet(FEATURES_PATH)
    bundles = []
    for horizon in HORIZONS_DAYS:
        horizon_bundles = load_grid_bundles(horizon, CONFIDENCE_LEVELS)
        bundles.extend(horizon_bundles.values())

    payload = evaluate_grid(bundles, features)
    write_results(payload, RESULTS_PATH)

    print("=" * 98)
    print("MODEL GRID TEST-SET EVALUATION SUMMARY")
    print("=" * 98)
    print(
        f"{'Pair':<10} {'Pinball Loss':<16} {'Breach (Target)':<20} {'Mean Haircut':<16} "
        f"{'vs Hist (%)':<16} {'vs Static (%)':<16}"
    )
    print("-" * 98)

    for horizon in HORIZONS_DAYS:
        for conf in CONFIDENCE_LEVELS:
            key = config_key(horizon, conf)
            cfg = payload["configs"][key]
            m_score = cfg["methods"]["model"]["overall"]
            loss = m_score["pinball_loss"]
            breach = m_score["breach_rate"]
            alpha = cfg["alpha"]
            haircut = m_score["mean_haircut"]
            imp_hist = cfg["comparison"]["pinball_improvement_vs_historical_pct"]
            imp_stat = cfg["comparison"]["pinball_improvement_vs_static_pct"]

            breach_str = f"{breach:>7.2%} ({alpha:>5.1%})"
            print(
                f"{key:<10} {loss:<16.6f} {breach_str:<20} {haircut:>12.2%}     "
                f"{imp_hist:>+11.2f}%   {imp_stat:>+11.2f}%"
            )

    print("-" * 98)
    print("QUANTILE CROSSING DIAGNOSTIC:")
    for horizon in HORIZONS_DAYS:
        h_key = f"h{horizon}"
        cross_data = payload["crossing"][h_key]
        n_rows = cross_data["n_rows"]
        rate = cross_data["crossing_rate"]
        print(f"  Horizon {horizon:>2d}d ({h_key}): test rows = {n_rows:,d} | crossing rate = {rate:>6.2%}")
    print("=" * 98)
    print(f"Results successfully written to: {RESULTS_PATH}")
    print("=" * 98)


if __name__ == "__main__":
    main()
