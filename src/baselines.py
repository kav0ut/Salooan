import os
from typing import Dict

import numpy as np
import pandas as pd

from src.config import (
    CONFIDENCE_LEVELS,
    FEATURES_PATH,
    HORIZONS_DAYS,
    TEST_FRACTION,
)
from src.split import prepare_horizon_table, split_by_date


def static_quantile(train: pd.DataFrame, confidence: float) -> float:
    alpha = 1.0 - confidence
    return float(np.quantile(train["target"].to_numpy(), alpha))


def historical_quantiles(train: pd.DataFrame, confidence: float) -> Dict[str, float]:
    alpha = 1.0 - confidence
    quantiles = {}
    for asset in sorted(train["asset"].unique()):
        asset_targets = train.loc[train["asset"] == asset, "target"].to_numpy()
        quantiles[asset] = float(np.quantile(asset_targets, alpha))
    return quantiles


def predict_static(test: pd.DataFrame, q: float) -> np.ndarray:
    return np.full(len(test), q, dtype=np.float64)


def predict_historical(test: pd.DataFrame, quantiles: Dict[str, float]) -> np.ndarray:
    test_assets = set(test["asset"].unique())
    missing = test_assets - set(quantiles.keys())
    if missing:
        raise ValueError(f"Asset(s) {missing} missing from fitted historical quantiles.")
    # Map each test row's asset to its fitted quantile, preserving test order
    return test["asset"].map(quantiles).to_numpy(dtype=np.float64)


def baseline_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    confidence: float,
) -> Dict[str, np.ndarray]:
    q_stat = static_quantile(train, confidence)
    q_hist = historical_quantiles(train, confidence)
    return {
        "static": predict_static(test, q_stat),
        "historical": predict_historical(test, q_hist),
    }


def main() -> None:
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"Feature table not found at {FEATURES_PATH}. Run 'python -m src.features' first.")

    features = pd.read_parquet(FEATURES_PATH)

    print("=" * 88)
    print("BASELINE QUANTILE PREDICTIONS & TEST BREACH RATES (DIAGNOSTIC)")
    print("=" * 88)

    for horizon in HORIZONS_DAYS:
        horizon_table = prepare_horizon_table(features, horizon_days=horizon)
        split = split_by_date(horizon_table, horizon_days=horizon, test_fraction=TEST_FRACTION)
        assets = sorted(split.test["asset"].unique())

        print(f"\n====================== Horizon: {horizon} Days ======================")

        for conf in CONFIDENCE_LEVELS:
            alpha = 1.0 - conf
            preds = baseline_predictions(split.train, split.test, conf)
            q_stat = static_quantile(split.train, conf)
            q_hist_dict = historical_quantiles(split.train, conf)

            print(f"\nConfidence: {int(conf * 100)}% (Target alpha: {alpha:.2f}) | Pooled Static Quantile: {q_stat:+.4f}")
            print(f"{'Asset':<10} {'Asset Class':<12} {'Hist Quantile':<15} {'Static Breach':<15} {'Hist Breach':<15}")
            print("-" * 67)

            test_targets = split.test["target"].to_numpy()
            static_preds = preds["static"]
            hist_preds = preds["historical"]

            for asset in assets:
                mask = (split.test["asset"] == asset).to_numpy()
                asset_targets = test_targets[mask]
                asset_class = split.test.loc[mask, "asset_class"].iloc[0]

                # Diagnostic breach rate: share of realized targets worse (more negative) than prediction
                static_breach = np.mean(asset_targets < static_preds[mask])
                hist_breach = np.mean(asset_targets < hist_preds[mask])
                hist_q = q_hist_dict[asset]

                print(f"{asset:<10} {asset_class:<12} {hist_q:>+13.4f}   {static_breach:>13.2%}   {hist_breach:>13.2%}")

            overall_static_breach = np.mean(test_targets < static_preds)
            overall_hist_breach = np.mean(test_targets < hist_preds)
            print("-" * 67)
            print(f"{'OVERALL':<10} {'all':<12} {'N/A':>13}   {overall_static_breach:>13.2%}   {overall_hist_breach:>13.2%}")

    print("\n" + "=" * 88)
    print("Baseline evaluation complete.")


if __name__ == "__main__":
    main()
