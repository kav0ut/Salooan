import os
from typing import Any, Dict, Mapping, Sequence, Union

import numpy as np
import pandas as pd

from src.baselines import historical_quantiles, static_quantile
from src.config import (
    DEFAULT_CONFIDENCE,
    DEFAULT_HORIZON_DAYS,
    DEMO_COLLATERAL_VALUE,
    FEATURE_COLUMNS,
    FEATURES_PATH,
    TEST_FRACTION,
    VOLATILITY_COL,
    VOLUME_COL,
)
from src.metrics import monotone_rearrangement
from src.split import prepare_horizon_table, split_by_date
from src.train import load_bundle


def to_haircut(q: Union[float, np.ndarray, Sequence[float]]) -> Union[float, np.ndarray]:
    if np.isscalar(q):
        # Quantile is negative for a drop; clip haircut to [0, 1]
        return float(np.clip(-float(q), 0.0, 1.0))
    return np.clip(-np.asarray(q, dtype=np.float64), 0.0, 1.0)


def max_loan(
    collateral_value: float,
    haircut: Union[float, np.ndarray, Sequence[float]],
) -> Union[float, np.ndarray]:
    if np.isscalar(haircut):
        return float(collateral_value * (1.0 - float(haircut)))
    return collateral_value * (1.0 - np.asarray(haircut, dtype=np.float64))


def latest_features(features: pd.DataFrame) -> pd.DataFrame:
    required_cols = ["asset", "date"] + FEATURE_COLUMNS
    missing = [col for col in required_cols if col not in features.columns]
    if missing:
        raise ValueError(f"Features DataFrame missing required columns: {missing}")

    # Exclude rows where feature values are incomplete; target NaN is permitted
    clean = features.dropna(subset=FEATURE_COLUMNS).copy()
    if clean.empty:
        raise ValueError("No rows with complete feature values found.")

    clean["date"] = pd.to_datetime(clean["date"])
    # Pick the latest available observation date per asset
    latest = clean.sort_values(by="date").drop_duplicates(subset=["asset"], keep="last")
    return latest.sort_values(by="asset").reset_index(drop=True)


def predict_quantile_return(bundle: Dict[str, Any], rows: pd.DataFrame) -> np.ndarray:
    bundle_features = bundle.get("feature_columns")
    if bundle_features != FEATURE_COLUMNS:
        raise ValueError(
            f"Bundle feature columns {bundle_features} do not match current FEATURE_COLUMNS {FEATURE_COLUMNS}."
        )

    missing = [col for col in FEATURE_COLUMNS if col not in rows.columns]
    if missing:
        raise ValueError(f"Rows are missing required feature columns: {missing}")

    if rows[FEATURE_COLUMNS].isna().any().any():
        raise ValueError("Input rows contain NaN values in required feature columns.")

    model = bundle["model"]
    preds = model.predict(rows[FEATURE_COLUMNS])
    return np.asarray(preds, dtype=np.float64)


def predict_multiquantile(
    bundles_by_confidence: Mapping[Union[float, str], Dict[str, Any]],
    X: pd.DataFrame,
    rearrange: bool = True,
) -> Dict[Union[float, str], np.ndarray]:
    predictions: Dict[Union[float, str], np.ndarray] = {}
    for conf, bundle in bundles_by_confidence.items():
        predictions[conf] = predict_quantile_return(bundle, X)

    if rearrange and len(predictions) >= 2:
        return monotone_rearrangement(predictions)

    return predictions


def predict_haircuts(
    bundle: Dict[str, Any],
    features: pd.DataFrame,
    collateral_value: float,
) -> pd.DataFrame:
    latest = latest_features(features)
    predicted_returns = predict_quantile_return(bundle, latest)
    haircuts = to_haircut(predicted_returns)
    loans = max_loan(collateral_value, haircuts)

    output = latest[["asset", "date"] + FEATURE_COLUMNS].copy()
    output["predicted_return"] = predicted_returns
    output["haircut"] = haircuts
    output["max_loan"] = loans
    return output


def main() -> None:
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"Feature table not found at {FEATURES_PATH}. Run 'python -m src.features' first.")

    bundle = load_bundle(DEFAULT_HORIZON_DAYS, DEFAULT_CONFIDENCE)
    features = pd.read_parquet(FEATURES_PATH)

    # Dynamic haircut predictions on the latest feature row per asset
    haircuts_df = predict_haircuts(bundle, features, collateral_value=DEMO_COLLATERAL_VALUE)

    # Fit baselines on the same training split for comparison
    horizon_table = prepare_horizon_table(features, horizon_days=DEFAULT_HORIZON_DAYS)
    split = split_by_date(horizon_table, horizon_days=DEFAULT_HORIZON_DAYS, test_fraction=TEST_FRACTION)

    q_stat = static_quantile(split.train, DEFAULT_CONFIDENCE)
    q_hist = historical_quantiles(split.train, DEFAULT_CONFIDENCE)

    static_hc = to_haircut(q_stat)
    hist_hc_map = {asset: to_haircut(q) for asset, q in q_hist.items()}

    haircuts_df["static_haircut"] = static_hc
    haircuts_df["hist_haircut"] = haircuts_df["asset"].map(hist_hc_map)
    haircuts_df["static_max_loan"] = max_loan(DEMO_COLLATERAL_VALUE, static_hc)
    haircuts_df["hist_max_loan"] = max_loan(DEMO_COLLATERAL_VALUE, haircuts_df["hist_haircut"])

    print("=" * 105)
    print("DYNAMIC COLLATERAL HAIRCUT & MAX LOAN ESTIMATION")
    print(f"Horizon: {DEFAULT_HORIZON_DAYS} Days | Confidence: {int(DEFAULT_CONFIDENCE * 100)}% | Demo Collateral: ${DEMO_COLLATERAL_VALUE:,.2f}")
    print("=" * 105)
    print(f"{'Asset':<10} {'As-of Date':<12} {'Vol(30d)':<10} {'AvgDolVol(30d)':<16} {'Dyn HC':<10} {'Hist HC':<10} {'Stat HC':<10} {'Dyn Loan':<12} {'Hist Loan':<12}")
    print("-" * 105)

    for _, row in haircuts_df.iterrows():
        asset = row["asset"]
        date_str = pd.to_datetime(row["date"]).strftime("%Y-%m-%d")
        vol = row[VOLATILITY_COL]
        dvol = row[VOLUME_COL]
        dyn_hc = row["haircut"]
        hist_hc = row["hist_haircut"]
        stat_hc = row["static_haircut"]
        dyn_loan = row["max_loan"]
        hist_loan = row["hist_max_loan"]

        print(f"{asset:<10} {date_str:<12} {vol:>8.2%}   ${dvol:>13,.0f}   {dyn_hc:>7.2%}    {hist_hc:>7.2%}    {stat_hc:>7.2%}    ${dyn_loan:>9,.2f}   ${hist_loan:>9,.2f}")

    print("=" * 105)


if __name__ == "__main__":
    main()
