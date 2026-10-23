import json
import os
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.baselines import historical_quantiles, static_quantile
from src.config import (
    FEATURE_COLUMNS,
    TEST_FRACTION,
    VOLATILITY_COL,
    VOLUME_COL,
)
from src.metrics import crossing_rate
from src.predict import (
    max_loan,
    predict_haircuts,
    predict_multiquantile,
    predict_quantile_return,
    to_haircut,
)
from src.split import prepare_horizon_table, split_by_date


def load_results(path: Union[str, os.PathLike]) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def calculator_result(
    bundle: Dict[str, Any],
    features: pd.DataFrame,
    asset: str,
    collateral_value: float,
    bundles_by_confidence: Optional[Mapping[Union[float, str], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    if collateral_value < 0:
        raise ValueError(f"Collateral value cannot be negative, got {collateral_value}")

    haircuts_df = predict_haircuts(bundle, features, collateral_value=collateral_value)
    asset_rows = haircuts_df[haircuts_df["asset"] == asset]
    if asset_rows.empty:
        available = sorted(haircuts_df["asset"].unique())
        raise ValueError(f"Unknown asset '{asset}'. Available assets: {available}")

    row = asset_rows.iloc[0]
    predicted_return = float(row["predicted_return"])
    haircut = float(row["haircut"])
    loan_val = float(row["max_loan"])

    if bundles_by_confidence is not None and len(bundles_by_confidence) >= 2:
        clean = features[features["asset"] == asset].dropna(subset=FEATURE_COLUMNS).copy()
        if not clean.empty:
            clean["date"] = pd.to_datetime(clean["date"])
            latest_row = clean.sort_values(by="date").iloc[[-1]]
            rearranged = predict_multiquantile(bundles_by_confidence, latest_row, rearrange=True)
            target_conf = float(bundle.get("confidence", 0.95))
            for k, preds in rearranged.items():
                if abs(float(k) - target_conf) < 1e-4:
                    predicted_return = float(preds[0])
                    haircut = float(to_haircut(predicted_return))
                    loan_val = float(max_loan(collateral_value, haircut))
                    break

    return {
        "asset": str(row["asset"]),
        "date": pd.to_datetime(row["date"]),
        "volatility": float(row[VOLATILITY_COL]),
        "avg_dollar_volume": float(row[VOLUME_COL]),
        "predicted_return": predicted_return,
        "haircut": haircut,
        "max_loan": loan_val,
        "collateral_value": float(collateral_value),
    }


def latest_crossing(
    bundles_by_confidence: Mapping[Union[float, str], Dict[str, Any]],
    features: pd.DataFrame,
    asset: str,
) -> Optional[Dict[float, float]]:
    if "asset" not in features.columns:
        raise ValueError("Features DataFrame missing 'asset' column.")
    if asset not in features["asset"].values:
        raise ValueError(f"Unknown asset '{asset}'. Available assets: {sorted(features['asset'].unique())}")

    asset_features = features[features["asset"] == asset].dropna(subset=FEATURE_COLUMNS)
    if asset_features.empty:
        raise ValueError(f"No complete feature rows for asset '{asset}'.")

    latest_row = asset_features.sort_values(by="date").iloc[[-1]]

    predictions: Dict[float, float] = {}
    metric_input: Dict[float, np.ndarray] = {}
    for conf, bundle in sorted(bundles_by_confidence.items(), key=lambda item: float(item[0])):
        c_float = float(conf)
        pred = float(predict_quantile_return(bundle, latest_row)[0])
        predictions[c_float] = pred
        metric_input[c_float] = np.array([pred])

    rate = crossing_rate(metric_input)
    if rate > 0.0:
        return predictions
    return None



def baseline_haircuts(
    features: pd.DataFrame,
    horizon_days: int,
    confidence: float,
    asset: str,
    train_end_date: Any,
) -> Dict[str, float]:
    horizon_table = prepare_horizon_table(features, horizon_days=horizon_days)
    cutoff = pd.to_datetime(train_end_date)
    train_rows = horizon_table[pd.to_datetime(horizon_table["date"]) <= cutoff]
    if train_rows.empty:
        raise ValueError("No training rows found up to train_end_date.")

    q_stat = static_quantile(train_rows, confidence)
    q_hist = historical_quantiles(train_rows, confidence)

    if asset not in q_hist:
        raise ValueError(f"Asset '{asset}' not found in training set. Available assets: {sorted(q_hist.keys())}")

    return {
        "static": float(to_haircut(q_stat)),
        "historical": float(to_haircut(q_hist[asset])),
    }


def risk_band(
    bundles: Optional[Union[Dict[str, Any], Mapping[Union[float, str], Dict[str, Any]]]] = None,
    features: Optional[pd.DataFrame] = None,
    prices: Optional[pd.DataFrame] = None,
    asset: Optional[str] = None,
    horizon_days: Optional[int] = None,
    lookback_days: int = 365,
    bundle: Optional[Dict[str, Any]] = None,
    bundles_by_confidence: Optional[Mapping[Union[float, str], Dict[str, Any]]] = None,
) -> pd.DataFrame:
    target_bundles = bundles
    if target_bundles is None:
        target_bundles = bundles_by_confidence if bundles_by_confidence is not None else bundle

    if target_bundles is None:
        raise ValueError("bundles or bundle must be provided.")
    if features is None or prices is None or asset is None or horizon_days is None:
        raise ValueError("features, prices, asset, and horizon_days are required.")

    if "asset" in features.columns:
        asset_features = features[features["asset"] == asset].copy()
    else:
        asset_features = features.copy()

    if asset_features.empty:
        raise ValueError(f"Asset '{asset}' not found in features.")

    # Drop rows where features are incomplete
    asset_features = asset_features.dropna(subset=FEATURE_COLUMNS).copy()
    if asset_features.empty:
        raise ValueError(f"No rows with complete feature values for asset '{asset}'.")

    if "asset" in prices.columns:
        asset_prices = prices[prices["asset"] == asset].copy()
    else:
        asset_prices = prices.copy()

    if asset_prices.empty:
        raise ValueError(f"Asset '{asset}' not found in prices.")

    if "close" not in asset_prices.columns:
        raise KeyError("Prices DataFrame must contain 'close' column.")
    if "date" not in asset_prices.columns:
        raise KeyError("Prices DataFrame must contain 'date' column.")

    asset_features["date"] = pd.to_datetime(asset_features["date"])
    asset_prices["date"] = pd.to_datetime(asset_prices["date"])

    merged = pd.merge(
        asset_prices[["date", "close"]],
        asset_features[["date"] + FEATURE_COLUMNS],
        on="date",
        how="inner",
    ).sort_values("date").reset_index(drop=True)

    result_cols = ["date", "close", "floor_date", "floor_price", "floor_90", "floor_95", "floor_99"]
    if merged.empty:
        return pd.DataFrame(columns=result_cols)

    if lookback_days is not None:
        if lookback_days <= 0:
            raise ValueError(f"lookback_days must be positive, got {lookback_days}")
        max_date = merged["date"].max()
        cutoff_date = max_date - pd.Timedelta(days=lookback_days)
        merged = merged[merged["date"] >= cutoff_date].reset_index(drop=True)

    merged["floor_date"] = merged["date"] + pd.Timedelta(days=horizon_days)

    if isinstance(target_bundles, dict) and "model" in target_bundles:
        q = predict_quantile_return(target_bundles, merged[FEATURE_COLUMNS])
        merged["floor_price"] = merged["close"] * (1.0 + q)
        merged["floor_90"] = merged["floor_price"]
        merged["floor_95"] = merged["floor_price"]
        merged["floor_99"] = merged["floor_price"]
    else:
        q_preds = predict_multiquantile(target_bundles, merged[FEATURE_COLUMNS], rearrange=True)
        q_map = {round(float(k), 2): v for k, v in q_preds.items()}
        merged["floor_90"] = merged["close"] * (1.0 + q_map[0.90]) if 0.90 in q_map else merged["close"]
        merged["floor_95"] = merged["close"] * (1.0 + q_map[0.95]) if 0.95 in q_map else merged["close"]
        merged["floor_99"] = merged["close"] * (1.0 + q_map[0.99]) if 0.99 in q_map else merged["close"]
        merged["floor_price"] = merged["floor_95"]

    return merged[result_cols].sort_values("date").reset_index(drop=True)


def training_cutoff(bundle: Dict[str, Any]) -> pd.Timestamp:
    val = bundle.get("train_end_date")
    if val is None:
        raise KeyError("Bundle missing 'train_end_date'.")
    return pd.to_datetime(val)


def newer_data_days(bundle: Dict[str, Any], features: pd.DataFrame) -> int:
    val = bundle.get("train_end_date")
    if val is None:
        raise KeyError("Bundle missing 'train_end_date'.")
    if features.empty or "date" not in features.columns:
        return 0
    cutoff = pd.to_datetime(val).floor("D")
    latest = pd.to_datetime(features["date"].max()).floor("D")
    return int((latest - cutoff).days)


def test_period_start(features: pd.DataFrame, horizon_days: int) -> pd.Timestamp:
    horizon_table = prepare_horizon_table(features, horizon_days=horizon_days)
    split = split_by_date(horizon_table, horizon_days=horizon_days, test_fraction=TEST_FRACTION)
    return split.test_start


test_period_start.__test__ = False


def evaluation_tables(
    results: Dict[str, Any],
    config_key: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    configs = results.get("configs", {})
    if config_key not in configs:
        available = list(configs.keys())
        raise KeyError(f"Config key '{config_key}' not found in results. Available keys: {available}")

    config_data = configs[config_key]

    summary_rows = []
    methods_dict = config_data.get("methods", {})
    method_order = ["model", "historical", "static"]
    ordered_methods = [m for m in method_order if m in methods_dict] + [
        m for m in methods_dict if m not in method_order
    ]

    for method_name in ordered_methods:
        method_data = methods_dict[method_name]
        ov = method_data.get("overall", {})
        summary_rows.append({
            "method": method_name,
            "n_rows": int(ov.get("n_rows", 0)),
            "pinball_loss": float(ov.get("pinball_loss", 0.0)),
            "breach_rate": float(ov.get("breach_rate", 0.0)),
            "mean_haircut": float(ov.get("mean_haircut", 0.0)),
        })
    summary_df = pd.DataFrame(summary_rows)

    per_asset_rows = []
    for method_name in ordered_methods:
        method_data = methods_dict[method_name]
        per_asset = method_data.get("per_asset", {})
        for asset in sorted(per_asset.keys()):
            score = per_asset[asset]
            per_asset_rows.append({
                "asset": str(asset),
                "method": str(method_name),
                "n_rows": int(score.get("n_rows", 0)),
                "pinball_loss": float(score.get("pinball_loss", 0.0)),
                "breach_rate": float(score.get("breach_rate", 0.0)),
                "mean_haircut": float(score.get("mean_haircut", 0.0)),
            })
    per_asset_df = pd.DataFrame(per_asset_rows)

    comparison_dict = config_data.get("comparison", {})
    info_dict = {
        "generated_at": results.get("generated_at"),
        "horizon_days": config_data.get("horizon_days"),
        "confidence": config_data.get("confidence"),
        "alpha": config_data.get("alpha"),
        "test_start": config_data.get("test_start"),
        "train_end_date": config_data.get("train_end_date"),
        "n_train_rows": config_data.get("n_train_rows"),
        "n_test_rows": config_data.get("n_test_rows"),
    }

    return summary_df, per_asset_df, comparison_dict, info_dict


def monthly_breach_rates(
    bundle: Dict[str, Any],
    features: pd.DataFrame,
) -> pd.DataFrame:
    columns = ["month", "n_rows", "n_breaches", "breach_rate", "target_alpha"]
    if features.empty:
        return pd.DataFrame(columns=columns)

    horizon_days = int(bundle["horizon_days"])
    confidence = float(bundle["confidence"])
    target_alpha = float(1.0 - confidence)
    target_col = f"target_{horizon_days}d"

    horizon_table = prepare_horizon_table(features, horizon_days=horizon_days)
    split = split_by_date(horizon_table, horizon_days=horizon_days, test_fraction=TEST_FRACTION)
    test_rows = split.test.copy()

    if test_rows.empty:
        return pd.DataFrame(columns=columns)

    q = predict_quantile_return(bundle, test_rows)
    test_rows["q"] = q
    target_series = test_rows[target_col] if target_col in test_rows.columns else test_rows["target"]
    # Strict inequality: breach occurs when realized return is worse than predicted quantile
    test_rows["is_breach"] = target_series < test_rows["q"]
    test_rows["month"] = pd.to_datetime(test_rows["date"]).dt.to_period("M").astype(str)

    monthly_stats = []
    for month, grp in test_rows.groupby("month", sort=True):
        n_rows = int(len(grp))
        n_breaches = int(grp["is_breach"].sum())
        rate = float(n_breaches / n_rows) if n_rows > 0 else 0.0
        monthly_stats.append({
            "month": str(month),
            "n_rows": n_rows,
            "n_breaches": n_breaches,
            "breach_rate": rate,
            "target_alpha": target_alpha,
        })

    return pd.DataFrame(monthly_stats)
