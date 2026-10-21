from dataclasses import dataclass
import os
from typing import List, Sequence, Union

import numpy as np
import pandas as pd

from src.config import (
    FEATURE_COLUMNS,
    FEATURES_PATH,
    HORIZONS_DAYS,
    TEST_FRACTION,
)


@dataclass
class Split:
    train: pd.DataFrame
    test: pd.DataFrame
    test_start: pd.Timestamp
    purged_rows: int


def prepare_horizon_table(features: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    target_col = f"target_{horizon_days}d"
    if target_col not in features.columns:
        raise KeyError(f"Target column '{target_col}' not found in features table.")

    # Drop rows with NaN target (such as the latest N calendar days reserved for prediction)
    table = features.dropna(subset=[target_col]).copy()
    table["target"] = table[target_col].astype("float64")

    # Keep only asset metadata, feature columns, and the single target column
    keep_cols = ["asset", "asset_class", "date"] + FEATURE_COLUMNS + ["target"]
    table = table[keep_cols].sort_values(by=["date", "asset"]).reset_index(drop=True)
    return table


def find_test_start(
    dates: Union[pd.Series, Sequence[pd.Timestamp], np.ndarray],
    test_fraction: float = TEST_FRACTION,
) -> pd.Timestamp:
    if not (0 < test_fraction < 1):
        raise ValueError(f"test_fraction must be between 0 and 1, got {test_fraction}")

    unique_dates = pd.Series(pd.to_datetime(dates)).drop_duplicates().sort_values().reset_index(drop=True)
    if len(unique_dates) == 0:
        raise ValueError("Cannot determine test start date on an empty sequence of dates.")

    # Determine split point by unique calendar dates across all assets
    n_test = max(1, int(round(len(unique_dates) * test_fraction)))
    split_idx = len(unique_dates) - n_test
    return pd.Timestamp(unique_dates.iloc[split_idx])


def split_by_date(
    table: pd.DataFrame,
    horizon_days: int,
    test_fraction: float = TEST_FRACTION,
) -> Split:
    if table.empty:
        raise ValueError("Cannot split an empty table.")

    table = table.copy()
    table["date"] = pd.to_datetime(table["date"])
    test_start = find_test_start(table["date"], test_fraction=test_fraction)

    # Test set includes all rows at or after the global test start date
    test_mask = table["date"] >= test_start
    test = table[test_mask].sort_values(by=["date", "asset"]).reset_index(drop=True)

    # Clean training rows must not leak forward into the test period via the forward return target
    train_mask = table["date"] + pd.Timedelta(days=horizon_days) < test_start
    train = table[train_mask].sort_values(by=["date", "asset"]).reset_index(drop=True)

    # Purged rows fall in [test_start - horizon_days, test_start)
    purged_mask = (table["date"] < test_start) & (~train_mask)
    purged_rows = int(purged_mask.sum())

    # Ensure all assets have samples in both train and test sets
    all_assets = set(table["asset"].unique())
    train_assets = set(train["asset"].unique())
    test_assets = set(test["asset"].unique())

    missing_train = all_assets - train_assets
    if missing_train:
        raise ValueError(f"Asset(s) {missing_train} have no observations in the training set.")

    missing_test = all_assets - test_assets
    if missing_test:
        raise ValueError(f"Asset(s) {missing_test} have no observations in the test set.")

    return Split(
        train=train,
        test=test,
        test_start=test_start,
        purged_rows=purged_rows,
    )


def main() -> None:
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"Feature table not found at {FEATURES_PATH}. Run 'python -m src.features' first.")

    features = pd.read_parquet(FEATURES_PATH)
    print(f"Loaded features from {FEATURES_PATH} ({len(features):,} rows).\n")

    print("=" * 70)
    print("CHRONOLOGICAL TIME-BASED SPLIT SUMMARY")
    print("=" * 70)

    for horizon in HORIZONS_DAYS:
        horizon_table = prepare_horizon_table(features, horizon_days=horizon)
        split = split_by_date(horizon_table, horizon_days=horizon, test_fraction=TEST_FRACTION)

        last_train_date = split.train["date"].max()
        first_test_date = split.test["date"].min()

        print(f"\n--- Horizon: {horizon} Days ---")
        print(f"  Test start date:  {split.test_start.strftime('%Y-%m-%d')}")
        print(f"  Last train date:  {last_train_date.strftime('%Y-%m-%d')}")
        print(f"  First test date:  {first_test_date.strftime('%Y-%m-%d')}")
        print(f"  Purged rows:      {split.purged_rows:,}")
        print(f"  Train row count:  {len(split.train):,}")
        print(f"  Test row count:   {len(split.test):,}")
        print(f"  Total rows:       {len(split.train) + len(split.test) + split.purged_rows:,}")

        print("  Rows per asset (Train / Test / Purged):")
        assets = sorted(horizon_table["asset"].unique())
        for asset in assets:
            n_tr = len(split.train[split.train["asset"] == asset])
            n_te = len(split.test[split.test["asset"] == asset])
            n_tot = len(horizon_table[horizon_table["asset"] == asset])
            n_purged = n_tot - n_tr - n_te
            asset_class = horizon_table[horizon_table["asset"] == asset]["asset_class"].iloc[0]
            print(f"    {asset:10s} ({asset_class:6s}): {n_tr:4d} train | {n_te:4d} test | {n_purged:3d} purged")

    print("\n" + "=" * 70)
    print("Split verification complete.")


if __name__ == "__main__":
    main()
