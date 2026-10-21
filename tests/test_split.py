import numpy as np
import pandas as pd
import pytest

from src.config import FEATURE_COLUMNS
from src.split import (
    Split,
    find_test_start,
    prepare_horizon_table,
    split_by_date,
)


@pytest.fixture
def synthetic_features() -> pd.DataFrame:
    # Daily calendar frequency for crypto (60 days)
    crypto_dates = pd.date_range("2023-01-01", periods=60, freq="D")
    n_crypto = len(crypto_dates)
    crypto_df = pd.DataFrame(
        {
            "asset": ["BTC"] * n_crypto,
            "asset_class": ["crypto"] * n_crypto,
            "date": crypto_dates,
            FEATURE_COLUMNS[0]: np.linspace(0.01, 0.05, n_crypto),
            FEATURE_COLUMNS[1]: np.linspace(1e8, 5e8, n_crypto),
            "target_7d": np.random.RandomState(1).randn(n_crypto) * 0.05,
            "target_14d": np.random.RandomState(2).randn(n_crypto) * 0.08,
        }
    )
    # Put NaNs at the end as in real processed data
    crypto_df.loc[n_crypto - 7 :, "target_7d"] = np.nan
    crypto_df.loc[n_crypto - 14 :, "target_14d"] = np.nan

    # Business day frequency for equities across the same range
    equity_dates = pd.bdate_range("2023-01-01", "2023-03-01")
    n_equity = len(equity_dates)
    equity_df = pd.DataFrame(
        {
            "asset": ["SPY"] * n_equity,
            "asset_class": ["equity"] * n_equity,
            "date": equity_dates,
            FEATURE_COLUMNS[0]: np.linspace(0.005, 0.02, n_equity),
            FEATURE_COLUMNS[1]: np.linspace(2e9, 4e9, n_equity),
            "target_7d": np.random.RandomState(3).randn(n_equity) * 0.02,
            "target_14d": np.random.RandomState(4).randn(n_equity) * 0.04,
        }
    )
    equity_df.loc[n_equity - 5 :, "target_7d"] = np.nan
    equity_df.loc[n_equity - 10 :, "target_14d"] = np.nan

    return pd.concat([crypto_df, equity_df], ignore_index=True)


def test_test_and_train_date_boundaries(synthetic_features):
    horizon = 7
    table = prepare_horizon_table(synthetic_features, horizon_days=horizon)
    split = split_by_date(table, horizon_days=horizon, test_fraction=0.2)

    # Every test row must have date >= test_start
    assert (split.test["date"] >= split.test_start).all()

    # Every train row must have date + N days < test_start
    assert (split.train["date"] + pd.Timedelta(days=horizon) < split.test_start).all()


def test_no_forward_target_leakage(synthetic_features):
    horizon = 7
    table = prepare_horizon_table(synthetic_features, horizon_days=horizon)
    split = split_by_date(table, horizon_days=horizon, test_fraction=0.2)

    # For every training row, the target realization date (date + N) must strictly precede test_start
    target_dates = split.train["date"] + pd.Timedelta(days=horizon)
    assert (target_dates < split.test_start).all()


def test_disjoint_sets_and_total_row_conservation(synthetic_features):
    horizon = 7
    table = prepare_horizon_table(synthetic_features, horizon_days=horizon)
    split = split_by_date(table, horizon_days=horizon, test_fraction=0.2)

    # Train and test subsets must not share any (asset, date) key
    train_keys = set(zip(split.train["asset"], split.train["date"]))
    test_keys = set(zip(split.test["asset"], split.test["date"]))
    assert train_keys.isdisjoint(test_keys)

    # Sum of train, test, and purged rows must equal total table rows
    assert len(split.train) + len(split.test) + split.purged_rows == len(table)


def test_purged_rows_are_in_gap_interval(synthetic_features):
    horizon = 7
    table = prepare_horizon_table(synthetic_features, horizon_days=horizon)
    split = split_by_date(table, horizon_days=horizon, test_fraction=0.2)

    gap_start = split.test_start - pd.Timedelta(days=horizon)
    gap_end = split.test_start

    # Identify rows in the table falling inside [test_start - N days, test_start)
    in_gap = table[(table["date"] >= gap_start) & (table["date"] < gap_end)]
    assert len(in_gap) == split.purged_rows

    # None of the gap rows may appear in train or test
    gap_keys = set(zip(in_gap["asset"], in_gap["date"]))
    train_keys = set(zip(split.train["asset"], split.train["date"]))
    test_keys = set(zip(split.test["asset"], split.test["date"]))

    assert gap_keys.isdisjoint(train_keys)
    assert gap_keys.isdisjoint(test_keys)


def test_both_assets_share_single_test_start(synthetic_features):
    horizon = 7
    table = prepare_horizon_table(synthetic_features, horizon_days=horizon)
    split = split_by_date(table, horizon_days=horizon, test_fraction=0.2)

    # Test start date is a single global date
    test_btc = split.test[split.test["asset"] == "BTC"]
    test_spy = split.test[split.test["asset"] == "SPY"]

    assert test_btc["date"].min() >= split.test_start
    assert test_spy["date"].min() >= split.test_start


def test_chronological_ordering_no_shuffle(synthetic_features):
    horizon = 7
    table = prepare_horizon_table(synthetic_features, horizon_days=horizon)
    split = split_by_date(table, horizon_days=horizon, test_fraction=0.2)

    # Strict chronological separation: max train date < min test date
    assert split.train["date"].max() < split.test["date"].min()


def test_nan_target_rows_are_dropped(synthetic_features):
    horizon = 7
    table = prepare_horizon_table(synthetic_features, horizon_days=horizon)

    # Table must not contain any NaN targets
    assert table["target"].notna().all()

    split = split_by_date(table, horizon_days=horizon, test_fraction=0.2)
    assert split.train["target"].notna().all()
    assert split.test["target"].notna().all()


def test_test_fraction_of_unique_dates(synthetic_features):
    horizon = 7
    table = prepare_horizon_table(synthetic_features, horizon_days=horizon)
    test_fraction = 0.25
    split = split_by_date(table, horizon_days=horizon, test_fraction=test_fraction)

    unique_dates = table["date"].drop_duplicates()
    unique_test_dates = split.test["date"].drop_duplicates()

    actual_fraction = len(unique_test_dates) / len(unique_dates)
    assert actual_fraction == pytest.approx(test_fraction, abs=0.03)


def test_different_horizon_changes_purge_gap(synthetic_features):
    table_7 = prepare_horizon_table(synthetic_features, horizon_days=7)
    split_7 = split_by_date(table_7, horizon_days=7, test_fraction=0.2)

    table_14 = prepare_horizon_table(synthetic_features, horizon_days=14)
    split_14 = split_by_date(table_14, horizon_days=14, test_fraction=0.2)

    # Longer horizon creates a wider purge gap and more purged rows
    assert split_14.purged_rows > split_7.purged_rows
    assert split_14.train["date"].max() <= split_7.train["date"].max()


def test_asset_missing_from_train_or_test_raises_value_error():
    # Asset with rows only in recent dates (will end up purely in test)
    table = pd.DataFrame(
        {
            "asset": ["LATE_ONLY"] * 10,
            "asset_class": ["crypto"] * 10,
            "date": pd.date_range("2023-01-20", periods=10, freq="D"),
            FEATURE_COLUMNS[0]: [0.02] * 10,
            FEATURE_COLUMNS[1]: [1e8] * 10,
            "target": [0.01] * 10,
        }
    )

    with pytest.raises(ValueError, match="no observations in the training set"):
        split_by_date(table, horizon_days=7, test_fraction=0.5)


def test_input_dataframe_not_mutated(synthetic_features):
    horizon = 7
    table = prepare_horizon_table(synthetic_features, horizon_days=horizon)
    table_copy = table.copy(deep=True)

    _ = split_by_date(table, horizon_days=horizon, test_fraction=0.2)

    pd.testing.assert_frame_equal(table, table_copy)
