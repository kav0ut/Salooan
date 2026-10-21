import math
import os
from typing import List

import numpy as np
import pandas as pd
import pytest

from src.config import (
    FEATURE_COLUMNS,
    HORIZONS_DAYS,
    VOLATILITY_COL,
    VOLATILITY_WINDOW,
    VOLUME_COL,
    VOLUME_WINDOW,
)
from src.features import (
    add_dollar_volume_avg,
    add_volatility,
    build_asset_features,
    build_feature_table,
    load_processed,
    save_features,
)


def test_volatility_matches_hand_computed():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=3, freq="D"),
            "log_return": [0.01, 0.02, 0.06],
        }
    )

    result = add_volatility(df, window=3, col_name="vol_test")

    # First 2 rows must be NaN (window=3)
    assert pd.isna(result["vol_test"].iloc[0])
    assert pd.isna(result["vol_test"].iloc[1])

    # Third row must match sqrt(0.0007)
    expected_std = math.sqrt(0.0007)
    assert result["vol_test"].iloc[2] == pytest.approx(expected_std, rel=1e-6)

    # Input DataFrame must not be mutated
    assert "vol_test" not in df.columns


def test_rolling_features_match_naive_loop():
    rng = np.random.RandomState(42)
    n = 50
    w = 5

    log_returns = rng.randn(n) * 0.02
    closes = 100.0 * np.exp(np.cumsum(log_returns))
    volumes = rng.uniform(1000, 5000, size=n)

    df = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=n, freq="D"),
            "close": closes,
            "volume": volumes,
            "log_return": log_returns,
        }
    )

    # Naive loop for sample std (ddof=1)
    naive_vol = []
    for i in range(n):
        if i < w - 1:
            naive_vol.append(np.nan)
        else:
            window_slice = log_returns[i - w + 1 : i + 1]
            mean_val = sum(window_slice) / w
            var_val = sum((x - mean_val) ** 2 for x in window_slice) / (w - 1)
            naive_vol.append(math.sqrt(var_val))

    # Naive loop for dollar volume average
    naive_dollar_vol = []
    for i in range(n):
        if i < w - 1:
            naive_dollar_vol.append(np.nan)
        else:
            window_dv = [
                c * v
                for c, v in zip(
                    closes[i - w + 1 : i + 1], volumes[i - w + 1 : i + 1]
                )
            ]
            naive_dollar_vol.append(sum(window_dv) / w)

    result = build_asset_features(
        df,
        vol_window=w,
        vol_col="vol_test",
        volume_window=w,
        volume_col="volume_test",
    )

    np.testing.assert_allclose(
        result["vol_test"].values, naive_vol, equal_nan=True, rtol=1e-6
    )
    np.testing.assert_allclose(
        result["volume_test"].values, naive_dollar_vol, equal_nan=True, rtol=1e-6
    )


def test_first_w_minus_1_rows_are_nan():
    w = 7
    n = 15
    df = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=n, freq="D"),
            "close": [10.0 + i for i in range(n)],
            "volume": [100.0] * n,
            "log_return": [0.01] * n,
        }
    )

    res_vol = add_volatility(df, window=w, col_name="vol")
    res_vol_vol = add_dollar_volume_avg(df, window=w, col_name="dvol")

    # Indices 0 to w-2 must be NaN
    assert res_vol["vol"].iloc[: w - 1].isna().all()
    assert res_vol_vol["dvol"].iloc[: w - 1].isna().all()

    # Index w-1 must be non-NaN
    assert pd.notna(res_vol["vol"].iloc[w - 1])
    assert pd.notna(res_vol_vol["dvol"].iloc[w - 1])


def test_dollar_volume_uses_close_times_volume():
    # Constant volume, increasing price
    df = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=3, freq="D"),
            "close": [10.0, 20.0, 30.0],
            "volume": [100.0, 100.0, 100.0],
        }
    )

    # Dollar volume values: [1000, 2000, 3000]. Mean over 3 rows = 2000.
    result = add_dollar_volume_avg(df, window=3, col_name="dvol")

    assert result["dvol"].iloc[2] == pytest.approx(2000.0)
    # Ensure it's not the raw volume mean of 100.0
    assert result["dvol"].iloc[2] != pytest.approx(100.0)


def test_no_lookahead_truncation():
    n = 60
    rng = np.random.RandomState(123)
    df = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=n, freq="D"),
            "close": 50.0 + rng.randn(n).cumsum(),
            "volume": rng.uniform(500, 2000, size=n),
            "log_return": rng.randn(n) * 0.015,
        }
    )

    # Compute features on full series
    df_full = build_asset_features(df)

    # Test at various cutoff points after incomplete window period
    cutoffs = [30, 35, 42, 50, 59]
    for t in cutoffs:
        df_trunc = df.iloc[: t + 1].copy()
        df_trunc_feat = build_asset_features(df_trunc)

        # Feature at date t must match exactly
        assert df_trunc_feat[VOLATILITY_COL].iloc[t] == pytest.approx(
            df_full[VOLATILITY_COL].iloc[t], rel=1e-9
        )
        assert df_trunc_feat[VOLUME_COL].iloc[t] == pytest.approx(
            df_full[VOLUME_COL].iloc[t], rel=1e-9
        )


def test_future_perturbation_does_not_affect_past():
    n = 50
    t = 35
    rng = np.random.RandomState(999)

    df_orig = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=n, freq="D"),
            "close": 100.0 + rng.randn(n).cumsum(),
            "volume": rng.uniform(1000, 5000, size=n),
            "log_return": rng.randn(n) * 0.02,
        }
    )

    feat_orig = build_asset_features(df_orig)

    # Create perturbed DataFrame where future after t is drastically changed
    df_perturbed = df_orig.copy(deep=True)
    df_perturbed.loc[t + 1 :, "close"] = df_perturbed.loc[t + 1 :, "close"] * 1000.0
    df_perturbed.loc[t + 1 :, "volume"] = df_perturbed.loc[t + 1 :, "volume"] * 50.0
    df_perturbed.loc[t + 1 :, "log_return"] = 0.50

    feat_perturbed = build_asset_features(df_perturbed)

    # Features at and before t must be identical
    pd.testing.assert_series_equal(
        feat_orig.loc[:t, VOLATILITY_COL],
        feat_perturbed.loc[:t, VOLATILITY_COL],
    )
    pd.testing.assert_series_equal(
        feat_orig.loc[:t, VOLUME_COL],
        feat_perturbed.loc[:t, VOLUME_COL],
    )


def test_features_ignore_targets():
    n = 45
    df = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=n, freq="D"),
            "close": [100.0 + i for i in range(n)],
            "volume": [1000.0] * n,
            "log_return": [0.01] * n,
            "target_7d": [0.05] * n,
            "target_14d": [0.10] * n,
        }
    )

    feat_clean = build_asset_features(df)

    # Replace target columns with extreme garbage and NaNs
    df_corrupted = df.copy(deep=True)
    df_corrupted["target_7d"] = np.nan
    df_corrupted["target_14d"] = 999999.0

    feat_corrupted = build_asset_features(df_corrupted)

    pd.testing.assert_series_equal(
        feat_clean[VOLATILITY_COL],
        feat_corrupted[VOLATILITY_COL],
    )
    pd.testing.assert_series_equal(
        feat_clean[VOLUME_COL],
        feat_corrupted[VOLUME_COL],
    )


def test_asset_isolation(monkeypatch):
    n = 40
    df_a = pd.DataFrame(
        {
            "asset": ["ASSET_A"] * n,
            "asset_class": ["crypto"] * n,
            "date": pd.date_range("2023-01-01", periods=n, freq="D"),
            "close": [100.0 + i for i in range(n)],
            "volume": [1000.0] * n,
            "log_return": [0.01] * n,
            "target_7d": [0.05] * n,
            "target_14d": [0.10] * n,
        }
    )

    df_b = pd.DataFrame(
        {
            "asset": ["ASSET_B"] * n,
            "asset_class": ["equity"] * n,
            "date": pd.date_range("2023-01-01", periods=n, freq="D"),
            "close": [50.0 + i * 2 for i in range(n)],
            "volume": [500.0] * n,
            "log_return": [0.02] * n,
            "target_7d": [0.03] * n,
            "target_14d": [0.06] * n,
        }
    )

    data_store = {"ASSET_A": df_a, "ASSET_B": df_b}

    def fake_load(asset: str) -> pd.DataFrame:
        return data_store[asset].copy()

    monkeypatch.setattr("src.features.load_processed", fake_load)

    # Build combined table
    table_1 = build_feature_table(assets=["ASSET_A", "ASSET_B"])
    feat_a_1 = table_1[table_1["asset"] == "ASSET_A"].reset_index(drop=True)

    # Modify asset B drastically
    df_b_modified = df_b.copy(deep=True)
    df_b_modified["close"] = df_b_modified["close"] * 10000.0
    df_b_modified["volume"] = 9999999.0
    df_b_modified["log_return"] = -0.80
    data_store["ASSET_B"] = df_b_modified

    table_2 = build_feature_table(assets=["ASSET_A", "ASSET_B"])
    feat_a_2 = table_2[table_2["asset"] == "ASSET_A"].reset_index(drop=True)

    # Asset A features must be completely unchanged
    pd.testing.assert_frame_equal(feat_a_1, feat_a_2)


def test_equity_weekend_gaps_window_counts_rows():
    # 7 business days spanning 9 or 10 calendar days
    bdate_range = pd.bdate_range("2023-01-02", periods=7)  # Mon to Wed next week
    df = pd.DataFrame(
        {
            "date": bdate_range,
            "close": [100.0 + i for i in range(7)],
            "volume": [1000.0] * 7,
            "log_return": [0.01] * 7,
        }
    )

    # Window of 5 rows
    result = build_asset_features(
        df,
        vol_window=5,
        vol_col="vol_5d",
        volume_window=5,
        volume_col="dvol_5d",
    )

    # Row index 4 is the 5th row (Friday). It must be non-NaN.
    assert pd.notna(result["vol_5d"].iloc[4])
    assert pd.notna(result["dvol_5d"].iloc[4])

    # Row index 5 is Monday (6th row, across weekend). It must be non-NaN.
    assert pd.notna(result["vol_5d"].iloc[5])
    assert pd.notna(result["dvol_5d"].iloc[5])


def test_nan_targets_kept_only_incomplete_windows_dropped(monkeypatch):
    n = 40
    # 40 rows: with window=30, the first 29 rows will be incomplete, leaving 11 rows.
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    df = pd.DataFrame(
        {
            "asset": ["TEST"] * n,
            "asset_class": ["crypto"] * n,
            "date": dates,
            "close": [100.0 + i for i in range(n)],
            "volume": [1000.0] * n,
            "log_return": [0.01] * n,
            "target_7d": [0.05] * (n - 3) + [np.nan, np.nan, np.nan],
            "target_14d": [0.10] * (n - 5) + [np.nan] * 5,
        }
    )

    monkeypatch.setattr("src.features.load_processed", lambda asset: df.copy())

    table = build_feature_table(assets=["TEST"])

    # Expected retained rows: 40 - (30 - 1) = 11 rows
    assert len(table) == 11

    # Verify that the last 3 rows still contain NaN in target_7d
    assert table["target_7d"].tail(3).isna().all()
    # Verify that the last 5 rows still contain NaN in target_14d
    assert table["target_14d"].tail(5).isna().all()

    # All retained rows must have non-NaN features
    assert table[VOLATILITY_COL].notna().all()
    assert table[VOLUME_COL].notna().all()


def test_input_not_mutated_and_build_is_deterministic():
    n = 35
    df = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=n, freq="D"),
            "close": [100.0 + i for i in range(n)],
            "volume": [1000.0] * n,
            "log_return": [0.01] * n,
        }
    )

    df_copy = df.copy(deep=True)

    result_1 = build_asset_features(df)
    # Input must not be mutated
    pd.testing.assert_frame_equal(df, df_copy)

    result_2 = build_asset_features(df)
    # Consecutive executions must yield identical results
    pd.testing.assert_frame_equal(result_1, result_2)


def test_save_features_atomic(tmp_path):
    dest_file = os.path.join(tmp_path, "test_features.parquet")
    df = pd.DataFrame(
        {
            "asset": ["BTCUSDT"],
            "asset_class": ["crypto"],
            "date": [pd.Timestamp("2023-01-31")],
            VOLATILITY_COL: [0.025],
            VOLUME_COL: [1e8],
            "target_7d": [0.03],
            "target_14d": [0.06],
        }
    )

    save_features(df, dest_path=dest_file)

    assert os.path.exists(dest_file)
    assert not os.path.exists(f"{dest_file}.tmp")

    loaded = pd.read_parquet(dest_file)
    pd.testing.assert_frame_equal(df, loaded)
