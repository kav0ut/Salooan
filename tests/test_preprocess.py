"""
Unit tests for data preprocessing and target generation (Milestone 2).

All tests use small synthetic DataFrames without making API calls or reading
actual production data.
"""
import os
import warnings
from typing import List

import numpy as np
import pandas as pd
import pytest

from src.preprocess import (
    add_forward_return,
    add_log_return,
    clean,
    find_gaps,
    load_raw,
    process_asset,
    save_processed,
)


# ---------------------------------------------------------------------------
# Test 1: Log returns against hand-computed values and NaN on first row
# ---------------------------------------------------------------------------
def test_add_log_return_hand_computed():
    """Verify log return calculation against hand-computed values.

    Log return formula: ln(close_t / close_{t-1}).
    The first row has no previous close, so its log return must be NaN.
    """
    # Synthetic closes: 100.0, 105.0 (+5%), 94.5 (-10% from 105)
    df = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=3, freq="D"),
            "close": [100.0, 105.0, 94.5],
        }
    )

    result = add_log_return(df)

    # 1. First row must be NaN
    assert pd.isna(result["log_return"].iloc[0])

    # 2. Hand-computed expected values
    expected_row1 = np.log(105.0 / 100.0)  # ~0.048790
    expected_row2 = np.log(94.5 / 105.0)   # ~-0.105360

    assert result["log_return"].iloc[1] == pytest.approx(expected_row1, rel=1e-6)
    assert result["log_return"].iloc[2] == pytest.approx(expected_row2, rel=1e-6)

    # 3. Input DataFrame must not be mutated
    assert "log_return" not in df.columns


# ---------------------------------------------------------------------------
# Test 2: Clean data - duplicates, sorting, invalid closes, immutability
# ---------------------------------------------------------------------------
def test_clean_filters_sorts_and_preserves_input():
    """Verify clean() sorts by date, drops duplicates (keeping first),
    removes non-positive or NaN closes, enforces float64 dtypes,
    and does not mutate input.
    """
    raw_data = {
        "date": [
            "2023-01-05",  # Out of order
            "2023-01-02",  # First occurrence of 2023-01-02 (valid close: 100)
            "2023-01-02",  # Duplicate date (close: 999 - should be dropped)
            "2023-01-03",  # Negative close (-10 - invalid, should be dropped)
            "2023-01-04",  # Zero close (0.0 - invalid, should be dropped)
            "2023-01-06",  # NaN close (invalid, should be dropped)
            "2023-01-07",  # Valid close (200)
        ],
        "close": [150.0, 100.0, 999.0, -10.0, 0.0, np.nan, 200.0],
        "open": ["140", "95", "900", "0", "0", "190", "195"],  # String types to test dtype casting
        "volume": [1000, 2000, 3000, 4000, 5000, 6000, 7000],
    }
    input_df = pd.DataFrame(raw_data)
    input_df_copy = input_df.copy(deep=True)

    cleaned = clean(input_df)

    # 1. Input DataFrame must not be mutated
    pd.testing.assert_frame_equal(input_df, input_df_copy)

    # 2. Only valid rows remain: 2023-01-02, 2023-01-05, 2023-01-07
    expected_dates = [
        pd.Timestamp("2023-01-02"),
        pd.Timestamp("2023-01-05"),
        pd.Timestamp("2023-01-07"),
    ]
    assert list(cleaned["date"]) == expected_dates

    # 3. Duplicate retained the first occurrence (close=100.0, not 999.0)
    assert cleaned.iloc[0]["close"] == 100.0

    # 4. Enforce float64 dtypes on numeric columns
    assert cleaned["close"].dtype == "float64"
    assert cleaned["open"].dtype == "float64"
    assert cleaned["volume"].dtype == "float64"


# ---------------------------------------------------------------------------
# Test 3: Gap detection warns for crypto, but not for equity weekends
# ---------------------------------------------------------------------------
def test_find_gaps_crypto_warning_and_equity_tolerance():
    """Verify find_gaps() raises UserWarning for crypto gaps,
    but does not warn for expected equity calendar gaps (weekends/holidays).
    """
    # Data with a weekend gap: Friday to Monday (missing Sat, Sun: 2 missing days)
    equity_df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2023-01-06"), pd.Timestamp("2023-01-09")],
            "asset_class": ["equity", "equity"],
        }
    )

    # Equity: finds the gap in calendar days, but should NOT issue a UserWarning
    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        equity_gaps = find_gaps(equity_df, asset_class="equity")
        assert len(equity_gaps) == 1
        assert equity_gaps[0] == (pd.Timestamp("2023-01-06"), pd.Timestamp("2023-01-09"), 2)
        # Verify no UserWarning was triggered
        user_warnings = [w for w in recorded_warnings if issubclass(w.category, UserWarning)]
        assert len(user_warnings) == 0

    # Crypto: missing calendar days should trigger a UserWarning
    crypto_df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-03")],
            "asset_class": ["crypto", "crypto"],
        }
    )
    with pytest.warns(UserWarning, match="Crypto data gap detected"):
        crypto_gaps = find_gaps(crypto_df, asset_class="crypto")
        assert len(crypto_gaps) == 1
        assert crypto_gaps[0] == (pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-03"), 1)

    # Crypto: continuous series with no gaps should not warn
    crypto_continuous = pd.DataFrame(
        {
            "date": [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-02")],
            "asset_class": ["crypto", "crypto"],
        }
    )
    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        gaps = find_gaps(crypto_continuous, asset_class="crypto")
        assert len(gaps) == 0
        user_warnings = [w for w in recorded_warnings if issubclass(w.category, UserWarning)]
        assert len(user_warnings) == 0


# ---------------------------------------------------------------------------
# Test 4: Crypto-style series - target_7d equals exact shift of 7 rows
# ---------------------------------------------------------------------------
def test_crypto_target_7d_equals_7_rows_ahead():
    """For continuous 24/7 crypto series, calendar t+7 corresponds exactly
    to the candle 7 rows ahead: target_7d = close_{t+7} / close_t - 1.
    """
    num_days = 15
    dates = pd.date_range("2023-01-01", periods=num_days, freq="D")
    # Linear prices: 100, 101, 102, ..., 114
    closes = [100.0 + i for i in range(num_days)]
    df = pd.DataFrame({"date": dates, "close": closes})

    result = add_forward_return(df, horizon_days=7)

    assert "target_7d" in result.columns

    # For index 0 to 7 (where t+7 is within dataset):
    for i in range(num_days - 7):
        expected_target = (closes[i + 7] / closes[i]) - 1.0
        assert result["target_7d"].iloc[i] == pytest.approx(expected_target, rel=1e-6)

    # The last 7 days cannot see 7 days ahead; they must be NaN
    for i in range(num_days - 7, num_days):
        assert pd.isna(result["target_7d"].iloc[i])


# ---------------------------------------------------------------------------
# Test 5: Equity-style series with weekend gaps - as-of uses right earlier close
# ---------------------------------------------------------------------------
def test_equity_forward_return_asof_weekend_gaps():
    """Verify as-of backward lookup handles weekends and non-trading days.

    When t+N lands on a weekend (Saturday or Sunday), the as-of lookup must
    match the last trading day on or before t+N (Friday's close).
    """
    # Business days over 2 weeks (Monday 2023-01-02 to Friday 2023-01-13)
    b_dates = pd.bdate_range("2023-01-02", "2023-01-13")
    # Closes for each business day:
    # 2023-01-02 (Mon): 100.0
    # 2023-01-03 (Tue): 102.0
    # 2023-01-04 (Wed): 104.0
    # 2023-01-05 (Thu): 106.0
    # 2023-01-06 (Fri): 110.0
    # 2023-01-09 (Mon): 112.0
    # 2023-01-10 (Tue): 114.0
    # 2023-01-11 (Wed): 116.0
    # 2023-01-12 (Thu): 118.0
    # 2023-01-13 (Fri): 120.0
    closes = [100.0, 102.0, 104.0, 106.0, 110.0, 112.0, 114.0, 116.0, 118.0, 120.0]
    df = pd.DataFrame({"date": b_dates, "close": closes})

    # Test horizon = 2 calendar days
    res_2d = add_forward_return(df, horizon_days=2, tolerance_days=5)

    # 1. Wednesday 2023-01-04 (index 2): t+2 is Friday 2023-01-06 (trading day, close=110.0)
    # Target = 110.0 / 104.0 - 1.0
    expected_wed = (110.0 / 104.0) - 1.0
    assert res_2d["target_2d"].iloc[2] == pytest.approx(expected_wed, rel=1e-6)

    # 2. Thursday 2023-01-05 (index 3): t+2 is Saturday 2023-01-07 (non-trading day).
    # Latest available close on or before Jan 07 is Friday Jan 06 (close=110.0).
    # Target = 110.0 / 106.0 - 1.0
    expected_thu = (110.0 / 106.0) - 1.0
    assert res_2d["target_2d"].iloc[3] == pytest.approx(expected_thu, rel=1e-6)

    # 3. Test horizon = 7 calendar days
    res_7d = add_forward_return(df, horizon_days=7, tolerance_days=5)

    # Monday 2023-01-02 (index 0): t+7 is Monday 2023-01-09 (trading day, close=112.0)
    expected_mon = (112.0 / 100.0) - 1.0
    assert res_7d["target_7d"].iloc[0] == pytest.approx(expected_mon, rel=1e-6)

    # Friday 2023-01-06 (index 4): t+7 is Friday 2023-01-13 (trading day, close=120.0)
    expected_fri = (120.0 / 110.0) - 1.0
    assert res_7d["target_7d"].iloc[4] == pytest.approx(expected_fri, rel=1e-6)


# ---------------------------------------------------------------------------
# Test 6: Last N days are NaN and tolerance stops stale match
# ---------------------------------------------------------------------------
def test_forward_return_end_nan_and_tolerance_cutoff():
    """Verify:
    1. Rows whose target date t+N falls beyond max_date are NaN.
    2. Tolerance stops matching stale prices when gap between t+N and
       the prior close exceeds tolerance_days.
    """
    # Part 1: Last N calendar days must have target = NaN
    df_recent = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", "2023-01-10", freq="D"),
            "close": [100.0] * 10,
        }
    )
    res_recent = add_forward_return(df_recent, horizon_days=5)
    # Dates 2023-01-06 to 2023-01-10 have t+5 > 2023-01-10 -> NaN
    for dt, val in zip(res_recent["date"], res_recent["target_5d"]):
        if dt + pd.Timedelta(days=5) > pd.Timestamp("2023-01-10"):
            assert pd.isna(val), f"Expected NaN for date {dt}"
        else:
            assert not pd.isna(val), f"Expected non-NaN for date {dt}"

    # Part 2: Tolerance stops stale match
    # Say asset traded on Jan 01 and Jan 02, then halted until Jan 15.
    halted_df = pd.DataFrame(
        {
            "date": [
                pd.Timestamp("2023-01-01"),
                pd.Timestamp("2023-01-02"),
                pd.Timestamp("2023-01-15"),
            ],
            "close": [100.0, 102.0, 150.0],
        }
    )
    # For date 2023-01-01 with horizon 7d:
    # t+7 is 2023-01-08.
    # The last trading close on or before 2023-01-08 is 2023-01-02 (gap = 6 days).
    # If tolerance is 5 days, 6 days > 5 days -> must return NaN!
    res_halted = add_forward_return(halted_df, horizon_days=7, tolerance_days=5)
    assert pd.isna(res_halted["target_7d"].iloc[0]), "Stale price within 6 days should exceed 5-day tolerance and be NaN"

    # If tolerance is relaxed to 7 days, 6 days <= 7 days -> matches Jan 02
    res_relaxed = add_forward_return(halted_df, horizon_days=7, tolerance_days=7)
    expected_return = (102.0 / 100.0) - 1.0
    assert res_relaxed["target_7d"].iloc[0] == pytest.approx(expected_return, rel=1e-6)


# ---------------------------------------------------------------------------
# Test 7: Target at t depends only on closes at t and t+N (no path leakage)
# ---------------------------------------------------------------------------
def test_target_depends_only_on_endpoints_no_path_leakage():
    """Verify changing intermediate prices between t and t+N does not affect target_{N}d.

    The forward return is close(t+N) / close(t) - 1 and must be completely
    independent of any interim volatility, spikes, or drops between t and t+N.
    """
    dates = pd.date_range("2023-01-01", periods=15, freq="D")

    # Baseline series: steady prices
    closes_baseline = [100.0] * 15
    df_base = pd.DataFrame({"date": dates, "close": closes_baseline})

    # Modified series: wild intermediate price fluctuations between t+1 and t+6
    closes_modified = list(closes_baseline)
    closes_modified[1] = 999.0
    closes_modified[2] = 10.0
    closes_modified[3] = 555.0
    closes_modified[4] = 20.0
    closes_modified[5] = 777.0
    closes_modified[6] = 30.0
    # Note: closes_modified[0] (t) and closes_modified[7] (t+7) remain 100.0
    df_mod = pd.DataFrame({"date": dates, "close": closes_modified})

    res_base = add_forward_return(df_base, horizon_days=7)
    res_mod = add_forward_return(df_mod, horizon_days=7)

    # target_7d at t=0 must be identical (0.0 in both cases)
    assert res_base["target_7d"].iloc[0] == pytest.approx(0.0)
    assert res_mod["target_7d"].iloc[0] == pytest.approx(0.0)
    assert res_base["target_7d"].iloc[0] == res_mod["target_7d"].iloc[0]


# ---------------------------------------------------------------------------
# Test 8: Running process_asset twice gives identical output (idempotence)
# ---------------------------------------------------------------------------
def test_process_asset_idempotence(tmp_path, monkeypatch):
    """Verify calling process_asset twice on the same raw file produces
    strictly identical results and writes identically.
    """
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Monkeypatch RAW_DIR and PROCESSED_DIR in src.preprocess
    monkeypatch.setattr("src.preprocess.RAW_DIR", str(raw_dir))
    monkeypatch.setattr("src.preprocess.PROCESSED_DIR", str(processed_dir))

    # Create synthetic raw parquet data for asset
    test_asset = "BTCUSDT"
    dates = pd.date_range("2023-01-01", periods=30, freq="D")
    raw_df = pd.DataFrame(
        {
            "date": dates,
            "open": np.linspace(100.0, 130.0, len(dates)),
            "high": np.linspace(105.0, 135.0, len(dates)),
            "low": np.linspace(95.0, 125.0, len(dates)),
            "close": np.linspace(101.0, 131.0, len(dates)),
            "volume": np.full(len(dates), 1000.0),
        }
    )
    raw_file = raw_dir / f"{test_asset}.parquet"
    raw_df.to_parquet(raw_file, index=False)

    # First run
    run1 = process_asset(test_asset)

    # Second run
    run2 = process_asset(test_asset)

    # Both returned DataFrames must be identical
    pd.testing.assert_frame_equal(run1, run2)

    # Processed parquet file on disk must match exactly
    processed_file = processed_dir / f"{test_asset}.parquet"
    assert processed_file.exists()
    disk_df = pd.read_parquet(processed_file)
    pd.testing.assert_frame_equal(run1, disk_df)
