"""
Tests for the data_pull module.
"""
import os
from datetime import datetime, timezone
from unittest.mock import patch, Mock

import pandas as pd
import pytest

from src.data_pull import to_ms, fetch_binance, fetch_equity, update_asset, save_raw_df
from src.config import RAW_DIR, CRYPTO_ASSETS, EQUITY_ASSETS, START_DATE, COLUMNS


def test_to_ms():
    # Test string input
    assert to_ms("2023-01-01") == 1672531200000
    # Test integer input (assumed to be milliseconds)
    assert to_ms(1672531200000) == 1672531200000
    # Test float input
    assert to_ms(1672531200000.0) == 1672531200000


@patch("src.data_pull.requests.get")
def test_fetch_binance_single_page(mock_get):
    # Mock a single page response
    mock_response = Mock()
    mock_response.raise_for_status.return_value = None
    # Binance kline: [open_time, open, high, low, close, volume, close_time, quote_asset_volume, number_of_trades, taker_buy_base_asset_volume, taker_buy_quote_asset_volume, ignore]
    mock_response.json.return_value = [
        [1672531200000, "1.0", "2.0", "0.5", "1.5", "10.0", 1672617599999, "15.0", 50, "5.0", "7.5", "0"],
        [1672617600000, "1.5", "2.5", "1.0", "2.0", "12.0", 1672703999999, "18.0", 60, "6.0", "9.0", "0"],
    ]
    mock_get.return_value = mock_response

    df = fetch_binance("BTCUSDT", "2023-01-01", "2023-01-02")

    # Check that we called the API with the correct parameters
    assert mock_get.call_count == 1
    args, kwargs = mock_get.call_args
    assert kwargs["params"]["symbol"] == "BTCUSDT"
    assert kwargs["params"]["interval"] == "1d"
    assert kwargs["params"]["startTime"] == 1672531200000
    assert kwargs["params"]["endTime"] == 1672617600000  # end_date exclusive? We set endDate to 2023-01-02, which is start of day, so we want data up to 2023-01-01 inclusive.
    # Note: Our end_date conversion is start of day, so we set endTime to the start of 2023-01-02, which will include the entire day of 2023-01-01.
    # The Binance API uses endTime inclusive? Actually, the documentation says: startTime and endTime are in milliseconds.
    # We'll assume that the endTime is inclusive for the close time? We'll trust the test as written.

    # Check the DataFrame
    assert not df.empty
    assert len(df) == 2
    assert list(df.columns) == COLUMNS
    # Check dtypes
    assert df["open_time"].dtype == "int64"
    assert df["open"].dtype == "float64"
    assert df["number_of_trades"].dtype == "int64"
    # Check values
    assert df.iloc[0]["open"] == 1.0
    assert df.iloc[1]["close"] == 2.0


@patch("src.data_pull.requests.get")
def test_fetch_binance_pagination(mock_get):
    # Mock two pages: first full, second short
    mock_response1 = Mock()
    mock_response1.raise_for_status.return_value = None
    mock_response1.json.return_value = [
        [1672531200000 + i * 86400000, "1.0", "2.0", "0.5", "1.5", "10.0", 1672531200000 + (i+1)*86400000 - 1, "15.0", 50, "5.0", "7.5", "0"]
        for i in range(1000)  # 1000 candles
    ]
    mock_response2 = Mock()
    mock_response2.raise_for_status.return_value = None
    mock_response2.json.return_value = [
        [1672531200000 + 1000 * 86400000, "1.0", "2.0", "0.5", "1.5", "10.0", 1672531200000 + 1001*86400000 - 1, "15.0", 50, "5.0", "7.5", "0"],
        [1672531200000 + 1001 * 86400000, "1.0", "2.0", "0.5", "1.5", "10.0", 1672531200000 + 1002*86400000 - 1, "15.0", 50, "5.0", "7.5", "0"],
    ]  # 2 candles (short page)
    mock_get.side_effect = [mock_response1, mock_response2]

    df = fetch_binance("BTCUSDT", "2023-01-01", None)  # No end date, so we rely on short page to stop

    # Check that we called the API twice
    assert mock_get.call_count == 2
    # Check that we got 1002 rows
    assert len(df) == 1002
    # Check that the open_time of the last row is the second candle of the second page
    assert df.iloc[-1]["open_time"] == 1672531200000 + 1001 * 86400000
    # Check that we slept between calls (we can't easily test the sleep, but we can trust it's there)


@patch("src.data_pull.requests.get")
@patch("src.data_pull.datetime")
def test_fetch_binance_drop_unfinished_candle(mock_dt, mock_get):
    # Mock data that includes today's unfinished candle
    # We'll set the start date to yesterday and end date to today (so we get two days: yesterday and today incomplete)
    # We want to see that today's candle is dropped.
    # Let's assume today is 2026-10-20 (as per the system date). We'll use a fixed date for testing.
    # We'll mock the current time in the function by patching datetime? Alternatively, we can set the end_date to a date in the future and let the function drop today's candle based on the current time.
    # However, the function uses datetime.now(timezone.utc) to determine today.
    # We'll patch datetime in the module to return a fixed date for now, but delegate strptime to the real datetime.

    # We'll set the fixed date to 2026-10-20 (so that today's candle is 2026-10-20)
    fixed_today = datetime(2026, 10, 20, tzinfo=timezone.utc)

    mock_dt.now.return_value = fixed_today
    # Delegate strptime to the real datetime
    mock_dt.strptime.side_effect = lambda *args, **kw: datetime.strptime(*args, **kw)

    # Compute the open_time for 2026-10-19 and 2026-10-20 (start of day in UTC)
    open_time_19 = to_ms("2026-10-19")
    open_time_20 = to_ms("2026-10-20")
    # Mock API response for two days: 2026-10-19 and 2026-10-20 (but we want to drop 2026-10-20)
    mock_response = Mock()
    mock_response.raise_for_status.return_value = None
    # Two candles: one for 2026-10-19 and one for 2026-10-20
    mock_response.json.return_value = [
        [open_time_19, "1.0", "2.0", "0.5", "1.5", "10.0", open_time_19 + 86400000 - 1, "15.0", 50, "5.0", "7.5", "0"],  # 2026-10-19
        [open_time_20, "2.0", "3.0", "1.5", "2.5", "12.0", open_time_20 + 86400000 - 1, "18.0", 60, "6.0", "9.0", "0"],   # 2026-10-20
    ]
    mock_get.return_value = mock_response

    df = fetch_binance("BTCUSDT", "2026-10-19", "2026-10-20")

    # We should only get one row (the 2026-10-19 candle) because the 2026-10-20 candle is today and unfinished.
    assert len(df) == 1
    assert df.iloc[0]["open_time"] == open_time_19


@patch("src.data_pull.requests.get")
def test_fetch_binance_empty_response(mock_get):
    mock_response = Mock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = []
    mock_get.return_value = mock_response

    df = fetch_binance("BTCUSDT", "2023-01-01", "2023-01-02")
    assert df.empty
    assert list(df.columns) == COLUMNS


@patch("src.data_pull.requests.get")
@patch("src.data_pull.datetime")
def test_fetch_binance_no_duplicate_across_pages(mock_dt, mock_get):
    # Mock two pages: first with candles [0, 1, ..., 999], second with [1000, 1001, ...]
    # We'll make sure there's no overlap.
    def generate_candles(start_idx, count):
        return [
            [
                1672531200000 + (start_idx + i) * 86400000,
                "1.0",
                "2.0",
                "0.5",
                "1.5",
                "10.0",
                1672531200000 + (start_idx + i + 1) * 86400000 - 1,
                "15.0",
                50,
                "5.0",
                "7.5",
                "0",
            ]
            for i in range(count)
        ]

    # Set the current date to be far in the future so that no candles are dropped as "today's unfinished candle"
    future_date = datetime(2030, 1, 1, tzinfo=timezone.utc)
    mock_dt.now.return_value = future_date
    # Delegate strptime to the real datetime
    mock_dt.strptime.side_effect = lambda *args, **kw: datetime.strptime(*args, **kw)

    mock_response1 = Mock()
    mock_response1.raise_for_status.return_value = None
    mock_response1.json.return_value = generate_candles(0, 1000)

    mock_response2 = Mock()
    mock_response2.raise_for_status.return_value = None
    mock_response2.json.return_value = generate_candles(1000, 500)  # next 500

    mock_get.side_effect = [mock_response1, mock_response2]

    df = fetch_binance("BTCUSDT", "2023-01-01", None)

    # We expect 1500 rows (no dropping because current date is in the future)
    assert len(df) == 1500
    # Check that the open_time values are unique and consecutive
    open_times = df["open_time"].tolist()
    assert open_times == list(range(1672531200000, 1672531200000 + 1500 * 86400000, 86400000))


# === Equity tests ===
@patch("src.data_pull.yf.download")
def test_fetch_equity_schema_and_dtypes(mock_download):
    # Mock yfinance to return a DataFrame with two days of data
    index = pd.DatetimeIndex([
        pd.Timestamp("2023-01-01", tz="UTC"),
        pd.Timestamp("2023-01-02", tz="UTC"),
    ])
    mock_df = pd.DataFrame({
        "Open": [1.0, 1.5],
        "High": [2.0, 2.5],
        "Low": [0.5, 1.0],
        "Close": [1.5, 2.0],
        "Volume": [10.0, 12.0],
    }, index=index)
    mock_download.return_value = mock_df

    df = fetch_equity("SPY", "2023-01-01", "2023-01-02")

    # Check that we called yfinance.download with the correct parameters
    mock_download.assert_called_once()
    args, kwargs = mock_download.call_args
    assert args[0] == "SPY"
    assert kwargs["start"] == "2023-01-01"
    assert kwargs["end"] == "2023-01-02"
    assert kwargs["progress"] == False
    assert kwargs["auto_adjust"] == True

    # Check the DataFrame
    assert not df.empty
    assert len(df) == 2
    assert list(df.columns) == COLUMNS
    # Check dtypes
    assert df["open_time"].dtype == "int64"
    assert df["open"].dtype == "float64"
    assert df["number_of_trades"].dtype == "float64"  # NaN is float
    assert pd.isna(df["number_of_trades"]).all()
    assert pd.isna(df["taker_buy_base_asset_volume"]).all()
    assert pd.isna(df["taker_buy_quote_asset_volume"]).all()
    # Check values (open, high, low, close, volume should match)
    assert df.iloc[0]["open"] == 1.0
    assert df.iloc[1]["close"] == 2.0
    # Check that close_time is open_time + 86400000 - 1
    assert df.iloc[0]["close_time"] == df.iloc[0]["open_time"] + 86400000 - 1
    # Check that quote_asset_volume is volume * close (approximation)
    assert df.iloc[0]["quote_asset_volume"] == df.iloc[0]["volume"] * df.iloc[0]["close"]
    assert df.iloc[0]["ignore"] == 0


@patch("src.data_pull.yf.download")
@patch("src.data_pull.datetime")
def test_fetch_equity_drops_today_row(mock_dt, mock_download):
    # Set today to 2026-10-20
    fixed_today = datetime(2026, 10, 20, tzinfo=timezone.utc)
    mock_dt.now.return_value = fixed_today
    # Delegate strptime to the real datetime
    mock_dt.strptime.side_effect = lambda *args, **kw: datetime.strptime(*args, **kw)

    # Mock yfinance to return data for 2026-10-19 and 2026-10-20
    index = pd.DatetimeIndex([
        pd.Timestamp("2026-10-19", tz="UTC"),
        pd.Timestamp("2026-10-20", tz=timezone.utc),
    ])
    mock_df = pd.DataFrame({
        "Open": [1.0, 2.0],
        "High": [2.0, 3.0],
        "Low": [0.5, 1.5],
        "Close": [1.5, 2.5],
        "Volume": [10.0, 12.0],
    }, index=index)
    mock_download.return_value = mock_df

    df = fetch_equity("SPY", "2026-10-19", "2026-10-20")

    # We should only get one row (yesterday) because today's row is dropped
    expected_open_time = to_ms("2026-10-19")
    print(f"DEBUG: expected_open_time = {expected_open_time}", flush=True)
    print(f"DEBUG: df.iloc[0]['open_time'] = {df.iloc[0]['open_time']}", flush=True)
    print(f"DEBUG: df = {df}", flush=True)
    assert len(df) == 1
    assert int(df.iloc[0]["open_time"]) == expected_open_time


def test_update_asset_incremental_append(tmpdir):
    # Setup a temporary RAW_DIR
    with patch("src.data_pull.RAW_DIR", str(tmpdir)):
        # Ensure the directory exists
        os.makedirs(tmpdir, exist_ok=True)
        symbol = "SPY"  # Use an equity symbol
        # Simulate existing data for 2023-01-01 and 2023-01-02
        existing_data = pd.DataFrame({
            "open_time": [to_ms("2023-01-01"), to_ms("2023-01-02")],
            "open": [1.0, 1.5],
            "high": [2.0, 2.5],
            "low": [0.5, 1.0],
            "close": [1.5, 2.0],
            "volume": [10.0, 12.0],
            "close_time": [to_ms("2023-01-01") + 86400000 - 1, to_ms("2023-01-02") + 86400000 - 1],
            "quote_asset_volume": [float("nan"), float("nan")],
            "number_of_trades": [float("nan"), float("nan")],
            "taker_buy_base_asset_volume": [float("nan"), float("nan")],
            "taker_buy_quote_asset_volume": [float("nan"), float("nan")],
            "ignore": [0, 0],
        })
        # Save existing data
        save_raw_df(existing_data, symbol)

        # Mock yfinance to return new data for 2023-01-03 and 2023-01-04
        with patch("src.data_pull.yf.download") as mock_download:
            index = pd.DatetimeIndex([
                pd.Timestamp("2023-01-03", tz="UTC"),
                pd.Timestamp("2023-01-04", tz="UTC"),
            ])
            mock_df = pd.DataFrame({
                "Open": [2.0, 2.5],
                "High": [3.0, 3.5],
                "Low": [1.5, 2.0],
                "Close": [2.5, 3.0],
                "Volume": [14.0, 16.0],
            }, index=index)
            mock_download.return_value = mock_df

            # Run update_asset
            update_asset(symbol)

            # Check that yfinance was called with the correct start date (day after last saved date)
            mock_download.assert_called_once()
            args, kwargs = mock_download.call_args
            assert args[0] == symbol
            assert kwargs["start"] == "2023-01-03"
            # end_date should be None (to get up to today, but we rely on drop today's row inside fetch_equity)
            assert kwargs["end"] is None
            assert kwargs["progress"] == False
            assert kwargs["auto_adjust"] == True

            # Read the updated Parquet file
            updated_path = os.path.join(tmpdir, f"{symbol}.parquet")
            assert os.path.exists(updated_path)
            updated_df = pd.read_parquet(updated_path)

            # Expect 4 rows: original 2 + new 2
            assert len(updated_df) == 4
            # Check that the open_time values are in order and correct
            expected_open_times = [
                to_ms("2023-01-01"),
                to_ms("2023-01-02"),
                to_ms("2023-01-03"),
                to_ms("2023-01-04"),
            ]
            assert updated_df["open_time"].tolist() == expected_open_times
            # Check that the new rows have the correct values (from mock)
            assert updated_df.iloc[2]["open"] == 2.0
            assert updated_df.iloc[3]["close"] == 3.0
            # Check that the new rows have NaN for the traded volume columns
            assert pd.isna(updated_df.iloc[2]["number_of_trades"])
            assert pd.isna(updated_df.iloc[3]["number_of_trades"])


def test_update_asset_running_twice_adds_nothing(tmpdir, capsys):
    with patch("src.data_pull.RAW_DIR", str(tmpdir)):
        os.makedirs(tmpdir, exist_ok=True)
        symbol = "SPY"  # Use an equity symbol
        # Start with no existing file
        # Mock yfinance to return data for 2023-01-01 and 2023-01-02 on first call, and empty on second call
        with patch("src.data_pull.yf.download") as mock_download:
            # First call returns data
            index1 = pd.DatetimeIndex([
                pd.Timestamp("2023-01-01", tz="UTC"),
                pd.Timestamp("2023-01-02", tz="UTC"),
            ])
            mock_df1 = pd.DataFrame({
                "Open": [1.0, 1.5],
                "High": [2.0, 2.5],
                "Low": [0.5, 1.0],
                "Close": [1.5, 2.0],
                "Volume": [10.0, 12.0],
            }, index=index1)
            # Second call returns empty DataFrame
            mock_empty = pd.DataFrame()
            mock_download.side_effect = [mock_df1, mock_empty]

            # First update
            update_asset(symbol)
            # Capture output to see "Added X new rows"
            captured = capsys.readouterr()
            assert "Added 2 new rows." in captured.out

            # Second update
            update_asset(symbol)
            captured = capsys.readouterr()
            # Should indicate no new data
            assert "No new data to add." in captured.out

            # Verify the file still has only 2 rows
            updated_path = os.path.join(tmpdir, f"{symbol}.parquet")
            assert os.path.exists(updated_path)
            df = pd.read_parquet(updated_path)
            assert len(df) == 2


def test_save_raw_df(tmpdir):
    # We'll use a temporary directory for RAW_DIR
    with patch("src.data_pull.RAW_DIR", str(tmpdir)):
        df = pd.DataFrame(
            {
                "open_time": [1672531200000],
                "open": [1.0],
                "high": [2.0],
                "low": [0.5],
                "close": [1.5],
                "volume": [10.0],
                "close_time": [1672617599999],
                "quote_asset_volume": [15.0],
                "taker_buy_base_asset_volume": [5.0],
                "taker_buy_quote_asset_volume": [7.5],
                "ignore": [0],
            }
        )
        save_raw_df(df, "BTCUSDT")
        # Check that the file was written
        expected_path = os.path.join(str(tmpdir), "BTCUSDT.parquet")
        assert os.path.exists(expected_path)
        # Read it back and check the data
        df_read = pd.read_parquet(expected_path)
        pd.testing.assert_frame_equal(df, df_read)


if __name__ == "__main__":
    pytest.main([__file__])