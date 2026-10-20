"""
Module responsible for pulling raw data from external sources (e.g., Binance, Yahoo Finance).
Reads configuration from src.config for asset symbols and date ranges.
Writes raw data files to data/raw directory.
"""
import os
import time
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import requests
import yfinance as yf

from src.config import BINANCE_URL, COLUMNS, RAW_DIR, CRYPTO_ASSETS, EQUITY_ASSETS, START_DATE


def to_ms(date_str: str) -> int:
    """Convert a date string in YYYY-MM-DD format to milliseconds since epoch (UTC start of day).
    If the input is already an integer, assume it's milliseconds and return it.
    """
    if isinstance(date_str, (int, float)):
        return int(date_str)
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def fetch_binance(symbol: str, start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
    """Fetch daily klines from Binance for a symbol and date range.
    Handles pagination (1000 limit per request), stops on a short page,
    sleeps 0.2 seconds between requests, and drops today's unfinished candle.
    Returns a DataFrame with the raw schema and correct dtypes.
    """
    start_ms = to_ms(start_date)
    end_ms = to_ms(end_date) if end_date else None

    all_data = []
    start_time = start_ms

    while True:
        params = {
            "symbol": symbol.upper(),
            "interval": "1d",
            "startTime": start_time,
            "limit": 1000,
        }
        if end_ms is not None:
            params["endTime"] = end_ms

        resp = requests.get(BINANCE_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if not data:
            break

        df = pd.DataFrame(data, columns=COLUMNS)
        all_data.append(df)

        # If we got less than the limit, we've reached the end
        if len(data) < 1000:
            break

        # Otherwise, set start_time to the open time of the next candle
        start_time = int(df.iloc[-1]["open_time"]) + 1
        time.sleep(0.2)

    if not all_data:
        return pd.DataFrame(columns=COLUMNS)

    df = pd.concat(all_data, ignore_index=True)
    # Drop duplicates on open_time (just in case)
    df = df.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)

    # Convert column types
    numeric_cols = ["open", "high", "low", "close", "volume", "quote_asset_volume",
                    "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume"]
    int_cols = ["open_time", "close_time", "number_of_trades", "ignore"]
    df[numeric_cols] = df[numeric_cols].astype("float64")
    df[int_cols] = df[int_cols].astype("int64")

    # Drop today's unfinished candle
    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    today_start_ms = int(today_start.timestamp() * 1000)
    df = df[df["open_time"] < today_start_ms].copy()

    return df


def fetch_equity(symbol: str, start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
    """Fetch daily equity data from Yahoo Finance for a symbol and date range.
    Uses auto_adjust=True, drops today's row, and maps to the raw schema.
    NaN is used for n_trades and taker_buy_base/quote_asset_volume.
    """
    # yfinance expects dates as strings; if end_date is None, it defaults to today (but we will drop today later)
    df = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=True)
    if df.empty:
        return pd.DataFrame(columns=COLUMNS)

    # If columns are MultiIndex (e.g., due to single symbol), flatten
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Reset index to have Date as a column
    df = df.reset_index()
    # Rename columns: the first column is the date, then standard OHLCV
    df.rename(columns={
        df.columns[0]: "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    }, inplace=True)

    # Convert date to milliseconds since epoch (UTC start of day)
    # Using .value // 10**6 gives milliseconds since epoch, assuming the timestamp is in UTC.
    df["open_time"] = df["date"].apply(lambda x: x.value // 10**6)
    # For equities, we don't have close_time, number_of_trades, or volume breakdowns.
    # We'll set close_time to open_time + 86400000 - 1 (end of day) as an approximation.
    df["close_time"] = df["open_time"] + 86400000 - 1
    df["quote_asset_volume"] = df["volume"] * df["close"]  # approximate; not used for equities
    df["number_of_trades"] = float("nan")
    df["taker_buy_base_asset_volume"] = float("nan")
    df["taker_buy_quote_asset_volume"] = float("nan")
    df["ignore"] = 0

    # Reorder columns to match COLUMNS
    df = df[COLUMNS]

    # Drop today's unfinished candle (same logic as Binance)
    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    today_start_ms = int(today_start.timestamp() * 1000)
    df = df[df["open_time"] < today_start_ms].copy()

    return df


def save_raw_df(df: pd.DataFrame, symbol: str) -> None:
    """Save a DataFrame to RAW_DIR as a Parquet file using an atomic write."""
    os.makedirs(RAW_DIR, exist_ok=True)
    temp_path = os.path.join(RAW_DIR, f"{symbol}.parquet.tmp")
    final_path = os.path.join(RAW_DIR, f"{symbol}.parquet")
    df.to_parquet(temp_path, index=False)
    os.replace(temp_path, final_path)


def update_asset(symbol: str) -> None:
    """Incrementally update raw data for a symbol.
    Reads existing Parquet (if any), fetches new data from the day after the last saved date,
    merges, drops duplicates, sorts, and saves atomically.
    Prints the number of rows added.
    """
    symbol_upper = symbol.upper()
    parquet_path = os.path.join(RAW_DIR, f"{symbol_upper}.parquet")

    # Determine start date for fetching new data
    if os.path.exists(parquet_path):
        try:
            existing_df = pd.read_parquet(parquet_path)
            if not existing_df.empty:
                # Get the maximum open_time (in milliseconds) and convert to date string
                last_open_time = existing_df["open_time"].max()
                last_date = pd.to_datetime(last_open_time, unit='ms', utc=True).date()
                # Start fetching from the next day
                start_date = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            else:
                # Existing file is empty; start from the configured START_DATE
                start_date = START_DATE
        except Exception:
            # If reading fails, start from START_DATE
            start_date = START_DATE
    else:
        # No existing file; start from START_DATE
        start_date = START_DATE

    # Fetch new data based on asset type
    if symbol_upper in CRYPTO_ASSETS:
        new_df = fetch_binance(symbol_upper, start_date)
    elif symbol_upper in EQUITY_ASSETS:
        new_df = fetch_equity(symbol_upper, start_date)
    else:
        raise ValueError(f"Unknown asset type: {symbol}")

    if new_df.empty:
        print(f"{symbol_upper}: No new data to add.")
        return

    # Combine with existing data (if any)
    if os.path.exists(parquet_path) and os.path.getsize(parquet_path) > 0:
        try:
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        except NameError:
            # existing_df not defined (empty file case)
            combined_df = new_df
    else:
        combined_df = new_df

    # Drop duplicates on open_time, keep the first occurrence
    combined_df = combined_df.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)

    # Save atomically
    save_raw_df(combined_df, symbol_upper)

    rows_added = len(new_df)
    print(f"{symbol_upper}: Added {rows_added} new rows.")


def main() -> None:
    """Main entry point: update all assets defined in config."""
    for symbol in CRYPTO_ASSETS + EQUITY_ASSETS:
        update_asset(symbol)


if __name__ == "__main__":
    main()