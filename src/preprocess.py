"""
Module responsible for cleaning and preprocessing raw asset data.

Reads raw parquet data from data/raw/, enforces schemas and dtypes,
calculates daily log returns, computes forward returns (targets) over
specified calendar-day horizons using as-of lookups, and writes
processed parquet files to data/processed/.
"""
import os
import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import (
    CRYPTO_ASSETS,
    EQUITY_ASSETS,
    HORIZONS_DAYS,
    PROCESSED_DIR,
    RAW_DIR,
    TARGET_ASOF_TOLERANCE_DAYS,
)


def load_raw(asset: str) -> pd.DataFrame:
    """Read data/raw/<asset>.parquet.

    Args:
        asset: The symbol name (e.g. 'BTCUSDT' or 'SPY').

    Returns:
        pd.DataFrame containing the raw ingested data.
    """
    file_path = os.path.join(RAW_DIR, f"{asset}.parquet")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Raw data file not found: {file_path}")
    return pd.read_parquet(file_path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw price DataFrame.

    - Ensures date column exists (converted from open_time if needed, normalized to midnight).
    - Sorts chronologically by date.
    - Drops duplicate dates (retaining the first occurrence).
    - Drops rows with NaN or non-positive close prices.
    - Enforces float64 dtypes on numeric price/volume columns.
    - Never forward-fills prices.
    - Returns a new DataFrame without mutating the input.
    """
    df = df.copy()

    # Convert open_time (ms) to normalized tz-naive datetime if date column is missing
    if "date" not in df.columns and "open_time" in df.columns:
        df["date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.floor("D").dt.tz_localize(None)
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        if hasattr(df["date"].dt, "tz") and df["date"].dt.tz is not None:
            df["date"] = df["date"].dt.tz_localize(None)
        df["date"] = df["date"].dt.floor("D")

    # Drop rows where close price is missing or non-positive
    df = df[df["close"].notna() & (df["close"] > 0)]

    # Sort chronologically by date
    df = df.sort_values("date")

    # Drop duplicate dates
    df = df.drop_duplicates(subset=["date"], keep="first")

    # Enforce dtypes on numeric columns
    df["close"] = df["close"].astype("float64")
    numeric_cols = [
        "open",
        "high",
        "low",
        "volume",
        "quote_asset_volume",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype("float64")

    return df.reset_index(drop=True)


def find_gaps(
    df: pd.DataFrame, asset_class: Optional[str] = None
) -> List[Tuple[pd.Timestamp, pd.Timestamp, int]]:
    """Report missing calendar days between consecutive rows.

    For equities, weekend and holiday gaps are expected and normal.
    For crypto, markets trade 24/7/365, so any missing calendar day triggers a warning.

    Args:
        df: DataFrame containing at least a 'date' column.
        asset_class: 'crypto' or 'equity'. If None, attempts to infer from df columns.

    Returns:
        List of tuples: (previous_date, current_date, missing_days_count).
    """
    if asset_class is not None:
        is_crypto = str(asset_class).lower() == "crypto"
    elif "asset_class" in df.columns and len(df) > 0:
        is_crypto = str(df["asset_class"].iloc[0]).lower() == "crypto"
    elif "asset" in df.columns and len(df) > 0:
        is_crypto = df["asset"].iloc[0] in CRYPTO_ASSETS
    else:
        is_crypto = False

    if len(df) < 2:
        return []

    df_sorted = df.sort_values("date").reset_index(drop=True)
    gaps: List[Tuple[pd.Timestamp, pd.Timestamp, int]] = []

    for i in range(1, len(df_sorted)):
        prev_date = df_sorted["date"].iloc[i - 1]
        curr_date = df_sorted["date"].iloc[i]
        diff_days = (curr_date - prev_date).days
        if diff_days > 1:
            missing_count = diff_days - 1
            gaps.append((prev_date, curr_date, missing_count))

    if is_crypto and gaps:
        warnings.warn(
            f"Crypto data gap detected: {len(gaps)} gap(s) with missing calendar days: {gaps}",
            UserWarning,
            stacklevel=2,
        )

    return gaps


def add_log_return(df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log return: ln(close_t / close_{t-1}).

    The first row will be NaN because there is no prior day close.
    Returns a new DataFrame and does not mutate the input.
    """
    df = df.copy()
    if df.empty:
        df["log_return"] = pd.Series(dtype="float64")
        return df

    # Log return is ln(P_t / P_{t-1}) = ln(P_t) - ln(P_{t-1})
    df["log_return"] = np.log(df["close"] / df["close"].shift(1)).astype("float64")
    return df


def add_forward_return(
    df: pd.DataFrame,
    horizon_days: int,
    tolerance_days: int = TARGET_ASOF_TOLERANCE_DAYS,
) -> pd.DataFrame:
    """Add target_{N}d column representing simple forward return over N calendar days.

    target_{N}d = close(t+N) / close(t) - 1.

    Because equity markets close on weekends/holidays, date t+N might not have a close.
    We use pd.merge_asof(..., direction='backward', tolerance=tolerance_days) to match
    the latest available close on or before date t+N.
    - If t+N extends beyond the latest date available in the dataset, target is NaN.
    - If the gap between t+N and the latest prior close exceeds tolerance_days, target is NaN.
    - Rows with NaN target are kept (required for live inference on the latest N days).

    Returns a new DataFrame and does not mutate the input.
    """
    df = df.copy()
    target_col = f"target_{horizon_days}d"

    if df.empty:
        df[target_col] = pd.Series(dtype="float64")
        return df

    max_date = df["date"].max()

    # Build left lookup table with target dates
    left = pd.DataFrame(
        {
            "_orig_idx": np.arange(len(df)),
            "orig_date": df["date"],
            "target_date": df["date"] + pd.Timedelta(days=horizon_days),
            "orig_close": df["close"],
        }
    )

    # Build right lookup table with available trading days and close prices
    right = pd.DataFrame(
        {
            "match_date": df["date"],
            "forward_close": df["close"],
        }
    )

    # As-of merge matching the latest close on or before target_date within tolerance
    merged = pd.merge_asof(
        left.sort_values("target_date"),
        right.sort_values("match_date"),
        left_on="target_date",
        right_on="match_date",
        direction="backward",
        tolerance=pd.Timedelta(days=tolerance_days),
    )

    # If target_date extends beyond the end of our dataset, the future hasn't occurred yet -> NaN
    merged.loc[merged["target_date"] > max_date, "forward_close"] = np.nan

    # Target match date must be strictly after the original date
    merged.loc[merged["match_date"] <= merged["orig_date"], "forward_close"] = np.nan

    # Restore original row ordering
    merged = merged.sort_values("_orig_idx").reset_index(drop=True)

    # Calculate simple forward return: close(t+N) / close(t) - 1
    forward_return = (merged["forward_close"] / merged["orig_close"]) - 1.0
    df[target_col] = forward_return.astype("float64").values

    return df


def save_processed(df: pd.DataFrame, asset: str) -> None:
    """Save processed DataFrame to data/processed/<asset>.parquet using an atomic write."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    temp_path = os.path.join(PROCESSED_DIR, f"{asset}.parquet.tmp")
    final_path = os.path.join(PROCESSED_DIR, f"{asset}.parquet")
    df.to_parquet(temp_path, index=False)
    os.replace(temp_path, final_path)


def process_asset(asset: str) -> pd.DataFrame:
    """Process a single asset from raw parquet to processed parquet.

    1. Load raw data.
    2. Attach asset metadata.
    3. Clean and normalize dates and dtypes.
    4. Detect calendar gaps.
    5. Calculate log returns.
    6. Calculate simple forward returns for each horizon in HORIZONS_DAYS.
    7. Atomically save to data/processed/<asset>.parquet.

    Returns:
        The processed pd.DataFrame.
    """
    df = load_raw(asset)

    # Attach asset metadata
    if "asset" not in df.columns:
        df["asset"] = asset
    if "asset_class" not in df.columns:
        df["asset_class"] = "crypto" if asset in CRYPTO_ASSETS else "equity"

    # Clean data
    df = clean(df)

    # Check for calendar gaps
    find_gaps(df, asset_class=df["asset_class"].iloc[0] if "asset_class" in df.columns else None)

    # Compute daily log return
    df = add_log_return(df)

    # Compute forward-return targets for each horizon
    for horizon in HORIZONS_DAYS:
        df = add_forward_return(df, horizon_days=horizon)

    # Atomically save processed data
    save_processed(df, asset)

    return df


def main() -> None:
    """Process all assets in CRYPTO_ASSETS and EQUITY_ASSETS and print summary metrics."""
    all_assets = CRYPTO_ASSETS + EQUITY_ASSETS
    print(f"Starting preprocessing for {len(all_assets)} assets...\n")

    for asset in all_assets:
        df = process_asset(asset)
        asset_class = "crypto" if asset in CRYPTO_ASSETS else "equity"
        gaps = find_gaps(df, asset_class=asset_class)

        min_date = df["date"].min().strftime("%Y-%m-%d")
        max_date = df["date"].max().strftime("%Y-%m-%d")
        nan_log_return = df["log_return"].isna().sum()

        print(f"=== Asset: {asset} ({asset_class}) ===")
        print(f"  Rows: {len(df):,}")
        print(f"  Date range: {min_date} to {max_date}")
        print(f"  NaN count (log_return): {nan_log_return}")
        for horizon in HORIZONS_DAYS:
            target_col = f"target_{horizon}d"
            nan_target = df[target_col].isna().sum()
            print(f"  NaN count ({target_col}): {nan_target}")
        if asset_class == "crypto":
            print(f"  Crypto gaps: {len(gaps)} gap(s) found" if gaps else "  Crypto gaps: None (0 gaps)")
        print()


if __name__ == "__main__":
    main()
