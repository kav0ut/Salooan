import os
from typing import List, Optional

import pandas as pd

from src.config import (
    CRYPTO_ASSETS,
    EQUITY_ASSETS,
    FEATURE_COLUMNS,
    HORIZONS_DAYS,
    PROCESSED_DIR,
    VOLATILITY_COL,
    VOLATILITY_WINDOW,
    VOLUME_COL,
    VOLUME_WINDOW,
)


def load_processed(asset: str) -> pd.DataFrame:
    file_path = os.path.join(PROCESSED_DIR, f"{asset}.parquet")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Processed file not found: {file_path}")
    return pd.read_parquet(file_path)


def add_volatility(
    df: pd.DataFrame,
    window: int = VOLATILITY_WINDOW,
    col_name: str = VOLATILITY_COL,
) -> pd.DataFrame:
    df = df.copy()
    df[col_name] = (
        df["log_return"]
        .rolling(window=window, min_periods=window)
        .std(ddof=1)
        .astype("float64")
    )
    return df


def add_dollar_volume_avg(
    df: pd.DataFrame,
    window: int = VOLUME_WINDOW,
    col_name: str = VOLUME_COL,
) -> pd.DataFrame:
    df = df.copy()
    dollar_vol = df["close"] * df["volume"]
    df[col_name] = (
        dollar_vol
        .rolling(window=window, min_periods=window)
        .mean()
        .astype("float64")
    )
    return df


def build_asset_features(
    df: pd.DataFrame,
    vol_window: int = VOLATILITY_WINDOW,
    vol_col: str = VOLATILITY_COL,
    volume_window: int = VOLUME_WINDOW,
    volume_col: str = VOLUME_COL,
) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)
    df = add_volatility(df, window=vol_window, col_name=vol_col)
    df = add_dollar_volume_avg(df, window=volume_window, col_name=volume_col)
    return df


def build_feature_table(assets: Optional[List[str]] = None) -> pd.DataFrame:
    if assets is None:
        assets = CRYPTO_ASSETS + EQUITY_ASSETS

    dfs = []
    for asset in assets:
        df_asset = load_processed(asset)
        df_feat = build_asset_features(df_asset)
        dfs.append(df_feat)

    target_cols = [f"target_{h}d" for h in HORIZONS_DAYS]
    feature_table_cols = ["asset", "asset_class", "date"] + FEATURE_COLUMNS + target_cols

    if not dfs:
        return pd.DataFrame(columns=feature_table_cols)

    combined = pd.concat(dfs, ignore_index=True)

    # Keep only the columns specified in design decisions
    cols_to_keep = [col for col in feature_table_cols if col in combined.columns]
    combined = combined[cols_to_keep]

    # Drop only rows with NaN in FEATURE_COLUMNS (incomplete rolling windows)
    combined = combined.dropna(subset=FEATURE_COLUMNS)

    # Sort chronologically by asset, then date
    combined = combined.sort_values(by=["asset", "date"]).reset_index(drop=True)

    return combined


def save_features(df: pd.DataFrame, dest_path: Optional[str] = None) -> None:
    if dest_path is None:
        dest_path = os.path.join(PROCESSED_DIR, "features.parquet")
    dest_dir = os.path.dirname(dest_path)
    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)
    temp_path = f"{dest_path}.tmp"
    df.to_parquet(temp_path, index=False)
    os.replace(temp_path, dest_path)


def main() -> None:
    print("Building feature table...")
    features_df = build_feature_table()
    save_features(features_df)
    print(f"Saved feature table with {len(features_df)} rows to {os.path.join(PROCESSED_DIR, 'features.parquet')}\n")

    target_cols = [f"target_{h}d" for h in HORIZONS_DAYS]

    # Group by asset and print summary metrics
    print("=" * 70)
    print("FEATURE TABLE SUMMARY PER ASSET")
    print("=" * 70)

    for asset, group in features_df.groupby("asset", sort=False):
        asset_class = group["asset_class"].iloc[0] if "asset_class" in group.columns else "unknown"
        min_date = group["date"].min()
        max_date = group["date"].max()
        row_count = len(group)

        # Count NaN targets
        nan_target_counts = {t_col: int(group[t_col].isna().sum()) for t_col in target_cols if t_col in group.columns}
        nan_target_str = ", ".join([f"{col}: {cnt}" for col, cnt in nan_target_counts.items()])

        print(f"\nAsset: {asset} ({asset_class})")
        print(f"  Rows: {row_count}")
        print(f"  Date range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
        print(f"  NaN targets: {nan_target_str}")
        print("  Feature column statistics:")
        desc = group[FEATURE_COLUMNS].describe().to_string()
        indented_desc = "\n".join("    " + line for line in desc.split("\n"))
        print(indented_desc)

    print("\n" + "=" * 70)
    print("Feature generation complete.")


if __name__ == "__main__":
    main()