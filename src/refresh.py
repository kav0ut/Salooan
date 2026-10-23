import json
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd


def expected_latest_candle(now: datetime) -> date:
    if now.tzinfo is None or now.tzinfo.utcoffset(now) is None:
        raise ValueError("now must be a timezone-aware datetime")
    now_utc = now.astimezone(timezone.utc)
    return now_utc.date() - timedelta(days=1)


def latest_saved_candle(raw_dir: Union[str, Path], assets: List[str]) -> Optional[date]:
    if not assets:
        return None

    str_raw_dir = str(raw_dir)
    latest_dates: List[date] = []

    for asset in assets:
        file_path = os.path.join(str_raw_dir, f"{asset}.parquet")
        if not os.path.exists(file_path):
            return None

        try:
            # Check schema to read only the date or open_time column
            import pyarrow.parquet as pq
            schema_names = pq.read_schema(file_path).names

            if "date" in schema_names:
                df = pd.read_parquet(file_path, columns=["date"])
                if df.empty or df["date"].dropna().empty:
                    return None
                val = df["date"].dropna().max()
                if isinstance(val, (datetime, pd.Timestamp)):
                    d = val.date()
                elif isinstance(val, date):
                    d = val
                else:
                    d = pd.to_datetime(val).date()
                latest_dates.append(d)
            elif "open_time" in schema_names:
                df = pd.read_parquet(file_path, columns=["open_time"])
                if df.empty or df["open_time"].dropna().empty:
                    return None
                val = df["open_time"].dropna().max()
                d = pd.to_datetime(val, unit="ms", utc=True).date()
                latest_dates.append(d)
            else:
                return None
        except Exception:
            return None

    if not latest_dates:
        return None

    # Return the oldest of the per-asset latest dates
    return min(latest_dates)


def read_status(path: Union[str, Path]) -> Dict[str, Any]:
    str_path = str(path)
    if not os.path.exists(str_path):
        return {}

    try:
        with open(str_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
        return {}
    except Exception:
        return {}


def write_status(path: Union[str, Path], status: Dict[str, Any]) -> None:
    str_path = str(path)
    dir_name = os.path.dirname(os.path.abspath(str_path))
    os.makedirs(dir_name, exist_ok=True)

    temp_path = f"{str_path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2, default=str)
    os.replace(temp_path, str_path)


def refresh_state(
    now: datetime,
    latest_saved: Optional[Union[date, str, datetime]],
    status: Dict[str, Any],
    min_interval_minutes: int,
) -> Dict[str, Any]:
    if now.tzinfo is None or now.tzinfo.utcoffset(now) is None:
        raise ValueError("now must be a timezone-aware datetime")

    now_utc = now.astimezone(timezone.utc)

    # Rule 1: No saved data allows refresh immediately
    if latest_saved is None:
        return {
            "allowed": True,
            "reason": "no_data",
            "message": "No saved data found; refresh allowed",
        }

    # Normalize latest_saved to date
    if isinstance(latest_saved, str):
        latest_date = pd.to_datetime(latest_saved).date()
    elif isinstance(latest_saved, (datetime, pd.Timestamp)):
        latest_date = latest_saved.date()
    else:
        latest_date = latest_saved

    expected = expected_latest_candle(now_utc)

    # Rule 2: Already up to date
    if latest_date >= expected:
        return {
            "allowed": False,
            "reason": "up_to_date",
            "message": f"Data is up to date as of {latest_date}",
        }

    # Rule 3: Cooldown after last attempt (success or failure)
    last_attempt_raw = status.get("last_attempt_utc")
    if last_attempt_raw is not None:
        try:
            if isinstance(last_attempt_raw, (datetime, pd.Timestamp)):
                last_attempt_dt = last_attempt_raw
            else:
                last_attempt_dt = datetime.fromisoformat(str(last_attempt_raw))

            if last_attempt_dt.tzinfo is None or last_attempt_dt.tzinfo.utcoffset(last_attempt_dt) is None:
                last_attempt_dt = last_attempt_dt.replace(tzinfo=timezone.utc)
            else:
                last_attempt_dt = last_attempt_dt.astimezone(timezone.utc)

            next_allowed_utc = last_attempt_dt + timedelta(minutes=min_interval_minutes)

            # Strictly less than the interval boundary triggers cooldown
            if now_utc < next_allowed_utc:
                return {
                    "allowed": False,
                    "reason": "cooldown",
                    "next_allowed_utc": next_allowed_utc,
                    "message": f"Next refresh possible at {next_allowed_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC",
                }
        except Exception:
            pass

    return {
        "allowed": True,
        "reason": "ok",
        "message": "Refresh allowed",
    }


def acquire_lock(path: Union[str, Path], now: datetime, timeout_minutes: int) -> bool:
    if now.tzinfo is None or now.tzinfo.utcoffset(now) is None:
        raise ValueError("now must be a timezone-aware datetime")

    str_path = str(path)
    now_utc = now.astimezone(timezone.utc)
    dir_name = os.path.dirname(os.path.abspath(str_path))
    os.makedirs(dir_name, exist_ok=True)

    # First attempt an exclusive creation
    try:
        with open(str_path, "x", encoding="utf-8") as f:
            json.dump({"acquired_at_utc": now_utc.isoformat(), "pid": os.getpid()}, f)
        return True
    except FileExistsError:
        pass

    # Lock file already exists: inspect age to see if it is stale
    try:
        with open(str_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        acquired_at_raw = data.get("acquired_at_utc")
        if acquired_at_raw:
            acquired_at = datetime.fromisoformat(str(acquired_at_raw))
            if acquired_at.tzinfo is None or acquired_at.tzinfo.utcoffset(acquired_at) is None:
                acquired_at = acquired_at.replace(tzinfo=timezone.utc)
            else:
                acquired_at = acquired_at.astimezone(timezone.utc)
        else:
            mtime = os.path.getmtime(str_path)
            acquired_at = datetime.fromtimestamp(mtime, tz=timezone.utc)
    except Exception:
        # Unreadable or corrupt lock file is considered stale
        try:
            mtime = os.path.getmtime(str_path)
            acquired_at = datetime.fromtimestamp(mtime, tz=timezone.utc)
        except Exception:
            acquired_at = now_utc - timedelta(minutes=timeout_minutes + 1)

    # If lock is still fresh, acquire fails
    if (now_utc - acquired_at) < timedelta(minutes=timeout_minutes):
        return False

    # Lock is stale: remove and retry exclusive creation
    try:
        os.remove(str_path)
    except OSError:
        return False

    try:
        with open(str_path, "x", encoding="utf-8") as f:
            json.dump({"acquired_at_utc": now_utc.isoformat(), "pid": os.getpid()}, f)
        return True
    except FileExistsError:
        return False


def release_lock(path: Union[str, Path]) -> None:
    str_path = str(path)
    try:
        if os.path.exists(str_path):
            os.remove(str_path)
    except OSError:
        pass


def run_refresh(
    ingest: Callable[[], Tuple[Dict[str, int], Dict[str, str]]],
    rebuild: Callable[[], None],
    now: datetime,
    status_path: Union[str, Path],
    lock_path: Union[str, Path],
) -> Dict[str, Any]:
    if now.tzinfo is None or now.tzinfo.utcoffset(now) is None:
        raise ValueError("now must be a timezone-aware datetime")
    now_utc = now.astimezone(timezone.utc)

    from src.config import REFRESH_LOCK_TIMEOUT_MINUTES
    if not acquire_lock(lock_path, now_utc, REFRESH_LOCK_TIMEOUT_MINUTES):
        return {
            "status": "already_running",
            "rows_added": {},
            "errors": {},
        }

    try:
        status_data = read_status(status_path)
        status_data["last_attempt_utc"] = now_utc.isoformat()

        rows_added, errors = ingest()
        total_rows_added = sum(rows_added.values()) if rows_added else 0

        # Scenario 1: All assets failed during ingestion
        if len(errors) > 0 and total_rows_added == 0:
            status_data["last_error"] = "; ".join(f"{k}: {v}" for k, v in errors.items())
            write_status(status_path, status_data)
            return {
                "status": "failed",
                "rows_added": rows_added,
                "errors": errors,
            }

        # Scenario 2: Nothing new and no errors
        if len(errors) == 0 and total_rows_added == 0:
            status_data["rows_added"] = rows_added
            status_data.pop("last_error", None)
            write_status(status_path, status_data)
            return {
                "status": "nothing_new",
                "rows_added": rows_added,
                "errors": {},
            }

        # Scenario 3: At least one asset gained rows -> run rebuild
        try:
            rebuild()
        except Exception as e:
            status_data["last_error"] = f"Rebuild failed: {e}"
            status_data["rows_added"] = rows_added
            write_status(status_path, status_data)
            all_errors = dict(errors)
            all_errors["rebuild"] = str(e)
            return {
                "status": "failed",
                "rows_added": rows_added,
                "errors": all_errors,
            }

        # Successful rebuild
        if len(errors) == 0:
            status_data["last_success_utc"] = now_utc.isoformat()
            status_data.pop("last_error", None)
            status_data["rows_added"] = rows_added
            write_status(status_path, status_data)
            return {
                "status": "ok",
                "rows_added": rows_added,
                "errors": {},
            }
        else:
            status_data["last_error"] = "; ".join(f"{k}: {v}" for k, v in errors.items())
            status_data["rows_added"] = rows_added
            write_status(status_path, status_data)
            return {
                "status": "partial",
                "rows_added": rows_added,
                "errors": errors,
            }
    finally:
        release_lock(lock_path)


def default_ingest() -> Tuple[Dict[str, int], Dict[str, str]]:
    from src.config import CRYPTO_ASSETS, EQUITY_ASSETS
    from src.data_pull import update_asset

    rows_added: Dict[str, int] = {}
    errors: Dict[str, str] = {}

    for asset in CRYPTO_ASSETS + EQUITY_ASSETS:
        try:
            n = update_asset(asset)
            rows_added[asset] = n
        except Exception as e:
            errors[asset] = str(e)

    return rows_added, errors


def default_rebuild() -> None:
    from src.config import CRYPTO_ASSETS, EQUITY_ASSETS
    from src.preprocess import process_asset
    from src.features import build_feature_table, save_features

    for asset in CRYPTO_ASSETS + EQUITY_ASSETS:
        process_asset(asset)

    features_df = build_feature_table()
    save_features(features_df)


def refresh_data(now: datetime) -> Dict[str, Any]:
    from src.config import REFRESH_STATUS_PATH, REFRESH_LOCK_PATH

    return run_refresh(
        ingest=default_ingest,
        rebuild=default_rebuild,
        now=now,
        status_path=REFRESH_STATUS_PATH,
        lock_path=REFRESH_LOCK_PATH,
    )

