import copy
import json
import os
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

from src.refresh import (
    acquire_lock,
    expected_latest_candle,
    latest_saved_candle,
    read_status,
    refresh_state,
    release_lock,
    write_status,
)


def test_expected_latest_candle_midnight():
    # 23:59 UTC on 2026-10-23 -> yesterday is 2026-10-22
    t_before_midnight = datetime(2026, 10, 23, 23, 59, 0, tzinfo=timezone.utc)
    assert expected_latest_candle(t_before_midnight) == date(2026, 10, 22)

    # 00:01 UTC on 2026-10-24 -> yesterday is 2026-10-23
    t_after_midnight = datetime(2026, 10, 24, 0, 1, 0, tzinfo=timezone.utc)
    assert expected_latest_candle(t_after_midnight) == date(2026, 10, 23)

    # Naive datetime raises ValueError
    naive_dt = datetime(2026, 10, 23, 12, 0, 0)
    with pytest.raises(ValueError, match="timezone-aware"):
        expected_latest_candle(naive_dt)


def test_latest_saved_candle(tmp_path):
    assets = ["BTC", "ETH", "ADA"]

    # Missing file returns None
    assert latest_saved_candle(tmp_path, assets) is None

    # Write files with date column
    df_btc = pd.DataFrame({"date": [date(2026, 10, 20), date(2026, 10, 22)]})
    df_eth = pd.DataFrame({"date": [date(2026, 10, 18), date(2026, 10, 19)]})
    df_ada = pd.DataFrame({"date": [date(2026, 10, 21)]})

    df_btc.to_parquet(tmp_path / "BTC.parquet")
    df_eth.to_parquet(tmp_path / "ETH.parquet")
    df_ada.to_parquet(tmp_path / "ADA.parquet")

    # Oldest of [2026-10-22, 2026-10-19, 2026-10-21] is 2026-10-19
    assert latest_saved_candle(tmp_path, assets) == date(2026, 10, 19)

    # Empty asset list returns None
    assert latest_saved_candle(tmp_path, []) is None


def test_latest_saved_candle_open_time(tmp_path):
    # Test support for raw open_time column in milliseconds
    assets = ["BTC", "ETH"]
    t1 = int(datetime(2026, 10, 20, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
    t2 = int(datetime(2026, 10, 21, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
    t3 = int(datetime(2026, 10, 18, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)

    pd.DataFrame({"open_time": [t1, t2]}).to_parquet(tmp_path / "BTC.parquet")
    pd.DataFrame({"open_time": [t3]}).to_parquet(tmp_path / "ETH.parquet")

    # Oldest is 2026-10-18
    assert latest_saved_candle(tmp_path, assets) == date(2026, 10, 18)


def test_read_write_status(tmp_path):
    status_path = tmp_path / "refresh_status.json"

    # Missing status file returns empty dict
    assert read_status(status_path) == {}

    # Round trip
    status = {
        "last_attempt_utc": "2026-10-23T12:00:00+00:00",
        "last_success_utc": "2026-10-23T12:00:00+00:00",
        "rows_added": {"BTC": 1, "ETH": 2},
    }
    write_status(status_path, status)
    assert read_status(status_path) == status

    # Atomic write leaves no temporary file
    assert not (tmp_path / "refresh_status.json.tmp").exists()

    # Corrupt JSON is treated as empty dict
    with open(status_path, "w", encoding="utf-8") as f:
        f.write("{corrupt json content")
    assert read_status(status_path) == {}


def test_refresh_state():
    now = datetime(2026, 10, 24, 12, 0, 0, tzinfo=timezone.utc)
    # expected latest candle for now is 2026-10-23

    # 1. Naive datetime raises ValueError
    with pytest.raises(ValueError, match="timezone-aware"):
        refresh_state(datetime(2026, 10, 24, 12, 0, 0), date(2026, 10, 20), {}, 60)

    # 2. None latest saved gives no_data and allowed
    st_none = refresh_state(now, None, {}, 60)
    assert st_none["allowed"] is True
    assert st_none["reason"] == "no_data"

    # 3. Up to date: latest candle equals yesterday (2026-10-23)
    st_uptodate = refresh_state(now, date(2026, 10, 23), {}, 60)
    assert st_uptodate["allowed"] is False
    assert st_uptodate["reason"] == "up_to_date"

    # 4. Stale candle with no status is allowed
    stale_date = date(2026, 10, 20)
    st_nostatus = refresh_state(now, stale_date, {}, 60)
    assert st_nostatus["allowed"] is True
    assert st_nostatus["reason"] == "ok"

    # 5. Stale candle with an attempt 10 minutes ago -> cooldown with right next_allowed_utc
    attempt_10m = now - timedelta(minutes=10)
    st_10m = refresh_state(now, stale_date, {"last_attempt_utc": attempt_10m.isoformat()}, 60)
    assert st_10m["allowed"] is False
    assert st_10m["reason"] == "cooldown"
    assert st_10m["next_allowed_utc"] == attempt_10m + timedelta(minutes=60)

    # 6. Attempt 2 hours ago is allowed
    attempt_2h = now - timedelta(hours=2)
    st_2h = refresh_state(now, stale_date, {"last_attempt_utc": attempt_2h.isoformat()}, 60)
    assert st_2h["allowed"] is True
    assert st_2h["reason"] == "ok"

    # 7. FAILED attempt also triggers the cooldown
    st_failed = refresh_state(
        now,
        stale_date,
        {"last_attempt_utc": attempt_10m.isoformat(), "last_error": "API error"},
        60,
    )
    assert st_failed["allowed"] is False
    assert st_failed["reason"] == "cooldown"
    assert st_failed["next_allowed_utc"] == attempt_10m + timedelta(minutes=60)

    # 8. Exactly at the interval boundary is allowed
    attempt_boundary = now - timedelta(minutes=60)
    st_boundary = refresh_state(now, stale_date, {"last_attempt_utc": attempt_boundary.isoformat()}, 60)
    assert st_boundary["allowed"] is True
    assert st_boundary["reason"] == "ok"


def test_lock_acquire_and_release(tmp_path):
    lock_path = tmp_path / "refresh.lock"
    now = datetime(2026, 10, 24, 12, 0, 0, tzinfo=timezone.utc)

    # Naive now raises ValueError
    with pytest.raises(ValueError, match="timezone-aware"):
        acquire_lock(lock_path, datetime(2026, 10, 24, 12, 0, 0), 15)

    # 1. Acquire succeeds once
    assert acquire_lock(lock_path, now, 15) is True

    # 2. A second acquire fails
    assert acquire_lock(lock_path, now, 15) is False

    # 3. Release allows a new acquire
    release_lock(lock_path)
    assert acquire_lock(lock_path, now, 15) is True

    # 4. A fresh lock (5 min later) is not replaced
    t_fresh = now + timedelta(minutes=5)
    assert acquire_lock(lock_path, t_fresh, 15) is False

    # 5. A stale lock (20 min later) is replaced
    t_stale = now + timedelta(minutes=20)
    assert acquire_lock(lock_path, t_stale, 15) is True


def test_no_mutation_of_inputs(tmp_path):
    now = datetime(2026, 10, 24, 12, 0, 0, tzinfo=timezone.utc)
    status = {"last_attempt_utc": (now - timedelta(minutes=10)).isoformat(), "key": "val"}
    status_copy = copy.deepcopy(status)

    refresh_state(now, date(2026, 10, 20), status, 60)
    assert status == status_copy

    write_status(tmp_path / "test.json", status)
    assert status == status_copy

    assets = ["BTCUSDT", "ETHUSDT"]
    assets_copy = list(assets)
    latest_saved_candle(tmp_path, assets)
    assert assets == assets_copy


def test_run_refresh_success(tmp_path):
    status_path = tmp_path / "status.json"
    lock_path = tmp_path / "refresh.lock"
    now = datetime(2026, 10, 24, 12, 0, 0, tzinfo=timezone.utc)

    rebuild_called = 0

    def fake_ingest():
        return {"BTC": 5, "ETH": 3}, {}

    def fake_rebuild():
        nonlocal rebuild_called
        rebuild_called += 1

    res = from_refresh_module().run_refresh(
        ingest=fake_ingest,
        rebuild=fake_rebuild,
        now=now,
        status_path=status_path,
        lock_path=lock_path,
    )

    assert res["status"] == "ok"
    assert res["rows_added"] == {"BTC": 5, "ETH": 3}
    assert res["errors"] == {}
    assert rebuild_called == 1
    # Lock is released
    assert not lock_path.exists()

    status = read_status(status_path)
    assert status["last_attempt_utc"] == now.isoformat()
    assert status["last_success_utc"] == now.isoformat()
    assert status["rows_added"] == {"BTC": 5, "ETH": 3}
    assert "last_error" not in status


def test_run_refresh_nothing_new(tmp_path):
    status_path = tmp_path / "status.json"
    lock_path = tmp_path / "refresh.lock"
    now = datetime(2026, 10, 24, 12, 0, 0, tzinfo=timezone.utc)

    rebuild_called = 0

    def fake_ingest():
        return {"BTC": 0, "ETH": 0}, {}

    def fake_rebuild():
        nonlocal rebuild_called
        rebuild_called += 1

    res = from_refresh_module().run_refresh(
        ingest=fake_ingest,
        rebuild=fake_rebuild,
        now=now,
        status_path=status_path,
        lock_path=lock_path,
    )

    assert res["status"] == "nothing_new"
    assert rebuild_called == 0
    assert not lock_path.exists()

    status = read_status(status_path)
    assert status["last_attempt_utc"] == now.isoformat()


def test_run_refresh_one_asset_fails(tmp_path):
    status_path = tmp_path / "status.json"
    lock_path = tmp_path / "refresh.lock"
    now = datetime(2026, 10, 24, 12, 0, 0, tzinfo=timezone.utc)

    rebuild_called = 0

    def fake_ingest():
        return {"BTC": 5}, {"ETH": "Network timeout"}

    def fake_rebuild():
        nonlocal rebuild_called
        rebuild_called += 1

    res = from_refresh_module().run_refresh(
        ingest=fake_ingest,
        rebuild=fake_rebuild,
        now=now,
        status_path=status_path,
        lock_path=lock_path,
    )

    assert res["status"] == "partial"
    assert res["rows_added"] == {"BTC": 5}
    assert "ETH" in res["errors"]
    assert rebuild_called == 1
    assert not lock_path.exists()

    status = read_status(status_path)
    assert status["last_attempt_utc"] == now.isoformat()
    assert "last_success_utc" not in status
    assert "ETH: Network timeout" in status["last_error"]


def test_run_refresh_every_asset_fails(tmp_path):
    status_path = tmp_path / "status.json"
    lock_path = tmp_path / "refresh.lock"
    now = datetime(2026, 10, 24, 12, 0, 0, tzinfo=timezone.utc)

    # Pre-populate status with existing fields
    write_status(status_path, {"last_success_utc": "2026-10-20T00:00:00+00:00", "other_field": 42})

    rebuild_called = 0

    def fake_ingest():
        return {}, {"BTC": "Error 1", "ETH": "Error 2"}

    def fake_rebuild():
        nonlocal rebuild_called
        rebuild_called += 1

    res = from_refresh_module().run_refresh(
        ingest=fake_ingest,
        rebuild=fake_rebuild,
        now=now,
        status_path=status_path,
        lock_path=lock_path,
    )

    assert res["status"] == "failed"
    assert rebuild_called == 0
    assert not lock_path.exists()

    status = read_status(status_path)
    assert status["last_attempt_utc"] == now.isoformat()
    assert status["last_success_utc"] == "2026-10-20T00:00:00+00:00"
    assert status["other_field"] == 42
    assert "Error 1" in status["last_error"]


def test_run_refresh_rebuild_raises(tmp_path):
    status_path = tmp_path / "status.json"
    lock_path = tmp_path / "refresh.lock"
    now = datetime(2026, 10, 24, 12, 0, 0, tzinfo=timezone.utc)

    def fake_ingest():
        return {"BTC": 5}, {}

    def fake_rebuild():
        raise RuntimeError("Disk write failed")

    res = from_refresh_module().run_refresh(
        ingest=fake_ingest,
        rebuild=fake_rebuild,
        now=now,
        status_path=status_path,
        lock_path=lock_path,
    )

    assert res["status"] == "failed"
    assert "Disk write failed" in res["errors"]["rebuild"]
    assert not lock_path.exists()

    status = read_status(status_path)
    assert status["last_attempt_utc"] == now.isoformat()
    assert "Disk write failed" in status["last_error"]


def test_run_refresh_already_running_and_stale_replacement(tmp_path):
    status_path = tmp_path / "status.json"
    lock_path = tmp_path / "refresh.lock"
    now = datetime(2026, 10, 24, 12, 0, 0, tzinfo=timezone.utc)

    # 1. Fresh lock held by another process (5 mins old)
    acquire_lock(lock_path, now - timedelta(minutes=5), 15)

    ingest_called = False
    rebuild_called = False

    def fake_ingest():
        nonlocal ingest_called
        ingest_called = True
        return {"BTC": 1}, {}

    def fake_rebuild():
        nonlocal rebuild_called
        rebuild_called = True

    res = from_refresh_module().run_refresh(
        ingest=fake_ingest,
        rebuild=fake_rebuild,
        now=now,
        status_path=status_path,
        lock_path=lock_path,
    )

    assert res["status"] == "already_running"
    assert not ingest_called
    assert not rebuild_called

    # 2. Stale lock (20 mins old) is replaced
    release_lock(lock_path)
    acquire_lock(lock_path, now - timedelta(minutes=20), 15)

    res_stale = from_refresh_module().run_refresh(
        ingest=fake_ingest,
        rebuild=fake_rebuild,
        now=now,
        status_path=status_path,
        lock_path=lock_path,
    )
    assert res_stale["status"] == "ok"
    assert ingest_called
    assert rebuild_called


def test_refresh_module_has_no_training_or_eval_references():
    import inspect
    import src.refresh as refresh_mod

    source = inspect.getsource(refresh_mod)
    assert "train_model" not in source
    assert "save_bundle" not in source
    assert "evaluate_config" not in source
    assert "evaluate_grid" not in source


def test_update_asset_returns_row_count(tmp_path, monkeypatch):
    import pandas as pd
    from src.data_pull import update_asset

    monkeypatch.setattr("src.data_pull.RAW_DIR", str(tmp_path))

    # Mock fetch_binance to return 2 rows
    sample_df = pd.DataFrame({
        "open_time": [1000, 2000],
        "open": [1.0, 2.0],
        "high": [1.1, 2.1],
        "low": [0.9, 1.9],
        "close": [1.05, 2.05],
        "volume": [100.0, 200.0],
        "close_time": [1999, 2999],
        "quote_asset_volume": [105.0, 410.0],
        "number_of_trades": [10, 20],
        "taker_buy_base_asset_volume": [50.0, 100.0],
        "taker_buy_quote_asset_volume": [52.5, 205.0],
        "ignore": [0, 0],
    })

    monkeypatch.setattr("src.data_pull.fetch_binance", lambda symbol, start: sample_df)
    n_added = update_asset("BTCUSDT")
    assert n_added == 2

    # Second call with empty DataFrame returns 0
    empty_df = pd.DataFrame(columns=sample_df.columns)
    monkeypatch.setattr("src.data_pull.fetch_binance", lambda symbol, start: empty_df)
    n_added_second = update_asset("BTCUSDT")
    assert n_added_second == 0


def from_refresh_module():
    import src.refresh as mod
    return mod

