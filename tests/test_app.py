import json
import os
import pytest
from streamlit.testing.v1 import AppTest

from src.config import (
    CONFIDENCE_LEVELS,
    CRYPTO_ASSETS,
    DEFAULT_CONFIDENCE,
    DEFAULT_HORIZON_DAYS,
    DEMO_COLLATERAL_VALUE,
    EQUITY_ASSETS,
    FEATURES_PATH,
    HORIZONS_DAYS,
    MODELS_DIR,
    RESULTS_PATH,
)
from src.train import config_key, model_path

missing_prereqs = []
for h in HORIZONS_DAYS:
    for c in CONFIDENCE_LEVELS:
        p = model_path(h, c)
        if not os.path.exists(p):
            missing_prereqs.append(f"Model bundle missing at {p}")

if not os.path.exists(FEATURES_PATH):
    missing_prereqs.append(f"Features file missing at {FEATURES_PATH}")
if not os.path.exists(RESULTS_PATH):
    missing_prereqs.append(f"Results file missing at {RESULTS_PATH}")

skip_if_prereqs_missing = pytest.mark.skipif(
    bool(missing_prereqs),
    reason="; ".join(missing_prereqs),
)


@skip_if_prereqs_missing
def test_app_smoke() -> None:
    at = AppTest.from_file("app.py", default_timeout=60).run()
    assert len(at.exception) == 0

    expected_assets = CRYPTO_ASSETS + EQUITY_ASSETS
    assert list(at.sidebar.selectbox[0].options) == expected_assets
    assert float(at.sidebar.number_input[0].value) == float(DEMO_COLLATERAL_VALUE)

    assert len(at.metric) >= 3
    metric_labels = [m.label for m in at.metric]
    assert "Recommended Haircut" in metric_labels
    assert "Maximum Loan Amount" in metric_labels

    at.sidebar.selectbox[0].select("DOGEUSDT").run()
    assert len(at.exception) == 0
    doge_haircut = float(at.metric[0].value.replace("%", "").strip())

    at.sidebar.selectbox[0].select("SPY").run()
    assert len(at.exception) == 0
    spy_haircut = float(at.metric[0].value.replace("%", "").strip())

    assert doge_haircut > spy_haircut


@skip_if_prereqs_missing
def test_app_runs_all_six_pairs_and_headline_label_mentions_selection() -> None:
    at = AppTest.from_file("app.py", default_timeout=60).run()
    assert len(at.exception) == 0

    for h in HORIZONS_DAYS:
        at.sidebar.radio[0].set_value(h).run()
        assert len(at.exception) == 0
        for c in CONFIDENCE_LEVELS:
            at.sidebar.select_slider[0].set_value(c).run()
            assert len(at.exception) == 0

            # Headline label mentions selected horizon and confidence
            metric_labels = [m.label for m in at.metric]
            conf_pct = round(c * 100)
            expected_metric_label = f"Predicted worst-case {h}-day drop ({conf_pct}% confidence)"
            assert expected_metric_label in metric_labels


@skip_if_prereqs_missing
def test_app_evaluation_tab_shows_selected_pair_from_results() -> None:
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        results = json.load(f)

    at = AppTest.from_file("app.py", default_timeout=60).run()
    assert len(at.exception) == 0

    # Select horizon 14 and confidence 0.99
    at.sidebar.radio[0].set_value(14).run()
    at.sidebar.select_slider[0].set_value(0.99).run()
    assert len(at.exception) == 0

    assert len(at.dataframe) >= 1
    summary_df = at.dataframe[0].value

    expected_methods = {"Model", "Historical", "Static"}
    assert set(summary_df["Method"]) == expected_methods

    cfg_14_99 = results["configs"]["h14_c99"]
    for _, row in summary_df.iterrows():
        method_key = row["Method"].lower()
        score = cfg_14_99["methods"][method_key]["overall"]
        assert row["Test Rows"] == score["n_rows"]
        assert row["Pinball Loss"] == f"{score['pinball_loss']:.6f}"
        assert row["Breach Rate"] == f"{score['breach_rate']:.2%}"
        assert row["Mean Haircut"] == f"{score['mean_haircut']:.2%}"


@skip_if_prereqs_missing
def test_app_friendly_failure_on_missing_model(monkeypatch) -> None:
    import streamlit as st
    st.cache_resource.clear()

    import src.train as train
    orig_load_bundle = train.load_bundle

    def fake_load_bundle(horizon_days: int, confidence: float, **kwargs):
        if horizon_days == 14 and round(confidence * 100) == 99:
            raise FileNotFoundError("Simulated missing model bundle")
        return orig_load_bundle(horizon_days, confidence, **kwargs)

    monkeypatch.setattr("src.train.load_bundle", fake_load_bundle)

    try:
        at = AppTest.from_file("app.py", default_timeout=60).run()
        at.sidebar.radio[0].set_value(14).run()
        at.sidebar.select_slider[0].set_value(0.99).run()

        # Must not raise an unhandled exception
        assert len(at.exception) == 0
        # Friendly error message appears mentioning python -m src.train
        assert len(at.error) >= 1
        assert any("python -m src.train" in err.value for err in at.error)
    finally:
        st.cache_resource.clear()


@skip_if_prereqs_missing
def test_app_refresh_up_to_date(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.refresh.refresh_state",
        lambda *a, **k: {"allowed": False, "reason": "up_to_date", "message": "Data is up to date as of 2026-10-22"},
    )
    at = AppTest.from_file("app.py", default_timeout=60).run()
    assert len(at.exception) == 0
    assert len(at.sidebar.button) >= 1
    assert at.sidebar.button[0].disabled is True
    assert any("up to date" in c.value for c in at.sidebar.caption)


@skip_if_prereqs_missing
def test_app_refresh_stale_and_allowed(monkeypatch) -> None:
    called_count = 0

    def fake_refresh(now):
        nonlocal called_count
        called_count += 1
        return {"status": "ok", "rows_added": {"BTCUSDT": 1}, "errors": {}}

    monkeypatch.setattr("src.refresh.refresh_data", fake_refresh)
    monkeypatch.setattr(
        "src.refresh.refresh_state",
        lambda *a, **k: {"allowed": True, "reason": "ok", "message": "Refresh allowed"},
    )

    at = AppTest.from_file("app.py", default_timeout=60).run()
    assert len(at.exception) == 0
    assert at.sidebar.button[0].disabled is False

    at.sidebar.button[0].click().run()
    assert len(at.exception) == 0
    assert called_count == 1
    assert any("refreshed successfully" in s.value for s in at.success)


@skip_if_prereqs_missing
def test_app_refresh_cooldown(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.refresh.refresh_state",
        lambda *a, **k: {
            "allowed": False,
            "reason": "cooldown",
            "next_allowed_utc": "2026-10-24 15:30:00 UTC",
            "message": "Next refresh possible at 2026-10-24 15:30:00 UTC",
        },
    )
    at = AppTest.from_file("app.py", default_timeout=60).run()
    assert len(at.exception) == 0
    assert at.sidebar.button[0].disabled is True
    assert any("Next refresh possible at" in c.value for c in at.sidebar.caption)


@skip_if_prereqs_missing
def test_app_refresh_failed(monkeypatch) -> None:
    def fake_refresh_fail(now):
        return {"status": "failed", "rows_added": {}, "errors": {"BTCUSDT": "Connection error"}}

    monkeypatch.setattr("src.refresh.refresh_data", fake_refresh_fail)
    monkeypatch.setattr(
        "src.refresh.refresh_state",
        lambda *a, **k: {"allowed": True, "reason": "ok", "message": "Refresh allowed"},
    )

    at = AppTest.from_file("app.py", default_timeout=60).run()
    at.sidebar.button[0].click().run()
    assert len(at.exception) == 0
    assert any("Data refresh failed" in err.value for err in at.error)
    # App still renders calculator metrics
    assert any("Recommended Haircut" in m.label for m in at.metric)


@skip_if_prereqs_missing
def test_app_never_calls_train_model(monkeypatch) -> None:
    def raise_on_train(*a, **k):
        raise AssertionError("train_model should never be called by app")

    monkeypatch.setattr("src.train.train_model", raise_on_train)
    at = AppTest.from_file("app.py", default_timeout=60).run()
    assert len(at.exception) == 0
    assert any("Recommended Haircut" in m.label for m in at.metric)




