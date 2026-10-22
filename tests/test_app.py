import os
import pytest
from streamlit.testing.v1 import AppTest

from src.config import (
    CRYPTO_ASSETS,
    DEFAULT_CONFIDENCE,
    DEFAULT_HORIZON_DAYS,
    DEMO_COLLATERAL_VALUE,
    EQUITY_ASSETS,
    FEATURES_PATH,
    MODELS_DIR,
    RESULTS_PATH,
)

model_file = os.path.join(
    MODELS_DIR,
    f"quantile_h{DEFAULT_HORIZON_DAYS}_c{round(DEFAULT_CONFIDENCE * 100)}.joblib",
)

missing_prereqs = []
if not os.path.exists(model_file):
    missing_prereqs.append(f"Model bundle missing at {model_file}")
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
def test_app_evaluation_tab() -> None:
    at = AppTest.from_file("app.py", default_timeout=60).run()
    assert len(at.exception) == 0

    assert len(at.dataframe) >= 1
    summary_df = at.dataframe[0].value

    expected_methods = {"Model", "Historical", "Static"}
    assert set(summary_df["Method"]) == expected_methods

    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        import json
        results = json.load(f)

    pct = round(DEFAULT_CONFIDENCE * 100)
    config_key = f"h{DEFAULT_HORIZON_DAYS}_c{pct}"
    cfg = results["configs"][config_key]

    for _, row in summary_df.iterrows():
        method_key = row["Method"].lower()
        score = cfg["methods"][method_key]["overall"]
        assert row["Test Rows"] == score["n_rows"]
        assert row["Pinball Loss"] == f"{score['pinball_loss']:.6f}"
        assert row["Breach Rate"] == f"{score['breach_rate']:.2%}"
        assert row["Mean Haircut"] == f"{score['mean_haircut']:.2%}"

