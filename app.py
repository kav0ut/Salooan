import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

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
    PROCESSED_DIR,
    RAW_DIR,
    REFRESH_MIN_INTERVAL_MINUTES,
    REFRESH_STATUS_PATH,
    RESULTS_PATH,
    RETRAIN_SUGGEST_AFTER_DAYS,
)
from src.dashboard import (
    baseline_haircuts,
    calculator_result,
    evaluation_tables,
    latest_crossing,
    load_results,
    newer_data_days,
    risk_band,
    training_cutoff,
)
from src.refresh import (
    latest_saved_candle,
    read_status,
    refresh_data,
    refresh_state,
)
import src.train as train
from src.train import config_key, model_path

st.set_page_config(page_title="Salooan", layout="wide")

now = datetime.now(timezone.utc)

st.title("Salooan")
st.markdown(
    "Dynamic collateral haircut estimation: predicting worst-case price drops over a "
    "liquidation horizon so lenders can safely size cash loans against volatile collateral."
)

if not os.path.exists(FEATURES_PATH):
    st.error(f"Feature table not found at {FEATURES_PATH}. Please generate features with: python -m src.features")
    st.stop()


@st.cache_resource
def get_bundle(horizon_days: int, confidence: float) -> Dict[str, Any]:
    return train.load_bundle(horizon_days, confidence)


@st.cache_resource
def get_horizon_bundles(horizon_days: int) -> Dict[float, Dict[str, Any]]:
    return train.load_grid_bundles(horizon_days, CONFIDENCE_LEVELS)


@st.cache_data
def get_features() -> pd.DataFrame:
    return pd.read_parquet(FEATURES_PATH)


@st.cache_data
def get_prices(asset: str) -> pd.DataFrame:
    price_file = os.path.join(PROCESSED_DIR, f"{asset}.parquet")
    if not os.path.exists(price_file):
        raise FileNotFoundError(f"Processed price file for {asset} not found at {price_file}.")
    return pd.read_parquet(price_file)


@st.cache_data
def get_results() -> Optional[Dict[str, Any]]:
    return load_results(RESULTS_PATH)


all_assets = CRYPTO_ASSETS + EQUITY_ASSETS

st.sidebar.header("Collateral & Loan")
selected_asset = st.sidebar.selectbox("Asset", all_assets, index=0)
collateral = st.sidebar.number_input(
    "Collateral Value ($)",
    min_value=0.0,
    value=float(DEMO_COLLATERAL_VALUE),
    step=100.0,
)

st.sidebar.markdown("---")
st.sidebar.subheader("Model Configuration")

default_h_idx = HORIZONS_DAYS.index(DEFAULT_HORIZON_DAYS) if DEFAULT_HORIZON_DAYS in HORIZONS_DAYS else 0
selected_horizon = st.sidebar.radio(
    "Liquidation Horizon",
    options=HORIZONS_DAYS,
    index=default_h_idx,
    format_func=lambda h: f"{h} days",
)

selected_confidence = st.sidebar.select_slider(
    "Confidence Level",
    options=CONFIDENCE_LEVELS,
    value=DEFAULT_CONFIDENCE,
    format_func=lambda c: f"{round(c * 100)}%",
)

# Friendly failure check for selected model bundle
selected_model_file = model_path(selected_horizon, selected_confidence)
if not os.path.exists(selected_model_file):
    st.error(
        f"Model file for horizon={selected_horizon}d and confidence={round(selected_confidence * 100)}% "
        f"not found at {selected_model_file}. Please train models with: python -m src.train"
    )
    st.stop()

try:
    bundle = get_bundle(selected_horizon, selected_confidence)
except FileNotFoundError as err:
    st.error(
        f"Model file not found: {err}. Please train models with: python -m src.train"
    )
    st.stop()

features = get_features()

# Data refresh section in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Data")

crypto_latest = latest_saved_candle(RAW_DIR, CRYPTO_ASSETS)
st.sidebar.text(f"Latest crypto candle: {crypto_latest if crypto_latest else 'None'}")

refresh_status = read_status(REFRESH_STATUS_PATH)
last_success = refresh_status.get("last_success_utc")
last_success_str = str(last_success)[:19].replace("T", " ") if last_success else "Never"
st.sidebar.text(f"Last success (UTC): {last_success_str}")

r_state = refresh_state(now, crypto_latest, refresh_status, REFRESH_MIN_INTERVAL_MINUTES)
refresh_disabled = not r_state["allowed"]

if st.sidebar.button("Refresh data", disabled=refresh_disabled):
    with st.spinner("Refreshing market data..."):
        res = refresh_data(now)
        st.session_state["refresh_res"] = res
        if res.get("status") in ("ok", "partial"):
            st.cache_data.clear()
        st.rerun()

if refresh_disabled:
    st.sidebar.caption(r_state["message"])

# Retraining suggestion if features are substantially newer than bundle train_end_date
days_newer = newer_data_days(bundle, features)
if days_newer > RETRAIN_SUGGEST_AFTER_DAYS:
    st.sidebar.info(
        f"Features contain data {days_newer} days newer than the model training cutoff ({bundle['train_end_date']}).\n\n"
        "Retraining recommended:\n"
        "`python -m src.train`\n"
        "`python -m src.evaluate`"
    )

# Display stored refresh result message
if "refresh_res" in st.session_state:
    res = st.session_state.pop("refresh_res")
    status = res.get("status")
    rows = res.get("rows_added", {})
    errors = res.get("errors", {})
    if status == "ok":
        rows_str = ", ".join(f"{k}: +{v}" for k, v in rows.items())
        st.success(f"Data refreshed successfully! Rows added: {rows_str}")
    elif status == "partial":
        rows_str = ", ".join(f"{k}: +{v}" for k, v in rows.items())
        err_str = "; ".join(f"{k}: {v}" for k, v in errors.items())
        st.warning(f"Data refresh completed with errors. Rows added: {rows_str}. Errors: {err_str}")
    elif status == "failed":
        err_str = "; ".join(f"{k}: {v}" for k, v in errors.items()) if errors else res.get("error", "Unknown error")
        st.error(f"Data refresh failed: {err_str}")
    elif status == "already_running":
        st.info("A data refresh is already running.")
    elif status == "nothing_new":
        st.info("Data is already up to date. No new candles found.")

tab_calc, tab_eval = st.tabs(["Haircut calculator", "Evaluation"])

with tab_calc:
    # Diagnostic warning if quantile crossing is detected for this asset across confidences
    try:
        horizon_bundles = get_horizon_bundles(selected_horizon)
        crossing_dict = latest_crossing(horizon_bundles, features, selected_asset)
        if crossing_dict is not None:
            details = ", ".join([f"{round(c * 100)}%: {val:.2%}" for c, val in crossing_dict.items()])
            st.warning(
                f"Quantile crossing detected for {selected_asset} at {selected_horizon}-day horizon: "
                f"predicted returns across confidence levels violate monotonic ordering ({details}). "
                "Higher confidence models should predict larger or equal projected drops."
            )
    except Exception:
        pass

    calc = calculator_result(bundle, features, selected_asset, collateral)

    conf_pct = round(selected_confidence * 100)
    col1, col2, col3 = st.columns(3)
    col1.metric("Recommended Haircut", f"{calc['haircut']:.2%}")
    col2.metric("Maximum Loan Amount", f"${calc['max_loan']:,.2f}")
    col3.metric(
        f"Predicted worst-case {selected_horizon}-day drop ({conf_pct}% confidence)",
        f"{calc['predicted_return']:.2%}",
    )

    as_of_date = pd.to_datetime(calc["date"]).strftime("%Y-%m-%d")
    if selected_asset in EQUITY_ASSETS:
        st.caption(f"As-of Date: {as_of_date} (last available trading day for equities)")
    else:
        st.caption(f"As-of Date: {as_of_date}")

    st.markdown("### Risk & Liquidity Indicators")
    f_col1, f_col2 = st.columns(2)
    f_col1.metric("30-Day Rolling Volatility", f"{calc['volatility']:.2%}")
    f_col2.metric("30-Day Avg Dollar Volume", f"${calc['avg_dollar_volume']:,.0f}")

    st.markdown("### Haircut Comparison vs Baselines")
    base_hc = baseline_haircuts(
        features,
        selected_horizon,
        selected_confidence,
        selected_asset,
        bundle["train_end_date"],
    )
    hc_df = pd.DataFrame({
        "Method": ["Dynamic Model", "Historical Baseline", "Static Baseline"],
        "Haircut": [calc["haircut"], base_hc["historical"], base_hc["static"]],
    })
    fig_hc = px.bar(
        hc_df,
        x="Method",
        y="Haircut",
        text=[f"{v:.1%}" for v in hc_df["Haircut"]],
        color="Method",
        color_discrete_map={
            "Dynamic Model": "#2563eb",
            "Historical Baseline": "#d97706",
            "Static Baseline": "#475569",
        },
    )
    fig_hc.update_layout(
        yaxis_tickformat=".0%",
        showlegend=False,
        height=320,
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(fig_hc, width="stretch")

    st.markdown("### Price History & Predicted Floor Line")
    prices = get_prices(selected_asset)
    band = risk_band(
        bundle,
        features,
        prices,
        selected_asset,
        selected_horizon,
        lookback_days=365,
    )
    t_cutoff = training_cutoff(bundle)

    fig_risk = go.Figure()
    fig_risk.add_trace(go.Scatter(
        x=band["date"],
        y=band["close"],
        mode="lines",
        name="Close Price",
        line=dict(color="#1e293b", width=1.8),
    ))
    fig_risk.add_trace(go.Scatter(
        x=band["floor_date"],
        y=band["floor_price"],
        mode="lines",
        name=f"Predicted Floor (t+{selected_horizon}d)",
        line=dict(color="#dc2626", width=1.5, dash="dash"),
    ))

    t_cutoff_date = str(t_cutoff.date())
    fig_risk.add_vline(
        x=t_cutoff_date,
        line_dash="dot",
        line_color="#64748b",
        annotation_text="model trained on data up to this date; everything after it is out-of-sample",
        annotation_position="top left",
    )
    fig_risk.update_layout(
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=450,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_risk, width="stretch")

    st.caption(
        f"The dashed red line is the model's predicted floor price for date t + {selected_horizon}. "
        "Breaches occur when the closing price dips below this floor. "
        "Predictions before the vertical dotted line were made on training data (in-sample); "
        f"predictions after it are strictly out-of-sample. The final {selected_horizon} days of the floor extend "
        "beyond the last price date into the future (live prediction)."
    )

with tab_eval:
    st.caption("Evaluation reflects the last python -m src.evaluate run, with its own test_start.")

    results = get_results()
    if results is None:
        st.warning(
            f"Evaluation results not found at {RESULTS_PATH}. "
            "Please generate evaluation results with: python -m src.evaluate"
        )
        st.stop()

    curr_config_key = config_key(selected_horizon, selected_confidence)
    if curr_config_key not in results.get("configs", {}):
        st.warning(
            f"Configuration '{curr_config_key}' not found in {RESULTS_PATH}. "
            "Please run evaluation with: python -m src.evaluate"
        )
        st.stop()

    summary_df, per_asset_df, comparison_dict, info_dict = evaluation_tables(results, curr_config_key)

    st.markdown("### Run Metadata & Split Information")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Out-of-Sample Start", str(info_dict["test_start"]))
    r2.metric("Training End Date", str(info_dict["train_end_date"]))
    r3.metric("Train / Test Rows", f"{info_dict['n_train_rows']:,} / {info_dict['n_test_rows']:,}")
    gen_time = str(info_dict.get("generated_at", ""))[:19].replace("T", " ")
    r4.metric("Generated At (UTC)", gen_time)

    st.markdown("### Model vs Baselines Comparison")
    c1, c2 = st.columns(2)
    p_stat = comparison_dict.get("pinball_improvement_vs_static_pct", 0.0)
    p_hist = comparison_dict.get("pinball_improvement_vs_historical_pct", 0.0)
    c1.metric("Pinball Improvement vs Static", f"{p_stat:+.2f}%")
    c2.metric("Pinball Improvement vs Historical", f"{p_hist:+.2f}%")

    st.markdown("### Overall Performance by Method")
    display_summary = summary_df.copy()
    display_summary["method"] = display_summary["method"].str.capitalize()
    display_summary["pinball_loss"] = display_summary["pinball_loss"].apply(lambda v: f"{v:.6f}")
    display_summary["breach_rate"] = display_summary["breach_rate"].apply(lambda v: f"{v:.2%}")
    display_summary["mean_haircut"] = display_summary["mean_haircut"].apply(lambda v: f"{v:.2%}")
    display_summary.columns = ["Method", "Test Rows", "Pinball Loss", "Breach Rate", "Mean Haircut"]
    st.dataframe(display_summary, width="stretch", hide_index=True)

    st.markdown("### Per-Asset Test Breach Rates")
    plot_df = per_asset_df.copy()
    plot_df["method"] = plot_df["method"].str.capitalize()
    fig_breach = px.bar(
        plot_df,
        x="asset",
        y="breach_rate",
        color="method",
        barmode="group",
        labels={"asset": "Asset", "breach_rate": "Breach Rate", "method": "Method"},
        color_discrete_map={
            "Model": "#2563eb",
            "Historical": "#d97706",
            "Static": "#475569",
        },
    )
    alpha_val = float(info_dict["alpha"])
    fig_breach.add_hline(
        y=alpha_val,
        line_dash="dash",
        line_color="#dc2626",
        annotation_text=f"Target alpha ({alpha_val:.2%})",
        annotation_position="top right",
    )
    fig_breach.update_layout(
        yaxis_tickformat=".1%",
        height=380,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_breach, width="stretch")

    st.caption(
        "Note on breach rates: Overlapping N-day returns make breaches cluster across consecutive days. "
        "With a 20% test fraction, realized breach rates carry natural clustering noise; small deviations "
        f"from the target alpha ({alpha_val:.1%}) do not indicate model failure."
    )

st.markdown("---")
st.caption("Educational machine learning project. This dashboard is for demonstration only and does not constitute financial or lending advice.")
