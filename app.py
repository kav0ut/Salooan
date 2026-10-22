import os
from typing import Any, Dict, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import (
    CRYPTO_ASSETS,
    DEFAULT_CONFIDENCE,
    DEFAULT_HORIZON_DAYS,
    DEMO_COLLATERAL_VALUE,
    EQUITY_ASSETS,
    FEATURES_PATH,
    MODELS_DIR,
    PROCESSED_DIR,
    RESULTS_PATH,
)
from src.dashboard import (
    baseline_haircuts,
    calculator_result,
    evaluation_tables,
    load_results,
    risk_band,
    test_period_start,
)
from src.train import load_bundle

st.set_page_config(page_title="Salooan", layout="wide")

st.title("Salooan")
st.markdown(
    "Dynamic collateral haircut estimation: predicting worst-case price drops over a "
    "liquidation horizon so lenders can safely size cash loans against volatile collateral."
)

model_file = os.path.join(
    MODELS_DIR,
    f"quantile_h{DEFAULT_HORIZON_DAYS}_c{round(DEFAULT_CONFIDENCE * 100)}.joblib",
)
if not os.path.exists(model_file):
    st.error(f"Model file not found at {model_file}. Please train the model with: python -m src.train")
    st.stop()

if not os.path.exists(FEATURES_PATH):
    st.error(f"Feature table not found at {FEATURES_PATH}. Please generate features with: python -m src.features")
    st.stop()


@st.cache_resource
def get_bundle() -> Dict[str, Any]:
    return load_bundle(DEFAULT_HORIZON_DAYS, DEFAULT_CONFIDENCE)


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


bundle = get_bundle()
features = get_features()

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
st.sidebar.text(f"Liquidation Horizon: {DEFAULT_HORIZON_DAYS} days")
st.sidebar.text(f"Confidence Level: {int(DEFAULT_CONFIDENCE * 100)}%")
st.sidebar.caption("Fixed configuration for Milestone 7 (grid selection in Milestone 8).")

tab_calc, tab_eval = st.tabs(["Haircut calculator", "Evaluation"])

with tab_calc:
    calc = calculator_result(bundle, features, selected_asset, collateral)

    col1, col2, col3 = st.columns(3)
    col1.metric("Recommended Haircut", f"{calc['haircut']:.2%}")
    col2.metric("Maximum Loan Amount", f"${calc['max_loan']:,.2f}")
    col3.metric(
        f"Worst-Case {DEFAULT_HORIZON_DAYS}d Return",
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
        DEFAULT_HORIZON_DAYS,
        DEFAULT_CONFIDENCE,
        selected_asset,
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
        DEFAULT_HORIZON_DAYS,
        lookback_days=365,
    )
    t_start = test_period_start(features, DEFAULT_HORIZON_DAYS)

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
        name=f"Predicted Floor (t+{DEFAULT_HORIZON_DAYS}d)",
        line=dict(color="#dc2626", width=1.5, dash="dash"),
    ))

    t_start_date = str(pd.to_datetime(t_start).date())
    fig_risk.add_vline(
        x=t_start_date,
        line_dash="dot",
        line_color="#64748b",
        annotation_text="Out-of-sample Start",
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
        "The dashed red line is the model's predicted floor price for date t + 7. "
        "Breaches occur when the closing price dips below this floor. "
        "Predictions before the vertical dotted line were made on training data (in-sample); "
        "predictions after it are strictly out-of-sample. The final 7 days of the floor extend "
        "beyond the last price date into the future (live prediction)."
    )

with tab_eval:
    results = get_results()
    if results is None:
        st.warning(
            f"Evaluation results not found at {RESULTS_PATH}. "
            "Please generate evaluation results with: python -m src.evaluate"
        )
        st.stop()

    pct = round(DEFAULT_CONFIDENCE * 100)
    config_key = f"h{DEFAULT_HORIZON_DAYS}_c{pct}"
    if config_key not in results.get("configs", {}):
        st.warning(
            f"Configuration '{config_key}' not found in {RESULTS_PATH}. "
            "Please run evaluation with: python -m src.evaluate"
        )
        st.stop()

    summary_df, per_asset_df, comparison_dict, info_dict = evaluation_tables(results, config_key)

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
