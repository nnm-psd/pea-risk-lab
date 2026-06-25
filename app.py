"""
PEA Risk Lab -- Streamlit entry point.

Run with: streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import risk_metrics as rm
import portfolio as pf
from config import full_universe, PEA_STOCKS, PEA_ETFS, DEFAULT_RISK_FREE_RATE
from data_fetch import fetch_price_history, fetch_benchmark_history, compute_log_returns

# ---------------------------------------------------------------------------
# Page config + visual identity: research-lab feel, not "trading desk".
# Generous whitespace, one focal chart per view, muted palette.
# ---------------------------------------------------------------------------
st.set_page_config(page_title="PEA Risk Lab", page_icon="◇", layout="wide")

LAB_CSS = """
<style>
    .stApp { background-color: #FAFAF8; }
    h1, h2, h3 { font-family: 'Georgia', serif; color: #1A1A2E; font-weight: 400; }
    .lab-subtitle { color: #6B6B76; font-size: 0.95rem; font-family: 'Helvetica', sans-serif;
                    margin-top: -0.6rem; margin-bottom: 1.6rem; }
    .metric-card { background-color: #FFFFFF; border: 1px solid #E8E6E1; border-radius: 6px;
                   padding: 1rem 1.2rem; }
    .stTabs [data-baseweb="tab"] { font-family: 'Helvetica', sans-serif; }
    div[data-testid="stMetricValue"] { font-family: 'Georgia', serif; }
</style>
"""
st.markdown(LAB_CSS, unsafe_allow_html=True)

ACCENT = "#2E5E4E"      # deep teal -- primary
ACCENT_SOFT = "#A8C3B8"
WARN = "#A4452E"        # muted terracotta -- risk / loss

st.title("PEA Risk Lab")
st.markdown(
    '<div class="lab-subtitle">Market risk and diversification, built from open data, '
    'for instruments eligible in a French PEA.</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar: universe selection
# ---------------------------------------------------------------------------
universe = full_universe()
labels_to_ticker = {f"{v} -- {k}": k for k, v in universe.items()}

st.sidebar.header("Select instruments")
selected_labels = st.sidebar.multiselect(
    "Stocks and ETFs (PEA-eligible)",
    options=sorted(labels_to_ticker.keys()),
    default=[],
    help="Pick one instrument to see its individual risk profile, or several "
         "to see how combining them changes overall risk.",
)
selected_tickers = [labels_to_ticker[lbl] for lbl in selected_labels]

period = st.sidebar.select_slider(
    "Lookback period", options=["1y", "2y", "3y", "5y"], value="3y"
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Data: Yahoo Finance via yfinance. Risk metrics computed locally from "
    "daily returns -- nothing here is a market-implied or vendor risk figure."
)

if not selected_tickers:
    st.info("Select at least one instrument from the sidebar to begin.")
    st.stop()

# ---------------------------------------------------------------------------
# Fetch data
# ---------------------------------------------------------------------------
with st.spinner("Fetching price history..."):
    try:
        prices = fetch_price_history(selected_tickers, period=period)
        benchmark = fetch_benchmark_history(period=period)
    except Exception as e:
        st.error(f"Data fetch failed: {e}")
        st.stop()

missing = [t for t in selected_tickers if t not in prices.columns]
if missing:
    st.warning(f"No data returned for: {', '.join(missing)}. Removed from analysis.")
    selected_tickers = [t for t in selected_tickers if t not in missing]

if prices.empty or len(selected_tickers) == 0:
    st.error("No usable price data for the current selection.")
    st.stop()

benchmark_returns = compute_log_returns(benchmark.to_frame("bm"))["bm"]

# ===========================================================================
# SINGLE-ASSET VIEW
# ===========================================================================
if len(selected_tickers) == 1:
    ticker = selected_tickers[0]
    name = universe[ticker]
    price_series = prices[ticker].dropna()

    st.header(name)
    summary = rm.single_asset_summary(price_series, benchmark_returns, DEFAULT_RISK_FREE_RATE)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Annualized volatility", f"{summary['annualized_volatility']*100:.1f}%")
    c2.metric("VaR (95%, 1-day)", f"{summary['var_95']*100:.2f}%")
    c3.metric("Max drawdown", f"{summary['max_drawdown']*100:.1f}%")
    c4.metric("Beta vs CAC 40", f"{summary['beta']:.2f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Annualized return", f"{summary['annualized_return']*100:.1f}%")
    c6.metric("CVaR (95%, 1-day)", f"{summary['cvar_95']*100:.2f}%")
    c7.metric("VaR (99%, 1-day)", f"{summary['var_99']*100:.2f}%")
    c8.metric("Sharpe ratio", f"{summary['sharpe_ratio']:.2f}")

    st.markdown("&nbsp;")

    tab1, tab2, tab3 = st.tabs(["Price & drawdown", "Return distribution", "Notes"])

    with tab1:
        dd = rm.drawdown_series(price_series)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=price_series.index, y=price_series, name="Price",
                                  line=dict(color=ACCENT, width=1.5), yaxis="y1"))
        fig.add_trace(go.Scatter(x=dd.index, y=dd * 100, name="Drawdown (%)",
                                  fill="tozeroy", line=dict(color=WARN, width=1),
                                  yaxis="y2", opacity=0.6))
        fig.update_layout(
            yaxis=dict(title="Price"),
            yaxis2=dict(title="Drawdown (%)", overlaying="y", side="right", showgrid=False),
            plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", y=1.1), height=420,
            margin=dict(t=30, l=10, r=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        returns = np.log(price_series / price_series.shift(1)).dropna()
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(x=returns * 100, nbinsx=80, marker_color=ACCENT_SOFT,
                                     name="Daily returns"))
        fig2.add_vline(x=-summary["var_95"] * 100, line_dash="dash", line_color=WARN,
                        annotation_text="VaR 95%")
        fig2.update_layout(
            xaxis_title="Daily return (%)", yaxis_title="Frequency",
            plot_bgcolor="white", paper_bgcolor="white", height=420,
            margin=dict(t=30, l=10, r=10, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.markdown(
            "- **VaR (95%)** reads as: on roughly 95% of trading days, the daily loss "
            "does not exceed this figure. It says nothing about the size of losses on "
            "the remaining 5% of days -- that's what CVaR is for.\n"
            "- **CVaR (95%)** is the average loss on exactly those worst 5% of days. "
            "It is always greater than or equal to VaR, and is generally the more "
            "honest tail-risk figure.\n"
            "- **Beta** is estimated by OLS against the CAC 40 over the selected lookback; "
            "it will shift with the window chosen.\n"
            "- Credit/default risk and ETF counterparty/replication risk are not yet "
            "covered for this instrument -- planned for a later phase."
        )

# ===========================================================================
# COMBINATION VIEW
# ===========================================================================
else:
    st.header(f"Combination of {len(selected_tickers)} instruments")

    weighting_mode = st.radio(
        "Weighting",
        options=["Equal-weight", "Optimize for minimum risk"],
        horizontal=True,
    )
    weighting_key = "min_variance" if weighting_mode.startswith("Optimize") else "equal"

    combo = pf.combination_summary(prices[selected_tickers], weighting=weighting_key)

    # --- the focal moment: weighted-average vol vs actual portfolio vol ---
    st.subheader("Diversification effect")
    cA, cB, cC = st.columns(3)
    cA.metric(
        "If risks just added up",
        f"{combo['weighted_average_volatility']*100:.1f}%",
        help="Weighted-average volatility of the individual instruments -- "
             "what you'd get if they were perfectly correlated.",
    )
    cB.metric(
        "Actual portfolio volatility",
        f"{combo['portfolio_volatility']*100:.1f}%",
        delta=f"-{combo['risk_reduction_pct']*100:.1f}%",
        delta_color="inverse",
        help="Volatility actually realized once correlation between the "
             "instruments is accounted for.",
    )
    cC.metric(
        "Diversification ratio",
        f"{combo['diversification_ratio']:.2f}",
        help="Weighted-average vol / portfolio vol. 1.0 = no benefit "
             "(perfectly correlated assets). Higher = more benefit.",
    )

    fig3 = go.Figure(go.Bar(
        x=["No diversification\n(weighted-avg)", "Actual portfolio"],
        y=[combo["weighted_average_volatility"] * 100, combo["portfolio_volatility"] * 100],
        marker_color=[ACCENT_SOFT, ACCENT],
        text=[f"{combo['weighted_average_volatility']*100:.1f}%",
              f"{combo['portfolio_volatility']*100:.1f}%"],
        textposition="outside",
    ))
    fig3.update_layout(
        yaxis_title="Annualized volatility (%)",
        plot_bgcolor="white", paper_bgcolor="white", height=380,
        margin=dict(t=20, l=10, r=10, b=10), showlegend=False,
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("&nbsp;")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Weights")
        weights_df = pd.DataFrame({
            "Instrument": [universe[t] for t in combo["tickers"]],
            "Ticker": combo["tickers"],
            "Weight": [f"{combo['weights'][t]*100:.1f}%" for t in combo["tickers"]],
            "Individual vol (ann.)": [f"{combo['individual_vols'][t]*100:.1f}%" for t in combo["tickers"]],
        })
        st.dataframe(weights_df, hide_index=True, use_container_width=True)

        st.metric("Portfolio VaR (95%, 1-day)", f"{combo['portfolio_var_95']*100:.2f}%")
        st.metric("Portfolio annualized return", f"{combo['portfolio_return']*100:.1f}%")

    with col_right:
        st.subheader("Correlation matrix")
        corr = combo["correlation_matrix"]
        fig4 = go.Figure(data=go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            colorscale=[[0, "#2E5E4E"], [0.5, "#FAFAF8"], [1, "#A4452E"]],
            zmin=-1, zmax=1, text=np.round(corr.values, 2), texttemplate="%{text}",
        ))
        fig4.update_layout(height=380, margin=dict(t=20, l=10, r=10, b=10))
        st.plotly_chart(fig4, use_container_width=True)

    st.caption(
        "Low or negative correlation between holdings is what drives the diversification "
        "ratio above 1.0. Try swapping one instrument for a less-correlated one and watch "
        "the bar chart move."
    )
