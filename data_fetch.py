"""
Data access layer. All network calls are isolated here so the risk/portfolio
modules stay pure and testable against synthetic data.
"""

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from config import BENCHMARK_TICKER


@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_price_history(tickers, period="3y", interval="1d"):
    """
    Fetch adjusted close prices for one or more tickers.

    Returns a DataFrame indexed by date, one column per ticker. Tickers that
    fail to download are dropped silently; caller should check for missing
    columns and surface a warning in the UI.
    """
    if isinstance(tickers, str):
        tickers = [tickers]

    data = yf.download(
        tickers,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )

    if isinstance(data.columns, pd.MultiIndex):
        # Newer yfinance versions return MultiIndex (Ticker, Price) columns
        # even for a single ticker when group_by="ticker" is set.
        available = [t for t in tickers if t in data.columns.get_level_values(0)]
        close = pd.concat({t: data[t]["Close"] for t in available}, axis=1)
    else:
        # Flat columns -- older yfinance behavior for a single ticker.
        close = data[["Close"]].rename(columns={"Close": tickers[0]})

    return close.dropna(how="all")


@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_benchmark_history(period="3y", interval="1d"):
    series = fetch_price_history(BENCHMARK_TICKER, period=period, interval=interval)
    return series[BENCHMARK_TICKER]


def compute_log_returns(price_df):
    """Daily log returns. Drops the first (NaN) row."""
    return np.log(price_df / price_df.shift(1)).dropna()