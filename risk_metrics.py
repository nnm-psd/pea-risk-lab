"""
Single-asset market risk metrics. Every function takes a return series
(daily log returns) and returns a plain float or small structure, so these
are easy to unit test against synthetic data independent of any data fetch.
"""

import numpy as np
import pandas as pd
from scipy import stats

from config import TRADING_DAYS_PER_YEAR


def annualized_volatility(returns):
    return returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)


def annualized_return(returns):
    """Geometric annualized return from daily log returns."""
    total_log_return = returns.sum()
    n_years = len(returns) / TRADING_DAYS_PER_YEAR
    if n_years <= 0:
        return np.nan
    return np.exp(total_log_return / n_years) - 1


def historical_var(returns, confidence=0.95):
    """
    Historical (empirical) VaR, expressed as a positive number representing
    the loss threshold over a single day at the given confidence level.
    E.g. VaR_95 = 0.02 means: on 95% of days, the loss does not exceed 2%.
    """
    alpha = 1 - confidence
    return -np.percentile(returns, alpha * 100)


def historical_cvar(returns, confidence=0.95):
    """
    Conditional VaR / Expected Shortfall: average loss in the tail beyond
    the VaR threshold. Always >= VaR, captures tail severity VaR ignores.
    """
    var = historical_var(returns, confidence)
    tail_losses = returns[returns <= -var]
    if len(tail_losses) == 0:
        return var
    return -tail_losses.mean()


def parametric_var(returns, confidence=0.95):
    """Variance-covariance VaR assuming normally distributed returns."""
    mu = returns.mean()
    sigma = returns.std(ddof=1)
    z = stats.norm.ppf(1 - confidence)
    return -(mu + z * sigma)


def max_drawdown(price_series):
    """
    Maximum peak-to-trough decline over the period, as a positive fraction.
    Operates on price levels, not returns.
    """
    cumulative_max = price_series.cummax()
    drawdown = (price_series - cumulative_max) / cumulative_max
    return -drawdown.min()


def drawdown_series(price_series):
    """Full drawdown time series, for charting."""
    cumulative_max = price_series.cummax()
    return (price_series - cumulative_max) / cumulative_max


def beta(asset_returns, benchmark_returns):
    """OLS beta of asset returns vs benchmark returns, aligned on date index."""
    aligned = pd.concat([asset_returns, benchmark_returns], axis=1, join="inner").dropna()
    aligned.columns = ["asset", "benchmark"]
    if len(aligned) < 2:
        return np.nan
    cov = np.cov(aligned["asset"], aligned["benchmark"])[0, 1]
    var = np.var(aligned["benchmark"], ddof=1)
    return cov / var if var != 0 else np.nan


def sharpe_ratio(returns, risk_free_rate=0.025):
    ann_ret = annualized_return(returns)
    ann_vol = annualized_volatility(returns)
    if ann_vol == 0 or np.isnan(ann_vol):
        return np.nan
    return (ann_ret - risk_free_rate) / ann_vol


def single_asset_summary(price_series, benchmark_returns=None, risk_free_rate=0.025):
    """Convenience wrapper bundling all single-asset metrics into one dict."""
    returns = np.log(price_series / price_series.shift(1)).dropna()
    summary = {
        "annualized_return": annualized_return(returns),
        "annualized_volatility": annualized_volatility(returns),
        "var_95": historical_var(returns, 0.95),
        "var_99": historical_var(returns, 0.99),
        "cvar_95": historical_cvar(returns, 0.95),
        "parametric_var_95": parametric_var(returns, 0.95),
        "max_drawdown": max_drawdown(price_series),
        "sharpe_ratio": sharpe_ratio(returns, risk_free_rate),
    }
    if benchmark_returns is not None:
        summary["beta"] = beta(returns, benchmark_returns)
    return summary
