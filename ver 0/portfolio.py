"""
Combination/diversification logic. This is the centerpiece of the app: given
several assets' return series, quantify how much risk is removed by holding
them together rather than separately.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from config import TRADING_DAYS_PER_YEAR


def returns_matrix(price_df):
    """Log returns for multiple tickers, aligned on common dates."""
    return np.log(price_df / price_df.shift(1)).dropna()


def annualized_cov_matrix(returns_df):
    return returns_df.cov() * TRADING_DAYS_PER_YEAR


def correlation_matrix(returns_df):
    return returns_df.corr()


def portfolio_volatility(weights, cov_matrix):
    weights = np.asarray(weights)
    return np.sqrt(weights @ cov_matrix.values @ weights)


def portfolio_return(weights, returns_df):
    """Annualized portfolio return given daily log returns and weights."""
    weights = np.asarray(weights)
    daily_portfolio_returns = returns_df.values @ weights
    n_years = len(daily_portfolio_returns) / TRADING_DAYS_PER_YEAR
    total_log_return = daily_portfolio_returns.sum()
    return np.exp(total_log_return / n_years) - 1 if n_years > 0 else np.nan


def weighted_average_volatility(weights, individual_vols):
    """
    The risk you'd have if there were NO diversification benefit at all --
    i.e. correlations were all +1. This is the baseline portfolio vol gets
    compared against.
    """
    return float(np.dot(weights, individual_vols))


def diversification_ratio(weights, individual_vols, cov_matrix):
    """
    weighted-average volatility / actual portfolio volatility.
    Ratio > 1 means diversification is reducing risk; ratio == 1 means the
    assets are perfectly correlated and diversification buys nothing.
    """
    w_avg_vol = weighted_average_volatility(weights, individual_vols)
    port_vol = portfolio_volatility(weights, cov_matrix)
    if port_vol == 0:
        return np.nan
    return w_avg_vol / port_vol


def equal_weights(n_assets):
    return np.repeat(1.0 / n_assets, n_assets)


def min_variance_weights(cov_matrix, long_only=True):
    """
    Solve for the minimum-variance long-only portfolio.
    Constraints: weights sum to 1; bounds [0, 1] per asset if long_only.
    """
    n = cov_matrix.shape[0]
    x0 = equal_weights(n)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n if long_only else [(-1.0, 1.0)] * n

    result = minimize(
        fun=lambda w: portfolio_volatility(w, cov_matrix) ** 2,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 500},
    )

    if not result.success:
        # fall back to equal weight rather than fail the app
        return equal_weights(n)
    return result.x


def portfolio_var_historical(weights, returns_df, confidence=0.95):
    """Historical VaR computed on the actual weighted daily portfolio return series."""
    weights = np.asarray(weights)
    daily_portfolio_returns = returns_df.values @ weights
    alpha = 1 - confidence
    return -np.percentile(daily_portfolio_returns, alpha * 100)


def combination_summary(price_df, weighting="equal"):
    """
    Full diversification analysis for a set of tickers.

    weighting: "equal" or "min_variance"
    Returns a dict with weights, individual vols, portfolio vol,
    weighted-average vol, diversification ratio, correlation matrix,
    and portfolio VaR.
    """
    returns_df = returns_matrix(price_df)
    tickers = list(returns_df.columns)
    cov_matrix = annualized_cov_matrix(returns_df)
    individual_vols = np.array([
        returns_df[t].std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR) for t in tickers
    ])

    if weighting == "min_variance":
        weights = min_variance_weights(cov_matrix)
    else:
        weights = equal_weights(len(tickers))

    port_vol = portfolio_volatility(weights, cov_matrix)
    w_avg_vol = weighted_average_volatility(weights, individual_vols)
    div_ratio = diversification_ratio(weights, individual_vols, cov_matrix)
    port_var_95 = portfolio_var_historical(weights, returns_df, 0.95)

    return {
        "tickers": tickers,
        "weights": dict(zip(tickers, weights)),
        "individual_vols": dict(zip(tickers, individual_vols)),
        "portfolio_volatility": port_vol,
        "weighted_average_volatility": w_avg_vol,
        "diversification_ratio": div_ratio,
        "risk_reduction_pct": (1 - port_vol / w_avg_vol) if w_avg_vol > 0 else np.nan,
        "portfolio_var_95": port_var_95,
        "correlation_matrix": correlation_matrix(returns_df),
        "portfolio_return": portfolio_return(weights, returns_df),
    }
