# PEA Risk Lab

A standalone Streamlit app for exploring market risk and diversification
across stocks and ETFs eligible in a French PEA, built entirely on open data.

This is deliberately a separate site/repo from other portfolio projects --
different visual language (research-lab aesthetic: whitespace, one focal
chart per view) rather than a dense "trading desk" dashboard.

## What it does today (Phase 1)

- **Single instrument view**: annualized volatility, historical VaR (95/99%),
  CVaR (95%), parametric VaR, beta vs CAC 40, max drawdown, Sharpe ratio,
  price + drawdown chart, return distribution with VaR marked.
- **Combination view**: pick 2+ instruments, choose equal-weight or a
  minimum-variance optimized portfolio, and see the diversification effect
  explicitly -- weighted-average volatility (what you'd get with perfect
  correlation) vs. actual portfolio volatility, plus a correlation heatmap.

## Running it

```bash
cd risk_lab
pip install -r requirements.txt
streamlit run app.py
```

Note: the data layer (`data_fetch.py`) calls Yahoo Finance via `yfinance`.
This requires normal internet access -- it was developed and unit-validated
against synthetic data, since the build sandbox has no route to Yahoo's
servers, so the live fetch should be smoke-tested on first run.

## Architecture

```
risk_lab/
  app.py            # Streamlit UI, two views: single asset / combination
  config.py         # PEA-eligible ticker universe, benchmark, constants
  data_fetch.py     # yfinance calls, isolated and cached (st.cache_data)
  risk_metrics.py   # pure functions: VaR, CVaR, vol, beta, drawdown, Sharpe
  portfolio.py      # covariance, diversification ratio, min-variance optimizer
```

`risk_metrics.py` and `portfolio.py` have no Streamlit or network
dependencies -- they take return/price series and return numbers, so they
can be unit tested against synthetic data independent of any live fetch
(see the validation script used during development, not included here but
trivial to reconstruct: generate correlated/uncorrelated synthetic return
series with `numpy.random`, check beta-vs-self == 1.0, CVaR >= VaR, and that
the min-variance optimizer never produces a higher-vol portfolio than
equal-weight).

## Roadmap (not yet built)

- **Phase 2 -- Credit/default risk for companies**: Merton structural model
  (distance-to-default from equity volatility + balance sheet leverage) and
  Altman Z-score as a cross-check. Framed explicitly as a market-implied
  proxy, since open CDS spread data doesn't exist -- this is a real
  limitation worth stating rather than hiding.
- **Phase 3 -- ETF-specific market risk**: tracking error vs stated
  benchmark, look-through risk into top holdings where disclosed.
- **Phase 4 -- ETF counterparty risk**: classify replication method
  (physical vs synthetic/swap-based) and extract swap counterparty exposure
  from KIID/prospectus documents. `config.py` already carries a
  `replication` field per ETF as a placeholder for this.

## Data sources

- Prices: Yahoo Finance (via `yfinance`)
- Benchmark: CAC 40 (`^FCHI`)
- Planned for later phases: AMF/ESMA fund register, issuer KIID/prospectus
  PDFs (Amundi, BNP Paribas Easy, etc.), FRED/ECB for risk-free rates
