"""
Ticker universe and global constants for the PEA Risk Lab.

The universe below is a curated starting point, not exhaustive. Extend the
dicts as needed -- the app reads tickers directly from these structures, so
adding a new entry is enough to make it selectable.

Yahoo Finance tickers are used since the app fetches data via yfinance.
"""

BENCHMARK_TICKER = "^FCHI"          # CAC 40, used for beta calculations
BENCHMARK_NAME = "CAC 40"

RISK_FREE_PROXY_TICKER = "^IRX"      # 13-week US T-bill as a rough proxy if FRED is not wired in yet
DEFAULT_RISK_FREE_RATE = 0.025       # fallback annualized rate if data fetch fails

# --- PEA-eligible single stocks (CAC 40 + a few large caps) ---------------
PEA_STOCKS = {
    "MC.PA": "LVMH",
    "OR.PA": "L'Oréal",
    "AIR.PA": "Airbus",
    "SAN.PA": "Sanofi",
    "BNP.PA": "BNP Paribas",
    "TTE.PA": "TotalEnergies",
    "AI.PA": "Air Liquide",
    "SU.PA": "Schneider Electric",
    "DG.PA": "Vinci",
    "EL.PA": "EssilorLuxottica",
    "RMS.PA": "Hermès",
    "SAF.PA": "Safran",
    "CS.PA": "AXA",
    "BN.PA": "Danone",
    "KER.PA": "Kering",
    "ENGI.PA": "Engie",
    "ML.PA": "Michelin",
    "STLAP.PA": "Stellantis",
    "CAP.PA": "Capgemini",
    "PUB.PA": "Publicis",
}

# --- PEA-eligible UCITS ETFs (mix of physical and synthetic replication) --
# replication field is a placeholder for Phase 4 (counterparty risk module).
PEA_ETFS = {
    "CW8.PA": {"name": "Amundi MSCI World", "replication": "synthetic"},
    "PUST.PA": {"name": "Amundi PEA Nasdaq-100", "replication": "synthetic"},
    "PSP5.PA": {"name": "Amundi PEA S&P 500", "replication": "synthetic"},
    "ESE.PA": {"name": "BNP Paribas Easy S&P 500", "replication": "synthetic"},
    "CACC.PA": {"name": "Amundi CAC 40 (Acc)", "replication": "physical"},
    "PCEU.PA": {"name": "Amundi PEA MSCI Europe", "replication": "physical"},
    "EWLD.PA": {"name": "Lyxor PEA Monde (MSCI World)", "replication": "synthetic"},
}

def full_universe():
    """Return a single {ticker: display_name} dict spanning stocks and ETFs."""
    universe = {t: f"{name} (stock)" for t, name in PEA_STOCKS.items()}
    universe.update({t: f"{v['name']} (ETF)" for t, v in PEA_ETFS.items()})
    return universe

TRADING_DAYS_PER_YEAR = 252