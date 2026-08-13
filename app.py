# ============================================================
# PAIRTRADE FINDER
# Cointegration / Pairs Trading Screener
# Monolithic Streamlit Application
# ============================================================
#
# Features:
#   - Yahoo Finance / yfinance data feed
#   - S&P 500
#   - Nasdaq 100
#   - Russell 2000 via iShares IWM holdings
#   - Futures
#   - Forex
#   - Top-50 crypto by market capitalization via CoinGecko
#   - Custom ticker universe
#   - Pearson correlation filter
#   - OLS Alpha / Beta
#   - Engle-Granger residual ADF
#   - Half-Life
#   - Rolling Z-Score
#   - Historical backtest
#   - Win Rate
#   - Total PnL
#   - Max Drawdown
#   - Interactive Plotly charts
#   - CSV / Excel export
#
# ============================================================

import io
import math
import itertools
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import statsmodels.api as sm

from scipy import stats
from statsmodels.tsa.stattools import adfuller


# ============================================================
# GLOBAL CONFIG
# ============================================================

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="PairTrade Finder",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #0e1117;
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3 {
        letter-spacing: -0.02em;
    }

    .metric-card {
        background: linear-gradient(
            135deg,
            rgba(30, 41, 59, 0.95),
            rgba(15, 23, 42, 0.95)
        );
        border: 1px solid rgba(148, 163, 184, 0.20);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 10px;
    }

    .status-ok {
        color: #22c55e;
        font-weight: 700;
    }

    .status-warning {
        color: #f59e0b;
        font-weight: 700;
    }

    .small-muted {
        color: #94a3b8;
        font-size: 0.85rem;
    }

    .pair-title {
        font-size: 1.35rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }

    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CONSTANTS
# ============================================================

FUTURES_UNIVERSES = {
    "CME Index Futures": {
        "ES=F": "S&P 500 E-mini",
        "NQ=F": "Nasdaq 100 E-mini",
        "YM=F": "Dow Jones E-mini",
        "RTY=F": "Russell 2000 E-mini",
        "EMD=F": "S&P MidCap 400 E-mini",
    },
    "CME Agricultural / Soft Commodities": {
        "ZC=F": "Corn",
        "ZS=F": "Soybeans",
        "ZW=F": "Wheat",
        "ZM=F": "Soybean Meal",
        "ZL=F": "Soybean Oil",
        "KC=F": "Coffee",
        "SB=F": "Sugar",
        "CC=F": "Cocoa",
        "CT=F": "Cotton",
        "LE=F": "Live Cattle",
        "HE=F": "Lean Hogs",
    },
    "CME Metals": {
        "GC=F": "Gold",
        "SI=F": "Silver",
        "HG=F": "Copper",
        "PL=F": "Platinum",
        "PA=F": "Palladium",
    },
    "CME / NYMEX Energy": {
        "CL=F": "WTI Crude Oil",
        "BZ=F": "Brent Crude",
        "NG=F": "Natural Gas",
        "RB=F": "Gasoline",
        "HO=F": "Heating Oil",
    },
}


FOREX_UNIVERSE = {
    "EURUSD=X": "EUR / USD",
    "GBPUSD=X": "GBP / USD",
    "USDJPY=X": "USD / JPY",
    "USDCHF=X": "USD / CHF",
    "AUDUSD=X": "AUD / USD",
    "NZDUSD=X": "NZD / USD",
    "USDCAD=X": "USD / CAD",
    "USDSEK=X": "USD / SEK",
    "USDNOK=X": "USD / NOK",
    "USDMXN=X": "USD / MXN",
    "EURGBP=X": "EUR / GBP",
    "EURJPY=X": "EUR / JPY",
    "GBPJPY=X": "GBP / JPY",
    "AUDJPY=X": "AUD / JPY",
    "CADJPY=X": "CAD / JPY",
    "CHFJPY=X": "CHF / JPY",
}


FALLBACK_CRYPTO = [
    "BTC-USD",
    "ETH-USD",
    "USDT-USD",
    "BNB-USD",
    "XRP-USD",
    "USDC-USD",
    "SOL-USD",
    "TRX-USD",
    "DOGE-USD",
    "ADA-USD",
    "BCH-USD",
    "LINK-USD",
    "XLM-USD",
    "LTC-USD",
    "AVAX-USD",
    "HBAR-USD",
    "SHIB-USD",
    "TON-USD",
    "DOT-USD",
    "UNI-USD",
    "AAVE-USD",
    "ETC-USD",
    "ATOM-USD",
    "XMR-USD",
    "FIL-USD",
    "NEAR-USD",
    "APT-USD",
    "ARB-USD",
    "OP-USD",
    "VET-USD",
    "MKR-USD",
    "ALGO-USD",
    "ICP-USD",
    "SUI-USD",
    "INJ-USD",
    "QNT-USD",
    "STX-USD",
    "IMX-USD",
    "GRT-USD",
    "RUNE-USD",
    "THETA-USD",
    "FTM-USD",
    "EGLD-USD",
    "FLOW-USD",
    "AXS-USD",
    "SAND-USD",
    "MANA-USD",
    "EOS-USD",
    "XTZ-USD",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_ticker(ticker):
    """
    Converts common exchange notation to Yahoo Finance notation.
    """
    ticker = str(ticker).strip().upper()

    if not ticker:
        return ""

    ticker = ticker.replace(" ", "")

    # Yahoo uses "-" instead of "." for many US share classes.
    if ticker.count(".") == 1 and not ticker.endswith(".X"):
        ticker = ticker.replace(".", "-")

    return ticker


def clean_ticker_list(values):
    """
    Normalize and deduplicate tickers.
    """
    result = []

    for value in values:
        if value is None:
            continue

        ticker = normalize_ticker(value)

        if ticker and ticker not in result:
            result.append(ticker)

    return result


def safe_float(value, default=np.nan):
    try:
        return float(value)
    except Exception:
        return default


def format_number(value, decimals=2):
    if value is None or not np.isfinite(value):
        return "—"

    return f"{value:,.{decimals}f}"


def format_pct(value, decimals=1):
    if value is None or not np.isfinite(value):
        return "—"

    return f"{value:.{decimals}f}%"


# ============================================================
# MARKET METADATA
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def load_sp500_universe():
    """
    Load S&P 500 constituents and GICS classifications.
    """

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    try:
        tables = pd.read_html(url)

        table = tables[0].copy()

        symbol_col = None

        for candidate in [
            "Symbol",
            "Ticker symbol",
            "Ticker",
        ]:
            if candidate in table.columns:
                symbol_col = candidate
                break

        if symbol_col is None:
            raise ValueError("S&P 500 ticker column not found")

        table["Ticker"] = table[symbol_col].astype(str).map(normalize_ticker)

        if "Security" in table.columns:
            table["Company"] = table["Security"]
        else:
            table["Company"] = table["Ticker"]

        if "GICS Sector" in table.columns:
            table["Sector"] = table["GICS Sector"]
        else:
            table["Sector"] = "Unknown"

        if "GICS Sub-Industry" in table.columns:
            table["Industry"] = table["GICS Sub-Industry"]
        else:
            table["Industry"] = "Unknown"

        table["Index"] = "S&P 500"

        return table[
            [
                "Ticker",
                "Company",
                "Sector",
                "Industry",
                "Index",
            ]
        ].drop_duplicates("Ticker")

    except Exception as exc:

        st.warning(
            f"Не удалось загрузить S&P 500 автоматически: {exc}"
        )

        return pd.DataFrame(
            columns=[
                "Ticker",
                "Company",
                "Sector",
                "Industry",
                "Index",
            ]
        )


@st.cache_data(ttl=3600, show_spinner=False)
def load_nasdaq100_universe():
    """
    Load Nasdaq-100 constituents.
    """

    url = "https://en.wikipedia.org/wiki/Nasdaq-100"

    try:

        tables = pd.read_html(url)

        selected = None

        for table in tables:

            columns = [str(c) for c in table.columns]

            ticker_candidates = [
                c for c in columns
                if "Ticker" in c or "Symbol" in c
            ]

            if ticker_candidates:
                selected = table.copy()
                break

        if selected is None:
            raise ValueError("Nasdaq-100 ticker table not found")

        columns = list(selected.columns)

        ticker_col = None

        for col in columns:
            col_str = str(col)

            if "Ticker" in col_str or "Symbol" in col_str:
                ticker_col = col
                break

        selected["Ticker"] = (
            selected[ticker_col]
            .astype(str)
            .map(normalize_ticker)
        )

        company_col = None

        for col in columns:
            if "Company" in str(col):
                company_col = col
                break

        if company_col:
            selected["Company"] = selected[company_col]
        else:
            selected["Company"] = selected["Ticker"]

        sector_col = None

        for col in columns:
            if "GICS Sector" in str(col):
                sector_col = col
                break

        industry_col = None

        for col in columns:
            if "GICS Sub-Industry" in str(col):
                industry_col = col
                break

        if sector_col:
            selected["Sector"] = selected[sector_col]
        else:
            selected["Sector"] = "Unknown"

        if industry_col:
            selected["Industry"] = selected[industry_col]
        else:
            selected["Industry"] = "Unknown"

        selected["Index"] = "Nasdaq 100"

        return selected[
            [
                "Ticker",
                "Company",
                "Sector",
                "Industry",
                "Index",
            ]
        ].drop_duplicates("Ticker")

    except Exception as exc:

        st.warning(
            f"Не удалось загрузить Nasdaq 100 автоматически: {exc}"
        )

        return pd.DataFrame(
            columns=[
                "Ticker",
                "Company",
                "Sector",
                "Industry",
                "Index",
            ]
        )


@st.cache_data(ttl=3600, show_spinner=False)
def load_russell2000_universe():
    """
    Load Russell 2000 proxy universe from iShares IWM holdings.

    iShares publishes the current holdings of the Russell 2000
    tracking ETF. The application attempts the downloadable CSV
    first and falls back to the HTML holdings table.
    """

    csv_url = (
        "https://www.ishares.com/us/products/239710/"
        "ishares-russell-2000-etf/"
        "1467271812596.ajax"
        "?fileType=csv"
        "&fileName=IWM_holdings"
        "&dataType=fund"
    )

    try:

        response = requests.get(
            csv_url,
            timeout=30,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 "
                    "(Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 "
                    "Chrome/120 Safari/537.36"
                )
            },
        )

        response.raise_for_status()

        raw = response.content

        table = None

        for skip in range(0, 16):

            try:
                candidate = pd.read_csv(
                    io.BytesIO(raw),
                    skiprows=skip,
                )

                normalized = {
                    str(c).strip().lower(): c
                    for c in candidate.columns
                }

                if "ticker" in normalized:
                    table = candidate
                    break

            except Exception:
                continue

        if table is None:
            raise ValueError(
                "Russell 2000 CSV did not contain Ticker column"
            )

        columns_lower = {
            str(c).strip().lower(): c
            for c in table.columns
        }

        ticker_col = columns_lower["ticker"]

        sector_col = columns_lower.get("sector")

        name_col = None

        for key, original in columns_lower.items():
            if key in {"name", "security"}:
                name_col = original
                break

        table["Ticker"] = (
            table[ticker_col]
            .astype(str)
            .map(normalize_ticker)
        )

        if name_col:
            table["Company"] = table[name_col].astype(str)
        else:
            table["Company"] = table["Ticker"]

        if sector_col:
            table["Sector"] = (
                table[sector_col]
                .fillna("Unknown")
                .astype(str)
            )
        else:
            table["Sector"] = "Unknown"

        table["Industry"] = "Unknown"
        table["Index"] = "Russell 2000"

        table = table[
            table["Ticker"].notna()
            & (table["Ticker"] != "")
            & (~table["Ticker"].isin(["USD", "N/A", "NAN"]))
        ]

        return table[
            [
                "Ticker",
                "Company",
                "Sector",
                "Industry",
                "Index",
            ]
        ].drop_duplicates("Ticker")

    except Exception:

        # Fallback: a small set of known current examples.
        # This keeps the application functional if iShares blocks
        # the downloadable CSV endpoint.

        fallback = [
            ("IWM", "iShares Russell 2000 ETF", "ETF"),
            ("MOGA", "Mogo", "Financials"),
            ("BTSG", "BrightSpring", "Health Care"),
            ("UMBF", "UMB Financial", "Financials"),
            ("CYTK", "Cytokinetics", "Health Care"),
            ("CTRE", "CareTrust REIT", "Real Estate"),
            ("ONB", "Old National Bancorp", "Financials"),
            ("KRYS", "Krystal Biotech", "Health Care"),
            ("GKOS", "Glaukos", "Health Care"),
            ("HUT", "Hut 8", "Information Technology"),
        ]

        return pd.DataFrame(
            fallback,
            columns=["Ticker", "Company", "Sector"],
        ).assign(
            Industry="Unknown",
            Index="Russell 2000",
        )


@st.cache_data(ttl=3600, show_spinner=False)
def load_us_universe(selected_indices):
    """
    Combine selected US index universes.
    """

    frames = []

    if "S&P 500" in selected_indices:
        frames.append(load_sp500_universe())

    if "Nasdaq 100" in selected_indices:
        frames.append(load_nasdaq100_universe())

    if "Russell 2000" in selected_indices:
        frames.append(load_russell2000_universe())

    if not frames:
        return pd.DataFrame(
            columns=[
                "Ticker",
                "Company",
                "Sector",
                "Industry",
                "Index",
            ]
        )

    result = pd.concat(
        frames,
        ignore_index=True,
    )

    result["Ticker"] = (
        result["Ticker"]
        .astype(str)
        .map(normalize_ticker)
    )

    result = result.drop_duplicates(
        subset=["Ticker"],
        keep="first",
    )

    return result


# ============================================================
# CRYPTO
# ============================================================

@st.cache_data(ttl=1800, show_spinner=False)
def load_top_crypto():
    """
    Load top 50 crypto assets by market cap using CoinGecko.
    """

    url = (
        "https://api.coingecko.com/api/v3/coins/markets"
        "?vs_currency=usd"
        "&order=market_cap_desc"
        "&per_page=50"
        "&page=1"
        "&sparkline=false"
    )

    try:

        response = requests.get(
            url,
            timeout=20,
            headers={
                "User-Agent": "PairTradeFinder/1.0"
            },
        )

        response.raise_for_status()

        data = response.json()

        rows = []

        for item in data:

            symbol = str(
                item.get("symbol", "")
            ).upper()

            if not symbol:
                continue

            ticker = f"{symbol}-USD"

            rows.append(
                {
                    "Ticker": ticker,
                    "Company": item.get(
                        "name",
                        symbol,
                    ),
                    "Sector": "Cryptocurrency",
                    "Industry": "Crypto",
                    "Index": "Top 50 Crypto",
                    "Market Cap": item.get(
                        "market_cap",
                        np.nan,
                    ),
                    "Rank": item.get(
                        "market_cap_rank",
                        np.nan,
                    ),
                }
            )

        result = pd.DataFrame(rows)

        if result.empty:
            raise ValueError(
                "CoinGecko returned no assets"
            )

        result = result.drop_duplicates(
            "Ticker"
        )

        return result

    except Exception as exc:

        st.warning(
            "CoinGecko недоступен. "
            "Используется резервный список crypto."
        )

        return pd.DataFrame(
            [
                {
                    "Ticker": ticker,
                    "Company": ticker.replace(
                        "-USD",
                        "",
                    ),
                    "Sector": "Cryptocurrency",
                    "Industry": "Crypto",
                    "Index": "Top 50 Crypto",
                    "Market Cap": np.nan,
                    "Rank": np.nan,
                }
                for ticker in FALLBACK_CRYPTO
            ]
        )


# ============================================================
# OTHER UNIVERSES
# ============================================================

def build_futures_universe(selected_groups):

    rows = []

    for group in selected_groups:

        mapping = FUTURES_UNIVERSES.get(
            group,
            {},
        )

        for ticker, name in mapping.items():

            rows.append(
                {
                    "Ticker": ticker,
                    "Company": name,
                    "Sector": group,
                    "Industry": "Futures",
                    "Index": "Futures",
                }
            )

    return pd.DataFrame(rows)


def build_forex_universe():

    rows = []

    for ticker, name in FOREX_UNIVERSE.items():

        rows.append(
            {
                "Ticker": ticker,
                "Company": name,
                "Sector": "Forex",
                "Industry": "FX",
                "Index": "Forex",
            }
        )

    return pd.DataFrame(rows)


def build_custom_universe(text):

    tickers = clean_ticker_list(
        text.replace(
            "\n",
            ",",
        ).split(",")
    )

    return pd.DataFrame(
        [
            {
                "Ticker": ticker,
                "Company": ticker,
                "Sector": "Custom",
                "Industry": "Custom",
                "Index": "Custom",
            }
            for ticker in tickers
        ]
    )


# ============================================================
# YFINANCE DATA
# ============================================================

def effective_yfinance_period(
    interval,
    requested_period,
):
    """
    Yahoo/yfinance imposes stricter limits on intraday data.

    15m:
        last ~60 days

    1h:
        up to ~730 days

    Daily:
        3mo - 2y as selected by user
    """

    if interval == "15m":
        return "60d"

    if interval == "1h":
        return "730d"

    return requested_period


@st.cache_data(
    ttl=1800,
    show_spinner=False,
)
def download_prices(
    tickers_tuple,
    interval,
    requested_period,
):
    """
    Batch download prices from Yahoo Finance.
    """

    tickers = list(tickers_tuple)

    if not tickers:
        return pd.DataFrame()

    period = effective_yfinance_period(
        interval,
        requested_period,
    )

    all_frames = []

    # Smaller chunks reduce Yahoo request failures.
    chunk_size = 40

    for start in range(
        0,
        len(tickers),
        chunk_size,
    ):

        chunk = tickers[
            start:start + chunk_size
        ]

        try:

            raw = yf.download(
                tickers=chunk,
                period=period,
                interval=interval,
                auto_adjust=True,
                progress=False,
                threads=True,
                group_by="column",
                repair=True,
                timeout=20,
            )

        except Exception:
            continue

        if raw is None or raw.empty:
            continue

        try:

            if isinstance(
                raw.columns,
                pd.MultiIndex,
            ):

                level0 = list(
                    raw.columns.get_level_values(0)
                )

                level1 = list(
                    raw.columns.get_level_values(1)
                )

                if "Close" in level0:

                    close = raw["Close"].copy()

                elif "Close" in level1:

                    close = raw.xs(
                        "Close",
                        axis=1,
                        level=1,
                    ).copy()

                else:
                    continue

            else:

                if "Close" not in raw.columns:
                    continue

                close = raw["Close"].copy()

                if isinstance(
                    close,
                    pd.Series,
                ):
                    close = close.to_frame()

            if isinstance(
                close,
                pd.Series,
            ):
                close = close.to_frame()

            close.columns = [
                normalize_ticker(c)
                for c in close.columns
            ]

            all_frames.append(close)

        except Exception:
            continue

    if not all_frames:
        return pd.DataFrame()

    result = pd.concat(
        all_frames,
        axis=1,
    )

    result = result.loc[
        ~result.index.duplicated(
            keep="last"
        )
    ]

    result = result.sort_index()

    result = result.loc[
        :,
        ~result.columns.duplicated()
    ]

    result = result.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return result


# ============================================================
# CORRELATION ENGINE
# ============================================================

def build_correlation_candidates(
    prices,
    threshold,
    max_pairs,
    correlation_basis="Returns",
):
    """
    Create candidate pairs using Pearson correlation.

    Returns:
        list of tuples:
            (ticker_y, ticker_x, correlation)
    """

    if prices.empty:
        return []

    if correlation_basis == "Returns":

        matrix = prices.pct_change(
            fill_method=None
        ).replace(
            [np.inf, -np.inf],
            np.nan,
        )

    else:

        matrix = prices.copy()

    corr_matrix = matrix.corr(
        method="pearson",
        min_periods=30,
    )

    candidates = []

    columns = list(
        corr_matrix.columns
    )

    for i in range(
        len(columns)
    ):

        ticker_a = columns[i]

        for j in range(
            i + 1,
            len(columns),
        ):

            ticker_b = columns[j]

            corr = safe_float(
                corr_matrix.loc[
                    ticker_a,
                    ticker_b,
                ]
            )

            if not np.isfinite(corr):
                continue

            if abs(corr) < threshold:
                continue

            candidates.append(
                (
                    ticker_a,
                    ticker_b,
                    corr,
                )
            )

    candidates.sort(
        key=lambda x: abs(x[2]),
        reverse=True,
    )

    return candidates[:max_pairs]


# ============================================================
# OLS / ADF / HALF LIFE
# ============================================================

def calculate_pair_statistics(
    y,
    x,
):
    """
    OLS:
        y = alpha + beta*x + residual

    Then ADF on residual.
    """

    df = pd.concat(
        [
            y.rename("y"),
            x.rename("x"),
        ],
        axis=1,
    ).dropna()

    if len(df) < 80:
        return None

    y_values = df["y"].astype(float)
    x_values = df["x"].astype(float)

    if (
        y_values.nunique() < 3
        or x_values.nunique() < 3
    ):
        return None

    try:

        X = sm.add_constant(
            x_values
        )

        model = sm.OLS(
            y_values,
            X,
        ).fit()

        alpha = safe_float(
            model.params.iloc[0]
        )

        beta = safe_float(
            model.params.iloc[1]
        )

        spread = (
            y_values
            - alpha
            - beta * x_values
        )

        spread = spread.replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna()

        if len(spread) < 60:
            return None

        try:

            adf_result = adfuller(
                spread.values,
                autolag="AIC",
            )

            adf_stat = safe_float(
                adf_result[0]
            )

            adf_pvalue = safe_float(
                adf_result[1]
            )

            adf_lag = int(
                adf_result[2]
            )

        except Exception:

            adf_stat = np.nan
            adf_pvalue = np.nan
            adf_lag = np.nan

        # Half-life:
        #
        # delta(S_t) = lambda*S_(t-1) + c + error
        #
        # HL = -ln(2) / lambda
        #

        lagged = spread.shift(1)
        delta = spread.diff()

        hl_df = pd.concat(
            [
                delta.rename("delta"),
                lagged.rename("lagged"),
            ],
            axis=1,
        ).dropna()

        half_life = np.nan

        if len(hl_df) >= 30:

            try:

                hl_X = sm.add_constant(
                    hl_df["lagged"]
                )

                hl_model = sm.OLS(
                    hl_df["delta"],
                    hl_X,
                ).fit()

                lam = safe_float(
                    hl_model.params.iloc[1]
                )

                if (
                    np.isfinite(lam)
                    and lam < 0
                ):
                    half_life = (
                        -math.log(2.0)
                        / lam
                    )

                    if half_life < 0:
                        half_life = np.nan

                    if half_life > 1_000_000:
                        half_life = np.nan

            except Exception:
                half_life = np.nan

        return {
            "alpha": alpha,
            "beta": beta,
            "adf_stat": adf_stat,
            "adf_pvalue": adf_pvalue,
            "adf_lag": adf_lag,
            "half_life": half_life,
            "spread": spread,
            "aligned_y": y_values,
            "aligned_x": x_values,
        }

    except Exception:
        return None


# ============================================================
# Z SCORE
# ============================================================

def calculate_zscore(
    spread,
    window,
):
    """
    Rolling Z-Score.
    """

    rolling_mean = (
        spread
        .rolling(
            window=window,
            min_periods=max(
                20,
                window // 2,
            ),
        )
        .mean()
    )

    rolling_std = (
        spread
        .rolling(
            window=window,
            min_periods=max(
                20,
                window // 2,
            ),
        )
        .std()
    )

    z = (
        spread - rolling_mean
    ) / rolling_std.replace(
        0,
        np.nan,
    )

    return z


# ============================================================
# BACKTEST ENGINE
# ============================================================

def backtest_pair(
    y,
    x,
    beta,
    alpha,
    entry_z,
    exit_z,
    z_window,
    initial_capital,
):
    """
    Historical pairs-trading simulation.

    Position:
        +1 = Long Spread
        -1 = Short Spread
         0 = Flat

    Long Spread:
        Long Y
        Short Beta * X

    Short Spread:
        Short Y
        Long Beta * X

    Position is opened using current bar prices
    after evaluating current Z-Score.

    PnL is then accumulated from subsequent price changes.
    """

    df = pd.concat(
        [
            y.rename("Y"),
            x.rename("X"),
        ],
        axis=1,
    ).dropna()

    if len(df) < max(
        80,
        z_window + 10,
    ):
        return {
            "win_rate": np.nan,
            "total_pnl": np.nan,
            "trades": 0,
            "max_drawdown": np.nan,
            "equity_curve": pd.Series(
                dtype=float
            ),
            "zscore": pd.Series(
                dtype=float
            ),
            "trades_detail": [],
        }

    spread = (
        df["Y"]
        - alpha
        - beta * df["X"]
    )

    z = calculate_zscore(
        spread,
        z_window,
    )

    equity = float(
        initial_capital
    )

    equity_values = []
    equity_dates = []

    position = 0

    shares_y = 0.0
    shares_x = 0.0

    entry_equity = equity
    entry_date = None
    entry_y = np.nan
    entry_x = np.nan
    entry_z = np.nan

    trades = []

    dates = list(
        df.index
    )

    for i in range(
        1,
        len(df),
    ):

        current_date = dates[i]
        previous_date = dates[i - 1]

        current_y = safe_float(
            df.loc[
                current_date,
                "Y",
            ]
        )

        previous_y = safe_float(
            df.loc[
                previous_date,
                "Y",
            ]
        )

        current_x = safe_float(
            df.loc[
                current_date,
                "X",
            ]
        )

        previous_x = safe_float(
            df.loc[
                previous_date,
                "X",
            ]
        )

        if not all(
            np.isfinite(v)
            for v in [
                current_y,
                previous_y,
                current_x,
                previous_x,
            ]
        ):
            equity_values.append(
                equity
            )
            equity_dates.append(
                current_date
            )
            continue

        # ----------------------------------------------------
        # Mark-to-market existing position
        # ----------------------------------------------------

        if position != 0:

            pnl_y = (
                shares_y
                * (current_y - previous_y)
            )

            pnl_x = (
                -position
                * beta
                * shares_x
                * (current_x - previous_x)
            )

            pnl = (
                position * pnl_y
                + pnl_x
            )

            equity += pnl

        equity_values.append(
            equity
        )

        equity_dates.append(
            current_date
        )

        current_z = safe_float(
            z.iloc[i]
        )

        if not np.isfinite(current_z):
            continue

        # ----------------------------------------------------
        # ENTRY
        # ----------------------------------------------------

        if position == 0:

            if current_z <= -entry_z:

                position = 1

                shares_y = (
                    initial_capital
                    / (
                        2.0
                        * max(
                            current_y,
                            1e-9,
                        )
                    )
                )

                shares_x = (
                    initial_capital
                    / (
                        2.0
                        * max(
                            current_x,
                            1e-9,
                        )
                    )
                )

                entry_equity = equity
                entry_date = current_date
                entry_y = current_y
                entry_x = current_x
                entry_z = current_z

            elif current_z >= entry_z:

                position = -1

                shares_y = (
                    initial_capital
                    / (
                        2.0
                        * max(
                            current_y,
                            1e-9,
                        )
                    )
                )

                shares_x = (
                    initial_capital
                    / (
                        2.0
                        * max(
                            current_x,
                            1e-9,
                        )
                    )
                )

                entry_equity = equity
                entry_date = current_date
                entry_y = current_y
                entry_x = current_x
                entry_z = current_z

        # ----------------------------------------------------
        # EXIT
        # ----------------------------------------------------

        elif abs(current_z) <= exit_z:

            trade_pnl = (
                equity
                - entry_equity
            )

            direction = (
                "LONG SPREAD"
                if position == 1
                else "SHORT SPREAD"
            )

            trades.append(
                {
                    "Entry Date": entry_date,
                    "Exit Date": current_date,
                    "Direction": direction,
                    "Entry Z": entry_z,
                    "Exit Z": current_z,
                    "Entry Y": entry_y,
                    "Exit Y": current_y,
                    "Entry X": entry_x,
                    "Exit X": current_x,
                    "PnL": trade_pnl,
                }
            )

            position = 0

            shares_y = 0.0
            shares_x = 0.0

            entry_date = None

    # --------------------------------------------------------
    # Force close open trade at final bar
    # --------------------------------------------------------

    if position != 0:

        final_date = dates[-1]

        final_y = safe_float(
            df.loc[
                final_date,
                "Y",
            ]
        )

        final_x = safe_float(
            df.loc[
                final_date,
                "X",
            ]
        )

        # No extra movement is added here because
        # the latest bar has already been marked-to-market.

        final_z = safe_float(
            z.iloc[-1]
        )

        trade_pnl = (
            equity
            - entry_equity
        )

        direction = (
            "LONG SPREAD"
            if position == 1
            else "SHORT SPREAD"
        )

        trades.append(
            {
                "Entry Date": entry_date,
                "Exit Date": final_date,
                "Direction": direction,
                "Entry Z": entry_z,
                "Exit Z": final_z,
                "Entry Y": entry_y,
                "Exit Y": final_y,
                "Entry X": entry_x,
                "Exit X": final_x,
                "PnL": trade_pnl,
            }
        )

    equity_series = pd.Series(
        equity_values,
        index=pd.DatetimeIndex(
            equity_dates
        ),
        name="Equity",
    )

    if equity_series.empty:

        max_drawdown = np.nan

    else:

        running_max = (
            equity_series
            .cummax()
        )

        drawdown = (
            equity_series
            - running_max
        )

        max_drawdown = safe_float(
            drawdown.min()
        )

    trade_count = len(
        trades
    )

    if trade_count > 0:

        winning = sum(
            1
            for trade in trades
            if trade["PnL"] > 0
        )

        win_rate = (
            winning
            / trade_count
            * 100.0
        )

    else:

        win_rate = np.nan

    total_pnl = (
        equity
        - initial_capital
    )

    return {
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "trades": trade_count,
        "max_drawdown": max_drawdown,
        "equity_curve": equity_series,
        "zscore": z,
        "spread": spread,
        "trades_detail": trades,
    }


# ============================================================
# FULL SCREEN
# ============================================================

def run_screener(
    prices,
    metadata,
    correlation_threshold,
    pvalue_threshold,
    winrate_threshold,
    max_pairs,
    max_backtests,
    z_window,
    entry_z,
    exit_z,
    initial_capital,
    correlation_basis,
    same_sector_only,
):
    """
    Main quantitative pipeline.
    """

    if prices.empty:
        return pd.DataFrame(), {}

    candidates = build_correlation_candidates(
        prices=prices,
        threshold=correlation_threshold,
        max_pairs=max_pairs,
        correlation_basis=correlation_basis,
    )

    if not candidates:
        return pd.DataFrame(), {}

    metadata_map = {}

    for _, row in metadata.iterrows():

        metadata_map[
            row["Ticker"]
        ] = row.to_dict()

    results = []

    pair_details = {}

    processed_backtests = 0

    for ticker_y, ticker_x, corr in candidates:

        if ticker_y not in prices.columns:
            continue

        if ticker_x not in prices.columns:
            continue

        meta_y = metadata_map.get(
            ticker_y,
            {
                "Company": ticker_y,
                "Sector": "Unknown",
                "Industry": "Unknown",
                "Index": "Unknown",
            },
        )

        meta_x = metadata_map.get(
            ticker_x,
            {
                "Company": ticker_x,
                "Sector": "Unknown",
                "Industry": "Unknown",
                "Index": "Unknown",
            },
        )

        if same_sector_only:

            sector_y = str(
                meta_y.get(
                    "Sector",
                    "Unknown",
                )
            )

            sector_x = str(
                meta_x.get(
                    "Sector",
                    "Unknown",
                )
            )

            if (
                sector_y != sector_x
                or sector_y == "Unknown"
            ):
                continue

        pair_df = pd.concat(
            [
                prices[ticker_y].rename(
                    ticker_y
                ),
                prices[ticker_x].rename(
                    ticker_x
                ),
            ],
            axis=1,
        ).dropna()

        if len(pair_df) < max(
            80,
            z_window + 20,
        ):
            continue

        stats_result = calculate_pair_statistics(
            pair_df[ticker_y],
            pair_df[ticker_x],
        )

        if stats_result is None:
            continue

        pvalue = safe_float(
            stats_result["adf_pvalue"]
        )

        if not np.isfinite(pvalue):
            continue

        if pvalue > pvalue_threshold:
            continue

        beta = safe_float(
            stats_result["beta"]
        )

        alpha = safe_float(
            stats_result["alpha"]
        )

        spread = stats_result[
            "spread"
        ]

        z = calculate_zscore(
            spread,
            z_window,
        )

        current_z = safe_float(
            z.dropna().iloc[-1]
            if not z.dropna().empty
            else np.nan
        )

        # ----------------------------------------------------
        # BACKTEST
        # ----------------------------------------------------

        backtest_result = {
            "win_rate": np.nan,
            "total_pnl": np.nan,
            "trades": 0,
            "max_drawdown": np.nan,
            "equity_curve": pd.Series(
                dtype=float
            ),
            "zscore": z,
            "spread": spread,
            "trades_detail": [],
        }

        if processed_backtests < max_backtests:

            backtest_result = backtest_pair(
                y=pair_df[ticker_y],
                x=pair_df[ticker_x],
                beta=beta,
                alpha=alpha,
                entry_z=entry_z,
                exit_z=exit_z,
                z_window=z_window,
                initial_capital=initial_capital,
            )

            processed_backtests += 1

        win_rate = safe_float(
            backtest_result[
                "win_rate"
            ]
        )

        if (
            np.isfinite(win_rate)
            and win_rate < winrate_threshold
        ):
            continue

        sector_y = str(
            meta_y.get(
                "Sector",
                "Unknown",
            )
        )

        sector_x = str(
            meta_x.get(
                "Sector",
                "Unknown",
            )
        )

        industry_y = str(
            meta_y.get(
                "Industry",
                "Unknown",
            )
        )

        industry_x = str(
            meta_x.get(
                "Industry",
                "Unknown",
            )
        )

        if sector_y == sector_x:

            pair_sector = sector_y

        else:

            pair_sector = (
                f"{sector_y} / {sector_x}"
            )

        if industry_y == industry_x:

            pair_industry = industry_y

        else:

            pair_industry = (
                f"{industry_y} / {industry_x}"
            )

        pair_name = (
            f"{ticker_y} / {ticker_x}"
        )

        result_row = {
            "Pair": pair_name,
            "Y": ticker_y,
            "X": ticker_x,
            "Sector": pair_sector,
            "Industry": pair_industry,
            "Corr": corr,
            "p-value": pvalue,
            "Beta": beta,
            "Alpha": alpha,
            "Half-Life": safe_float(
                stats_result[
                    "half_life"
                ]
            ),
            "Current Z": current_z,
            "WinRate %": win_rate,
            "Total PnL": safe_float(
                backtest_result[
                    "total_pnl"
                ]
            ),
            "Trades": int(
                backtest_result[
                    "trades"
                ]
            ),
            "Max Drawdown": safe_float(
                backtest_result[
                    "max_drawdown"
                ]
            ),
            "Observations": len(
                pair_df
            ),
        }

        results.append(
            result_row
        )

        pair_details[
            pair_name
        ] = {
            "y": ticker_y,
            "x": ticker_x,
            "metadata_y": meta_y,
            "metadata_x": meta_x,
            "alpha": alpha,
            "beta": beta,
            "corr": corr,
            "pvalue": pvalue,
            "half_life": safe_float(
                stats_result[
                    "half_life"
                ]
            ),
            "spread": spread,
            "zscore": backtest_result[
                "zscore"
            ],
            "equity_curve": backtest_result[
                "equity_curve"
            ],
            "trades": backtest_result[
                "trades_detail"
            ],
            "backtest": backtest_result,
            "prices_y": pair_df[
                ticker_y
            ],
            "prices_x": pair_df[
                ticker_x
            ],
        }

    results_df = pd.DataFrame(
        results
    )

    if results_df.empty:
        return results_df, pair_details

    results_df = results_df.sort_values(
        by=[
            "p-value",
            "WinRate %",
            "Corr",
        ],
        ascending=[
            True,
            False,
            False,
        ],
        na_position="last",
    )

    results_df = results_df.reset_index(
        drop=True
    )

    return (
        results_df,
        pair_details,
    )


# ============================================================
# PLOTLY CHARTS
# ============================================================

def make_zscore_chart(
    zscore,
    entry_z,
    exit_z,
    pair_name,
):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=zscore.index,
            y=zscore.values,
            mode="lines",
            name="Z-Score",
            line=dict(
                color="#38bdf8",
                width=2,
            ),
        )
    )

    fig.add_hline(
        y=entry_z,
        line_color="#ef4444",
        line_dash="dash",
        annotation_text="+ Entry",
    )

    fig.add_hline(
        y=-entry_z,
        line_color="#22c55e",
        line_dash="dash",
        annotation_text="- Entry",
    )

    fig.add_hline(
        y=exit_z,
        line_color="#a3a3a3",
        line_dash="dot",
        annotation_text="+ Exit",
    )

    fig.add_hline(
        y=-exit_z,
        line_color="#a3a3a3",
        line_dash="dot",
        annotation_text="- Exit",
    )

    fig.add_hline(
        y=0,
        line_color="#ffffff",
        line_width=1,
    )

    fig.update_layout(
        title=f"Z-Score — {pair_name}",
        template="plotly_dark",
        height=450,
        hovermode="x unified",
        legend=dict(
            orientation="h"
        ),
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20,
        ),
    )

    return fig


def make_normalized_price_chart(
    y,
    x,
    ticker_y,
    ticker_x,
):
    aligned = pd.concat(
        [
            y.rename(ticker_y),
            x.rename(ticker_x),
        ],
        axis=1,
    ).dropna()

    if aligned.empty:
        return go.Figure()

    normalized = (
        aligned
        / aligned.iloc[0]
        * 100.0
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=normalized.index,
            y=normalized[ticker_y],
            mode="lines",
            name=ticker_y,
            line=dict(
                color="#38bdf8",
                width=2,
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=normalized.index,
            y=normalized[ticker_x],
            mode="lines",
            name=ticker_x,
            line=dict(
                color="#f59e0b",
                width=2,
            ),
        )
    )

    fig.update_layout(
        title="Normalized Prices (Base = 100)",
        template="plotly_dark",
        height=450,
        hovermode="x unified",
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20,
        ),
    )

    return fig


def make_equity_chart(
    equity,
    pair_name,
    initial_capital,
):
    fig = go.Figure()

    if equity is not None and not equity.empty:

        fig.add_trace(
            go.Scatter(
                x=equity.index,
                y=equity.values,
                mode="lines",
                name="Equity",
                fill="tozeroy",
                line=dict(
                    color="#22c55e",
                    width=2,
                ),
            )
        )

    fig.add_hline(
        y=initial_capital,
        line_color="#94a3b8",
        line_dash="dash",
        annotation_text="Initial Capital",
    )

    fig.update_layout(
        title=f"Equity Curve — {pair_name}",
        template="plotly_dark",
        height=400,
        hovermode="x unified",
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20,
        ),
    )

    return fig


# ============================================================
# EXPORT
# ============================================================

@st.cache_data
def dataframe_to_csv(df):
    return df.to_csv(
        index=False
    ).encode(
        "utf-8-sig"
    )


def dataframe_to_excel(df):
    output = io.BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl",
    ) as writer:

        df.to_excel(
            writer,
            sheet_name="Screener Results",
            index=False,
        )

    output.seek(0)

    return output.getvalue()


def trades_to_dataframe(
    trades,
):
    if not trades:
        return pd.DataFrame()

    return pd.DataFrame(
        trades
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    "## 📊 PairTrade Finder"
)

st.sidebar.caption(
    "Cointegration & Pairs Trading Screener"
)

st.sidebar.divider()

market_type = st.sidebar.selectbox(
    "🌍 Market",
    [
        "US Stocks",
        "Futures",
        "Forex",
        "Crypto",
        "Custom Tickers",
    ],
)


# ============================================================
# MARKET SELECTION
# ============================================================

metadata = pd.DataFrame()

selected_sectors = []
selected_industries = []

if market_type == "US Stocks":

    selected_indices = st.sidebar.multiselect(
        "📚 Universe",
        [
            "S&P 500",
            "Nasdaq 100",
            "Russell 2000",
        ],
        default=[
            "S&P 500"
        ],
    )

    if selected_indices:

        with st.spinner(
            "Загрузка списков акций..."
        ):

            metadata = load_us_universe(
                tuple(selected_indices)
            )

        if metadata.empty:

            st.sidebar.error(
                "Не удалось получить US universe."
            )

        sector_options = sorted(
            [
                x
                for x in metadata[
                    "Sector"
                ]
                .dropna()
                .unique()
                if str(x).strip()
                and str(x) != "Unknown"
            ]
        )

        selected_sectors = st.sidebar.multiselect(
            "🏭 Sectors",
            sector_options,
            default=[],
        )

        industry_source = metadata.copy()

        if selected_sectors:

            industry_source = (
                industry_source[
                    industry_source[
                        "Sector"
                    ].isin(
                        selected_sectors
                    )
                ]
            )

        industry_options = sorted(
            [
                x
                for x in industry_source[
                    "Industry"
                ]
                .dropna()
                .unique()
                if str(x).strip()
                and str(x) != "Unknown"
            ]
        )

        selected_industries = st.sidebar.multiselect(
            "🔬 Industries",
            industry_options,
            default=[],
        )

        if selected_sectors:

            metadata = metadata[
                metadata[
                    "Sector"
                ].isin(
                    selected_sectors
                )
            ]

        if selected_industries:

            metadata = metadata[
                metadata[
                    "Industry"
                ].isin(
                    selected_industries
                )
            ]

else:

    if market_type == "Futures":

        future_groups = st.sidebar.multiselect(
            "📚 Futures Groups",
            list(
                FUTURES_UNIVERSES.keys()
            ),
            default=[
                "CME Index Futures"
            ],
        )

        metadata = build_futures_universe(
            future_groups
        )

    elif market_type == "Forex":

        metadata = build_forex_universe()

    elif market_type == "Crypto":

        with st.spinner(
            "Загрузка Top-50 crypto..."
        ):
            metadata = load_top_crypto()

    elif market_type == "Custom Tickers":

        custom_text = st.sidebar.text_area(
            "Введите тикеры через запятую",
            value=(
                "AAPL,MSFT,GOOGL,AMZN,"
                "META,NVDA,AMD,INTC"
            ),
            height=120,
        )

        metadata = build_custom_universe(
            custom_text
        )


# ============================================================
# DATA SETTINGS
# ============================================================

st.sidebar.divider()

interval = st.sidebar.selectbox(
    "⏱ Timeframe",
    [
        "1d",
        "1h",
        "15m",
    ],
    index=0,
)

history_period = st.sidebar.selectbox(
    "📅 History",
    [
        "3mo",
        "6mo",
        "1y",
        "2y",
    ],
    index=2,
)

if interval == "15m":

    st.sidebar.warning(
        "15m: Yahoo ограничивает "
        "историю примерно последними 60 днями."
    )

elif interval == "1h":

    st.sidebar.info(
        "1h: Yahoo ограничивает "
        "историю примерно последними 730 днями."
    )


# ============================================================
# UNIVERSE LIMIT
# ============================================================

st.sidebar.divider()

max_universe = st.sidebar.slider(
    "Максимум тикеров для скринера",
    min_value=20,
    max_value=400,
    value=100,
    step=10,
    help=(
        "Ограничение защищает бесплатный "
        "Yahoo/yfinance от слишком большого "
        "числа запросов."
    ),
)


# ============================================================
# QUANT SETTINGS
# ============================================================

st.sidebar.markdown(
    "### 🧮 Quant Filters"
)

correlation_threshold = st.sidebar.slider(
    "Pearson |Correlation|",
    min_value=0.50,
    max_value=0.99,
    value=0.80,
    step=0.01,
)

pvalue_threshold = st.sidebar.slider(
    "ADF p-value",
    min_value=0.01,
    max_value=0.20,
    value=0.05,
    step=0.01,
)

winrate_threshold = st.sidebar.slider(
    "Minimum Win Rate %",
    min_value=0.0,
    max_value=100.0,
    value=50.0,
    step=1.0,
)

correlation_basis = st.sidebar.selectbox(
    "Correlation basis",
    [
        "Returns",
        "Prices",
    ],
    index=0,
)

same_sector_only = st.sidebar.checkbox(
    "Только пары внутри одного сектора",
    value=False,
)


# ============================================================
# BACKTEST SETTINGS
# ============================================================

st.sidebar.markdown(
    "### 📈 Backtest"
)

entry_z = st.sidebar.slider(
    "Entry Z-Score",
    min_value=1.0,
    max_value=4.0,
    value=2.0,
    step=0.1,
)

exit_z = st.sidebar.slider(
    "Exit Z-Score",
    min_value=0.0,
    max_value=1.0,
    value=0.2,
    step=0.1,
)

z_window = st.sidebar.slider(
    "Rolling Z-Score Window",
    min_value=20,
    max_value=250,
    value=60,
    step=10,
)

initial_capital = st.sidebar.number_input(
    "Initial Capital ($)",
    min_value=1000.0,
    max_value=1_000_000.0,
    value=10_000.0,
    step=1000.0,
)

max_candidate_pairs = st.sidebar.slider(
    "Максимум корреляционных кандидатов",
    min_value=100,
    max_value=20_000,
    value=3_000,
    step=100,
)

max_backtests = st.sidebar.slider(
    "Максимум Backtests",
    min_value=25,
    max_value=2_000,
    value=300,
    step=25,
)


# ============================================================
# APPLY UNIVERSE LIMIT
# ============================================================

if not metadata.empty:

    metadata = metadata.copy()

    metadata["Ticker"] = (
        metadata["Ticker"]
        .astype(str)
        .map(normalize_ticker)
    )

    metadata = metadata[
        metadata["Ticker"] != ""
    ]

    metadata = metadata.drop_duplicates(
        "Ticker"
    )

    # Random-free deterministic sampling:
    # alphabetic ordering.
    metadata = metadata.sort_values(
        "Ticker"
    )

    if len(metadata) > max_universe:

        metadata = metadata.head(
            max_universe
        )

# ============================================================
# RUN BUTTON
# ============================================================

st.sidebar.divider()

run_button = st.sidebar.button(
    "🚀 ЗАПУСТИТЬ СКРИНЕР",
    type="primary",
    use_container_width=True,
)


# ============================================================
# HEADER
# ============================================================

st.title(
    "📊 PairTrade Finder"
)

st.caption(
    "Cointegration • OLS • ADF • Half-Life • Z-Score • Backtest"
)

if not metadata.empty:

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Universe",
        len(metadata),
    )

    c2.metric(
        "Timeframe",
        interval,
    )

    c3.metric(
        "History",
        effective_yfinance_period(
            interval,
            history_period,
        ),
    )

    c4.metric(
        "Corr Threshold",
        f"{correlation_threshold:.2f}",
    )

else:

    st.info(
        "Выберите рынок и параметры слева."
    )


# ============================================================
# RUN SCREENER
# ============================================================

if run_button:

    if metadata.empty:

        st.error(
            "Universe пуст. "
            "Выберите тикеры или другой рынок."
        )

        st.stop()

    tickers = clean_ticker_list(
        metadata["Ticker"].tolist()
    )

    if len(tickers) < 2:

        st.error(
            "Нужно минимум 2 тикера."
        )

        st.stop()

    with st.spinner(
        f"Загрузка {len(tickers)} тикеров..."
    ):

        prices = download_prices(
            tuple(tickers),
            interval,
            history_period,
        )

    if prices.empty:

        st.error(
            "Yahoo Finance не вернул данные."
        )

        st.info(
            "Попробуйте Daily timeframe "
            "или уменьшите universe."
        )

        st.stop()

    # Keep only tickers with enough data.
    min_required = max(
        80,
        z_window + 20,
    )

    valid_columns = []

    for column in prices.columns:

        observations = (
            prices[column]
            .dropna()
            .shape[0]
        )

        if observations >= min_required:
            valid_columns.append(
                column
            )

    prices = prices[
        valid_columns
    ]

    metadata = metadata[
        metadata[
            "Ticker"
        ].isin(
            prices.columns
        )
    ].copy()

    if len(prices.columns) < 2:

        st.error(
            "После очистки осталось "
            "меньше двух ликвидных тикеров."
        )

        st.stop()

    st.session_state[
        "prices"
    ] = prices

    st.session_state[
        "metadata"
    ] = metadata

    with st.spinner(
        "Расчёт корреляций, OLS, ADF, Half-Life и Backtest..."
    ):

        results_df, pair_details = run_screener(
            prices=prices,
            metadata=metadata,
            correlation_threshold=correlation_threshold,
            pvalue_threshold=pvalue_threshold,
            winrate_threshold=winrate_threshold,
            max_pairs=max_candidate_pairs,
            max_backtests=max_backtests,
            z_window=z_window,
            entry_z=entry_z,
            exit_z=exit_z,
            initial_capital=initial_capital,
            correlation_basis=correlation_basis,
            same_sector_only=same_sector_only,
        )

    st.session_state[
        "results"
    ] = results_df

    st.session_state[
        "pair_details"
    ] = pair_details

    st.session_state[
        "run_timestamp"
    ] = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    st.rerun()


# ============================================================
# RESULTS
# ============================================================

results_df = st.session_state.get(
    "results",
    pd.DataFrame(),
)

pair_details = st.session_state.get(
    "pair_details",
    {},
)


if results_df.empty:

    st.markdown(
        """
        ## 👈 Настройте параметры слева

        Затем нажмите:

        **🚀 ЗАПУСТИТЬ СКРИМЕР**

        Приложение загрузит цены, рассчитает корреляции,
        OLS hedge ratio, ADF cointegration test,
        half-life, Z-score и исторический backtest.
        """
    )

    with st.expander(
        "ℹ️ Как работает математическое ядро"
    ):

        st.markdown(
            """
            **1. Pearson Correlation**

            Сначала отбираются пары с высокой корреляцией.

            **2. OLS**

            Для пары строится:

            `Y = Alpha + Beta × X + Spread`

            **3. ADF**

            ADF применяется к остаткам OLS.

            Низкий p-value означает, что нулевая гипотеза
            о unit root отвергается.

            **4. Half-Life**

            Оценивается скорость возврата spread к среднему.

            **5. Z-Score**

            `Z = (Spread - Rolling Mean) / Rolling Std`

            **6. Trading**

            При Z >= Entry:

            `SHORT SPREAD`

            При Z <= -Entry:

            `LONG SPREAD`

            Выход происходит около нулевой линии
            или внутри заданного Exit Z.
            """
        )

    st.stop()


# ============================================================
# TOP KPIs
# ============================================================

st.success(
    f"Скринер завершён. "
    f"Найдено {len(results_df)} подходящих пар."
)

if "run_timestamp" in st.session_state:

    st.caption(
        "Последний запуск: "
        + st.session_state[
            "run_timestamp"
        ]
    )


k1, k2, k3, k4, k5 = st.columns(5)

k1.metric(
    "Pairs",
    len(results_df),
)

k2.metric(
    "Median Corr",
    format_number(
        results_df["Corr"].median(),
        3,
    ),
)

k3.metric(
    "Median ADF p",
    format_number(
        results_df["p-value"].median(),
        4,
    ),
)

k4.metric(
    "Median Win Rate",
    format_pct(
        results_df["WinRate %"].median(),
        1,
    ),
)

k5.metric(
    "Total PnL Median",
    "$"
    + format_number(
        results_df["Total PnL"].median(),
        2,
    ),
)


# ============================================================
# TABS
# ============================================================

tab_results, tab_inspector = st.tabs(
    [
        "📋 Screener Results",
        "🔎 Pair Inspector",
    ]
)


# ============================================================
# TAB 1
# ============================================================

with tab_results:

    st.subheader(
        "Screener Results"
    )

    display_columns = [
        "Pair",
        "Sector",
        "Industry",
        "Corr",
        "p-value",
        "Beta",
        "Alpha",
        "Half-Life",
        "Current Z",
        "WinRate %",
        "Total PnL",
        "Trades",
        "Max Drawdown",
        "Observations",
    ]

    display_df = results_df[
        display_columns
    ].copy()

    st.dataframe(
        display_df,
        use_container_width=True,
        height=600,
        hide_index=True,
        column_config={
            "Corr": st.column_config.NumberColumn(
                "Corr",
                format="%.3f",
            ),
            "p-value": st.column_config.NumberColumn(
                "p-value",
                format="%.4f",
            ),
            "Beta": st.column_config.NumberColumn(
                "Beta",
                format="%.4f",
            ),
            "Alpha": st.column_config.NumberColumn(
                "Alpha",
                format="%.4f",
            ),
            "Half-Life": st.column_config.NumberColumn(
                "Half-Life",
                format="%.2f",
            ),
            "Current Z": st.column_config.NumberColumn(
                "Current Z",
                format="%.2f",
            ),
            "WinRate %": st.column_config.NumberColumn(
                "WinRate %",
                format="%.1f%%",
            ),
            "Total PnL": st.column_config.NumberColumn(
                "Total PnL",
                format="$%.2f",
            ),
            "Max Drawdown": st.column_config.NumberColumn(
                "Max Drawdown",
                format="$%.2f",
            ),
        },
    )

    st.markdown(
        "### 📥 Export"
    )

    csv_data = dataframe_to_csv(
        results_df
    )

    excel_data = dataframe_to_excel(
        results_df
    )

    download_col1, download_col2 = st.columns(2)

    with download_col1:

        st.download_button(
            label="⬇️ Скачать CSV",
            data=csv_data,
            file_name=(
                "pairtrade_screener_results.csv"
            ),
            mime="text/csv",
            use_container_width=True,
        )

    with download_col2:

        st.download_button(
            label="⬇️ Скачать Excel",
            data=excel_data,
            file_name=(
                "pairtrade_screener_results.xlsx"
            ),
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
            use_container_width=True,
        )

    st.markdown(
        "### 📊 Quick Distribution"
    )

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:

        fig = go.Figure()

        fig.add_trace(
            go.Histogram(
                x=results_df[
                    "p-value"
                ],
                nbinsx=30,
                name="ADF p-value",
                marker_color="#38bdf8",
            )
        )

        fig.update_layout(
            title="ADF p-value Distribution",
            template="plotly_dark",
            height=350,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with chart_col2:

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=results_df[
                    "Corr"
                ],
                y=results_df[
                    "WinRate %"
                ],
                mode="markers",
                text=results_df[
                    "Pair"
                ],
                marker=dict(
                    size=9,
                    color=results_df[
                        "p-value"
                    ],
                    colorscale="Turbo",
                    showscale=True,
                    colorbar=dict(
                        title="p-value"
                    ),
                ),
            )
        )

        fig.update_layout(
            title="Correlation vs Win Rate",
            template="plotly_dark",
            height=350,
            xaxis_title="Correlation",
            yaxis_title="Win Rate %",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


# ============================================================
# TAB 2 — PAIR INSPECTOR
# ============================================================

with tab_inspector:

    st.subheader(
        "Pair Inspector"
    )

    pair_options = results_df[
        "Pair"
    ].tolist()

    selected_pair = st.selectbox(
        "Выберите пару",
        pair_options,
        index=0,
    )

    if (
        selected_pair
        not in pair_details
    ):

        st.error(
            "Подробные данные пары отсутствуют."
        )

    else:

        detail = pair_details[
            selected_pair
        ]

        ticker_y = detail[
            "y"
        ]

        ticker_x = detail[
            "x"
        ]

        beta = detail[
            "beta"
        ]

        alpha = detail[
            "alpha"
        ]

        corr = detail[
            "corr"
        ]

        pvalue = detail[
            "pvalue"
        ]

        half_life = detail[
            "half_life"
        ]

        zscore = detail[
            "zscore"
        ]

        equity = detail[
            "equity_curve"
        ]

        trades = detail[
            "trades"
        ]

        backtest = detail[
            "backtest"
        ]

        st.markdown(
            f"""
            <div class="pair-title">
                {selected_pair}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption(
            f"{ticker_y} = Alpha + Beta × {ticker_x} + Spread"
        )

        m1, m2, m3, m4, m5, m6 = st.columns(6)

        m1.metric(
            "Correlation",
            format_number(
                corr,
                3,
            ),
        )

        m2.metric(
            "ADF p-value",
            format_number(
                pvalue,
                5,
            ),
        )

        m3.metric(
            "Beta",
            format_number(
                beta,
                4,
            ),
        )

        m4.metric(
            "Half-Life",
            format_number(
                half_life,
                2,
            ),
        )

        m5.metric(
            "Win Rate",
            format_pct(
                backtest[
                    "win_rate"
                ],
                1,
            ),
        )

        m6.metric(
            "PnL",
            "$"
            + format_number(
                backtest[
                    "total_pnl"
                ],
                2,
            ),
        )

        # ----------------------------------------------------
        # Z-SCORE
        # ----------------------------------------------------

        st.plotly_chart(
            make_zscore_chart(
                zscore=zscore,
                entry_z=entry_z,
                exit_z=exit_z,
                pair_name=selected_pair,
            ),
            use_container_width=True,
        )

        # ----------------------------------------------------
        # NORMALIZED PRICES
        # ----------------------------------------------------

        st.plotly_chart(
            make_normalized_price_chart(
                y=detail[
                    "prices_y"
                ],
                x=detail[
                    "prices_x"
                ],
                ticker_y=ticker_y,
                ticker_x=ticker_x,
            ),
            use_container_width=True,
        )

        # ----------------------------------------------------
        # EQUITY CURVE
        # ----------------------------------------------------

        st.plotly_chart(
            make_equity_chart(
                equity=equity,
                pair_name=selected_pair,
                initial_capital=initial_capital,
            ),
            use_container_width=True,
        )

        # ----------------------------------------------------
        # TRADE TABLE
        # ----------------------------------------------------

        st.markdown(
            "### 💰 Historical Trades"
        )

        trades_df = trades_to_dataframe(
            trades
        )

        if trades_df.empty:

            st.info(
                "Backtest не создал ни одной сделки "
                "по заданным Entry / Exit параметрам."
            )

        else:

            st.dataframe(
                trades_df,
                use_container_width=True,
                hide_index=True,
            )

            trade_csv = dataframe_to_csv(
                trades_df
            )

            st.download_button(
                "⬇️ Скачать сделки CSV",
                trade_csv,
                file_name=(
                    selected_pair
                    .replace(
                        "/",
                        "_",
                    )
                    + "_trades.csv"
                ),
                mime="text/csv",
            )

        # ----------------------------------------------------
        # QUANT DETAILS
        # ----------------------------------------------------

        with st.expander(
            "🧮 Quantitative Details"
        ):

            detail_col1, detail_col2 = st.columns(2)

            with detail_col1:

                st.write(
                    f"**Dependent asset Y:** `{ticker_y}`"
                )

                st.write(
                    f"**Independent asset X:** `{ticker_x}`"
                )

                st.write(
                    f"**Alpha:** `{alpha:.6f}`"
                )

                st.write(
                    f"**Beta:** `{beta:.6f}`"
                )

            with detail_col2:

                st.write(
                    f"**Correlation:** `{corr:.6f}`"
                )

                st.write(
                    f"**ADF p-value:** `{pvalue:.8f}`"
                )

                st.write(
                    f"**Half-Life:** `{format_number(half_life, 2)} bars`"
                )

                st.write(
                    f"**Trades:** `{backtest['trades']}`"
                )

        # ----------------------------------------------------
        # SIGNAL
        # ----------------------------------------------------

        current_z = safe_float(
            zscore.dropna().iloc[-1]
            if not zscore.dropna().empty
            else np.nan
        )

        if np.isfinite(current_z):

            if current_z >= entry_z:

                st.error(
                    f"🔴 Current signal: SHORT SPREAD "
                    f"(Z = {current_z:.2f})"
                )

            elif current_z <= -entry_z:

                st.success(
                    f"🟢 Current signal: LONG SPREAD "
                    f"(Z = {current_z:.2f})"
                )

            elif abs(current_z) <= exit_z:

                st.info(
                    f"⚪ Current signal: EXIT / MEAN "
                    f"(Z = {current_z:.2f})"
                )

            else:

                st.warning(
                    f"🟡 Current signal: WAIT "
                    f"(Z = {current_z:.2f})"
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    """
    PairTrade Finder — research / educational quantitative tool.
    Data supplied by public/free market-data sources.
    Backtest results are hypothetical and do not include commissions,
    slippage, borrow fees, dividends, taxes or execution latency.
    """
)
