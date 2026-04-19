"""
Data provider module — wraps yfinance to deliver quotes, option chains,
expirations, and historical OHLCV data in the app's canonical types.

All public functions raise ValueError on bad tickers or data gaps so
callers can display friendly error messages.
"""

from __future__ import annotations

import logging
import os
import ssl
import warnings
from datetime import date, datetime, timedelta
from typing import Optional

# ---------------------------------------------------------------------------
# SSL bypass for corporate proxies with self-signed certificates.
# yfinance 1.x uses curl_cffi which validates SSL strictly by default.
# ---------------------------------------------------------------------------
os.environ.setdefault("CURL_CA_BUNDLE", "")
os.environ.setdefault("REQUESTS_CA_BUNDLE", "")

try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

# Patch curl_cffi sessions to disable cert verification before yfinance imports it.
# Store the original __init__ on the class itself so importlib.reload() cannot
# create a recursive wrapper chain (_patched -> _patched -> ... -> RecursionError).
try:
    import curl_cffi.requests as _cffi
    if not hasattr(_cffi.Session, "_orig_init_unpatched"):
        _cffi.Session._orig_init_unpatched = _cffi.Session.__init__

        def _patched_init(self, *args, **kwargs):
            kwargs.setdefault("verify", False)
            _cffi.Session._orig_init_unpatched(self, *args, **kwargs)

        _cffi.Session.__init__ = _patched_init
except Exception:
    pass

warnings.filterwarnings("ignore", message="Unverified HTTPS request")

import pandas as pd
import yfinance as yf

from greeks.black_scholes import backfill_greeks
from strategies.models import OptionContract, Quote
from data.common import (
    safe_float as _safe_float,
    safe_int as _safe_int,
    nearest_weekly_expiration as _nearest_weekly_expiration,
    days_to_expiration,
    earnings_warning,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ticker(symbol: str) -> yf.Ticker:
    return yf.Ticker(symbol.upper().strip())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_quote(symbol: str) -> Quote:
    """
    Fetch real-time (15-min delayed) quote for *symbol*.

    Raises ValueError if the ticker is invalid or price data is unavailable.
    """
    t = _ticker(symbol)
    info = t.fast_info

    price = _safe_float(getattr(info, "last_price", None))
    if price == 0.0:
        # Fallback: grab last close from 2-day history
        hist = t.history(period="2d")
        if hist.empty:
            raise ValueError(f"No price data found for '{symbol}'. Check the ticker.")
        price = float(hist["Close"].iloc[-1])

    prev_close = _safe_float(getattr(info, "previous_close", None))
    if prev_close == 0.0:
        prev_close = price

    change = price - prev_close
    change_pct = change / prev_close if prev_close != 0 else 0.0
    volume = _safe_int(getattr(info, "last_volume", None))
    avg_vol = _safe_int(getattr(info, "three_month_average_volume", None))
    market_cap = _safe_float(getattr(info, "market_cap", None)) or None

    # Earnings date (best-effort)
    earnings_date: Optional[str] = None
    try:
        cal = t.calendar
        if cal is not None and not cal.empty:
            ed = cal.get("Earnings Date")
            if ed is not None and len(ed) > 0:
                earnings_date = pd.Timestamp(ed.iloc[0]).date().isoformat()
    except Exception:
        pass

    return Quote(
        ticker=symbol.upper(),
        price=price,
        prev_close=prev_close,
        change=change,
        change_pct=change_pct,
        volume=volume,
        avg_volume=avg_vol,
        market_cap=market_cap,
        earnings_date=earnings_date,
    )


def get_expirations(symbol: str) -> tuple[str, ...]:
    """Return all available option expiration dates as ISO strings."""
    t = _ticker(symbol)
    exps = t.options
    if not exps:
        raise ValueError(f"No option expirations found for '{symbol}'.")
    return exps  # yfinance already returns sorted ISO date strings


def get_nearest_weekly_expiration(symbol: str) -> str:
    """Return the nearest weekly (Friday) expiration for *symbol*."""
    return _nearest_weekly_expiration(get_expirations(symbol))


def get_option_chain(
    symbol: str,
    expiration: str,
) -> tuple[list[OptionContract], list[OptionContract]]:
    """
    Fetch the full option chain for *symbol* at *expiration*.

    Returns (calls, puts) as lists of OptionContract.
    Raises ValueError on bad ticker or missing chain.
    """
    t = _ticker(symbol)
    try:
        chain = t.option_chain(expiration)
    except Exception as exc:
        raise ValueError(f"Could not fetch option chain for '{symbol}' at {expiration}: {exc}") from exc

    # Spot price and DTE for Black-Scholes backfill
    spot_price = _safe_float(getattr(t.fast_info, "last_price", None))
    if spot_price == 0.0:
        hist = t.history(period="2d")
        spot_price = float(hist["Close"].iloc[-1]) if not hist.empty else 0.0
    dte = days_to_expiration(expiration)

    def _parse_row(row: pd.Series, option_type: str) -> OptionContract:
        iv = _safe_float(row.get("impliedVolatility")) or None
        delta = _safe_float(row.get("delta")) or None
        gamma = _safe_float(row.get("gamma")) or None
        theta = _safe_float(row.get("theta")) or None
        vega = _safe_float(row.get("vega")) or None

        # Backfill missing Greeks via Black-Scholes when IV is available
        if iv and spot_price > 0 and any(g is None for g in (delta, gamma, theta, vega)):
            strike = _safe_float(row.get("strike"))
            bs = backfill_greeks(strike, spot_price, dte, iv, option_type)
            if bs is not None:
                delta = delta if delta is not None else bs.delta
                gamma = gamma if gamma is not None else bs.gamma
                theta = theta if theta is not None else bs.theta
                vega = vega if vega is not None else bs.vega

        return OptionContract(
            strike=_safe_float(row.get("strike")),
            expiration=expiration,
            option_type=option_type,
            bid=_safe_float(row.get("bid")),
            ask=_safe_float(row.get("ask")),
            last=_safe_float(row.get("lastPrice")),
            volume=_safe_int(row.get("volume")),
            open_interest=_safe_int(row.get("openInterest")),
            implied_volatility=iv,
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
        )

    calls = [_parse_row(row, "call") for _, row in chain.calls.iterrows()]
    puts = [_parse_row(row, "put") for _, row in chain.puts.iterrows()]
    return calls, puts


def get_historical(symbol: str, months: int = 6) -> pd.DataFrame:
    """
    Fetch historical daily OHLCV data for *symbol* going back *months* months.

    Returns a DataFrame with columns: Open, High, Low, Close, Volume.
    Index is a DatetimeIndex sorted ascending.
    Raises ValueError if insufficient data is returned.
    """
    period = f"{max(1, months)}mo"
    t = _ticker(symbol)
    df = t.history(period=period)
    if df.empty or len(df) < 20:
        raise ValueError(f"Insufficient historical data for '{symbol}'.")
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df



# days_to_expiration and earnings_warning are imported from data.common
