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

# Patch curl_cffi sessions to disable cert verification before yfinance imports it
try:
    import curl_cffi.requests as _cffi
    _orig_init = _cffi.Session.__init__

    def _patched_init(self, *args, **kwargs):
        kwargs.setdefault("verify", False)
        _orig_init(self, *args, **kwargs)

    _cffi.Session.__init__ = _patched_init
except Exception:
    pass

warnings.filterwarnings("ignore", message="Unverified HTTPS request")

import pandas as pd
import yfinance as yf

from strategies.models import OptionContract, Quote

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ticker(symbol: str) -> yf.Ticker:
    return yf.Ticker(symbol.upper().strip())


def _nearest_weekly_expiration(expirations: tuple[str, ...]) -> str:
    """
    Return the nearest expiration that falls on a Friday (weekly option)
    or, if none found within 14 days, the nearest expiration overall.
    """
    today = date.today()
    candidates = []
    for exp_str in expirations:
        exp = date.fromisoformat(exp_str)
        if exp >= today:
            candidates.append(exp)
    if not candidates:
        raise ValueError("No future expirations found.")
    # Prefer Friday expirations (weekday() == 4) within the next ~14 days
    friday_candidates = [e for e in candidates if e.weekday() == 4]
    if friday_candidates:
        return friday_candidates[0].isoformat()
    # Fallback: nearest expiration regardless of day
    return sorted(candidates)[0].isoformat()


def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val) if val is not None and not pd.isna(val) else default
    except (TypeError, ValueError):
        return default


def _safe_int(val, default: int = 0) -> int:
    try:
        return int(val) if val is not None and not pd.isna(val) else default
    except (TypeError, ValueError):
        return default


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

    def _parse_row(row: pd.Series, option_type: str) -> OptionContract:
        return OptionContract(
            strike=_safe_float(row.get("strike")),
            expiration=expiration,
            option_type=option_type,
            bid=_safe_float(row.get("bid")),
            ask=_safe_float(row.get("ask")),
            last=_safe_float(row.get("lastPrice")),
            volume=_safe_int(row.get("volume")),
            open_interest=_safe_int(row.get("openInterest")),
            implied_volatility=_safe_float(row.get("impliedVolatility")) or None,
            delta=_safe_float(row.get("delta")) or None,
            gamma=_safe_float(row.get("gamma")) or None,
            theta=_safe_float(row.get("theta")) or None,
            vega=_safe_float(row.get("vega")) or None,
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


def days_to_expiration(expiration: str) -> int:
    """Return calendar days from today to *expiration*."""
    exp = date.fromisoformat(expiration)
    return max(0, (exp - date.today()).days)


def earnings_warning(quote: Quote, expiration: str) -> Optional[str]:
    """
    Return a warning string if an earnings date falls before or on the expiration,
    otherwise return None.
    """
    if not quote.earnings_date:
        return None
    try:
        ed = date.fromisoformat(quote.earnings_date)
        exp = date.fromisoformat(expiration)
        if ed <= exp:
            return (
                f"Earnings expected on {quote.earnings_date} — before expiration. "
                "IV crush risk is elevated."
            )
    except Exception:
        pass
    return None
