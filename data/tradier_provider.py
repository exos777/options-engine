"""
Tradier data provider — real-time quotes, option chains (with full Greeks),
expirations, and historical OHLCV via the Tradier REST API.

Mirrors the public API of data/provider.py so callers can swap seamlessly.
Auth: Bearer token from Streamlit secrets (TRADIER_TOKEN) or env var.
"""

from __future__ import annotations

import logging
import os
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import requests

from greeks.black_scholes import backfill_greeks
from strategies.models import OptionContract, Quote
from data.common import (
    safe_float as _safe_float,
    safe_int as _safe_int,
    nearest_weekly_expiration as _nearest_weekly_expiration_impl,
    days_to_expiration,
    earnings_warning,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Auth + transport
# ---------------------------------------------------------------------------

_BASE_URL = "https://api.tradier.com/v1"
_TIMEOUT_SECONDS = 15


def _get_token() -> str:
    """Resolve the Tradier API token from Streamlit secrets or env vars."""
    token: Optional[str] = None
    try:
        import streamlit as st  # type: ignore
        if hasattr(st, "secrets"):
            token = st.secrets.get("TRADIER_TOKEN")
    except Exception:
        token = None
    if not token:
        token = os.environ.get("TRADIER_TOKEN", "")
    return (token or "").strip()


def tradier_available() -> bool:
    """Return True if a Tradier API token is configured."""
    return bool(_get_token())


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {_get_token()}",
        "Accept": "application/json",
    }


def _get(path: str, params: Optional[dict] = None) -> dict:
    """GET against the Tradier REST API. Raises on auth/network errors."""
    if not _get_token():
        raise RuntimeError("TRADIER_TOKEN not configured")
    resp = requests.get(
        f"{_BASE_URL}{path}",
        headers=_headers(),
        params=params or {},
        timeout=_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    return resp.json() or {}


def _as_list(node) -> list:
    """Tradier returns a single-item dict when only one row exists; coerce to list."""
    if node is None:
        return []
    if isinstance(node, list):
        return node
    return [node]


# ---------------------------------------------------------------------------
# Public API — mirrors data/provider.py
# ---------------------------------------------------------------------------

def get_quote(symbol: str) -> Quote:
    """Fetch real-time quote for *symbol*. Raises ValueError on bad ticker."""
    sym = symbol.upper().strip()
    data = _get("/markets/quotes", {"symbols": sym, "greeks": "false"})
    quotes_node = (data.get("quotes") or {})
    raw = quotes_node.get("quote")
    if isinstance(raw, list):
        raw = raw[0] if raw else None
    if not raw:
        raise ValueError(f"No quote data found for '{sym}'. Check the ticker.")

    price = _safe_float(raw.get("last")) or _safe_float(raw.get("close"))
    if price == 0.0:
        raise ValueError(f"No price data found for '{sym}'.")
    prev_close = _safe_float(raw.get("prevclose")) or price
    change = price - prev_close
    change_pct = change / prev_close if prev_close != 0 else 0.0
    volume = _safe_int(raw.get("volume"))
    avg_vol = _safe_int(raw.get("average_volume"))

    return Quote(
        ticker=sym,
        price=price,
        prev_close=prev_close,
        change=change,
        change_pct=change_pct,
        volume=volume,
        avg_volume=avg_vol,
        market_cap=None,
        earnings_date=None,
    )


def get_expirations(symbol: str) -> tuple[str, ...]:
    """Return all available option expiration dates as ISO strings."""
    sym = symbol.upper().strip()
    data = _get(
        "/markets/options/expirations",
        {"symbol": sym, "includeAllRoots": "true", "strikes": "false"},
    )
    exps_node = data.get("expirations") or {}
    raw = exps_node.get("date") or []
    if isinstance(raw, str):
        raw = [raw]
    if not raw:
        raise ValueError(f"No option expirations found for '{sym}'.")
    return tuple(sorted(raw))


def get_nearest_weekly_expiration(symbol: str) -> str:
    """Return the nearest weekly (Friday) expiration for *symbol*."""
    return _nearest_weekly_expiration_impl(get_expirations(symbol))


def get_option_chain(
    symbol: str,
    expiration: str,
) -> tuple[list[OptionContract], list[OptionContract]]:
    """
    Fetch the full option chain for *symbol* at *expiration*.

    Uses greeks=true to get delta/gamma/theta/vega + mid_iv server-side.
    Falls back to Black-Scholes backfill for any missing Greeks.

    Returns (calls, puts). Raises ValueError on bad ticker / missing chain.
    """
    sym = symbol.upper().strip()
    data = _get(
        "/markets/options/chains",
        {"symbol": sym, "expiration": expiration, "greeks": "true"},
    )
    options_node = data.get("options") or {}
    raw_options = _as_list(options_node.get("option"))

    if not raw_options:
        raise ValueError(
            f"Could not fetch option chain for '{sym}' at {expiration}."
        )

    # Spot price for Black-Scholes backfill (only if needed).
    try:
        spot_price = get_quote(sym).price
    except Exception:
        spot_price = 0.0
    dte = days_to_expiration(expiration)

    calls: list[OptionContract] = []
    puts: list[OptionContract] = []

    for opt in raw_options:
        option_type = (opt.get("option_type") or "").lower()
        if option_type not in ("call", "put"):
            continue

        greeks = opt.get("greeks") or {}
        iv = _safe_float(greeks.get("mid_iv")) or None
        delta = greeks.get("delta")
        gamma = greeks.get("gamma")
        theta = greeks.get("theta")
        vega = greeks.get("vega")
        # Tradier returns numeric Greeks already; coerce defensively.
        delta = _safe_float(delta) if delta is not None else None
        gamma = _safe_float(gamma) if gamma is not None else None
        theta = _safe_float(theta) if theta is not None else None
        vega = _safe_float(vega) if vega is not None else None

        strike = _safe_float(opt.get("strike"))

        # Backfill missing Greeks via Black-Scholes when IV is available.
        if iv and spot_price > 0 and any(g is None for g in (delta, gamma, theta, vega)):
            bs = backfill_greeks(strike, spot_price, dte, iv, option_type)
            if bs is not None:
                delta = delta if delta is not None else bs.delta
                gamma = gamma if gamma is not None else bs.gamma
                theta = theta if theta is not None else bs.theta
                vega = vega if vega is not None else bs.vega

        contract = OptionContract(
            strike=strike,
            expiration=expiration,
            option_type=option_type,
            bid=_safe_float(opt.get("bid")),
            ask=_safe_float(opt.get("ask")),
            last=_safe_float(opt.get("last")),
            volume=_safe_int(opt.get("volume")),
            open_interest=_safe_int(opt.get("open_interest")),
            implied_volatility=iv,
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
        )
        if option_type == "call":
            calls.append(contract)
        else:
            puts.append(contract)

    return calls, puts


def get_historical(symbol: str, months: int = 6) -> pd.DataFrame:
    """
    Fetch daily OHLCV from Tradier for *symbol* going back *months* months.

    Returns a DataFrame indexed by date with columns:
    Open, High, Low, Close, Volume.
    """
    sym = symbol.upper().strip()
    end = date.today()
    start = end - timedelta(days=max(1, months) * 31)

    data = _get(
        "/markets/history",
        {
            "symbol": sym,
            "interval": "daily",
            "start": start.isoformat(),
            "end": end.isoformat(),
        },
    )
    history_node = data.get("history") or {}
    days = _as_list(history_node.get("day"))
    if not days or len(days) < 20:
        raise ValueError(f"Insufficient historical data for '{sym}'.")

    df = pd.DataFrame(days)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        },
        inplace=True,
    )
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.sort_index(inplace=True)
    return df


# days_to_expiration and earnings_warning are imported from data.common
