"""
Tradier data provider — real-time quotes, option chains (with full Greeks),
expirations, and historical OHLCV via the Tradier REST API.

Mirrors the public API of data/provider.py so callers can swap seamlessly.
Auth: Bearer token from Streamlit secrets (TRADIER_TOKEN) or env var.
"""

from __future__ import annotations

import logging
import os
import time
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
_MAX_RETRIES = 3
_RETRY_BACKOFF_SECONDS = 0.75


def _clean_token(raw: str) -> str:
    """
    Normalise a token value.

    Strips whitespace and any surrounding quotes. Quotes are required by
    TOML in secrets.toml but are literal characters in an env var — pasting
    the quoted form into Railway yields `Bearer "abc"` and a 401, so we
    defend against it here.
    """
    token = (raw or "").strip()
    for quote in ('"', "'"):
        if len(token) >= 2 and token.startswith(quote) and token.endswith(quote):
            token = token[1:-1].strip()
            break
    return token


def _get_token() -> str:
    """Resolve the Tradier API token.

    Priority:
      1. TRADIER_TOKEN environment variable (Railway, Docker, .env)
      2. Streamlit secrets (local secrets.toml, Streamlit Cloud)
    """
    # Env var is checked first — always works on Railway.
    token = _clean_token(os.environ.get("TRADIER_TOKEN", ""))
    if token:
        return token
    # Fallback: Streamlit secrets (local dev / Streamlit Cloud).
    try:
        import streamlit as st  # type: ignore
        if hasattr(st, "secrets"):
            token = _clean_token(st.secrets.get("TRADIER_TOKEN") or "")
    except Exception:
        pass
    return token


def tradier_available() -> bool:
    """Return True if a Tradier API token is configured (not that it works)."""
    return bool(_get_token())


def tradier_token_fingerprint() -> str:
    """Short non-secret identifier for the active token (cache-key use)."""
    token = _get_token()
    return f"{len(token)}:{token[-4:]}" if token else "none"


def tradier_status() -> tuple[bool, str]:
    """
    Verify the configured token against the live API.

    Returns (ok, message). Unlike tradier_available(), this makes a real
    request, so a stale or malformed token is reported as broken instead
    of showing a false "connected" state.
    """
    token = _get_token()
    if not token:
        return False, "No TRADIER_TOKEN configured."
    try:
        resp = _request("/user/profile")
    except Exception as exc:  # noqa: BLE001 — network/DNS/proxy failure
        return False, f"Could not reach Tradier ({type(exc).__name__})."

    if resp.status_code == 401:
        return False, (
            f"Token rejected (401). The key ending '...{token[-4:]}' "
            f"({len(token)} chars) is invalid, expired, or was pasted "
            "with surrounding quotes."
        )
    # /user/profile is 404/403 on market-data-only keys — the token itself
    # is still valid, so confirm with a lightweight market-data call.
    if resp.status_code != 200:
        try:
            probe = _request("/markets/quotes", {"symbols": "AAPL"})
        except Exception as exc:  # noqa: BLE001
            return False, f"Could not reach Tradier ({type(exc).__name__})."
        if probe.status_code == 401:
            return False, (
                f"Token rejected (401). The key ending '...{token[-4:]}' "
                f"({len(token)} chars) is invalid, expired, or was pasted "
                "with surrounding quotes."
            )
        if probe.status_code != 200:
            return False, f"Tradier returned HTTP {probe.status_code}."
    return True, f"Tradier connected (key ...{token[-4:]})."


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {_get_token()}",
        "Accept": "application/json",
    }


def _request(path: str, params: Optional[dict] = None) -> requests.Response:
    """
    GET against the Tradier REST API with retry on transport failures.

    Connection resets are retried (corporate proxies drop TLS handshakes
    intermittently); HTTP status codes are not — a 401 is a real answer
    and retrying it just wastes time.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(_MAX_RETRIES):
        try:
            return requests.get(
                f"{_BASE_URL}{path}",
                headers=_headers(),
                params=params or {},
                timeout=_TIMEOUT_SECONDS,
            )
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_RETRY_BACKOFF_SECONDS * (attempt + 1))
    raise last_exc  # type: ignore[misc]


def _get(path: str, params: Optional[dict] = None) -> dict:
    """GET against the Tradier REST API. Raises on auth/network errors."""
    if not _get_token():
        raise RuntimeError("TRADIER_TOKEN not configured")
    resp = _request(path, params)
    resp.raise_for_status()
    return resp.json() or {}


def _clean_symbol(symbol: str) -> str:
    """
    Normalise and validate a ticker.

    Raises ValueError on a blank symbol so callers get an actionable
    message instead of an opaque HTTP 400 from `?symbol=`.
    """
    sym = (symbol or "").strip().upper()
    if not sym:
        raise ValueError("No ticker symbol provided.")
    return sym


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
    sym = _clean_symbol(symbol)
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
    sym = _clean_symbol(symbol)
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
    sym = _clean_symbol(symbol)
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


def get_earnings_date(symbol: str) -> Optional[str]:
    """
    Best-effort next-earnings date (ISO string) via the Tradier beta
    fundamentals calendar. Returns None on any failure or if the account
    tier doesn't include fundamentals — callers must treat None as
    "unknown", never as "no earnings".
    """
    sym = _clean_symbol(symbol)
    try:
        data = _get("/markets/fundamentals/calendars", {"symbols": sym})
    except Exception:
        return None

    today = date.today()
    best: Optional[date] = None

    def _walk(node) -> None:
        nonlocal best
        if isinstance(node, dict):
            ev_type = str(node.get("event_type", node.get("type", ""))).lower()
            raw_date = node.get("begin_date_time") or node.get("date")
            if "earnings" in ev_type and raw_date:
                try:
                    d = date.fromisoformat(str(raw_date)[:10])
                    if d >= today and (best is None or d < best):
                        best = d
                except ValueError:
                    pass
            for v in node.values():
                _walk(v)
        elif isinstance(node, list):
            for v in node:
                _walk(v)

    _walk(data)
    return best.isoformat() if best else None


def get_historical(symbol: str, months: int = 6) -> pd.DataFrame:
    """
    Fetch daily OHLCV from Tradier for *symbol* going back *months* months.

    Returns a DataFrame indexed by date with columns:
    Open, High, Low, Close, Volume.
    """
    sym = _clean_symbol(symbol)
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
