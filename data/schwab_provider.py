"""
Schwab data provider — real-time quotes and option chains via schwab-py.

Mirrors the public API of data/provider.py so callers can swap seamlessly.
Authentication:
  - Local: uses schwab_token.json (created via browser OAuth on first run)
  - Streamlit Cloud: reads token JSON from st.secrets["SCHWAB_TOKEN_JSON"]
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy Schwab client singleton
# ---------------------------------------------------------------------------

_client = None


def _get_keys() -> tuple[str, str]:
    """Resolve Schwab API key + secret from env, .env, or st.secrets."""
    app_key = os.getenv("SCHWAB_APP_KEY", "")
    app_secret = os.getenv("SCHWAB_APP_SECRET", "")

    # Try Streamlit secrets if env vars are empty/placeholder
    if not app_key or app_key in ("your_key_here", "your_app_key_here"):
        try:
            import streamlit as st
            app_key = st.secrets.get("SCHWAB_APP_KEY", "")
            app_secret = st.secrets.get("SCHWAB_APP_SECRET", "")
        except Exception:
            pass

    # Try dotenv as last resort
    if not app_key or app_key in ("your_key_here", "your_app_key_here"):
        try:
            from dotenv import load_dotenv
            load_dotenv(Path(__file__).resolve().parent.parent / ".env")
            app_key = os.getenv("SCHWAB_APP_KEY", "")
            app_secret = os.getenv("SCHWAB_APP_SECRET", "")
        except ImportError:
            pass

    return app_key, app_secret


def _token_path() -> Path:
    return Path(__file__).resolve().parent.parent / "schwab_token.json"


def get_client():
    """Return an authenticated Schwab client (cached as singleton)."""
    global _client
    if _client is not None:
        return _client

    import schwab

    app_key, app_secret = _get_keys()
    if not app_key or app_key in ("your_key_here", "your_app_key_here"):
        raise RuntimeError(
            "SCHWAB_APP_KEY not configured. "
            "Set it in .env, environment variables, or Streamlit secrets."
        )

    token_path = _token_path()

    # 1. Try loading from existing token file
    if token_path.exists():
        try:
            _client = schwab.auth.client_from_token_file(
                str(token_path), app_key, app_secret
            )
            logger.info("Schwab: authenticated from token file")
            return _client
        except Exception as e:
            logger.warning("Schwab: token file invalid (%s), will re-auth", e)

    # 2. Try loading token JSON from Streamlit secrets (Cloud deployment)
    _on_cloud = not token_path.exists()  # no local token file = likely Cloud
    try:
        import streamlit as st
        token_json = st.secrets.get("SCHWAB_TOKEN_JSON", "")
        if token_json:
            import tempfile
            tmp_token = Path(tempfile.gettempdir()) / "schwab_token.json"
            # st.secrets may return a non-str type; force to str
            tmp_token.write_text(str(token_json))
            logger.info("Schwab: wrote token to %s (%d bytes)", tmp_token, tmp_token.stat().st_size)
            _client = schwab.auth.client_from_token_file(
                str(tmp_token), app_key, app_secret
            )
            logger.info("Schwab: authenticated from Streamlit secrets token")
            return _client
        elif _on_cloud:
            import streamlit as st
            st.error("SCHWAB_TOKEN_JSON not found in Streamlit secrets. "
                     "Run schwab_auth.py locally and paste the token JSON into secrets.")
            st.stop()
    except Exception as e:
        logger.error("Schwab token from secrets failed: %s", e, exc_info=True)
        if _on_cloud:
            import streamlit as st
            st.error(f"Schwab auth failed: {e}")
            st.stop()
        logger.warning("Schwab: failed to auth from st.secrets token: %s", e)

    # 3. Fall back to browser-based login flow (local dev only)
    callback_url = "https://127.0.0.1"
    _client = schwab.auth.client_from_login_flow(
        app_key, app_secret, callback_url, str(token_path)
    )
    logger.info("Schwab: authenticated via browser login flow")
    return _client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(val, default: float = 0.0) -> float:
    try:
        if val is None:
            return default
        return float(val)
    except (TypeError, ValueError):
        return default


def _safe_int(val, default: int = 0) -> int:
    try:
        if val is None:
            return default
        return int(val)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Public API — mirrors data/provider.py
# ---------------------------------------------------------------------------

from greeks.black_scholes import backfill_greeks
from strategies.models import OptionContract, Quote


def get_quote(symbol: str) -> Quote:
    """Fetch real-time quote from Schwab."""
    c = get_client()
    resp = c.get_quote(symbol.upper().strip())
    resp.raise_for_status()
    data = resp.json()

    q = data[symbol.upper().strip()]["quote"]

    price = _safe_float(q.get("lastPrice"))
    prev_close = _safe_float(q.get("closePrice")) or price
    change = price - prev_close
    change_pct = change / prev_close if prev_close != 0 else 0.0
    volume = _safe_int(q.get("totalVolume"))
    avg_vol = _safe_int(q.get("averageVolume10Days"))  # Schwab offers 10-day avg

    # Schwab doesn't provide earnings date in quotes — leave None
    return Quote(
        ticker=symbol.upper().strip(),
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
    """Return available option expiration dates as ISO strings."""
    import schwab.client

    c = get_client()
    resp = c.get_option_chain(
        symbol.upper().strip(),
        contract_type=schwab.client.Client.Options.ContractType.CALL,
        strike_count=1,
    )
    resp.raise_for_status()
    chain = resp.json()

    # Extract unique expiration dates from callExpDateMap keys
    # Keys look like "2026-03-20:4" (date:DTE)
    exp_dates = set()
    for key in chain.get("callExpDateMap", {}):
        date_str = key.split(":")[0]
        exp_dates.add(date_str)
    for key in chain.get("putExpDateMap", {}):
        date_str = key.split(":")[0]
        exp_dates.add(date_str)

    if not exp_dates:
        raise ValueError(f"No option expirations found for '{symbol}'.")

    return tuple(sorted(exp_dates))


def _nearest_weekly_expiration(expirations: tuple[str, ...]) -> str:
    """Return the nearest Friday expiration, or nearest overall."""
    today = date.today()
    candidates = []
    for exp_str in expirations:
        exp = date.fromisoformat(exp_str)
        if exp >= today:
            candidates.append(exp)
    if not candidates:
        raise ValueError("No future expirations found.")
    friday_candidates = [e for e in candidates if e.weekday() == 4]
    if friday_candidates:
        return friday_candidates[0].isoformat()
    return sorted(candidates)[0].isoformat()


def get_nearest_weekly_expiration(symbol: str) -> str:
    """Return the nearest weekly (Friday) expiration for symbol."""
    return _nearest_weekly_expiration(get_expirations(symbol))


def get_option_chain(
    symbol: str,
    expiration: str,
) -> tuple[list[OptionContract], list[OptionContract]]:
    """
    Fetch the full option chain for symbol at expiration from Schwab.
    Returns (calls, puts) as lists of OptionContract.
    """
    import schwab.client

    c = get_client()

    exp_date = date.fromisoformat(expiration)
    resp = c.get_option_chain(
        symbol.upper().strip(),
        contract_type=schwab.client.Client.Options.ContractType.ALL,
        from_date=exp_date,
        to_date=exp_date,
        include_underlying_quote=True,
    )
    resp.raise_for_status()
    chain = resp.json()

    # Get spot price from underlying quote
    spot_price = _safe_float(
        chain.get("underlyingPrice")
        or chain.get("underlying", {}).get("last")
    )
    dte = days_to_expiration(expiration)

    def _parse_contracts(exp_date_map: dict, option_type: str) -> list[OptionContract]:
        contracts = []
        for date_key, strikes in exp_date_map.items():
            for strike_str, contract_list in strikes.items():
                for contract in contract_list:
                    iv = _safe_float(contract.get("volatility")) / 100 if contract.get("volatility") else None
                    delta = _safe_float(contract.get("delta")) or None
                    gamma = _safe_float(contract.get("gamma")) or None
                    theta = _safe_float(contract.get("theta")) or None
                    vega = _safe_float(contract.get("vega")) or None

                    strike = _safe_float(contract.get("strikePrice"))

                    # Backfill missing Greeks via Black-Scholes
                    if iv and spot_price > 0 and any(g is None for g in (delta, gamma, theta, vega)):
                        bs = backfill_greeks(strike, spot_price, dte, iv, option_type)
                        if bs is not None:
                            delta = delta if delta is not None else bs.delta
                            gamma = gamma if gamma is not None else bs.gamma
                            theta = theta if theta is not None else bs.theta
                            vega = vega if vega is not None else bs.vega

                    contracts.append(OptionContract(
                        strike=strike,
                        expiration=expiration,
                        option_type=option_type,
                        bid=_safe_float(contract.get("bid")),
                        ask=_safe_float(contract.get("ask")),
                        last=_safe_float(contract.get("last")),
                        volume=_safe_int(contract.get("totalVolume")),
                        open_interest=_safe_int(contract.get("openInterest")),
                        implied_volatility=iv,
                        delta=delta,
                        gamma=gamma,
                        theta=theta,
                        vega=vega,
                    ))
        return contracts

    calls = _parse_contracts(chain.get("callExpDateMap", {}), "call")
    puts = _parse_contracts(chain.get("putExpDateMap", {}), "put")
    return calls, puts


def get_historical(symbol: str, months: int = 6) -> pd.DataFrame:
    """
    Fetch historical daily OHLCV from Schwab.
    Falls back to yfinance if Schwab historical isn't available.
    """
    import schwab.client

    c = get_client()
    end = datetime.now()
    start = end - timedelta(days=months * 30)

    try:
        resp = c.get_price_history_every_day(
            symbol.upper().strip(),
            start_datetime=start,
            end_datetime=end,
        )
        resp.raise_for_status()
        data = resp.json()

        candles = data.get("candles", [])
        if not candles or len(candles) < 20:
            raise ValueError("Insufficient data from Schwab")

        df = pd.DataFrame(candles)
        df["datetime"] = pd.to_datetime(df["datetime"], unit="ms")
        df.set_index("datetime", inplace=True)
        df.rename(columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }, inplace=True)
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.sort_index(inplace=True)
        return df
    except Exception as e:
        logger.warning("Schwab historical failed (%s), falling back to yfinance", e)
        from data.provider import get_historical as yf_historical
        return yf_historical(symbol, months)


def days_to_expiration(expiration: str) -> int:
    """Return calendar days from today to expiration."""
    exp = date.fromisoformat(expiration)
    return max(0, (exp - date.today()).days)


def earnings_warning(quote: Quote, expiration: str) -> Optional[str]:
    """Return a warning if earnings falls before expiration."""
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
