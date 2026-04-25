"""
Shared data utilities used by both provider.py and schwab_provider.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

import pandas as pd

from strategies.models import OptionContract, Quote


@dataclass
class SelectedContract:
    """Result of a single-strike / single-expiration contract lookup."""
    contract: Optional[OptionContract]
    dte: int
    error: Optional[str] = None

    @property
    def found(self) -> bool:
        return self.contract is not None


def safe_float(val, default: float = 0.0) -> float:
    """Safely convert a value to float."""
    try:
        if val is None:
            return default
        if isinstance(val, float) and pd.isna(val):
            return default
        return float(val)
    except (TypeError, ValueError):
        return default


def safe_int(val, default: int = 0) -> int:
    """Safely convert a value to int."""
    try:
        if val is None:
            return default
        if isinstance(val, float) and pd.isna(val):
            return default
        return int(val)
    except (TypeError, ValueError):
        return default


def nearest_weekly_expiration(expirations: tuple[str, ...]) -> str:
    """Return the best expiration in the 7-14 DTE window, or nearest Friday."""
    today = date.today()
    candidates = []
    for exp_str in expirations:
        exp = date.fromisoformat(exp_str)
        dte = (exp - today).days
        if exp >= today:
            candidates.append((exp, dte))

    if not candidates:
        raise ValueError("No future expirations found.")

    ideal = [(exp, dte) for exp, dte in candidates if 7 <= dte <= 14]
    if ideal:
        return sorted(ideal, key=lambda x: x[1])[0][0].isoformat()

    fridays = [(exp, dte) for exp, dte in candidates if exp.weekday() == 4 and dte > 0]
    if fridays:
        return sorted(fridays, key=lambda x: x[1])[0][0].isoformat()

    return sorted(candidates, key=lambda x: x[1])[0][0].isoformat()


def get_next_weekly_expiration(
    base_date: str,
    available_expirations: tuple[str, ...] | list[str],
) -> Optional[str]:
    """
    Return the first expiration in *available_expirations* that is on or
    after *base_date* + 7 days.

    Used to default the Roll Leg 2 expiration to one week past Leg 1 and
    snap forward to the next valid weekly when the exact +7 date isn't
    listed.

    Fallback: if no expiration falls within 14 days of base_date, return
    the nearest available expiration to base_date+7. Returns None only if
    *available_expirations* is empty.
    """
    if not available_expirations:
        return None

    try:
        base = date.fromisoformat(base_date)
    except (TypeError, ValueError):
        return None

    target = base + timedelta(days=7)
    fourteen_days_out = base + timedelta(days=14)

    parsed: list[tuple[date, str]] = []
    for exp_str in available_expirations:
        try:
            parsed.append((date.fromisoformat(exp_str), exp_str))
        except (TypeError, ValueError):
            continue

    if not parsed:
        return None

    # Primary: first expiration >= target_date AND within the 14-day window
    in_window = [
        (exp, raw) for exp, raw in parsed
        if target <= exp <= fourteen_days_out
    ]
    if in_window:
        in_window.sort(key=lambda p: p[0])
        return in_window[0][1]

    # Fallback: closest expiration to target by absolute distance
    parsed.sort(key=lambda p: abs((p[0] - target).days))
    return parsed[0][1]


def days_to_expiration(expiration: str) -> int:
    """Return calendar days from today to expiration."""
    exp = date.fromisoformat(expiration)
    return max(0, (exp - date.today()).days)


def get_selected_contract(
    dp,
    ticker: str,
    expiration: str,
    strike: float,
    strategy: str,
) -> SelectedContract:
    """
    Fetch the option chain for *expiration* and return the contract at
    *strike* matching *strategy* ("CSP" -> puts, "CC" -> calls).

    DTE is always computed from *expiration*, independent of whether the
    strike is found — callers need the DTE even for "not found" UI state.
    """
    try:
        dte = days_to_expiration(expiration)
    except Exception as exc:
        return SelectedContract(
            contract=None,
            dte=0,
            error=f"Invalid expiration {expiration!r}: {exc}",
        )

    try:
        calls, puts = dp.get_option_chain(ticker, expiration)
    except Exception as exc:
        return SelectedContract(
            contract=None,
            dte=dte,
            error=f"Could not fetch option chain: {exc}",
        )

    chain = puts if strategy == "CSP" else calls
    contract = next(
        (c for c in chain if abs(c.strike - strike) < 0.01),
        None,
    )
    if contract is None:
        return SelectedContract(
            contract=None,
            dte=dte,
            error="Strike not found for selected expiration.",
        )

    return SelectedContract(contract=contract, dte=dte, error=None)


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
