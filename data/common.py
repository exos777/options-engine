"""
Shared data utilities used by both provider.py and schwab_provider.py.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

import pandas as pd

from strategies.models import Quote


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
