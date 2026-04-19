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
