"""
Shared scoring utilities used by both covered_call.py and cash_secured_put.py.

These sub-scorers and helpers were duplicated across both scoring engines.
Centralising them here ensures consistency and reduces maintenance burden.

IMPORTANT: Do NOT change scoring weights or trading logic — only shared helpers.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Optional

from strategies.models import (
    OptionContract,
    RegimeResult,
    RiskProfile,
    ScoredOption,
    SupportResistanceLevel,
    TechnicalIndicators,
)


# ---------------------------------------------------------------------------
# Max delta defaults by risk profile
# ---------------------------------------------------------------------------

MAX_DELTA_DEFAULT: dict[RiskProfile, float] = {
    RiskProfile.CONSERVATIVE: 0.30,
    RiskProfile.BALANCED:     0.40,
    RiskProfile.AGGRESSIVE:   0.50,
}


# ---------------------------------------------------------------------------
# Sub-scorers shared between CC and CSP (all return 0-100)
# ---------------------------------------------------------------------------

def premium_score(annualised_return: float, risk_profile: RiskProfile) -> float:
    """
    Map annualised return to score using an S-curve (logistic).

    Targets represent "excellent" premium for weekly sellers:
      Conservative ~25% APY, Balanced ~40%, Aggressive ~60%+
    """
    targets = {
        RiskProfile.CONSERVATIVE: 0.25,
        RiskProfile.BALANCED:     0.40,
        RiskProfile.AGGRESSIVE:   0.60,
    }
    target = targets[risk_profile]
    k = 8.0
    score = 100.0 / (1.0 + math.exp(-k * (annualised_return / target - 0.5)))
    return min(100.0, max(0.0, score))


def delta_score(delta: Optional[float], max_delta: float) -> float:
    """Low absolute delta = high score. Penalises contracts above max_delta."""
    if delta is None:
        return 50.0
    d = abs(delta)
    if d > max_delta:
        return max(0.0, 20.0 - (d - max_delta) * 200)
    return 30.0 + (max_delta - d) / max_delta * 70.0


def liquidity_score(oi: int, spread_pct: float) -> float:
    """Combine open interest and bid-ask spread quality into 0-100."""
    oi_score = min(100.0, math.log1p(max(0, oi)) / math.log1p(500) * 100)
    spread_score = max(0.0, 100.0 - spread_pct * 333)
    return (oi_score + spread_score) / 2


def theta_score(
    theta: Optional[float],
    risk_profile: RiskProfile,
    current_price: float = 100.0,
) -> float:
    """
    Higher absolute theta = more daily decay income = better for sellers.
    Targets are price-relative (bps per day).
    """
    if theta is None:
        return 50.0
    theta_bps = abs(theta) / current_price * 10000 if current_price > 0 else 0
    targets = {
        RiskProfile.CONSERVATIVE: 3.0,
        RiskProfile.BALANCED:     5.0,
        RiskProfile.AGGRESSIVE:   8.0,
    }
    target = targets[risk_profile]
    k = 2.0
    score = 100 * (1 - math.exp(-k * theta_bps / target))
    return min(100.0, max(0.0, score))


def vega_penalty(vega: Optional[float], risk_profile: RiskProfile) -> float:
    """Return a multiplier (0.60-1.0). High vega = IV expansion risk."""
    if vega is None:
        return 1.0
    thresholds = {
        RiskProfile.CONSERVATIVE: 0.15,
        RiskProfile.BALANCED:     0.25,
        RiskProfile.AGGRESSIVE:   0.35,
    }
    threshold = thresholds[risk_profile]
    abs_vega = abs(vega)
    if abs_vega <= threshold:
        return 1.0
    excess = abs_vega - threshold
    return max(0.60, 1.0 - excess * 2.0)


# ---------------------------------------------------------------------------
# Shared penalty logic
# ---------------------------------------------------------------------------

def apply_risk_penalties(
    composite: float,
    contract: OptionContract,
    regime: RegimeResult,
    risk_profile: RiskProfile,
    earnings_date: Optional[str] = None,
) -> tuple[float, bool]:
    """
    Apply regime, earnings, and vega penalties with a capped floor.
    Returns (adjusted_composite, earnings_in_window).
    """
    penalty = 1.0

    # Regime penalty
    if regime.trade_bias == "skip":
        penalty *= 0.35
    elif regime.trade_bias == "caution":
        penalty *= 0.75

    # Earnings penalty
    earnings_in_window = False
    if earnings_date:
        try:
            ed = date.fromisoformat(earnings_date)
            exp = date.fromisoformat(contract.expiration)
            if ed <= exp:
                earnings_in_window = True
                penalty *= 0.65
        except Exception:
            pass

    # Vega risk penalty
    penalty *= vega_penalty(contract.vega, risk_profile)

    # Floor: never reduce by more than 55% (except "skip")
    if regime.trade_bias != "skip":
        penalty = max(0.45, penalty)

    composite = min(100.0, composite * penalty)
    return composite, earnings_in_window
