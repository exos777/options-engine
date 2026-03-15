"""
Covered call scoring engine.

For each call option in the chain, calculates all relevant metrics and
produces a composite score 0–100 that balances:

  - Premium / annualised return          (weight: 20%)
  - Delta / assignment risk              (weight: 25%)
  - Liquidity (OI + bid-ask spread)      (weight: 20%)
  - Chart alignment                      (weight: 20%)
  - Cost basis compliance                (weight: 15%)

Weights are adjusted based on the user's risk profile.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Optional

import pandas as pd

from strategies.models import (
    FilterParams,
    OptionContract,
    RegimeResult,
    RiskProfile,
    ScoredOption,
    SupportResistanceLevel,
    TechnicalIndicators,
)
from scoring.regime import chart_score_multiplier
from indicators.support_resistance import is_near_level


# ---------------------------------------------------------------------------
# Weight profiles
# ---------------------------------------------------------------------------

_WEIGHTS: dict[RiskProfile, dict[str, float]] = {
    # Conservative: safety first — delta and liquidity dominate
    RiskProfile.CONSERVATIVE: {
        "premium":   0.15,
        "theta":     0.15,
        "delta":     0.30,
        "liquidity": 0.20,
        "chart":     0.15,
        "basis":     0.05,
    },
    # Balanced: income generation with controlled risk
    RiskProfile.BALANCED: {
        "premium":   0.25,
        "theta":     0.20,
        "delta":     0.20,
        "liquidity": 0.15,
        "chart":     0.10,
        "basis":     0.10,
    },
    # Aggressive: maximize premium and theta income
    RiskProfile.AGGRESSIVE: {
        "premium":   0.35,
        "theta":     0.25,
        "delta":     0.15,
        "liquidity": 0.10,
        "chart":     0.10,
        "basis":     0.05,
    },
}

# Delta thresholds by risk profile (max delta considered acceptable)
_MAX_DELTA_DEFAULT: dict[RiskProfile, float] = {
    RiskProfile.CONSERVATIVE: 0.30,
    RiskProfile.BALANCED:     0.40,
    RiskProfile.AGGRESSIVE:   0.50,
}


# ---------------------------------------------------------------------------
# Individual sub-scorers (all return 0–100)
# ---------------------------------------------------------------------------

def _premium_score(annualised_return: float, risk_profile: RiskProfile) -> float:
    """
    Map annualised return → score.
    Conservative: peak at ~15% APY; Balanced ~25%; Aggressive ~40%+
    """
    targets = {
        RiskProfile.CONSERVATIVE: 0.15,
        RiskProfile.BALANCED:     0.25,
        RiskProfile.AGGRESSIVE:   0.40,
    }
    target = targets[risk_profile]
    # Logistic-style score: 100 × (1 - exp(-k × r/target))
    # k chosen so that at r == target we get ~63 points
    k = 2.5
    score = 100 * (1 - math.exp(-k * annualised_return / target))
    return min(100.0, max(0.0, score))


def _delta_score(delta: Optional[float], max_delta: float) -> float:
    """
    Low delta = high score. Penalises contracts above max_delta heavily.
    delta is expected as a positive number (call delta 0–1).
    """
    if delta is None:
        return 50.0  # unknown — neutral
    d = abs(delta)
    if d > max_delta:
        # Hard penalty above the limit
        return max(0.0, 20.0 - (d - max_delta) * 200)
    # Linear interpolation: delta=0 → 100, delta=max_delta → 30
    return 30.0 + (max_delta - d) / max_delta * 70.0


def _liquidity_score(oi: int, spread_pct: float) -> float:
    """
    Combine open interest and bid-ask spread quality into 0–100.
    """
    # OI component: 50 → 50pts, 500 → 100pts (log scale)
    oi_score = min(100.0, math.log1p(max(0, oi)) / math.log1p(500) * 100)
    # Spread component: 0% → 100pts, 30% → 0pts
    spread_score = max(0.0, 100.0 - spread_pct * 333)
    return (oi_score + spread_score) / 2


def _chart_score(
    strike: float,
    current_price: float,
    indicators: TechnicalIndicators,
    regime: RegimeResult,
    resistance_levels: list[SupportResistanceLevel],
) -> float:
    """
    Covered call chart alignment score.

    Higher score when:
    - Bullish trend → strike is comfortably above current price
    - Near resistance → strike slightly above resistance (let it get called)
    - Neutral / overextended → closer OTM is OK
    """
    multiplier = chart_score_multiplier(regime, "covered_call")

    distance_pct = (strike - current_price) / current_price

    from strategies.models import ChartRegime
    primary = regime.primary

    # Preferred OTM distance by regime
    if primary == ChartRegime.BULLISH:
        sweet_spot = 0.03   # 3% OTM ideal
    elif primary == ChartRegime.NEAR_RESISTANCE:
        # Near resistance: slightly above the resistance level is good
        sweet_spot = 0.01
    elif primary == ChartRegime.OVEREXTENDED:
        sweet_spot = 0.02
    else:
        sweet_spot = 0.02

    # Score based on proximity to sweet-spot OTM distance
    raw = max(0.0, 1.0 - abs(distance_pct - sweet_spot) / 0.08) * 100
    # Bonus if strike sits just above a resistance level
    for rl in resistance_levels:
        if 0 <= (strike - rl.price) / rl.price <= 0.01:
            raw = min(100.0, raw + 10)

    return raw * multiplier


def _theta_score(theta: Optional[float], risk_profile: RiskProfile) -> float:
    """
    Higher absolute theta = more daily decay income = better for sellers.
    theta is negative for sold options; we score on abs(theta).

    Uses the same logistic curve as premium_score, targeted at daily decay.
    """
    if theta is None:
        return 50.0  # neutral when Greeks unavailable
    targets = {
        RiskProfile.CONSERVATIVE: 0.05,  # $0.05/day target
        RiskProfile.BALANCED:     0.10,  # $0.10/day target
        RiskProfile.AGGRESSIVE:   0.20,  # $0.20/day target
    }
    target = targets[risk_profile]
    k = 2.5
    score = 100 * (1 - math.exp(-k * abs(theta) / target))
    return min(100.0, max(0.0, score))


def _vega_penalty(vega: Optional[float], risk_profile: RiskProfile) -> float:
    """
    Return a composite multiplier (0.60–1.0).
    High vega = risk of IV expansion hurting the short position.
    """
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


def _expected_move_score(
    strike: float,
    current_price: float,
    expected_move: float,
) -> float:
    """
    Reward covered call strikes outside the expected move range.

    Upper bound = current_price + expected_move
    strike above upper_bound → outside EM → safer → high score (→ multiplier > 1.0)
    strike inside EM         → risky      → low score (→ multiplier < 1.0)
    Returns 0–100; converted to a composite multiplier by caller.
    """
    if expected_move <= 0:
        return 60.0  # neutral — maps to multiplier ~1.0

    upper_bound = current_price + expected_move
    distance_beyond = strike - upper_bound

    if distance_beyond >= 0:
        pct_beyond = distance_beyond / expected_move
        return min(100.0, 60.0 + pct_beyond * 200)
    else:
        pct_inside = abs(distance_beyond) / expected_move
        return max(0.0, 60.0 - pct_inside * 120)


def _basis_score(
    strike: float,
    cost_basis: Optional[float],
    premium: float,
    allow_below_basis: bool,
) -> float:
    """
    Score how well the strike respects the user's cost basis.
    - Strike > cost_basis → full score (captures some capital gain)
    - Strike between cost_basis-premium and cost_basis → partial (break-even zone)
    - Strike < cost_basis-premium → negative territory
    """
    if cost_basis is None:
        return 75.0  # neutral if unknown

    effective_breakeven = cost_basis - premium
    if strike >= cost_basis:
        return 100.0
    elif strike >= effective_breakeven:
        # In the break-even zone: interpolate 60–99
        pct_in = (strike - effective_breakeven) / (cost_basis - effective_breakeven)
        return 60.0 + pct_in * 39.0
    else:
        if not allow_below_basis:
            return 0.0
        # Allowed below basis: penalise proportional to how far below
        drop = (effective_breakeven - strike) / effective_breakeven
        return max(0.0, 40.0 - drop * 200)


# ---------------------------------------------------------------------------
# Per-contract metrics
# ---------------------------------------------------------------------------

def _calc_annualised_return(premium: float, stock_price: float, dte: int) -> float:
    """Premium as a fraction of stock price, annualised."""
    if stock_price <= 0 or dte <= 0:
        return 0.0
    return (premium / stock_price) * (365 / dte)


def _calc_max_profit(strike: float, cost_basis: Optional[float], premium: float) -> Optional[float]:
    if cost_basis is None:
        return None
    return (strike - cost_basis) + premium


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def score_covered_calls(
    calls: list[OptionContract],
    current_price: float,
    dte: int,
    indicators: TechnicalIndicators,
    regime: RegimeResult,
    support_levels: list[SupportResistanceLevel],
    resistance_levels: list[SupportResistanceLevel],
    params: FilterParams,
    earnings_date: Optional[str] = None,
    expected_move: Optional[float] = None,
) -> list[ScoredOption]:
    """
    Score each call option contract and return a list of ScoredOption,
    filtered by the user's FilterParams constraints.

    Contracts that fail hard filters (spread, OI, premium, delta) are excluded.
    """
    weights = _WEIGHTS[params.risk_profile]
    max_delta = params.max_delta or _MAX_DELTA_DEFAULT[params.risk_profile]
    scored: list[ScoredOption] = []

    for c in calls:
        # --- Hard filters ---
        if c.open_interest < params.min_open_interest:
            continue
        if c.bid_ask_spread_pct > params.max_spread_pct:
            continue
        if c.mid < params.min_premium:
            continue
        delta_abs = abs(c.delta) if c.delta is not None else None
        if delta_abs is not None and delta_abs > max_delta:
            continue
        # Only OTM or ATM calls for covered calls
        if c.strike < current_price * 0.97:
            continue
        # Optionally enforce cost basis
        if (
            params.cost_basis is not None
            and not params.allow_below_basis
            and c.strike < params.cost_basis - c.mid
        ):
            continue

        # --- Metrics ---
        premium = c.mid
        ann_return = _calc_annualised_return(premium, current_price, dte)
        distance_pct = (c.strike - current_price) / current_price
        break_even = current_price - premium  # break-even on downside

        # --- Sub-scores ---
        ps = _premium_score(ann_return, params.risk_profile)
        ts = _theta_score(c.theta, params.risk_profile)
        ds = _delta_score(c.delta, max_delta)
        ls = _liquidity_score(c.open_interest, c.bid_ask_spread_pct)
        cs = _chart_score(c.strike, current_price, indicators, regime, resistance_levels)
        bs = _basis_score(c.strike, params.cost_basis, premium, params.allow_below_basis)

        # --- Weighted composite ---
        composite = (
            weights["premium"]   * ps
            + weights["theta"]   * ts
            + weights["delta"]   * ds
            + weights["liquidity"] * ls
            + weights["chart"]   * cs
            + weights["basis"]   * bs
        )

        # --- Regime modifier ---
        if regime.trade_bias == "skip":
            composite *= 0.35
        elif regime.trade_bias == "caution":
            composite *= 0.75

        # --- Earnings penalty (binary event risk) ---
        earnings_in_window = False
        if earnings_date:
            try:
                ed = date.fromisoformat(earnings_date)
                exp = date.fromisoformat(c.expiration)
                if ed <= exp:
                    earnings_in_window = True
                    composite *= 0.65
            except Exception:
                pass

        # --- Expected move modifier (outside EM = bonus, inside = penalty) ---
        em = expected_move if expected_move is not None else (
            current_price * c.implied_volatility * math.sqrt(max(dte, 1) / 365)
            if c.implied_volatility
            else indicators.atr_14 * math.sqrt(max(dte, 1))
        )
        ems = _expected_move_score(c.strike, current_price, em)
        composite *= (0.80 + (ems / 100.0) * 0.30)

        # --- Vega risk modifier ---
        composite *= _vega_penalty(c.vega, params.risk_profile)

        composite = min(100.0, composite)

        near_sup = any(is_near_level(c.strike, s) for s in support_levels)
        near_res = any(is_near_level(c.strike, r) for r in resistance_levels)

        scored.append(
            ScoredOption(
                contract=c,
                premium=premium,
                annualized_return=ann_return,
                distance_pct=distance_pct,
                break_even=break_even,
                score=round(composite, 2),
                premium_score=round(ps, 2),
                theta_score=round(ts, 2),
                delta_score=round(ds, 2),
                liquidity_score=round(ls, 2),
                chart_score=round(cs, 2),
                basis_score=round(bs, 2),
                near_support=near_sup,
                near_resistance=near_res,
                above_cost_basis=(
                    c.strike >= (params.cost_basis or 0.0)
                    if params.cost_basis is not None else None
                ),
                earnings_in_window=earnings_in_window,
            )
        )

    scored.sort(key=lambda x: x.score, reverse=True)
    return scored
