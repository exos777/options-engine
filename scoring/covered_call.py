"""
Covered call scoring engine.

For each call option in the chain, calculates all relevant metrics and
produces a composite score 0–100 that balances:

  - Premium / annualised return
  - Theta / daily decay income
  - Delta / assignment risk
  - Liquidity (OI + bid-ask spread)
  - Chart alignment
  - Expected move (outside EM = safer)
  - Cost basis compliance

Weights are adjusted based on the user's risk profile.
All modifiers (regime, earnings, vega) are applied as a single capped
penalty to prevent excessive stacking.
"""

from __future__ import annotations

import math
from typing import Optional

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
from scoring.common import (
    MAX_DELTA_DEFAULT,
    apply_risk_penalties,
    delta_score as _delta_score,
    liquidity_score as _liquidity_score,
    premium_score as _premium_score,
    theta_score as _theta_score,
)
from indicators.support_resistance import is_near_level


# ---------------------------------------------------------------------------
# Weight profiles (must sum to 1.0)
# ---------------------------------------------------------------------------

_WEIGHTS: dict[RiskProfile, dict[str, float]] = {
    # Conservative: safety first — delta and liquidity dominate
    RiskProfile.CONSERVATIVE: {
        "premium":   0.12,
        "theta":     0.12,
        "delta":     0.28,
        "liquidity": 0.18,
        "chart":     0.12,
        "em":        0.13,
        "basis":     0.05,
    },
    # Balanced: income generation with controlled risk
    RiskProfile.BALANCED: {
        "premium":   0.22,
        "theta":     0.18,
        "delta":     0.18,
        "liquidity": 0.12,
        "chart":     0.08,
        "em":        0.12,
        "basis":     0.10,
    },
    # Aggressive: maximize premium and theta income
    RiskProfile.AGGRESSIVE: {
        "premium":   0.28,
        "theta":     0.22,
        "delta":     0.13,
        "liquidity": 0.08,
        "chart":     0.08,
        "em":        0.14,
        "basis":     0.07,
    },
}


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


def _expected_move_score(
    strike: float,
    current_price: float,
    expected_move: float,
) -> float:
    """
    Reward covered call strikes outside the expected move range.

    Upper bound = current_price + expected_move
    strike above upper_bound → outside EM → safer → high score
    strike inside EM         → risky      → low score
    Returns 0–100 (used as a weighted sub-score, not a multiplier).
    """
    if expected_move <= 0:
        return 50.0  # neutral when EM data unavailable

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
    expected_move: float = 0.0,
) -> list[ScoredOption]:
    """
    Score each call option contract and return a list of ScoredOption,
    filtered by the user's FilterParams constraints.

    Contracts that fail hard filters (spread, OI, premium, delta) are excluded.
    """
    # Adaptive weight redistribution: if no cost basis, redistribute
    # the "basis" weight proportionally to other factors
    weights = dict(_WEIGHTS[params.risk_profile])
    if params.cost_basis is None:
        basis_w = weights.pop("basis")
        total = sum(weights.values())
        for k in weights:
            weights[k] += basis_w * (weights[k] / total)
    max_delta = params.max_delta or MAX_DELTA_DEFAULT[params.risk_profile]
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

        # --- Expected move ---
        em = expected_move if expected_move > 0 else (
            current_price * c.implied_volatility * math.sqrt(max(dte, 1) / 365)
            if c.implied_volatility
            else indicators.atr_14 * math.sqrt(max(dte, 1))
        )

        # --- Sub-scores ---
        ps = _premium_score(ann_return, params.risk_profile)
        ts = _theta_score(c.theta, params.risk_profile, current_price)
        ds = _delta_score(c.delta, max_delta)
        ls = _liquidity_score(c.open_interest, c.bid_ask_spread_pct)
        cs = _chart_score(c.strike, current_price, indicators, regime, resistance_levels)
        ems = _expected_move_score(c.strike, current_price, em)
        bs = _basis_score(c.strike, params.cost_basis, premium, params.allow_below_basis)

        # --- Weighted composite (EM is now a sub-score, not a multiplier) ---
        composite = (
            weights["premium"]   * ps
            + weights["theta"]   * ts
            + weights["delta"]   * ds
            + weights["liquidity"] * ls
            + weights["chart"]   * cs
            + weights["em"]      * ems
        )
        if "basis" in weights:
            composite += weights["basis"] * bs

        # --- Capped risk modifiers (prevent excessive stacking) ---
        composite, earnings_in_window = apply_risk_penalties(
            composite, c, regime, params.risk_profile, earnings_date,
        )

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
                expected_move_score=round(ems, 2),
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
