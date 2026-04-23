"""
Cash-secured put scoring engine.

For each put option, calculates all relevant metrics and produces a
composite score 0–100 that balances:

  - Premium / annualised return
  - Theta / daily decay income
  - Delta / assignment risk
  - Liquidity (OI + bid-ask spread)
  - Chart alignment (near support)
  - Expected move (outside EM = safer)
  - Desired buy price proximity

Weights are adjusted based on the user's risk profile.
All modifiers (regime, earnings, vega) are applied as a single capped
penalty to prevent excessive stacking.
IV rank defaults to 50 (neutral) when 52-week IV data is unavailable.
"""

from __future__ import annotations

import math
from typing import Optional

from strategies.models import (
    ChartRegime,
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
        "buy_price": 0.05,
    },
    # Balanced: wheel-aligned income with assignment quality focus
    RiskProfile.BALANCED: {
        "premium":   0.15,
        "theta":     0.10,
        "delta":     0.15,
        "liquidity": 0.10,
        "chart":     0.15,
        "em":        0.15,
        "buy_price": 0.20,
    },
    # Aggressive: maximize premium and theta income
    RiskProfile.AGGRESSIVE: {
        "premium":   0.28,
        "theta":     0.22,
        "delta":     0.13,
        "liquidity": 0.08,
        "chart":     0.08,
        "em":        0.14,
        "buy_price": 0.07,
    },
}


def _chart_score(
    strike: float,
    current_price: float,
    indicators: TechnicalIndicators,
    regime: RegimeResult,
    support_levels: list[SupportResistanceLevel],
) -> float:
    """
    CSP chart alignment score.

    Favors strikes:
    - At or below the nearest support level
    - Outside the expected weekly move
    - In bullish / near-support regimes
    """
    multiplier = chart_score_multiplier(regime, "cash_secured_put")
    distance_pct = (current_price - strike) / current_price  # positive = OTM

    # Base score: reward OTM distance up to sweet spot
    primary = regime.primary
    if primary in (ChartRegime.BULLISH, ChartRegime.NEAR_SUPPORT):
        sweet_spot = 0.03
    elif primary == ChartRegime.NEUTRAL:
        sweet_spot = 0.04
    else:
        sweet_spot = 0.05  # bearish: need more cushion to score well

    raw = max(0.0, 1.0 - abs(distance_pct - sweet_spot) / 0.08) * 100

    # Strong bonus: strike is at or just below a support level
    for sl in support_levels:
        if sl.price >= strike and (sl.price - strike) / sl.price <= 0.015:
            raw = min(100.0, raw + 15 * sl.strength)

    # Penalty: strike is above a support level (support won't help if assigned)
    for sl in support_levels:
        if strike > sl.price and (strike - sl.price) / sl.price <= 0.02:
            raw = max(0.0, raw - 10)

    return raw * multiplier


def _iv_rank_score(current_iv: float, iv_52w_high: float, iv_52w_low: float) -> float:
    """High IV rank = better premium selling environment."""
    if iv_52w_high == iv_52w_low:
        return 50.0
    iv_rank = (current_iv - iv_52w_low) / (iv_52w_high - iv_52w_low) * 100
    return min(100.0, iv_rank)


def _expected_move_score(
    strike: float,
    current_price: float,
    expected_move: float,
) -> float:
    """
    Reward strikes outside the expected move range.

    Expected move defines the 1-sigma range:
    lower_bound = current_price - expected_move

    strike below lower_bound → outside EM → safer → high score
    strike inside EM         → risky      → low score
    Returns 0–100 (used as a weighted sub-score, not a multiplier).
    """
    lower_bound = current_price - expected_move

    if expected_move <= 0:
        return 50.0  # neutral if no IV data

    # How far outside the expected move is the strike?
    distance_beyond = lower_bound - strike

    if distance_beyond >= 0:
        # Outside expected move — reward it
        pct_beyond = distance_beyond / expected_move
        return min(100.0, 60.0 + pct_beyond * 200)
    else:
        # Inside expected move — penalize it
        pct_inside = abs(distance_beyond) / expected_move
        return max(0.0, 60.0 - pct_inside * 120)


def _buy_price_score(
    strike: float,
    desired_buy_price: Optional[float],
    premium: float,
) -> float:
    """
    Score how close the net cost (strike - premium) is to the user's
    desired acquisition price.
    """
    if desired_buy_price is None:
        return 75.0  # neutral

    net_cost = strike - premium
    deviation_pct = abs(net_cost - desired_buy_price) / desired_buy_price

    if deviation_pct <= 0.01:
        return 100.0
    elif deviation_pct <= 0.03:
        return 85.0 - deviation_pct * 300
    elif deviation_pct <= 0.08:
        return 75.0 - deviation_pct * 250
    else:
        return max(0.0, 50.0 - deviation_pct * 100)


# ---------------------------------------------------------------------------
# Per-contract metrics
# ---------------------------------------------------------------------------

def _calc_annualised_return(
    premium: float,
    strike: float,
    dte: int,
) -> float:
    """Premium / capital at risk, annualised. Capital = strike × 100."""
    if strike <= 0 or dte <= 0:
        return 0.0
    return (premium / strike) * (365 / dte)


def _assignment_attractiveness(
    strike: float,
    current_price: float,
    desired_buy_price: Optional[float],
    support_levels: list[SupportResistanceLevel],
) -> str:
    """Plain-English note on assignment attractiveness."""
    net = strike  # simplified; full net = strike - premium at time of assignment
    if desired_buy_price and abs(net - desired_buy_price) / desired_buy_price <= 0.05:
        return "Near desired buy price — assignment is attractive."
    near_sup = any(
        abs(strike - s.price) / s.price <= 0.02 for s in support_levels
    )
    if near_sup:
        return "Strike is near support — assignment at a historically defended level."
    if strike < current_price * 0.95:
        return "Strike is well below current price — assignment is a significant discount."
    return "Strike is close to current price — assignment likely if stock pulls back modestly."


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def score_cash_secured_puts(
    puts: list[OptionContract],
    current_price: float,
    dte: int,
    indicators: TechnicalIndicators,
    regime: RegimeResult,
    support_levels: list[SupportResistanceLevel],
    resistance_levels: list[SupportResistanceLevel],
    params: FilterParams,
    iv_52w_high: Optional[float] = None,
    iv_52w_low: Optional[float] = None,
    earnings_date: Optional[str] = None,
    expected_move: float = 0.0,
    iv_percentile: Optional[float] = None,
) -> list[ScoredOption]:
    """
    Score each put option and return a filtered, sorted list of ScoredOption.

    iv_52w_high / iv_52w_low: optional 52-week IV range for the underlying.
    When provided, IV rank is computed per-contract using each contract's
    implied_volatility as the current IV reading.  When omitted, iv_rank_score
    defaults to 50 (neutral) so the weight is effectively zeroed out.
    """
    if dte < 5:
        return []
    if dte > 21:
        return []
    if iv_percentile is not None and iv_percentile < 25:
        return []

    # Adaptive weight redistribution: if no desired buy price, redistribute
    # the "buy_price" weight proportionally to other factors
    weights = dict(_WEIGHTS[params.risk_profile])
    if params.desired_buy_price is None:
        bp_w = weights.pop("buy_price")
        total = sum(weights.values())
        for k in weights:
            weights[k] += bp_w * (weights[k] / total)
    max_delta = params.max_delta or MAX_DELTA_DEFAULT[params.risk_profile]
    scored: list[ScoredOption] = []

    for c in puts:
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
        # Only OTM or ATM puts for CSPs
        if c.strike > current_price * 1.03:
            continue

        # --- Metrics ---
        premium = c.mid
        ann_return = _calc_annualised_return(premium, c.strike, dte)
        distance_pct = (current_price - c.strike) / current_price
        break_even = c.strike - premium  # net cost basis if assigned

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
        cs = _chart_score(c.strike, current_price, indicators, regime, support_levels)
        ems = _expected_move_score(c.strike, current_price, em)
        bps = _buy_price_score(c.strike, params.desired_buy_price, premium)
        # iv_rank computed for storage but not weighted (kept for future use)
        ivs = (
            _iv_rank_score(c.implied_volatility, iv_52w_high, iv_52w_low)
            if (c.implied_volatility and iv_52w_high is not None and iv_52w_low is not None)
            else 50.0
        )

        # --- Weighted composite (EM is now a sub-score, not a multiplier) ---
        composite = (
            weights["premium"]   * ps
            + weights["theta"]   * ts
            + weights["delta"]   * ds
            + weights["liquidity"] * ls
            + weights["chart"]   * cs
            + weights["em"]      * ems
        )
        if "buy_price" in weights:
            composite += weights["buy_price"] * bps

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
                basis_score=round(bps, 2),
                iv_rank_score=round(ivs, 2),
                expected_move_score=round(ems, 2),
                near_support=near_sup,
                near_resistance=near_res,
                above_cost_basis=None,
                earnings_in_window=earnings_in_window,
            )
        )

    scored.sort(key=lambda x: x.score, reverse=True)
    return scored
