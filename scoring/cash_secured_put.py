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
from datetime import date
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
    # Balanced: income generation with controlled assignment risk
    RiskProfile.BALANCED: {
        "premium":   0.22,
        "theta":     0.18,
        "delta":     0.18,
        "liquidity": 0.12,
        "chart":     0.08,
        "em":        0.12,
        "buy_price": 0.10,
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

_MAX_DELTA_DEFAULT: dict[RiskProfile, float] = {
    RiskProfile.CONSERVATIVE: 0.30,
    RiskProfile.BALANCED:     0.40,
    RiskProfile.AGGRESSIVE:   0.50,
}


# ---------------------------------------------------------------------------
# Sub-scorers (all return 0–100)
# ---------------------------------------------------------------------------

def _premium_score(annualised_return: float, risk_profile: RiskProfile) -> float:
    """
    Map annualised return → score using an S-curve (logistic).

    Targets represent "excellent" premium for weekly sellers:
      Conservative ~25% APY, Balanced ~40%, Aggressive ~60%+
    The curve midpoint is at 50% of target (score=50), giving good
    differentiation in the 15–60% APY range where weeklies operate.
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


def _delta_score(delta: Optional[float], max_delta: float) -> float:
    """
    Low absolute delta = high score for puts (we want OTM).
    delta here is the put delta, typically negative; we use abs().
    """
    if delta is None:
        return 50.0
    d = abs(delta)
    if d > max_delta:
        return max(0.0, 20.0 - (d - max_delta) * 200)
    return 30.0 + (max_delta - d) / max_delta * 70.0


def _liquidity_score(oi: int, spread_pct: float) -> float:
    oi_score = min(100.0, math.log1p(max(0, oi)) / math.log1p(500) * 100)
    spread_score = max(0.0, 100.0 - spread_pct * 333)
    return (oi_score + spread_score) / 2


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


def _theta_score(
    theta: Optional[float],
    risk_profile: RiskProfile,
    current_price: float = 100.0,
) -> float:
    """
    Higher absolute theta = more daily decay income = better for sellers.
    theta is negative for sold options; we score on abs(theta).

    Targets are price-relative (basis points per day) so the curve
    differentiates meaningfully across stock price ranges.
    """
    if theta is None:
        return 50.0
    # Theta as basis points of stock price per day
    theta_bps = abs(theta) / current_price * 10000 if current_price > 0 else 0
    # Targets in bps/day (calibrated for weekly OTM options)
    targets = {
        RiskProfile.CONSERVATIVE: 3.0,   # e.g. $0.06/day on $200 stock
        RiskProfile.BALANCED:     5.0,   # e.g. $0.10/day on $200 stock
        RiskProfile.AGGRESSIVE:   8.0,   # e.g. $0.16/day on $200 stock
    }
    target = targets[risk_profile]
    k = 2.0
    score = 100 * (1 - math.exp(-k * theta_bps / target))
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
    expected_move: Optional[float] = None,
) -> list[ScoredOption]:
    """
    Score each put option and return a filtered, sorted list of ScoredOption.

    iv_52w_high / iv_52w_low: optional 52-week IV range for the underlying.
    When provided, IV rank is computed per-contract using each contract's
    implied_volatility as the current IV reading.  When omitted, iv_rank_score
    defaults to 50 (neutral) so the weight is effectively zeroed out.
    """
    # Adaptive weight redistribution: if no desired buy price, redistribute
    # the "buy_price" weight proportionally to other factors
    weights = dict(_WEIGHTS[params.risk_profile])
    if params.desired_buy_price is None:
        bp_w = weights.pop("buy_price")
        total = sum(weights.values())
        for k in weights:
            weights[k] += bp_w * (weights[k] / total)
    max_delta = params.max_delta or _MAX_DELTA_DEFAULT[params.risk_profile]
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
        em = expected_move if expected_move is not None else (
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
        penalty = 1.0

        # Regime penalty
        if regime.trade_bias == "skip":
            penalty *= 0.35
        elif regime.trade_bias == "caution":
            penalty *= 0.75

        # Earnings penalty (binary event risk)
        earnings_in_window = False
        if earnings_date:
            try:
                ed = date.fromisoformat(earnings_date)
                exp = date.fromisoformat(c.expiration)
                if ed <= exp:
                    earnings_in_window = True
                    penalty *= 0.65
            except Exception:
                pass

        # Vega risk penalty
        penalty *= _vega_penalty(c.vega, params.risk_profile)

        # Floor: never reduce by more than 55% (except "skip" regime
        # which is an explicit do-not-trade signal)
        if regime.trade_bias != "skip":
            penalty = max(0.45, penalty)

        composite *= penalty
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
