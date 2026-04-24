"""
New CSP scoring formula under evaluation.

    score = premium_weight * premium
          + theta_weight * |theta|
          + distance_weight * dist_pct
          - delta_penalty * |delta|
          - spread_penalty * spread_pct

If break_even > desired_buy_price → apply LARGE_PENALTY.
"""

from __future__ import annotations

from scoring_lab.data import StrikeData


PREMIUM_WEIGHT = 0.25
THETA_WEIGHT = 0.20
DISTANCE_WEIGHT = 0.20
DELTA_PENALTY = 0.20
SPREAD_PENALTY = 0.15
LARGE_PENALTY = 1000.0


def new_score(strike_data: StrikeData, desired_buy_price: float) -> float:
    """Return the raw composite score for one strike under the new formula."""
    score = (
        PREMIUM_WEIGHT * strike_data.premium
        + THETA_WEIGHT * abs(strike_data.theta)
        + DISTANCE_WEIGHT * strike_data.dist_pct
        - DELTA_PENALTY * abs(strike_data.delta)
        - SPREAD_PENALTY * strike_data.spread_pct
    )
    if strike_data.break_even > desired_buy_price:
        score -= LARGE_PENALTY
    return score


def new_score_all(
    strikes: list[StrikeData],
    desired_buy_price: float,
) -> list[tuple[StrikeData, float]]:
    """Score every strike and return (strike, score) pairs, best first."""
    pairs = [(s, new_score(s, desired_buy_price)) for s in strikes]
    pairs.sort(key=lambda p: p[1], reverse=True)
    return pairs
