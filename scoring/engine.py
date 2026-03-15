"""
Recommendation engine.

Orchestrates the full pipeline:
  1. Receives scored options (already filtered)
  2. Selects the top 3 with distinct labels: Safest Income, Best Balance, Max Premium
  3. Generates plain-English explanations for each
  4. Builds the ScreenerResult for the UI layer

Plain-English explanation rules:
  - Safest Income  → lowest delta / widest OTM from the top-10 scored
  - Best Balance   → highest composite score overall
  - Max Premium    → highest annualised return from the top-10 scored
"""

from __future__ import annotations

import math
from datetime import date

import pandas as pd

from strategies.models import (
    ChartRegime,
    FilterParams,
    Quote,
    Recommendation,
    RecommendationLabel,
    RegimeResult,
    ScreenerResult,
    ScoredOption,
    Strategy,
    SupportResistanceLevel,
    TechnicalIndicators,
)


# ---------------------------------------------------------------------------
# Explanation generation
# ---------------------------------------------------------------------------

def _regime_phrase(regime: RegimeResult) -> str:
    phrases = {
        ChartRegime.BULLISH:          "the stock is in an uptrend",
        ChartRegime.BEARISH:          "the chart is in a downtrend",
        ChartRegime.NEUTRAL:          "the chart is range-bound",
        ChartRegime.OVEREXTENDED:     "the stock looks extended to the upside",
        ChartRegime.NEAR_SUPPORT:     "price is sitting near a support zone",
        ChartRegime.NEAR_RESISTANCE:  "price is approaching a resistance level",
    }
    return phrases.get(regime.primary, "the chart is mixed")


def _explain_covered_call(
    option: ScoredOption,
    regime: RegimeResult,
    indicators: TechnicalIndicators,
    label: RecommendationLabel,
) -> str:
    c = option.contract
    distance = option.distance_pct * 100
    ann_ret = option.annualized_return * 100
    regime_phrase = _regime_phrase(regime)

    if regime.trade_bias == "skip":
        return (
            f"No strong trade recommended — {regime_phrase}. "
            f"The ${c.strike:.2f} call offers {ann_ret:.1f}% annualised, "
            "but chart conditions do not support selling calls at this time. "
            "Consider waiting for a clearer setup."
        )

    base = f"The ${c.strike:.2f} call is {distance:.1f}% above the current price"
    premium_note = f"and earns ${option.premium:.2f} in premium ({ann_ret:.1f}% annualised)"

    if label == RecommendationLabel.SAFEST_INCOME:
        delta_note = (
            f" with a low delta of {abs(c.delta):.2f}" if c.delta else ""
        )
        return (
            f"{base}{delta_note}, {premium_note}. "
            f"This is the safest income option because {regime_phrase} "
            "and the strike gives the stock plenty of room to run before assignment. "
            f"Break-even on the downside is ${option.break_even:.2f}."
        )

    if label == RecommendationLabel.BEST_BALANCE:
        chart_note = ""
        if option.near_resistance:
            chart_note = " The strike sits near resistance, making assignment at a reasonable price likely."
        return (
            f"{base}, {premium_note}. "
            f"This is the best balance of premium, safety, and chart context because {regime_phrase}.{chart_note} "
            f"Break-even is ${option.break_even:.2f}."
        )

    if label == RecommendationLabel.MAX_PREMIUM:
        caution = ""
        if option.contract.delta and abs(option.contract.delta) > 0.35:
            caution = (
                f" Note: delta of {abs(c.delta):.2f} means higher assignment risk — "
                "suitable if you are comfortable being called away."
            )
        return (
            f"{base}, {premium_note} — the highest available premium. "
            f"Chart context: {regime_phrase}.{caution} "
            f"Break-even is ${option.break_even:.2f}."
        )

    return f"{base}, {premium_note}."


def _explain_csp(
    option: ScoredOption,
    regime: RegimeResult,
    indicators: TechnicalIndicators,
    label: RecommendationLabel,
) -> str:
    c = option.contract
    distance = option.distance_pct * 100
    ann_ret = option.annualized_return * 100
    regime_phrase = _regime_phrase(regime)

    if regime.trade_bias == "skip":
        return (
            f"No strong trade recommended — {regime_phrase}. "
            f"The ${c.strike:.2f} put offers ${option.premium:.2f} premium ({ann_ret:.1f}% annualised), "
            "but the chart suggests increased downside risk. "
            "Consider waiting for a reversal or stronger support confirmation."
        )

    base = f"The ${c.strike:.2f} put is {distance:.1f}% below the current price"
    premium_note = f"collecting ${option.premium:.2f} ({ann_ret:.1f}% annualised)"
    breakeven_note = f"Net cost basis if assigned: ${option.break_even:.2f}"

    if label == RecommendationLabel.SAFEST_INCOME:
        sup_note = " The strike is near a support zone, giving a historically defended entry." if option.near_support else ""
        return (
            f"{base}, {premium_note}.{sup_note} "
            f"This is the safest entry because {regime_phrase} and assignment "
            f"at this level represents a meaningful discount. {breakeven_note}."
        )

    if label == RecommendationLabel.BEST_BALANCE:
        return (
            f"{base}, {premium_note}. "
            f"Best balance of premium and safety because {regime_phrase}. "
            f"The strike respects chart structure without sacrificing too much income. "
            f"{breakeven_note}."
        )

    if label == RecommendationLabel.MAX_PREMIUM:
        caution = ""
        if c.delta and abs(c.delta) > 0.35:
            caution = (
                f" Higher delta ({abs(c.delta):.2f}) means assignment is more likely — "
                "only suitable if you want to own the stock near this level."
            )
        return (
            f"{base}, {premium_note} — the highest available income.{caution} "
            f"Chart context: {regime_phrase}. {breakeven_note}."
        )

    return f"{base}, {premium_note}. {breakeven_note}."


# ---------------------------------------------------------------------------
# Label assignment
# ---------------------------------------------------------------------------

def _pick_recommendations(
    scored: list[ScoredOption],
    strategy: Strategy,
    regime: RegimeResult,
    indicators: TechnicalIndicators,
) -> list[Recommendation]:
    """
    Pick three recommendations with distinct labels from the scored list.

    - Best Balance  : highest composite score
    - Safest Income : lowest delta (most OTM) among top-10
    - Max Premium   : highest annualised return among top-10
    """
    if not scored:
        return []

    top10 = scored[:10]

    # Best Balance: top scorer
    best_balance = top10[0]

    # Safest Income: lowest absolute delta (or greatest distance_pct if delta unknown)
    def safest_key(o: ScoredOption) -> float:
        if o.contract.delta is not None:
            return abs(o.contract.delta)
        return -o.distance_pct  # proxy: further OTM = safer

    safest = min(top10, key=safest_key)

    # Max Premium: highest annualised return
    max_prem = max(top10, key=lambda o: o.annualized_return)

    explain_fn = _explain_covered_call if strategy == Strategy.COVERED_CALL else _explain_csp

    recs: list[Recommendation] = []
    seen_strikes: set[float] = set()

    for label, option in [
        (RecommendationLabel.BEST_BALANCE, best_balance),
        (RecommendationLabel.SAFEST_INCOME, safest),
        (RecommendationLabel.MAX_PREMIUM, max_prem),
    ]:
        if option.contract.strike in seen_strikes:
            # Avoid duplicate strikes — find next best candidate
            candidates = [o for o in top10 if o.contract.strike not in seen_strikes]
            if not candidates:
                continue
            option = candidates[0]
        seen_strikes.add(option.contract.strike)
        recs.append(
            Recommendation(
                label=label,
                option=option,
                explanation=explain_fn(option, regime, indicators, label),
            )
        )

    return recs


# ---------------------------------------------------------------------------
# DataFrame builder for the ranked table
# ---------------------------------------------------------------------------

def _to_dataframe(scored: list[ScoredOption], strategy: Strategy) -> pd.DataFrame:
    """Convert a list of ScoredOption into a display DataFrame."""
    rows = []
    for s in scored:
        c = s.contract
        row = {
            "Strike": c.strike,
            "Premium": round(s.premium, 2),
            "Ann. Return": f"{s.annualized_return * 100:.1f}%",
            "Delta": round(abs(c.delta), 3) if c.delta is not None else "—",
            "IV": f"{c.implied_volatility * 100:.1f}%" if c.implied_volatility else "—",
            "OI": c.open_interest,
            "Spread %": f"{c.bid_ask_spread_pct * 100:.1f}%",
            "Dist %": f"{s.distance_pct * 100:.1f}%",
            "Break-even": round(s.break_even, 2),
            "Score": round(s.score, 1),
            "Near S/R": ("S" if s.near_support else "") + ("R" if s.near_resistance else ""),
        }
        if strategy == Strategy.COVERED_CALL:
            row["Above Basis"] = (
                "✓" if s.above_cost_basis else ("✗" if s.above_cost_basis is False else "—")
            )
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Expected move
# ---------------------------------------------------------------------------

def _expected_move(price: float, iv: float | None, dte: int) -> float:
    """
    One standard-deviation expected move: price × IV × sqrt(dte/365).
    Returns 0 if IV is unavailable.
    """
    if not iv or dte <= 0:
        return 0.0
    return price * iv * math.sqrt(dte / 365)


def _avg_iv(scored: list[ScoredOption]) -> float | None:
    """Average IV across near-the-money strikes."""
    ivs = [
        o.contract.implied_volatility
        for o in scored
        if o.contract.implied_volatility
        and 0.01 <= o.distance_pct <= 0.10
    ]
    return sum(ivs) / len(ivs) if ivs else None


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

def run_screener(
    quote: Quote,
    expiration: str,
    dte: int,
    scored_options: list[ScoredOption],
    indicators: TechnicalIndicators,
    regime: RegimeResult,
    support_levels: list[SupportResistanceLevel],
    resistance_levels: list[SupportResistanceLevel],
    params: FilterParams,
    warnings: list[str] | None = None,
) -> ScreenerResult:
    """
    Assemble the final ScreenerResult from all computed components.

    Parameters
    ----------
    scored_options : already filtered and sorted ScoredOption list
    warnings       : any pre-computed warning strings (earnings, etc.)
    """
    recs = _pick_recommendations(scored_options, params.strategy, regime, indicators)
    df = _to_dataframe(scored_options, params.strategy)

    avg_iv = _avg_iv(scored_options)
    exp_move = _expected_move(quote.price, avg_iv, dte)

    # Regime-level warning
    w = list(warnings or [])
    if regime.trade_bias == "skip" and regime.skip_reason:
        w.append(regime.skip_reason)
    if regime.trade_bias == "caution":
        w.append(
            f"Chart regime is {regime.primary.value} — strike selection has been tightened. "
            "Review recommendations carefully."
        )

    return ScreenerResult(
        quote=quote,
        expiration=expiration,
        dte=dte,
        indicators=indicators,
        regime=regime,
        support_levels=support_levels,
        resistance_levels=resistance_levels,
        expected_move=exp_move,
        all_options=df,
        recommendations=recs,
        warnings=w,
    )
