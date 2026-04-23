"""
Recommendation engine.

Orchestrates the full pipeline:
  1. Receives scored options (already filtered)
  2. Selects the top 3 with distinct labels: Aggressive, Balanced, Conservative
  3. Generates plain-English explanations with position sizing
  4. Builds the ScreenerResult for the UI layer

Leg assignment:
  - Leg 1 Aggressive   → highest annualised return from top-10
  - Leg 2 Balanced     → highest composite score overall
  - Leg 3 Conservative → lowest delta / widest OTM from top-10
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


def _vega_label(vega: float | None) -> str:
    if vega is None:
        return "Unknown"
    av = abs(vega)
    if av <= 0.15:
        return "Low"
    if av <= 0.30:
        return "Medium"
    return "High ⚠️"


def _em_context(
    strike: float,
    current_price: float,
    expected_move: float,
    is_call: bool,
) -> str:
    if expected_move <= 0:
        return ""
    if is_call:
        boundary = current_price + expected_move
        outside = strike >= boundary
        label = "Upper EM boundary"
    else:
        boundary = current_price - expected_move
        outside = strike <= boundary
        label = "Lower EM boundary"
    status = "✅ outside" if outside else "⚠️ inside"
    return f"Strike is {status} the expected move ({label}: ${boundary:.2f})."


def _explain_covered_call(
    option: ScoredOption,
    regime: RegimeResult,
    indicators: TechnicalIndicators,
    label: RecommendationLabel,
    expected_move: float = 0.0,
    current_price: float = 0.0,
) -> str:
    c = option.contract
    distance = option.distance_pct * 100
    ann_ret = option.annualized_return * 100
    regime_phrase = _regime_phrase(regime)
    theta_note = f"  Theta: ${abs(c.theta):.3f}/day." if c.theta else ""
    vega_note = f"  Vega risk: {_vega_label(c.vega)}." if c.vega else ""
    em_note = _em_context(c.strike, current_price, expected_move, is_call=True)
    em_part = f"  {em_note}" if em_note else ""

    if regime.trade_bias == "skip":
        return (
            f"No strong trade recommended — {regime_phrase}. "
            f"The ${c.strike:.2f} call offers {ann_ret:.1f}% annualised, "
            "but chart conditions do not support selling calls at this time. "
            "Consider waiting for a clearer setup."
        )

    base = f"The ${c.strike:.2f} call is {distance:.1f}% above the current price"
    income = f"earning ${option.premium:.2f} premium ({ann_ret:.1f}% annualised)"
    be = f"Break-even: ${option.break_even:.2f}."

    if label == RecommendationLabel.AGGRESSIVE:
        caution = ""
        if c.delta and abs(c.delta) > 0.35:
            caution = f"  Note: delta {abs(c.delta):.2f} — higher assignment risk."
        return (
            f"{base}, {income} — highest available yield.{caution}{theta_note}{vega_note}{em_part}  "
            f"{be}  Chart context: {regime_phrase}."
        )

    if label == RecommendationLabel.BALANCED:
        chart_note = "  Strike near resistance — assignment at a reasonable level." if option.near_resistance else ""
        return (
            f"{base}, {income}.{theta_note}{vega_note}{em_part}{chart_note}  "
            f"{be}  Best balance of premium, theta, and safety because {regime_phrase}."
        )

    if label == RecommendationLabel.CONSERVATIVE:
        delta_note = f"  Delta: {abs(c.delta):.2f}." if c.delta else ""
        return (
            f"{base}, {income}.{delta_note}{theta_note}{vega_note}{em_part}  "
            f"{be}  Conservative choice: {regime_phrase}, plenty of room before assignment."
        )

    return f"{base}, {income}.  {be}"


def _explain_csp(
    option: ScoredOption,
    regime: RegimeResult,
    indicators: TechnicalIndicators,
    label: RecommendationLabel,
    expected_move: float = 0.0,
    current_price: float = 0.0,
) -> str:
    c = option.contract
    distance = option.distance_pct * 100
    ann_ret = option.annualized_return * 100
    regime_phrase = _regime_phrase(regime)
    theta_note = f"  Theta: ${abs(c.theta):.3f}/day." if c.theta else ""
    vega_note = f"  Vega risk: {_vega_label(c.vega)}." if c.vega else ""
    em_note = _em_context(c.strike, current_price, expected_move, is_call=False)
    em_part = f"  {em_note}" if em_note else ""
    be = f"Net cost basis if assigned: ${option.break_even:.2f}."

    if regime.trade_bias == "skip":
        return (
            f"No strong trade recommended — {regime_phrase}. "
            f"The ${c.strike:.2f} put offers ${option.premium:.2f} ({ann_ret:.1f}% annualised), "
            "but the chart suggests elevated downside risk. "
            "Consider waiting for a reversal or stronger support."
        )

    base = f"The ${c.strike:.2f} put is {distance:.1f}% below the current price"
    income = f"collecting ${option.premium:.2f} ({ann_ret:.1f}% annualised)"

    if label == RecommendationLabel.AGGRESSIVE:
        caution = ""
        if c.delta and abs(c.delta) > 0.35:
            caution = f"  Delta {abs(c.delta):.2f} — assignment more likely, only suitable if willing to own."
        return (
            f"{base}, {income} — highest available income.{caution}{theta_note}{vega_note}{em_part}  "
            f"{be}  Chart context: {regime_phrase}."
        )

    if label == RecommendationLabel.BALANCED:
        return (
            f"{base}, {income}.{theta_note}{vega_note}{em_part}  "
            f"{be}  Best balance of premium, theta, and safety because {regime_phrase}."
        )

    if label == RecommendationLabel.CONSERVATIVE:
        sup_note = "  Strike near support — historically defended entry." if option.near_support else ""
        return (
            f"{base}, {income}.{theta_note}{vega_note}{em_part}{sup_note}  "
            f"{be}  Conservative choice: {regime_phrase}, assignment at a meaningful discount."
        )

    return f"{base}, {income}.  {be}"


# ---------------------------------------------------------------------------
# Label assignment
# ---------------------------------------------------------------------------

def _position_size(score: float) -> tuple[str, str]:
    """Return (size_label, reason) based on composite score."""
    if score >= 75:
        return "\u2705 Full Size", "Strong setup \u2014 all signals aligned"
    if score >= 60:
        return "\u26a1 Half Size", "Good setup \u2014 some signals mixed"
    if score >= 45:
        return "\u26a0\ufe0f Quarter Size", "Weak setup \u2014 proceed with caution"
    return "\u26d4 No Trade", "Score too low \u2014 skip this week"


def _pick_recommendations(
    scored: list[ScoredOption],
    strategy: Strategy,
    regime: RegimeResult,
    indicators: TechnicalIndicators,
    expected_move: float = 0.0,
    current_price: float = 0.0,
) -> list[Recommendation]:
    """
    Pick three wheel-strategy recommendations with distinct labels.

    - Leg 1 Aggressive  : highest annualised yield from top-10
    - Leg 2 Balanced    : highest composite score (best risk-adjusted income)
    - Leg 3 Conservative: lowest assignment risk (lowest delta) from top-10
    """
    if not scored:
        return []

    top10 = scored[:10]

    # Leg 2 — Balanced: highest composite score
    balanced = top10[0]

    # Leg 3 — Conservative: lowest absolute delta
    def safest_key(o: ScoredOption) -> float:
        if o.contract.delta is not None:
            return abs(o.contract.delta)
        return -o.distance_pct

    conservative = min(top10, key=safest_key)

    # Leg 1 — Aggressive: highest annualized income
    aggressive = max(top10, key=lambda o: o.annualized_return)

    explain_fn = _explain_covered_call if strategy == Strategy.COVERED_CALL else _explain_csp

    recs: list[Recommendation] = []
    seen_strikes: set[float] = set()

    for label, option in [
        (RecommendationLabel.AGGRESSIVE, aggressive),
        (RecommendationLabel.BALANCED, balanced),
        (RecommendationLabel.CONSERVATIVE, conservative),
    ]:
        if option.contract.strike in seen_strikes:
            candidates = [o for o in top10 if o.contract.strike not in seen_strikes]
            if not candidates:
                continue
            option = candidates[0]
        seen_strikes.add(option.contract.strike)

        size, reason = _position_size(option.score)
        recs.append(
            Recommendation(
                label=label,
                option=option,
                explanation=explain_fn(
                    option, regime, indicators, label,
                    expected_move=expected_move,
                    current_price=current_price,
                ),
                position_size=size,
                position_size_reason=reason,
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
            "Theta": round(c.theta, 4) if c.theta is not None else "—",
            "Vega": round(c.vega, 4) if c.vega is not None else "—",
            "IV": f"{c.implied_volatility * 100:.1f}%" if c.implied_volatility else "—",
            "OI": c.open_interest,
            "Spread %": f"{c.bid_ask_spread_pct * 100:.1f}%",
            "Dist %": f"{s.distance_pct * 100:.1f}%",
            "Break-even": round(s.break_even, 2),
            "Score": round(s.score, 1),
            "Near S/R": ("S" if s.near_support else "") + ("R" if s.near_resistance else ""),
            "Earnings": "⚠️" if s.earnings_in_window else "",
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
    avg_iv = _avg_iv(scored_options)
    exp_move = _expected_move(quote.price, avg_iv, dte)

    recs = _pick_recommendations(
        scored_options, params.strategy, regime, indicators,
        expected_move=exp_move,
        current_price=quote.price,
    )
    df = _to_dataframe(scored_options, params.strategy)

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
