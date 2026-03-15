"""
Chart regime classifier.

Takes technical indicators and S/R levels and produces a RegimeResult that
drives how the scoring engine weights strikes.

Classification rules (applied in order, first match wins for primary):
  BEARISH        — price < SMA20 < SMA50  (confirmed downtrend)
  OVEREXTENDED   — price > SMA20 > SMA50 AND RSI >= 70
  NEAR_RESISTANCE— price within 2% of nearest resistance
  NEAR_SUPPORT   — price within 2% of nearest support
  BULLISH        — price > SMA20 AND price > SMA50
  NEUTRAL        — price between SMA20 and SMA50 (mixed signals)

Trade bias:
  "go"      — conditions are good for the strategy
  "caution" — conditions are marginal; tighten strike selection
  "skip"    — conditions argue against the strategy this week
"""

from __future__ import annotations

from typing import Optional

from strategies.models import (
    ChartRegime,
    RegimeResult,
    SupportResistanceLevel,
    TechnicalIndicators,
)
from indicators.support_resistance import is_near_level, nearest_resistance, nearest_support


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NEAR_LEVEL_PCT = 0.02   # within 2% of a S/R level counts as "near"
_RSI_OVERBOUGHT = 70
_RSI_OVERSOLD   = 30


def _describe(
    primary: ChartRegime,
    secondary: Optional[ChartRegime],
    ind: TechnicalIndicators,
) -> str:
    parts = [primary.value]
    if secondary:
        parts.append(secondary.value)
    parts.append(
        f"| Price ${ind.current_price:.2f}  "
        f"SMA20 ${ind.sma_20:.2f}  "
        f"SMA50 ${ind.sma_50:.2f}  "
        f"RSI {ind.rsi_14:.1f}  "
        f"ATR ${ind.atr_14:.2f}"
    )
    return "  ".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_regime(
    indicators: TechnicalIndicators,
    support_levels: list[SupportResistanceLevel],
    resistance_levels: list[SupportResistanceLevel],
) -> RegimeResult:
    """
    Classify the current chart regime from technical data.

    Returns a RegimeResult with primary classification, optional secondary,
    a human-readable description, and a trade_bias flag.
    """
    price = indicators.current_price
    sma20 = indicators.sma_20
    sma50 = indicators.sma_50
    rsi = indicators.rsi_14

    nr = nearest_resistance(resistance_levels, price)
    ns = nearest_support(support_levels, price)

    near_res = nr is not None and is_near_level(price, nr, _NEAR_LEVEL_PCT)
    near_sup = ns is not None and is_near_level(price, ns, _NEAR_LEVEL_PCT)

    # --- Primary classification ---
    if price < sma20 and sma20 < sma50:
        primary = ChartRegime.BEARISH
    elif price > sma20 and sma20 > sma50 and rsi >= _RSI_OVERBOUGHT:
        primary = ChartRegime.OVEREXTENDED
    elif near_res and price > sma20:
        primary = ChartRegime.NEAR_RESISTANCE
    elif near_sup and price > sma50:
        primary = ChartRegime.NEAR_SUPPORT
    elif price > sma20 and price > sma50:
        primary = ChartRegime.BULLISH
    else:
        primary = ChartRegime.NEUTRAL

    # --- Secondary classification (overlay) ---
    secondary: Optional[ChartRegime] = None
    if primary != ChartRegime.NEAR_RESISTANCE and near_res:
        secondary = ChartRegime.NEAR_RESISTANCE
    elif primary != ChartRegime.NEAR_SUPPORT and near_sup:
        secondary = ChartRegime.NEAR_SUPPORT
    elif primary not in (ChartRegime.BEARISH, ChartRegime.OVEREXTENDED):
        if rsi >= _RSI_OVERBOUGHT:
            secondary = ChartRegime.OVEREXTENDED
        elif rsi <= _RSI_OVERSOLD:
            secondary = ChartRegime.NEAR_SUPPORT  # oversold near potential turn

    # --- Trade bias ---
    skip_reason: Optional[str] = None

    if primary == ChartRegime.BEARISH:
        trade_bias = "skip"
        skip_reason = (
            "The chart is in a confirmed downtrend (price below SMA20 and SMA50). "
            "Covered calls carry assignment risk on a falling stock; "
            "cash-secured puts face a falling knife. Consider waiting for stabilisation."
        )
    elif primary == ChartRegime.OVEREXTENDED:
        trade_bias = "caution"
    elif primary == ChartRegime.NEAR_RESISTANCE:
        trade_bias = "caution"
    elif primary in (ChartRegime.BULLISH, ChartRegime.NEAR_SUPPORT):
        trade_bias = "go"
    else:  # NEUTRAL
        trade_bias = "caution"

    return RegimeResult(
        primary=primary,
        secondary=secondary,
        description=_describe(primary, secondary, indicators),
        trade_bias=trade_bias,
        skip_reason=skip_reason,
    )


# ---------------------------------------------------------------------------
# Chart multipliers consumed by scoring modules
# ---------------------------------------------------------------------------

def chart_score_multiplier(regime: RegimeResult, strategy: str) -> float:
    """
    Return a multiplier in [0.0, 1.0] that scales the chart alignment
    component of the scoring engine.

    strategy: "covered_call" or "cash_secured_put"
    """
    primary = regime.primary

    # Covered calls
    if strategy == "covered_call":
        return {
            ChartRegime.BULLISH: 1.0,
            ChartRegime.NEAR_SUPPORT: 0.9,
            ChartRegime.NEUTRAL: 0.75,
            ChartRegime.NEAR_RESISTANCE: 0.85,  # closer strikes make sense here
            ChartRegime.OVEREXTENDED: 0.6,       # be careful; potential pullback
            ChartRegime.BEARISH: 0.2,
        }.get(primary, 0.75)

    # Cash secured puts
    if strategy == "cash_secured_put":
        return {
            ChartRegime.BULLISH: 0.9,
            ChartRegime.NEAR_SUPPORT: 1.0,
            ChartRegime.NEUTRAL: 0.75,
            ChartRegime.NEAR_RESISTANCE: 0.7,
            ChartRegime.OVEREXTENDED: 0.5,
            ChartRegime.BEARISH: 0.2,
        }.get(primary, 0.75)

    return 0.75
