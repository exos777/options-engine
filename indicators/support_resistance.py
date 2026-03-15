"""
Support and resistance detection using price pivot highs / lows.

Algorithm:
1. Walk the Close series and find local pivots (swing highs / swing lows)
   using a configurable left/right bar window.
2. Cluster nearby pivots within a tolerance band (% of price).
3. Score each cluster by the number of touches.
4. Return the top N support and resistance levels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.models import SupportResistanceLevel


# ---------------------------------------------------------------------------
# Pivot detection
# ---------------------------------------------------------------------------

def _find_pivots(
    high: pd.Series,
    low: pd.Series,
    left: int = 5,
    right: int = 5,
) -> tuple[list[float], list[float]]:
    """
    Detect swing highs and swing lows.

    A pivot high at index i means high[i] is the highest value in
    [i-left, i+right]. Similarly for pivot lows.

    Returns (pivot_highs, pivot_lows) as lists of price levels.
    """
    highs = high.values
    lows = low.values
    n = len(highs)

    pivot_highs: list[float] = []
    pivot_lows: list[float] = []

    for i in range(left, n - right):
        window_high = highs[i - left : i + right + 1]
        window_low = lows[i - left : i + right + 1]

        if highs[i] == np.max(window_high):
            pivot_highs.append(float(highs[i]))
        if lows[i] == np.min(window_low):
            pivot_lows.append(float(lows[i]))

    return pivot_highs, pivot_lows


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def _cluster_levels(
    prices: list[float],
    tolerance_pct: float = 0.015,
) -> list[tuple[float, int]]:
    """
    Group nearby price levels within *tolerance_pct* of each other.

    Returns a list of (representative_price, touch_count) sorted by
    touch_count descending.
    """
    if not prices:
        return []

    sorted_prices = sorted(prices)
    clusters: list[list[float]] = []

    current_cluster: list[float] = [sorted_prices[0]]
    for p in sorted_prices[1:]:
        ref = np.mean(current_cluster)
        if abs(p - ref) / ref <= tolerance_pct:
            current_cluster.append(p)
        else:
            clusters.append(current_cluster)
            current_cluster = [p]
    clusters.append(current_cluster)

    result = [(float(np.mean(c)), len(c)) for c in clusters]
    result.sort(key=lambda x: x[1], reverse=True)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_support_resistance(
    df: pd.DataFrame,
    current_price: float,
    left: int = 5,
    right: int = 5,
    tolerance_pct: float = 0.015,
    top_n: int = 5,
) -> tuple[list[SupportResistanceLevel], list[SupportResistanceLevel]]:
    """
    Detect support and resistance levels from an OHLCV DataFrame.

    Parameters
    ----------
    df : DataFrame with columns High, Low, Close
    current_price : float — current stock price (used for classifying S vs R)
    left, right : pivot detection window bars
    tolerance_pct : cluster grouping tolerance
    top_n : maximum number of levels to return per side

    Returns
    -------
    (support_levels, resistance_levels) sorted by strength descending
    """
    if len(df) < left + right + 1:
        return [], []

    pivot_highs, pivot_lows = _find_pivots(df["High"], df["Low"], left, right)

    # All pivots are candidates for both support and resistance based on
    # their position relative to the current price.
    all_pivots = pivot_highs + pivot_lows
    clustered = _cluster_levels(all_pivots, tolerance_pct)

    if not clustered:
        return [], []

    max_touches = max(c[1] for c in clustered)

    support_levels: list[SupportResistanceLevel] = []
    resistance_levels: list[SupportResistanceLevel] = []

    for price_level, touches in clustered:
        strength = touches / max_touches  # normalised 0–1
        if price_level < current_price:
            support_levels.append(
                SupportResistanceLevel(
                    price=price_level,
                    level_type="support",
                    strength=strength,
                    touches=touches,
                )
            )
        else:
            resistance_levels.append(
                SupportResistanceLevel(
                    price=price_level,
                    level_type="resistance",
                    strength=strength,
                    touches=touches,
                )
            )

    # Sort: support descending by price (closest first), resistance ascending
    support_levels.sort(key=lambda x: x.price, reverse=True)
    resistance_levels.sort(key=lambda x: x.price)

    return support_levels[:top_n], resistance_levels[:top_n]


def nearest_support(
    support_levels: list[SupportResistanceLevel],
    current_price: float,
) -> SupportResistanceLevel | None:
    """Return the nearest support level below the current price, or None."""
    below = [s for s in support_levels if s.price < current_price]
    if not below:
        return None
    return max(below, key=lambda x: x.price)


def nearest_resistance(
    resistance_levels: list[SupportResistanceLevel],
    current_price: float,
) -> SupportResistanceLevel | None:
    """Return the nearest resistance level above the current price, or None."""
    above = [r for r in resistance_levels if r.price > current_price]
    if not above:
        return None
    return min(above, key=lambda x: x.price)


def is_near_level(
    price: float,
    level: SupportResistanceLevel,
    tolerance_pct: float = 0.02,
) -> bool:
    """Return True if *price* is within *tolerance_pct* of *level.price*."""
    return abs(price - level.price) / level.price <= tolerance_pct
