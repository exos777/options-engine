"""Tests for indicators/support_resistance.py"""

import numpy as np
import pandas as pd
import pytest

from indicators.support_resistance import (
    find_support_resistance,
    is_near_level,
    nearest_resistance,
    nearest_support,
)
from strategies.models import SupportResistanceLevel


def _make_oscillating_df(low: float = 90.0, high: float = 110.0, n: int = 60) -> pd.DataFrame:
    """Create a DataFrame that oscillates between low and high."""
    prices = []
    for i in range(n):
        if i % 10 < 5:
            prices.append(high - (i % 5) * 0.5)
        else:
            prices.append(low + (i % 5) * 0.5)

    prices = np.array(prices, dtype=float)
    highs = prices + 1.0
    lows = prices - 1.0
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"Open": prices, "High": highs, "Low": lows, "Close": prices, "Volume": [1e6] * n},
        index=idx,
    )


class TestFindSupportResistance:
    def test_returns_two_lists(self, bullish_df):
        supports, resistances = find_support_resistance(bullish_df, 108.0)
        assert isinstance(supports, list)
        assert isinstance(resistances, list)

    def test_support_below_price(self, bullish_df):
        supports, _ = find_support_resistance(bullish_df, 108.0)
        for s in supports:
            assert s.price < 108.0

    def test_resistance_above_price(self, bullish_df):
        _, resistances = find_support_resistance(bullish_df, 95.0)
        for r in resistances:
            assert r.price > 95.0

    def test_strength_in_0_1(self, bullish_df):
        supports, resistances = find_support_resistance(bullish_df, 108.0)
        for level in supports + resistances:
            assert 0.0 <= level.strength <= 1.0

    def test_oscillating_produces_both_sides(self):
        df = _make_oscillating_df(low=90.0, high=110.0)
        supports, resistances = find_support_resistance(df, current_price=100.0)
        assert len(supports) > 0
        assert len(resistances) > 0

    def test_empty_df_returns_empty_lists(self):
        tiny = pd.DataFrame(
            {"Open": [100.0], "High": [101.0], "Low": [99.0], "Close": [100.0], "Volume": [1e6]}
        )
        supports, resistances = find_support_resistance(tiny, 100.0)
        assert supports == []
        assert resistances == []

    def test_top_n_respected(self, bullish_df):
        supports, resistances = find_support_resistance(bullish_df, 108.0, top_n=3)
        assert len(supports) <= 3
        assert len(resistances) <= 3


class TestNearestLevel:
    def test_nearest_support_returns_closest_below(self):
        levels = [
            SupportResistanceLevel(90.0, "support", 0.8, 4),
            SupportResistanceLevel(95.0, "support", 0.6, 3),
            SupportResistanceLevel(80.0, "support", 0.4, 2),
        ]
        result = nearest_support(levels, 100.0)
        assert result.price == 95.0

    def test_nearest_resistance_returns_closest_above(self):
        levels = [
            SupportResistanceLevel(110.0, "resistance", 0.8, 4),
            SupportResistanceLevel(115.0, "resistance", 0.6, 3),
            SupportResistanceLevel(120.0, "resistance", 0.4, 2),
        ]
        result = nearest_resistance(levels, 100.0)
        assert result.price == 110.0

    def test_nearest_support_none_when_all_above(self):
        levels = [SupportResistanceLevel(110.0, "support", 0.8, 4)]
        result = nearest_support(levels, 100.0)
        assert result is None

    def test_nearest_resistance_none_when_all_below(self):
        levels = [SupportResistanceLevel(90.0, "resistance", 0.8, 4)]
        result = nearest_resistance(levels, 100.0)
        assert result is None


class TestIsNearLevel:
    def test_within_tolerance(self):
        level = SupportResistanceLevel(100.0, "support", 0.8, 4)
        assert is_near_level(101.5, level, tolerance_pct=0.02) is True

    def test_outside_tolerance(self):
        level = SupportResistanceLevel(100.0, "support", 0.8, 4)
        assert is_near_level(105.0, level, tolerance_pct=0.02) is False

    def test_exact_match(self):
        level = SupportResistanceLevel(100.0, "support", 0.8, 4)
        assert is_near_level(100.0, level) is True
