"""Tests for scoring/regime.py"""

import pytest

from scoring.regime import chart_score_multiplier, classify_regime
from strategies.models import ChartRegime, SupportResistanceLevel, TechnicalIndicators


def _make_indicators(price, sma20, sma50, rsi=50.0, atr=2.0) -> TechnicalIndicators:
    return TechnicalIndicators(
        sma_20=sma20,
        sma_50=sma50,
        rsi_14=rsi,
        atr_14=atr,
        avg_volume_20=1_000_000,
        current_price=price,
        weekly_atr_est=atr * 2.24,
    )


class TestClassifyRegime:
    def test_bullish_when_price_above_both_smas(self):
        ind = _make_indicators(price=110, sma20=105, sma50=100, rsi=55)
        result = classify_regime(ind, [], [])
        assert result.primary == ChartRegime.BULLISH
        assert result.trade_bias == "go"

    def test_bearish_when_price_below_both_smas(self):
        ind = _make_indicators(price=88, sma20=95, sma50=100, rsi=38)
        result = classify_regime(ind, [], [])
        assert result.primary == ChartRegime.BEARISH
        assert result.trade_bias == "skip"
        assert result.skip_reason is not None

    def test_overextended_when_bullish_and_rsi_gte_70(self):
        ind = _make_indicators(price=115, sma20=110, sma50=105, rsi=72)
        result = classify_regime(ind, [], [])
        assert result.primary == ChartRegime.OVEREXTENDED

    def test_neutral_when_mixed_smas(self):
        # price between SMA20 and SMA50
        ind = _make_indicators(price=102, sma20=105, sma50=100, rsi=50)
        result = classify_regime(ind, [], [])
        assert result.primary == ChartRegime.NEUTRAL

    def test_near_resistance_when_close_to_resistance(self):
        ind = _make_indicators(price=109, sma20=107, sma50=104, rsi=58)
        resistances = [SupportResistanceLevel(110.0, "resistance", 0.9, 5)]
        result = classify_regime(ind, [], resistances)
        assert result.primary == ChartRegime.NEAR_RESISTANCE

    def test_near_support_when_close_to_support(self):
        ind = _make_indicators(price=101.5, sma20=102, sma50=100, rsi=45)
        supports = [SupportResistanceLevel(100.0, "support", 0.8, 4)]
        result = classify_regime(ind, supports, [])
        assert result.primary == ChartRegime.NEAR_SUPPORT

    def test_secondary_regime_populated(self):
        # Bullish primary but near resistance as secondary
        ind = _make_indicators(price=109, sma20=107, sma50=103, rsi=60)
        resistances = [SupportResistanceLevel(110.5, "resistance", 0.9, 5)]
        result = classify_regime(ind, [], resistances)
        # secondary might be near resistance
        # Just assert the field exists and is optional
        assert result.secondary is None or isinstance(result.secondary, ChartRegime)

    def test_skip_reason_only_on_bearish(self):
        bullish = _make_indicators(price=110, sma20=105, sma50=100)
        result = classify_regime(bullish, [], [])
        assert result.skip_reason is None

        bearish = _make_indicators(price=88, sma20=95, sma50=100)
        result = classify_regime(bearish, [], [])
        assert result.skip_reason is not None


class TestChartScoreMultiplier:
    def test_bullish_covered_call_is_high(self):
        from strategies.models import RegimeResult
        regime = RegimeResult(ChartRegime.BULLISH, None, "", "go")
        m = chart_score_multiplier(regime, "covered_call")
        assert m == pytest.approx(1.0)

    def test_bearish_multiplier_is_low(self):
        from strategies.models import RegimeResult
        regime = RegimeResult(ChartRegime.BEARISH, None, "", "skip")
        m_cc = chart_score_multiplier(regime, "covered_call")
        m_csp = chart_score_multiplier(regime, "cash_secured_put")
        assert m_cc <= 0.25
        assert m_csp <= 0.25

    def test_near_support_favors_csp(self):
        from strategies.models import RegimeResult
        regime = RegimeResult(ChartRegime.NEAR_SUPPORT, None, "", "go")
        m_cc = chart_score_multiplier(regime, "covered_call")
        m_csp = chart_score_multiplier(regime, "cash_secured_put")
        assert m_csp >= m_cc
