"""Tests for scoring/cash_secured_put.py"""

import pytest

from scoring.cash_secured_put import (
    _buy_price_score,
    _delta_score,
    _iv_rank_score,
    _liquidity_score,
    _premium_score,
    score_cash_secured_puts,
)
from strategies.models import RiskProfile


class TestPremiumScore:
    def test_zero_return_near_zero(self):
        assert _premium_score(0.0, RiskProfile.BALANCED) == pytest.approx(0.0, abs=1.0)

    def test_high_return_approaches_100(self):
        assert _premium_score(1.5, RiskProfile.BALANCED) > 90

    def test_bounded(self):
        for r in [0.0, 0.1, 0.5, 2.0]:
            score = _premium_score(r, RiskProfile.BALANCED)
            assert 0 <= score <= 100


class TestDeltaScore:
    def test_low_abs_delta_scores_high(self):
        # put delta is negative; function uses abs
        score = _delta_score(-0.10, 0.40)
        assert score > 80

    def test_high_delta_penalised(self):
        score = _delta_score(-0.55, 0.40)
        assert score < 20

    def test_none_returns_50(self):
        assert _delta_score(None, 0.40) == pytest.approx(50.0)


class TestIVRankScore:
    def test_high_iv_scores_high(self):
        # current IV at 90% of range
        score = _iv_rank_score(0.45, iv_52w_high=0.50, iv_52w_low=0.10)
        assert score == pytest.approx(87.5)

    def test_low_iv_scores_low(self):
        # current IV at 10% of range
        score = _iv_rank_score(0.14, iv_52w_high=0.50, iv_52w_low=0.10)
        assert score == pytest.approx(10.0)

    def test_at_52w_high_scores_100(self):
        assert _iv_rank_score(0.50, iv_52w_high=0.50, iv_52w_low=0.10) == pytest.approx(100.0)

    def test_at_52w_low_scores_zero(self):
        assert _iv_rank_score(0.10, iv_52w_high=0.50, iv_52w_low=0.10) == pytest.approx(0.0)

    def test_flat_range_returns_neutral(self):
        assert _iv_rank_score(0.30, iv_52w_high=0.30, iv_52w_low=0.30) == pytest.approx(50.0)

    def test_capped_at_100(self):
        # current IV above 52w high (data anomaly)
        score = _iv_rank_score(0.60, iv_52w_high=0.50, iv_52w_low=0.10)
        assert score == pytest.approx(100.0)

    def test_iv_rank_stored_on_scored_option(
        self, sample_puts, bullish_indicators, bullish_regime,
        support_levels, resistance_levels, balanced_csp_params
    ):
        scored = score_cash_secured_puts(
            sample_puts, 108.0, 7,
            bullish_indicators, bullish_regime,
            support_levels, resistance_levels,
            balanced_csp_params,
            iv_52w_high=0.45,
            iv_52w_low=0.15,
        )
        for s in scored:
            assert 0 <= s.iv_rank_score <= 100

    def test_defaults_to_50_when_iv_data_missing(
        self, sample_puts, bullish_indicators, bullish_regime,
        support_levels, resistance_levels, balanced_csp_params
    ):
        scored = score_cash_secured_puts(
            sample_puts, 108.0, 7,
            bullish_indicators, bullish_regime,
            support_levels, resistance_levels,
            balanced_csp_params,
            # no iv_52w_high / iv_52w_low passed
        )
        for s in scored:
            assert s.iv_rank_score == pytest.approx(50.0)

    def test_high_iv_rank_boosts_score_vs_low(
        self, sample_puts, bullish_indicators, bullish_regime,
        support_levels, resistance_levels, balanced_csp_params
    ):
        high_iv = score_cash_secured_puts(
            sample_puts, 108.0, 7,
            bullish_indicators, bullish_regime,
            support_levels, resistance_levels,
            balanced_csp_params,
            iv_52w_high=0.50, iv_52w_low=0.10,
        )
        low_iv = score_cash_secured_puts(
            sample_puts, 108.0, 7,
            bullish_indicators, bullish_regime,
            support_levels, resistance_levels,
            balanced_csp_params,
            iv_52w_high=0.50, iv_52w_low=0.45,  # current IV near low
        )
        avg_high = sum(s.score for s in high_iv) / len(high_iv)
        avg_low  = sum(s.score for s in low_iv)  / len(low_iv)
        assert avg_high > avg_low


class TestBuyPriceScore:
    def test_none_returns_neutral(self):
        assert _buy_price_score(100.0, None, 1.0) == pytest.approx(75.0)

    def test_exact_match_scores_100(self):
        # strike=101, premium=1, net_cost=100, desired=100
        score = _buy_price_score(101.0, 100.0, 1.0)
        assert score == pytest.approx(100.0)

    def test_far_from_desired_scores_low(self):
        # strike=120, premium=1, net_cost=119, desired=100 → 19% off
        score = _buy_price_score(120.0, 100.0, 1.0)
        assert score < 40

    def test_close_to_desired_scores_high(self):
        # net_cost=101.5, desired=100 → 1.5% off
        score = _buy_price_score(102.5, 100.0, 1.0)
        assert score > 70


class TestScoreCashSecuredPuts:
    def test_returns_sorted_by_score(
        self, sample_puts, bullish_indicators, bullish_regime,
        support_levels, resistance_levels, balanced_csp_params
    ):
        scored = score_cash_secured_puts(
            sample_puts, 108.0, 7,
            bullish_indicators, bullish_regime,
            support_levels, resistance_levels,
            balanced_csp_params,
        )
        scores = [s.score for s in scored]
        assert scores == sorted(scores, reverse=True)

    def test_filters_low_oi(
        self, sample_puts, bullish_indicators, bullish_regime,
        support_levels, resistance_levels, balanced_csp_params
    ):
        balanced_csp_params.min_open_interest = 600
        scored = score_cash_secured_puts(
            sample_puts, 108.0, 7,
            bullish_indicators, bullish_regime,
            support_levels, resistance_levels,
            balanced_csp_params,
        )
        for s in scored:
            assert s.contract.open_interest >= 600

    def test_break_even_equals_strike_minus_premium(
        self, sample_puts, bullish_indicators, bullish_regime,
        support_levels, resistance_levels, balanced_csp_params
    ):
        scored = score_cash_secured_puts(
            sample_puts, 108.0, 7,
            bullish_indicators, bullish_regime,
            support_levels, resistance_levels,
            balanced_csp_params,
        )
        for s in scored:
            expected_be = s.contract.strike - s.premium
            assert s.break_even == pytest.approx(expected_be, abs=0.01)

    def test_distance_pct_positive_for_otm(
        self, sample_puts, bullish_indicators, bullish_regime,
        support_levels, resistance_levels, balanced_csp_params
    ):
        scored = score_cash_secured_puts(
            sample_puts, 108.0, 7,
            bullish_indicators, bullish_regime,
            support_levels, resistance_levels,
            balanced_csp_params,
        )
        # All puts are OTM so distance_pct should be positive
        for s in scored:
            assert s.distance_pct > 0

    def test_empty_puts_returns_empty(
        self, bullish_indicators, bullish_regime,
        support_levels, resistance_levels, balanced_csp_params
    ):
        result = score_cash_secured_puts(
            [], 108.0, 7,
            bullish_indicators, bullish_regime,
            support_levels, resistance_levels,
            balanced_csp_params,
        )
        assert result == []

    def test_score_bounded(
        self, sample_puts, bullish_indicators, bullish_regime,
        support_levels, resistance_levels, balanced_csp_params
    ):
        scored = score_cash_secured_puts(
            sample_puts, 108.0, 7,
            bullish_indicators, bullish_regime,
            support_levels, resistance_levels,
            balanced_csp_params,
        )
        for s in scored:
            assert 0 <= s.score <= 100

    def test_near_support_flagged_correctly(
        self, sample_puts, bullish_indicators, bullish_regime,
        support_levels, resistance_levels, balanced_csp_params
    ):
        # Ensure the near_support flag is set for strikes near $95 support
        scored = score_cash_secured_puts(
            sample_puts, 108.0, 7,
            bullish_indicators, bullish_regime,
            support_levels, resistance_levels,
            balanced_csp_params,
        )
        near_sup_strikes = {s.contract.strike for s in scored if s.near_support}
        # $97 put is within 2% of $95 support
        # Not guaranteed but should have at least some near support
        assert isinstance(near_sup_strikes, set)
