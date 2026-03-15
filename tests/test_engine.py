"""Tests for scoring/engine.py"""

import pytest

from scoring.covered_call import score_covered_calls
from scoring.cash_secured_put import score_cash_secured_puts
from scoring.engine import run_screener, _pick_recommendations, _expected_move
from strategies.models import (
    RecommendationLabel,
    Strategy,
)


class TestExpectedMove:
    def test_zero_when_no_iv(self):
        assert _expected_move(100.0, None, 7) == 0.0

    def test_zero_when_dte_zero(self):
        assert _expected_move(100.0, 0.30, 0) == 0.0

    def test_positive_value(self):
        result = _expected_move(100.0, 0.30, 7)
        assert result > 0

    def test_higher_iv_larger_move(self):
        low_iv = _expected_move(100.0, 0.20, 7)
        high_iv = _expected_move(100.0, 0.40, 7)
        assert high_iv > low_iv


class TestPickRecommendations:
    def test_returns_up_to_three(
        self, sample_calls, bullish_indicators, bullish_regime,
        support_levels, resistance_levels, balanced_cc_params
    ):
        scored = score_covered_calls(
            sample_calls, 108.0, 7,
            bullish_indicators, bullish_regime,
            support_levels, resistance_levels,
            balanced_cc_params,
        )
        recs = _pick_recommendations(scored, Strategy.COVERED_CALL, bullish_regime, bullish_indicators)
        assert len(recs) <= 3

    def test_no_duplicate_strikes(
        self, sample_calls, bullish_indicators, bullish_regime,
        support_levels, resistance_levels, balanced_cc_params
    ):
        scored = score_covered_calls(
            sample_calls, 108.0, 7,
            bullish_indicators, bullish_regime,
            support_levels, resistance_levels,
            balanced_cc_params,
        )
        recs = _pick_recommendations(scored, Strategy.COVERED_CALL, bullish_regime, bullish_indicators)
        strikes = [r.option.contract.strike for r in recs]
        assert len(strikes) == len(set(strikes))

    def test_empty_scored_returns_empty(self, bullish_regime, bullish_indicators):
        recs = _pick_recommendations([], Strategy.COVERED_CALL, bullish_regime, bullish_indicators)
        assert recs == []

    def test_labels_are_valid(
        self, sample_calls, bullish_indicators, bullish_regime,
        support_levels, resistance_levels, balanced_cc_params
    ):
        scored = score_covered_calls(
            sample_calls, 108.0, 7,
            bullish_indicators, bullish_regime,
            support_levels, resistance_levels,
            balanced_cc_params,
        )
        recs = _pick_recommendations(scored, Strategy.COVERED_CALL, bullish_regime, bullish_indicators)
        valid_labels = set(RecommendationLabel)
        for r in recs:
            assert r.label in valid_labels

    def test_each_recommendation_has_explanation(
        self, sample_calls, bullish_indicators, bullish_regime,
        support_levels, resistance_levels, balanced_cc_params
    ):
        scored = score_covered_calls(
            sample_calls, 108.0, 7,
            bullish_indicators, bullish_regime,
            support_levels, resistance_levels,
            balanced_cc_params,
        )
        recs = _pick_recommendations(scored, Strategy.COVERED_CALL, bullish_regime, bullish_indicators)
        for r in recs:
            assert isinstance(r.explanation, str)
            assert len(r.explanation) > 10

    def test_skip_regime_explanation_mentions_conditions(
        self, sample_calls, bullish_indicators, bearish_regime,
        support_levels, resistance_levels, balanced_cc_params
    ):
        from scoring.covered_call import score_covered_calls as scc
        scored = scc(
            sample_calls, 108.0, 7,
            bullish_indicators, bearish_regime,
            support_levels, resistance_levels,
            balanced_cc_params,
        )
        recs = _pick_recommendations(scored, Strategy.COVERED_CALL, bearish_regime, bullish_indicators)
        if recs:
            # All explanations should mention chart/conditions in bearish regime
            for r in recs:
                text = r.explanation.lower()
                assert any(w in text for w in ["chart", "trend", "downtrend", "condition", "bearish"])


class TestRunScreener:
    def test_full_pipeline_covered_call(
        self, sample_quote, sample_calls, bullish_indicators, bullish_regime,
        support_levels, resistance_levels, balanced_cc_params
    ):
        scored = score_covered_calls(
            sample_calls, sample_quote.price, 7,
            bullish_indicators, bullish_regime,
            support_levels, resistance_levels,
            balanced_cc_params,
        )
        result = run_screener(
            quote=sample_quote,
            expiration="2024-12-27",
            dte=7,
            scored_options=scored,
            indicators=bullish_indicators,
            regime=bullish_regime,
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            params=balanced_cc_params,
        )
        assert result.quote.ticker == "TEST"
        assert result.dte == 7
        assert not result.all_options.empty
        assert len(result.recommendations) <= 3

    def test_full_pipeline_csp(
        self, sample_quote, sample_puts, bullish_indicators, bullish_regime,
        support_levels, resistance_levels, balanced_csp_params
    ):
        scored = score_cash_secured_puts(
            sample_puts, sample_quote.price, 7,
            bullish_indicators, bullish_regime,
            support_levels, resistance_levels,
            balanced_csp_params,
        )
        result = run_screener(
            quote=sample_quote,
            expiration="2024-12-27",
            dte=7,
            scored_options=scored,
            indicators=bullish_indicators,
            regime=bullish_regime,
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            params=balanced_csp_params,
        )
        assert not result.all_options.empty

    def test_warnings_propagated(
        self, sample_quote, bullish_indicators, bullish_regime,
        support_levels, resistance_levels, balanced_cc_params
    ):
        result = run_screener(
            quote=sample_quote,
            expiration="2024-12-27",
            dte=7,
            scored_options=[],
            indicators=bullish_indicators,
            regime=bullish_regime,
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            params=balanced_cc_params,
            warnings=["Test warning"],
        )
        assert "Test warning" in result.warnings

    def test_bearish_regime_adds_skip_warning(
        self, sample_quote, bullish_indicators, bearish_regime,
        support_levels, resistance_levels, balanced_cc_params
    ):
        result = run_screener(
            quote=sample_quote,
            expiration="2024-12-27",
            dte=7,
            scored_options=[],
            indicators=bullish_indicators,
            regime=bearish_regime,
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            params=balanced_cc_params,
        )
        assert any("downtrend" in w.lower() or "bearish" in w.lower() for w in result.warnings)

    def test_dataframe_has_expected_columns(
        self, sample_quote, sample_calls, bullish_indicators, bullish_regime,
        support_levels, resistance_levels, balanced_cc_params
    ):
        scored = score_covered_calls(
            sample_calls, sample_quote.price, 7,
            bullish_indicators, bullish_regime,
            support_levels, resistance_levels,
            balanced_cc_params,
        )
        result = run_screener(
            quote=sample_quote,
            expiration="2024-12-27",
            dte=7,
            scored_options=scored,
            indicators=bullish_indicators,
            regime=bullish_regime,
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            params=balanced_cc_params,
        )
        required_cols = {"Strike", "Premium", "Delta", "OI", "Score"}
        assert required_cols.issubset(set(result.all_options.columns))
