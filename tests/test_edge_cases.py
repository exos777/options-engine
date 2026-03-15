"""
Edge case tests: missing Greeks, empty chain, illiquid strikes, invalid tickers.
"""

import pytest

from scoring.covered_call import score_covered_calls
from scoring.cash_secured_put import score_cash_secured_puts
from strategies.models import (
    FilterParams,
    OptionContract,
    RiskProfile,
    Strategy,
)


def _minimal_call(
    strike: float,
    bid: float = 1.0,
    ask: float = 1.20,
    oi: int = 200,
    delta=None,
    iv=None,
) -> OptionContract:
    return OptionContract(
        strike=strike,
        expiration="2024-12-27",
        option_type="call",
        bid=bid,
        ask=ask,
        last=(bid + ask) / 2,
        volume=20,
        open_interest=oi,
        implied_volatility=iv,
        delta=delta,
    )


def _minimal_put(
    strike: float,
    bid: float = 1.0,
    ask: float = 1.20,
    oi: int = 200,
    delta=None,
    iv=None,
) -> OptionContract:
    return OptionContract(
        strike=strike,
        expiration="2024-12-27",
        option_type="put",
        bid=bid,
        ask=ask,
        last=(bid + ask) / 2,
        volume=20,
        open_interest=oi,
        implied_volatility=iv,
        delta=delta,
    )


@pytest.fixture
def params_no_basis() -> FilterParams:
    return FilterParams(
        strategy=Strategy.COVERED_CALL,
        risk_profile=RiskProfile.BALANCED,
        max_delta=0.50,
        min_open_interest=10,
        max_spread_pct=0.50,
        min_premium=0.05,
        cost_basis=None,
    )


@pytest.fixture
def csp_params() -> FilterParams:
    return FilterParams(
        strategy=Strategy.CASH_SECURED_PUT,
        risk_profile=RiskProfile.BALANCED,
        max_delta=0.50,
        min_open_interest=10,
        max_spread_pct=0.50,
        min_premium=0.05,
    )


class TestMissingGreeks:
    def test_call_without_delta_still_scores(
        self, bullish_indicators, bullish_regime,
        support_levels, resistance_levels, params_no_basis
    ):
        calls = [_minimal_call(110.0, delta=None, iv=None)]
        scored = score_covered_calls(
            calls, 108.0, 7,
            bullish_indicators, bullish_regime,
            support_levels, resistance_levels,
            params_no_basis,
        )
        assert len(scored) == 1
        assert 0 <= scored[0].score <= 100

    def test_put_without_delta_still_scores(
        self, bullish_indicators, bullish_regime,
        support_levels, resistance_levels, csp_params
    ):
        puts = [_minimal_put(105.0, delta=None, iv=None)]
        scored = score_cash_secured_puts(
            puts, 108.0, 7,
            bullish_indicators, bullish_regime,
            support_levels, resistance_levels,
            csp_params,
        )
        assert len(scored) == 1
        assert 0 <= scored[0].score <= 100

    def test_missing_iv_shows_dash_in_output(
        self, sample_quote, bullish_indicators, bullish_regime,
        support_levels, resistance_levels, params_no_basis
    ):
        from scoring.engine import run_screener
        calls = [_minimal_call(110.0, iv=None)]
        scored = score_covered_calls(
            calls, 108.0, 7,
            bullish_indicators, bullish_regime,
            support_levels, resistance_levels,
            params_no_basis,
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
            params=params_no_basis,
        )
        assert "—" in result.all_options["IV"].values


class TestEmptyChain:
    def test_empty_call_chain(
        self, bullish_indicators, bullish_regime,
        support_levels, resistance_levels, params_no_basis
    ):
        result = score_covered_calls(
            [], 108.0, 7,
            bullish_indicators, bullish_regime,
            support_levels, resistance_levels,
            params_no_basis,
        )
        assert result == []

    def test_empty_put_chain(
        self, bullish_indicators, bullish_regime,
        support_levels, resistance_levels, csp_params
    ):
        result = score_cash_secured_puts(
            [], 108.0, 7,
            bullish_indicators, bullish_regime,
            support_levels, resistance_levels,
            csp_params,
        )
        assert result == []


class TestIlliquidStrikes:
    def test_zero_oi_filtered(
        self, bullish_indicators, bullish_regime,
        support_levels, resistance_levels, params_no_basis
    ):
        calls = [_minimal_call(110.0, oi=0)]
        params_no_basis.min_open_interest = 1
        result = score_covered_calls(
            calls, 108.0, 7,
            bullish_indicators, bullish_regime,
            support_levels, resistance_levels,
            params_no_basis,
        )
        assert result == []

    def test_wide_spread_filtered(
        self, bullish_indicators, bullish_regime,
        support_levels, resistance_levels, params_no_basis
    ):
        # bid=0.01, ask=5.00 → enormous spread
        calls = [_minimal_call(110.0, bid=0.01, ask=5.00, oi=500)]
        params_no_basis.max_spread_pct = 0.30
        result = score_covered_calls(
            calls, 108.0, 7,
            bullish_indicators, bullish_regime,
            support_levels, resistance_levels,
            params_no_basis,
        )
        assert result == []

    def test_below_min_premium_filtered(
        self, bullish_indicators, bullish_regime,
        support_levels, resistance_levels, params_no_basis
    ):
        calls = [_minimal_call(110.0, bid=0.01, ask=0.03, oi=500)]
        params_no_basis.min_premium = 0.10
        result = score_covered_calls(
            calls, 108.0, 7,
            bullish_indicators, bullish_regime,
            support_levels, resistance_levels,
            params_no_basis,
        )
        assert result == []


class TestOptionContractHelpers:
    def test_mid_is_average_of_bid_ask(self):
        c = _minimal_call(110.0, bid=1.0, ask=1.40)
        assert c.mid == pytest.approx(1.20)

    def test_mid_falls_back_to_last_if_no_bid_ask(self):
        c = OptionContract(
            strike=110.0, expiration="2024-12-27", option_type="call",
            bid=0.0, ask=0.0, last=1.50,
            volume=10, open_interest=100,
        )
        # When bid and ask are both 0, mid falls back to last price
        assert c.mid == pytest.approx(1.50)
        assert c.last == 1.50

    def test_bid_ask_spread_pct_calculation(self):
        c = _minimal_call(110.0, bid=1.0, ask=1.20)
        # mid=1.10, spread=0.20, pct=0.20/1.10 ≈ 0.182
        assert c.bid_ask_spread_pct == pytest.approx(0.20 / 1.10, rel=0.01)

    def test_bid_ask_spread_pct_returns_1_when_mid_zero(self):
        c = OptionContract(
            strike=110.0, expiration="2024-12-27", option_type="call",
            bid=0.0, ask=0.0, last=0.0,
            volume=0, open_interest=0,
        )
        assert c.bid_ask_spread_pct == pytest.approx(1.0)
