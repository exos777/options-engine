"""Tests for greeks/black_scholes.py"""

import pytest

from greeks.black_scholes import compute_greeks, backfill_greeks


class TestCallGreeks:
    """Black-Scholes Greeks for call options."""

    def test_atm_call_delta_near_half(self):
        g = compute_greeks(S=100, K=100, T=30/365, sigma=0.30, option_type="call")
        assert g is not None
        # ATM call delta is slightly above 0.50 due to drift
        assert 0.48 <= g.delta <= 0.60

    def test_deep_otm_call_delta_near_zero(self):
        g = compute_greeks(S=100, K=150, T=7/365, sigma=0.25, option_type="call")
        assert g is not None
        assert g.delta < 0.05

    def test_deep_itm_call_delta_near_one(self):
        g = compute_greeks(S=100, K=60, T=30/365, sigma=0.25, option_type="call")
        assert g is not None
        assert g.delta > 0.95

    def test_theta_is_negative(self):
        g = compute_greeks(S=100, K=100, T=30/365, sigma=0.30, option_type="call")
        assert g is not None
        assert g.theta < 0  # long call loses value over time

    def test_vega_is_positive(self):
        g = compute_greeks(S=100, K=100, T=30/365, sigma=0.30, option_type="call")
        assert g is not None
        assert g.vega > 0

    def test_gamma_is_positive(self):
        g = compute_greeks(S=100, K=100, T=30/365, sigma=0.30, option_type="call")
        assert g is not None
        assert g.gamma > 0

    def test_gamma_highest_atm(self):
        atm = compute_greeks(S=100, K=100, T=30/365, sigma=0.30, option_type="call")
        otm = compute_greeks(S=100, K=115, T=30/365, sigma=0.30, option_type="call")
        assert atm.gamma > otm.gamma


class TestPutGreeks:
    """Black-Scholes Greeks for put options."""

    def test_put_call_parity_delta(self):
        """Put delta = Call delta - 1."""
        call = compute_greeks(S=100, K=100, T=30/365, sigma=0.30, option_type="call")
        put = compute_greeks(S=100, K=100, T=30/365, sigma=0.30, option_type="put")
        assert call is not None and put is not None
        assert put.delta == pytest.approx(call.delta - 1.0, abs=0.001)

    def test_put_delta_negative(self):
        g = compute_greeks(S=100, K=100, T=30/365, sigma=0.30, option_type="put")
        assert g is not None
        assert g.delta < 0

    def test_deep_otm_put_delta_near_zero(self):
        g = compute_greeks(S=100, K=60, T=7/365, sigma=0.25, option_type="put")
        assert g is not None
        assert abs(g.delta) < 0.05

    def test_put_gamma_equals_call_gamma(self):
        call = compute_greeks(S=100, K=100, T=30/365, sigma=0.30, option_type="call")
        put = compute_greeks(S=100, K=100, T=30/365, sigma=0.30, option_type="put")
        assert call.gamma == pytest.approx(put.gamma, abs=0.0001)

    def test_put_vega_equals_call_vega(self):
        call = compute_greeks(S=100, K=100, T=30/365, sigma=0.30, option_type="call")
        put = compute_greeks(S=100, K=100, T=30/365, sigma=0.30, option_type="put")
        assert call.vega == pytest.approx(put.vega, abs=0.0001)


class TestEdgeCases:
    """Edge cases and invalid inputs."""

    def test_zero_time_call_itm(self):
        g = compute_greeks(S=110, K=100, T=0, sigma=0.30, option_type="call")
        assert g is not None
        assert g.delta == 1.0
        assert g.gamma == 0.0
        assert g.theta == 0.0

    def test_zero_time_call_otm(self):
        g = compute_greeks(S=90, K=100, T=0, sigma=0.30, option_type="call")
        assert g is not None
        assert g.delta == 0.0

    def test_zero_time_put_itm(self):
        g = compute_greeks(S=90, K=100, T=0, sigma=0.30, option_type="put")
        assert g is not None
        assert g.delta == -1.0

    def test_zero_sigma_returns_none(self):
        assert compute_greeks(S=100, K=100, T=30/365, sigma=0.0, option_type="call") is None

    def test_zero_spot_returns_none(self):
        assert compute_greeks(S=0, K=100, T=30/365, sigma=0.30, option_type="call") is None

    def test_zero_strike_returns_none(self):
        assert compute_greeks(S=100, K=0, T=30/365, sigma=0.30, option_type="call") is None


class TestBackfillGreeks:
    """Test the convenience wrapper."""

    def test_backfill_with_valid_inputs(self):
        g = backfill_greeks(strike=100, spot=100, dte=7, iv=0.30, option_type="call")
        assert g is not None
        assert 0.40 <= g.delta <= 0.65

    def test_backfill_with_none_iv(self):
        assert backfill_greeks(strike=100, spot=100, dte=7, iv=None, option_type="call") is None

    def test_backfill_with_zero_iv(self):
        assert backfill_greeks(strike=100, spot=100, dte=7, iv=0.0, option_type="call") is None

    def test_backfill_dte_zero_handled(self):
        # DTE=0 gets clamped to 1 day internally
        g = backfill_greeks(strike=100, spot=100, dte=0, iv=0.30, option_type="call")
        assert g is not None

    def test_put_backfill_delta_negative(self):
        g = backfill_greeks(strike=95, spot=100, dte=7, iv=0.30, option_type="put")
        assert g is not None
        assert g.delta < 0


class TestRealisticWeeklyValues:
    """
    Sanity-check that Greeks for typical weekly option scenarios
    are in the right ballpark.
    """

    def test_weekly_otm_call_on_200_stock(self):
        """$200 stock, $210 strike, 7 DTE, 30% IV."""
        g = compute_greeks(S=200, K=210, T=7/365, sigma=0.30, option_type="call")
        assert g is not None
        # Delta should be 0.15-0.30 for 5% OTM weekly
        assert 0.05 <= g.delta <= 0.35
        # Theta should be meaningful (negative, several cents/day)
        assert g.theta < -0.01
        # Vega should be positive
        assert g.vega > 0

    def test_weekly_otm_put_on_200_stock(self):
        """$200 stock, $190 strike, 7 DTE, 30% IV."""
        g = compute_greeks(S=200, K=190, T=7/365, sigma=0.30, option_type="put")
        assert g is not None
        # Delta should be -0.15 to -0.30 for 5% OTM weekly
        assert -0.35 <= g.delta <= -0.05
        assert g.theta < -0.01
