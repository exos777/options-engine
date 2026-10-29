"""Tests for scoring/csp_premium.py — same-day CSP scoring."""

from __future__ import annotations

import pytest

from scoring.csp_premium import (
    MAX_SAME_DAY_CAPTURE,
    PremiumParams,
    delta_score,
    edge_score,
    estimate_decay_today,
    expected_move_score,
    liquidity_score,
    score_premium_puts,
)
from strategies.models import OptionContract, SupportResistanceLevel

SPOT = 100.0


def _put(strike, bid, ask, delta, theta=-0.30, oi=1000, vol=500):
    return OptionContract(
        strike=strike, expiration="2026-10-30", option_type="put",
        bid=bid, ask=ask, last=bid, volume=vol, open_interest=oi,
        implied_volatility=0.50,
        delta=-delta if delta is not None else None, theta=theta,
    )


# ── decay estimate ──────────────────────────────

def test_decay_scales_with_hours_left():
    put = _put(95, 1.00, 1.02, 0.15, theta=-0.40)
    assert estimate_decay_today(put, 1, 6.5) == pytest.approx(0.40)
    assert estimate_decay_today(put, 1, 3.25) == pytest.approx(0.20)
    assert estimate_decay_today(put, 1, 0.0) == 0.0


def test_decay_capped_at_share_of_credit():
    put = _put(95, 0.50, 0.52, 0.15, theta=-2.00)      # 0DTE-style theta
    assert estimate_decay_today(put, 0, 6.5) == pytest.approx(0.51 * MAX_SAME_DAY_CAPTURE)


def test_decay_halved_for_earnings_before_expiry():
    put = _put(95, 1.00, 1.02, 0.15, theta=-0.40)
    assert estimate_decay_today(put, 1, 6.5, event_before_expiry=True) == pytest.approx(0.20)


def test_decay_without_theta_spreads_credit_over_sessions():
    put = _put(95, 1.00, 1.00, 0.15, theta=None)
    assert estimate_decay_today(put, 4, 6.5) == pytest.approx(0.25)
    assert estimate_decay_today(put, 0, 3.25) == pytest.approx(0.50)


# ── subscores ───────────────────────────────────

def test_edge_score_scales_and_caps():
    assert edge_score(0.0025) == pytest.approx(50)
    assert edge_score(0.02) == 100


def test_delta_score_prefers_lower_delta_for_same_day():
    assert delta_score(0.15) == 100
    assert delta_score(0.05) == pytest.approx(60)
    assert delta_score(0.30) == pytest.approx(50)
    assert delta_score(None) == 50


def test_liquidity_rewards_depth_not_just_spread():
    thin = liquidity_score(0.03, volume=100, open_interest=150, max_spread_pct=0.10)
    deep = liquidity_score(0.03, volume=2000, open_interest=5000, max_spread_pct=0.10)
    assert deep > thin
    assert liquidity_score(0.01, 2000, 5000, 0.10) == pytest.approx(100)
    assert liquidity_score(0.10, 0, 0, 0.10) == 0


def test_expected_move_score():
    assert expected_move_score(0.06, 0.05) == 100
    assert expected_move_score(0.025, 0.05) == pytest.approx(50)


# ── filters ─────────────────────────────────────

def test_filters_itm_wide_thin_illiquid_and_out_of_band_delta():
    puts = [
        _put(101, 2.00, 2.05, 0.55),                    # ITM
        _put(98, 1.00, 1.30, 0.30),                     # spread 26% > 10%
        _put(97, 0.80, 0.82, 0.22, oi=10, vol=20),      # illiquid
        _put(96, 0.60, 0.62, 0.45),                     # delta too high
        _put(90, 0.03, 0.04, 0.03),                     # bid below min
        _put(95, 0.40, 0.41, 0.12),                     # ✓
    ]
    assert [p.strike for p in score_premium_puts(puts, SPOT, 1)] == [95]


def test_drops_puts_whose_decay_cannot_beat_the_spread():
    slow = _put(95, 0.40, 0.48, 0.12, theta=-0.05)      # 0.05 decay < 0.08 spread
    assert score_premium_puts([slow], SPOT, 5) == []


def test_rejects_expirations_beyond_a_week():
    assert score_premium_puts([_put(97, 0.80, 0.82, 0.22)], SPOT, 8) == []


# ── the AMZN 235 vs 237.5 case ──────────────────

def test_deeper_market_can_beat_slightly_richer_strike():
    """Same spread, similar decay: 235's bigger volume/OI should now count."""
    richer_thin = _put(97.5, 1.00, 1.04, 0.20, theta=-0.50, oi=120, vol=110)
    deep = _put(95, 0.80, 0.83, 0.15, theta=-0.45, oi=6000, vol=3000)
    picks = score_premium_puts([richer_thin, deep], SPOT, 1, hours_left=6.5)
    assert picks[0].strike == 95


# ── plan ────────────────────────────────────────

def test_plan_numbers():
    [pick] = score_premium_puts(
        [_put(95, 1.00, 1.02, 0.15, theta=-0.40)], SPOT, 1, hours_left=6.5,
        params=PremiumParams(stop_multiple=2.0),
    )
    assert pick.credit == pytest.approx(1.01)
    assert pick.est_decay == pytest.approx(0.40)
    assert pick.net_edge == pytest.approx(0.38)
    assert pick.edge_pct == pytest.approx(0.38 / 95)
    assert pick.target_price == pytest.approx(0.61)
    assert pick.stop_at == pytest.approx(2.02)
    assert pick.assigned_cost == pytest.approx(93.99)
    assert pick.cash_required == 9500
    assert pick.capture_pct == pytest.approx(0.40 / 1.01)


def test_less_time_left_means_less_edge():
    put = [_put(95, 1.00, 1.02, 0.15, theta=-0.40)]
    morning = score_premium_puts(put, SPOT, 1, hours_left=6.5)[0]
    late = score_premium_puts(put, SPOT, 1, hours_left=1.0)[0]
    assert late.net_edge < morning.net_edge
    assert late.score < morning.score


def test_support_and_expected_move_flags():
    supports = [SupportResistanceLevel(price=96.0, level_type="support",
                                       strength=0.8, touches=4)]
    [pick] = score_premium_puts(
        [_put(94, 0.50, 0.51, 0.12, theta=-0.30)], SPOT, 1,
        expected_move=4.0, supports=supports,
    )
    assert pick.support == 96.0
    assert pick.outside_expected_move
