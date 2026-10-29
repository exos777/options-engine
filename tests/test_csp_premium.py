"""Tests for scoring/csp_premium.py — short-dated CSP premium scoring."""

from __future__ import annotations

import pytest

from scoring.csp_premium import (
    PremiumParams,
    delta_score,
    expected_move_score,
    liquidity_score,
    premium_score,
    score_premium_puts,
)
from strategies.models import OptionContract, SupportResistanceLevel

SPOT = 100.0


def _put(strike, bid, ask, delta, oi=1000, vol=0):
    return OptionContract(
        strike=strike, expiration="2026-10-30", option_type="put",
        bid=bid, ask=ask, last=bid, volume=vol, open_interest=oi,
        implied_volatility=0.50, delta=-delta if delta is not None else None,
    )


# ── subscores ───────────────────────────────────

def test_premium_score_scales_and_caps():
    assert premium_score(0.0015) == pytest.approx(50)
    assert premium_score(0.0100) == 100


def test_delta_score_peaks_in_sweet_spot():
    assert delta_score(0.20) == 100
    assert delta_score(0.08) == pytest.approx(60)
    assert delta_score(0.35) == pytest.approx(40)
    assert delta_score(None) == 50


def test_expected_move_and_liquidity_scores():
    assert expected_move_score(0.06, 0.05) == 100
    assert expected_move_score(0.025, 0.05) == pytest.approx(50)
    assert liquidity_score(0.01, 0.10) == 100
    assert liquidity_score(0.10, 0.10) == 0


# ── filters ─────────────────────────────────────

def test_filters_itm_wide_thin_illiquid_and_out_of_band_delta():
    puts = [
        _put(101, 2.00, 2.05, 0.55),            # ITM
        _put(98, 1.00, 1.30, 0.30),             # spread 26% > 10%
        _put(97, 0.80, 0.82, 0.22, oi=10),      # illiquid
        _put(96, 0.60, 0.62, 0.45),             # delta too high
        _put(90, 0.03, 0.04, 0.03),             # bid below min
        _put(95, 0.40, 0.41, 0.12),             # ✓
    ]
    picks = score_premium_puts(puts, SPOT, 3)
    assert [p.strike for p in picks] == [95]


def test_volume_counts_as_liquidity_for_fresh_dailies():
    put = _put(97, 0.80, 0.82, 0.22, oi=0, vol=400)
    assert len(score_premium_puts([put], SPOT, 0)) == 1


def test_rejects_expirations_beyond_a_week():
    assert score_premium_puts([_put(97, 0.80, 0.82, 0.22)], SPOT, 8) == []


# ── ranking + plan ──────────────────────────────

def test_plan_numbers():
    [pick] = score_premium_puts(
        [_put(95, 1.00, 1.02, 0.20)], SPOT, 2,
        params=PremiumParams(profit_target=0.5, stop_multiple=2.0),
    )
    assert pick.credit == pytest.approx(1.01)
    assert pick.yield_per_day == pytest.approx(1.01 / 95 / 2)
    assert pick.buy_back_at == pytest.approx(0.51, abs=0.01)
    assert pick.stop_at == pytest.approx(2.02)
    assert pick.assigned_cost == pytest.approx(93.99)
    assert pick.cash_required == 9500


def test_zero_dte_yield_counts_as_one_day():
    [pick] = score_premium_puts([_put(97, 0.50, 0.51, 0.20)], SPOT, 0)
    assert pick.yield_per_day == pytest.approx(0.505 / 97)


def test_higher_iv_rank_raises_score():
    put = [_put(95, 0.60, 0.61, 0.20)]
    low = score_premium_puts(put, SPOT, 2, iv_rank=10)[0].score
    high = score_premium_puts(put, SPOT, 2, iv_rank=90)[0].score
    assert high - low == pytest.approx(80 * 0.20, abs=0.1)


def test_support_and_expected_move_flags():
    supports = [SupportResistanceLevel(price=96.0, level_type="support",
                                       strength=0.8, touches=4)]
    [pick] = score_premium_puts(
        [_put(94, 0.50, 0.51, 0.15)], SPOT, 2,
        expected_move=4.0, supports=supports,
    )
    assert pick.support == 96.0
    assert pick.outside_expected_move           # 6% OTM vs 4% move


def test_best_first():
    puts = [_put(97, 0.80, 0.82, 0.22), _put(93, 0.20, 0.21, 0.09)]
    picks = score_premium_puts(puts, SPOT, 2, iv_rank=50)
    assert picks[0].strike == 97
    assert picks[0].score > picks[1].score
