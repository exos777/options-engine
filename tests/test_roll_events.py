"""Tests for scoring/roll_events.py and event-aware roll scoring."""

from __future__ import annotations

import math
from datetime import date

import pytest

from scoring.position_decision import OpenPosition, RollCandidate
from scoring.roll_events import (
    EVENT_PENALTY,
    EventContext,
    assess_candidate,
    implied_event_move,
    last_expiration_before,
)
from scoring.roll_scoring import score_roll_candidate

TODAY = date(2026, 10, 30)
LEG1 = date(2026, 11, 6)
LEG2 = date(2026, 11, 13)


def _ctx(earnings=None, leg1_iv=0.30, rv=0.28, flagged=False, spot=100.0):
    return EventContext(
        today=TODAY, leg1_expiration=LEG1, leg2_expiration=LEG2, spot=spot,
        leg1_iv=leg1_iv, realized_vol=rv, earnings_date=earnings,
        earnings_flagged=flagged,
    )


# ── context ─────────────────────────────────────

def test_where_earnings_fall():
    assert _ctx(date(2026, 11, 4)).earnings_before_leg1
    assert _ctx(date(2026, 11, 4)).earnings_before_leg2
    mid = _ctx(date(2026, 11, 10))
    assert not mid.earnings_before_leg1 and mid.earnings_before_leg2
    later = _ctx(date(2026, 11, 20))
    assert not later.earnings_before_leg1 and not later.earnings_before_leg2
    assert _ctx(flagged=True).earnings_before_leg2


def test_baseline_iv_skips_a_leg1_that_contains_the_event():
    assert _ctx(date(2026, 11, 10)).base_iv == 0.30          # Leg 1 clean
    assert _ctx(date(2026, 11, 4)).base_iv == 0.28           # Leg 1 has it → realised


def test_implied_event_move():
    # 60% IV vs 30% baseline over 14 days on a $100 stock.
    expected = 100 * math.sqrt((0.60 ** 2 - 0.30 ** 2) * 14 / 365)
    assert implied_event_move(100, 0.60, 0.30, 14) == pytest.approx(expected)
    assert implied_event_move(100, 0.25, 0.30, 14) == 0.0
    assert implied_event_move(100, 0.60, 0.0, 14) == 0.0


# ── per-candidate assessment ────────────────────

def _assess(strike, iv, ctx, sto=3.0, btc=2.0, strategy="CSP"):
    return assess_candidate(strike=strike, sell_to_open=sto, buy_to_close=btc,
                            leg2_iv=iv, dte=14, strategy=strategy, ctx=ctx)


def test_no_event_leaves_credit_alone():
    ev = _assess(95, 0.31, _ctx())
    assert not ev.spans_event
    assert ev.adjusted_credit == pytest.approx(1.0)
    assert ev.penalty == 0 and not ev.reject_reason


def test_strike_inside_earnings_move_is_rejected():
    ctx = _ctx(date(2026, 11, 10))
    move = implied_event_move(100, 0.60, 0.30, 14)            # ≈ $10.0
    ev = _assess(95, 0.60, ctx)
    assert ev.kind == "earnings" and ev.inside_move
    assert ev.event_move == pytest.approx(move)
    assert "earnings move" in ev.reject_reason


def test_strike_outside_earnings_move_is_allowed_with_adjusted_credit():
    ctx = _ctx(date(2026, 11, 10))
    ev = _assess(85, 0.60, ctx, sto=3.0, btc=2.0)
    assert ev.spans_event and not ev.inside_move and not ev.reject_reason
    assert ev.penalty == EVENT_PENALTY
    assert ev.event_premium == pytest.approx(3.0 * (1 - 0.30 / 0.60))  # $1.50
    assert ev.adjusted_credit == pytest.approx(1.0 - 1.5)              # debit without it


def test_call_side_uses_the_upper_move():
    ctx = _ctx(date(2026, 11, 10))
    assert _assess(105, 0.60, ctx, strategy="CC").inside_move
    assert not _assess(115, 0.60, ctx, strategy="CC").inside_move


def test_unmeasurable_earnings_stay_strict():
    ctx = _ctx(date(2026, 11, 10), leg1_iv=0.0, rv=0.0)
    ev = _assess(85, 0.0, ctx)
    assert "event move unknown" in ev.reject_reason


def test_iv_term_spike_without_a_date_is_treated_as_an_event():
    ctx = _ctx()                                               # no earnings known
    assert ctx.term_spike(0.45)
    ev = _assess(85, 0.45, ctx)
    assert ev.kind == "iv_spike"
    assert not _assess(95, 0.33, ctx).spans_event              # 1.1× — normal


def test_last_expiration_before_the_report():
    exps = ["2026-10-30", "2026-11-06", "2026-11-13", "2026-11-20"]
    assert last_expiration_before(exps, date(2026, 11, 18), after=LEG1) == "2026-11-13"
    assert last_expiration_before(exps, date(2026, 11, 10), after=LEG1) is None


# ── wired into the roll scorer ──────────────────

def _cand(strike, bid, iv, delta=0.15):
    return RollCandidate(
        strike=strike, expiration=LEG2.isoformat(), dte=14, bid=bid, ask=bid + 0.1,
        mid=bid + 0.05, delta=delta, open_interest=1000, roll_credit=bid - 2.0,
        roll_type="down", spread_pct=0.03, buy_to_close=2.0, sell_to_open=bid, iv=iv,
    )


POS = OpenPosition(strategy="CSP", strike=98, original_premium=2.5,
                   current_bid=1.9, current_ask=2.0)


def test_scorer_rejects_inside_move_and_penalizes_outside():
    ctx = _ctx(date(2026, 11, 10))
    inside = score_roll_candidate(_cand(95, 4.0, 0.60), POS, underlying_price=100,
                                  event_ctx=ctx)
    assert inside.rejected and "earnings move" in inside.rejection_reason

    outside_event = score_roll_candidate(_cand(88, 3.0, 0.60), POS,
                                         underlying_price=100, event_ctx=ctx)
    no_event = score_roll_candidate(_cand(88, 3.0, 0.30), POS,
                                    underlying_price=100, event_ctx=_ctx())
    assert not outside_event.rejected
    assert outside_event.score < no_event.score
    assert "spans earnings Nov 10" in outside_event.explanation
    assert "event premium" in outside_event.explanation


def test_scorer_without_context_keeps_the_legacy_earnings_rule():
    s = score_roll_candidate(_cand(88, 3.0, 0.60), POS, has_earnings=True)
    assert s.rejected and "Earnings" in s.rejection_reason
