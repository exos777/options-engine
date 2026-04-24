"""Tests for scoring/position_decision.py — Roll vs Assign decision engine."""

import pytest

from scoring.position_decision import (
    OpenPosition,
    RollCandidate,
    PositionDecision,
    score_wait,
    _calc_expected_move,
    _expected_move_score_csp,
    _expected_move_score_cc,
    _csp_roll_has_merit,
    _cc_roll_has_merit,
    _prefer_assignment_over_roll,
    _position_size_guidance,
    _build_roll_candidates_filtered,
    evaluate_position,
    find_best_roll_for_premium,
)
from strategies.models import RegimeResult, ChartRegime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_csp(**kwargs) -> OpenPosition:
    if "total_premium" in kwargs:
        kwargs["total_premium_collected"] = kwargs.pop("total_premium")
    if "desired_buy" in kwargs:
        kwargs["desired_buy_price"] = kwargs.pop("desired_buy")
    defaults = dict(
        strategy="CSP",
        strike=370.0,
        original_premium=4.50,
        current_bid=2.00,
        current_ask=2.20,
        total_premium_collected=kwargs.get("total_premium_collected", 4.50),
        desired_buy_price=0.0,
    )
    defaults.update(kwargs)
    return OpenPosition(**defaults)


def make_cc(**kwargs) -> OpenPosition:
    defaults = dict(
        strategy="CC",
        strike=400.0,
        original_premium=3.00,
        current_bid=1.50,
        current_ask=1.70,
        total_premium_collected=3.00,
        cost_basis=365.0,
    )
    defaults.update(kwargs)
    return OpenPosition(**defaults)


def make_roll(**kwargs) -> RollCandidate:
    defaults = dict(
        strike=370.0,
        expiration="2025-05-02",
        dte=7,
        bid=3.00,
        ask=3.20,
        mid=3.10,
        delta=0.30,
        open_interest=500,
        roll_credit=0.50,
        roll_type="out",
        spread_pct=0.0,
    )
    defaults.update(kwargs)
    return RollCandidate(**defaults)


# ── Close Price Mode Tests ─────────────────────

def test_close_cost_conservative():
    pos = make_csp(current_bid=2.00, current_ask=2.20)
    pos.close_price_mode = "conservative"
    assert pos.close_cost == pytest.approx(2.20)


def test_close_cost_realistic():
    pos = make_csp(current_bid=2.00, current_ask=2.20)
    pos.close_price_mode = "realistic"
    assert pos.close_cost == pytest.approx(2.10)


def test_close_cost_optimistic():
    pos = make_csp(current_bid=2.00, current_ask=2.20)
    pos.close_price_mode = "optimistic"
    mid = 2.10
    expected = (2.00 + mid) / 2
    assert pos.close_cost == pytest.approx(expected)


# ── Spread PCT Filter Tests ────────────────────

def test_spread_pct_calculation():
    c = make_roll(strike=365)
    c.bid = 2.00
    c.ask = 2.60
    c.mid = 2.30
    c.spread_pct = (c.ask - c.bid) / max(c.mid, 0.01)
    assert c.spread_pct == pytest.approx(0.2608, rel=0.01)


def test_spread_pct_filtered_out():
    """Candidates with spread > 15% should be excluded."""
    wide_spread = make_roll()
    wide_spread.bid = 1.00
    wide_spread.ask = 1.50
    wide_spread.mid = 1.25
    wide_spread.spread_pct = (1.50 - 1.00) / 1.25
    assert wide_spread.spread_pct > 0.15


# ── Wait Decision Tests ────────────────────────

def test_wait_recommended_otm_high_dte():
    pos = make_csp(
        strike=370,
        original_premium=4.50,
        current_ask=2.00,
        current_bid=1.80,
    )
    score, explanation = score_wait(
        pos,
        current_price=380.0,
        dte_remaining=8,
    )
    assert score >= 60
    assert "otm" in explanation.lower() or "theta" in explanation.lower()


def test_wait_not_recommended_after_50_pct_capture():
    pos = make_csp(
        original_premium=4.50,
        current_ask=1.50,
        current_bid=1.30,
    )
    pos.close_price_mode = "realistic"
    score, _ = score_wait(
        pos,
        current_price=375.0,
        dte_remaining=4,
    )
    assert score < 50


# ── Expected Move Tests ────────────────────────

def test_expected_move_calculation():
    em = _calc_expected_move(
        current_price=391.0,
        implied_volatility=0.45,
        dte=7,
    )
    assert 20 < em < 35


def test_csp_outside_em_rewarded():
    score, reason = _expected_move_score_csp(
        strike=355.0,
        current_price=391.0,
        expected_move=25.0,
        wants_assignment=False,
    )
    assert score > 0
    assert "outside" in reason.lower()


def test_csp_inside_em_penalized_when_unwanted():
    score, reason = _expected_move_score_csp(
        strike=375.0,
        current_price=391.0,
        expected_move=25.0,
        wants_assignment=False,
    )
    assert score < 0
    assert "inside" in reason.lower()


def test_cc_outside_em_rewarded():
    score, reason = _expected_move_score_cc(
        strike=425.0,
        current_price=391.0,
        expected_move=25.0,
        wants_to_keep_shares=True,
    )
    assert score > 0


# ── CSP Roll Merit Tests ───────────────────────

def test_csp_roll_rejected_no_improvement():
    pos = make_csp(
        strike=370,
        total_premium=4.50,
        desired_buy=365,
    )
    candidate = make_roll(
        strike=370,
        roll_credit=-0.10,
        roll_type="out",
    )
    has_merit, reason = _csp_roll_has_merit(pos, candidate)
    assert not has_merit
    assert "does not improve" in reason.lower()


def test_csp_roll_accepted_lowers_strike_with_credit():
    pos = make_csp(strike=370, total_premium=4.50)
    candidate = make_roll(
        strike=360,
        roll_credit=0.80,
        roll_type="down",
    )
    has_merit, reason = _csp_roll_has_merit(pos, candidate)
    assert has_merit
    assert "lowers strike" in reason.lower()


def test_csp_roll_accepted_improves_net_cost():
    pos = make_csp(
        strike=370,
        total_premium=4.50,
        desired_buy=365,
    )
    candidate = make_roll(
        strike=368,
        roll_credit=1.20,
        roll_type="down",
    )
    has_merit, reason = _csp_roll_has_merit(pos, candidate)
    assert has_merit


# ── CC Roll Merit Tests ────────────────────────

def test_cc_roll_rejected_below_cost_basis():
    pos = make_cc(cost_basis=365.0)
    regime = RegimeResult(
        primary=ChartRegime.NEUTRAL,
        secondary=None,
        description="",
        trade_bias="caution",
    )
    candidate = make_roll(
        strike=360,
        roll_type="down",
    )
    has_merit, reason = _cc_roll_has_merit(pos, candidate, regime)
    assert not has_merit
    assert "cost basis" in reason.lower()


def test_cc_roll_accepted_raises_strike():
    pos = make_cc(strike=400, cost_basis=362.30)
    regime = RegimeResult(
        primary=ChartRegime.BULLISH,
        secondary=None,
        description="",
        trade_bias="go",
    )
    candidate = make_roll(
        strike=410,
        roll_credit=0.50,
        roll_type="up",
    )
    has_merit, reason = _cc_roll_has_merit(pos, candidate, regime)
    assert has_merit
    assert "raises" in reason.lower()


# ── Do Not Roll Just to Avoid Assignment ──────

def test_prefer_assignment_over_marginal_roll():
    pos = make_csp(desired_buy=365)
    new_assign, new_roll, note = _prefer_assignment_over_roll(
        pos, 65.0, 70.0, wants_assignment=True,
    )
    assert new_assign >= 65
    assert new_roll <= 70
    assert "wheel-worthy" in note.lower() or "preferred" in note.lower()


# ── Position Sizing Tests ──────────────────────

def test_full_size_clean_conditions():
    regime = RegimeResult(
        primary=ChartRegime.BULLISH,
        secondary=None,
        description="",
        trade_bias="go",
    )
    size, reason = _position_size_guidance(
        regime=regime,
        has_earnings=False,
        dte_remaining=10,
        times_rolled=0,
        confidence=80.0,
    )
    assert "Full" in size


# ── Best Roll for Maximum Premium ──────────────

def test_best_roll_picks_max_credit():
    pos = make_csp(strike=370, total_premium=4.50)
    cands = [
        make_roll(strike=365, roll_credit=0.40, spread_pct=0.05),
        make_roll(strike=360, roll_credit=0.90, spread_pct=0.05),
        make_roll(strike=368, roll_credit=0.55, spread_pct=0.05),
    ]
    best = find_best_roll_for_premium(cands, pos)
    assert best is not None
    assert best.roll_credit == pytest.approx(0.90)


def test_best_roll_excludes_thin_credit():
    pos = make_csp(strike=370, total_premium=4.50)
    cands = [
        make_roll(strike=365, roll_credit=0.15, spread_pct=0.05),
        make_roll(strike=360, roll_credit=0.10, spread_pct=0.05),
    ]
    best = find_best_roll_for_premium(cands, pos)
    assert best is None


def test_best_roll_excludes_wide_spread():
    pos = make_csp(strike=370, total_premium=4.50)
    cands = [
        make_roll(strike=365, roll_credit=0.80, spread_pct=0.25),
    ]
    best = find_best_roll_for_premium(cands, pos)
    assert best is None


def test_best_roll_csp_rejects_strike_above_original():
    pos = make_csp(strike=370, total_premium=4.50)
    cands = [
        make_roll(strike=375, roll_credit=0.80, spread_pct=0.05),
    ]
    best = find_best_roll_for_premium(cands, pos)
    assert best is None


def test_best_roll_cc_rejects_strike_below_cost_basis():
    pos = make_cc(strike=400, cost_basis=365.0)
    cands = [
        make_roll(strike=360, roll_credit=0.80, spread_pct=0.05),
    ]
    best = find_best_roll_for_premium(cands, pos)
    assert best is None


def test_best_roll_returns_none_on_empty():
    pos = make_csp(strike=370, total_premium=4.50)
    assert find_best_roll_for_premium([], pos) is None


def test_no_exposure_poor_conditions():
    regime = RegimeResult(
        primary=ChartRegime.BEARISH,
        secondary=None,
        description="",
        trade_bias="skip",
    )
    size, reason = _position_size_guidance(
        regime=regime,
        has_earnings=True,
        dte_remaining=3,
        times_rolled=3,
        confidence=35.0,
    )
    assert "No New" in size or "Quarter" in size
