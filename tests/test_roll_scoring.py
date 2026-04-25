"""Tests for scoring/roll_scoring.py."""

import pytest

from scoring.position_decision import OpenPosition, RollCandidate
from scoring.roll_scoring import (
    AGGRESSIVE_DELTA_MIN,
    SAFE_DELTA_MAX,
    format_roll_recommendation,
    pick_top_rolls,
    score_roll_candidate,
)


# ── Helpers ─────────────────────────────────────

def make_csp(**kw) -> OpenPosition:
    defaults = dict(
        strategy="CSP",
        strike=370.0,
        original_premium=4.50,
        current_bid=2.00,
        current_ask=2.20,
        total_premium_collected=4.50,
        desired_buy_price=365.0,
    )
    defaults.update(kw)
    return OpenPosition(**defaults)


def make_cc(**kw) -> OpenPosition:
    defaults = dict(
        strategy="CC",
        strike=400.0,
        original_premium=3.00,
        current_bid=1.50,
        current_ask=1.70,
        total_premium_collected=3.00,
        cost_basis=365.0,
    )
    defaults.update(kw)
    return OpenPosition(**defaults)


def make_roll(**kw) -> RollCandidate:
    defaults = dict(
        strike=370.0,
        expiration="2026-05-02",
        dte=10,
        bid=3.00,
        ask=3.20,
        mid=3.10,
        delta=0.25,
        open_interest=500,
        roll_credit=0.50,
        roll_type="out",
        spread_pct=0.05,
        buy_to_close=2.20,
        sell_to_open=2.70,
    )
    defaults.update(kw)
    return RollCandidate(**defaults)


# ── Hard filters ────────────────────────────────

def test_csp_rejects_wide_spread():
    pos = make_csp()
    c = make_roll(spread_pct=0.20)
    s = score_roll_candidate(c, pos)
    assert s.rejected
    assert "Spread" in s.rejection_reason


def test_csp_rejects_low_open_interest():
    pos = make_csp()
    c = make_roll(open_interest=10)
    s = score_roll_candidate(c, pos, min_open_interest=50)
    assert s.rejected
    assert "OI" in s.rejection_reason


def test_csp_rejects_dte_outside_window():
    pos = make_csp()
    c_short = make_roll(dte=5)
    c_long = make_roll(dte=30)
    assert score_roll_candidate(c_short, pos).rejected
    assert score_roll_candidate(c_long, pos).rejected


def test_csp_rejects_when_earnings_in_window():
    pos = make_csp()
    c = make_roll()
    s = score_roll_candidate(c, pos, has_earnings=True)
    assert s.rejected
    assert "Earnings" in s.rejection_reason


def test_csp_rejects_debit_without_strike_improvement():
    pos = make_csp(strike=370)
    # Big debit, no strike change
    c = make_roll(strike=370, roll_credit=-0.50)
    s = score_roll_candidate(c, pos)
    assert s.rejected


def test_csp_accepts_small_debit_with_big_strike_improvement():
    pos = make_csp(strike=370)
    c = make_roll(strike=362, roll_credit=-0.10)
    s = score_roll_candidate(c, pos)
    assert not s.rejected
    assert s.score > 0


def test_cc_rejects_strike_below_cost_basis():
    pos = make_cc(strike=400, cost_basis=365.0)
    c = make_roll(strike=360, roll_credit=0.50)
    s = score_roll_candidate(c, pos)
    assert s.rejected
    assert "cost basis" in s.rejection_reason.lower()


# ── Sub-score behavior ──────────────────────────

def test_csp_higher_credit_yields_higher_credit_subscore():
    pos = make_csp()
    s_low = score_roll_candidate(
        make_roll(roll_credit=0.10), pos,
    )
    s_high = score_roll_candidate(
        make_roll(roll_credit=1.00), pos,
    )
    assert s_high.sub_scores["credit"] > s_low.sub_scores["credit"]


def test_csp_rewards_lower_effective_cost():
    pos = make_csp(strike=370, original_premium=4.50, desired_buy_price=360.0)
    # Same strike, higher credit -> lower effective cost
    c_thin = make_roll(strike=370, roll_credit=0.10)
    c_fat = make_roll(strike=370, roll_credit=2.00)
    s_thin = score_roll_candidate(c_thin, pos)
    s_fat = score_roll_candidate(c_fat, pos)
    assert (
        s_fat.sub_scores["effective_cost"]
        > s_thin.sub_scores["effective_cost"]
    )


def test_csp_lower_delta_scores_better_on_delta_axis():
    pos = make_csp()
    s_low = score_roll_candidate(make_roll(delta=0.10), pos)
    s_high = score_roll_candidate(make_roll(delta=0.40), pos)
    assert s_low.sub_scores["delta"] > s_high.sub_scores["delta"]


def test_better_liquidity_scores_higher():
    pos = make_csp()
    s_thin = score_roll_candidate(
        make_roll(open_interest=60, spread_pct=0.13), pos,
    )
    s_deep = score_roll_candidate(
        make_roll(open_interest=2000, spread_pct=0.02), pos,
    )
    assert s_deep.sub_scores["liquidity"] > s_thin.sub_scores["liquidity"]


# ── Best != highest premium ─────────────────────

def test_highest_premium_not_always_chosen():
    """A higher-premium candidate with much worse liquidity should lose."""
    pos = make_csp()
    high_prem_thin = make_roll(
        strike=370, roll_credit=2.50, delta=0.45,
        open_interest=60, spread_pct=0.14,
    )
    moderate_prem_clean = make_roll(
        strike=365, roll_credit=0.80, delta=0.20,
        open_interest=2000, spread_pct=0.02,
    )
    picks = pick_top_rolls([high_prem_thin, moderate_prem_clean], pos)
    assert picks.balanced is not None
    # Balanced should be the cleaner, lower-premium candidate
    assert picks.balanced.strike == 365.0


# ── pick_top_rolls bucketing ────────────────────

def test_pick_top_rolls_buckets_by_delta():
    pos = make_csp()
    safe = make_roll(strike=350, roll_credit=0.40, delta=0.12)
    balanced = make_roll(strike=365, roll_credit=0.90, delta=0.25)
    aggro = make_roll(strike=368, roll_credit=1.40, delta=0.40)
    picks = pick_top_rolls([safe, balanced, aggro], pos)

    assert picks.safe is not None
    assert picks.aggressive is not None
    assert picks.balanced is not None
    assert picks.safe.delta <= SAFE_DELTA_MAX
    assert picks.aggressive.delta > AGGRESSIVE_DELTA_MIN


def test_pick_top_rolls_handles_empty():
    pos = make_csp()
    assert pick_top_rolls([], pos).balanced is None


def test_safe_falls_back_when_no_low_delta_candidate():
    """When every candidate has delta > SAFE_DELTA_MAX (e.g. high-IV
    names like TSLA), Safe must still surface the lowest-delta one
    instead of being blank."""
    pos = make_csp()
    a = make_roll(strike=372, roll_credit=1.05, delta=0.42)
    b = make_roll(strike=370, roll_credit=0.80, delta=0.35)
    picks = pick_top_rolls([a, b], pos)
    assert picks.safe is not None
    # Lowest delta wins the Safe slot.
    assert picks.safe.delta == 0.35


def test_aggressive_excludes_itm_strikes_for_cc():
    """Aggressive must not pick an in-the-money strike for a CC,
    even if its premium is the highest. Spot is $400, so strikes
    below $400 are ITM and must be skipped."""
    pos = make_cc(strike=400.0, cost_basis=365.0)
    # Highest credit, but ITM (strike below spot 400).
    itm_juicy = make_roll(
        strike=395, roll_credit=8.00, delta=0.55,
        open_interest=500, spread_pct=0.04,
    )
    # OTM with smaller credit — should win Aggressive.
    otm_solid = make_roll(
        strike=405, roll_credit=2.50, delta=0.35,
        open_interest=500, spread_pct=0.04,
    )
    otm_safe = make_roll(
        strike=420, roll_credit=0.60, delta=0.15,
        open_interest=500, spread_pct=0.04,
    )
    picks = pick_top_rolls(
        [itm_juicy, otm_solid, otm_safe], pos,
        underlying_price=400.0,
    )
    assert picks.aggressive is not None
    assert picks.aggressive.strike >= 400.0
    assert picks.aggressive is otm_solid


def test_aggressive_excludes_itm_strikes_for_csp():
    """For a CSP, ITM means strike above spot. The juiciest premium
    is on the ITM 375 strike but Aggressive must skip it and pick
    the highest-credit OTM (≤ spot) candidate instead."""
    pos = make_csp(strike=370.0, desired_buy_price=365.0)
    itm_juicy = make_roll(
        strike=375, roll_credit=8.00, delta=0.55,
        open_interest=500, spread_pct=0.04,
    )
    otm_high_credit = make_roll(
        strike=368, roll_credit=4.00, delta=0.45,
        open_interest=500, spread_pct=0.04,
    )
    otm_solid = make_roll(
        strike=365, roll_credit=2.50, delta=0.30,
        open_interest=500, spread_pct=0.04,
    )
    otm_safe = make_roll(
        strike=355, roll_credit=0.60, delta=0.15,
        open_interest=500, spread_pct=0.04,
    )
    picks = pick_top_rolls(
        [itm_juicy, otm_high_credit, otm_solid, otm_safe], pos,
        underlying_price=370.0,
    )
    assert picks.aggressive is not None
    # Core invariant: Aggressive must not be in-the-money.
    assert picks.aggressive.strike <= 370.0
    assert picks.aggressive is not itm_juicy


def test_aggressive_picks_highest_credit_distinct_from_balanced():
    """Aggressive should pick the highest-credit (closest-to-assignment)
    candidate and differ from Balanced when an alternative exists."""
    pos = make_csp()
    # Same composite score is unlikely; rig it so Balanced wins on score
    # but a different strike has higher credit.
    cleaner = make_roll(
        strike=365, roll_credit=0.80, delta=0.20,
        open_interest=2000, spread_pct=0.02,
    )
    juicier = make_roll(
        strike=370, roll_credit=2.00, delta=0.40,
        open_interest=400, spread_pct=0.06,
    )
    picks = pick_top_rolls([cleaner, juicier], pos)
    assert picks.aggressive is not None
    assert picks.balanced is not None
    assert picks.aggressive is not picks.balanced
    # Aggressive should be the higher-credit strike.
    assert picks.aggressive.roll_credit >= picks.balanced.roll_credit


def test_pick_top_rolls_skips_rejected():
    pos = make_csp()
    bad = make_roll(spread_pct=0.20)  # rejected
    good = make_roll(strike=365, roll_credit=0.80)
    picks = pick_top_rolls([bad, good], pos)
    assert picks.balanced is good
    assert all(not s.rejected for _, s in picks.all_scored)


# ── Recommendation text ─────────────────────────

def test_format_recommendation_includes_strike_exp_and_credit():
    pos = make_csp()
    c = make_roll(strike=370, expiration="2026-05-15", roll_credit=6.85)
    score = score_roll_candidate(c, pos)
    text = format_roll_recommendation(c, pos, score, bucket="balanced")
    assert "370.00P" in text
    assert "2026-05-15" in text
    assert "+$6.85" in text
    assert "best balanced roll" in text
