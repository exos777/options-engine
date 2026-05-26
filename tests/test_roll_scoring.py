"""Tests for scoring/roll_scoring.py."""

import pytest

from scoring.position_decision import OpenPosition, RollCandidate
from scoring.roll_scoring import (
    DELTA_HARD_CAP,
    SAFE_DELTA_MAX,
    Signals,
    derive_signals,
    format_roll_recommendation,
    pick_top_rolls,
    score_roll_candidate,
)
from strategies.models import (
    SupportResistanceLevel,
    TechnicalIndicators,
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


def make_indicators(**kw) -> TechnicalIndicators:
    defaults = dict(
        sma_20=370.0,
        sma_50=365.0,
        rsi_14=58.0,
        atr_14=5.0,
        avg_volume_20=1_000_000,
        current_price=372.0,
        weekly_atr_est=11.0,
        adx_14=22.0,
        squeeze_on=False,
        vwap=371.0,
    )
    defaults.update(kw)
    return TechnicalIndicators(**defaults)


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


def test_rejects_high_delta_hard_cap():
    """|delta| >= DELTA_HARD_CAP (0.35) is rejected outright."""
    pos = make_csp()
    s = score_roll_candidate(make_roll(delta=DELTA_HARD_CAP), pos)
    assert s.rejected
    assert "Delta" in s.rejection_reason


def test_csp_rejects_itm_strike_when_spot_known():
    """Put with strike above spot is ITM and must be rejected."""
    pos = make_csp(strike=370.0)
    c = make_roll(strike=375.0)
    s = score_roll_candidate(c, pos, underlying_price=370.0)
    assert s.rejected
    assert "ITM" in s.rejection_reason


def test_cc_rejects_itm_strike_when_spot_known():
    pos = make_cc(strike=400.0)
    c = make_roll(strike=395.0, delta=0.30)
    s = score_roll_candidate(c, pos, underlying_price=400.0)
    assert s.rejected
    assert "ITM" in s.rejection_reason


def test_csp_rejects_debit_without_strike_improvement():
    pos = make_csp(strike=370)
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
    c = make_roll(strike=360, roll_credit=0.50, delta=0.25)
    s = score_roll_candidate(c, pos)
    assert s.rejected
    assert "cost basis" in s.rejection_reason.lower()


# ── Sub-score behaviour ──────────────────────────

def test_higher_credit_yields_higher_premium_subscore():
    pos = make_csp()
    s_low = score_roll_candidate(make_roll(roll_credit=0.10), pos)
    s_high = score_roll_candidate(make_roll(roll_credit=1.00), pos)
    assert s_high.sub_scores["premium"] > s_low.sub_scores["premium"]


def test_lower_delta_yields_smaller_delta_penalty():
    pos = make_csp()
    s_low = score_roll_candidate(make_roll(delta=0.10), pos)
    s_high = score_roll_candidate(make_roll(delta=0.30), pos)
    assert s_low.delta_penalty < s_high.delta_penalty


def test_better_liquidity_scores_higher():
    pos = make_csp()
    s_thin = score_roll_candidate(
        make_roll(open_interest=60, spread_pct=0.13), pos,
    )
    s_deep = score_roll_candidate(
        make_roll(open_interest=2000, spread_pct=0.02), pos,
    )
    assert s_deep.sub_scores["liquidity"] > s_thin.sub_scores["liquidity"]


def test_higher_theta_per_day_scores_higher():
    """Same DTE — higher mid (more daily decay) wins on theta."""
    pos = make_csp()
    thin = make_roll(mid=0.30, bid=0.25, ask=0.35, dte=14)
    fat = make_roll(mid=2.50, bid=2.45, ask=2.55, dte=14)
    s_thin = score_roll_candidate(thin, pos)
    s_fat = score_roll_candidate(fat, pos)
    assert s_fat.sub_scores["theta"] > s_thin.sub_scores["theta"]


def test_distance_score_rewards_otm_distance():
    pos = make_csp()
    near = make_roll(strike=369.0)
    far = make_roll(strike=355.0)
    s_near = score_roll_candidate(near, pos, underlying_price=370.0)
    s_far = score_roll_candidate(far, pos, underlying_price=370.0)
    assert s_far.sub_scores["distance"] > s_near.sub_scores["distance"]


def test_csp_structure_score_rewards_strike_at_support():
    """Strike sitting on a support level should boost structure score."""
    pos = make_csp()
    supports = [SupportResistanceLevel(
        price=365.0, level_type="support", strength=0.9, touches=4,
    )]
    at_support = make_roll(strike=365.0, delta=0.25)
    away = make_roll(strike=350.0, delta=0.10)
    s_at = score_roll_candidate(
        at_support, pos,
        underlying_price=372.0, support_levels=supports,
    )
    s_away = score_roll_candidate(
        away, pos,
        underlying_price=372.0, support_levels=supports,
    )
    assert s_at.sub_scores["structure"] > s_away.sub_scores["structure"]


# ── Best != highest premium ─────────────────────

def test_highest_premium_not_always_chosen():
    """A high-premium candidate with bad liquidity / high delta should lose."""
    pos = make_csp()
    high_prem_dirty = make_roll(
        strike=370, roll_credit=2.50, delta=0.32,
        open_interest=60, spread_pct=0.14,
    )
    moderate_prem_clean = make_roll(
        strike=365, roll_credit=0.80, delta=0.20,
        open_interest=2000, spread_pct=0.02,
    )
    picks = pick_top_rolls(
        [high_prem_dirty, moderate_prem_clean], pos,
        underlying_price=370.0,
    )
    assert picks.balanced is not None
    assert picks.balanced.strike == 365.0


# ── Signals derivation ──────────────────────────

def test_signals_default_to_neutral_without_indicators():
    sig = derive_signals(None, 0.0, None, None)
    assert sig == Signals()


def test_direction_bias_bullish():
    ind = make_indicators(
        current_price=380.0, sma_20=375.0, sma_50=365.0, rsi_14=62.0,
    )
    sig = derive_signals(ind, 380.0, [], [])
    assert sig.direction == "bullish"


def test_direction_bias_bearish():
    ind = make_indicators(
        current_price=350.0, sma_20=355.0, sma_50=365.0, rsi_14=38.0,
    )
    sig = derive_signals(ind, 350.0, [], [])
    assert sig.direction == "bearish"


def test_volatility_compressed_when_squeeze_on():
    ind = make_indicators(squeeze_on=True)
    sig = derive_signals(ind, 372.0, [], [])
    assert sig.volatility == "compressed"


def test_chart_structure_near_support():
    ind = make_indicators(current_price=366.0)
    supports = [SupportResistanceLevel(
        price=365.0, level_type="support", strength=0.8, touches=3,
    )]
    sig = derive_signals(ind, 366.0, supports, [])
    assert sig.structure == "near_support"


# ── pick_top_rolls bucketing ────────────────────

def test_pick_top_rolls_buckets_by_delta():
    pos = make_csp()
    safe = make_roll(strike=350, roll_credit=0.40, delta=0.12)
    balanced = make_roll(strike=365, roll_credit=0.90, delta=0.22)
    aggro = make_roll(strike=368, roll_credit=1.40, delta=0.32)
    picks = pick_top_rolls([safe, balanced, aggro], pos, underlying_price=370.0)

    assert picks.safe is not None
    assert picks.balanced is not None
    assert picks.premium is not None
    assert picks.aggressive is picks.premium  # alias still works
    assert picks.safe.delta <= SAFE_DELTA_MAX


def test_pick_top_rolls_handles_empty():
    pos = make_csp()
    assert pick_top_rolls([], pos).balanced is None


def test_pick_top_rolls_rejects_all_high_delta_returns_empty():
    """Every candidate has |delta| >= 0.35 — picks must be empty."""
    pos = make_csp()
    a = make_roll(strike=370, roll_credit=1.00, delta=0.40)
    b = make_roll(strike=368, roll_credit=1.50, delta=0.42)
    picks = pick_top_rolls([a, b], pos, underlying_price=370.0)
    assert picks.balanced is None
    assert picks.safe is None
    assert picks.premium is None


def test_safe_falls_back_when_no_low_delta_candidate():
    """When no candidate has |delta| <= SAFE_DELTA_MAX, fall back to lowest delta."""
    pos = make_csp()
    a = make_roll(strike=372, roll_credit=1.05, delta=0.32)
    b = make_roll(strike=370, roll_credit=0.80, delta=0.28)
    picks = pick_top_rolls([a, b], pos, underlying_price=375.0)
    assert picks.safe is not None
    assert picks.safe.delta == 0.28


def test_premium_excludes_itm_strikes_for_cc():
    """Premium pick must skip ITM call strikes."""
    pos = make_cc(strike=400.0, cost_basis=365.0)
    # ITM (strike < spot 400) AND high delta — would be rejected anyway.
    itm_juicy = make_roll(
        strike=395, roll_credit=8.00, delta=0.30,
        open_interest=500, spread_pct=0.04,
    )
    otm_solid = make_roll(
        strike=405, roll_credit=2.50, delta=0.30,
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
    assert picks.premium is not None
    assert picks.premium.strike >= 400.0


def test_premium_excludes_itm_strikes_for_csp():
    """For CSP, ITM means strike above spot. Premium must pick OTM."""
    pos = make_csp(strike=370.0, desired_buy_price=365.0)
    itm = make_roll(
        strike=375, roll_credit=8.00, delta=0.30,
        open_interest=500, spread_pct=0.04,
    )
    otm_high = make_roll(
        strike=368, roll_credit=4.00, delta=0.32,
        open_interest=500, spread_pct=0.04,
    )
    otm_solid = make_roll(
        strike=365, roll_credit=2.50, delta=0.25,
        open_interest=500, spread_pct=0.04,
    )
    otm_safe = make_roll(
        strike=355, roll_credit=0.60, delta=0.15,
        open_interest=500, spread_pct=0.04,
    )
    picks = pick_top_rolls(
        [itm, otm_high, otm_solid, otm_safe], pos,
        underlying_price=370.0,
    )
    assert picks.premium is not None
    assert picks.premium.strike <= 370.0


def test_premium_picks_highest_credit_distinct_from_balanced():
    pos = make_csp()
    cleaner = make_roll(
        strike=365, roll_credit=0.80, delta=0.18,
        open_interest=2000, spread_pct=0.02,
    )
    juicier = make_roll(
        strike=368, roll_credit=2.00, delta=0.30,
        open_interest=400, spread_pct=0.06,
    )
    picks = pick_top_rolls(
        [cleaner, juicier], pos,
        underlying_price=370.0,
    )
    assert picks.premium is not None
    assert picks.balanced is not None
    assert picks.premium is not picks.balanced
    assert picks.premium.roll_credit >= picks.balanced.roll_credit


def test_pick_top_rolls_skips_rejected():
    pos = make_csp()
    bad = make_roll(spread_pct=0.20)
    good = make_roll(strike=365, roll_credit=0.80)
    picks = pick_top_rolls([bad, good], pos, underlying_price=370.0)
    assert picks.balanced is good
    assert all(not s.rejected for _, s in picks.all_scored)


# ── Recommendation text ─────────────────────────

def test_format_recommendation_includes_strike_exp_and_credit():
    pos = make_csp()
    c = make_roll(
        strike=365, expiration="2026-05-15", roll_credit=1.85, delta=0.20,
    )
    score = score_roll_candidate(c, pos, underlying_price=370.0)
    text = format_roll_recommendation(c, pos, score, bucket="balanced")
    assert "365.00P" in text
    assert "2026-05-15" in text
    assert "+$1.85" in text
    assert "best balanced roll" in text
