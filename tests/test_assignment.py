"""Tests for scoring/assignment.py — chart-aware probability of assignment."""

from __future__ import annotations

import pytest

from indicators.technical import calculate_full_indicators
from scoring.assignment import (
    MAX_DRIFT,
    AssignmentEstimate,
    SafePick,
    SidePicks,
    TechnicalView,
    compare_sides,
    estimate_assignment,
    pick_low_assignment,
    prob_itm,
    technical_view,
)
from strategies.models import OptionContract, SupportResistanceLevel

SPOT = 100.0
WEEK = 7


def _opt(strike, kind="put", bid=1.0, ask=1.05, iv=0.40, delta=None, oi=500):
    return OptionContract(
        strike=strike, expiration="2026-11-13", option_type=kind, bid=bid, ask=ask,
        last=bid, volume=100, open_interest=oi, implied_volatility=iv, delta=delta,
    )


FLAT = TechnicalView()                                    # no drift, no RV


# ── probability model ───────────────────────────

def test_prob_itm_basics():
    years = WEEK / 365
    assert prob_itm(100, 100, years, 0.4, 0.0, "put") == pytest.approx(0.5, abs=0.02)
    far = prob_itm(100, 90, years, 0.4, 0.0, "put")
    assert 0 < far < 0.1
    # Upward drift: puts less likely ITM, calls more likely.
    assert prob_itm(100, 95, years, 0.4, 0.5, "put") < prob_itm(100, 95, years, 0.4, 0.0, "put")
    assert prob_itm(100, 105, years, 0.4, 0.5, "call") > prob_itm(100, 105, years, 0.4, 0.0, "call")


def test_estimate_matches_market_with_no_chart_signal():
    est = estimate_assignment(_opt(95), SPOT, WEEK, FLAT)
    assert est.prob == pytest.approx(est.market_prob)
    assert est.drivers == []
    assert est.touch_prob == pytest.approx(min(1.0, 2 * est.prob))


def test_no_iv_means_no_estimate():
    assert estimate_assignment(_opt(95, iv=None), SPOT, WEEK, FLAT) is None


def test_realised_vol_below_iv_lowers_the_odds_except_through_earnings():
    calm = TechnicalView(realized_vol=0.20)
    est = estimate_assignment(_opt(95, iv=0.40), SPOT, WEEK, calm)
    assert est.prob < est.market_prob
    assert "moves less than IV implies" in est.drivers[0][0]
    through = estimate_assignment(_opt(95, iv=0.40), SPOT, WEEK, calm, earnings_in_window=True)
    assert through.prob == pytest.approx(through.market_prob)
    assert "earnings" in through.drivers[0][0]


def test_drift_drivers_explain_the_change():
    bull = TechnicalView(drift=0.30, parts=[("uptrend", 0.20), ("MACD rising", 0.10)])
    est = estimate_assignment(_opt(95), SPOT, WEEK, bull)
    assert est.prob < est.market_prob                       # uptrend helps a put
    assert est.market_prob + sum(d for _, d in est.drivers) == pytest.approx(est.prob)
    labels = [l for l, _ in est.drivers]
    assert labels == ["uptrend", "MACD rising"]


def test_support_between_price_and_strike_is_a_cushion():
    sup = [SupportResistanceLevel(price=97.0, level_type="support", strength=0.8, touches=4)]
    plain = estimate_assignment(_opt(95), SPOT, WEEK, FLAT)
    cushioned = estimate_assignment(_opt(95), SPOT, WEEK, FLAT, supports=sup)
    assert cushioned.prob == pytest.approx(plain.prob * (1 - 0.25 * 0.8))
    assert cushioned.cushion == 97.0
    # A level below the strike is no cushion.
    low = [SupportResistanceLevel(price=93.0, level_type="support", strength=0.9, touches=5)]
    assert estimate_assignment(_opt(95), SPOT, WEEK, FLAT, supports=low).cushion is None


# ── technical view from real indicator series ───

def test_technical_view_reads_trend(bullish_df, bearish_df):
    bull = technical_view(calculate_full_indicators(bullish_df), float(bullish_df["Close"].iloc[-1]))
    bear = technical_view(calculate_full_indicators(bearish_df), float(bearish_df["Close"].iloc[-1]))
    assert bull.drift > 0 and bull.bias == "bullish"
    assert bear.drift < 0 and bear.bias == "bearish"
    for v in (bull, bear):
        assert abs(v.drift) <= MAX_DRIFT
        assert v.realized_vol > 0
        assert v.reasons


def test_technical_view_handles_missing_data():
    assert technical_view(None, 100).drift == 0.0


# ── picks ───────────────────────────────────────

def _ladder():
    # Puts: closer = more premium and more risk.
    return [
        _opt(99, bid=2.40, ask=2.50), _opt(97, bid=1.50, ask=1.56),
        _opt(95, bid=0.90, ask=0.94), _opt(92, bid=0.40, ask=0.42),
        _opt(90, bid=0.20, ask=0.21), _opt(102, bid=3.0, ask=3.1),   # ITM — excluded
        _opt(88, bid=0.10, ask=0.30),                                # spread too wide
    ]


def test_picks_most_premium_under_the_limit():
    side = pick_low_assignment(_ladder(), "CSP", SPOT, WEEK, FLAT, max_prob=0.15)
    assert side.picks
    assert all(p.estimate.prob <= 0.15 for p in side.picks)
    assert [p.credit for p in side.picks] == sorted((p.credit for p in side.picks), reverse=True)
    strikes = [p.strike for p in side.picks]
    assert 102 not in strikes and 88 not in strikes and 99 not in strikes


def test_a_tighter_limit_picks_a_safer_strike():
    loose = pick_low_assignment(_ladder(), "CSP", SPOT, WEEK, FLAT, max_prob=0.25).best
    tight = pick_low_assignment(_ladder(), "CSP", SPOT, WEEK, FLAT, max_prob=0.05).best
    assert tight.strike < loose.strike
    assert tight.estimate.prob < loose.estimate.prob


def test_desired_buy_price_and_cost_basis_are_respected():
    csp = pick_low_assignment(_ladder(), "CSP", SPOT, WEEK, FLAT, max_prob=0.5,
                              desired_buy_price=93)
    assert all(p.strike <= 93 for p in csp.picks)
    calls = [_opt(k, "call", bid=b, ask=b + 0.04) for k, b in ((103, 1.2), (106, 0.6), (110, 0.2))]
    cc = pick_low_assignment(calls, "CC", SPOT, WEEK, FLAT, max_prob=0.5, cost_basis=105)
    assert all(p.strike >= 105 for p in cc.picks)
    assert pick_low_assignment(calls, "CC", SPOT, WEEK, FLAT, max_prob=0.001).excluded_reason


# ── CC vs CSP verdict ───────────────────────────

def _pick(side, strike, yield_pct, prob=0.08):
    est = AssignmentEstimate(prob=prob, market_prob=prob, delta=0.1)
    return SafePick(contract=_opt(strike, "put" if side == "CSP" else "call"),
                    side=side, credit=1.0, yield_pct=yield_pct, estimate=est)


def test_verdict_prefers_the_side_that_pays_more():
    csp = SidePicks("CSP", [_pick("CSP", 95, 0.012)])
    cc = SidePicks("CC", [_pick("CC", 105, 0.006)])
    assert compare_sides(csp, cc, FLAT).side == "CSP"


def test_chart_breaks_a_near_tie():
    csp = SidePicks("CSP", [_pick("CSP", 95, 0.010)])
    cc = SidePicks("CC", [_pick("CC", 105, 0.0105)])
    bull = TechnicalView(drift=0.3, bias="bullish", parts=[("uptrend", 0.3)])
    bear = TechnicalView(drift=-0.3, bias="bearish", parts=[("downtrend", -0.3)])
    assert compare_sides(csp, cc, bull).side == "CSP"
    assert compare_sides(csp, cc, bear).side == "CC"


def test_verdict_when_one_or_no_side_qualifies():
    assert compare_sides(SidePicks("CSP"), SidePicks("CC", [_pick("CC", 105, 0.01)]), FLAT).side == "CC"
    assert compare_sides(SidePicks("CSP"), SidePicks("CC"), FLAT).side is None
