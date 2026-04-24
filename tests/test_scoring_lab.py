"""Tests for the scoring comparison lab."""

import pandas as pd
import pytest

from scoring_lab.current_scorer import current_score, current_score_all
from scoring_lab.data import Snapshot, StrikeData
from scoring_lab.new_scorer import (
    DELTA_PENALTY,
    DISTANCE_WEIGHT,
    LARGE_PENALTY,
    PREMIUM_WEIGHT,
    SPREAD_PENALTY,
    THETA_WEIGHT,
    new_score,
    new_score_all,
)
from scoring_lab.outcomes import DOWNSIDE_SHOCK_PCT, simulate
from scoring_lab.report import build_ticker_report, run_comparison
from scoring_lab.snapshots import all_snapshots


# ── Fixtures ────────────────────────────────────

def make_strike(**kw) -> StrikeData:
    defaults = dict(
        strike=360.0,
        premium=3.00,
        delta=-0.25,
        theta=-0.08,
        vega=0.15,
        iv=0.40,
        open_interest=500,
        spread_pct=0.05,
        dist_pct=0.05,
        break_even=357.0,
        underlying_price=380.0,
        dte=10,
    )
    defaults.update(kw)
    return StrikeData(**defaults)


def make_snapshot(strikes: list[StrikeData]) -> Snapshot:
    return Snapshot(
        ticker="TEST",
        underlying_price=380.0,
        desired_buy_price=360.0,
        dte=10,
        strikes=strikes,
    )


# ── new_score formula ────────────────────────────

def test_new_score_matches_formula_below_desired():
    s = make_strike(strike=355.0, premium=3.00, break_even=352.0)
    desired = 360.0
    expected = (
        PREMIUM_WEIGHT * 3.00
        + THETA_WEIGHT * 0.08
        + DISTANCE_WEIGHT * 0.05
        - DELTA_PENALTY * 0.25
        - SPREAD_PENALTY * 0.05
    )
    assert new_score(s, desired) == pytest.approx(expected)


def test_new_score_applies_large_penalty_above_desired():
    above = make_strike(strike=370.0, break_even=367.0)
    below = make_strike(strike=355.0, break_even=352.0)
    s_above = new_score(above, desired_buy_price=360.0)
    s_below = new_score(below, desired_buy_price=360.0)
    assert s_below - s_above >= LARGE_PENALTY * 0.99


def test_new_score_all_sorts_descending():
    strikes = [
        make_strike(strike=350.0, premium=2.0, break_even=348.0),
        make_strike(strike=358.0, premium=4.0, break_even=354.0),
        make_strike(strike=370.0, premium=6.0, break_even=364.0),  # above desired
    ]
    ranked = new_score_all(strikes, desired_buy_price=360.0)
    scores = [sc for _, sc in ranked]
    assert scores == sorted(scores, reverse=True)
    # Strike above desired should be last (LARGE_PENALTY)
    assert ranked[-1][0].strike == 370.0


# ── current_score wrapper ────────────────────────

def test_current_score_all_returns_scored_options():
    snaps = all_snapshots()
    scored = current_score_all(snaps[0])
    assert len(scored) > 0
    for s in scored:
        assert 0.0 <= s.score <= 100.0


def test_current_score_matches_one_strike():
    snap = all_snapshots()[0]
    target = snap.strikes[len(snap.strikes) // 2]
    score = current_score(target, snap)
    assert score > 0.0


# ── snapshots ────────────────────────────────────

def test_all_snapshots_has_four_tickers():
    snaps = all_snapshots()
    tickers = [s.ticker for s in snaps]
    assert set(tickers) == {"TSLA", "NVDA", "AAPL", "SPY"}


def test_snapshot_strikes_are_plausible():
    for snap in all_snapshots():
        assert len(snap.strikes) >= 10
        for s in snap.strikes:
            assert s.premium > 0
            assert -1.0 <= s.delta <= 0.0  # puts have non-positive delta
            assert s.theta <= 0
            assert 0.0 <= s.spread_pct <= 1.0
            assert s.open_interest > 0


# ── outcomes ─────────────────────────────────────

def test_simulate_expires_equals_premium():
    s = make_strike(premium=2.50)
    o = simulate(s, desired_buy_price=360.0)
    assert o.pnl_expires == pytest.approx(2.50)


def test_simulate_downside_shock_can_be_negative():
    s = make_strike(strike=380.0, premium=5.00, underlying_price=400.0)
    # shocked spot = 360 -> assigned loss = 20 -> pnl = 5 - 20 = -15
    o = simulate(s, desired_buy_price=360.0)
    assert o.pnl_downside_shock == pytest.approx(
        5.00 - (380.0 - 400.0 * (1 - DOWNSIDE_SHOCK_PCT))
    )
    assert o.downside_risk > 0


# ── report ──────────────────────────────────────

def test_build_ticker_report_returns_dataframes():
    snap = all_snapshots()[0]
    r = build_ticker_report(snap)
    assert isinstance(r.picks_table, pd.DataFrame)
    assert isinstance(r.summary_table, pd.DataFrame)
    assert r.winner in {"NEW", "CURRENT", "TIE"}
    assert "Engine" in r.picks_table.columns
    assert "Avg Premium" in r.summary_table.columns


def test_run_comparison_covers_all_tickers(capsys):
    reports = run_comparison()
    assert len(reports) == 4
    captured = capsys.readouterr()
    assert "TSLA" in captured.out
    assert "Overall" in captured.out
