"""Tests for ui/scanner.py — pure scan/render logic (no network)."""

from datetime import date, timedelta

import pandas as pd
import pytest

from strategies.models import ChartRegime
from ui.scanner import (
    DEFAULT_WATCHLIST,
    ScanRow,
    _COLOR_MUTED,
    _COLOR_OK,
    _COLOR_STRONG,
    _COLOR_WEAK,
    _atm_iv,
    _earnings_in_window,
    _effective_score,
    _iv_rank,
    _pick_expiration,
    _row_color,
)
from strategies.models import OptionContract


# ── _pick_expiration ────────────────────────────

def _exp(days_out: int) -> str:
    return (date.today() + timedelta(days=days_out)).isoformat()


def test_monday_picks_5_to_7_dte_window():
    exps = [_exp(1), _exp(6), _exp(14), _exp(30)]
    chosen, dte = _pick_expiration(exps, "Monday")
    assert chosen == _exp(6)
    assert dte == 6


def test_friday_picks_0_to_3_dte_window():
    exps = [_exp(1), _exp(6), _exp(14)]
    chosen, dte = _pick_expiration(exps, "Friday")
    assert chosen == _exp(1)
    assert dte == 1


def test_pick_expiration_falls_back_to_closest_when_window_empty():
    # No expiration in the 5–7 window — should snap to the nearest.
    exps = [_exp(2), _exp(15)]
    chosen, dte = _pick_expiration(exps, "Monday")
    assert chosen == _exp(2)  # closer to target 6 than 15


def test_pick_expiration_ignores_past_dates():
    exps = [_exp(-3), _exp(6)]
    chosen, _ = _pick_expiration(exps, "Monday")
    assert chosen == _exp(6)


def test_pick_expiration_raises_when_no_future_expirations():
    with pytest.raises(ValueError):
        _pick_expiration([_exp(-1), _exp(-5)], "Monday")


# ── _atm_iv ─────────────────────────────────────

def _opt(strike: float, iv: float | None) -> OptionContract:
    return OptionContract(
        strike=strike, expiration="2026-01-01", option_type="put",
        bid=1.0, ask=1.1, last=1.05, volume=10, open_interest=100,
        implied_volatility=iv,
    )


def test_atm_iv_averages_near_the_money_strikes_only():
    contracts = [
        _opt(100, 0.50),   # ATM — counted
        _opt(102, 0.60),   # within 5% — counted
        _opt(130, 0.90),   # far OTM — excluded
    ]
    assert _atm_iv(contracts, spot=100.0) == pytest.approx(0.55)


def test_atm_iv_returns_zero_when_no_iv_data():
    assert _atm_iv([_opt(100, None)], spot=100.0) == 0.0


# ── _iv_rank ────────────────────────────────────

def test_iv_rank_high_when_iv_above_realised_range():
    # Flat-ish prices → low realised vol → a 60% IV ranks near the top.
    close = pd.Series([100 + (i % 3) * 0.1 for i in range(60)])
    df = pd.DataFrame({"Close": close})
    assert _iv_rank(df, atm_iv=0.60) >= 80.0


def test_iv_rank_neutral_when_history_too_short():
    df = pd.DataFrame({"Close": [100, 101, 102]})
    assert _iv_rank(df, atm_iv=0.40) == 50.0


def test_iv_rank_neutral_when_no_iv():
    df = pd.DataFrame({"Close": list(range(100, 160))})
    assert _iv_rank(df, atm_iv=0.0) == 50.0


# ── _earnings_in_window ─────────────────────────

def test_earnings_in_window_true_when_before_expiration():
    assert _earnings_in_window(_exp(3), _exp(7)) is True


def test_earnings_in_window_false_when_after_expiration():
    assert _earnings_in_window(_exp(10), _exp(7)) is False


def test_earnings_in_window_false_when_unknown():
    assert _earnings_in_window(None, _exp(7)) is False


# ── _effective_score ────────────────────────────

def test_effective_score_respects_strategy_filter():
    row = ScanRow(ticker="X", best_csp_score=72.0, best_cc_score=55.0)
    assert _effective_score(row, "CSP only") == 72.0
    assert _effective_score(row, "CC only") == 55.0
    assert _effective_score(row, "Both") == 72.0


# ── _row_color ──────────────────────────────────

def test_row_color_buckets_by_score():
    assert _row_color(ScanRow(ticker="X"), 80.0) == _COLOR_STRONG
    assert _row_color(ScanRow(ticker="X"), 60.0) == _COLOR_OK
    assert _row_color(ScanRow(ticker="X"), 30.0) == _COLOR_WEAK


def test_row_color_muted_for_no_trade_and_earnings():
    no_trade = ScanRow(ticker="X", has_no_trade=True, no_trade_reason="x")
    earnings = ScanRow(ticker="X", earnings_warning=True)
    errored = ScanRow(ticker="X", error="boom")
    assert _row_color(no_trade, 90.0) == _COLOR_MUTED
    assert _row_color(earnings, 90.0) == _COLOR_MUTED
    assert _row_color(errored, 90.0) == _COLOR_MUTED


# ── watchlist default ───────────────────────────

def test_default_watchlist_has_ten_tickers():
    assert len(DEFAULT_WATCHLIST) == 10
    assert "TSLA" in DEFAULT_WATCHLIST
