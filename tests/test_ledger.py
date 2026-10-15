"""Tests for portfolio/ledger.py — wheel cycle accounting."""

from __future__ import annotations

from datetime import date

import pytest

from portfolio.ledger import build_ledger
from portfolio.models import Trade, TradeAction

EXP1 = date(2026, 10, 9)
EXP2 = date(2026, 10, 16)
EXP3 = date(2026, 10, 23)


def _opt(d, action, kind, strike, exp, price=0.0, qty=1, fees=0.0, ticker="TSLA", tid=None):
    return Trade(
        trade_date=d, ticker=ticker, action=action, quantity=qty, price=price,
        option_type=kind, strike=strike, expiration=exp, fees=fees, id=tid,
    )


def test_cash_flow_signs():
    sto = _opt(date(2026, 10, 1), TradeAction.SELL_TO_OPEN, "put", 240, EXP1, 3.10, qty=2, fees=1.30)
    assert sto.cash_flow == pytest.approx(620 - 1.30)
    btc = _opt(date(2026, 10, 1), TradeAction.BUY_TO_CLOSE, "put", 240, EXP1, 1.00)
    assert btc.cash_flow == pytest.approx(-100)
    put_assign = _opt(date(2026, 10, 9), TradeAction.ASSIGNED, "put", 240, EXP1)
    assert put_assign.cash_flow == pytest.approx(-24_000)
    call_away = _opt(date(2026, 10, 9), TradeAction.ASSIGNED, "call", 250, EXP1)
    assert call_away.cash_flow == pytest.approx(25_000)


def test_open_csp_is_a_csp_phase_position():
    ledger = build_ledger([
        _opt(date(2026, 10, 1), TradeAction.SELL_TO_OPEN, "put", 240, EXP1, 3.00, qty=2),
    ])
    pos = ledger.positions["TSLA"]
    assert pos.phase == "CSP"
    assert pos.net_premium == pytest.approx(600)
    assert pos.csp_collateral == pytest.approx(48_000)
    assert pos.capital_at_risk == pytest.approx(48_000)
    assert pos.adjusted_basis is None
    assert len(ledger.open_shorts) == 1
    assert ledger.closed == []


def test_expired_put_closes_the_cycle_as_pure_premium():
    ledger = build_ledger([
        _opt(date(2026, 10, 1), TradeAction.SELL_TO_OPEN, "put", 240, EXP1, 3.00, fees=0.65),
        _opt(date(2026, 10, 9), TradeAction.EXPIRED, "put", 240, EXP1),
    ])
    assert ledger.positions == {}
    [cycle] = ledger.closed
    assert cycle.realized_pnl == pytest.approx(299.35)
    assert cycle.max_capital == pytest.approx(24_000)
    assert cycle.days == 8
    assert cycle.return_pct == pytest.approx(299.35 / 24_000)
    assert cycle.annualized_pct == pytest.approx(299.35 / 24_000 * 365 / 8)


def test_full_wheel_cycle_tracks_adjusted_basis_and_pnl():
    trades = [
        _opt(date(2026, 10, 1), TradeAction.SELL_TO_OPEN, "put", 240, EXP1, 3.00),
        _opt(date(2026, 10, 9), TradeAction.ASSIGNED, "put", 240, EXP1),
        _opt(date(2026, 10, 10), TradeAction.SELL_TO_OPEN, "call", 245, EXP2, 2.00),
    ]
    pos = build_ledger(trades).positions["TSLA"]
    assert pos.phase == "CC"
    assert pos.shares == 100
    assert pos.avg_share_cost == pytest.approx(240)
    assert pos.net_premium == pytest.approx(500)
    assert pos.adjusted_basis == pytest.approx(235)     # 240 − 5.00 premium
    assert pos.capital_at_risk == pytest.approx(24_000)  # calls need no cash

    trades.append(_opt(date(2026, 10, 16), TradeAction.ASSIGNED, "call", 245, EXP2))
    ledger = build_ledger(trades)
    assert ledger.positions == {}
    [cycle] = ledger.closed
    assert cycle.net_premium == pytest.approx(500)
    assert cycle.share_pnl == pytest.approx(500)       # bought 240, sold 245
    assert cycle.realized_pnl == pytest.approx(1_000)
    assert cycle.started == date(2026, 10, 1)
    assert cycle.ended == date(2026, 10, 16)


def test_roll_keeps_cycle_open_and_nets_the_debit():
    ledger = build_ledger([
        _opt(date(2026, 10, 1), TradeAction.SELL_TO_OPEN, "put", 240, EXP1, 3.00),
        _opt(date(2026, 10, 8), TradeAction.BUY_TO_CLOSE, "put", 240, EXP1, 4.00),
        _opt(date(2026, 10, 8), TradeAction.SELL_TO_OPEN, "put", 235, EXP2, 4.50),
    ])
    pos = ledger.positions["TSLA"]
    assert ledger.closed == []
    assert pos.net_premium == pytest.approx(300 - 400 + 450)
    [short] = pos.open_shorts
    assert (short.strike, short.expiration) == (235, EXP2)


def test_same_date_trades_replay_in_entry_order():
    # BTC and STO of a roll share a date; ids keep the close before the open
    # so the cycle never looks flat in between.
    ledger = build_ledger([
        _opt(date(2026, 10, 8), TradeAction.SELL_TO_OPEN, "put", 235, EXP2, 4.50, tid=3),
        _opt(date(2026, 10, 8), TradeAction.BUY_TO_CLOSE, "put", 240, EXP1, 4.00, tid=2),
        _opt(date(2026, 10, 1), TradeAction.SELL_TO_OPEN, "put", 240, EXP1, 3.00, tid=1),
    ])
    assert ledger.closed == []
    assert ledger.positions["TSLA"].open_shorts[0].strike == 235


def test_adding_to_a_short_averages_the_open_price():
    ledger = build_ledger([
        _opt(date(2026, 10, 1), TradeAction.SELL_TO_OPEN, "put", 240, EXP1, 3.00, qty=1),
        _opt(date(2026, 10, 2), TradeAction.SELL_TO_OPEN, "put", 240, EXP1, 2.00, qty=3),
    ])
    [short] = ledger.open_shorts
    assert short.contracts == 4
    assert short.avg_open_price == pytest.approx((3.00 + 6.00) / 4)
    assert short.opened_on == date(2026, 10, 1)


def test_partial_close_leaves_remaining_contracts_open():
    ledger = build_ledger([
        _opt(date(2026, 10, 1), TradeAction.SELL_TO_OPEN, "put", 240, EXP1, 3.00, qty=3),
        _opt(date(2026, 10, 3), TradeAction.BUY_TO_CLOSE, "put", 240, EXP1, 1.00, qty=2),
    ])
    [short] = ledger.open_shorts
    assert short.contracts == 1
    assert ledger.positions["TSLA"].csp_collateral == pytest.approx(24_000)


def test_unmatched_close_warns_and_records_no_cycle():
    ledger = build_ledger([
        _opt(date(2026, 10, 9), TradeAction.EXPIRED, "put", 240, EXP1),
    ])
    assert ledger.closed == []
    assert ledger.positions == {}
    assert any("no matching open contract" in w for w in ledger.warnings)


def test_monthly_premium_and_totals_across_tickers():
    ledger = build_ledger([
        _opt(date(2026, 9, 25), TradeAction.SELL_TO_OPEN, "put", 240, EXP1, 3.00),
        _opt(date(2026, 10, 1), TradeAction.SELL_TO_OPEN, "put", 100, EXP2, 1.50, ticker="AMD"),
        _opt(date(2026, 10, 5), TradeAction.BUY_TO_CLOSE, "put", 240, EXP1, 0.50),
    ])
    assert ledger.monthly_premium == {
        "2026-09": pytest.approx(300),
        "2026-10": pytest.approx(150 - 50),
    }
    assert ledger.total_premium == pytest.approx(400)
    assert ledger.realized_pnl == pytest.approx(250)       # TSLA cycle closed
    assert list(ledger.positions) == ["AMD"]
    assert ledger.capital_at_risk == pytest.approx(10_000)


def test_manual_share_trades_use_average_cost():
    ledger = build_ledger([
        Trade(date(2026, 10, 1), "AAPL", TradeAction.BUY_SHARES, 100, 200.0),
        Trade(date(2026, 10, 2), "AAPL", TradeAction.BUY_SHARES, 100, 220.0),
        Trade(date(2026, 10, 3), "AAPL", TradeAction.SELL_SHARES, 100, 230.0),
    ])
    pos = ledger.positions["AAPL"]
    assert pos.shares == 100
    assert pos.avg_share_cost == pytest.approx(210)
    assert pos.phase == "CC"
