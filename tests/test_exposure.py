"""Tests for portfolio/exposure.py — capital committed vs limits."""

from __future__ import annotations

from datetime import date

from portfolio.exposure import compute_exposure, max_new_csp_contracts
from portfolio.ledger import build_ledger
from portfolio.models import Trade, TradeAction

EXP = date(2026, 10, 23)


def _positions():
    ledger = build_ledger([
        Trade(date(2026, 10, 1), "TSLA", TradeAction.SELL_TO_OPEN, 1, 3.0,
              option_type="put", strike=240, expiration=EXP),
        Trade(date(2026, 10, 1), "AMD", TradeAction.BUY_SHARES, 100, 150.0),
    ])
    return list(ledger.positions.values())


def test_totals_and_rows_sorted_by_size():
    report = compute_exposure(_positions(), account_size=100_000)
    assert [r.ticker for r in report.rows] == ["TSLA", "AMD"]
    assert report.total == 24_000 + 15_000
    assert report.pct_of_account == 0.39
    assert report.free_cash == 61_000
    assert report.warnings == ["TSLA is 24% of the account (limit 20%)."]


def test_total_limit_warning():
    report = compute_exposure(_positions(), account_size=40_000, max_ticker_pct=1.0)
    assert len(report.warnings) == 1
    assert "98% of the account is committed" in report.warnings[0]


def test_no_account_size_means_no_limit_checks():
    report = compute_exposure(_positions(), account_size=0)
    assert report.warnings == []
    assert report.pct_of_account == 0.0


def test_max_new_csp_contracts_respects_both_limits():
    report = compute_exposure(_positions(), account_size=100_000)
    # Per-ticker room for a new name: 20k → 1 contract at $150.
    assert max_new_csp_contracts(report, "NVDA", 150) == 1
    # TSLA is already over its 20% limit.
    assert max_new_csp_contracts(report, "TSLA", 240) == 0
    # Loose per-ticker cap: total room = 80k − 39k = 41k → 2 at $200.
    assert max_new_csp_contracts(report, "NVDA", 200, max_ticker_pct=1.0) == 2
    assert max_new_csp_contracts(compute_exposure([], 0), "NVDA", 100) == 0
