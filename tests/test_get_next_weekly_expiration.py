"""Tests for data.common.get_next_weekly_expiration."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from data.common import days_to_expiration, get_next_weekly_expiration


def _iso(d: date) -> str:
    return d.isoformat()


# ── Normal case: exact +7 hits a listed expiration ──

def test_exact_seven_day_match():
    base = date(2026, 5, 1)
    available = [
        _iso(date(2026, 5, 1)),
        _iso(date(2026, 5, 8)),     # +7 exact
        _iso(date(2026, 5, 15)),
    ]
    assert get_next_weekly_expiration(_iso(base), available) == _iso(date(2026, 5, 8))


# ── Snap: exact +7 missing, take next available ──

def test_snaps_to_next_available_when_exact_date_missing():
    base = date(2026, 5, 1)
    available = [
        _iso(date(2026, 5, 1)),
        _iso(date(2026, 5, 9)),     # +8 days; first one >= +7
        _iso(date(2026, 5, 15)),
    ]
    assert get_next_weekly_expiration(_iso(base), available) == _iso(date(2026, 5, 9))


def test_skips_expirations_before_target():
    base = date(2026, 5, 1)
    available = [
        _iso(date(2026, 5, 1)),
        _iso(date(2026, 5, 4)),     # too early
        _iso(date(2026, 5, 6)),     # too early
        _iso(date(2026, 5, 10)),    # +9, valid
    ]
    assert get_next_weekly_expiration(_iso(base), available) == _iso(date(2026, 5, 10))


# ── Fallback: no expiration within 14 days, take nearest ──

def test_fallback_when_no_expiration_in_window():
    base = date(2026, 5, 1)
    # Target = May 8, window = May 8..15. None inside window.
    # Available are far before/after target — should pick closest by abs distance.
    available = [
        _iso(date(2026, 5, 1)),     # 7 days before target
        _iso(date(2026, 5, 30)),    # 22 days after target
    ]
    # |May 1 - May 8| = 7, |May 30 - May 8| = 22 → May 1 wins
    assert get_next_weekly_expiration(_iso(base), available) == _iso(date(2026, 5, 1))


def test_fallback_when_only_far_future_expirations():
    base = date(2026, 5, 1)
    available = [
        _iso(date(2026, 6, 30)),
        _iso(date(2026, 7, 15)),
    ]
    # Both are far past the 14-day window; nearest to May 8 is June 30.
    assert get_next_weekly_expiration(_iso(base), available) == _iso(date(2026, 6, 30))


# ── Empty / invalid input ──

def test_empty_expirations_returns_none():
    assert get_next_weekly_expiration("2026-05-01", []) is None


def test_invalid_base_date_returns_none():
    assert get_next_weekly_expiration("not-a-date", ["2026-05-01"]) is None


def test_invalid_expirations_filtered_out():
    base = date(2026, 5, 1)
    available = ["bad-date", _iso(date(2026, 5, 8))]
    assert get_next_weekly_expiration(_iso(base), available) == _iso(date(2026, 5, 8))


# ── DTE alignment: DTE uses Leg 2 expiration ──

def test_dte_uses_leg2_expiration():
    today = date.today()
    leg1 = today + timedelta(days=3)
    available = [
        _iso(leg1),
        _iso(leg1 + timedelta(days=7)),   # Leg 2 candidate
    ]
    leg2 = get_next_weekly_expiration(_iso(leg1), available)
    assert leg2 is not None
    # The DTE the UI will display equals (Leg 2 - today).days
    assert days_to_expiration(leg2) == 10  # 3 + 7
