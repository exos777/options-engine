"""Tests for portfolio/alerts.py — exit signals on open shorts."""

from __future__ import annotations

from datetime import date, time

from portfolio.alerts import position_alerts, sort_alerts
from portfolio.ledger import OpenShort

TODAY = date(2026, 10, 15)


def _short(kind="put", strike=100.0, exp=date(2026, 10, 23), price=2.00):
    return OpenShort(
        ticker="TSLA", option_type=kind, strike=strike, expiration=exp,
        contracts=1, avg_open_price=price, opened_on=date(2026, 10, 10),
    )


def _kinds(alerts):
    return {a.kind for a in alerts}


def test_quiet_position_has_no_alerts():
    alerts = position_alerts(_short(), spot=110, mark=1.50, atr=3, today=TODAY)
    assert alerts == []


def test_profit_take_at_target():
    alerts = position_alerts(_short(price=2.00), spot=110, mark=0.90, today=TODAY)
    assert _kinds(alerts) == {"profit_take"}
    assert "55%" in alerts[0].message


def test_profit_target_is_configurable():
    s = _short(price=2.00)
    assert position_alerts(s, mark=1.20, today=TODAY, profit_target=0.50) == []
    assert _kinds(position_alerts(s, mark=1.20, today=TODAY, profit_target=0.40)) == {"profit_take"}


def test_strike_tested_near_expiry_within_one_atr():
    s = _short(strike=100, exp=date(2026, 10, 16))     # 1 DTE
    alerts = position_alerts(s, spot=102, atr=3, today=TODAY)
    assert _kinds(alerts) == {"strike_tested"}
    assert alerts[0].severity == "action"


def test_not_tested_with_plenty_of_time_or_room():
    far_dated = _short(strike=100, exp=date(2026, 10, 30))
    assert position_alerts(far_dated, spot=102, atr=3, today=TODAY) == []
    near_expiry = _short(strike=100, exp=date(2026, 10, 16))
    assert position_alerts(near_expiry, spot=110, atr=3, today=TODAY) == []


def test_itm_with_time_left_is_a_warning_not_tested():
    alerts = position_alerts(_short(kind="put", strike=100), spot=95, atr=3, today=TODAY)
    assert _kinds(alerts) == {"itm"}
    alerts = position_alerts(_short(kind="call", strike=100), spot=105, atr=3, today=TODAY)
    assert _kinds(alerts) == {"itm"}


def test_itm_near_expiry_reports_tested_only():
    s = _short(kind="call", strike=100, exp=date(2026, 10, 16))
    alerts = position_alerts(s, spot=108, atr=3, today=TODAY)
    assert _kinds(alerts) == {"strike_tested"}


def test_expired_but_still_open():
    alerts = position_alerts(_short(exp=date(2026, 10, 10)), spot=110, today=TODAY)
    assert _kinds(alerts) == {"expired"}


def test_earnings_before_expiration():
    alerts = position_alerts(_short(), earnings_date=date(2026, 10, 20), today=TODAY)
    assert _kinds(alerts) == {"earnings"}
    assert position_alerts(_short(), earnings_date=date(2026, 11, 1), today=TODAY) == []


def test_call_below_adjusted_basis():
    s = _short(kind="call", strike=95)
    alerts = position_alerts(s, spot=90, adjusted_basis=97.50, today=TODAY)
    assert _kinds(alerts) == {"below_basis"}
    assert position_alerts(_short(kind="call", strike=100), spot=90,
                           adjusted_basis=97.50, today=TODAY) == []


def test_sort_puts_actions_first():
    alerts = (
        position_alerts(_short(), earnings_date=date(2026, 10, 20), today=TODAY)
        + position_alerts(_short(price=2.0), mark=0.5, today=TODAY)
    )
    assert [a.severity for a in sort_alerts(alerts)] == ["action", "warning"]


def test_day_trade_alert_late_in_the_session():
    opened_today = OpenShort(
        ticker="TSLA", option_type="put", strike=100.0, expiration=date(2026, 10, 23),
        contracts=1, avg_open_price=2.0, opened_on=TODAY,
    )
    late = position_alerts(opened_today, today=TODAY, now_time=time(15, 35))
    assert _kinds(late) == {"day_trade"}
    assert late[0].severity == "action"
    assert position_alerts(opened_today, today=TODAY, now_time=time(11, 0)) == []
    # Opened on an earlier day — not a same-day trade.
    assert position_alerts(_short(), today=TODAY, now_time=time(15, 35)) == []
