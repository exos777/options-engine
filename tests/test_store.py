"""Tests for data/store.py — preferences, trades, IV history, URL handling."""

from __future__ import annotations

from datetime import date

from data import store
from portfolio.models import Trade, TradeAction


def test_pref_round_trip_and_default():
    assert store.get_pref("missing", default=[1]) == [1]
    store.set_pref("watch", ["AMD", "TSLA"])
    assert store.get_pref("watch") == ["AMD", "TSLA"]
    store.set_pref("watch", ["NVDA"])          # upsert overwrites
    assert store.get_pref("watch") == ["NVDA"]


def test_trades_add_list_delete():
    t1 = store.add_trade(Trade(
        trade_date=date(2026, 10, 1), ticker="tsla ",
        action=TradeAction.SELL_TO_OPEN, quantity=1, price=3.10,
        option_type="put", strike=240.0, expiration=date(2026, 10, 9),
        fees=0.65,
    ))
    store.add_trade(Trade(
        trade_date=date(2026, 9, 1), ticker="AMD",
        action=TradeAction.BUY_SHARES, quantity=100, price=150.0,
    ))
    trades = store.list_trades()
    assert [t.ticker for t in trades] == ["AMD", "TSLA"]   # date order
    tsla = store.list_trades("tsla")[0]
    assert tsla.id == t1
    assert tsla.action == TradeAction.SELL_TO_OPEN
    assert tsla.expiration == date(2026, 10, 9)
    assert tsla.fees == 0.65

    store.delete_trade(t1)
    assert [t.ticker for t in store.list_trades()] == ["AMD"]


def test_iv_history_upserts_one_reading_per_day():
    store.record_iv("spy", date(2026, 10, 1), 0.20)
    store.record_iv("SPY", date(2026, 10, 1), 0.22)    # same day → replaced
    store.record_iv("SPY", date(2026, 10, 2), 0.25)
    store.record_iv("SPY", date(2026, 10, 3), 0.0)     # ignored
    assert store.iv_history("SPY", date(2026, 1, 1)) == [
        (date(2026, 10, 1), 0.22),
        (date(2026, 10, 2), 0.25),
    ]
    assert store.iv_history("SPY", date(2026, 10, 2)) == [(date(2026, 10, 2), 0.25)]


def test_database_url_normalises_railway_postgres(monkeypatch):
    monkeypatch.delenv("OPTIONS_DB_URL")
    monkeypatch.setenv("DATABASE_URL", "postgres://u:p@h:5432/railway")
    assert store.database_url() == "postgresql+psycopg://u:p@h:5432/railway"
    assert store.backend_name() == "Postgres"
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@h/db")
    assert store.database_url() == "postgresql+psycopg://u:p@h/db"


def test_database_url_defaults_to_local_sqlite(monkeypatch):
    monkeypatch.delenv("OPTIONS_DB_URL")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert store.database_url().endswith("options_engine.db")
    assert store.backend_name() == "SQLite"
