"""
Persistent storage — survives app restarts and Railway redeploys.

Backend selection (first match wins):
    OPTIONS_DB_URL   explicit SQLAlchemy URL (tests point this at a temp file)
    DATABASE_URL     set automatically by Railway's Postgres plugin
    (default)        local SQLite file at data/options_engine.db

Railway's container filesystem is wiped on every deploy, so anything that
must persist in production (the trade ledger, watchlist, preferences, the
IV history behind IV rank) has to live in Postgres there. Locally the
SQLite file gives the same behaviour with zero setup.

Tables
------
kv          JSON preferences keyed by name (watchlist, form state, limits)
trades      the wheel ledger — one row per fill / lifecycle event
iv_history  one ATM implied-vol reading per ticker per day
"""

from __future__ import annotations

import json
import os
import pathlib
import threading
from datetime import date, datetime
from typing import Any, Optional

from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    delete,
    select,
)
from sqlalchemy.engine import Engine

from portfolio.models import Trade, TradeAction

_DEFAULT_SQLITE = pathlib.Path(__file__).parent / "options_engine.db"

_metadata = MetaData()

kv_table = Table(
    "kv",
    _metadata,
    Column("key", String(128), primary_key=True),
    Column("value", Text, nullable=False),
)

trades_table = Table(
    "trades",
    _metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("trade_date", Date, nullable=False),
    Column("ticker", String(16), nullable=False, index=True),
    Column("action", String(24), nullable=False),
    Column("option_type", String(4), nullable=True),
    Column("strike", Float, nullable=True),
    Column("expiration", Date, nullable=True),
    Column("quantity", Integer, nullable=False),
    Column("price", Float, nullable=False),
    Column("fees", Float, nullable=False, default=0.0),
    Column("notes", Text, nullable=False, default=""),
    Column("created_at", DateTime, nullable=False),
)

iv_history_table = Table(
    "iv_history",
    _metadata,
    Column("ticker", String(16), primary_key=True),
    Column("as_of", Date, primary_key=True),
    Column("atm_iv", Float, nullable=False),
)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

_engines: dict[str, Engine] = {}
_engines_lock = threading.Lock()


def database_url() -> str:
    """Resolve the SQLAlchemy URL for the active backend."""
    url = os.getenv("OPTIONS_DB_URL") or os.getenv("DATABASE_URL") or ""
    url = url.strip()
    if not url:
        return f"sqlite:///{_DEFAULT_SQLITE}"
    # Railway / Heroku hand out postgres:// — SQLAlchemy needs the driver.
    if url.startswith("postgres://"):
        url = "postgresql+psycopg://" + url[len("postgres://"):]
    elif url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://"):]
    return url


def backend_name() -> str:
    """'Postgres' or 'SQLite' — for the UI's storage badge."""
    return "Postgres" if database_url().startswith("postgresql") else "SQLite"


def get_engine() -> Engine:
    """Engine for the current URL, created (with tables) on first use."""
    url = database_url()
    with _engines_lock:
        engine = _engines.get(url)
        if engine is None:
            kwargs: dict[str, Any] = {"future": True, "pool_pre_ping": True}
            if url.startswith("sqlite"):
                # The scanner records IV from worker threads.
                kwargs["connect_args"] = {"check_same_thread": False}
            engine = create_engine(url, **kwargs)
            _metadata.create_all(engine)
            _engines[url] = engine
    return engine


def _upsert(conn, table: Table, values: dict, keys: list[str]) -> None:
    """INSERT … ON CONFLICT DO UPDATE, for both SQLite and Postgres."""
    if conn.dialect.name == "postgresql":
        from sqlalchemy.dialects.postgresql import insert
    else:
        from sqlalchemy.dialects.sqlite import insert
    stmt = insert(table).values(**values)
    stmt = stmt.on_conflict_do_update(
        index_elements=keys,
        set_={k: stmt.excluded[k] for k in values if k not in keys},
    )
    conn.execute(stmt)


# ---------------------------------------------------------------------------
# Preferences (JSON key/value)
# ---------------------------------------------------------------------------

def get_pref(key: str, default: Any = None) -> Any:
    """Return the JSON-decoded preference, or *default* if unset."""
    with get_engine().connect() as conn:
        raw = conn.execute(
            select(kv_table.c.value).where(kv_table.c.key == key)
        ).scalar_one_or_none()
    if raw is None:
        return default
    try:
        return json.loads(raw)
    except ValueError:
        return default


def set_pref(key: str, value: Any) -> None:
    """Store *value* (anything JSON-serialisable) under *key*."""
    with get_engine().begin() as conn:
        _upsert(
            conn, kv_table,
            {"key": key, "value": json.dumps(value, default=str)},
            ["key"],
        )


# ---------------------------------------------------------------------------
# Trades
# ---------------------------------------------------------------------------

def _row_to_trade(row) -> Trade:
    return Trade(
        id=row.id,
        trade_date=row.trade_date,
        ticker=row.ticker,
        action=TradeAction(row.action),
        option_type=row.option_type,
        strike=row.strike,
        expiration=row.expiration,
        quantity=row.quantity,
        price=row.price,
        fees=row.fees or 0.0,
        notes=row.notes or "",
    )


def add_trade(trade: Trade) -> int:
    """Insert a trade and return its new id."""
    with get_engine().begin() as conn:
        result = conn.execute(
            trades_table.insert().values(
                trade_date=trade.trade_date,
                ticker=trade.ticker.upper().strip(),
                action=trade.action.value,
                option_type=trade.option_type,
                strike=trade.strike,
                expiration=trade.expiration,
                quantity=trade.quantity,
                price=trade.price,
                fees=trade.fees,
                notes=trade.notes,
                created_at=datetime.now(),
            )
        )
        return int(result.inserted_primary_key[0])


def list_trades(ticker: Optional[str] = None) -> list[Trade]:
    """All trades in entry order (trade date, then insertion order)."""
    stmt = select(trades_table).order_by(
        trades_table.c.trade_date, trades_table.c.id,
    )
    if ticker:
        stmt = stmt.where(trades_table.c.ticker == ticker.upper().strip())
    with get_engine().connect() as conn:
        return [_row_to_trade(r) for r in conn.execute(stmt)]


def delete_trade(trade_id: int) -> None:
    with get_engine().begin() as conn:
        conn.execute(delete(trades_table).where(trades_table.c.id == trade_id))


# ---------------------------------------------------------------------------
# IV history
# ---------------------------------------------------------------------------

def record_iv(ticker: str, as_of: date, atm_iv: float) -> None:
    """Store today's ATM IV for *ticker* (one reading per day; last wins)."""
    if atm_iv <= 0:
        return
    with get_engine().begin() as conn:
        _upsert(
            conn, iv_history_table,
            {"ticker": ticker.upper(), "as_of": as_of, "atm_iv": float(atm_iv)},
            ["ticker", "as_of"],
        )


def iv_history(ticker: str, since: date) -> list[tuple[date, float]]:
    """(date, atm_iv) readings for *ticker* on or after *since*, oldest first."""
    stmt = (
        select(iv_history_table.c.as_of, iv_history_table.c.atm_iv)
        .where(iv_history_table.c.ticker == ticker.upper())
        .where(iv_history_table.c.as_of >= since)
        .order_by(iv_history_table.c.as_of)
    )
    with get_engine().connect() as conn:
        return [(r.as_of, r.atm_iv) for r in conn.execute(stmt)]
