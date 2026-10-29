"""
Ledger tab — the wheel journal.

Sections, top to bottom:
    🔔 Alerts          exit signals on every open short (live data)
    📂 Open Positions   each ticker's in-progress wheel cycle
    🧮 Exposure         capital committed vs account size and limits
    📈 Performance      premium by month, closed cycles, annualized return
    ✍️ Log a Trade      single fills, plus a two-leg roll helper
    🗂️ Trade History    every row, with delete and CSV export
"""

from __future__ import annotations

import io
import os
from datetime import date, datetime
from typing import Optional

import pandas as pd
import streamlit as st

from data import store
from portfolio.alerts import Alert, position_alerts, sort_alerts
from portfolio.exposure import (
    DEFAULT_MAX_TICKER_PCT,
    DEFAULT_MAX_TOTAL_PCT,
    compute_exposure,
)
from portfolio.ledger import Ledger, OpenShort, WheelPosition, build_ledger
from portfolio.models import Trade, TradeAction
from ui.position_management import PM_PREFILL_KEY

_SETTINGS_KEY = "ledger.settings"

# Set by queue_trade_prefill(); applied before the trade form's widgets render.
LEDGER_PREFILL_KEY = "_ledger_trade_prefill"
_DEFAULT_SETTINGS = {
    "account_size": 0.0,
    "max_ticker_pct": DEFAULT_MAX_TICKER_PCT,
    "max_total_pct": DEFAULT_MAX_TOTAL_PCT,
    "profit_target": 0.50,
}
_LIVE_TTL_SECONDS = 300

_SEVERITY_STYLE = {
    "action": ("#3a1d1d", "#f85149", "🔴"),
    "warning": ("#3a3010", "#d29922", "🟡"),
    "info": ("#15304a", "#58a6ff", "🔵"),
}


# ---------------------------------------------------------------------------
# Settings + data loading
# ---------------------------------------------------------------------------

def load_settings() -> dict:
    try:
        saved = store.get_pref(_SETTINGS_KEY, {}) or {}
    except Exception:
        saved = {}
    return {**_DEFAULT_SETTINGS, **saved}


def load_ledger() -> tuple[list[Trade], Ledger]:
    trades = store.list_trades()
    return trades, build_ledger(trades)


def position_for(ticker: str) -> Optional[WheelPosition]:
    """The open wheel cycle for *ticker*, or None. Never raises."""
    if not ticker:
        return None
    try:
        return build_ledger(store.list_trades(ticker)).positions.get(ticker.upper())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Live data for alerts
# ---------------------------------------------------------------------------

def _atr(hist: pd.DataFrame, period: int = 14) -> Optional[float]:
    if hist is None or len(hist) < period + 1:
        return None
    high, low, close = hist["High"], hist["Low"], hist["Close"]
    tr = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()],
        axis=1,
    ).max(axis=1)
    value = tr.rolling(period).mean().iloc[-1]
    return float(value) if pd.notna(value) else None


def _parse_date(value) -> Optional[date]:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _fetch_live(dp, shorts: list[OpenShort]) -> dict:
    """Spot / ATR / earnings per ticker and mark per contract. Best-effort."""
    tickers = sorted({s.ticker for s in shorts})
    out: dict = {"tickers": {}, "marks": {}, "errors": []}
    for t in tickers:
        info: dict = {"spot": None, "atr": None, "earnings": None}
        try:
            quote = dp.get_quote(t)
            info["spot"] = quote.price
            info["earnings"] = _parse_date(getattr(quote, "earnings_date", None))
        except Exception as exc:  # noqa: BLE001
            out["errors"].append(f"{t}: quote unavailable ({exc})")
        if info["earnings"] is None and hasattr(dp, "get_earnings_date"):
            try:
                info["earnings"] = _parse_date(dp.get_earnings_date(t))
            except Exception:  # noqa: BLE001
                pass
        try:
            info["atr"] = _atr(dp.get_historical(t, months=2))
        except Exception:  # noqa: BLE001
            pass
        out["tickers"][t] = info

    for s in shorts:
        if s.dte() < 0:
            continue
        try:
            calls, puts = dp.get_option_chain(s.ticker, s.expiration.isoformat())
            chain = puts if s.option_type == "put" else calls
            match = next(
                (c for c in chain if abs(c.strike - s.strike) < 1e-6), None,
            )
            if match is not None:
                out["marks"][_short_key(s)] = match.mid
        except Exception as exc:  # noqa: BLE001
            out["errors"].append(
                f"{s.ticker} {s.strike:g} {s.expiration}: chain unavailable ({exc})"
            )
    return out


def _short_key(s: OpenShort) -> str:
    return f"{s.ticker}|{s.option_type}|{s.strike}|{s.expiration}"


def _live_data(dp, shorts: list[OpenShort], force: bool) -> Optional[dict]:
    if dp is None or not shorts:
        return None
    key = tuple(sorted(_short_key(s) for s in shorts))
    cached = st.session_state.get("ledger_live")
    fresh = (
        cached
        and cached["key"] == key
        and (datetime.now() - cached["at"]).total_seconds() < _LIVE_TTL_SECONDS
    )
    if force or not fresh:
        with st.spinner("Checking open positions against live prices…"):
            data = _fetch_live(dp, shorts)
        cached = {"key": key, "at": datetime.now(), "data": data}
        st.session_state["ledger_live"] = cached
    return cached["data"]


def collect_alerts(ledger: Ledger, live: Optional[dict], profit_target: float) -> list[Alert]:
    alerts: list[Alert] = []
    for pos in ledger.positions.values():
        for s in pos.open_shorts:
            info = (live or {}).get("tickers", {}).get(s.ticker, {})
            alerts += position_alerts(
                s,
                spot=info.get("spot"),
                mark=(live or {}).get("marks", {}).get(_short_key(s)),
                atr=info.get("atr"),
                earnings_date=info.get("earnings"),
                adjusted_basis=pos.adjusted_basis,
                profit_target=profit_target,
            )
    return sort_alerts(alerts)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def _render_storage_badge() -> None:
    backend = store.backend_name()
    on_railway = bool(os.getenv("RAILWAY_ENVIRONMENT") or os.getenv("RAILWAY_PROJECT_ID"))
    # An explicit OPTIONS_DB_URL means the SQLite file was placed on purpose
    # (e.g. on a Railway volume); only the default in-container path is lost.
    configured = bool(os.getenv("OPTIONS_DB_URL") or os.getenv("DATABASE_URL"))
    if backend == "SQLite" and on_railway and not configured:
        st.error(
            "Saving to a local SQLite file, which Railway wipes on every "
            "deploy. Either attach a Volume at `/data` and set "
            "`OPTIONS_DB_URL=sqlite:////data/options_engine.db`, or add a "
            "Postgres database and reference its `DATABASE_URL`.",
            icon="⚠️",
        )
    else:
        st.caption(f"💾 Stored in {backend} ({_storage_location()})")


def _storage_location() -> str:
    url = store.database_url()
    if url.startswith("sqlite:///"):
        return url[len("sqlite:///"):]
    return url.split("@")[-1]   # host/db only — never show credentials


def _render_alerts(alerts: list[Alert], live: Optional[dict]) -> None:
    st.markdown("#### 🔔 Alerts")
    if live is None:
        st.caption("No open option positions to watch.")
        return
    if not alerts:
        st.success("All open positions look fine — nothing needs action.", icon="✅")
    for a in alerts:
        bg, border, dot = _SEVERITY_STYLE.get(a.severity, _SEVERITY_STYLE["info"])
        st.markdown(
            f'<div style="background:{bg};border-left:4px solid {border};'
            f'border-radius:6px;padding:8px 12px;margin-bottom:6px;">'
            f"{dot} <b>{a.ticker} {a.contract}</b> — {a.message}</div>",
            unsafe_allow_html=True,
        )
    for err in live.get("errors", []):
        st.caption(f"⚠️ {err}")


def _manage_prefill(pos: WheelPosition, s: OpenShort) -> dict:
    prefill = {
        "pm_ticker": s.ticker,
        "pm_strategy": "CSP" if s.option_type == "put" else "CC",
        "pm_strike": float(s.strike),
        "pm_premium": round(float(s.avg_open_price), 2),
        "pm_leg1_sel": s.expiration.isoformat(),
        "pm_leg1_txt": s.expiration.isoformat(),
    }
    if pos.adjusted_basis is not None:
        prefill["pm_cost_basis"] = round(float(pos.adjusted_basis), 2)
    return prefill


def _render_positions(ledger: Ledger, live: Optional[dict]) -> None:
    st.markdown("#### 📂 Open Positions")
    if not ledger.positions:
        st.info("No open wheel positions. Log a trade below to start.", icon="✍️")
        return

    for pos in sorted(ledger.positions.values(), key=lambda p: p.ticker):
        spot = (live or {}).get("tickers", {}).get(pos.ticker, {}).get("spot")
        with st.container(border=True):
            c = st.columns(6)
            c[0].metric(pos.ticker, f"{pos.phase} phase", f"since {pos.started:%b %d}", delta_color="off")
            c[1].metric("Shares", f"{pos.shares:,}")
            c[2].metric(
                "Avg cost", f"${pos.avg_share_cost:,.2f}" if pos.avg_share_cost else "—",
            )
            c[3].metric("Premium this cycle", f"${pos.net_premium:,.2f}")
            c[4].metric(
                "Adjusted basis",
                f"${pos.adjusted_basis:,.2f}" if pos.adjusted_basis is not None else "—",
                (f"spot ${spot:,.2f}" if spot else None),
                delta_color="off",
            )
            c[5].metric("Capital at risk", f"${pos.capital_at_risk:,.0f}")

            for s in pos.open_shorts:
                kind = "Put" if s.option_type == "put" else "Call"
                mark = (live or {}).get("marks", {}).get(_short_key(s))
                captured = (
                    f" · {1 - mark / s.avg_open_price:.0%} captured (mark {mark:.2f})"
                    if mark is not None and s.avg_open_price > 0 else ""
                )
                row = st.columns([6, 1.3, 1.3, 1.3])
                row[0].markdown(
                    f"**{s.contracts}× {s.strike:g} {kind}** · exp {s.expiration:%a %b %d} "
                    f"({s.dte()}d) · sold @ {s.avg_open_price:.2f} = "
                    f"${s.premium_collected:,.0f}{captured}"
                )
                k = _short_key(s)
                if row[1].button("🔄 Manage", key=f"ldg_manage_{k}", use_container_width=True):
                    st.session_state[PM_PREFILL_KEY] = _manage_prefill(pos, s)
                    st.toast(f"Loaded {s.ticker} {s.strike:g}{kind[0]} into 🔄 Manage Position.")
                if row[2].button("Expired", key=f"ldg_exp_{k}", use_container_width=True):
                    _log_lifecycle(s, TradeAction.EXPIRED)
                if row[3].button("Assigned", key=f"ldg_asg_{k}", use_container_width=True):
                    _log_lifecycle(s, TradeAction.ASSIGNED)


def _log_lifecycle(s: OpenShort, action: TradeAction) -> None:
    store.add_trade(Trade(
        trade_date=min(date.today(), s.expiration),
        ticker=s.ticker,
        action=action,
        quantity=s.contracts,
        option_type=s.option_type,
        strike=s.strike,
        expiration=s.expiration,
    ))
    st.session_state.pop("ledger_live", None)
    st.rerun()


def _render_exposure(ledger: Ledger, settings: dict) -> None:
    st.markdown("#### 🧮 Exposure")
    account = float(settings["account_size"] or 0)
    report = compute_exposure(
        list(ledger.positions.values()), account,
        settings["max_ticker_pct"], settings["max_total_pct"],
    )
    if account <= 0:
        st.caption("Set your account size under ⚙️ Ledger settings to see limits.")
    c = st.columns(3)
    c[0].metric("Capital committed", f"${report.total:,.0f}")
    c[1].metric(
        "Of account", f"{report.pct_of_account:.0%}" if account > 0 else "—",
        f"limit {settings['max_total_pct']:.0%}", delta_color="off",
    )
    c[2].metric("Free cash", f"${report.free_cash:,.0f}" if account > 0 else "—")
    for w in report.warnings:
        st.warning(w, icon="⚠️")
    if report.rows:
        st.dataframe(
            pd.DataFrame([{
                "Ticker": r.ticker,
                "Phase": r.phase,
                "CSP collateral": r.csp_collateral,
                "Share cost": r.share_cost,
                "Total": r.total,
                "% of account": r.pct_of_account * 100 if account > 0 else None,
            } for r in report.rows]),
            hide_index=True, use_container_width=True,
            column_config={
                "CSP collateral": st.column_config.NumberColumn(format="$%.0f"),
                "Share cost": st.column_config.NumberColumn(format="$%.0f"),
                "Total": st.column_config.NumberColumn(format="$%.0f"),
                "% of account": st.column_config.ProgressColumn(
                    format="%.0f%%", min_value=0, max_value=100,
                ),
            },
        )


def _render_performance(ledger: Ledger) -> None:
    st.markdown("#### 📈 Performance")
    this_month = date.today().strftime("%Y-%m")
    wins = [cyc for cyc in ledger.closed if cyc.realized_pnl > 0]
    c = st.columns(4)
    c[0].metric("Total premium (net)", f"${ledger.total_premium:,.2f}")
    c[1].metric("This month", f"${ledger.monthly_premium.get(this_month, 0.0):,.2f}")
    c[2].metric("Realized P&L (closed cycles)", f"${ledger.realized_pnl:,.2f}")
    c[3].metric(
        "Closed cycles",
        str(len(ledger.closed)),
        f"{len(wins)}/{len(ledger.closed)} profitable" if ledger.closed else None,
        delta_color="off",
    )

    if ledger.monthly_premium:
        st.bar_chart(
            pd.DataFrame(
                {"Premium": list(ledger.monthly_premium.values())},
                index=list(ledger.monthly_premium.keys()),
            ),
            height=220,
        )

    if ledger.closed:
        st.dataframe(
            pd.DataFrame([{
                "Ticker": cyc.ticker,
                "Started": cyc.started,
                "Ended": cyc.ended,
                "Days": cyc.days,
                "Premium": cyc.net_premium,
                "Share P&L": cyc.share_pnl,
                "Realized P&L": cyc.realized_pnl,
                "Max capital": cyc.max_capital,
                "Return %": cyc.return_pct * 100,
                "Annualized %": cyc.annualized_pct * 100,
            } for cyc in reversed(ledger.closed)]),
            hide_index=True, use_container_width=True,
            column_config={
                "Premium": st.column_config.NumberColumn(format="$%.2f"),
                "Share P&L": st.column_config.NumberColumn(format="$%.2f"),
                "Realized P&L": st.column_config.NumberColumn(format="$%.2f"),
                "Max capital": st.column_config.NumberColumn(format="$%.0f"),
                "Return %": st.column_config.NumberColumn(format="%.2f%%"),
                "Annualized %": st.column_config.NumberColumn(format="%.1f%%"),
            },
        )


def queue_trade_prefill(
    ticker: str, strike: float, expiration: date, price: float,
    option_type: str = "put", note: str = "",
) -> None:
    """
    Pre-fill the Log a Trade form with a sell-to-open. Called by other
    tabs (the scanner's Log button); applied the next time the form renders.
    """
    st.session_state[LEDGER_PREFILL_KEY] = {
        "ldg_f_action": TradeAction.SELL_TO_OPEN,
        "ldg_f_ticker": ticker,
        "ldg_f_date": date.today(),
        "ldg_f_qty": 1,
        "ldg_f_type": option_type,
        "ldg_f_strike": float(strike),
        "ldg_f_exp": expiration,
        "ldg_f_price": round(float(price), 2),
        "ldg_f_notes": note,
    }


def _apply_trade_prefill() -> bool:
    prefill = st.session_state.pop(LEDGER_PREFILL_KEY, None)
    if not prefill:
        return False
    for k, v in prefill.items():
        st.session_state[k] = v
    return True


def _render_trade_form() -> None:
    st.markdown("#### ✍️ Log a Trade")
    if _apply_trade_prefill():
        st.info(
            "Pre-filled from the scanner — set your actual fill price and "
            "contracts, then **➕ Add trade**.",
            icon="📒",
        )
    tab_single, tab_roll = st.tabs(["Single fill", "Roll (close + reopen)"])

    with tab_single:
        with st.form("ledger_trade_form", clear_on_submit=True):
            c = st.columns(4)
            action = c[0].selectbox(
                "Action", list(TradeAction), format_func=lambda a: a.label,
                key="ldg_f_action",
            )
            ticker = c[1].text_input("Ticker", key="ldg_f_ticker").upper().strip()
            trade_date = c[2].date_input("Trade date", key="ldg_f_date")
            quantity = c[3].number_input(
                "Contracts (or shares)", min_value=1, step=1, key="ldg_f_qty",
            )
            c = st.columns(5)
            option_type = c[0].radio(
                "Type", ["put", "call"], horizontal=True, key="ldg_f_type",
            )
            strike = c[1].number_input("Strike", min_value=0.0, step=0.5, key="ldg_f_strike")
            expiration = c[2].date_input("Expiration", key="ldg_f_exp")
            price = c[3].number_input(
                "Price / share", min_value=0.0, step=0.01, key="ldg_f_price",
                help="Option premium per share, or share price for share trades. "
                     "Not needed for Expired / Assigned.",
            )
            fees = c[4].number_input("Fees ($)", min_value=0.0, step=0.01, key="ldg_f_fees")
            notes = st.text_input("Notes", placeholder="optional", key="ldg_f_notes")
            submitted = st.form_submit_button("➕ Add trade", type="primary")

        if submitted:
            error = _validate(action, ticker, strike, price)
            if error:
                st.error(error)
            else:
                trade = Trade(
                    trade_date=trade_date,
                    ticker=ticker,
                    action=action,
                    quantity=int(quantity),
                    price=float(price),
                    option_type=option_type if action.is_option else None,
                    strike=float(strike) if action.is_option else None,
                    expiration=expiration if action.is_option else None,
                    fees=float(fees),
                    notes=notes,
                )
                store.add_trade(trade)
                st.session_state.pop("ledger_live", None)
                st.toast(f"Logged: {trade.describe()}")
                st.rerun()

    with tab_roll:
        _render_roll_form()


def _validate(action: TradeAction, ticker: str, strike: float, price: float) -> str:
    if not ticker:
        return "Enter a ticker."
    if action.is_option and strike <= 0:
        return "Enter the option's strike."
    if action in (
        TradeAction.SELL_TO_OPEN, TradeAction.BUY_TO_CLOSE,
        TradeAction.BUY_SHARES, TradeAction.SELL_SHARES,
    ) and price <= 0:
        return "Enter the fill price."
    return ""


def _render_roll_form() -> None:
    trades, ledger = load_ledger()
    shorts = ledger.open_shorts
    if not shorts:
        st.caption("No open options to roll.")
        return
    labels = {
        f"{s.ticker} {s.contracts}× {s.strike:g}{'P' if s.option_type == 'put' else 'C'} "
        f"{s.expiration}": s
        for s in shorts
    }
    with st.form("ledger_roll_form", clear_on_submit=True):
        choice = st.selectbox("Contract to roll", list(labels))
        c = st.columns(5)
        roll_date = c[0].date_input("Date", value=date.today())
        btc_price = c[1].number_input("Buy-to-close price", min_value=0.0, step=0.01)
        new_strike = c[2].number_input("New strike", min_value=0.0, step=0.5)
        new_exp = c[3].date_input("New expiration", value=date.today())
        sto_price = c[4].number_input("New sell price", min_value=0.0, step=0.01)
        fees = st.number_input("Total fees ($)", min_value=0.0, step=0.01)
        submitted = st.form_submit_button("🔁 Log roll", type="primary")

    if submitted:
        old = labels[choice]
        if new_strike <= 0 or sto_price <= 0:
            st.error("Enter the new strike and sell price.")
            return
        store.add_trade(Trade(
            trade_date=roll_date, ticker=old.ticker,
            action=TradeAction.BUY_TO_CLOSE, quantity=old.contracts,
            price=float(btc_price), option_type=old.option_type,
            strike=old.strike, expiration=old.expiration,
            fees=float(fees) / 2, notes="roll",
        ))
        store.add_trade(Trade(
            trade_date=roll_date, ticker=old.ticker,
            action=TradeAction.SELL_TO_OPEN, quantity=old.contracts,
            price=float(sto_price), option_type=old.option_type,
            strike=float(new_strike), expiration=new_exp,
            fees=float(fees) / 2, notes="roll",
        ))
        net = (sto_price - btc_price) * 100 * old.contracts - fees
        st.session_state.pop("ledger_live", None)
        st.toast(f"Rolled {old.ticker} for a net {'credit' if net >= 0 else 'debit'} of ${abs(net):,.2f}")
        st.rerun()


def trades_to_csv(trades: list[Trade]) -> str:
    buf = io.StringIO()
    pd.DataFrame([{
        "id": t.id,
        "date": t.trade_date,
        "ticker": t.ticker,
        "action": t.action.label,
        "type": t.option_type or "",
        "strike": t.strike,
        "expiration": t.expiration,
        "quantity": t.quantity,
        "price": t.price,
        "fees": t.fees,
        "cash_flow": round(t.cash_flow, 2),
        "notes": t.notes,
    } for t in trades]).to_csv(buf, index=False)
    return buf.getvalue()


def _render_history(trades: list[Trade], ledger: Ledger) -> None:
    st.markdown("#### 🗂️ Trade History")
    for w in ledger.warnings:
        st.warning(w, icon="⚠️")
    if not trades:
        st.caption("No trades logged yet.")
        return
    st.dataframe(
        pd.DataFrame([{
            "ID": t.id,
            "Date": t.trade_date,
            "Trade": t.describe(),
            "Cash": t.cash_flow,
            "Notes": t.notes,
        } for t in reversed(trades)]),
        hide_index=True, use_container_width=True,
        column_config={"Cash": st.column_config.NumberColumn(format="$%.2f")},
    )
    c = st.columns([2, 1, 2])
    c[0].download_button(
        "⬇️ Export CSV", trades_to_csv(trades),
        file_name=f"wheel_ledger_{date.today()}.csv", mime="text/csv",
        use_container_width=True,
    )
    by_id = {t.id: t for t in trades}
    to_delete = c[2].selectbox(
        "Delete trade", [None] + list(reversed(list(by_id))),
        format_func=lambda i: "— pick a trade ID —" if i is None else f"#{i} {by_id[i].describe()}",
        key="ledger_delete_sel",
    )
    if to_delete is not None and c[1].button("🗑️ Delete", use_container_width=True):
        store.delete_trade(to_delete)
        st.session_state.pop("ledger_live", None)
        st.rerun()


def _render_settings(settings: dict) -> dict:
    with st.expander("⚙️ Ledger settings"):
        c = st.columns(4)
        account = c[0].number_input(
            "Account size ($)", min_value=0.0, step=1000.0,
            value=float(settings["account_size"]),
        )
        max_ticker = c[1].slider(
            "Max per ticker %", 5, 100, int(round(settings["max_ticker_pct"] * 100)), step=5,
        )
        max_total = c[2].slider(
            "Max committed %", 10, 100, int(round(settings["max_total_pct"] * 100)), step=5,
        )
        target = c[3].slider(
            "Profit-take alert at %", 25, 90, int(round(settings["profit_target"] * 100)), step=5,
        )
    new = {
        "account_size": account,
        "max_ticker_pct": max_ticker / 100,
        "max_total_pct": max_total / 100,
        "profit_target": target / 100,
    }
    if new != settings:
        try:
            store.set_pref(_SETTINGS_KEY, new)
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Could not save settings: {exc}")
    return new


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def render_ledger(dp=None) -> None:
    st.subheader("📒 Wheel Ledger")
    st.caption("Every put and call you sell, tracked through the full wheel cycle.")

    try:
        trades, ledger = load_ledger()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not open the ledger database: {exc}", icon="🚨")
        return

    _render_storage_badge()
    settings = _render_settings(load_settings())

    shorts = ledger.open_shorts
    refresh = False
    if shorts:
        refresh = st.button("↻ Refresh live prices", key="ledger_refresh")
    live = _live_data(dp, shorts, force=refresh)
    alerts = collect_alerts(ledger, live, settings["profit_target"])

    _render_alerts(alerts, live)
    st.divider()
    _render_positions(ledger, live)
    st.divider()
    _render_exposure(ledger, settings)
    st.divider()
    _render_performance(ledger)
    st.divider()
    _render_trade_form()
    st.divider()
    _render_history(trades, ledger)
