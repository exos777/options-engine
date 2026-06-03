"""
Daily Scanner — the morning watchlist sweep for a pure premium seller.

Answers one question: "Where is the best premium to sell today?"

For every ticker on the watchlist it pulls a live quote, the nearest
session-appropriate option chain (5–7 DTE Monday mode, 0–3 DTE Friday
mode), and 6 months of history; runs the existing CSP and CC scoring
engines; and surfaces the single best strike for each side plus a
0–100 composite score.

Scanning is parallelised across tickers (I/O-bound Tradier calls) and
results are cached for 5 minutes.
"""

from __future__ import annotations

import json
import math
import pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Callable, Optional

import numpy as np
import streamlit as st

from data.common import days_to_expiration
from indicators.support_resistance import find_support_resistance
from indicators.technical import calculate_full_indicators
from scoring.cash_secured_put import score_cash_secured_puts
from scoring.covered_call import score_covered_calls
from scoring.regime import classify_regime
from strategies.models import (
    ChartRegime,
    FilterParams,
    OptionContract,
    RiskProfile,
    Strategy,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_WATCHLIST: list[str] = [
    "AAPL", "AMD", "AMZN", "GOOG", "INTC",
    "META", "MSFT", "MU", "NFLX", "NVDA",
    "PLTR", "QQQ", "SNOW", "SPY", "TSLA",
]

# Session → (lo_dte, hi_dte, target_dte) preferred expiration window.
_SESSION_WINDOWS = {
    "Monday": (5, 7, 6),
    "Friday": (0, 3, 1),
}

_CACHE_TTL_SECONDS = 300
_MAX_WORKERS = 8

# Shared scan filters — a generic premium-seller profile (no per-ticker
# cost basis / desired buy price; those are a Screener-tab concern).
_SCAN_MAX_DELTA = 0.40
_SCAN_MIN_OI = 50
_SCAN_MAX_SPREAD = 0.15      # CLAUDE.md hard filter: spread > 15% → no trade
_SCAN_MIN_PREMIUM = 0.10

# Row background / border colours by score bucket.
_COLOR_STRONG = ("#0d2a17", "#3fb950")
_COLOR_OK = ("#2a2410", "#d29922")
_COLOR_WEAK = ("#2a1416", "#f85149")
_COLOR_MUTED = ("#1a1a20", "#6e7681")


# ---------------------------------------------------------------------------
# Watchlist persistence (survives reruns, refreshes, and restarts)
# ---------------------------------------------------------------------------

_PREFS_PATH = (
    pathlib.Path(__file__).parent.parent / "data" / "scanner_prefs.json"
)


def _load_saved_watchlist() -> list[str]:
    """Return the persisted watchlist, or the default if none is saved."""
    try:
        prefs = json.loads(_PREFS_PATH.read_text(encoding="utf-8"))
        saved = prefs.get("watchlist")
        if isinstance(saved, list) and saved:
            # Keep only clean, de-duplicated symbols.
            out: list[str] = []
            for sym in saved:
                s = str(sym).strip().upper()
                if s and s not in out:
                    out.append(s)
            if out:
                return out
    except Exception:
        pass
    return list(DEFAULT_WATCHLIST)


def _save_watchlist(watchlist: list[str]) -> None:
    """Persist the watchlist to disk. Best-effort — never raises."""
    try:
        _PREFS_PATH.parent.mkdir(parents=True, exist_ok=True)
        _PREFS_PATH.write_text(
            json.dumps({"watchlist": watchlist}, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


def _parse_watchlist(raw: str) -> list[str]:
    """Parse a comma/newline-separated string into clean unique symbols."""
    watchlist: list[str] = []
    for chunk in raw.replace("\n", ",").split(","):
        sym = chunk.strip().upper()
        if sym and sym not in watchlist:
            watchlist.append(sym)
    return watchlist


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ScanRow:
    """One ticker's scan result. All metrics pre-computed for fast render."""
    ticker: str
    price: float = 0.0
    change_pct: float = 0.0
    volume: int = 0
    regime: Optional[ChartRegime] = None
    regime_label: str = "—"
    trade_bias: str = ""
    iv_rank: float = 0.0
    # Best CSP
    best_csp_strike: Optional[float] = None
    best_csp_premium: float = 0.0
    best_csp_ann_pct: float = 0.0
    best_csp_delta: float = 0.0
    best_csp_theta_day: float = 0.0
    best_csp_score: float = 0.0
    # Best CC
    best_cc_strike: Optional[float] = None
    best_cc_premium: float = 0.0
    best_cc_ann_pct: float = 0.0
    best_cc_theta_day: float = 0.0
    best_cc_score: float = 0.0
    # Composite + context
    score: float = 0.0
    expiration: str = ""
    dte: int = 0
    earnings_warning: bool = False
    has_no_trade: bool = False
    no_trade_reason: str = ""
    error: str = ""


# ---------------------------------------------------------------------------
# Scan helpers
# ---------------------------------------------------------------------------

def _pick_expiration(
    expirations: tuple[str, ...] | list[str],
    session_mode: str,
) -> tuple[str, int]:
    """Pick the expiration closest to the session's preferred DTE window."""
    today = date.today()
    parsed: list[tuple[str, int]] = []
    for exp in expirations:
        try:
            dte = (date.fromisoformat(exp) - today).days
        except (TypeError, ValueError):
            continue
        if dte >= 0:
            parsed.append((exp, dte))
    if not parsed:
        raise ValueError("No future expirations available.")

    lo, hi, target = _SESSION_WINDOWS.get(session_mode, _SESSION_WINDOWS["Monday"])
    in_window = [(e, d) for e, d in parsed if lo <= d <= hi]
    pool = in_window or parsed
    return min(pool, key=lambda pair: abs(pair[1] - target))


def _atm_iv(contracts: list[OptionContract], spot: float) -> float:
    """Average implied vol of strikes within 5% of spot. 0.0 if unavailable."""
    if spot <= 0:
        return 0.0
    ivs = [
        c.implied_volatility
        for c in contracts
        if c.implied_volatility and abs(c.strike - spot) / spot <= 0.05
    ]
    return sum(ivs) / len(ivs) if ivs else 0.0


def _iv_rank(hist_df, atm_iv: float) -> float:
    """
    IV rank 0–100: where current ATM IV sits within the trailing
    realised-volatility range. A practical proxy when historical implied
    vol isn't available — high rank = IV is rich vs how the stock has
    actually moved, i.e. a good premium-selling environment.
    """
    if atm_iv <= 0 or hist_df is None or len(hist_df) < 40:
        return 50.0
    close = hist_df["Close"].astype(float)
    log_ret = np.log(close / close.shift(1)).dropna()
    realised = (log_ret.rolling(20).std() * math.sqrt(252)).dropna()
    if realised.empty:
        return 50.0
    lo, hi = float(realised.min()), float(realised.max())
    if hi <= lo:
        return 50.0
    rank = (atm_iv - lo) / (hi - lo) * 100.0
    return max(0.0, min(100.0, rank))


def _earnings_in_window(earnings_date: Optional[str], expiration: str) -> bool:
    """True if a known earnings date falls on/before the expiration."""
    if not earnings_date:
        return False
    try:
        ed = date.fromisoformat(earnings_date)
        exp = date.fromisoformat(expiration)
        return date.today() <= ed <= exp
    except (TypeError, ValueError):
        return False


def _scan_params(strategy: Strategy) -> FilterParams:
    """Generic premium-seller filter profile shared by every scan."""
    return FilterParams(
        strategy=strategy,
        risk_profile=RiskProfile.BALANCED,
        max_delta=_SCAN_MAX_DELTA,
        min_open_interest=_SCAN_MIN_OI,
        max_spread_pct=_SCAN_MAX_SPREAD,
        min_premium=_SCAN_MIN_PREMIUM,
    )


def _scan_one(dp, ticker: str, session_mode: str) -> ScanRow:
    """Scan a single ticker. Never raises — failures are captured on the row."""
    row = ScanRow(ticker=ticker)

    # 1. Quote
    try:
        quote = dp.get_quote(ticker)
    except Exception as exc:  # noqa: BLE001 — surface as row error
        row.error = f"Quote unavailable ({exc})"
        return row
    row.price = quote.price
    row.change_pct = quote.change_pct
    row.volume = quote.volume

    # 2. Expiration for this session
    try:
        expirations = dp.get_expirations(ticker)
        expiration, dte = _pick_expiration(expirations, session_mode)
    except Exception as exc:  # noqa: BLE001
        row.error = f"No options ({exc})"
        return row
    row.expiration = expiration
    row.dte = dte

    # 3. Option chain
    try:
        calls, puts = dp.get_option_chain(ticker, expiration)
    except Exception as exc:  # noqa: BLE001
        row.error = f"Chain unavailable ({exc})"
        return row

    # 4. History → indicators → S/R → regime
    try:
        hist = dp.get_historical(ticker, months=6)
        indicators = calculate_full_indicators(hist).snapshot
        supports, resistances = find_support_resistance(hist, quote.price)
        regime = classify_regime(indicators, supports, resistances)
    except Exception as exc:  # noqa: BLE001
        row.error = f"Indicators unavailable ({exc})"
        return row
    row.regime = regime.primary
    row.regime_label = regime.primary.value
    row.trade_bias = regime.trade_bias

    # 5. IV rank + expected move
    atm_iv = _atm_iv(puts + calls, quote.price)
    row.iv_rank = _iv_rank(hist, atm_iv)
    expected_move = (
        quote.price * atm_iv * math.sqrt(max(dte, 1) / 365)
        if atm_iv > 0
        else indicators.atr_14 * math.sqrt(max(dte, 1))
    )

    # 6. Earnings (best-effort)
    earnings_date = getattr(quote, "earnings_date", None)
    if earnings_date is None and hasattr(dp, "get_earnings_date"):
        try:
            earnings_date = dp.get_earnings_date(ticker)
        except Exception:  # noqa: BLE001
            earnings_date = None
    row.earnings_warning = _earnings_in_window(earnings_date, expiration)

    # 7. Score both sides. Friday session relaxes the 5-DTE floor.
    min_dte = 0 if session_mode == "Friday" else 5

    csp_scored = score_cash_secured_puts(
        puts, quote.price, dte, indicators, regime,
        supports, resistances, _scan_params(Strategy.CASH_SECURED_PUT),
        earnings_date=earnings_date,
        expected_move=expected_move,
        min_dte=min_dte,
    )
    cc_scored = score_covered_calls(
        calls, quote.price, dte, indicators, regime,
        supports, resistances, _scan_params(Strategy.COVERED_CALL),
        earnings_date=earnings_date,
        expected_move=expected_move,
        min_dte=min_dte,
    )

    if csp_scored:
        best = csp_scored[0]
        c = best.contract
        row.best_csp_strike = c.strike
        row.best_csp_premium = best.premium
        row.best_csp_ann_pct = best.annualized_return * 100.0
        row.best_csp_delta = abs(c.delta) if c.delta is not None else 0.0
        row.best_csp_theta_day = abs(c.theta) if c.theta is not None else 0.0
        row.best_csp_score = best.score

    if cc_scored:
        best = cc_scored[0]
        c = best.contract
        row.best_cc_strike = c.strike
        row.best_cc_premium = best.premium
        row.best_cc_ann_pct = best.annualized_return * 100.0
        row.best_cc_theta_day = abs(c.theta) if c.theta is not None else 0.0
        row.best_cc_score = best.score

    row.score = max(row.best_csp_score, row.best_cc_score)

    # 8. Hard-filter "no trade" reasons (IV-rank check is applied at render
    #    time so the sidebar slider stays instant).
    if regime.trade_bias == "skip":
        row.has_no_trade = True
        row.no_trade_reason = "Bearish regime"
    elif row.earnings_warning:
        row.has_no_trade = True
        days = days_to_expiration(earnings_date) if earnings_date else 0
        row.no_trade_reason = f"Earnings {days}d"
    elif not csp_scored and not cc_scored:
        row.has_no_trade = True
        row.no_trade_reason = "No liquid options"

    return row


def scan_watchlist(
    dp,
    tickers: list[str],
    session_mode: str,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> list[ScanRow]:
    """Scan every ticker in parallel. Results preserve watchlist order."""
    tickers = [t for t in tickers if t]
    if not tickers:
        return []

    results: dict[str, ScanRow] = {}
    total = len(tickers)
    with ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, total)) as pool:
        futures = {
            pool.submit(_scan_one, dp, t, session_mode): t for t in tickers
        }
        for done, future in enumerate(as_completed(futures), start=1):
            ticker = futures[future]
            try:
                results[ticker] = future.result()
            except Exception as exc:  # noqa: BLE001
                results[ticker] = ScanRow(ticker=ticker, error=str(exc))
            if progress_cb:
                progress_cb(done, total)

    return [results[t] for t in tickers if t in results]


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------

def _effective_score(row: ScanRow, strategy_filter: str) -> float:
    """The score that matters given the CSP/CC/Both filter."""
    if strategy_filter == "CSP only":
        return row.best_csp_score
    if strategy_filter == "CC only":
        return row.best_cc_score
    return max(row.best_csp_score, row.best_cc_score)


def _row_color(row: ScanRow, eff_score: float) -> tuple[str, str]:
    """(background, border) for a row, given its effective score."""
    if row.error or row.has_no_trade or row.earnings_warning:
        return _COLOR_MUTED
    if eff_score >= 70:
        return _COLOR_STRONG
    if eff_score >= 50:
        return _COLOR_OK
    return _COLOR_WEAK


def _fmt_money(v: float) -> str:
    return f"${v:,.2f}" if v else "—"


def _fmt_strike(v: Optional[float]) -> str:
    return f"${v:,.2f}" if v else "—"


def _cell(label: str, value: str, width: str = "1") -> str:
    return (
        f'<div style="flex:{width};min-width:0;">'
        f'<div style="font-size:10px;color:#8b949e;text-transform:uppercase;'
        f'letter-spacing:.04em;">{label}</div>'
        f'<div style="font-size:14px;color:#e6edf3;font-weight:600;">{value}</div>'
        f"</div>"
    )


def _render_row(row: ScanRow, friday: bool, eff_score: float) -> None:
    """One scan result: a colour-coded card plus an Action button."""
    bg, border = _row_color(row, eff_score)
    card_col, action_col = st.columns([13, 2])

    with card_col:
        if row.error:
            st.markdown(
                f'<div style="background:{bg};border:1px solid {border};'
                f'border-left:4px solid {border};border-radius:8px;'
                f'padding:10px 14px;margin-bottom:6px;">'
                f'<span style="font-weight:700;color:#e6edf3;">{row.ticker}</span>'
                f'<span style="color:#f85149;margin-left:12px;">{row.error}</span>'
                f"</div>",
                unsafe_allow_html=True,
            )
            return

        change_color = "#3fb950" if row.change_pct >= 0 else "#f85149"
        price_html = (
            f'{_fmt_money(row.price)} '
            f'<span style="color:{change_color};font-size:11px;">'
            f'{row.change_pct * 100:+.1f}%</span>'
        )
        # Friday session swaps CSP Ann% for weekend theta.
        if friday:
            weekend_theta = row.best_csp_theta_day * 2
            third_label = "Weekend Theta"
            third_value = (
                f"${weekend_theta:.2f} over wknd" if weekend_theta else "—"
            )
        else:
            third_label = "CSP Ann%"
            third_value = (
                f"{row.best_csp_ann_pct:.1f}%" if row.best_csp_strike else "—"
            )

        cells = "".join([
            _cell("Ticker", f"<b>{row.ticker}</b>", "1.1"),
            _cell("Price", price_html, "1.4"),
            _cell("Regime", row.regime_label, "1.4"),
            _cell("IV Rank", f"{row.iv_rank:.0f}", "0.9"),
            _cell("Best CSP", _fmt_strike(row.best_csp_strike), "1.1"),
            _cell("CSP Prem", _fmt_money(row.best_csp_premium), "1.1"),
            _cell(third_label, third_value, "1.5"),
            _cell("Best CC", _fmt_strike(row.best_cc_strike), "1.1"),
            _cell("CC Prem", _fmt_money(row.best_cc_premium), "1.1"),
            _cell(
                "Theta/day",
                f"${row.best_cc_theta_day:.2f}" if row.best_cc_strike else "—",
                "1.1",
            ),
            _cell(
                "Score",
                f'<span style="color:{border};">{eff_score:.0f}</span>',
                "0.9",
            ),
        ])
        badge = ""
        if row.earnings_warning:
            badge = (
                '<span style="color:#d29922;font-size:11px;margin-left:8px;">'
                "⚠️ earnings</span>"
            )
        elif row.has_no_trade:
            badge = (
                '<span style="color:#8b949e;font-size:11px;margin-left:8px;">'
                f"⛔ {row.no_trade_reason}</span>"
            )
        st.markdown(
            f'<div style="background:{bg};border:1px solid {border};'
            f'border-left:4px solid {border};border-radius:8px;'
            f'padding:10px 14px;margin-bottom:6px;display:flex;gap:14px;'
            f'align-items:center;">{cells}'
            f'<div style="flex:0 0 auto;">{badge}</div></div>',
            unsafe_allow_html=True,
        )

    with action_col:
        if row.earnings_warning:
            st.button(
                "⚠️ Earnings", key=f"scan_act_{row.ticker}",
                disabled=True, use_container_width=True,
            )
        elif row.has_no_trade:
            st.button(
                row.no_trade_reason, key=f"scan_act_{row.ticker}",
                disabled=True, use_container_width=True,
            )
        elif eff_score >= 70:
            if st.button(
                "📈 Open", key=f"scan_act_{row.ticker}",
                type="primary", use_container_width=True,
            ):
                st.session_state["selected_ticker"] = row.ticker
                st.session_state["active_tab"] = "screener"
                st.session_state["quick_select"] = row.ticker
                st.session_state["auto_run"] = True
                st.rerun()
        else:
            st.button(
                "Skip", key=f"scan_act_{row.ticker}",
                disabled=True, use_container_width=True,
            )


def _render_summary(rows: list[ScanRow], strategy_filter: str) -> None:
    """Five summary cards beneath the table."""
    tradeable = [
        r for r in rows
        if not r.error and not r.has_no_trade and not r.earnings_warning
    ]
    best_csp = max(
        (r for r in tradeable if r.best_csp_strike),
        key=lambda r: r.best_csp_score, default=None,
    )
    best_cc = max(
        (r for r in tradeable if r.best_cc_strike),
        key=lambda r: r.best_cc_score, default=None,
    )
    strong = sum(
        1 for r in tradeable
        if _effective_score(r, strategy_filter) >= 70
    )
    earnings_ct = sum(1 for r in rows if r.earnings_warning)
    no_trade_ct = sum(
        1 for r in rows if r.has_no_trade and not r.earnings_warning
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        if best_csp:
            st.metric(
                "🅿️ Best CSP",
                f"{best_csp.ticker} {_fmt_strike(best_csp.best_csp_strike)}",
                f"Score {best_csp.best_csp_score:.0f}",
            )
        else:
            st.metric("🅿️ Best CSP", "—", "No setup")
    with c2:
        if best_cc:
            st.metric(
                "🅲 Best CC",
                f"{best_cc.ticker} {_fmt_strike(best_cc.best_cc_strike)}",
                f"Score {best_cc.best_cc_score:.0f}",
            )
        else:
            st.metric("🅲 Best CC", "—", "No setup")
    with c3:
        st.metric("✅ Strong Setups", str(strong), "score ≥ 70")
    with c4:
        st.metric("⚠️ Earnings Warnings", str(earnings_ct))
    with c5:
        st.metric("⛔ No Trade", str(no_trade_ct), "hard filters")


def _render_controls(dp) -> dict:
    """Scanner control panel. Returns the resolved settings + scan trigger."""
    st.markdown("#### 🔍 Scan Controls")
    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.3, 1.3])
    with c1:
        session_mode = st.radio(
            "Session",
            options=["Monday", "Friday"],
            help="Monday → 5–7 DTE expirations · Friday → 0–3 DTE",
            key="scanner_session",
        )
    with c2:
        strategy_filter = st.radio(
            "Strategy",
            options=["Both", "CSP only", "CC only"],
            key="scanner_strategy_filter",
        )
    with c3:
        min_score = st.slider(
            "Min score", 40, 80, 60, step=5, key="scanner_min_score",
        )
    with c4:
        min_iv_rank = st.slider(
            "Min IV rank", 20, 80, 30, step=5, key="scanner_min_iv",
        )

    # Seed the editor from the persisted watchlist on first render, then
    # let Streamlit's widget state own it for the rest of the session.
    if "scanner_watchlist_raw" not in st.session_state:
        st.session_state["scanner_watchlist_raw"] = ", ".join(
            _load_saved_watchlist()
        )

    watchlist_raw = st.text_area(
        "Watchlist (comma or newline separated)",
        height=68,
        key="scanner_watchlist_raw",
    )
    watchlist = _parse_watchlist(watchlist_raw)

    # Persist whenever the parsed list changes so edits survive restarts.
    if watchlist and watchlist != st.session_state.get("scanner_watchlist_saved"):
        _save_watchlist(watchlist)
        st.session_state["scanner_watchlist_saved"] = watchlist

    scan_clicked = st.button(
        "🔄 Scan All",
        type="primary",
        use_container_width=True,
        disabled=(dp is None or not watchlist),
        key="scanner_scan_btn",
    )

    return {
        "session_mode": session_mode,
        "strategy_filter": strategy_filter,
        "min_score": min_score,
        "min_iv_rank": min_iv_rank,
        "watchlist": watchlist,
        "scan_clicked": scan_clicked,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def render_scanner(dp=None) -> None:
    """Render the Daily Scanner tab."""
    st.subheader("☀️ Daily Scanner")
    st.caption("Where is the best premium to sell today?")

    if dp is None:
        st.warning("No data provider available — cannot scan.", icon="⚠️")
        return

    cfg = _render_controls(dp)
    st.divider()

    cache_key = (tuple(cfg["watchlist"]), cfg["session_mode"])
    cached_at: Optional[datetime] = st.session_state.get("scanner_scanned_at")
    cached_key = st.session_state.get("scanner_cache_key")
    rows: Optional[list[ScanRow]] = st.session_state.get("scanner_results")

    is_stale = (
        cached_at is None
        or cached_key != cache_key
        or (datetime.now() - cached_at).total_seconds() > _CACHE_TTL_SECONDS
    )

    # Scan on explicit click, or when the cache key changed and we already
    # have stale data to refresh. First visit waits for the button.
    if cfg["scan_clicked"] or (rows is not None and is_stale):
        progress = st.progress(0.0, text="Scanning watchlist…")

        def _bump(done: int, total: int) -> None:
            progress.progress(
                done / total, text=f"Scanned {done}/{total} tickers…",
            )

        rows = scan_watchlist(
            dp, cfg["watchlist"], cfg["session_mode"], progress_cb=_bump,
        )
        progress.empty()
        st.session_state["scanner_results"] = rows
        st.session_state["scanner_scanned_at"] = datetime.now()
        st.session_state["scanner_cache_key"] = cache_key
        cached_at = st.session_state["scanner_scanned_at"]

    if not rows:
        st.info(
            "Click **🔄 Scan All** to sweep your watchlist for the best "
            "premium-selling setups.",
            icon="👆",
        )
        return

    # Apply the live IV-rank filter (kept out of the cached scan so the
    # slider stays instant).
    for r in rows:
        if (
            not r.error
            and not r.has_no_trade
            and not r.earnings_warning
            and r.iv_rank < cfg["min_iv_rank"]
        ):
            r.has_no_trade = True
            r.no_trade_reason = f"IV too low ({r.iv_rank:.0f})"
        elif r.no_trade_reason.startswith("IV too low") and r.iv_rank >= cfg["min_iv_rank"]:
            # Slider was lowered — clear a previously-applied IV flag.
            r.has_no_trade = False
            r.no_trade_reason = ""

    friday = cfg["session_mode"] == "Friday"
    strategy_filter = cfg["strategy_filter"]

    # Sort: tradeable rows by score desc, muted rows last.
    def _sort_key(r: ScanRow) -> tuple[int, float]:
        muted = bool(r.error or r.has_no_trade or r.earnings_warning)
        return (1 if muted else 0, -_effective_score(r, strategy_filter))

    visible = sorted(
        (
            r for r in rows
            if r.error
            or r.has_no_trade
            or r.earnings_warning
            or _effective_score(r, strategy_filter) >= cfg["min_score"]
            # Always show muted rows; gate only by min_score for tradeable.
        ),
        key=_sort_key,
    )
    # Rows below min_score that are otherwise tradeable are hidden — show a
    # note so the count isn't a mystery.
    hidden = len(rows) - len(visible)

    stamp = cached_at.strftime("%I:%M %p").lstrip("0") if cached_at else "—"
    st.caption(
        f"Last scanned: {stamp}  ·  {len(rows)} tickers  ·  "
        f"session: {cfg['session_mode']} ({'0–3' if friday else '5–7'} DTE)"
        + (f"  ·  {hidden} below min score hidden" if hidden > 0 else "")
    )

    if not visible:
        st.info("No tickers meet the current filters.", icon="🔍")
        return

    for r in visible:
        _render_row(r, friday, _effective_score(r, strategy_filter))

    st.divider()
    _render_summary(rows, strategy_filter)
