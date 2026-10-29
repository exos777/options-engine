"""
Daily Scanner — CSP premium mode.

Answers one question: "Which put should I sell right now for the most
premium, on a stock I'd be happy to own if it goes wrong?"

For every ticker on the watchlist (quality names only — assignment is the
acceptable worst case) it pulls a live quote, the option chain for the
EARLIEST expiration (a daily if the ticker lists them, else the nearest
weekly; never more than 7 days out), and 6 months of history; scores
every OTM put with scoring/csp_premium.py; and surfaces the best one with
a trade plan: buy-back target, stop, and the cost if assigned.

Earnings are allowed but flagged. Scanning is parallelised across tickers
(I/O-bound API calls); history, expirations and earnings dates are cached
for the day, quotes and chains for 5 minutes.
"""

from __future__ import annotations

import json
import math
import pathlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Callable, Optional
from zoneinfo import ZoneInfo

import numpy as np
import streamlit as st

from data import store
from indicators.support_resistance import find_support_resistance
from indicators.technical import calculate_full_indicators
from portfolio.exposure import compute_exposure, max_new_csp_contracts
from scoring.csp_premium import (
    CLOSE_BY,
    MAX_DTE,
    SESSION_HOURS,
    PremiumParams,
    score_premium_puts,
)
from scoring.regime import classify_regime
from strategies.models import ChartRegime, OptionContract
from ui import ledger as ledger_ui

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CORE = [
    "AAPL", "AMD", "AMZN", "GOOG", "INTC", "META", "MSFT", "MU", "NFLX",
    "NVDA", "PLTR", "QQQ", "SNOW", "SPY", "TSLA",
]
_QUALITY_ADDED = [
    "ADBE", "ARM", "AVGO", "COST", "CRM", "CRWD", "IWM", "JPM", "LLY",
    "MA", "ORCL", "PANW", "SHOP", "SMH", "UBER", "V", "WMT", "XLK",
]
_SIMILAR_ADDED = [
    # Semis & hardware
    "AMAT", "ASML", "DELL", "LRCX", "MRVL", "QCOM", "TSM",
    # Software & networking
    "ANET", "INTU", "NOW",
    # Consumer internet, media, finance, health
    "ABNB", "DASH", "DIS", "GS", "UNH",
]

# Quality names with liquid daily/weekly options — stocks worth owning if a
# put is assigned: the trader's core list, then large profitable companies
# and broad ETFs with tight option markets. Speculative high-IV names
# (COIN, HOOD, SMCI, MSTR) are left out on purpose; add them by hand.
DEFAULT_WATCHLIST: list[str] = sorted(_CORE + _QUALITY_ADDED + _SIMILAR_ADDED)

# Earlier defaults. A saved watchlist identical to one of these was never
# customised, so it is upgraded to the current default on load.
_LEGACY_DEFAULTS: list[frozenset[str]] = [
    frozenset(_CORE),
    frozenset(_CORE + _QUALITY_ADDED),
]

_CACHE_TTL_SECONDS = 300
_MAX_WORKERS = 8

_MARKET_TZ = ZoneInfo("America/New_York")
_MARKET_OPEN = time(9, 30)
_MARKET_CLOSE = time(16, 0)

# Runner-up strikes kept per ticker for the "Compare strikes" table.
_ALTERNATIVES = 3

# Score penalty when the chart is in a confirmed downtrend: still shown
# (high IV often comes with selloffs) but ranked below calmer setups.
_DOWNTREND_PENALTY = 10.0

# True IV rank needs this many stored daily ATM-IV readings; until then
# the realised-vol proxy is used and the cell is marked "≈".
_IV_HISTORY_MIN_POINTS = 20
_IV_HISTORY_LOOKBACK_DAYS = 365

# Earnings inside this many days are listed under the scan results.
_EARNINGS_HORIZON_DAYS = 14

# ATM IV this many times the 20-day realised vol, and at least this high
# in absolute terms, flags an IV spike — the market pricing an event
# (usually earnings) even when no date is known. The absolute floor keeps
# calm index ETFs (0DTE SPY at 25% IV vs a 10% realised move) quiet.
_IV_SPIKE_RATIO = 2.5
_IV_SPIKE_MIN_IV = 0.40

# Row background / border colours by score bucket.
_COLOR_STRONG = ("#0d2a17", "#3fb950")
_COLOR_OK = ("#2a2410", "#d29922")
_COLOR_WEAK = ("#2a1416", "#f85149")
_COLOR_MUTED = ("#1a1a20", "#6e7681")


# ---------------------------------------------------------------------------
# Watchlist persistence (survives reruns, refreshes, restarts and deploys)
# ---------------------------------------------------------------------------

_WATCHLIST_PREF = "scanner.watchlist"

# Pre-database location. Still read as a fallback so an existing local
# watchlist carries over the first time the app runs with the store.
_PREFS_PATH = (
    pathlib.Path(__file__).parent.parent / "data" / "scanner_prefs.json"
)


def _clean_symbols(saved) -> list[str]:
    """Keep only clean, de-duplicated, upper-cased symbols."""
    out: list[str] = []
    if isinstance(saved, list):
        for sym in saved:
            s = str(sym).strip().upper()
            if s and s not in out:
                out.append(s)
    return out


def _load_saved_watchlist() -> list[str]:
    """Return the persisted watchlist, or the default if none is saved."""
    try:
        saved = store.get_pref(_WATCHLIST_PREF)
    except Exception:
        saved = None
    if saved is None:
        try:
            prefs = json.loads(_PREFS_PATH.read_text(encoding="utf-8"))
            saved = prefs.get("watchlist")
        except Exception:
            saved = None
    cleaned = _clean_symbols(saved)
    if not cleaned or frozenset(cleaned) in _LEGACY_DEFAULTS:
        return list(DEFAULT_WATCHLIST)
    return cleaned


def _save_watchlist(watchlist: list[str]) -> None:
    """Persist the watchlist. Best-effort — never raises."""
    try:
        store.set_pref(_WATCHLIST_PREF, list(watchlist))
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
    regime: Optional[ChartRegime] = None
    regime_label: str = "—"
    downtrend: bool = False
    atm_iv: float = 0.0
    iv_rank: float = 0.0
    iv_rank_source: str = "rv"      # "iv" = true IV rank, "rv" = proxy
    expected_move: float = 0.0
    realized_vol: float = 0.0       # 20-day annualised
    earnings_date: Optional[str] = None
    earnings_warning: bool = False  # earnings on/before expiration
    expiration: str = ""
    dte: int = 0
    hours_left: float = 0.0         # session hours left when scanned
    # Best put
    strike: Optional[float] = None
    credit: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    volume: int = 0
    open_interest: int = 0
    delta: Optional[float] = None
    est_decay: float = 0.0          # expected decay today, per share
    net_edge: float = 0.0           # est_decay − spread, per share
    edge_pct: float = 0.0           # net_edge / strike
    event_haircut: bool = False     # decay estimate halved for earnings
    distance_pct: float = 0.0
    outside_expected_move: bool = False
    support: Optional[float] = None
    spread_pct: float = 0.0
    target_price: float = 0.0       # where the put trades if decay holds
    stop_at: float = 0.0
    assigned_cost: float = 0.0
    cash_required: float = 0.0
    score: float = 0.0
    # Runner-up strikes for the "Compare strikes" table (best first).
    alternatives: list = field(default_factory=list)
    has_no_trade: bool = False
    no_trade_reason: str = ""
    error: str = ""

    @property
    def tradeable(self) -> bool:
        return not self.error and not self.has_no_trade and self.strike is not None

    @property
    def iv_spike_ratio(self) -> float:
        """ATM IV ÷ 20-day realised vol (0 when either is unknown)."""
        if self.atm_iv <= 0 or self.realized_vol <= 0:
            return 0.0
        return self.atm_iv / self.realized_vol

    @property
    def iv_spike(self) -> bool:
        return (
            self.iv_spike_ratio >= _IV_SPIKE_RATIO
            and self.atm_iv >= _IV_SPIKE_MIN_IV
        )


# ---------------------------------------------------------------------------
# Scan helpers
# ---------------------------------------------------------------------------

def _pick_expiration(
    expirations: tuple[str, ...] | list[str],
    now: Optional[datetime] = None,
    max_dte: int = MAX_DTE,
) -> tuple[str, int]:
    """
    The earliest tradeable expiration within *max_dte* days.

    Tickers with daily expirations resolve to the nearest day; weekly-only
    tickers to the nearest Friday. Today's expiration counts until the
    4:00 pm ET close, after which it is dead and the next one is used.
    """
    now = now or datetime.now(_MARKET_TZ)
    today = now.date()
    after_close = now.time() >= _MARKET_CLOSE
    parsed: list[tuple[str, int]] = []
    for exp in expirations:
        try:
            dte = (date.fromisoformat(exp) - today).days
        except (TypeError, ValueError):
            continue
        if dte < 0 or dte > max_dte or (dte == 0 and after_close):
            continue
        parsed.append((exp, dte))
    if not parsed:
        raise ValueError(f"No expiration within {max_dte} days.")
    return min(parsed, key=lambda pair: pair[1])


def session_hours_left(now: Optional[datetime] = None) -> float:
    """
    Regular-session hours left to hold a same-day trade. Before the open
    — or after the close, when planning for the next session — that's a
    full 6.5-hour day.
    """
    now = (now or datetime.now(_MARKET_TZ)).astimezone(_MARKET_TZ)
    t = now.time()
    if t < _MARKET_OPEN or t >= _MARKET_CLOSE:
        return SESSION_HOURS
    close = datetime.combine(now.date(), _MARKET_CLOSE, tzinfo=_MARKET_TZ)
    return max((close - now).total_seconds() / 3600.0, 0.0)


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


def _true_iv_rank(
    history: list[float], atm_iv: float,
    min_points: int = _IV_HISTORY_MIN_POINTS,
) -> Optional[float]:
    """
    Classic IV rank 0–100: today's ATM IV within its own trailing range.
    None until *min_points* past readings exist.
    """
    if atm_iv <= 0 or len(history) < min_points:
        return None
    values = [v for v in history if v > 0] + [atm_iv]
    lo, hi = min(values), max(values)
    if hi <= lo:
        return 50.0
    return max(0.0, min(100.0, (atm_iv - lo) / (hi - lo) * 100.0))


def _resolve_iv_rank(ticker: str, hist_df, atm_iv: float) -> tuple[float, str]:
    """
    Record today's ATM IV, then return (rank, source): the true IV rank
    when enough history is stored, else the realised-vol proxy.
    """
    today = date.today()
    history: list[float] = []
    try:
        since = today - timedelta(days=_IV_HISTORY_LOOKBACK_DAYS)
        history = [v for d, v in store.iv_history(ticker, since) if d != today]
        store.record_iv(ticker, today, atm_iv)
    except Exception:  # noqa: BLE001 — storage is best-effort for scans
        pass
    rank = _true_iv_rank(history, atm_iv)
    if rank is not None:
        return rank, "iv"
    return _iv_rank(hist_df, atm_iv), "rv"


def _market_today() -> date:
    """Today's date in New York — the server may run in UTC."""
    return datetime.now(_MARKET_TZ).date()


def _earnings_in_window(earnings_date: Optional[str], expiration: str) -> bool:
    """True if a known earnings date falls on/before the expiration."""
    if not earnings_date:
        return False
    try:
        ed = date.fromisoformat(earnings_date)
        exp = date.fromisoformat(expiration)
        return _market_today() <= ed <= exp
    except (TypeError, ValueError):
        return False


def _realized_vol(hist_df, window: int = 20) -> float:
    """Annualised close-to-close volatility over the last *window* days."""
    if hist_df is None or len(hist_df) < window + 1:
        return 0.0
    close = hist_df["Close"].astype(float)
    log_ret = np.log(close / close.shift(1)).dropna().tail(window)
    vol = float(log_ret.std() * math.sqrt(252))
    return vol if math.isfinite(vol) else 0.0


def _when(d: date, today: Optional[date] = None) -> str:
    """'today' / 'tomorrow' / 'in 3d' / 'Mon Nov 03'."""
    days = (d - (today or _market_today())).days
    if days == 0:
        return "today"
    if days == 1:
        return "tomorrow"
    if 1 < days <= 6:
        return f"in {days}d ({d:%a})"
    return f"{d:%a %b %d}"


def _resolve_earnings(dp, ticker: str, quote) -> Optional[str]:
    """
    Next earnings date (ISO) or None. The provider's dedicated lookup wins;
    the quote's own date is used only if it hasn't already passed.
    """
    earnings_date = None
    if hasattr(dp, "get_earnings_date"):
        try:
            earnings_date = _cached_daily(
                dp, "earnings", ticker, lambda: dp.get_earnings_date(ticker),
            )
        except Exception:  # noqa: BLE001
            earnings_date = None
    if not earnings_date:
        quoted = getattr(quote, "earnings_date", None)
        if quoted and str(quoted)[:10] >= _market_today().isoformat():
            earnings_date = str(quoted)[:10]
    return earnings_date


# Per-day cache for data that doesn't change intraday (history, the list of
# expirations, earnings dates). Cuts a rescan to quote + chain per ticker,
# which keeps a 30+ ticker watchlist inside Tradier's per-minute quota.
_daily_cache: dict[tuple, object] = {}
_daily_cache_lock = threading.Lock()


def _cached_daily(dp, kind: str, ticker: str, fetch: Callable[[], object]):
    key = (getattr(dp, "__name__", type(dp).__name__), kind, ticker, date.today())
    with _daily_cache_lock:
        if key in _daily_cache:
            return _daily_cache[key]
    value = fetch()
    with _daily_cache_lock:
        if len(_daily_cache) > 2000:       # drop previous days
            for k in [k for k in _daily_cache if k[3] != date.today()]:
                del _daily_cache[k]
        _daily_cache[key] = value
    return value


def _scan_one(
    dp, ticker: str,
    params: Optional[PremiumParams] = None,
    now: Optional[datetime] = None,
) -> ScanRow:
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

    # 2. Earliest expiration, at most a week out
    try:
        expirations = _cached_daily(
            dp, "expirations", ticker, lambda: dp.get_expirations(ticker),
        )
    except Exception as exc:  # noqa: BLE001
        row.error = f"No options ({exc})"
        return row
    try:
        expiration, dte = _pick_expiration(expirations, now)
    except ValueError:
        row.has_no_trade = True
        row.no_trade_reason = f"No expiry ≤ {MAX_DTE}d"
        return row
    row.expiration = expiration
    row.dte = dte

    # 3. Option chain (puts only matter)
    try:
        calls, puts = dp.get_option_chain(ticker, expiration)
    except Exception as exc:  # noqa: BLE001
        row.error = f"Chain unavailable ({exc})"
        return row

    # 4. History → indicators → support → regime
    try:
        hist = _cached_daily(
            dp, "history", ticker, lambda: dp.get_historical(ticker, months=6),
        )
        indicators = calculate_full_indicators(hist).snapshot
        supports, resistances = find_support_resistance(hist, quote.price)
        regime = classify_regime(indicators, supports, resistances)
    except Exception as exc:  # noqa: BLE001
        row.error = f"Indicators unavailable ({exc})"
        return row
    row.regime = regime.primary
    row.regime_label = regime.primary.value
    row.downtrend = regime.trade_bias == "skip"

    # 5. IV, IV rank, expected move
    atm_iv = _atm_iv(puts + calls, quote.price)
    row.atm_iv = atm_iv
    row.realized_vol = _realized_vol(hist)
    row.iv_rank, row.iv_rank_source = _resolve_iv_rank(ticker, hist, atm_iv)
    row.expected_move = (
        quote.price * atm_iv * math.sqrt(max(dte, 1) / 365)
        if atm_iv > 0
        else indicators.atr_14 * math.sqrt(max(dte, 1))
    )

    # 6. Earnings — allowed, just flagged
    earnings_date = _resolve_earnings(dp, ticker, quote)
    row.earnings_date = earnings_date
    row.earnings_warning = _earnings_in_window(earnings_date, expiration)

    # 7. Best put for a same-day round trip
    row.hours_left = session_hours_left(now)
    picks = score_premium_puts(
        puts, quote.price, dte,
        hours_left=row.hours_left,
        iv_rank=row.iv_rank,
        expected_move=row.expected_move,
        supports=supports,
        event_before_expiry=row.earnings_warning,
        params=params,
    )
    if not picks:
        row.has_no_trade = True
        row.no_trade_reason = "No liquid put beats its spread today"
        return row

    penalty = _DOWNTREND_PENALTY if row.downtrend else 0.0
    best = picks[0]
    c = best.contract
    row.strike = best.strike
    row.credit = best.credit
    row.bid, row.ask = c.bid, c.ask
    row.volume, row.open_interest = c.volume or 0, c.open_interest or 0
    row.delta = best.delta
    row.est_decay = best.est_decay
    row.net_edge = best.net_edge
    row.edge_pct = best.edge_pct
    row.event_haircut = best.event_haircut
    row.distance_pct = best.distance_pct
    row.outside_expected_move = best.outside_expected_move
    row.support = best.support
    row.spread_pct = best.spread_pct
    row.target_price = best.target_price
    row.stop_at = best.stop_at
    row.assigned_cost = best.assigned_cost
    row.cash_required = best.cash_required
    row.score = max(0.0, best.score - penalty)
    row.alternatives = [
        {
            "strike": p.strike,
            "bid": p.contract.bid,
            "ask": p.contract.ask,
            "spread_pct": p.spread_pct,
            "volume": p.contract.volume or 0,
            "open_interest": p.contract.open_interest or 0,
            "delta": p.delta,
            "est_decay": p.est_decay,
            "net_edge": p.net_edge,
            "score": max(0.0, p.score - penalty),
        }
        for p in picks[:_ALTERNATIVES]
    ]
    return row


def scan_watchlist(
    dp,
    tickers: list[str],
    params: Optional[PremiumParams] = None,
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
            pool.submit(_scan_one, dp, t, params): t for t in tickers
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


def apply_iv_floor(rows: list[ScanRow], min_iv_rank: float) -> None:
    """Mark tradeable rows below the IV-rank floor as no-trade (reversible)."""
    for r in rows:
        if r.no_trade_reason.startswith("IV rank"):
            r.has_no_trade = False
            r.no_trade_reason = ""
        if r.tradeable and r.iv_rank < min_iv_rank:
            r.has_no_trade = True
            r.no_trade_reason = f"IV rank {r.iv_rank:.0f} < {min_iv_rank:.0f}"


def upcoming_earnings(
    rows: list[ScanRow], horizon_days: int = _EARNINGS_HORIZON_DAYS,
) -> list[tuple[str, date]]:
    """(ticker, earnings date) for rows reporting within *horizon_days*."""
    today = _market_today()
    out: list[tuple[str, date]] = []
    for r in rows:
        try:
            ed = date.fromisoformat(r.earnings_date) if r.earnings_date else None
        except (TypeError, ValueError):
            ed = None
        if ed and 0 <= (ed - today).days <= horizon_days:
            out.append((r.ticker, ed))
    return sorted(out, key=lambda pair: pair[1])


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------

def _row_color(row: ScanRow) -> tuple[str, str]:
    """(background, border) for a row by score; muted if not tradeable."""
    if not row.tradeable:
        return _COLOR_MUTED
    if row.score >= 70:
        return _COLOR_STRONG
    if row.score >= 50:
        return _COLOR_OK
    return _COLOR_WEAK


def _fmt_money(v: float) -> str:
    return f"${v:,.2f}" if v else "—"


def _cell(label: str, value: str, width: str = "1") -> str:
    return (
        f'<div style="flex:{width};min-width:0;">'
        f'<div style="font-size:10px;color:#8b949e;text-transform:uppercase;'
        f'letter-spacing:.04em;">{label}</div>'
        f'<div style="font-size:14px;color:#e6edf3;font-weight:600;">{value}</div>'
        f"</div>"
    )


def _event_banner(row: ScanRow) -> str:
    """
    Loud strip at the top of a card explaining WHY premium is high:
    earnings before this put expires, else an unexplained IV spike.
    Empty when neither applies.
    """
    pill = (
        "display:inline-block;padding:2px 8px;border-radius:10px;"
        "font-size:11px;font-weight:700;letter-spacing:.04em;margin-right:8px;"
    )
    if row.earnings_warning and row.earnings_date:
        ed = date.fromisoformat(row.earnings_date)
        spike = (
            f" IV is {row.iv_spike_ratio:.1f}× its normal move."
            if row.iv_spike else ""
        )
        return (
            f'<div style="background:#3d2a0a;border:1px solid #d29922;'
            f'border-radius:6px;padding:6px 10px;margin-bottom:8px;'
            f'font-size:12px;color:#f0c674;">'
            f'<span style="{pill}background:#d29922;color:#0d1117;">'
            f"📅 EARNINGS {_when(ed).upper()}</span>"
            f"<b>{ed:%a %b %d}</b> — before this put expires. The high premium "
            f"is earnings premium. Closing today keeps you out of the gap, but "
            f"that premium won't bleed off intraday — it collapses after the "
            f"report — so today's decay estimate is halved. If you can't close "
            f"by {CLOSE_BY}, you're holding through earnings.{spike}</div>"
        )
    if row.iv_spike:
        return (
            f'<div style="background:#2a1a3d;border:1px solid #a371f7;'
            f'border-radius:6px;padding:6px 10px;margin-bottom:8px;'
            f'font-size:12px;color:#d2b8ff;">'
            f'<span style="{pill}background:#a371f7;color:#0d1117;">⚡ IV SPIKE</span>'
            f"IV {row.atm_iv * 100:.0f}% is {row.iv_spike_ratio:.1f}× the stock's "
            f"20-day move ({row.realized_vol * 100:.0f}%) — the market is pricing "
            f"an event. No earnings date found: check for earnings or news.</div>"
        )
    return ""


def _badges(row: ScanRow) -> str:
    out = []
    if row.earnings_date and not row.earnings_warning:
        ed = date.fromisoformat(row.earnings_date)
        if 0 <= (ed - _market_today()).days <= _EARNINGS_HORIZON_DAYS:
            out.append(f"📅 Next earnings {ed:%a %b %d} — after this expiration")
    if row.downtrend:
        out.append(f"📉 Downtrend (−{_DOWNTREND_PENALTY:.0f} score)")
    if row.tradeable and row.outside_expected_move:
        out.append("🛡️ Outside expected move")
    if row.tradeable and row.support is not None:
        out.append(f"🧱 Support ${row.support:,.2f} above strike")
    return "  ·  ".join(out)


def _plan_line(row: ScanRow, fits: Optional[int]) -> str:
    discount = (row.price - row.assigned_cost) / row.price if row.price else 0.0
    captured = row.est_decay / row.credit if row.credit else 0.0
    parts = [
        f"🎯 Target ~<b>${row.target_price:.2f}</b> by the close if the stock "
        f"holds ({captured:.0%} of the credit)",
        f"🛑 Stop at <b>${row.stop_at:.2f}</b>",
        f"⏰ Close by <b>{CLOSE_BY}</b>",
        f"🏠 Can't close? Assigned at <b>${row.assigned_cost:,.2f}</b> net "
        f"({discount:.1%} below today) · ${row.cash_required:,.0f} cash",
    ]
    if fits is not None:
        parts.append(
            f"✅ fits {fits} contract{'s' if fits != 1 else ''}" if fits > 0
            else "⛔ over your exposure limits"
        )
    return "  ·  ".join(parts)


def _render_row(row: ScanRow, fits: Optional[int]) -> None:
    """One scan result: a colour-coded card plus a Log action."""
    bg, border = _row_color(row)
    card_col, action_col = st.columns([13, 2])

    with card_col:
        if row.error or not row.tradeable:
            reason = row.error or row.no_trade_reason
            color = "#f85149" if row.error else "#8b949e"
            extra = _badges(row)
            if row.earnings_warning and row.earnings_date:
                ed = date.fromisoformat(row.earnings_date)
                extra = f"📅 Earnings {_when(ed)}" + (f"  ·  {extra}" if extra else "")
            st.markdown(
                f'<div style="background:{bg};border:1px solid {border};'
                f'border-left:4px solid {border};border-radius:8px;'
                f'padding:10px 14px;margin-bottom:6px;">'
                f'<span style="font-weight:700;color:#e6edf3;">{row.ticker}</span>'
                f'<span style="color:{color};margin-left:12px;">⛔ {reason}</span>'
                + (f'<span style="color:#8b949e;font-size:11px;margin-left:12px;">{extra}</span>'
                   if extra else "")
                + "</div>",
                unsafe_allow_html=True,
            )
            return

        change_color = "#3fb950" if row.change_pct >= 0 else "#f85149"
        price_html = (
            f'{_fmt_money(row.price)} '
            f'<span style="color:{change_color};font-size:11px;">'
            f'{row.change_pct * 100:+.1f}%</span>'
        )
        iv_html = (
            f"{row.atm_iv * 100:.0f}% · "
            f"{row.iv_rank:.0f}{'' if row.iv_rank_source == 'iv' else '≈'}"
        )
        delta_txt = f" · Δ{row.delta:.2f}" if row.delta is not None else ""
        cells = "".join([
            _cell(
                "Ticker",
                f"<b>{row.ticker}</b>"
                + (" 📅" if row.earnings_warning else " ⚡" if row.iv_spike else ""),
                "0.9",
            ),
            _cell("Price", price_html, "1.3"),
            _cell("Expires", f"{row.expiration[5:]} · {row.dte}d", "1.1"),
            _cell("IV · Rank", iv_html, "1.1"),
            _cell("Sell put", f"${row.strike:g}{delta_txt}", "1.2"),
            _cell(
                "Bid / Ask",
                f"${row.bid:.2f} / ${row.ask:.2f}"
                f"<span style='font-size:11px;color:#8b949e;'> ({row.spread_pct * 100:.0f}%)</span>",
                "1.5",
            ),
            _cell("Vol / OI", f"{row.volume:,} / {row.open_interest:,}", "1.1"),
            _cell(
                "Decay today",
                f"${row.est_decay * 100:,.0f}" + (
                    "<span style='font-size:11px;color:#d29922;'> (halved)</span>"
                    if row.event_haircut else ""
                ),
                "0.9",
            ),
            _cell(
                "Net edge",
                f"${row.net_edge * 100:,.0f}"
                f"<span style='font-size:11px;color:#8b949e;'> ({row.edge_pct * 100:.2f}%)</span>",
                "1.2",
            ),
            _cell("OTM", f"{row.distance_pct * 100:.1f}%", "0.7"),
            _cell("Score", f'<span style="color:{border};">{row.score:.0f}</span>', "0.6"),
        ])
        badges = _badges(row)
        st.markdown(
            f'<div style="background:{bg};border:1px solid {border};'
            f'border-left:4px solid {border};border-radius:8px;'
            f'padding:10px 14px;margin-bottom:6px;">'
            f"{_event_banner(row)}"
            f'<div style="display:flex;gap:14px;align-items:center;">{cells}</div>'
            f'<div style="font-size:12px;color:#c9d1d9;margin-top:6px;">'
            f"{_plan_line(row, fits)}</div>"
            + (f'<div style="font-size:11px;color:#d29922;margin-top:4px;">{badges}</div>'
               if badges else "")
            + "</div>",
            unsafe_allow_html=True,
        )
        if len(row.alternatives) > 1:
            with st.expander(f"Compare strikes — {row.ticker}"):
                _render_alternatives(row)

    with action_col:
        if st.button(
            "📒 Log", key=f"scan_log_{row.ticker}",
            type="primary" if row.score >= 70 else "secondary",
            use_container_width=True,
            help="Pre-fill this put in 📒 Ledger → Log a Trade",
        ):
            ledger_ui.queue_trade_prefill(
                ticker=row.ticker,
                strike=row.strike,
                expiration=date.fromisoformat(row.expiration),
                price=row.credit,
                note=f"scanner score {row.score:.0f}" + (
                    f" · earnings {row.earnings_date}" if row.earnings_warning else ""
                ),
            )
            st.toast(
                f"{row.ticker} ${row.strike:g}P pre-filled — open the 📒 Ledger tab."
            )


def _render_alternatives(row: ScanRow) -> None:
    """Top strikes side by side, so the pick is explainable."""
    st.dataframe(
        [{
            "Strike": a["strike"],
            "Bid": a["bid"],
            "Ask": a["ask"],
            "Spread %": a["spread_pct"] * 100,
            "Volume": a["volume"],
            "Open int.": a["open_interest"],
            "Delta": a["delta"],
            "Decay today $": a["est_decay"] * 100,
            "Net edge $": a["net_edge"] * 100,
            "Score": a["score"],
        } for a in row.alternatives],
        hide_index=True, use_container_width=True,
        column_config={
            "Strike": st.column_config.NumberColumn(format="$%g"),
            "Bid": st.column_config.NumberColumn(format="$%.2f"),
            "Ask": st.column_config.NumberColumn(format="$%.2f"),
            "Spread %": st.column_config.NumberColumn(format="%.1f%%"),
            "Delta": st.column_config.NumberColumn(format="%.2f"),
            "Decay today $": st.column_config.NumberColumn(format="$%.0f"),
            "Net edge $": st.column_config.NumberColumn(format="$%.0f"),
            "Score": st.column_config.NumberColumn(format="%.0f"),
        },
    )
    st.caption(
        "Per contract. Decay today = theta × session hours left, capped at 70% "
        "of the credit. Net edge = decay − spread (you pay the spread "
        "getting in and out). Score: net edge 35% · liquidity (spread, "
        "volume, open interest) 25% · delta 20% · expected move 10% · "
        "IV rank 5% · support 5%."
    )


def _render_summary(rows: list[ScanRow]) -> None:
    """Summary cards beneath the list."""
    tradeable = [r for r in rows if r.tradeable]
    top = max(tradeable, key=lambda r: r.score, default=None)
    richest = max(tradeable, key=lambda r: r.iv_rank, default=None)
    best_edge = max(tradeable, key=lambda r: r.net_edge * 100, default=None)
    strong = sum(1 for r in tradeable if r.score >= 70)
    earnings_ct = sum(1 for r in tradeable if r.earnings_warning)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        if top:
            st.metric("🏆 Top pick", f"{top.ticker} ${top.strike:g}P", f"Score {top.score:.0f}")
        else:
            st.metric("🏆 Top pick", "—", "No setup")
    with c2:
        if richest:
            st.metric("🔥 Richest IV", richest.ticker, f"Rank {richest.iv_rank:.0f}")
        else:
            st.metric("🔥 Richest IV", "—")
    with c3:
        if best_edge:
            st.metric(
                "💰 Biggest net edge", best_edge.ticker,
                f"${best_edge.net_edge * 100:,.0f}/contract today",
            )
        else:
            st.metric("💰 Biggest net edge", "—")
    with c4:
        st.metric("✅ Strong setups", str(strong), "score ≥ 70")
    with c5:
        st.metric("📅 Earnings in window", str(earnings_ct), "gap risk", delta_color="off")

    upcoming = upcoming_earnings(rows)
    if upcoming:
        st.markdown(f"##### 📅 Earnings in the next {_EARNINGS_HORIZON_DAYS} days")
        st.markdown("  ·  ".join(
            f"**{t}** {d:%a %b %d} ({(d - date.today()).days}d)"
            for t, d in upcoming
        ))


def _reset_watchlist() -> None:
    st.session_state["scanner_watchlist_raw"] = ", ".join(DEFAULT_WATCHLIST)


def _render_controls(dp) -> dict:
    """Scanner control panel. Returns the resolved settings + scan trigger."""
    st.markdown("#### 🔍 Scan Controls")
    st.caption(
        "Sells puts on the earliest expiration — a daily if the ticker has "
        f"one, else the nearest weekly, never more than {MAX_DTE} days out. "
        "Today's expiration counts until the 4:00 pm ET close."
    )
    c1, c2 = st.columns(2)
    with c1:
        min_score = st.slider(
            "Min score", 30, 80, 50, step=5, key="scanner_min_score",
        )
    with c2:
        min_iv_rank = st.slider(
            "Min IV rank", 0, 80, 30, step=5, key="scanner_min_iv",
            help="Hide tickers whose IV isn't rich versus their own history.",
        )

    # Seed the editor from the persisted watchlist on first render, then
    # let Streamlit's widget state own it for the rest of the session.
    if "scanner_watchlist_raw" not in st.session_state:
        st.session_state["scanner_watchlist_raw"] = ", ".join(
            _load_saved_watchlist()
        )

    watchlist_raw = st.text_area(
        "Watchlist — stocks you'd be happy to own if assigned",
        height=90,
        key="scanner_watchlist_raw",
    )
    watchlist = _parse_watchlist(watchlist_raw)

    # Persist whenever the parsed list changes so edits survive restarts.
    if watchlist and watchlist != st.session_state.get("scanner_watchlist_saved"):
        _save_watchlist(watchlist)
        st.session_state["scanner_watchlist_saved"] = watchlist

    b1, b2 = st.columns([4, 1])
    with b1:
        scan_clicked = st.button(
            "🔄 Scan All",
            type="primary",
            use_container_width=True,
            disabled=(dp is None or not watchlist),
            key="scanner_scan_btn",
        )
    with b2:
        st.button(
            "↺ Default list", on_click=_reset_watchlist,
            use_container_width=True, key="scanner_reset_btn",
            help=f"Restore the {len(DEFAULT_WATCHLIST)}-ticker quality list.",
        )

    return {
        "min_score": min_score,
        "min_iv_rank": min_iv_rank,
        "watchlist": watchlist,
        "scan_clicked": scan_clicked,
    }


def _exposure_fits() -> Callable[[ScanRow], Optional[int]]:
    """
    fits(row) → contracts of that put that fit the Ledger's exposure
    limits, or None when no account size is set.
    """
    settings = ledger_ui.load_settings()
    account = float(settings.get("account_size") or 0)
    if account <= 0:
        return lambda row: None
    try:
        _, ledger = ledger_ui.load_ledger()
    except Exception:  # noqa: BLE001
        return lambda row: None
    report = compute_exposure(
        list(ledger.positions.values()), account,
        settings["max_ticker_pct"], settings["max_total_pct"],
    )

    def fits(row: ScanRow) -> Optional[int]:
        return max_new_csp_contracts(
            report, row.ticker, row.strike or 0.0,
            settings["max_ticker_pct"], settings["max_total_pct"],
        )
    return fits


def _session_note(now: Optional[datetime] = None) -> tuple[str, bool]:
    """(message, is_warning) describing the session for same-day trades."""
    now = (now or datetime.now(_MARKET_TZ)).astimezone(_MARKET_TZ)
    t = now.time()
    if now.weekday() >= 5 or t >= _MARKET_CLOSE:
        return ("Market closed — decay estimates are for the next full session.", False)
    if t < _MARKET_OPEN:
        return ("Market opens 9:30 am ET — estimates assume a full session.", False)
    left = session_hours_left(now)
    if left < 0.5:
        return (
            f"Under 30 minutes left in the session — too late to open a "
            f"same-day trade (close by {CLOSE_BY}).",
            True,
        )
    h, m = int(left), int(round((left - int(left)) * 60))
    return (
        f"Session: {h}h {m:02d}m left — decay estimates cover the rest of today. "
        f"Close everything by {CLOSE_BY}.",
        False,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def render_scanner(dp=None) -> None:
    """Render the Daily Scanner tab."""
    st.subheader("☀️ Daily Scanner — Same-Day CSP")
    st.caption(
        "The best put to sell now and buy back before today's close, on a "
        "stock you'd own — assignment is only the worst case if you can't "
        "close. Ranked by what today's decay should earn after the spread."
    )

    if dp is None:
        st.warning("No data provider available — cannot scan.", icon="⚠️")
        return

    cfg = _render_controls(dp)
    st.divider()

    fits = _exposure_fits()
    params = PremiumParams()

    note, warn = _session_note()
    (st.warning if warn else st.info)(note, icon="⏰")

    cache_key = (tuple(cfg["watchlist"]), date.today().isoformat())
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

        rows = scan_watchlist(dp, cfg["watchlist"], params, progress_cb=_bump)
        progress.empty()
        st.session_state["scanner_results"] = rows
        st.session_state["scanner_scanned_at"] = datetime.now()
        st.session_state["scanner_cache_key"] = cache_key
        cached_at = st.session_state["scanner_scanned_at"]

    if not rows:
        st.info(
            "Click **🔄 Scan All** to find the richest puts to sell on your "
            "watchlist.",
            icon="👆",
        )
        return

    # IV-rank floor is applied live so the slider stays instant.
    apply_iv_floor(rows, cfg["min_iv_rank"])

    # Tradeable rows by score, then everything muted at the bottom.
    visible = sorted(
        (r for r in rows if not r.tradeable or r.score >= cfg["min_score"]),
        key=lambda r: (0 if r.tradeable else 1, -r.score),
    )
    hidden = len(rows) - len(visible)

    stamp = cached_at.strftime("%I:%M %p").lstrip("0") if cached_at else "—"
    st.caption(
        f"Last scanned: {stamp}  ·  {len(rows)} tickers  ·  earliest expiration ≤ {MAX_DTE}d"
        + (f"  ·  {hidden} below min score hidden" if hidden > 0 else "")
    )
    if any(r.iv_rank_source != "iv" for r in rows if r.tradeable):
        st.caption(
            "IV Rank ≈ = estimated from realised volatility. Each scan stores "
            f"today's implied vol; after {_IV_HISTORY_MIN_POINTS} days of "
            "history a ticker switches to true IV rank."
        )

    if not visible:
        st.info("No tickers meet the current filters.", icon="🔍")
        return

    for r in visible:
        _render_row(r, fits(r) if r.tradeable else None)

    st.divider()
    _render_summary(rows)
