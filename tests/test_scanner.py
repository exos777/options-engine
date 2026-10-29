"""Tests for ui/scanner.py — pure scan/render logic (no network)."""

from datetime import date, datetime, time, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest

from strategies.models import ChartRegime
import ui.scanner as scanner
from ui.scanner import (
    DEFAULT_WATCHLIST,
    ScanRow,
    _COLOR_MUTED,
    _COLOR_OK,
    _COLOR_STRONG,
    _COLOR_WEAK,
    _atm_iv,
    _MARKET_TZ,
    _earnings_in_window,
    _iv_rank,
    _load_saved_watchlist,
    _parse_watchlist,
    _pick_expiration,
    _row_color,
    _resolve_iv_rank,
    _save_watchlist,
    _event_banner,
    _realized_vol,
    _resolve_earnings,
    _scan_one,
    _session_note,
    _true_iv_rank,
    _when,
    apply_iv_floor,
    session_hours_left,
    upcoming_earnings,
)
from data import store
from strategies.models import OptionContract


# ── _pick_expiration ────────────────────────────

def _exp(days_out: int) -> str:
    return (date.today() + timedelta(days=days_out)).isoformat()


def _at(hour: int, minute: int = 0) -> datetime:
    """Today in market time, at the given clock time."""
    return datetime.combine(date.today(), time(hour, minute), tzinfo=_MARKET_TZ)


MORNING = _at(10)
EVENING = _at(16, 30)


def test_pick_expiration_picks_earliest_daily():
    exps = [_exp(1), _exp(6), _exp(14), _exp(30)]
    assert _pick_expiration(exps, MORNING) == (_exp(1), 1)


def test_pick_expiration_picks_nearest_weekly_when_no_dailies():
    exps = [_exp(3), _exp(10), _exp(17)]
    assert _pick_expiration(exps, MORNING) == (_exp(3), 3)


def test_pick_expiration_ignores_past_dates():
    chosen, _ = _pick_expiration([_exp(-3), _exp(6)], MORNING)
    assert chosen == _exp(6)


def test_pick_expiration_uses_today_while_market_open():
    assert _pick_expiration([_exp(0), _exp(1)], MORNING) == (_exp(0), 0)


def test_pick_expiration_skips_today_after_the_close():
    assert _pick_expiration([_exp(0), _exp(1)], EVENING) == (_exp(1), 1)


def test_pick_expiration_never_more_than_a_week_out():
    assert _pick_expiration([_exp(7), _exp(14)], MORNING) == (_exp(7), 7)
    with pytest.raises(ValueError):
        _pick_expiration([_exp(8), _exp(14)], MORNING)


def test_pick_expiration_raises_when_no_future_expirations():
    with pytest.raises(ValueError):
        _pick_expiration([_exp(-1), _exp(-5)], MORNING)


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


# ── _row_color ──────────────────────────────────

def _tradeable(score: float) -> ScanRow:
    return ScanRow(ticker="X", strike=100.0, score=score)


def test_row_color_buckets_by_score():
    assert _row_color(_tradeable(80.0)) == _COLOR_STRONG
    assert _row_color(_tradeable(60.0)) == _COLOR_OK
    assert _row_color(_tradeable(30.0)) == _COLOR_WEAK


def test_row_color_muted_when_not_tradeable_but_earnings_stays_colored():
    no_trade = ScanRow(ticker="X", strike=100.0, score=90.0,
                       has_no_trade=True, no_trade_reason="x")
    errored = ScanRow(ticker="X", error="boom")
    no_pick = ScanRow(ticker="X")
    assert _row_color(no_trade) == _COLOR_MUTED
    assert _row_color(errored) == _COLOR_MUTED
    assert _row_color(no_pick) == _COLOR_MUTED
    earnings = ScanRow(ticker="X", strike=100.0, score=90.0, earnings_warning=True)
    assert _row_color(earnings) == _COLOR_STRONG     # allowed, just flagged


# ── IV floor ────────────────────────────────────

def test_apply_iv_floor_is_reversible():
    rows = [ScanRow(ticker="A", strike=10.0, score=80, iv_rank=20),
            ScanRow(ticker="B", strike=10.0, score=80, iv_rank=60)]
    apply_iv_floor(rows, 30)
    assert [r.tradeable for r in rows] == [False, True]
    assert rows[0].no_trade_reason == "IV rank 20 < 30"
    apply_iv_floor(rows, 10)
    assert [r.tradeable for r in rows] == [True, True]


# ── _scan_one end to end (fake provider) ────────

@pytest.fixture(autouse=True)
def _fresh_daily_cache():
    """The per-day data cache is module-global; isolate each test."""
    scanner._daily_cache.clear()
    yield
    scanner._daily_cache.clear()


class _FakeDP:
    """Spot 100, one expiration in 2 days, a ladder of OTM puts."""
    __name__ = "fake_dp"

    def __init__(self, exp_days=2, earnings=None, closes=None):
        self.exp = _exp(exp_days)
        self.earnings = earnings
        self.closes = closes or [100 + (i % 5) for i in range(130)]

    def get_quote(self, t):
        return SimpleNamespace(price=100.0, change_pct=-0.01, earnings_date=self.earnings)

    def get_expirations(self, t):
        return (self.exp, _exp(30))

    def get_option_chain(self, t, exp):
        def put(k, bid, ask, d):
            return OptionContract(strike=k, expiration=exp, option_type="put",
                                  bid=bid, ask=ask, last=bid, volume=500,
                                  open_interest=1000, implied_volatility=0.45, delta=-d,
                                  theta=-bid * 0.6)
        puts = [put(99, 1.40, 1.45, 0.40), put(97, 0.80, 0.83, 0.22),
                put(95, 0.40, 0.42, 0.12), put(90, 0.04, 0.06, 0.03)]
        return [], puts

    def get_historical(self, t, months=6):
        n = len(self.closes)
        return pd.DataFrame({
            "Open": self.closes, "High": [c + 1 for c in self.closes],
            "Low": [c - 1 for c in self.closes], "Close": self.closes,
            "Volume": [1_000_000] * n,
        }, index=pd.date_range(end=date.today(), periods=n, freq="D"))


def test_scan_one_picks_a_put_and_builds_a_plan():
    row = _scan_one(_FakeDP(), "TEST", now=MORNING)
    assert row.error == ""
    assert row.tradeable
    assert row.expiration == _exp(2) and row.dte == 2
    assert row.strike in (97, 95)                 # 0.40Δ and 0.03Δ excluded
    assert row.hours_left == pytest.approx(6.0)   # 10:00 am → 6h to the close
    assert row.net_edge == pytest.approx(row.est_decay - (row.ask - row.bid))
    assert row.target_price == pytest.approx(row.credit - row.est_decay, abs=0.01)
    assert row.assigned_cost == pytest.approx(row.strike - row.credit, abs=0.01)
    assert row.cash_required == row.strike * 100
    assert row.alternatives and row.alternatives[0]["strike"] == row.strike
    assert not row.earnings_warning and not row.event_haircut


def test_scan_one_flags_but_allows_earnings_and_halves_decay():
    plain = _scan_one(_FakeDP(), "TEST", now=MORNING)
    row = _scan_one(_FakeDP(earnings=_exp(1)), "TEST", now=MORNING)
    assert row.tradeable and row.earnings_warning and row.event_haircut
    if row.strike == plain.strike:
        assert row.est_decay == pytest.approx(plain.est_decay / 2)


def test_scan_one_no_trade_when_nothing_within_a_week():
    row = _scan_one(_FakeDP(exp_days=9), "TEST", now=MORNING)
    assert not row.tradeable and row.error == ""
    assert row.no_trade_reason == "No expiry ≤ 7d"


# ── watchlist default ───────────────────────────

def test_default_watchlist_is_alphabetical_and_contains_core_names():
    assert DEFAULT_WATCHLIST == sorted(DEFAULT_WATCHLIST)
    for sym in ("TSLA", "NVDA", "MSFT", "SPY", "QQQ", "PLTR"):
        assert sym in DEFAULT_WATCHLIST


# ── watchlist parsing ───────────────────────────

def test_parse_watchlist_dedupes_uppercases_and_splits():
    raw = "tsla, NVDA\nmsft , tsla,, spy"
    assert _parse_watchlist(raw) == ["TSLA", "NVDA", "MSFT", "SPY"]


def test_parse_watchlist_empty_string_yields_empty_list():
    assert _parse_watchlist("   ") == []


# ── watchlist persistence ───────────────────────

def test_save_and_load_watchlist_round_trips(tmp_path, monkeypatch):
    prefs = tmp_path / "scanner_prefs.json"
    monkeypatch.setattr(scanner, "_PREFS_PATH", prefs)
    custom = ["AMD", "PLTR", "SNOW"]
    _save_watchlist(custom)
    assert _load_saved_watchlist() == custom


def test_load_watchlist_falls_back_to_default_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(scanner, "_PREFS_PATH", tmp_path / "nope.json")
    assert _load_saved_watchlist() == list(DEFAULT_WATCHLIST)


def test_load_watchlist_cleans_dupes_and_falls_back_on_garbage(tmp_path, monkeypatch):
    prefs = tmp_path / "scanner_prefs.json"
    monkeypatch.setattr(scanner, "_PREFS_PATH", prefs)
    prefs.write_text('{"watchlist": ["aapl", "AAPL", " msft "]}', encoding="utf-8")
    assert _load_saved_watchlist() == ["AAPL", "MSFT"]

    prefs.write_text("not json at all", encoding="utf-8")
    assert _load_saved_watchlist() == list(DEFAULT_WATCHLIST)


def test_watchlist_persists_in_store_over_legacy_file(tmp_path, monkeypatch):
    legacy = tmp_path / "scanner_prefs.json"
    monkeypatch.setattr(scanner, "_PREFS_PATH", legacy)
    legacy.write_text('{"watchlist": ["AAPL"]}', encoding="utf-8")
    assert _load_saved_watchlist() == ["AAPL"]       # legacy file carried over
    _save_watchlist(["NVDA", "AMD"])
    assert _load_saved_watchlist() == ["NVDA", "AMD"]  # store wins once saved


# ── true IV rank from stored history ────────────

def test_true_iv_rank_needs_enough_history():
    assert _true_iv_rank([0.30] * 5, atm_iv=0.40, min_points=20) is None
    assert _true_iv_rank([0.30] * 25, atm_iv=0.0) is None


def test_true_iv_rank_places_current_iv_in_its_range():
    history = [0.20 + i * 0.01 for i in range(21)]     # 0.20 … 0.40
    assert _true_iv_rank(history, atm_iv=0.30) == pytest.approx(50.0)
    assert _true_iv_rank(history, atm_iv=0.50) == pytest.approx(100.0)
    assert _true_iv_rank([0.30] * 21, atm_iv=0.30) == 50.0


def test_resolve_iv_rank_records_and_switches_to_true_rank():
    close = pd.Series([100 + (i % 3) * 0.1 for i in range(60)])
    df = pd.DataFrame({"Close": close})

    rank, source = _resolve_iv_rank("ZZZ", df, 0.30)
    assert source == "rv"
    assert store.iv_history("ZZZ", date.today()) == [(date.today(), 0.30)]

    for i in range(1, 21):
        store.record_iv("ZZZ", date.today() - timedelta(days=i), 0.20 + i * 0.01)
    rank, source = _resolve_iv_rank("ZZZ", df, 0.30)
    assert source == "iv"
    assert rank == pytest.approx((0.30 - 0.21) / (0.40 - 0.21) * 100)


# ── upcoming earnings ───────────────────────────

def test_upcoming_earnings_lists_only_horizon_sorted():
    rows = [
        ScanRow(ticker="AAA", earnings_date=_exp(10)),
        ScanRow(ticker="BBB", earnings_date=_exp(3)),
        ScanRow(ticker="CCC", earnings_date=_exp(30)),
        ScanRow(ticker="DDD", earnings_date=_exp(-1)),
        ScanRow(ticker="EEE", earnings_date="garbage"),
        ScanRow(ticker="FFF"),
    ]
    assert [t for t, _ in upcoming_earnings(rows, horizon_days=14)] == ["BBB", "AAA"]


# ── earnings + IV spike ─────────────────────────

class _EarningsDP(_FakeDP):
    def __init__(self, lookup, **kw):
        super().__init__(**kw)
        self.lookup = lookup

    def get_earnings_date(self, t):
        return self.lookup


def test_resolve_earnings_prefers_provider_lookup():
    quote = SimpleNamespace(earnings_date=_exp(20))
    assert _resolve_earnings(_EarningsDP(_exp(1)), "T", quote) == _exp(1)


def test_resolve_earnings_ignores_a_past_quote_date():
    assert _resolve_earnings(_EarningsDP(None), "T", SimpleNamespace(earnings_date=_exp(-90))) is None
    assert _resolve_earnings(_EarningsDP(None), "T", SimpleNamespace(earnings_date=_exp(3))) == _exp(3)


def test_scan_one_uses_provider_earnings_lookup():
    row = _scan_one(_EarningsDP(_exp(1)), "TEST", now=MORNING)
    assert row.earnings_date == _exp(1)
    assert row.earnings_warning and row.tradeable


def test_realized_vol_and_iv_spike():
    flat = pd.DataFrame({"Close": [100.0] * 30})
    assert _realized_vol(flat) == 0.0
    assert _realized_vol(pd.DataFrame({"Close": [100.0] * 5})) == 0.0
    wiggly = pd.DataFrame({"Close": [100 * (1.01 if i % 2 else 0.99) for i in range(40)]})
    assert _realized_vol(wiggly) > 0.2

    row = ScanRow(ticker="AMZN", atm_iv=1.34, realized_vol=0.30)
    assert row.iv_spike_ratio == pytest.approx(1.34 / 0.30)
    assert row.iv_spike
    assert not ScanRow(ticker="X", atm_iv=0.35, realized_vol=0.30).iv_spike
    assert not ScanRow(ticker="X", atm_iv=0.35).iv_spike
    # Calm index ETF: high ratio but low absolute IV — not an event.
    assert not ScanRow(ticker="SPY", atm_iv=0.25, realized_vol=0.09).iv_spike
    # Busy but normal: 2.2× is below the 2.5× bar.
    assert not ScanRow(ticker="X", atm_iv=0.66, realized_vol=0.30).iv_spike


def test_when_labels():
    today = date(2026, 10, 29)
    assert _when(date(2026, 10, 29), today) == "today"
    assert _when(date(2026, 10, 30), today) == "tomorrow"
    assert _when(date(2026, 11, 2), today) == "in 4d (Mon)"
    assert _when(date(2026, 11, 20), today) == "Fri Nov 20"


def test_event_banner_explains_the_premium():
    earnings = ScanRow(ticker="AMZN", strike=237.5, earnings_date=_exp(0),
                       earnings_warning=True, atm_iv=1.34, realized_vol=0.30)
    html = _event_banner(earnings)
    assert "📅 EARNINGS TODAY" in html and "earnings premium" in html
    assert "4.5×" in html

    spike = ScanRow(ticker="AMZN", strike=237.5, atm_iv=1.34, realized_vol=0.30)
    html = _event_banner(spike)
    assert "⚡ IV SPIKE" in html and "No earnings date found" in html

    assert _event_banner(ScanRow(ticker="X", atm_iv=0.3, realized_vol=0.3)) == ""


# ── session timing ──────────────────────────────

def test_session_hours_left():
    assert session_hours_left(_at(8)) == 6.5            # before the open
    assert session_hours_left(_at(9, 30)) == pytest.approx(6.5)
    assert session_hours_left(_at(13)) == pytest.approx(3.0)
    assert session_hours_left(_at(15, 45)) == pytest.approx(0.25)
    assert session_hours_left(_at(17)) == 6.5           # planning the next session


def test_session_note_warns_in_the_last_half_hour():
    weekday = date.today()
    while weekday.weekday() >= 5:
        weekday += timedelta(days=1)
    at = lambda h, m=0: datetime.combine(weekday, time(h, m), tzinfo=_MARKET_TZ)
    msg, warn = _session_note(at(15, 40))
    assert warn and "too late" in msg
    msg, warn = _session_note(at(11))
    assert not warn and "5h 00m left" in msg
    assert "opens" in _session_note(at(8))[0]
    assert "closed" in _session_note(at(16, 30))[0]


# ── watchlist defaults ──────────────────────────

def test_default_watchlist_extends_core_with_similar_names():
    assert len(DEFAULT_WATCHLIST) == len(set(DEFAULT_WATCHLIST)) == 48
    for sym in ("AAPL", "AMD", "AMZN", "GOOG", "INTC", "META", "MSFT", "MU",
                "NFLX", "NVDA", "PLTR", "QQQ", "SNOW", "SPY", "TSLA",
                "AVGO", "AMAT", "TSM", "NOW", "IWM"):
        assert sym in DEFAULT_WATCHLIST
    for speculative in ("COIN", "HOOD", "SMCI", "MSTR"):
        assert speculative not in DEFAULT_WATCHLIST


def test_saved_old_default_upgrades_but_custom_list_is_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(scanner, "_PREFS_PATH", tmp_path / "none.json")
    old_core = ["AAPL", "AMD", "AMZN", "GOOG", "INTC", "META", "MSFT", "MU",
                "NFLX", "NVDA", "PLTR", "QQQ", "SNOW", "SPY", "TSLA"]
    _save_watchlist(old_core)
    assert _load_saved_watchlist() == list(DEFAULT_WATCHLIST)
    _save_watchlist(old_core[:-1] + ["COIN"])
    assert _load_saved_watchlist() == old_core[:-1] + ["COIN"]
