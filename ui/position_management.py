"""
Streamlit UI for the Roll vs Assign position management decision engine.
"""

from __future__ import annotations

from datetime import date
import json
import pathlib

import streamlit as st

import math

from data import store
from data.common import (
    days_to_expiration,
    get_next_weekly_expiration,
    get_selected_contract,
)
from indicators.support_resistance import find_support_resistance
from indicators.technical import calculate_indicators
from scoring.regime import classify_regime
from scoring.position_decision import (
    OpenPosition,
    RollCandidate,
    evaluate_position,
    find_best_roll_for_premium,
    new_effective_cost,
    new_exit_profit,
    roll_vs_assign_verdict,
    verdict_confidence,
)
from scoring.roll_events import (
    EventContext,
    implied_event_move,
    last_expiration_before,
)
from scoring.roll_scoring import (
    RollPicks,
    RollScore,
    format_roll_recommendation,
    pick_top_rolls,
)
from strategies.models import (
    ChartRegime,
    RegimeResult,
    SupportResistanceLevel,
    TechnicalIndicators,
)


# ---------------------------------------------------------------------------
# Form-state persistence (database-backed, survives Railway redeploys)
# ---------------------------------------------------------------------------

_PM_PREFS_KEY = "position_manager.form"

# Pre-database location — read once as a fallback so old entries carry over.
_PM_PREFS_PATH = (
    pathlib.Path(__file__).parent.parent / "data" / "pm_prefs.json"
)

# Keys that are saved across sessions (all pm_* widget keys).
_PM_PERSIST_KEYS: list[str] = [
    "pm_ticker",
    "pm_strategy",
    "pm_strike",
    "pm_premium",
    "pm_price",
    "pm_buy_price",
    "pm_cost_basis",
    "pm_wants_assign",
    "pm_keep_shares",
    "pm_earnings",
    "pm_leg1_sel",
    "pm_leg2_sel",
    "pm_leg1_txt",
    "pm_leg2_txt",
]

# Set by the Ledger tab's "Manage" button; applied before widgets render.
PM_PREFILL_KEY = "_pm_prefill"


def _load_pm_prefs() -> None:
    """Restore saved form values into session_state on first page load."""
    if st.session_state.get("_pm_prefs_loaded"):
        return
    st.session_state["_pm_prefs_loaded"] = True
    try:
        prefs = store.get_pref(_PM_PREFS_KEY)
    except Exception:
        prefs = None
    if prefs is None and _PM_PREFS_PATH.exists():
        try:
            prefs = json.loads(_PM_PREFS_PATH.read_text(encoding="utf-8"))
        except Exception:
            prefs = None
    for k, v in (prefs or {}).items():
        if k in _PM_PERSIST_KEYS and k not in st.session_state:
            st.session_state[k] = v


def _apply_prefill() -> None:
    """Load a position sent over from the Ledger tab into the form."""
    prefill = st.session_state.pop(PM_PREFILL_KEY, None)
    if not prefill:
        return
    for k, v in prefill.items():
        st.session_state[k] = v
    # Force a fresh quote / live-data fetch for the new contract.
    st.session_state.pop("_pm_spot_ticker", None)
    st.session_state.pop("pm_leg1_prev", None)


def _save_pm_prefs() -> None:
    """Persist current form values after every render (only when changed)."""
    prefs = {
        k: st.session_state[k]
        for k in _PM_PERSIST_KEYS
        if k in st.session_state
    }
    if prefs == st.session_state.get("_pm_prefs_saved"):
        return
    try:
        store.set_pref(_PM_PREFS_KEY, prefs)
        st.session_state["_pm_prefs_saved"] = prefs
    except Exception:
        pass


_REC_COLORS = {
    "assign": ("#d29922", "#3a3010"),
    "roll": ("#58a6ff", "#15304a"),
    "close": ("#3fb950", "#1a4a2e"),
    "wait": ("#8b949e", "#21262d"),
}


def _spread_color(pct: float) -> str:
    if pct <= 0.05:
        return "#3fb950"
    if pct <= 0.10:
        return "#d29922"
    return "#f85149"


def _fetch_chart_context(
    dp,
    ticker: str,
    current_price: float,
) -> tuple[
    "TechnicalIndicators | None",
    list[SupportResistanceLevel],
    list[SupportResistanceLevel],
]:
    """
    Fetch ~6 months of OHLCV and derive (indicators, supports, resistances)
    for the roll scoring engine.

    Cached in session_state per (ticker, today) so we don't refetch on
    every Streamlit rerun.
    """
    if dp is None or not ticker:
        return None, [], []

    cache_key = f"{ticker}_{date.today().isoformat()}"
    cache = st.session_state.get("pm_chart_cache", {})
    if cache_key in cache:
        return cache[cache_key]

    try:
        df = dp.get_historical(ticker, months=6)
    except Exception:
        cache[cache_key] = (None, [], [])
        st.session_state["pm_chart_cache"] = cache
        return None, [], []

    try:
        indicators = calculate_indicators(df)
    except Exception:
        indicators = None

    try:
        supports, resistances = find_support_resistance(df, current_price)
    except Exception:
        supports, resistances = [], []

    # 20-day realised vol: the "no event" baseline for roll event checks.
    try:
        close = df["Close"].astype(float)
        rets = (close / close.shift(1)).apply(math.log).dropna().tail(20)
        rv = float(rets.std() * math.sqrt(252))
        st.session_state[f"pm_rv_{cache_key}"] = rv if math.isfinite(rv) else 0.0
    except Exception:
        st.session_state[f"pm_rv_{cache_key}"] = 0.0

    result = (indicators, supports, resistances)
    cache[cache_key] = result
    st.session_state["pm_chart_cache"] = cache
    return result


def _fetch_roll_candidates(
    dp,
    ticker: str,
    pos: OpenPosition,
    leg2_exp: str,
) -> list[RollCandidate]:
    """Fetch the Leg 2 option chain and build RollCandidate list."""
    candidates: list[RollCandidate] = []
    if not leg2_exp:
        return candidates

    try:
        dte = days_to_expiration(leg2_exp)
    except Exception:
        return candidates

    try:
        calls, puts = dp.get_option_chain(ticker, leg2_exp)
    except Exception:
        return candidates

    chain_type = "puts" if pos.strategy == "CSP" else "calls"
    chain = puts if chain_type == "puts" else calls
    strike_lo = pos.strike * 0.90
    strike_hi = pos.strike * 1.10
    buy_to_close = pos.current_ask  # pay the ask to close short

    for opt in chain:
        if not (strike_lo <= opt.strike <= strike_hi):
            continue
        if opt.mid <= 0:
            continue
        mid = opt.mid
        bid = opt.bid
        ask = opt.ask
        spread_pct = (ask - bid) / max(mid, 0.01)

        sell_to_open = bid  # receive the bid when opening the new short
        roll_credit = sell_to_open - buy_to_close

        if opt.strike < pos.strike:
            roll_type = "down"
        elif opt.strike > pos.strike:
            roll_type = "up"
        else:
            roll_type = "out"

        candidates.append(RollCandidate(
            strike=opt.strike,
            expiration=leg2_exp,
            dte=dte,
            bid=bid,
            ask=ask,
            mid=mid,
            delta=abs(opt.delta) if opt.delta is not None else 0.0,
            open_interest=opt.open_interest or 0,
            roll_credit=roll_credit,
            roll_type=roll_type,
            spread_pct=spread_pct,
            buy_to_close=buy_to_close,
            sell_to_open=sell_to_open,
            iv=opt.implied_volatility or 0.0,
        ))
    return candidates


def _lookup_earnings(dp, ticker: str, quote) -> str | None:
    """Next earnings date (ISO) — the provider's lookup, else the quote's."""
    found = None
    if hasattr(dp, "get_earnings_date"):
        try:
            found = dp.get_earnings_date(ticker)
        except Exception:
            found = None
    if not found:
        quoted = getattr(quote, "earnings_date", None)
        if quoted and str(quoted)[:10] >= date.today().isoformat():
            found = str(quoted)[:10]
    return found


def _do_fetch(dp, ticker: str, expiration: str, strike: float, strategy: str) -> None:
    """Fetch live quote + option data for the selected expiration/strike."""
    # Clear the previous contract's quote first: a failed lookup must never
    # leave another contract's bid/ask behind to be used as the close price.
    for k in ("pm_auto_bid", "pm_auto_ask", "pm_auto_iv", "pm_auto_delta"):
        st.session_state[k] = 0.0
    st.session_state["pm_btc"] = 0.0
    try:
        quote = dp.get_quote(ticker)
        st.session_state["pm_auto_price"] = quote.price
        # Write to the widget key directly so the field reflects live price.
        st.session_state["pm_price"] = quote.price

        st.session_state["pm_earnings_date"] = _lookup_earnings(dp, ticker, quote)

        selected = get_selected_contract(dp, ticker, expiration, strike, strategy)
        st.session_state["pm_auto_dte"] = selected.dte

        if selected.found:
            contract = selected.contract
            st.session_state["pm_auto_bid"] = contract.bid
            st.session_state["pm_auto_ask"] = contract.ask
            # Buying back a short pays the ask.
            st.session_state["pm_btc"] = float(contract.ask)
            st.session_state["pm_auto_iv"] = contract.implied_volatility or 0.0
            st.session_state["pm_auto_delta"] = (
                abs(contract.delta) if contract.delta else 0.0
            )
            st.session_state["pm_fetch_ok"] = True
            st.session_state["pm_fetch_err"] = ""
        else:
            st.session_state["pm_fetch_ok"] = False
            st.session_state["pm_fetch_err"] = (
                selected.error
                or "Strike not found for selected expiration."
            )
    except Exception as e:
        st.session_state["pm_fetch_ok"] = False
        st.session_state["pm_fetch_err"] = str(e)


def _btc_known(current_ask: float) -> bool:
    """Every roll nets against the buy-to-close price — refuse to guess it."""
    if current_ask > 0:
        return True
    st.error(
        "Buy-to-close price for Leg 1 is $0, so roll credits would leave out "
        "the cost of closing your current option. Click **🔄 Fetch Live "
        "Data**, or enter the Leg 1 ask from your broker.",
        icon="🚨",
    )
    return False


def render_position_manager(dp=None) -> None:
    _load_pm_prefs()
    _apply_prefill()

    st.subheader("\U0001f504 Roll vs Assign Decision Engine")
    st.caption(
        "Evaluate an open CSP or CC position and get a recommendation: "
        "assign, roll, close, or wait."
    )

    # ── Ticker + Fetch row ─────────────────────────────
    col_ticker, col_leg1, col_leg2, col_fetch = st.columns([2, 2, 2, 1])
    with col_ticker:
        ticker = st.text_input(
            "Ticker", value="TSLA", key="pm_ticker",
        ).upper().strip()

    # Auto-populate spot price as soon as the ticker is entered.
    # This runs on every render but is cheap — one fast_info call —
    # and is guarded so it only fires when the ticker changes.
    if dp is not None and ticker:
        if st.session_state.get("_pm_spot_ticker") != ticker:
            try:
                _q = dp.get_quote(ticker)
                st.session_state["pm_auto_price"] = _q.price
                st.session_state["pm_price"] = _q.price
                st.session_state["_pm_spot_ticker"] = ticker
            except Exception:
                pass

    if dp is not None and ticker:
        try:
            exps = dp.get_expirations(ticker)
            exp_list = list(exps)
        except Exception:
            exp_list = []
    else:
        exp_list = []

    with col_leg1:
        if exp_list:
            expiration = st.selectbox(
                "Leg 1 Expiration Date (current)", options=exp_list,
                index=0, key="pm_leg1_sel",
            )
        else:
            expiration = st.text_input(
                "Leg 1 Expiration Date (current)", placeholder="YYYY-MM-DD",
                key="pm_leg1_txt",
            ).strip()

    # When Leg 1 changes, auto-default Leg 2 = next weekly >= Leg 1 + 7d.
    # User overrides to Leg 2 stick until Leg 1 changes again.
    if expiration:
        prev_leg1 = st.session_state.get("pm_leg1_prev")
        if prev_leg1 != expiration and exp_list:
            auto_leg2 = get_next_weekly_expiration(expiration, exp_list)
            if auto_leg2 is not None:
                st.session_state["pm_leg2_sel"] = auto_leg2
            st.session_state["pm_leg1_prev"] = expiration

    with col_leg2:
        if exp_list:
            leg2_expiration = st.selectbox(
                "Leg 2 Expiration Date (new)", options=exp_list,
                key="pm_leg2_sel",
            )
        else:
            leg2_expiration = st.text_input(
                "Leg 2 Expiration Date (new)", placeholder="YYYY-MM-DD",
                key="pm_leg2_txt",
            ).strip()

    with col_fetch:
        st.write("")
        fetch_btn = st.button(
            "\U0001f504 Fetch Live Data",
            use_container_width=True, key="pm_fetch_btn",
            disabled=(dp is None),
        )

    # ── Inputs ─────────────────────────────────────────
    col_left, col_mid, col_right = st.columns(3)

    with col_left:
        strategy = st.radio(
            "Position Type", ["CC", "CSP"],
            horizontal=True, key="pm_strategy",
        )
        strike = st.number_input(
            "Strike Price ($)", min_value=0.0, value=370.0,
            step=1.0, key="pm_strike",
        )
        original_premium = st.number_input(
            "Original Premium ($)", min_value=0.0,
            value=4.50, step=0.10, key="pm_premium",
            help="Premium collected when this short was opened",
        )
        times_rolled = 0

    # ── Fetch Leg 1's live quote ───────────────────────
    # On the button, and whenever the contract on screen hasn't been fetched
    # yet — including the first render after saved values or a Ledger
    # pre-fill load. Otherwise its ask stays $0 and every roll looks like
    # closing the current leg is free.
    fetch_key = f"{ticker}_{expiration}_{strike}_{strategy}"
    if dp is not None and ticker and expiration and (
        fetch_btn or st.session_state.get("pm_fetched_key") != fetch_key
    ):
        with st.spinner(f"Fetching {ticker} {expiration} data…"):
            _do_fetch(dp, ticker, expiration, strike, strategy)
        st.session_state["pm_fetched_key"] = fetch_key

    # ── Live-data status ───────────────────────────────
    _has_auto = st.session_state.get("pm_fetch_ok", False)
    _fetch_err = st.session_state.get("pm_fetch_err", "")
    if _fetch_err:
        st.warning(
            f"Could not fetch live data: {_fetch_err}\n"
            "Please enter values manually.",
        )
    elif _has_auto:
        _ap = st.session_state.get("pm_auto_price", 0)
        _ab = st.session_state.get("pm_auto_bid", 0)
        _aa = st.session_state.get("pm_auto_ask", 0)
        _ai = st.session_state.get("pm_auto_iv", 0)
        st.caption(
            f"✅ Live data fetched — "
            f"Stock: ${_ap:.2f} · "
            f"Bid: ${_ab:.2f} · "
            f"Ask: ${_aa:.2f} · "
            f"IV: {_ai * 100:.1f}%"
        )

    with col_mid:
        current_bid = float(st.session_state.get("pm_auto_bid", 0.0))
        current_ask = st.number_input(
            "Buy to Close — Leg 1 ask ($)",
            min_value=0.0, step=0.01, key="pm_btc",
            help="What it costs to buy back the current short — its ask. "
                 "Filled from live data; override with your broker's quote.",
        )
        # A manual override has no matching bid: treat the ask as the mid.
        if current_bid <= 0 or current_bid > current_ask:
            current_bid = current_ask
        implied_vol = float(st.session_state.get("pm_auto_iv", 0.0))
        close_mode = "realistic"

        try:
            leg1_dte = days_to_expiration(expiration) if expiration else 0
        except Exception:
            leg1_dte = 0
        try:
            dte_remaining = (
                days_to_expiration(leg2_expiration) if leg2_expiration else 0
            )
        except Exception:
            dte_remaining = 0

        st.metric(
            "DTE Remaining",
            f"{dte_remaining} days",
            help="Calculated from the Leg 2 expiration (the new option).",
        )
        st.caption(f"Leg 1 (current): {leg1_dte} DTE")

    with col_right:
        current_price = st.number_input(
            "Current Stock Price ($)", min_value=0.0, step=0.01,
            value=st.session_state.get("pm_auto_price", 375.0),
            key="pm_price",
        )
        if strategy == "CSP":
            desired_buy = st.number_input(
                "Desired Buy Price ($)", min_value=0.0,
                value=365.0, step=1.0, key="pm_buy_price",
            )
            cost_basis = 0.0
            wants_assignment = st.checkbox(
                "I am willing to be assigned", value=True,
                key="pm_wants_assign",
                help="Check if you'd be happy owning shares at strike",
            )
            wants_to_keep = False
        else:
            desired_buy = 0.0
            cost_basis = st.number_input(
                "Cost Basis ($)", min_value=0.0, value=365.0,
                step=1.0, key="pm_cost_basis",
            )
            wants_assignment = True
            wants_to_keep = st.checkbox(
                "I want to keep my shares", value=False,
                key="pm_keep_shares",
            )

        earnings_iso = st.session_state.get("pm_earnings_date")
        earnings_dt = date.fromisoformat(earnings_iso) if earnings_iso else None
        if earnings_dt is not None:
            st.caption(f"📅 Next earnings: **{earnings_dt:%a %b %d, %Y}** (auto-detected)")
            manual_earnings = False
        else:
            manual_earnings = st.checkbox(
                "Earnings before Leg 2 expires",
                key="pm_earnings",
                help="No earnings date was found automatically. Tick this if "
                     "you know the company reports before the new expiration.",
            )

    # ── Evaluate ───────────────────────────────────────
    if st.button(
        "\U0001f9e0 Evaluate Position",
        type="primary", use_container_width=True, key="pm_eval",
    ) and _btc_known(current_ask):
        pos = OpenPosition(
            strategy=strategy,
            strike=strike,
            original_premium=original_premium,
            current_bid=current_bid,
            current_ask=current_ask,
            close_price_mode=close_mode,
            times_rolled=times_rolled,
            total_premium_collected=original_premium,
            desired_buy_price=desired_buy,
            cost_basis=cost_basis,
        )

        if dp is not None and ticker and leg2_expiration:
            with st.spinner(f"Fetching {leg2_expiration} roll candidates…"):
                roll_cands = _fetch_roll_candidates(
                    dp, ticker, pos, leg2_expiration,
                )
            indicators_obj, supports, resistances = _fetch_chart_context(
                dp, ticker, current_price,
            )
        else:
            roll_cands = []
            indicators_obj, supports, resistances = None, [], []

        if indicators_obj is not None:
            regime = classify_regime(indicators_obj, supports, resistances)
        else:
            regime = RegimeResult(
                primary=ChartRegime.NEUTRAL,
                secondary=None,
                description="Neutral (no chart data)",
                trade_bias="caution",
            )

        event_ctx = _build_event_context(
            ticker, expiration, leg2_expiration, current_price,
            implied_vol, earnings_dt, manual_earnings,
        )
        has_earnings = event_ctx is not None and (
            event_ctx.earnings_before_leg1 or event_ctx.earnings_before_leg2
        )

        # Expected move from Leg 2 IV * sqrt(Leg 2 DTE / 365) * spot
        if implied_vol > 0 and dte_remaining > 0 and current_price > 0:
            expected_move = (
                current_price * implied_vol * math.sqrt(dte_remaining / 365)
            )
        else:
            expected_move = 0.0

        decision = evaluate_position(
            pos=pos,
            current_price=current_price,
            regime=regime,
            support_levels=supports,
            resistance_levels=resistances,
            roll_candidates=roll_cands,
            next_cc_premium=0.0,
            has_earnings=has_earnings,
            implied_volatility=implied_vol,
            dte_remaining=dte_remaining,
            wants_assignment=wants_assignment,
            wants_to_keep_shares=wants_to_keep,
        )

        _render_decision(
            decision, pos, current_price, implied_vol, dte_remaining,
            dp=dp, ticker=ticker, wants_assignment=wants_assignment,
            indicators=indicators_obj,
            support_levels=supports,
            resistance_levels=resistances,
            expected_move=expected_move,
            has_earnings=has_earnings,
            event_ctx=event_ctx,
            expirations=exp_list,
        )

    _save_pm_prefs()


def _build_event_context(
    ticker: str,
    leg1_exp: str,
    leg2_exp: str,
    spot: float,
    leg1_iv: float,
    earnings_dt: date | None,
    manual_earnings: bool,
) -> EventContext | None:
    """Everything the roll scorer needs to price events around this roll."""
    try:
        leg1 = date.fromisoformat(leg1_exp)
        leg2 = date.fromisoformat(leg2_exp)
    except (TypeError, ValueError):
        return None
    rv = float(st.session_state.get(f"pm_rv_{ticker}_{date.today().isoformat()}", 0.0))
    return EventContext(
        today=date.today(),
        leg1_expiration=leg1,
        leg2_expiration=leg2,
        spot=spot,
        leg1_iv=leg1_iv,
        realized_vol=rv,
        earnings_date=earnings_dt,
        earnings_flagged=manual_earnings,
    )


def _render_event_panel(
    ctx: EventContext | None,
    candidates: list[RollCandidate],
    expirations: list[str],
    strategy: str,
) -> None:
    """Explain how earnings / an IV-implied event shape this roll."""
    if ctx is None:
        return
    # Measure the event on the candidate nearest the money.
    near = min(candidates, key=lambda c: abs(c.strike - ctx.spot), default=None)
    leg2_iv = near.iv if near is not None else 0.0
    base = ctx.base_iv
    move = (
        implied_event_move(ctx.spot, leg2_iv, base, near.dte)
        if near is not None else 0.0
    )
    move_txt = (
        f" The options price about **±${move:.2f} (±{move / ctx.spot:.1%})** "
        f"for it." if move > 0 and ctx.spot > 0 else ""
    )
    side = "below" if strategy == "CSP" else "above"
    ed = ctx.earnings_date

    if ctx.earnings_before_leg1 and ed is not None:
        st.warning(
            f"📅 **Earnings {ed:%a %b %d} lands before your current option "
            f"expires ({ctx.leg1_expiration:%b %d}).** Buying it back now pays "
            f"for the earnings premium in Leg 1 — if the strike isn't under "
            f"pressure, waiting until after the report usually makes the "
            f"buy-back cheaper (IV crush). Rolling now keeps you exposed to "
            f"the report on Leg 2 anyway.{move_txt}",
        )
    elif ctx.earnings_before_leg2:
        when = f"{ed:%a %b %d}" if ed is not None else "before Leg 2 expires"
        avoid = (
            last_expiration_before(expirations, ed, ctx.leg1_expiration)
            if ed is not None else None
        )
        avoid_txt = (
            f" To stay clear of the report, pick Leg 2 = **{avoid}**."
            if avoid else ""
        )
        st.warning(
            f"📅 **Earnings {when} falls inside the new expiration "
            f"({ctx.leg2_expiration:%b %d}).** Leg 2's credit includes "
            f"earnings premium, which pays for gap risk rather than decay.{move_txt} "
            f"Strikes inside that move are rejected; the rest are ranked on "
            f"their credit *without* the earnings premium and must sit "
            f"{side} the move.{avoid_txt}",
        )
    elif near is not None and ctx.term_spike(leg2_iv):
        st.info(
            f"⚡ **Leg 2 is priced at {leg2_iv:.0%} IV vs a {base:.0%} baseline** "
            f"— the market expects an event before {ctx.leg2_expiration:%b %d} "
            f"(earnings, product news, a Fed day…). It is treated like "
            f"earnings: strikes inside the event move are rejected and credits "
            f"are compared without the event premium.{move_txt}",
            icon="⚡",
        )


def _conf_color(level: str) -> str:
    return {
        "High": "#3fb950",
        "Medium": "#d29922",
        "Low": "#f85149",
    }.get(level, "#8b949e")


def _render_decision(
    decision,
    pos: OpenPosition,
    current_price: float,
    implied_vol: float,
    dte_remaining: int,
    dp=None,
    ticker: str = "",
    wants_assignment: bool = False,
    indicators: TechnicalIndicators | None = None,
    support_levels: list[SupportResistanceLevel] | None = None,
    resistance_levels: list[SupportResistanceLevel] | None = None,
    expected_move: float = 0.0,
    has_earnings: bool = False,
    event_ctx: EventContext | None = None,
    expirations: list[str] | None = None,
) -> None:
    _render_event_panel(
        event_ctx, decision.roll_candidates, expirations or [], pos.strategy,
    )
    picks = pick_top_rolls(
        decision.roll_candidates,
        pos,
        indicators=indicators,
        support_levels=support_levels or [],
        resistance_levels=resistance_levels or [],
        expected_move=expected_move,
        underlying_price=current_price,
        has_earnings=has_earnings,
        event_ctx=event_ctx,
    )
    best_roll = picks.balanced
    verdict = roll_vs_assign_verdict(pos, best_roll)
    confidence = verdict_confidence(pos, best_roll, verdict["recommendation"])

    # If we have a scored roll, replace the verdict's terse explanation
    # with the richer one that names the contributing factors.
    if best_roll is not None and verdict["recommendation"] == "ROLL":
        bal_score = picks.score_by_strike.get(best_roll.strike)
        if bal_score is not None:
            verdict["explanation"] = format_roll_recommendation(
                best_roll, pos, bal_score, bucket="balanced",
            )

    rec = verdict["recommendation"]
    if rec == "ROLL":
        color = "#58a6ff"
        icon = "\U0001f504"
    else:
        color = "#d29922"
        icon = "\U0001f4cc"

    st.markdown("---")
    st.markdown(
        (
            '<div style="'
            "background:#161b22;"
            "border:1px solid #30363d;"
            f"border-left:4px solid {color};"
            "border-radius:8px;"
            "padding:16px;"
            'margin:8px 0;">'
            f'<div style="font-size:20px;font-weight:700;color:{color};">'
            f"{icon} Recommendation: {rec}"
            "</div>"
            '<div style="font-size:14px;color:#e6edf3;margin-top:8px;">'
            f"{verdict['explanation'].replace('$', '&#36;')}"
            "</div>"
            '<div style="font-size:13px;color:#8b949e;margin-top:10px;">'
            f"Confidence: "
            f'<b style="color:{_conf_color(confidence)};">{confidence}</b>'
            "</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    if best_roll is None:
        st.markdown("**\U0001f4cc Assign**")
        st.metric(
            "Cost basis if assigned",
            f"${verdict['assignment_cost']:.2f}",
        )
        return

    _render_roll_math_card(pos, best_roll, verdict["assignment_cost"])
    _render_roll_alternatives(picks, pos)


def _render_roll_math_card(
    pos: OpenPosition,
    best_roll: RollCandidate,
    assignment_cost: float,
) -> None:
    btc = best_roll.buy_to_close
    sto = best_roll.sell_to_open
    net = sto - btc

    kind = "P" if pos.strategy == "CSP" else "C"
    st.markdown("**Roll Math**")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Buy to Close · Leg 1", f"${btc:.2f}")
        st.caption(f"${pos.strike:g}{kind} — current short, pay the ask")
    with m2:
        st.metric("Sell to Open · Leg 2", f"${sto:.2f}")
        st.caption(
            f"${best_roll.strike:g}{kind} exp {best_roll.expiration} — receive the bid"
        )
    with m3:
        if net >= 0:
            st.metric("Net Credit", f"+${net:.2f}")
        else:
            st.metric("Net Debit", f"-${abs(net):.2f}")
        st.caption(
            f"${sto:.2f} − ${btc:.2f} = {'+' if net >= 0 else '−'}${abs(net):.2f} "
            f"per share (${abs(net) * 100:,.0f} per contract)"
        )

    # Outcome line
    st.markdown("**If rolled and later assigned:**")
    if pos.strategy == "CSP":
        new_cost = new_effective_cost(pos, best_roll)
        st.metric(
            "New Effective Cost",
            f"${new_cost:.2f}",
            f"vs assign ${assignment_cost:.2f}",
            delta_color="inverse" if new_cost > assignment_cost else "normal",
        )
    else:
        profit = new_exit_profit(pos, best_roll)
        st.metric(
            "New Exit Profit",
            f"${profit:.2f}",
            f"strike ${best_roll.strike:.2f} vs cost basis ${pos.cost_basis:.2f}",
        )


def _render_roll_alternatives(picks: RollPicks, pos: OpenPosition) -> None:
    """Render Safe / Balanced / Premium cards in a 3-column row."""
    if not (picks.safe or picks.balanced or picks.premium):
        return

    st.markdown("---")
    st.markdown("### \U0001f3af Top 3 Roll Alternatives")
    if pos.strategy == "CSP":
        st.caption(
            "Safe: lower delta / farther OTM · "
            "Balanced: best overall · "
            "Premium: higher credit but still acceptable"
        )
    else:
        st.caption(
            "Safe: farther OTM / lower call-away risk · "
            "Balanced: best overall · "
            "Premium: higher credit but still acceptable"
        )

    cols = st.columns(3)
    bucket_specs = [
        ("safe", "\U0001f6e1️ Safe", picks.safe, "#3fb950"),
        ("balanced", "⚖️ Balanced", picks.balanced, "#58a6ff"),
        ("premium", "\U0001f4b0 Premium", picks.premium, "#f85149"),
    ]
    for col, (bucket, label, cand, color) in zip(cols, bucket_specs):
        with col:
            if cand is None:
                st.markdown(f"**{label}**")
                st.caption("No qualifying candidate.")
                continue

            score = picks.score_by_strike.get(cand.strike)
            score_val = score.score if score else 0.0
            credit_sign = "+" if cand.roll_credit >= 0 else "-"

            st.markdown(
                (
                    '<div style="'
                    "background:#161b22;"
                    "border:1px solid #30363d;"
                    f"border-left:4px solid {color};"
                    "border-radius:8px;"
                    'padding:12px 14px;margin-bottom:6px;">'
                    f'<div style="font-weight:700;color:{color};">'
                    f"{label}"
                    "</div>"
                    '<div style="font-size:13px;color:#8b949e;'
                    'margin-top:4px;">Score '
                    f"{score_val:.1f}/100"
                    "</div>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
            st.metric(
                "Strike / DTE",
                f"${cand.strike:.2f}",
                f"{cand.dte} DTE · |Δ| {cand.delta:.2f}",
            )
            st.metric(
                "Net " + ("Credit" if cand.roll_credit >= 0 else "Debit"),
                f"{credit_sign}${abs(cand.roll_credit):.2f}",
                f"STO ${cand.sell_to_open:.2f} · BTC ${cand.buy_to_close:.2f}",
            )
            ev = score.event if score else None
            if ev is not None and ev.spans_event:
                icon = "📅" if ev.kind == "earnings" else "⚡"
                st.caption(
                    f"{icon} Spans {'earnings' if ev.kind == 'earnings' else 'an event'}"
                    + (f" · outside ±${ev.event_move:.2f} move" if ev.event_move else "")
                    + (f" · ${ev.event_premium:.2f} event premium → "
                       f"{'+' if ev.adjusted_credit >= 0 else '−'}${abs(ev.adjusted_credit):.2f} "
                       f"without it" if ev.event_premium else "")
                )
            if pos.strategy == "CSP":
                st.metric(
                    "New Effective Cost",
                    f"${new_effective_cost(pos, cand):.2f}",
                )
            else:
                st.metric(
                    "New Exit Profit",
                    f"${new_exit_profit(pos, cand):.2f}",
                )
            if score and score.explanation:
                st.caption(score.explanation)
