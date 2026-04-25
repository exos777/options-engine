"""
Streamlit UI for the Roll vs Assign position management decision engine.
"""

from __future__ import annotations

from datetime import date

import streamlit as st

import math

from data.common import (
    days_to_expiration,
    get_next_weekly_expiration,
    get_selected_contract,
)
from indicators.support_resistance import find_support_resistance
from indicators.technical import calculate_indicators
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
        ))
    return candidates


def _do_fetch(dp, ticker: str, expiration: str, strike: float, strategy: str) -> None:
    """Fetch live quote + option data for the selected expiration/strike."""
    try:
        quote = dp.get_quote(ticker)
        st.session_state["pm_auto_price"] = quote.price

        if quote.earnings_date:
            try:
                ed = date.fromisoformat(quote.earnings_date)
                exp = date.fromisoformat(expiration)
                st.session_state["pm_auto_earnings"] = ed <= exp
            except Exception:
                st.session_state["pm_auto_earnings"] = False
        else:
            st.session_state["pm_auto_earnings"] = False

        selected = get_selected_contract(dp, ticker, expiration, strike, strategy)
        st.session_state["pm_auto_dte"] = selected.dte

        if selected.found:
            contract = selected.contract
            st.session_state["pm_auto_bid"] = contract.bid
            st.session_state["pm_auto_ask"] = contract.ask
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


def render_position_manager(dp=None) -> None:
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
            "Position Type", ["CSP", "CC"],
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

    # ── Auto-fetch on button press ─────────────────────
    if fetch_btn and dp is not None and ticker and expiration:
        with st.spinner(f"Fetching {ticker} {expiration} data…"):
            _do_fetch(dp, ticker, expiration, strike, strategy)

    # ── Auto-fetch when key inputs change ──────────────
    fetch_key = f"{ticker}_{expiration}_{strike}_{strategy}"
    prev_key = st.session_state.get("pm_last_fetch_key", "")
    if (
        dp is not None
        and ticker
        and expiration
        and fetch_key != prev_key
        and prev_key != ""
    ):
        _do_fetch(dp, ticker, expiration, strike, strategy)
    if ticker and expiration:
        st.session_state["pm_last_fetch_key"] = fetch_key

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
        current_ask = float(st.session_state.get("pm_auto_ask", 0.0))
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

        has_earnings_default = st.session_state.get("pm_auto_earnings", False)
        has_earnings = st.checkbox(
            "Earnings within DTE window",
            value=has_earnings_default,
            key="pm_earnings",
        )

        regime_choice = st.selectbox(
            "Chart Regime",
            [r.value for r in ChartRegime],
            index=0, key="pm_regime",
        )

    # ── Evaluate ───────────────────────────────────────
    if st.button(
        "\U0001f9e0 Evaluate Position",
        type="primary", use_container_width=True, key="pm_eval",
    ):
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

        regime = RegimeResult(
            primary=ChartRegime(regime_choice),
            secondary=None,
            description=regime_choice,
            trade_bias=(
                "go" if regime_choice in ("Bullish", "Near Support")
                else "skip" if regime_choice == "Bearish"
                else "caution"
            ),
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
            support_levels=[],
            resistance_levels=[],
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
) -> None:
    picks = pick_top_rolls(
        decision.roll_candidates,
        pos,
        indicators=indicators,
        support_levels=support_levels or [],
        resistance_levels=resistance_levels or [],
        expected_move=expected_move,
        underlying_price=current_price,
        has_earnings=has_earnings,
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

    st.markdown("**Roll Math**")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Sell to Open", f"${sto:.2f}")
    with m2:
        st.metric("Buy to Close", f"${btc:.2f}")
    with m3:
        if net >= 0:
            st.metric("Net Credit", f"+${net:.2f}")
        else:
            st.metric("Net Debit", f"-${abs(net):.2f}")

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
    """Render Safe / Balanced / Aggressive cards in a 3-column row."""
    if not (picks.safe or picks.balanced or picks.aggressive):
        return

    st.markdown("---")
    st.markdown("### \U0001f3af Top 3 Roll Alternatives")
    if pos.strategy == "CSP":
        st.caption(
            "Safe: lower delta / farther OTM · "
            "Balanced: best risk/reward · "
            "Aggressive: higher credit, higher assignment risk"
        )
    else:
        st.caption(
            "Safe: farther OTM / lower call-away risk · "
            "Balanced: best risk/reward · "
            "Aggressive: higher premium, higher call-away probability"
        )

    cols = st.columns(3)
    bucket_specs = [
        ("safe", "\U0001f6e1️ Safe", picks.safe, "#3fb950"),
        ("balanced", "⚖️ Balanced", picks.balanced, "#58a6ff"),
        ("aggressive", "\U0001f525 Aggressive", picks.aggressive, "#f85149"),
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
