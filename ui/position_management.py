"""
Streamlit UI for the Roll vs Assign position management decision engine.
"""

from __future__ import annotations

from datetime import date

import streamlit as st

from data.common import days_to_expiration, get_selected_contract
from scoring.position_decision import (
    OpenPosition,
    RollCandidate,
    evaluate_position,
    find_best_roll_for_premium,
    roll_vs_assign_verdict,
    verdict_confidence,
)
from strategies.models import (
    ChartRegime,
    RegimeResult,
    SupportResistanceLevel,
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


def _fetch_roll_candidates(
    dp,
    ticker: str,
    pos: OpenPosition,
) -> list[RollCandidate]:
    """Fetch option chains for 7-21 DTE expirations and build RollCandidate list."""
    candidates: list[RollCandidate] = []
    try:
        expirations = dp.get_expirations(ticker)
    except Exception:
        return candidates

    today = date.today()
    strike_lo = pos.strike * 0.90
    strike_hi = pos.strike * 1.10
    chain_type = "puts" if pos.strategy == "CSP" else "calls"

    for exp_str in expirations:
        try:
            dte = (date.fromisoformat(exp_str) - today).days
        except Exception:
            continue
        if not (7 <= dte <= 21):
            continue
        try:
            calls, puts = dp.get_option_chain(ticker, exp_str)
        except Exception:
            continue

        chain = puts if chain_type == "puts" else calls
        for opt in chain:
            if not (strike_lo <= opt.strike <= strike_hi):
                continue
            if opt.mid <= 0:
                continue
            mid = opt.mid
            bid = opt.bid
            ask = opt.ask
            spread_pct = (ask - bid) / max(mid, 0.01)
            roll_credit = mid - pos.close_cost

            if opt.strike < pos.strike:
                roll_type = "down" if pos.strategy == "CSP" else "down"
            elif opt.strike > pos.strike:
                roll_type = "up"
            else:
                roll_type = "out"

            candidates.append(RollCandidate(
                strike=opt.strike,
                expiration=exp_str,
                dte=dte,
                bid=bid,
                ask=ask,
                mid=mid,
                delta=abs(opt.delta) if opt.delta is not None else 0.0,
                open_interest=opt.open_interest or 0,
                roll_credit=roll_credit,
                roll_type=roll_type,
                spread_pct=spread_pct,
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
    col_ticker, col_exp, col_fetch = st.columns([2, 2, 1])
    with col_ticker:
        ticker = st.text_input(
            "Ticker", value="TSLA", key="pm_ticker",
        ).upper().strip()
    with col_exp:
        if dp is not None and ticker:
            try:
                exps = dp.get_expirations(ticker)
                exp_list = list(exps)
            except Exception:
                exp_list = []
        else:
            exp_list = []

        if exp_list:
            expiration = st.selectbox(
                "Expiration Date", options=exp_list,
                index=0, key="pm_expiration_sel",
            )
        else:
            expiration = st.text_input(
                "Expiration Date", placeholder="YYYY-MM-DD",
                key="pm_expiration_txt",
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

        if expiration:
            try:
                dte_remaining = days_to_expiration(expiration)
            except Exception:
                dte_remaining = 0
        else:
            dte_remaining = 0

        st.metric(
            "DTE Remaining",
            f"{dte_remaining} days",
            help="Auto-calculated from the selected expiration date.",
        )

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

        if dp is not None and ticker:
            with st.spinner("Fetching roll candidates…"):
                roll_cands = _fetch_roll_candidates(dp, ticker, pos)
        else:
            roll_cands = []

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
) -> None:
    best_roll = find_best_roll_for_premium(decision.roll_candidates, pos)
    verdict = roll_vs_assign_verdict(pos, best_roll)
    confidence = verdict_confidence(pos, best_roll, verdict["recommendation"])

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

    st.markdown("**Comparison**")
    assign_col, roll_col = st.columns(2)

    with assign_col:
        st.markdown("**\U0001f4cc Assign**")
        st.metric(
            "Cost basis if assigned",
            f"${verdict['assignment_cost']:.2f}",
        )

    with roll_col:
        st.markdown("**\U0001f504 Roll**")
        if best_roll is not None:
            direction = "Down" if best_roll.strike < pos.strike else (
                "Up" if best_roll.strike > pos.strike else "Same"
            )
            st.metric(
                "New strike",
                f"${best_roll.strike:.2f}",
                f"{direction} · {best_roll.dte} DTE",
            )
            st.metric(
                "Net roll credit",
                f"${best_roll.roll_credit:.2f}",
            )
            new_cost = verdict["new_effective_cost"]
            st.metric(
                "New effective cost",
                f"${new_cost:.2f}" if new_cost is not None else "—",
            )
        else:
            st.caption("No profitable roll available.")
