"""
Streamlit UI for the Roll vs Assign position management decision engine.
"""

from __future__ import annotations

import streamlit as st

from data.common import days_to_expiration
from scoring.position_decision import (
    OpenPosition,
    RollCandidate,
    evaluate_position,
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


def _do_fetch(dp, ticker: str, expiration: str, strike: float, strategy: str) -> None:
    """Fetch live quote + option data and store in session state."""
    try:
        quote = dp.get_quote(ticker)
        st.session_state["pm_auto_price"] = quote.price

        if quote.earnings_date:
            try:
                from datetime import date
                ed = date.fromisoformat(quote.earnings_date)
                exp = date.fromisoformat(expiration)
                st.session_state["pm_auto_earnings"] = ed <= exp
            except Exception:
                st.session_state["pm_auto_earnings"] = False
        else:
            st.session_state["pm_auto_earnings"] = False

        calls, puts = dp.get_option_chain(ticker, expiration)
        chain = puts if strategy == "CSP" else calls

        contract = next(
            (c for c in chain if abs(c.strike - strike) < 0.01),
            None,
        )
        if contract:
            st.session_state["pm_auto_bid"] = contract.bid
            st.session_state["pm_auto_ask"] = contract.ask
            st.session_state["pm_auto_iv"] = contract.implied_volatility or 0.0
            st.session_state["pm_auto_delta"] = (
                abs(contract.delta) if contract.delta else 0.0
            )
            st.session_state["pm_auto_dte"] = days_to_expiration(expiration)
            st.session_state["pm_fetch_ok"] = True
            st.session_state["pm_fetch_err"] = ""
        else:
            st.session_state["pm_fetch_ok"] = False
            st.session_state["pm_fetch_err"] = (
                f"No {strategy} contract at strike ${strike:.2f} "
                f"for {expiration}. Enter values manually."
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
            "Original Premium Collected ($)", min_value=0.0,
            value=4.50, step=0.10, key="pm_premium",
        )
        total_premium = st.number_input(
            "Total Premium Collected ($)", min_value=0.0,
            value=4.50, step=0.10, key="pm_total_prem",
        )
        times_rolled = st.number_input(
            "Times Rolled", min_value=0, max_value=10,
            value=0, key="pm_rolled",
        )

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
        current_bid = st.number_input(
            "Current Option Bid ($)", min_value=0.0, step=0.01,
            value=st.session_state.get("pm_auto_bid", 2.00),
            key="pm_bid",
        )
        current_ask = st.number_input(
            "Current Option Ask ($)", min_value=0.0, step=0.01,
            value=st.session_state.get("pm_auto_ask", 2.20),
            key="pm_ask",
        )
        close_mode = st.radio(
            "Close Cost Estimate",
            ["realistic", "conservative", "optimistic"],
            index=0, horizontal=True, key="pm_close_mode",
            help=(
                "Realistic = mid price\n"
                "Conservative = ask price\n"
                "Optimistic = between bid and mid"
            ),
        )
        implied_vol = st.number_input(
            "Current IV (decimal, e.g. 0.45 = 45%)",
            min_value=0.0, max_value=5.0, step=0.01,
            value=st.session_state.get("pm_auto_iv", 0.0),
            key="pm_iv",
            help="Implied volatility of current option",
        )
        dte_remaining = st.number_input(
            "DTE Remaining", min_value=0, max_value=30,
            value=st.session_state.get("pm_auto_dte", 5),
            key="pm_dte",
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

    # ── Close cost estimates ───────────────────────────
    mid_val = (current_bid + current_ask) / 2
    opt_val = (current_bid + mid_val) / 2

    st.markdown("---")
    est1, est2, est3 = st.columns(3)
    est1.metric("Conservative Close", f"${current_ask:.2f}")
    est2.metric("Realistic Close", f"${mid_val:.2f}")
    est3.metric("Optimistic Close", f"${opt_val:.2f}")

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
            total_premium_collected=total_premium,
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

        decision = evaluate_position(
            pos=pos,
            current_price=current_price,
            regime=regime,
            support_levels=[],
            resistance_levels=[],
            roll_candidates=[],
            next_cc_premium=0.0,
            has_earnings=has_earnings,
            implied_volatility=implied_vol,
            dte_remaining=dte_remaining,
            wants_assignment=wants_assignment,
            wants_to_keep_shares=wants_to_keep,
        )

        _render_decision(decision, pos, current_price, implied_vol, dte_remaining)


def _render_decision(
    decision,
    pos: OpenPosition,
    current_price: float,
    implied_vol: float,
    dte_remaining: int,
) -> None:
    from scoring.position_decision import _calc_expected_move

    st.markdown("---")

    # ── Recommendation card ────────────────────────────
    rec_map = {
        "\U0001f4cc Accept Assignment": ("assign", "\U0001f4cc Accept Assignment", "#d29922"),
        "\U0001f4cc Let Shares Be Called Away": ("assign", "\U0001f4cc Let Shares Be Called Away", "#d29922"),
        "\U0001f504 Roll Position": ("roll", "\U0001f504 Roll Position", "#58a6ff"),
        "\U0001f4b0 Close for Profit": ("close", "\U0001f4b0 Close for Profit", "#3fb950"),
        "⏸️ Wait — Let Theta Work": ("wait", "⏸️ Wait — Let Theta Work", "#8b949e"),
    }
    _, label, color = rec_map.get(
        decision.recommendation,
        ("wait", decision.recommendation, "#8b949e"),
    )

    st.markdown(
        '<div style="'
        "background:#161b22;"
        "border:1px solid #30363d;"
        "border-left:4px solid {color};"
        "border-radius:8px;"
        "padding:16px;"
        'margin:8px 0;">'
        '<div style="font-size:20px;font-weight:700;color:{color};">'
        "{label}"
        "</div>"
        '<div style="font-size:13px;color:#8b949e;margin-top:4px;">'
        "Confidence: {confidence:.0f}/100"
        "</div>"
        '<div style="font-size:14px;color:#e6edf3;margin-top:8px;">'
        "{explanation}"
        "</div>"
        '<div style="margin-top:12px;padding-top:8px;border-top:1px solid #21262d;">'
        '<span style="font-size:13px;font-weight:700;color:#e6edf3;">'
        "Position Sizing: </span>"
        '<span style="font-size:13px;color:{color};">{pos_size}</span>'
        '<div style="font-size:11px;color:#8b949e;margin-top:2px;">'
        "{pos_reason}"
        "</div></div>"
        "</div>".format(
            color=color,
            label=label,
            confidence=decision.confidence,
            explanation=decision.explanation.replace("$", "&#36;"),
            pos_size=decision.position_size,
            pos_reason=decision.position_size_reason,
        ),
        unsafe_allow_html=True,
    )

    # ── Warnings ───────────────────────────────────────
    for w in decision.warnings:
        st.info(w)

    # ── Score comparison table ─────────────────────────
    st.markdown("**Score Comparison**")
    score_labels = {
        "assign": "\U0001f4cc Assign",
        "roll": "\U0001f504 Roll",
        "close": "\U0001f4b0 Close",
        "wait": "⏸️ Wait",
    }
    import pandas as pd
    score_rows = []
    best_key = max(decision.scores, key=lambda k: decision.scores[k])
    for key, val in decision.scores.items():
        score_rows.append({
            "Action": score_labels.get(key, key),
            "Score": round(val, 1),
            "Best": "★" if key == best_key else "",
        })
    score_df = pd.DataFrame(score_rows)
    st.dataframe(
        score_df, hide_index=True, use_container_width=True,
        column_config={
            "Score": st.column_config.ProgressColumn(
                min_value=0, max_value=100,
            ),
        },
    )

    # ── Expected move ──────────────────────────────────
    em = _calc_expected_move(current_price, implied_vol, dte_remaining)
    if em > 0:
        lower_em = current_price - em
        upper_em = current_price + em
        em_pct = em / current_price * 100
        st.markdown(
            '<div style="background:#161b22;border:1px solid #30363d;'
            'border-radius:6px;padding:10px 14px;margin:8px 0;'
            'font-size:13px;color:#8b949e;">'
            "Expected Move: &plusmn;&#36;{em} (&plusmn;{pct}%) &nbsp;&middot;&nbsp; "
            '<span style="color:#3fb950;">CSP safe: below &#36;{lo}</span>'
            " &nbsp;&middot;&nbsp; "
            '<span style="color:#58a6ff;">CC safe: above &#36;{hi}</span>'
            "</div>".format(
                em=f"{em:.2f}", pct=f"{em_pct:.1f}",
                lo=f"{lower_em:.2f}", hi=f"{upper_em:.2f}",
            ),
            unsafe_allow_html=True,
        )

    # ── Roll candidates ────────────────────────────────
    if decision.roll_candidates:
        st.markdown("**Roll Candidates**")
        roll_rows = []
        for c in decision.roll_candidates:
            roll_rows.append({
                "Strike": f"${c.strike:.2f}",
                "Expiration": c.expiration,
                "DTE": c.dte,
                "Credit": f"${c.roll_credit:.2f}",
                "Delta": f"{c.delta:.2f}",
                "Spread": f"{c.spread_pct * 100:.1f}%",
                "Type": c.roll_type,
                "OI": c.open_interest,
            })
        roll_df = pd.DataFrame(roll_rows)
        st.dataframe(roll_df, hide_index=True, use_container_width=True)

    # ── What happens next ──────────────────────────────
    st.markdown("---")
    st.markdown("**\U0001f4cb What Happens Next**")
    _safe = decision.next_wheel_step.replace("$", "&#36;")
    st.markdown(
        '<div style="background:#1e3a5f;border-left:4px solid #58a6ff;'
        'border-radius:4px;padding:10px 14px;margin:8px 0;'
        'font-size:14px;color:#e6edf3;">'
        "\U0001f4a1 " + _safe + "</div>",
        unsafe_allow_html=True,
    )
