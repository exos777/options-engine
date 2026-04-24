"""
Streamlit UI for the Roll vs Assign position management decision engine.
"""

from __future__ import annotations

from datetime import date

import streamlit as st

from data.common import days_to_expiration
from scoring.position_decision import (
    OpenPosition,
    RollCandidate,
    evaluate_position,
    find_best_roll_for_premium,
    new_effective_cost,
    roll_vs_assign_verdict,
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

    # ── Best Roll for Maximum Premium ──────────────────
    best_roll = find_best_roll_for_premium(decision.roll_candidates, pos)
    new_net_roll = 0.0
    _render_best_roll_card(best_roll, pos)
    if best_roll is not None:
        new_net_roll = (
            best_roll.strike
            - pos.total_premium_collected
            - best_roll.roll_credit
        )

    # ── Assignment Analysis ────────────────────────────
    best_cc = None
    best_cc_dte = 0
    if wants_assignment and pos.strategy == "CSP":
        best_cc, best_cc_dte = _find_best_immediate_cc(dp, ticker, pos)
        _render_assignment_card(
            pos, current_price, best_cc, best_cc_dte,
        )

    # ── Side-by-Side Comparison ────────────────────────
    if wants_assignment and pos.strategy == "CSP":
        _render_roll_vs_assign_comparison(
            pos, best_roll, new_net_roll, best_cc,
        )

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


def _render_best_roll_card(
    best_roll: RollCandidate | None,
    pos: OpenPosition,
) -> None:
    st.markdown("---")

    verdict = roll_vs_assign_verdict(pos, best_roll)

    if best_roll is None:
        st.warning(
            "\U0001f504 No profitable roll found. "
            "No candidates generate meaningful credit."
        )
        _render_simple_verdict(verdict)
        return

    st.markdown("### \U0001f504 Best Roll — Roll Math")

    new_cost = new_effective_cost(pos, best_roll)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Original Premium", f"${pos.original_premium:.2f}")
    with c2:
        st.metric("Cost to Close", f"${pos.close_cost:.2f}")
    with c3:
        st.metric("Net Roll Credit", f"${best_roll.roll_credit:.2f}")
    with c4:
        direction = "Down" if best_roll.strike < pos.strike else (
            "Up" if best_roll.strike > pos.strike else "Same"
        )
        st.metric(
            "New Strike",
            f"${best_roll.strike:.2f}",
            f"{direction} · {best_roll.dte} DTE",
        )
    with c5:
        st.metric(
            "New Effective Cost",
            f"${new_cost:.2f}",
            f"vs assign ${verdict['assignment_cost']:.2f}",
        )

    st.caption(
        f"New premium: ${best_roll.mid:.2f} · "
        f"Delta: {best_roll.delta:.2f} · "
        f"OI: {best_roll.open_interest} · "
        f"Spread: {best_roll.spread_pct * 100:.1f}% · "
        f"Exp: {best_roll.expiration}"
    )

    _render_simple_verdict(verdict)


def _render_simple_verdict(verdict: dict) -> None:
    rec = verdict["recommendation"]
    if rec == "ROLL":
        color = "#58a6ff"
        icon = "\U0001f504"
    else:
        color = "#d29922"
        icon = "\U0001f4cc"

    st.markdown(
        (
            '<div style="'
            "background:#161b22;"
            "border:1px solid #30363d;"
            f"border-left:4px solid {color};"
            "border-radius:8px;"
            "padding:14px 16px;"
            'margin:8px 0;">'
            f'<div style="font-size:18px;font-weight:700;color:{color};">'
            f"{icon} Recommendation: {rec}"
            "</div>"
            '<div style="font-size:14px;color:#e6edf3;margin-top:6px;">'
            f"{verdict['explanation'].replace('$', '&#36;')}"
            "</div>"
            '<div style="font-size:13px;color:#8b949e;margin-top:6px;">'
            f"<b>Next action:</b> "
            f"{verdict['next_action'].replace('$', '&#36;')}"
            "</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _find_best_immediate_cc(
    dp,
    ticker: str,
    pos: OpenPosition,
):
    """Find the highest-premium CC 2-8% above strike within 7-14 DTE."""
    if dp is None or not ticker:
        return None, 0

    best_cc = None
    best_cc_dte = 0
    best_premium = 0.0
    try:
        expirations = dp.get_expirations(ticker)
    except Exception:
        return None, 0

    today = date.today()
    lo_strike = pos.strike * 1.02
    hi_strike = pos.strike * 1.08

    for exp_str in expirations:
        try:
            dte = (date.fromisoformat(exp_str) - today).days
        except Exception:
            continue
        if not (7 <= dte <= 14):
            continue
        try:
            calls, _puts = dp.get_option_chain(ticker, exp_str)
        except Exception:
            continue

        cc_candidates = [
            c for c in calls
            if lo_strike <= c.strike <= hi_strike
            and (c.open_interest or 0) >= 50
            and c.mid >= 0.50
        ]
        if not cc_candidates:
            continue

        cc = max(cc_candidates, key=lambda c: c.mid)
        if cc.mid > best_premium:
            best_cc = cc
            best_cc_dte = dte
            best_premium = cc.mid

    return best_cc, best_cc_dte


def _render_assignment_card(
    pos: OpenPosition,
    current_price: float,
    best_cc,
    best_cc_dte: int,
) -> None:
    net_cost = pos.net_assigned_cost

    st.markdown("---")
    st.markdown("### ✅ Assignment Analysis")

    assign_col1, assign_col2, assign_col3 = st.columns(3)

    with assign_col1:
        st.metric(
            "You Get Assigned At",
            f"${pos.strike:.2f}",
            "per share",
        )
        st.metric(
            "Net Cost Basis",
            f"${net_cost:.2f}",
            f"after ${pos.total_premium_collected:.2f} premium",
        )

    with assign_col2:
        discount = current_price - net_cost
        disc_label = "Buying at discount" if discount > 0 else "Paying premium"
        st.metric(
            "vs Current Price",
            f"${current_price:.2f}",
            f"{disc_label}: ${abs(discount):.2f}",
        )

        if pos.desired_buy_price > 0:
            vs_desired = net_cost - pos.desired_buy_price
            above_below = "above" if vs_desired > 0 else "below"
            st.metric(
                "vs Desired Buy Price",
                f"${pos.desired_buy_price:.2f}",
                f"${abs(vs_desired):.2f} {above_below} target",
                delta_color="inverse" if vs_desired > 0 else "normal",
            )

    with assign_col3:
        if best_cc is not None:
            st.metric(
                "Best Immediate CC",
                f"${best_cc.strike:.2f} strike",
                f"${best_cc.mid:.2f} premium · {best_cc_dte} DTE",
            )
            new_basis = net_cost - best_cc.mid
            st.metric(
                "Basis After CC",
                f"${new_basis:.2f}",
                f"-${best_cc.mid:.2f} from premium",
            )
        else:
            st.metric(
                "Immediate CC",
                "No data",
                "Enter IV to calculate",
            )

    if best_cc is not None:
        profit_if_called = best_cc.strike - net_cost + best_cc.mid
        st.success(
            f"✅ If you accept assignment at ${pos.strike:.2f}:\n\n"
            f"Net cost basis: ${net_cost:.2f}\n\n"
            f"Immediately sell ${best_cc.strike:.2f} CC "
            f"for ${best_cc.mid:.2f}\n\n"
            f"New basis: ${net_cost - best_cc.mid:.2f}\n\n"
            f"If called away at ${best_cc.strike:.2f}: "
            f"Profit = ${profit_if_called:.2f}/share ✅\n\n"
            f"If CC expires: Sell another CC, "
            f"keep reducing basis"
        )
    else:
        st.success(
            f"✅ If assigned at ${pos.strike:.2f}:\n\n"
            f"Net cost basis: ${net_cost:.2f}\n\n"
            f"Immediately sell CC above ${pos.strike:.2f} for 7-14 DTE\n\n"
            f"Target strike: "
            f"${pos.strike * 1.03:.2f}–${pos.strike * 1.05:.2f}"
        )

    st.markdown("**Break-even Analysis:**")
    be_col1, be_col2, be_col3 = st.columns(3)
    with be_col1:
        st.metric(
            "Break-even (no CC)",
            f"${net_cost:.2f}",
            "stock must reach this",
        )
    with be_col2:
        if best_cc is not None:
            be_with_cc = net_cost - best_cc.mid
            st.metric(
                "Break-even (with 1 CC)",
                f"${be_with_cc:.2f}",
                f"-${best_cc.mid:.2f} from CC",
            )
        else:
            st.metric("Break-even (with 1 CC)", "—", "no CC data")
    with be_col3:
        est_monthly_cc = (best_cc.mid if best_cc is not None else 2.0) * 4
        be_monthly = net_cost - est_monthly_cc
        st.metric(
            "Est. Break-even (4 CCs)",
            f"${be_monthly:.2f}",
            f"-${est_monthly_cc:.2f} est.",
        )


def _render_roll_vs_assign_comparison(
    pos: OpenPosition,
    best_roll: RollCandidate | None,
    new_net_roll: float,
    best_cc,
) -> None:
    net_cost = pos.net_assigned_cost

    st.markdown("---")
    st.markdown("### ⚖️ Roll vs Assign — Side by Side")

    comp_col1, comp_col2 = st.columns(2)

    with comp_col1:
        st.markdown("**\U0001f504 If You Roll**")
        if best_roll is not None:
            st.markdown(
                f"- Roll to **${best_roll.strike:.2f}** "
                f"exp {best_roll.expiration}\n"
                f"- Collect **${best_roll.roll_credit:.2f}** credit\n"
                f"- New net cost if assigned: "
                f"**${new_net_roll:.2f}**\n"
                f"- Assignment delayed {best_roll.dte} days\n"
                f"- Capital still at risk: "
                f"**${best_roll.strike * 100:.0f}**"
            )
        else:
            st.markdown("No profitable roll available")

    with comp_col2:
        st.markdown("**✅ If You Accept Assignment**")
        cc_prem_str = f"${best_cc.mid:.2f}" if best_cc is not None else "N/A"
        cc_prem_val = best_cc.mid if best_cc is not None else 0.0
        st.markdown(
            f"- Own 100 shares at **${net_cost:.2f}**\n"
            f"- Sell CC immediately for **{cc_prem_str}**\n"
            f"- New basis: "
            f"**${net_cost - cc_prem_val:.2f}**\n"
            f"- Start earning CC income weekly\n"
            f"- Capital deployed: "
            f"**${net_cost * 100:.0f}**"
        )
