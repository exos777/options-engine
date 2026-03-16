"""
Streamlit rendering helpers for the recommendations and market overview sections.

All functions accept Streamlit-compatible data and call st.* directly.
They are separated from app/main.py to keep the main file focused on layout.
"""

from __future__ import annotations

import streamlit as st

from strategies.models import (
    ChartRegime,
    Recommendation,
    RecommendationLabel,
    ScreenerResult,
    Strategy,
)


# ---------------------------------------------------------------------------
# Colour / badge helpers
# ---------------------------------------------------------------------------

_LABEL_EMOJI = {
    RecommendationLabel.SAFEST_INCOME: "🛡️",
    RecommendationLabel.BEST_BALANCE:  "⚖️",
    RecommendationLabel.MAX_PREMIUM:   "💰",
}

_REGIME_COLOR = {
    ChartRegime.BULLISH:          "normal",
    ChartRegime.NEUTRAL:          "off",
    ChartRegime.BEARISH:          "inverse",
    ChartRegime.OVEREXTENDED:     "off",
    ChartRegime.NEAR_SUPPORT:     "normal",
    ChartRegime.NEAR_RESISTANCE:  "off",
}


# ---------------------------------------------------------------------------
# Market overview
# ---------------------------------------------------------------------------

def render_market_overview(result: ScreenerResult, strategy: Strategy) -> None:
    """Render the market overview metric strip and warnings."""
    q = result.quote
    ind = result.indicators

    # Price metrics row — exactly 5 columns, no clutter
    col1, col2, col3, col4, col5 = st.columns(5)

    pct = q.change_pct * 100
    pct_label = f"{pct:+.2f}%"

    col1.metric("Price", f"${q.price:.2f}", pct_label)
    col2.metric("SMA 20", f"${ind.sma_20:.2f}")
    col3.metric("SMA 50", f"${ind.sma_50:.2f}")
    col4.metric("RSI (14)", f"{ind.rsi_14:.1f}")
    col5.metric("ATR (14)", f"${ind.atr_14:.2f}")

    st.caption(
        f"**Expiration:** {result.expiration}  ·  "
        f"**DTE:** {result.dte}  ·  "
        f"**Weekly ATR est.:** ${ind.weekly_atr_est:.2f}"
        + (
            f"  ·  **Expected move ±:** ${result.expected_move:.2f}"
            if result.expected_move > 0
            else ""
        )
    )

    # Regime badge
    primary = result.regime.primary
    regime_delta_color = _REGIME_COLOR.get(primary, "off")
    badge_col, _, warn_col = st.columns([2, 1, 5])
    badge_col.metric(
        "Chart Regime",
        primary.value,
        result.regime.secondary.value if result.regime.secondary else "",
        delta_color=regime_delta_color,
    )

    # Warnings
    for w in result.warnings:
        st.warning(w, icon="⚠️")

    # Earnings warning (shown as error for high visibility)
    if q.earnings_date:
        from data.provider import earnings_warning
        ew = earnings_warning(q, result.expiration)
        if ew:
            st.error(ew, icon="🗓️")


# ---------------------------------------------------------------------------
# Signal Dashboard (collapsed expander)
# ---------------------------------------------------------------------------

def render_signal_dashboard(result: ScreenerResult) -> None:
    """Render collapsed Signal Dashboard expander with ADX, VWAP, TTM Squeeze."""
    ind = result.indicators
    q = result.quote

    with st.expander("📊 Signal Dashboard", expanded=False):
        c1, c2, c3, c4 = st.columns(4)

        # RSI
        rsi_v = ind.rsi_14
        if rsi_v >= 70:
            rsi_signal, rsi_color = "Overbought ⚠️", "inverse"
        elif rsi_v <= 30:
            rsi_signal, rsi_color = "Oversold 🔵", "normal"
        else:
            rsi_signal, rsi_color = "Neutral", "off"
        c1.metric("RSI (14)", f"{rsi_v:.1f}", rsi_signal, delta_color=rsi_color)

        # ADX
        adx_v = ind.adx_14
        if adx_v > 25:
            adx_signal, adx_color = "Trending 📈", "off"
        elif adx_v < 20:
            adx_signal, adx_color = "Ranging ✅", "normal"
        else:
            adx_signal, adx_color = "Transitioning", "off"
        c2.metric("ADX (14)", f"{adx_v:.1f}", adx_signal, delta_color=adx_color)

        # VWAP
        vwap_v = ind.vwap
        if vwap_v > 0:
            diff_pct = (q.price - vwap_v) / vwap_v * 100
            vwap_signal = f"{diff_pct:+.1f}% {'above' if diff_pct >= 0 else 'below'}"
            vwap_color = "normal" if diff_pct >= 0 else "inverse"
            c3.metric("VWAP", f"${vwap_v:.2f}", vwap_signal, delta_color=vwap_color)
        else:
            c3.metric("VWAP", "N/A")

        # TTM Squeeze
        sq_label = "ON 🔴 — compression" if ind.squeeze_on else "OFF 🟢 — expansion"
        sq_color = "inverse" if ind.squeeze_on else "normal"
        c4.metric("TTM Squeeze", sq_label, delta_color=sq_color)

        # SMA context row
        st.caption(
            f"SMA20 ${ind.sma_20:.2f} {'✅ price above' if q.price > ind.sma_20 else '⚠️ price below'}  ·  "
            f"SMA50 ${ind.sma_50:.2f} {'✅ price above' if q.price > ind.sma_50 else '⚠️ price below'}"
        )

        # Contextual warnings
        if ind.squeeze_on:
            st.warning(
                "TTM Squeeze ON — Bollinger Bands are inside Keltner Channels. "
                "Volatility compression in progress; premium may expand after breakout. "
                "Consider waiting for squeeze release before selling.",
                icon="🔴",
            )
        if adx_v < 20:
            st.info(
                f"ADX {adx_v:.1f} indicates a ranging/low-trend market — "
                "ideal conditions for premium selling strategies.",
                icon="✅",
            )
        elif adx_v > 40:
            st.warning(
                f"ADX {adx_v:.1f} — strong trend in progress. "
                "Be cautious selling against the trend.",
                icon="⚠️",
            )


# ---------------------------------------------------------------------------
# Single recommendation card
# ---------------------------------------------------------------------------

def render_recommendation_card(
    rec: Recommendation,
    strategy: Strategy,
    current_price: float = 0.0,
    expected_move: float = 0.0,
) -> None:
    """Render one recommendation inside a styled expander card."""
    c = rec.option.contract
    opt = rec.option
    emoji = _LABEL_EMOJI.get(rec.label, "📋")

    with st.expander(
        f"{emoji} **{rec.label.value}** — ${c.strike:.2f} "
        f"| Premium ${opt.premium:.2f} "
        f"| Score {opt.score:.0f}/100",
        expanded=True,
    ):
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Strike", f"${c.strike:.2f}")
        m1.metric("Premium", f"${opt.premium:.2f}")

        m2.metric(
            "Delta",
            f"{abs(c.delta):.3f}" if c.delta is not None else "—",
        )
        m2.metric(
            "IV",
            f"{c.implied_volatility * 100:.1f}%" if c.implied_volatility else "—",
        )

        m3.metric("Open Interest", f"{c.open_interest:,}")
        m3.metric("Bid-Ask Spread", f"{c.bid_ask_spread_pct * 100:.1f}%")

        m4.metric(
            "Ann. Return",
            f"{opt.annualized_return * 100:.1f}%",
        )
        m4.metric("Break-even", f"${opt.break_even:.2f}")

        m5.metric(
            "Theta / day",
            f"${c.theta:.4f}" if c.theta is not None else "—",
        )
        m5.metric(
            "Vega / 1% IV",
            f"${c.vega:.4f}" if c.vega is not None else "—",
        )

        st.caption(
            f"Distance from price: **{opt.distance_pct * 100:.1f}%**  ·  "
            f"Score breakdown — Premium: {opt.premium_score:.0f}  "
            f"Theta: {opt.theta_score:.0f}  "
            f"Delta: {opt.delta_score:.0f}  "
            f"Liquidity: {opt.liquidity_score:.0f}  "
            f"Chart: {opt.chart_score:.0f}  "
            f"Basis: {opt.basis_score:.0f}"
        )

        # Plain English explanation
        st.info(rec.explanation, icon="💡")

        # Seller's summary
        ss1, ss2, ss3, ss4 = st.columns(4)

        # Theta income
        ss1.metric(
            "Theta / day",
            f"${abs(c.theta):.3f}" if c.theta else "—",
            help="Daily premium decay earned (per contract = ×100)",
        )
        ss1.metric("Ann. Yield", f"{opt.annualized_return * 100:.1f}%")

        # Expected move
        em_status = "—"
        em_boundary = "—"
        em_buffer = "—"
        if expected_move > 0 and current_price > 0:
            em_pct = expected_move / current_price * 100
            if strategy == Strategy.COVERED_CALL:
                boundary = current_price + expected_move
                outside = c.strike >= boundary
                buffer = c.strike - boundary
            else:
                boundary = current_price - expected_move
                outside = c.strike <= boundary
                buffer = boundary - c.strike
            em_status = "✅ Outside EM" if outside else "⚠️ Inside EM"
            em_boundary = f"${boundary:.2f} (±{em_pct:.1f}%)"
            em_buffer = f"${abs(buffer):.2f} {'buffer' if outside else 'exposure'}"

        ss2.metric("EM Boundary", em_boundary)
        ss2.metric("Strike vs EM", em_status)

        # Vega risk
        vega_risk = "—"
        if c.vega:
            av = abs(c.vega)
            vega_risk = "Low" if av <= 0.15 else ("Medium" if av <= 0.30 else "High ⚠️")
        ss3.metric("Vega Risk", vega_risk, help="IV expansion risk for short positions")
        ss3.metric("EM Buffer", em_buffer)

        # Earnings + basis
        earnings_status = "🚨 In Window" if opt.earnings_in_window else "✅ Clear"
        ss4.metric("Earnings", earnings_status)
        if opt.above_cost_basis is True:
            ss4.metric("vs Cost Basis", "✅ Above")
        elif opt.above_cost_basis is False:
            ss4.metric("vs Cost Basis", "⚠️ Below")

        # S/R context badges
        tags = []
        if opt.near_support:
            tags.append("🟢 Near Support")
        if opt.near_resistance:
            tags.append("🔴 Near Resistance")
        if tags:
            st.caption("  ".join(tags))


# ---------------------------------------------------------------------------
# Full recommendations section
# ---------------------------------------------------------------------------

def render_recommendations(
    result: ScreenerResult,
    strategy: Strategy,
    expected_move: float = 0.0,
) -> None:
    """Render all recommendations, or a no-trade message if list is empty."""
    st.subheader("Recommendations")

    if not result.recommendations:
        if result.regime.trade_bias == "skip":
            st.error(
                result.regime.skip_reason
                or "No trade recommended based on current chart conditions.",
                icon="🚫",
            )
        else:
            st.warning(
                "No qualifying strikes found. Try relaxing your filters "
                "(min OI, max spread, min premium, max delta).",
                icon="🔍",
            )
        return

    for rec in result.recommendations:
        render_recommendation_card(
            rec, strategy,
            current_price=result.quote.price,
            expected_move=expected_move,
        )


# ---------------------------------------------------------------------------
# Ranked option table
# ---------------------------------------------------------------------------

def render_option_table(result: ScreenerResult) -> None:
    """Render the full sortable scored option table."""
    st.subheader("All Qualifying Strikes")

    df = result.all_options
    if df.empty:
        st.info("No qualifying strikes to display.")
        return

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Strike":     st.column_config.NumberColumn(format="$%.2f"),
            "Premium":    st.column_config.NumberColumn(format="$%.2f"),
            "Break-even": st.column_config.NumberColumn(format="$%.2f"),
            "Theta":      st.column_config.NumberColumn(format="$%.4f"),
            "Vega":       st.column_config.NumberColumn(format="$%.4f"),
            "Score":      st.column_config.ProgressColumn(min_value=0, max_value=100),
            "Earnings":   st.column_config.TextColumn(help="⚠️ Earnings event before expiration"),
        },
    )
