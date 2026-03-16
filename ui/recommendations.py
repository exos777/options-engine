"""
Streamlit rendering helpers for the recommendations and market overview sections.

All functions accept Streamlit-compatible data and call st.* directly.
They are separated from app/main.py to keep the main file focused on layout.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from indicators.technical import FullIndicators
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

    # Earnings warning (shown as error for high visibility)
    if q.earnings_date:
        from data.provider import earnings_warning
        ew = earnings_warning(q, result.expiration)
        if ew:
            st.error(ew, icon="🗓️")


# ---------------------------------------------------------------------------
# Price Forecast (probability-based, replaces old signal dashboard)
# ---------------------------------------------------------------------------

def render_price_forecast(
    result: ScreenerResult,
    strategy: Strategy,
    full_ind: FullIndicators | None,
) -> None:
    """
    Render a comprehensive probability-based price forecast using all 9
    indicator signals.  Replaces the old warning banner with an actionable
    bullish/bearish probability, confidence level, and indicator breakdown.
    """
    price = result.quote.price
    dte = result.dte
    expected_move = result.expected_move
    upper = price + expected_move
    lower = price - expected_move
    indicators = result.indicators

    # ── Gather all indicator values ──────────────────
    rsi = indicators.rsi_14
    sma20 = indicators.sma_20
    sma50 = indicators.sma_50
    adx = getattr(indicators, "adx_14", 0) or 0
    vwap = getattr(indicators, "vwap", 0) or 0
    squeeze_on = getattr(indicators, "squeeze_on", False)

    # MACD values from full_ind
    macd_line = 0.0
    hist_rising = False
    last_hist = 0.0
    if full_ind is not None and hasattr(full_ind, "macd_data"):
        macd_line = float(full_ind.macd_data.macd_line.iloc[-1])
        last_hist = float(full_ind.macd_data.histogram.iloc[-1])
        prev_hist = float(full_ind.macd_data.histogram.iloc[-2])
        hist_rising = last_hist > prev_hist

    # BB pct_b
    pct_b = 0.5  # neutral default
    if full_ind is not None and hasattr(full_ind, "bb"):
        pb = full_ind.bb.pct_b.iloc[-1]
        if not pd.isna(pb):
            pct_b = float(pb)

    regime = result.regime.primary

    # ── Calculate adjustments ────────────────────────
    adjustments: dict[str, int] = {}

    # 1. RSI
    if rsi < 30:        adjustments["RSI"] = +15
    elif rsi < 35:      adjustments["RSI"] = +10
    elif rsi < 40:      adjustments["RSI"] = +5
    elif rsi <= 60:     adjustments["RSI"] = 0
    elif rsi < 65:      adjustments["RSI"] = -5
    elif rsi < 70:      adjustments["RSI"] = -10
    else:               adjustments["RSI"] = -15

    # 2. MACD
    if macd_line > 0 and hist_rising:       adjustments["MACD"] = +12
    elif macd_line > 0 and not hist_rising:  adjustments["MACD"] = +6
    elif macd_line < 0 and hist_rising:     adjustments["MACD"] = -3
    else:                                   adjustments["MACD"] = -12

    # 3. SMA20
    pct_from_sma20 = (price - sma20) / sma20 * 100
    if pct_from_sma20 > 3:      adjustments["SMA20"] = +10
    elif pct_from_sma20 > 1:    adjustments["SMA20"] = +6
    elif pct_from_sma20 > 0:    adjustments["SMA20"] = +3
    elif pct_from_sma20 > -1:   adjustments["SMA20"] = -3
    elif pct_from_sma20 > -3:   adjustments["SMA20"] = -6
    else:                       adjustments["SMA20"] = -10

    # 4. SMA50
    pct_from_sma50 = (price - sma50) / sma50 * 100
    if pct_from_sma50 > 3:      adjustments["SMA50"] = +10
    elif pct_from_sma50 > 1:    adjustments["SMA50"] = +6
    elif pct_from_sma50 > 0:    adjustments["SMA50"] = +3
    elif pct_from_sma50 > -1:   adjustments["SMA50"] = -3
    elif pct_from_sma50 > -3:   adjustments["SMA50"] = -6
    else:                       adjustments["SMA50"] = -10

    # 5. Bollinger Bands
    if pct_b < 0:           adjustments["BB"] = +10
    elif pct_b < 0.2:       adjustments["BB"] = +7
    elif pct_b < 0.4:       adjustments["BB"] = +3
    elif pct_b <= 0.6:      adjustments["BB"] = 0
    elif pct_b < 0.8:       adjustments["BB"] = -3
    elif pct_b <= 1.0:      adjustments["BB"] = -7
    else:                   adjustments["BB"] = -10

    # 6. ADX (trend strength × direction from regime)
    if adx < 20:
        adjustments["ADX"] = 0
    elif regime == ChartRegime.BEARISH:
        adjustments["ADX"] = -8 if adx > 30 else -4
    elif regime == ChartRegime.BULLISH:
        adjustments["ADX"] = +8 if adx > 30 else +4
    else:
        adjustments["ADX"] = 0

    # 7. VWAP
    if vwap > 0:
        pct_from_vwap = (price - vwap) / vwap * 100
        if pct_from_vwap > 2:       adjustments["VWAP"] = +8
        elif pct_from_vwap > 0:     adjustments["VWAP"] = +4
        elif pct_from_vwap > -2:    adjustments["VWAP"] = -4
        else:                       adjustments["VWAP"] = -8
    else:
        adjustments["VWAP"] = 0

    # 8. TTM Squeeze
    if squeeze_on:
        if regime == ChartRegime.BULLISH:     adjustments["Squeeze"] = +3
        elif regime == ChartRegime.BEARISH:   adjustments["Squeeze"] = -3
        else:                                 adjustments["Squeeze"] = 0
    else:
        if last_hist > 0 and hist_rising:         adjustments["Squeeze"] = +7
        elif last_hist > 0:                       adjustments["Squeeze"] = +3
        elif last_hist < 0 and not hist_rising:   adjustments["Squeeze"] = -7
        else:                                     adjustments["Squeeze"] = -3

    # 9. Chart Regime
    regime_adj = {
        ChartRegime.BULLISH:         +10,
        ChartRegime.NEAR_SUPPORT:    +6,
        ChartRegime.NEUTRAL:          0,
        ChartRegime.NEAR_RESISTANCE: -6,
        ChartRegime.BEARISH:         -10,
        ChartRegime.OVEREXTENDED:    -8,
    }
    adjustments["Regime"] = regime_adj.get(regime, 0)

    # ── Votes (simple +1 / -1 / 0 per indicator) ────
    votes = {k: (1 if v > 0 else -1 if v < 0 else 0) for k, v in adjustments.items()}
    bullish_signals = sum(1 for v in votes.values() if v > 0)
    bearish_signals = sum(1 for v in votes.values() if v < 0)
    neutral_count   = sum(1 for v in votes.values() if v == 0)

    # ── Final probability (weighted) ─────────────────
    total_adj = sum(adjustments.values())
    bullish_prob = max(15.0, min(85.0, 50.0 + total_adj))
    bearish_prob = 100.0 - bullish_prob

    direction = (
        "BULLISH" if bullish_prob > 52
        else "BEARISH" if bullish_prob < 48
        else "NEUTRAL"
    )
    direction_emoji = "🟢" if direction == "BULLISH" else "🔴" if direction == "BEARISH" else "🟡"
    majority_count = max(bullish_signals, bearish_signals)

    # ── Bias color theme ─────────────────────────────
    if direction == "BULLISH":
        bias_bg, bias_color, bias_border = "#1a4a2e", "#3fb950", "#3fb950"
    elif direction == "BEARISH":
        bias_bg, bias_color, bias_border = "#4a1a1a", "#f85149", "#f85149"
    else:
        bias_bg, bias_color, bias_border = "#3a3a1a", "#d29922", "#d29922"

    em_pct = expected_move / price * 100 if price > 0 else 0.0

    # ── CSP / CC detail lines ─────────────────────────
    csp_detail = "Outside 1&#963; expected move"
    cc_detail  = "Outside 1&#963; expected move"
    if result.recommendations:
        best = result.recommendations[0]
        c = best.option.contract
        opt = best.option
        delta_str  = f"&#916;{abs(c.delta):.2f}" if c.delta else ""
        theta_str  = f"&#952;${abs(c.theta):.2f}/day" if c.theta else ""
        detail_str = f"Best: ${c.strike:.2f} &middot; {delta_str} &middot; ${opt.premium:.2f} premium &middot; {theta_str}".strip(" &middot;")
        if strategy == Strategy.CASH_SECURED_PUT:
            csp_detail = detail_str
        else:
            cc_detail = detail_str

    # ── Indicator breakdown rows (with Reason) ────────
    def _reason(key: str) -> str:
        if key == "RSI":
            if rsi < 35:   return "Approaching oversold"
            if rsi > 65:   return "Approaching overbought"
            return "Neutral zone"
        if key == "MACD":
            if macd_line > 0 and hist_rising:  return "Positive + rising"
            if macd_line > 0:                  return "Positive + falling"
            if macd_line < 0 and hist_rising:  return "Negative + rising"
            return "Negative + falling"
        if key == "SMA20":
            pct = pct_from_sma20
            return f"Price {abs(pct):.1f}% {'above' if pct >= 0 else 'below'} SMA20"
        if key == "SMA50":
            pct = pct_from_sma50
            return f"Price {abs(pct):.1f}% {'above' if pct >= 0 else 'below'} SMA50"
        if key == "BB":
            if pct_b < 0.2:    return "Lower band — oversold zone"
            if pct_b > 0.8:    return "Upper band — overbought zone"
            return "Mid-band — neutral"
        if key == "ADX":
            if adx < 20:       return "Ranging — no trend bias"
            return f"{'Bullish' if regime == ChartRegime.BULLISH else 'Bearish'} trend confirmed"
        if key == "VWAP":
            if vwap <= 0:      return "No VWAP data"
            pct = (price - vwap) / vwap * 100
            return f"Price {abs(pct):.1f}% {'above' if pct >= 0 else 'below'} VWAP"
        if key == "Squeeze":
            if squeeze_on:     return "Compression — volatility pending"
            return "Volatility releasing"
        if key == "Regime":
            return regime.value
        return ""

    indicator_rows = [
        ("RSI(14)",    f"{rsi:.1f}",
         "🟢" if votes["RSI"] > 0 else "🔴" if votes["RSI"] < 0 else "⚪",
         _reason("RSI")),
        ("MACD",       f"{macd_line:.2f}",
         "🟢" if votes["MACD"] > 0 else "🔴" if votes["MACD"] < 0 else "⚪",
         _reason("MACD")),
        ("SMA20",      f"${sma20:.2f}",
         "🟢" if votes["SMA20"] > 0 else "🔴" if votes["SMA20"] < 0 else "⚪",
         _reason("SMA20")),
        ("SMA50",      f"${sma50:.2f}",
         "🟢" if votes["SMA50"] > 0 else "🔴" if votes["SMA50"] < 0 else "⚪",
         _reason("SMA50")),
        ("Bollinger",  f"{pct_b:.2f} %B",
         "🟢" if votes["BB"] > 0 else "🔴" if votes["BB"] < 0 else "⚪",
         _reason("BB")),
        ("ADX(14)",    f"{adx:.1f}",
         "🟢" if votes["ADX"] > 0 else "🔴" if votes["ADX"] < 0 else "⚪",
         _reason("ADX")),
        ("VWAP",       f"${vwap:.2f}" if vwap > 0 else "N/A",
         "🟢" if votes["VWAP"] > 0 else "🔴" if votes["VWAP"] < 0 else "⚪",
         _reason("VWAP")),
        ("TTM Squeeze","ON \U0001f534" if squeeze_on else "OFF \U0001f7e2",
         "🟢" if votes["Squeeze"] > 0 else "🔴" if votes["Squeeze"] < 0 else "⚪",
         _reason("Squeeze")),
        ("Regime",     regime.value,
         "🟢" if votes["Regime"] > 0 else "🔴" if votes["Regime"] < 0 else "⚪",
         _reason("Regime")),
    ]

    # ── Render 2-zone bar (HTML via .format — no f-string to avoid indent issue) ──
    html = (
'<div style="display:grid;grid-template-columns:1fr 1fr;gap:0;'
'border-radius:8px;overflow:hidden;border:1px solid #30363d;margin:8px 0;">'

'<div style="background:{bias_bg};border-right:1px solid #30363d;padding:14px 16px;">'
'<div style="font-size:17px;font-weight:700;color:{bias_color};">'
'{direction_emoji} {direction} {bullish_prob}%'
'</div>'
'<div style="font-size:11px;color:{bias_color};opacity:0.8;margin-top:2px;">'
'{majority_count}/9 {direction_lower} signals'
'</div>'
'<div style="font-size:12px;color:#8b949e;margin-top:8px;">'
'{dte} DTE &nbsp;&middot;&nbsp; &plusmn;${expected_move} (&plusmn;{em_pct}%)'
'</div>'
'<div style="font-size:13px;color:#e6edf3;margin-top:4px;">'
'Range: ${lower} &#8212; ${upper}'
'</div>'
'</div>'

'<div style="background:#161b22;padding:14px 16px;">'
'<div style="margin-bottom:8px;">'
'<div style="font-size:11px;color:#3fb950;font-weight:700;margin-bottom:2px;">'
'&#128154; CSP &#8212; Sell Puts BELOW'
'</div>'
'<div style="font-size:22px;font-weight:700;color:#3fb950;">${lower}</div>'
'<div style="font-size:10px;color:#8b949e;">{csp_detail}</div>'
'</div>'
'<div style="border-top:1px solid #21262d;margin:8px 0;"></div>'
'<div>'
'<div style="font-size:11px;color:#58a6ff;font-weight:700;margin-bottom:2px;">'
'&#128153; CC &#8212; Sell Calls ABOVE'
'</div>'
'<div style="font-size:22px;font-weight:700;color:#58a6ff;">${upper}</div>'
'<div style="font-size:10px;color:#8b949e;">{cc_detail}</div>'
'</div>'
'</div>'
'</div>'
    ).format(
        bias_bg=bias_bg,
        bias_color=bias_color,
        direction_emoji=direction_emoji,
        direction=direction,
        bullish_prob=f"{bullish_prob:.0f}",
        majority_count=majority_count,
        direction_lower=direction.lower(),
        dte=dte,
        expected_move=f"{expected_move:.2f}",
        em_pct=f"{em_pct:.1f}",
        lower=f"{lower:.2f}",
        upper=f"{upper:.2f}",
        csp_detail=csp_detail,
        cc_detail=cc_detail,
    )
    st.markdown(html, unsafe_allow_html=True)

    # ── Indicator breakdown expander ─────────────────
    with st.expander("🔍 Indicator breakdown", expanded=False):
        df = pd.DataFrame(
            indicator_rows,
            columns=["Indicator", "Value", "Vote", "Reason"],
        )
        st.dataframe(df, hide_index=True, use_container_width=True)
        st.caption(
            f"{bullish_signals} bullish / {bearish_signals} bearish / "
            f"{neutral_count} neutral  ·  "
            f"weighted score {total_adj:+.0f} → {bullish_prob:.0f}% bullish"
        )

    st.caption(
        "Estimates use technical indicators only. "
        "Not financial advice. Options involve substantial risk."
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
