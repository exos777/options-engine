"""
Streamlit rendering helpers for the recommendations and market overview sections.

All functions accept Streamlit-compatible data and call st.* directly.
They are separated from app/main.py to keep the main file focused on layout.
"""

from __future__ import annotations

import math

import numpy as np
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
    RecommendationLabel.AGGRESSIVE:   "🔥",
    RecommendationLabel.BALANCED:     "⚖️",
    RecommendationLabel.CONSERVATIVE: "🛡️",
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

def render_market_overview(
    result: ScreenerResult,
    strategy: Strategy,
    full_ind: FullIndicators | None = None,
) -> None:
    """Render the market overview — 2-row premium-seller dashboard."""
    q = result.quote
    ind = result.indicators
    em = result.expected_move

    # ── RSI label ──────────────────────────────────────────────────────────
    rsi = ind.rsi_14
    if rsi < 30:        rsi_label = "🟢 Oversold"
    elif rsi < 40:      rsi_label = "🟡 Near Oversold"
    elif rsi <= 60:     rsi_label = "⚪ Neutral"
    elif rsi < 70:      rsi_label = "🟡 Near Overbought"
    else:               rsi_label = "🔴 Overbought"

    # ── ATR % of price ─────────────────────────────────────────────────────
    atr = ind.atr_14
    atr_pct = atr / q.price * 100 if q.price > 0 else 0.0
    atr_label = f"{atr_pct:.1f}% of price"

    # ── Vol Environment (IV30 vs HV30) ─────────────────────────────────────
    iv30_val: float | None = None
    hv30_val: float | None = None

    if em > 0 and q.price > 0 and result.dte > 0:
        iv30_val = (em / q.price) / math.sqrt(max(result.dte, 1) / 365) * 100

    if full_ind is not None:
        close = full_ind.close.dropna()
        if len(close) >= 31:
            log_ret = np.log(close / close.shift(1)).dropna()
            hv30_val = float(log_ret.tail(30).std() * math.sqrt(252) * 100)

    if iv30_val is not None and hv30_val is not None and hv30_val > 0:
        iv_ratio = iv30_val / hv30_val
        if iv_ratio > 1.2:
            vol_label = "🟢 IV Rich"
            vol_detail = "good to sell"
        elif iv_ratio < 0.8:
            vol_label = "🔴 IV Cheap"
            vol_detail = "consider waiting"
        else:
            vol_label = "🟡 IV Fair"
            vol_detail = "neutral"
        vol_display = f"IV {iv30_val:.0f}% / HV {hv30_val:.0f}%"
    elif iv30_val is not None:
        vol_label = f"IV ~{iv30_val:.0f}%"
        vol_detail = "HV unavailable"
        vol_display = vol_label
    else:
        vol_label = "—"
        vol_detail = ""
        vol_display = "—"

    # ── Regime ─────────────────────────────────────────────────────────────
    primary = result.regime.primary
    regime_delta_color = _REGIME_COLOR.get(primary, "off")
    secondary_label = result.regime.secondary.value if result.regime.secondary else ""

    # ── Row 1: 5 metrics ───────────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)

    pct = q.change_pct * 100
    col1.metric("Price", f"${q.price:.2f}", f"{pct:+.2f}%")
    col2.metric("RSI (14)", f"{rsi:.1f}", rsi_label)
    col3.metric("ATR (14)", f"${atr:.2f}", atr_label)
    col4.metric("Vol Environment", vol_display, vol_detail)
    col5.metric(
        "Chart Regime",
        primary.value,
        secondary_label,
        delta_color=regime_delta_color,
    )

    # ── Row 2: expected move bar ────────────────────────────────────────────
    if em > 0:
        lower = q.price - em
        upper = q.price + em
        em_pct = em / q.price * 100
        row2 = (
            '<p style="font-size:24px;color:#8b949e;margin:6px 0 4px;">'
            '<b>Expiration:</b> {exp} &nbsp;&middot;&nbsp; '
            '<b>DTE:</b> {dte} &nbsp;&middot;&nbsp; '
            '<b>Expected Move:</b> &plusmn;${em} (&plusmn;{emp}%) &nbsp;&middot;&nbsp; '
            '<span style="color:#3fb950;font-weight:600;">CSP safe: below ${lo}</span>'
            ' &nbsp;&middot;&nbsp; '
            '<span style="color:#58a6ff;font-weight:600;">CC safe: above ${hi}</span>'
            '</p>'
        ).format(
            exp=result.expiration, dte=result.dte,
            em=f"{em:.2f}", emp=f"{em_pct:.1f}",
            lo=f"{lower:.2f}", hi=f"{upper:.2f}",
        )
        st.markdown(row2, unsafe_allow_html=True)
    else:
        st.caption(f"**Expiration:** {result.expiration}  ·  **DTE:** {result.dte}")

    # ── TTM Squeeze alert ───────────────────────────────────────────────────
    if getattr(ind, "squeeze_on", False):
        st.warning(
            "⚡ ⚡ Squeeze ACTIVE — Bollinger Bands inside Keltner Channels. "
            "Big move imminent. Consider waiting for squeeze release before selling.",
            icon="⚡",
        )

    # ── Direction gauge ─────────────────────────────────────────────────────
    render_direction_gauge(result, result.indicators, full_ind)

    # ── Earnings warning ────────────────────────────────────────────────────
    if q.earnings_date:
        from data.provider import earnings_warning
        ew = earnings_warning(q, result.expiration)
        if ew:
            st.error(ew, icon="🗓️")


# ---------------------------------------------------------------------------
# Shared directional probability calculation
# ---------------------------------------------------------------------------

def _calculate_direction_probability(
    result: ScreenerResult,
    indicators,
    full_ind: FullIndicators | None,
) -> dict:
    """Single source of truth for directional probability."""
    price = result.quote.price
    rsi = indicators.rsi_14
    sma20 = indicators.sma_20
    sma50 = indicators.sma_50
    adx = getattr(indicators, "adx_14", 0) or 0
    vwap = getattr(indicators, "vwap", 0) or 0
    squeeze_on = getattr(indicators, "squeeze_on", False)

    macd_val = 0.0
    hist_rising = False
    last_hist = 0.0
    if full_ind is not None and hasattr(full_ind, "macd_data"):
        macd_val = float(full_ind.macd_data.macd_line.iloc[-1])
        last_hist = float(full_ind.macd_data.histogram.iloc[-1])
        prev_hist = float(full_ind.macd_data.histogram.iloc[-2])
        hist_rising = last_hist > prev_hist

    pct_b = 0.5
    if full_ind is not None and hasattr(full_ind, "bb"):
        pb = full_ind.bb.pct_b.iloc[-1]
        if not pd.isna(pb):
            pct_b = float(pb)

    regime = result.regime.primary

    adj: dict[str, int] = {}

    if rsi < 30:        adj["RSI"] = +15
    elif rsi < 35:      adj["RSI"] = +10
    elif rsi < 40:      adj["RSI"] = +5
    elif rsi <= 60:     adj["RSI"] = 0
    elif rsi < 65:      adj["RSI"] = -5
    elif rsi < 70:      adj["RSI"] = -10
    else:               adj["RSI"] = -15

    if macd_val > 0 and hist_rising:    adj["MACD"] = +12
    elif macd_val > 0:                  adj["MACD"] = +6
    elif macd_val < 0 and hist_rising:  adj["MACD"] = -3
    else:                               adj["MACD"] = -12

    pct20 = (price - sma20) / sma20 * 100
    if pct20 > 3:    adj["SMA20"] = +10
    elif pct20 > 1:  adj["SMA20"] = +6
    elif pct20 > 0:  adj["SMA20"] = +3
    elif pct20 > -1: adj["SMA20"] = -3
    elif pct20 > -3: adj["SMA20"] = -6
    else:            adj["SMA20"] = -10

    pct50 = (price - sma50) / sma50 * 100
    if pct50 > 3:    adj["SMA50"] = +10
    elif pct50 > 1:  adj["SMA50"] = +6
    elif pct50 > 0:  adj["SMA50"] = +3
    elif pct50 > -1: adj["SMA50"] = -3
    elif pct50 > -3: adj["SMA50"] = -6
    else:            adj["SMA50"] = -10

    if pct_b < 0:      adj["BB"] = +10
    elif pct_b < 0.2:  adj["BB"] = +7
    elif pct_b < 0.4:  adj["BB"] = +3
    elif pct_b <= 0.6: adj["BB"] = 0
    elif pct_b < 0.8:  adj["BB"] = -3
    elif pct_b <= 1.0: adj["BB"] = -7
    else:              adj["BB"] = -10

    if adx < 20:
        adj["ADX"] = 0
    elif regime == ChartRegime.BEARISH:
        adj["ADX"] = -8 if adx > 30 else -4
    elif regime == ChartRegime.BULLISH:
        adj["ADX"] = +8 if adx > 30 else +4
    else:
        adj["ADX"] = 0

    if vwap > 0:
        pct_vwap = (price - vwap) / vwap * 100
        if pct_vwap > 2:    adj["VWAP"] = +8
        elif pct_vwap > 0:  adj["VWAP"] = +4
        elif pct_vwap > -2: adj["VWAP"] = -4
        else:               adj["VWAP"] = -8
    else:
        adj["VWAP"] = 0

    if squeeze_on:
        if regime == ChartRegime.BULLISH:      adj["Squeeze"] = +3
        elif regime == ChartRegime.BEARISH:    adj["Squeeze"] = -3
        else:                                  adj["Squeeze"] = 0
    else:
        if last_hist > 0 and hist_rising:      adj["Squeeze"] = +7
        elif last_hist > 0:                    adj["Squeeze"] = +3
        elif last_hist < 0 and not hist_rising: adj["Squeeze"] = -7
        else:                                  adj["Squeeze"] = -3

    regime_map = {
        ChartRegime.BULLISH:         +10,
        ChartRegime.NEAR_SUPPORT:    +6,
        ChartRegime.NEUTRAL:          0,
        ChartRegime.NEAR_RESISTANCE: -6,
        ChartRegime.BEARISH:         -10,
        ChartRegime.OVEREXTENDED:    -8,
    }
    adj["Regime"] = regime_map.get(regime, 0)

    total = sum(adj.values())
    bull_prob = max(15.0, min(85.0, 50.0 + total))
    bear_prob = 100.0 - bull_prob

    if bull_prob > 50:
        display_pct = bull_prob
        display_label = "Bullish"
    elif bull_prob < 50:
        display_pct = bear_prob
        display_label = "Bearish"
    else:
        display_pct = 50.0
        display_label = "Neutral"

    bull_count = sum(1 for v in adj.values() if v > 0)
    bear_count = sum(1 for v in adj.values() if v < 0)
    majority = max(bull_count, bear_count)
    agreement = majority / len(adj) if adj else 0

    if agreement >= 0.77:
        confidence = "High Confidence"
    elif agreement >= 0.55:
        confidence = "Moderate Confidence"
    else:
        confidence = "Low Confidence"

    return {
        "bull_prob": bull_prob,
        "bear_prob": bear_prob,
        "display_pct": display_pct,
        "display_label": display_label,
        "bull_count": bull_count,
        "bear_count": bear_count,
        "confidence": confidence,
        "adjustments": adj,
        "total_adj": total,
    }


# ---------------------------------------------------------------------------
# Direction gauge
# ---------------------------------------------------------------------------

def render_direction_gauge(
    result: ScreenerResult,
    indicators,
    full_ind: FullIndicators | None,
) -> None:
    """Compact BEAR/BULL direction gauge with probability bar."""
    prob = _calculate_direction_probability(result, indicators, full_ind)
    bull_prob = prob["bull_prob"]
    display_pct = prob["display_pct"]
    display_label = prob["display_label"]
    ticker = result.quote.ticker
    dte = result.dte

    if display_label == "Bullish":
        emoji = "🟢"
        color = "#3fb950"
    elif display_label == "Bearish":
        emoji = "🔴"
        color = "#f85149"
    else:
        emoji = "🟡"
        color = "#d29922"

    if display_label == "Bullish":
        if display_pct >= 70:
            strat = "Strong Bullish \u2192 Sell CSPs Aggressively"
        else:
            strat = "Leaning Bullish \u2192 Good for CSPs"
        s_color = "#3fb950"
    elif display_label == "Bearish":
        if display_pct >= 70:
            strat = "Strong Bearish \u2192 Sell Covered Calls"
        else:
            strat = "Leaning Bearish \u2192 Cautious on CSPs"
        s_color = "#f85149"
    else:
        strat = "Neutral \u2192 Selective on Both"
        s_color = "#d29922"

    bar_width = 28
    marker_pos = int(bull_prob / 100 * bar_width)
    bar = "\u2501" * marker_pos + "\u25CF" + "\u2501" * (bar_width - marker_pos)

    st.markdown(
        '<div style="'
        "background:#161b22;"
        "border:1px solid #30363d;"
        "border-radius:8px;"
        "padding:12px 16px;"
        "margin:8px 0;"
        'font-family:monospace;">'
        '<div style="font-size:12px;color:#8b949e;margin-bottom:6px;">'
        "\U0001f4ca Market Direction \u2014 {ticker}"
        " &nbsp;\u00b7&nbsp; {dte} Day Outlook"
        "</div>"
        '<div style="display:flex;align-items:center;gap:8px;font-size:13px;">'
        '<span style="color:#f85149;font-weight:700;">BEAR \u25c0</span>'
        '<span style="color:{color};flex:1;letter-spacing:1px;">{bar}</span>'
        '<span style="color:#3fb950;font-weight:700;">\u25b6 BULL</span>'
        '<span style="color:{color};font-weight:700;font-size:15px;">'
        "{display_pct}% {display_label} {emoji}</span>"
        " "
        '<span style="color:{s_color};font-size:12px;'
        "background:{s_color}22;"
        "padding:2px 8px;"
        "border-radius:4px;"
        'border:1px solid {s_color}44;">'
        "{strat}</span>"
        "</div></div>".format(
            ticker=ticker,
            dte=dte,
            color=color,
            bar=bar,
            display_pct=f"{display_pct:.0f}",
            display_label=display_label,
            emoji=emoji,
            s_color=s_color,
            strat=strat,
        ),
        unsafe_allow_html=True,
    )


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
    indicator signals.  Uses the shared probability calculation so gauge
    and forecast always agree.
    """
    prob = _calculate_direction_probability(result, result.indicators, full_ind)
    adjustments = prob["adjustments"]
    total_adj = prob["total_adj"]
    bullish_prob = prob["bull_prob"]
    bearish_prob = prob["bear_prob"]
    display_pct = prob["display_pct"]
    display_label = prob["display_label"]

    price = result.quote.price
    dte = result.dte
    expected_move = result.expected_move
    upper = price + expected_move
    lower = price - expected_move
    indicators = result.indicators
    regime = result.regime.primary

    rsi = indicators.rsi_14
    sma20 = indicators.sma_20
    sma50 = indicators.sma_50
    adx = getattr(indicators, "adx_14", 0) or 0
    vwap = getattr(indicators, "vwap", 0) or 0
    squeeze_on = getattr(indicators, "squeeze_on", False)

    macd_line = 0.0
    hist_rising = False
    if full_ind is not None and hasattr(full_ind, "macd_data"):
        macd_line = float(full_ind.macd_data.macd_line.iloc[-1])
        last_hist = float(full_ind.macd_data.histogram.iloc[-1])
        prev_hist = float(full_ind.macd_data.histogram.iloc[-2])
        hist_rising = last_hist > prev_hist

    pct_b = 0.5
    if full_ind is not None and hasattr(full_ind, "bb"):
        pb = full_ind.bb.pct_b.iloc[-1]
        if not pd.isna(pb):
            pct_b = float(pb)

    pct_from_sma20 = (price - sma20) / sma20 * 100
    pct_from_sma50 = (price - sma50) / sma50 * 100

    votes = {k: (1 if v > 0 else -1 if v < 0 else 0) for k, v in adjustments.items()}
    bullish_signals = sum(1 for v in votes.values() if v > 0)
    bearish_signals = sum(1 for v in votes.values() if v < 0)
    neutral_count   = sum(1 for v in votes.values() if v == 0)

    direction = display_label.upper()
    if display_label == "Bullish":
        direction_emoji = "🟢"
        bias_bg, bias_color, bias_border = "#1a4a2e", "#3fb950", "#3fb950"
    elif display_label == "Bearish":
        direction_emoji = "🔴"
        bias_bg, bias_color, bias_border = "#4a1a1a", "#f85149", "#f85149"
    else:
        direction_emoji = "🟡"
        bias_bg, bias_color, bias_border = "#3a3a1a", "#d29922", "#d29922"

    majority_count = max(bullish_signals, bearish_signals)
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
'{direction_emoji} {direction} {display_pct}%'
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
        display_pct=f"{display_pct:.0f}",
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
            f"weighted score {total_adj:+.0f} → {display_pct:.0f}% {display_label.lower()}"
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
            f"${abs(c.theta):.3f}" if c.theta is not None else "—",
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

        # Plain English explanation (escape $ to avoid LaTeX rendering)
        _safe_text = rec.explanation.replace("$", "&#36;")
        st.markdown(
            '<div style="background:#1e3a5f;border-left:4px solid #58a6ff;'
            'border-radius:4px;padding:10px 14px;margin:8px 0;'
            'font-size:14px;color:#e6edf3;">'
            "\U0001f4a1 " + _safe_text + "</div>",
            unsafe_allow_html=True,
        )

        # Seller's summary
        ss1, ss2 = st.columns(2)

        ss1.metric("Ann. Yield", f"{opt.annualized_return * 100:.1f}%")

        # Earnings + basis
        earnings_status = "🚨 In Window" if opt.earnings_in_window else "✅ Clear"
        ss2.metric("Earnings", earnings_status)
        if opt.above_cost_basis is True:
            ss2.metric("vs Cost Basis", "✅ Above")
        elif opt.above_cost_basis is False:
            ss2.metric("vs Cost Basis", "⚠️ Below")

        # Assignment probability
        assignment_prob = abs(c.delta or 0) * 100
        if assignment_prob < 20:
            _ap_label = f"\U0001f4cc Assignment Probability: {assignment_prob:.0f}% \u2705 Low"
            _ap_color = "#3fb950"
        elif assignment_prob < 35:
            _ap_label = f"\U0001f4cc Assignment Probability: {assignment_prob:.0f}% \u26a0\ufe0f Moderate"
            _ap_color = "#d29922"
        else:
            _ap_label = f"\U0001f4cc Assignment Probability: {assignment_prob:.0f}% \U0001f534 High"
            _ap_color = "#f85149"
        st.markdown(
            '<span style="color:{c};font-weight:600;">{l}</span>'.format(
                c=_ap_color, l=_ap_label,
            ),
            unsafe_allow_html=True,
        )

        # Position sizing
        st.markdown(
            f"**Position Size:** {rec.position_size}  \u2014  {rec.position_size_reason}"
        )

        # S/R context badges
        tags = []
        if opt.near_support:
            tags.append("🟢 Near Support")
        if opt.near_resistance:
            tags.append("🔴 Near Resistance")
        if tags:
            st.caption("  ".join(tags))

        # What happens next
        _render_what_happens_next(rec, strategy, current_price)


def _render_what_happens_next(
    rec: Recommendation,
    strategy: Strategy,
    current_price: float,
) -> None:
    """Show the wheel-strategy next steps for a recommendation."""
    c = rec.option.contract
    premium = rec.option.premium

    if strategy == Strategy.CASH_SECURED_PUT:
        effective_basis = c.strike - premium
        target_lo = c.strike * 1.03
        target_hi = c.strike * 1.05
        st.markdown(
            "<b>What happens next:</b><br><br>"
            "\u2705 <b>If expires worthless:</b> "
            "Keep {premium} premium. Sell another CSP next week.<br><br>"
            "\U0001f4cc <b>If assigned at {strike}:</b> "
            "You buy 100 shares. Effective cost basis: {basis}<br><br>"
            "\u27a1\ufe0f <b>Immediate next step:</b> "
            "Sell covered call above {strike} targeting "
            "{lo}\u2013{hi} strike for 7\u201314 DTE.".format(
                premium=f"${premium:.2f}",
                strike=f"${c.strike:.2f}",
                basis=f"${effective_basis:.2f}",
                lo=f"${target_lo:.2f}",
                hi=f"${target_hi:.2f}",
            ),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<b>What happens next:</b><br><br>"
            "\u2705 <b>If expires worthless:</b> "
            "Keep {premium} premium. "
            "Sell another CC next week.<br><br>"
            "\U0001f4cc <b>If called away at {strike}:</b> "
            "Position closed. Wheel complete \u2014 start new CSP cycle.".format(
                premium=f"${premium:.2f}",
                strike=f"${c.strike:.2f}",
            ),
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Wheel summary card
# ---------------------------------------------------------------------------

def render_wheel_summary(
    result: ScreenerResult,
    strategy: Strategy,
) -> None:
    """Render the wheel strategy summary card at the top of recommendations."""
    recs = result.recommendations
    if not recs:
        no_trade_reason = (
            result.warnings[0] if result.warnings
            else "Conditions unfavorable"
        )
        st.error(f"\u26d4 NO TRADE \u2014 {no_trade_reason}")
        return

    total_premium = sum(r.option.premium for r in recs)
    avg_breakeven = sum(r.option.break_even for r in recs) / len(recs)
    best_size = recs[0].position_size

    st.markdown(
        '<div style="'
        "background:#161b22;"
        "border:1px solid #30363d;"
        "border-left:4px solid #3fb950;"
        "border-radius:8px;"
        "padding:16px;"
        'margin:8px 0;">'
        '<div style="font-size:14px;font-weight:700;'
        'color:#e6edf3;margin-bottom:12px;">'
        "\U0001f3a1 Wheel Strategy Summary \u2014 {ticker}"
        "</div>"
        '<div style="display:grid;'
        "grid-template-columns:repeat(4,1fr);"
        'gap:12px;font-family:monospace;">'
        '<div><div style="color:#8b949e;font-size:11px;">LEGS AVAILABLE</div>'
        '<div style="color:#e6edf3;font-size:16px;font-weight:700;">'
        "{legs}/3</div></div>"
        '<div><div style="color:#8b949e;font-size:11px;">TOTAL PREMIUM</div>'
        '<div style="color:#3fb950;font-size:16px;font-weight:700;">'
        "${total_premium}</div></div>"
        '<div><div style="color:#8b949e;font-size:11px;">AVG BREAK-EVEN</div>'
        '<div style="color:#58a6ff;font-size:16px;font-weight:700;">'
        "${avg_breakeven}</div></div>"
        '<div><div style="color:#8b949e;font-size:11px;">POSITION SIZE</div>'
        '<div style="font-size:14px;font-weight:700;">'
        "{best_size}</div></div>"
        "</div></div>".format(
            ticker=result.quote.ticker,
            legs=len(recs),
            total_premium=f"{total_premium:.2f}",
            avg_breakeven=f"{avg_breakeven:.2f}",
            best_size=best_size,
        ),
        unsafe_allow_html=True,
    )


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

    render_wheel_summary(result, strategy)

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
