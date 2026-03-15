"""
Plotly chart builder for the options screener.

Three-panel figure:
  - Top panel    (55%):  candlestick, SMA20, SMA50, Bollinger Bands,
                         support/resistance lines, strike markers
  - Middle panel (20%):  volume bars
  - Bottom panel (25%):  MACD line, signal line, histogram
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from indicators.technical import FullIndicators
from strategies.models import Recommendation, SupportResistanceLevel


# ---------------------------------------------------------------------------
# Colour constants
# ---------------------------------------------------------------------------

_COL_BULL    = "#26a69a"   # teal  – up candles / positive MACD histogram
_COL_BEAR    = "#ef5350"   # red   – down candles / negative MACD histogram
_COL_SMA20   = "#f7c948"   # gold
_COL_SMA50   = "#64b5f6"   # blue
_COL_BB      = "rgba(100,181,246,0.15)"  # semi-transparent blue fill
_COL_BB_LINE = "rgba(100,181,246,0.50)"  # BB band borders
_COL_SUPPORT = "#26a69a"   # teal
_COL_RESIST  = "#ef5350"   # red
_COL_SAFEST  = "#66bb6a"   # green
_COL_BALANCE = "#f7c948"   # gold
_COL_MAXPREM = "#ef5350"   # red
_COL_VOLUME  = "#90a4ae"   # grey
_COL_MACD    = "#ce93d8"   # purple
_COL_SIGNAL  = "#f7c948"   # gold


def _label_color(label_str: str) -> str:
    s = label_str.lower()
    if "safest" in s:
        return _COL_SAFEST
    if "balance" in s:
        return _COL_BALANCE
    return _COL_MAXPREM


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------

_COL_EM_FILL = "rgba(255,165,0,0.08)"   # light orange shaded zone
_COL_EM_LINE = "rgba(255,165,0,0.70)"   # orange dashed boundary lines


def build_price_chart(
    full_ind: FullIndicators,
    support_levels: list[SupportResistanceLevel],
    resistance_levels: list[SupportResistanceLevel],
    recommendations: list[Recommendation],
    ticker: str,
    expiration: str,
    current_price: float,
    expected_move: float = 0.0,
) -> go.Figure:
    """
    Build and return the three-panel Plotly figure.

    Panels: candlestick+indicators | volume | MACD
    """
    df_index = full_ind.dates

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.55, 0.20, 0.25],
        subplot_titles=[
            f"{ticker}  –  Expiration {expiration}",
            "Volume",
            "MACD (12 / 26 / 9)",
        ],
    )

    # ------------------------------------------------------------------ #
    # Panel 1 — Candlestick
    # ------------------------------------------------------------------ #
    fig.add_trace(
        go.Candlestick(
            x=df_index,
            open=full_ind.open,
            high=full_ind.high,
            low=full_ind.low,
            close=full_ind.close,
            name="Price",
            increasing_line_color=_COL_BULL,
            decreasing_line_color=_COL_BEAR,
            increasing_fillcolor=_COL_BULL,
            decreasing_fillcolor=_COL_BEAR,
            showlegend=False,
        ),
        row=1, col=1,
    )

    # ------------------------------------------------------------------ #
    # Bollinger Bands (upper / lower fill + middle)
    # ------------------------------------------------------------------ #
    bb = full_ind.bb

    # Filled region between upper and lower
    fig.add_trace(
        go.Scatter(
            x=list(df_index) + list(df_index[::-1]),
            y=list(bb.upper) + list(bb.lower[::-1]),
            fill="toself",
            fillcolor=_COL_BB,
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
            name="BB Band",
        ),
        row=1, col=1,
    )
    # Upper band line
    fig.add_trace(
        go.Scatter(x=df_index, y=bb.upper, mode="lines",
                   line=dict(color=_COL_BB_LINE, width=1, dash="dot"),
                   name="BB Upper", showlegend=False),
        row=1, col=1,
    )
    # Lower band line
    fig.add_trace(
        go.Scatter(x=df_index, y=bb.lower, mode="lines",
                   line=dict(color=_COL_BB_LINE, width=1, dash="dot"),
                   name="BB Lower", showlegend=False),
        row=1, col=1,
    )

    # ------------------------------------------------------------------ #
    # SMA 20 & 50
    # ------------------------------------------------------------------ #
    fig.add_trace(
        go.Scatter(x=df_index, y=full_ind.sma20, mode="lines",
                   name="SMA 20", line=dict(color=_COL_SMA20, width=1.5)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df_index, y=full_ind.sma50, mode="lines",
                   name="SMA 50", line=dict(color=_COL_SMA50, width=1.5)),
        row=1, col=1,
    )

    # ------------------------------------------------------------------ #
    # Support & resistance horizontal lines
    # ------------------------------------------------------------------ #
    x0, x1 = df_index[0], df_index[-1]

    for i, sl in enumerate(support_levels):
        fig.add_shape(type="line", x0=x0, x1=x1, y0=sl.price, y1=sl.price,
                      line=dict(color=_COL_SUPPORT, width=1, dash="dot"), row=1, col=1)
        if i == 0:
            fig.add_annotation(x=x1, y=sl.price, text=f"  S ${sl.price:.2f}",
                                showarrow=False, font=dict(color=_COL_SUPPORT, size=10),
                                xanchor="left", row=1, col=1)

    for i, rl in enumerate(resistance_levels):
        fig.add_shape(type="line", x0=x0, x1=x1, y0=rl.price, y1=rl.price,
                      line=dict(color=_COL_RESIST, width=1, dash="dot"), row=1, col=1)
        if i == 0:
            fig.add_annotation(x=x1, y=rl.price, text=f"  R ${rl.price:.2f}",
                                showarrow=False, font=dict(color=_COL_RESIST, size=10),
                                xanchor="left", row=1, col=1)

    # ------------------------------------------------------------------ #
    # Expected move zone (±1σ shaded orange region)
    # ------------------------------------------------------------------ #
    if expected_move > 0:
        em_upper = current_price + expected_move
        em_lower = current_price - expected_move

        # Shaded fill between upper and lower bounds
        fig.add_trace(
            go.Scatter(
                x=list(df_index) + list(df_index[::-1]),
                y=[em_upper] * len(df_index) + [em_lower] * len(df_index),
                fill="toself",
                fillcolor=_COL_EM_FILL,
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
                name="1σ EM Zone",
            ),
            row=1, col=1,
        )
        # Upper bound dashed line
        fig.add_shape(type="line", x0=x0, x1=x1, y0=em_upper, y1=em_upper,
                      line=dict(color=_COL_EM_LINE, width=1.5, dash="dash"), row=1, col=1)
        fig.add_annotation(x=x1, y=em_upper,
                           text=f"  1σ Upper ${em_upper:.2f}",
                           showarrow=False, font=dict(color=_COL_EM_LINE, size=10),
                           xanchor="left", row=1, col=1)
        # Lower bound dashed line
        fig.add_shape(type="line", x0=x0, x1=x1, y0=em_lower, y1=em_lower,
                      line=dict(color=_COL_EM_LINE, width=1.5, dash="dash"), row=1, col=1)
        fig.add_annotation(x=x1, y=em_lower,
                           text=f"  1σ Lower ${em_lower:.2f}",
                           showarrow=False, font=dict(color=_COL_EM_LINE, size=10),
                           xanchor="left", row=1, col=1)

    # Current price reference line
    fig.add_shape(type="line", x0=x0, x1=x1, y0=current_price, y1=current_price,
                  line=dict(color="#ffffff", width=1, dash="dash"), row=1, col=1)

    # ------------------------------------------------------------------ #
    # Recommended strike lines
    # ------------------------------------------------------------------ #
    for rec in recommendations:
        strike = rec.option.contract.strike
        color = _label_color(rec.label.value)
        fig.add_shape(type="line", x0=x0, x1=x1, y0=strike, y1=strike,
                      line=dict(color=color, width=1.5, dash="dashdot"), row=1, col=1)
        fig.add_annotation(x=x1, y=strike, text=f"  {rec.label.value} ${strike:.2f}",
                            showarrow=False, font=dict(color=color, size=10),
                            xanchor="left", row=1, col=1)

    # ------------------------------------------------------------------ #
    # Panel 2 — Volume
    # ------------------------------------------------------------------ #
    close = full_ind.close
    bar_colors = [
        _COL_BULL if c >= p else _COL_BEAR
        for c, p in zip(close, close.shift(1).fillna(close))
    ]
    fig.add_trace(
        go.Bar(x=df_index, y=full_ind.volume, name="Volume",
               marker_color=bar_colors, showlegend=False),
        row=2, col=1,
    )
    # Avg volume line
    fig.add_trace(
        go.Scatter(x=df_index, y=full_ind.avg_volume, mode="lines",
                   name="Avg Vol", line=dict(color=_COL_SMA20, width=1, dash="dot"),
                   showlegend=False),
        row=2, col=1,
    )

    # ------------------------------------------------------------------ #
    # Panel 3 — MACD
    # ------------------------------------------------------------------ #
    md = full_ind.macd_data

    hist_colors = [
        _COL_BULL if v >= 0 else _COL_BEAR
        for v in md.histogram.fillna(0)
    ]
    fig.add_trace(
        go.Bar(x=df_index, y=md.histogram, name="MACD Hist",
               marker_color=hist_colors, showlegend=False),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df_index, y=md.macd_line, mode="lines",
                   name="MACD", line=dict(color=_COL_MACD, width=1.5)),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df_index, y=md.signal_line, mode="lines",
                   name="Signal", line=dict(color=_COL_SIGNAL, width=1.5)),
        row=3, col=1,
    )
    # Zero line on MACD panel
    fig.add_shape(type="line", x0=x0, x1=x1, y0=0, y1=0,
                  line=dict(color="rgba(255,255,255,0.3)", width=1), row=3, col=1)

    # ------------------------------------------------------------------ #
    # Layout
    # ------------------------------------------------------------------ #
    fig.update_layout(
        template="plotly_dark",
        height=680,
        margin=dict(l=10, r=90, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Vol", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)

    return fig
