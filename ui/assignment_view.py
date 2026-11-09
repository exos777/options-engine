"""
Screener panel: "Max premium, low assignment" — CC vs CSP for the week.

For both sides it estimates each OTM strike's probability of assignment
from the whole chart (scoring/assignment.py), keeps the strikes under the
trader's limit, and shows the ones paying the most — plus a verdict on
which side the week favors.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

import streamlit as st

from scoring.assignment import (
    SafePick,
    SidePicks,
    compare_sides,
    pick_low_assignment,
    technical_view,
)

_SIDE_NAME = {"CSP": "Cash-Secured Put", "CC": "Covered Call"}
_SIDE_COLOR = {"CSP": "#58a6ff", "CC": "#3fb950"}
_HARD_MAX_SPREAD = 0.15     # CLAUDE.md hard filter


def _earnings_in_window(earnings_date: Optional[str], expiration: str) -> Optional[date]:
    try:
        ed = date.fromisoformat(str(earnings_date)[:10]) if earnings_date else None
        exp = date.fromisoformat(expiration)
    except (TypeError, ValueError):
        return None
    return ed if ed is not None and date.today() <= ed <= exp else None


def _pick_card(pick: SafePick, rank: int) -> None:
    est = pick.estimate
    kind = "P" if pick.side == "CSP" else "C"
    color = _SIDE_COLOR[pick.side]
    delta_txt = f"delta {est.delta:.2f}" if est.delta is not None else "delta n/a"
    label = "Top pick" if rank == 0 else f"#{rank + 1}"
    st.markdown(
        f'<div style="background:#161b22;border:1px solid #30363d;'
        f'border-left:4px solid {color};border-radius:8px;padding:10px 12px;'
        f'margin-bottom:6px;">'
        f'<div style="font-size:11px;color:#8b949e;text-transform:uppercase;">{label}</div>'
        f'<div style="font-size:18px;font-weight:700;color:#e6edf3;">'
        f'Sell ${pick.strike:g}{kind} · ${pick.credit:.2f} '
        f'<span style="font-size:13px;color:#8b949e;">(${pick.credit_per_contract:,.0f}/contract, '
        f'{pick.yield_pct:.2%} of collateral)</span></div>'
        f'<div style="font-size:14px;color:#e6edf3;margin-top:4px;">'
        f'Chance of assignment <b style="color:{color};">{est.prob:.0%}</b> '
        f'<span style="color:#8b949e;">· {delta_txt} · market {est.market_prob:.0%} · '
        f'chance it is touched before expiry ~{est.touch_prob:.0%}</span></div>'
        f'<div style="font-size:12px;color:#c9d1d9;margin-top:4px;">'
        f'📏 {est.weekly_ranges:.1f} weekly ranges {"below" if pick.side == "CSP" else "above"} price'
        + (
            f' · 🧱 behind {", ".join(est.levels_between[:4])}'
            + (f" +{len(est.levels_between) - 4} more" if len(est.levels_between) > 4 else "")
            if est.levels_between else " · no level in front of the strike"
        )
        + "</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    with st.expander("Why this probability", expanded=False):
        lines = [f"Options market (IV, no drift): **{est.market_prob:.1%}**"]
        for text, change in est.drivers:
            # Escape "$": two of them in one line render as a LaTeX formula.
            text = text.replace("$", "\\$")
            sign = "−" if change < 0 else "+"
            lines.append(
                f"{sign}{abs(change) * 100:.1f} pts · {text}" if change else f"· {text}"
            )
        lines.append(f"**= {est.prob:.1%}** chance of assignment at expiration")
        st.markdown("  \n".join(lines))


def _side_column(side: SidePicks, note: str = "") -> None:
    st.markdown(f"##### {_SIDE_NAME[side.side]}s")
    if note:
        st.caption(note)
    if not side.picks:
        st.info(side.excluded_reason or "No qualifying strikes.", icon="ℹ️")
        return
    for i, pick in enumerate(side.picks):
        _pick_card(pick, i)


def render_assignment_panel(
    result,
    full_ind,
    calls: list,
    puts: list,
    *,
    min_open_interest: int = 0,
    desired_buy_price: Optional[float] = None,
    cost_basis: Optional[float] = None,
    earnings_date: Optional[str] = None,
) -> None:
    """Render the CC-vs-CSP "max premium, low assignment" section."""
    st.subheader("🎯 Max Premium · Low Assignment")
    st.caption(
        "Assignment odds from the whole chart, not just delta. Over one week "
        "what matters most is how far the strike is in typical weekly ranges "
        "(ATR) and what stands in front of it: swing support/resistance, "
        "SMA20/50, EMA21, VWAP, Bollinger bands, last week's and the 20-day "
        "low/high — stacked, with clusters counted as stronger zones. Also "
        "used: how much the stock really moves vs IV, a TTM squeeze, and a "
        "small pull from trend (SMA · ADX · MACD) and overbought/oversold "
        "(RSI · Bollinger). Nothing guarantees zero assignment — set the most "
        "risk you'll accept."
    )
    max_prob = st.slider(
        "Max chance of assignment", 3, 30, 10, step=1, format="%d%%",
        key="asg_max_prob",
    ) / 100.0

    spot = result.quote.price
    view = technical_view(full_ind, spot)
    ed = _earnings_in_window(earnings_date, result.expiration)
    if ed is not None:
        st.warning(
            f"📅 Earnings {ed:%a %b %d} lands before this expiration — the "
            f"stock can gap through any strike. IV is used in full (no "
            f"realised-vol discount), but a probability can't capture a gap.",
            icon="⚠️",
        )

    common = dict(
        max_prob=max_prob,
        supports=result.support_levels,
        resistances=result.resistance_levels,
        earnings_in_window=ed is not None,
        max_spread_pct=_HARD_MAX_SPREAD,
        min_open_interest=min_open_interest,
    )
    csp = pick_low_assignment(
        puts, "CSP", spot, result.dte, view,
        desired_buy_price=desired_buy_price, **common,
    )
    cc = pick_low_assignment(
        calls, "CC", spot, result.dte, view,
        cost_basis=cost_basis, **common,
    )

    verdict = compare_sides(csp, cc, view)
    color = _SIDE_COLOR.get(verdict.side or "", "#8b949e")
    st.markdown(
        f'<div style="background:#0d1117;border:1px solid {color};border-radius:8px;'
        f'padding:12px 14px;margin:6px 0 10px 0;">'
        f'<div style="font-size:16px;font-weight:700;color:{color};">{verdict.headline}</div>'
        + "".join(
            f'<div style="font-size:13px;color:#c9d1d9;margin-top:4px;">{r}</div>'
            for r in verdict.reasons
        )
        + "</div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)
    with left:
        _side_column(
            csp,
            f"Strikes at or below your desired buy price ${desired_buy_price:,.2f}."
            if desired_buy_price else "",
        )
    with right:
        _side_column(
            cc,
            "Needs 100 shares per contract."
            + (f" Strikes at or above your cost basis ${cost_basis:,.2f}." if cost_basis else ""),
        )
