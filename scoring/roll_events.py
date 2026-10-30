"""
Event awareness for rolls — earnings, and any other event the option
market is pricing between today and the new expiration.

An event (earnings, FDA date, product launch, Fed day…) shows up in the
options as extra implied volatility on every expiration that spans it.
Two consequences for a roll:

  * Leg 2's credit is inflated by event premium. That premium is not
    income you keep for decay — it pays for the gap risk, then collapses
    after the event. So rolls are compared on an event-adjusted credit.
  * The stock is expected to jump by the event move. A Leg 2 strike
    inside that move is likely to be tested by the event itself.

The event move is measured from the IV term structure: the variance
Leg 2 carries beyond a "no event" baseline (Leg 1's IV when Leg 1 ends
before the event, else realised volatility) is attributed to the event.

Pure functions only — no Streamlit, no network.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Optional

# Leg 2 IV this many times the baseline flags an event even when no
# earnings date is known.
TERM_SPIKE_RATIO = 1.30

# Score penalty for a roll spanning an event, outside the event move.
EVENT_PENALTY = 10.0

# Leg 1 IV is only a usable "no event" baseline with this many days left:
# options a day or two from expiry carry noisy, inflated IV.
MIN_BASELINE_DTE = 5

# Roll window — a "pick this expiration instead" suggestion must fit it.
ROLL_MIN_DTE = 7
ROLL_MAX_DTE = 21


@dataclass
class EventContext:
    """What is known about events around a roll. Built once per evaluation."""
    today: date
    leg1_expiration: date
    leg2_expiration: date
    spot: float
    leg1_iv: float = 0.0          # current short's IV
    realized_vol: float = 0.0     # 20-day, annualised
    earnings_date: Optional[date] = None
    earnings_flagged: bool = False   # user ticked "earnings" without a date

    # -- where the earnings fall -------------------------------------------

    @property
    def earnings_before_leg1(self) -> bool:
        """Report lands before the CURRENT short expires."""
        return (
            self.earnings_date is not None
            and self.today <= self.earnings_date <= self.leg1_expiration
        )

    @property
    def earnings_before_leg2(self) -> bool:
        """Report lands before the NEW short expires (rolling into it)."""
        return self.earnings_before(self.leg2_expiration)

    def earnings_before(self, expiration: date) -> bool:
        """Report lands on/before *expiration* (a manual flag counts too)."""
        if self.earnings_date is None:
            return self.earnings_flagged
        return self.today <= self.earnings_date <= expiration

    # -- baseline volatility ------------------------------------------------

    @property
    def base_iv(self) -> float:
        """
        IV with no event in it: Leg 1's when it ends before the event and
        still has MIN_BASELINE_DTE days left, else 20-day realised vol.
        """
        leg1_dte = (self.leg1_expiration - self.today).days
        leg1_usable = (
            self.leg1_iv > 0
            and not self.earnings_before_leg1
            and leg1_dte >= MIN_BASELINE_DTE
        )
        if leg1_usable or (self.realized_vol <= 0 and self.leg1_iv > 0
                           and not self.earnings_before_leg1):
            return self.leg1_iv
        return self.realized_vol

    def term_spike(self, leg2_iv: float) -> bool:
        """Leg 2 priced well above the baseline — an unnamed event."""
        base = self.base_iv
        return base > 0 and leg2_iv >= base * TERM_SPIKE_RATIO


def implied_event_move(spot: float, event_iv: float, base_iv: float, dte: int) -> float:
    """
    ±$ move the market prices for the event: the variance in *event_iv*
    over *dte* days beyond the no-event baseline.
    """
    if spot <= 0 or event_iv <= 0 or base_iv <= 0 or dte <= 0:
        return 0.0
    excess = max(event_iv ** 2 - base_iv ** 2, 0.0) * dte / 365.0
    return spot * math.sqrt(excess)


def measure_event_move(
    ctx: EventContext, leg2_iv: float, dte: int,
) -> tuple[float, str]:
    """
    (±$ move, how it was measured) for a Leg 2 that spans an event:

      "event" — the event's own move, from IV above the baseline
      "full"  — the event can't be separated (Leg 2 IV isn't above the
                baseline), so Leg 2's whole expected move, which includes
                the event, is used instead — the conservative bound
      "none"  — no IV to measure anything
    """
    move = implied_event_move(ctx.spot, leg2_iv, ctx.base_iv, dte)
    if move > 0:
        return move, "event"
    if leg2_iv > 0 and ctx.spot > 0 and dte > 0:
        return ctx.spot * leg2_iv * math.sqrt(dte / 365.0), "full"
    return 0.0, "none"


@dataclass
class CandidateEvent:
    """How one Leg 2 candidate relates to the event."""
    kind: Optional[str] = None       # "earnings" | "iv_spike" | None
    event_move: float = 0.0          # ±$ expected on the event (0 = unknown)
    move_basis: str = "none"         # "event" | "full" | "none" — see measure_event_move
    inside_move: bool = False        # strike within the event move
    adjusted_credit: float = 0.0     # roll credit without event premium
    event_premium: float = 0.0       # per share, on the new leg
    penalty: float = 0.0
    reject_reason: str = ""

    @property
    def spans_event(self) -> bool:
        return self.kind is not None


def assess_candidate(
    *,
    strike: float,
    sell_to_open: float,
    buy_to_close: float,
    leg2_iv: float,
    dte: int,
    strategy: str,
    ctx: EventContext,
    expiration: Optional[date] = None,
) -> CandidateEvent:
    """
    Classify a candidate against the event context and adjust its credit.
    *expiration* is the candidate's own expiry (defaults to Leg 2's) — a
    later expiration offered as a Safe alternative may span the report
    even when the chosen Leg 2 doesn't.
    """
    credit = sell_to_open - buy_to_close
    out = CandidateEvent(adjusted_credit=credit)

    if ctx.earnings_before(expiration or ctx.leg2_expiration):
        out.kind = "earnings"
    elif ctx.term_spike(leg2_iv):
        out.kind = "iv_spike"
    else:
        return out

    base = ctx.base_iv
    if leg2_iv > 0 and base > 0 and leg2_iv > base:
        # Premium scales roughly with IV: strip the event share of the new leg.
        out.event_premium = sell_to_open * (1.0 - base / leg2_iv)
        out.adjusted_credit = credit - out.event_premium
    out.event_move, out.move_basis = measure_event_move(ctx, leg2_iv, dte)

    if out.event_move > 0:
        if strategy == "CSP":
            out.inside_move = strike > ctx.spot - out.event_move
        else:
            out.inside_move = strike < ctx.spot + out.event_move
        if out.inside_move:
            what = "earnings" if out.kind == "earnings" else "event"
            out.reject_reason = (
                f"Inside the ±${out.event_move:.2f} {what} move — "
                f"likely tested by the {what}"
            )
        else:
            out.penalty = EVENT_PENALTY
    elif out.kind == "earnings":
        # Spans earnings and there's no IV to size any move: stay strict.
        out.reject_reason = "Earnings inside DTE window (no IV to size the move)"
    else:
        out.penalty = EVENT_PENALTY
    return out


def last_expiration_before(
    expirations: list[str],
    event: date,
    after: date,
    today: Optional[date] = None,
    min_dte: int = ROLL_MIN_DTE,
    max_dte: int = ROLL_MAX_DTE,
) -> Optional[str]:
    """
    Latest expiration after *after* that ends before *event* AND is a
    valid roll target (min_dte–max_dte days out). None if there isn't one —
    then every roll spans the event.
    """
    today = today or date.today()
    best: Optional[str] = None
    for exp in expirations:
        try:
            d = date.fromisoformat(exp)
        except (TypeError, ValueError):
            continue
        dte = (d - today).days
        if (
            after < d < event
            and min_dte <= dte <= max_dte
            and (best is None or d > date.fromisoformat(best))
        ):
            best = exp
    return best
