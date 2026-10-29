"""
Same-day CSP scorer — sell a put and buy it back before today's close.

The trade: sell a put on the earliest expiration (daily or weekly, never
more than 7 days out) on a stock you'd own, let it decay for the rest of
the session, and buy it back the same day. Assignment is only the
worst case if the position can't be closed.

So the score is built around what a same-day round trip actually earns:

    est. decay today = |theta| × (session hours left ÷ 6.5),
                       capped at 70% of the credit
    net edge         = est. decay today − bid/ask spread
                       (the spread is paid once in, once out)

ranked per dollar of collateral, with liquidity (spread, today's volume,
open interest) a close second because the position must be exited today.
Earnings before expiration halve the decay estimate: event premium does
not bleed off intraday, it collapses after the report.

Pure functions only — no Streamlit, no network.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from strategies.models import OptionContract, SupportResistanceLevel

MAX_DTE = 7
SESSION_HOURS = 6.5

# Score weights (sum to 100).
W_EDGE = 35
W_LIQUIDITY = 25
W_DELTA = 20
W_EXPECTED_MOVE = 10
W_IV_RANK = 5
W_SUPPORT = 5

# Net edge (share of collateral, one session) that earns a full edge score.
FULL_SCORE_EDGE = 0.005

# A same-day hold rarely captures the whole credit — exits come before the
# final minutes, when 0DTE decay is steepest and gamma most dangerous.
MAX_SAME_DAY_CAPTURE = 0.70

# Earnings before expiration: event premium doesn't decay intraday.
EVENT_DECAY_HAIRCUT = 0.50

# Delta sweet spot for a same-day hold: far enough out that an intraday
# move doesn't blow through the stop, close enough to decay meaningfully.
SWEET_DELTA_LO = 0.10
SWEET_DELTA_HI = 0.20

# Liquidity: spread ≤ this earns full spread points; volume / OI earn full
# depth points at these levels (log-scaled below them).
TIGHT_SPREAD = 0.02
FULL_VOLUME = 2000
FULL_OPEN_INTEREST = 5000

CLOSE_BY = "3:45 pm ET"


@dataclass
class PremiumParams:
    min_delta: float = 0.05
    max_delta: float = 0.35
    max_spread_pct: float = 0.10     # paid on entry AND exit — keep tight
    min_liquidity: int = 100         # open interest OR today's volume
    min_bid: float = 0.05
    stop_multiple: float = 2.0       # buy back if the put doubles


@dataclass
class PremiumPick:
    contract: OptionContract
    credit: float                    # mid, per share
    delta: Optional[float]           # absolute value
    est_decay: float                 # expected decay today, per share
    net_edge: float                  # est_decay − spread, per share
    edge_pct: float                  # net_edge / strike
    distance_pct: float              # (spot − strike) / spot
    outside_expected_move: bool
    support: Optional[float]         # support level between strike and spot
    spread_pct: float
    score: float
    event_haircut: bool = False
    subscores: dict = field(default_factory=dict)
    # Trade plan
    target_price: float = 0.0        # where the put should trade if decay holds
    stop_at: float = 0.0             # stop buy-to-close price
    assigned_cost: float = 0.0       # cost per share if assigned
    cash_required: float = 0.0       # collateral per contract

    @property
    def strike(self) -> float:
        return self.contract.strike

    @property
    def spread(self) -> float:
        return max(self.contract.ask - self.contract.bid, 0.0)

    @property
    def capture_pct(self) -> float:
        return self.est_decay / self.credit if self.credit > 0 else 0.0


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def estimate_decay_today(
    contract: OptionContract,
    dte: int,
    hours_left: float,
    event_before_expiry: bool = False,
) -> float:
    """
    Expected drop in the put's price over the rest of today's session,
    per share, if the stock holds still.
    """
    credit = contract.mid
    if credit <= 0:
        return 0.0
    session_frac = max(0.0, min(hours_left, SESSION_HOURS)) / SESSION_HOURS
    if contract.theta is not None:
        decay = abs(contract.theta) * session_frac
    else:
        # No greeks: spread the premium evenly over the sessions left.
        decay = credit * session_frac / max(dte, 1)
    decay = min(decay, credit * MAX_SAME_DAY_CAPTURE)
    if event_before_expiry:
        decay *= EVENT_DECAY_HAIRCUT
    return decay


def edge_score(edge_pct: float) -> float:
    return max(0.0, min(100.0, edge_pct / FULL_SCORE_EDGE * 100.0))


def delta_score(delta: Optional[float]) -> float:
    if delta is None:
        return 50.0
    if SWEET_DELTA_LO <= delta <= SWEET_DELTA_HI:
        return 100.0
    if delta < SWEET_DELTA_LO:
        # 0.10 → 100, 0.05 → 60: safe but barely decays.
        return max(0.0, 100.0 - (SWEET_DELTA_LO - delta) / 0.05 * 40.0)
    # 0.20 → 100, 0.30 → 50, 0.35 → 25: a small move hits the stop.
    return max(0.0, 100.0 - (delta - SWEET_DELTA_HI) / 0.10 * 50.0)


def expected_move_score(distance: float, expected_move_pct: float) -> float:
    if expected_move_pct <= 0:
        return 50.0
    return max(0.0, min(100.0, distance / expected_move_pct * 100.0))


def _log_scaled(value: int, full: int) -> float:
    if value <= 0:
        return 0.0
    return min(1.0, math.log10(value + 1) / math.log10(full + 1))


def liquidity_score(
    spread_pct: float, volume: int, open_interest: int, max_spread_pct: float,
) -> float:
    """
    Can this be closed today near the mid? Half the score is the spread,
    the rest is depth: today's volume (who's trading it now) and open
    interest (standing interest at this strike).
    """
    if spread_pct <= TIGHT_SPREAD:
        spread_part = 1.0
    elif spread_pct >= max_spread_pct:
        spread_part = 0.0
    else:
        spread_part = (max_spread_pct - spread_pct) / (max_spread_pct - TIGHT_SPREAD)
    return 100.0 * (
        0.50 * spread_part
        + 0.30 * _log_scaled(volume, FULL_VOLUME)
        + 0.20 * _log_scaled(open_interest, FULL_OPEN_INTEREST)
    )


def _support_between(
    strike: float, spot: float, supports: list[SupportResistanceLevel],
) -> Optional[float]:
    """Strongest-touched support level sitting between strike and spot."""
    inside = [s for s in supports if strike <= s.price < spot]
    if not inside:
        return None
    return max(inside, key=lambda s: (s.touches, s.price)).price


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

def score_premium_puts(
    puts: list[OptionContract],
    spot: float,
    dte: int,
    *,
    hours_left: float = SESSION_HOURS,
    iv_rank: float = 50.0,
    expected_move: float = 0.0,
    supports: Optional[list[SupportResistanceLevel]] = None,
    event_before_expiry: bool = False,
    params: Optional[PremiumParams] = None,
) -> list[PremiumPick]:
    """
    Every OTM put worth a same-day round trip, best first. A put whose
    expected decay today doesn't beat its spread is dropped.
    """
    params = params or PremiumParams()
    supports = supports or []
    if spot <= 0 or dte < 0 or dte > MAX_DTE:
        return []

    em_pct = expected_move / spot if expected_move > 0 else 0.0
    iv_component = max(0.0, min(100.0, iv_rank))
    picks: list[PremiumPick] = []

    for c in puts:
        if c.strike >= spot or c.bid < params.min_bid:
            continue
        volume, oi = c.volume or 0, c.open_interest or 0
        if max(oi, volume) < params.min_liquidity:
            continue
        spread_pct = c.bid_ask_spread_pct
        if spread_pct > params.max_spread_pct:
            continue
        delta = abs(c.delta) if c.delta is not None else None
        if delta is not None and not (params.min_delta <= delta <= params.max_delta):
            continue

        credit = c.mid
        decay = estimate_decay_today(c, dte, hours_left, event_before_expiry)
        net_edge = decay - max(c.ask - c.bid, 0.0)
        if net_edge <= 0:
            continue
        edge_pct = net_edge / c.strike
        distance = (spot - c.strike) / spot
        support = _support_between(c.strike, spot, supports)

        subs = {
            "edge": edge_score(edge_pct),
            "liquidity": liquidity_score(spread_pct, volume, oi, params.max_spread_pct),
            "delta": delta_score(delta),
            "expected_move": expected_move_score(distance, em_pct),
            "iv_rank": iv_component,
            "support": 100.0 if support is not None else 30.0,
        }
        score = (
            subs["edge"] * W_EDGE
            + subs["liquidity"] * W_LIQUIDITY
            + subs["delta"] * W_DELTA
            + subs["expected_move"] * W_EXPECTED_MOVE
            + subs["iv_rank"] * W_IV_RANK
            + subs["support"] * W_SUPPORT
        ) / 100.0

        picks.append(PremiumPick(
            contract=c,
            credit=credit,
            delta=delta,
            est_decay=decay,
            net_edge=net_edge,
            edge_pct=edge_pct,
            distance_pct=distance,
            outside_expected_move=em_pct > 0 and distance >= em_pct,
            support=support,
            spread_pct=spread_pct,
            score=round(score, 1),
            event_haircut=event_before_expiry,
            subscores=subs,
            target_price=round(max(credit - decay, 0.01), 2),
            stop_at=round(credit * params.stop_multiple, 2),
            assigned_cost=round(c.strike - credit, 2),
            cash_required=c.strike * 100.0,
        ))

    picks.sort(key=lambda p: p.score, reverse=True)
    return picks
