"""
CSP premium scorer — short-dated puts sold for premium on stocks you'd own.

The trade: sell a put on the earliest expiration (daily or weekly, never
more than 7 days out), collect the richest premium the stock offers, and
buy it back before expiration. Assignment is the acceptable worst case,
which is why the universe is limited to stocks the trader would hold.

Ranking favours premium per day on the cash set aside, then how rich IV
is versus its own history, then safety (delta, distance outside the
expected move, support between strike and spot) and a tight spread —
the spread is paid twice when the plan is to buy back.

Pure functions only — no Streamlit, no network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from strategies.models import OptionContract, SupportResistanceLevel

MAX_DTE = 7

# Score weights (sum to 100).
W_PREMIUM = 35
W_IV_RANK = 20
W_DELTA = 20
W_EXPECTED_MOVE = 10
W_LIQUIDITY = 10
W_SUPPORT = 5

# Premium per day (credit / strike / days) that earns a full premium score.
FULL_SCORE_YIELD_PER_DAY = 0.0030

# Delta sweet spot: far enough out to buy back calmly, close enough to pay.
SWEET_DELTA_LO = 0.15
SWEET_DELTA_HI = 0.25

# A spread this tight (fraction of mid) earns a full liquidity score.
TIGHT_SPREAD = 0.02


@dataclass
class PremiumParams:
    min_delta: float = 0.08
    max_delta: float = 0.35
    max_spread_pct: float = 0.10     # paid on entry AND exit — keep tight
    min_liquidity: int = 50          # open interest OR today's volume
    min_bid: float = 0.05
    profit_target: float = 0.50      # buy back once this share is captured
    stop_multiple: float = 2.0       # buy back if the put doubles


@dataclass
class PremiumPick:
    contract: OptionContract
    credit: float                    # mid, per share
    delta: Optional[float]           # absolute value
    yield_pct: float                 # credit / strike
    yield_per_day: float
    distance_pct: float              # (spot − strike) / spot
    outside_expected_move: bool
    support: Optional[float]         # support level between strike and spot
    spread_pct: float
    score: float
    subscores: dict = field(default_factory=dict)
    # Trade plan
    buy_back_at: float = 0.0         # target buy-to-close price
    stop_at: float = 0.0             # stop buy-to-close price
    assigned_cost: float = 0.0       # cost per share if assigned
    cash_required: float = 0.0       # collateral per contract

    @property
    def strike(self) -> float:
        return self.contract.strike


# ---------------------------------------------------------------------------
# Subscores (each 0–100)
# ---------------------------------------------------------------------------

def premium_score(yield_per_day: float) -> float:
    return max(0.0, min(100.0, yield_per_day / FULL_SCORE_YIELD_PER_DAY * 100.0))


def delta_score(delta: Optional[float]) -> float:
    if delta is None:
        return 50.0
    if SWEET_DELTA_LO <= delta <= SWEET_DELTA_HI:
        return 100.0
    if delta < SWEET_DELTA_LO:
        # 0.15 → 100, 0.08 → 60: safe but thin.
        return max(0.0, 100.0 - (SWEET_DELTA_LO - delta) / 0.07 * 40.0)
    # 0.25 → 100, 0.35 → 40: rich but close to the money.
    return max(0.0, 100.0 - (delta - SWEET_DELTA_HI) / 0.10 * 60.0)


def expected_move_score(distance: float, expected_move_pct: float) -> float:
    if expected_move_pct <= 0:
        return 50.0
    return max(0.0, min(100.0, distance / expected_move_pct * 100.0))


def liquidity_score(spread_pct: float, max_spread_pct: float) -> float:
    if spread_pct <= TIGHT_SPREAD:
        return 100.0
    if spread_pct >= max_spread_pct:
        return 0.0
    return 100.0 * (max_spread_pct - spread_pct) / (max_spread_pct - TIGHT_SPREAD)


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
    iv_rank: float = 50.0,
    expected_move: float = 0.0,
    supports: Optional[list[SupportResistanceLevel]] = None,
    params: Optional[PremiumParams] = None,
) -> list[PremiumPick]:
    """Every eligible OTM put, best first. Empty if nothing qualifies."""
    params = params or PremiumParams()
    supports = supports or []
    if spot <= 0 or dte < 0 or dte > MAX_DTE:
        return []

    days = max(dte, 1)
    em_pct = expected_move / spot if expected_move > 0 else 0.0
    iv_component = max(0.0, min(100.0, iv_rank))
    picks: list[PremiumPick] = []

    for c in puts:
        if c.strike >= spot or c.bid < params.min_bid:
            continue
        if max(c.open_interest or 0, c.volume or 0) < params.min_liquidity:
            continue
        spread = c.bid_ask_spread_pct
        if spread > params.max_spread_pct:
            continue
        delta = abs(c.delta) if c.delta is not None else None
        if delta is not None and not (params.min_delta <= delta <= params.max_delta):
            continue

        credit = c.mid
        yield_pct = credit / c.strike
        ypd = yield_pct / days
        distance = (spot - c.strike) / spot
        support = _support_between(c.strike, spot, supports)

        subs = {
            "premium": premium_score(ypd),
            "iv_rank": iv_component,
            "delta": delta_score(delta),
            "expected_move": expected_move_score(distance, em_pct),
            "liquidity": liquidity_score(spread, params.max_spread_pct),
            "support": 100.0 if support is not None else 30.0,
        }
        score = (
            subs["premium"] * W_PREMIUM
            + subs["iv_rank"] * W_IV_RANK
            + subs["delta"] * W_DELTA
            + subs["expected_move"] * W_EXPECTED_MOVE
            + subs["liquidity"] * W_LIQUIDITY
            + subs["support"] * W_SUPPORT
        ) / 100.0

        picks.append(PremiumPick(
            contract=c,
            credit=credit,
            delta=delta,
            yield_pct=yield_pct,
            yield_per_day=ypd,
            distance_pct=distance,
            outside_expected_move=em_pct > 0 and distance >= em_pct,
            support=support,
            spread_pct=spread,
            score=round(score, 1),
            subscores=subs,
            buy_back_at=round(credit * (1.0 - params.profit_target), 2),
            stop_at=round(credit * params.stop_multiple, 2),
            assigned_cost=round(c.strike - credit, 2),
            cash_required=c.strike * 100.0,
        ))

    picks.sort(key=lambda p: p.score, reverse=True)
    return picks
