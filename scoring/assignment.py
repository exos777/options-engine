"""
Probability of assignment that uses the whole chart, not just delta.

Delta is the options market's risk-neutral view: it assumes no drift and
prices volatility at implied vol. This module starts from that and adjusts
it, step by step, for what the technical picture adds:

    1. Market (IV)       P(expire ITM) from the option's own IV, no drift
    2. Realised vs IV    implied vol usually overstates the move a stock
                         actually makes; blend in 20-day realised vol
                         (skipped when earnings land before expiration —
                         then IV's event premium is real)
    3. Trend & momentum  a modest annual drift from price vs SMA20/50
                         (scaled by ADX), MACD histogram, RSI and Bollinger
                         %B — capped low: over one week trend is mostly
                         noise; overbought/oversold snap-backs matter more
    4. Levels            every floor (puts) or ceiling (calls) standing
                         between price and strike — swing pivots, SMA20/50,
                         EMA21, VWAP, Bollinger bands, last week's and the
                         20-day low/high — each shaves the probability;
                         stacked with diminishing returns, levels within a
                         quarter ATR merge into one stronger level

On a one-week trade the strike's distance in typical weekly ranges
(ATR × √sessions) and what stands in between matter more than direction,
so the model leans on volatility and levels, not trend. A TTM squeeze
(coiled volatility) cancels the realised-vol discount: a breakout is due.

Each step's effect is kept as a "driver" so the UI can show exactly why
the number differs from delta. Pure functions — no Streamlit, no network.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from strategies.models import OptionContract, SupportResistanceLevel

# Drift contributions, annualised (0.25 ≈ +0.48% over a 7-day week).
# Trend is deliberately small: over a week it is mostly noise.
TREND_DRIFT = 0.25
MACD_DRIFT = 0.06
RSI_EXTREME_DRIFT = 0.20      # mean reversion beyond 70 / below 30
RSI_MOMENTUM_DRIFT = 0.04     # 60–70 / 30–40: momentum persists a little
BOLLINGER_DRIFT = 0.15        # outside the bands: snap-back tendency
MAX_DRIFT = 0.40
BIAS_THRESHOLD = 0.05         # trend + MACD drift that counts as a direction

# Realised-vol blend: weight on realised, and never below this share of IV.
RV_WEIGHT = 0.40
MIN_SIGMA_SHARE_OF_IV = 0.70

# Levels between price and strike: each cuts the probability by up to
# LEVEL_CUSHION × its weight (0–1); stacked multiplicatively, total cut
# capped. Levels within LEVEL_MERGE_ATR of each other are one zone.
LEVEL_CUSHION = 0.20
MAX_TOTAL_CUSHION = 0.40
LEVEL_MERGE_ATR = 0.25
CONFLUENCE_BONUS = 0.15        # weight added per extra level in a zone
NEAR_SPOT_ATR = 0.20           # a level this close to price is about to be
                               # tested — it counts half

# Fixed weights for indicator-derived levels (pivots use their strength).
LEVEL_WEIGHTS = {
    "SMA50": 0.60,
    "SMA20": 0.45,
    "EMA21": 0.40,
    "VWAP": 0.40,
    "Bollinger band": 0.50,
    "last week's": 0.50,
    "20-day": 0.60,
}

PROB_FLOOR, PROB_CAP = 0.002, 0.99


# ---------------------------------------------------------------------------
# Technical view — computed once per ticker
# ---------------------------------------------------------------------------

@dataclass
class Level:
    price: float
    name: str
    weight: float          # 0–1
    side: str              # "support" | "resistance" | "both" (moving averages)


@dataclass
class TechnicalView:
    drift: float = 0.0                       # annualised, capped
    parts: list[tuple[str, float]] = field(default_factory=list)
    realized_vol: float = 0.0                # 20-day annualised
    bias: str = "neutral"                    # bullish / neutral / bearish
    atr: float = 0.0                         # 14-day, $
    squeeze: bool = False                    # TTM squeeze on
    levels: list[Level] = field(default_factory=list)  # indicator levels

    @property
    def reasons(self) -> list[str]:
        return [label for label, d in self.parts if d != 0]


def _last(series) -> Optional[float]:
    try:
        v = float(series.dropna().iloc[-1])
        return v if math.isfinite(v) else None
    except Exception:
        return None


def technical_view(full_ind, spot: float) -> TechnicalView:
    """Condense the indicator series into a drift + realised vol."""
    view = TechnicalView()
    if full_ind is None or spot <= 0:
        return view

    parts: list[tuple[str, float]] = []

    # Trend: price vs SMA20 / SMA50, strength from ADX.
    sma20, sma50 = _last(full_ind.sma20), _last(full_ind.sma50)
    adx = _last(full_ind.adx14) or 20.0
    strength = 0.5 + 0.5 * max(0.0, min(1.0, (adx - 15.0) / 25.0))
    if sma20 and sma50:
        if spot > sma20 > sma50:
            parts.append((f"uptrend (above SMA20/50, ADX {adx:.0f})", TREND_DRIFT * strength))
        elif spot < sma20 < sma50:
            parts.append((f"downtrend (below SMA20/50, ADX {adx:.0f})", -TREND_DRIFT * strength))
        elif spot > sma50:
            parts.append(("above SMA50, mixed short term", TREND_DRIFT * 0.5 * strength))
        elif spot < sma50:
            parts.append(("below SMA50, mixed short term", -TREND_DRIFT * 0.5 * strength))

    # Momentum: MACD histogram.
    hist = _last(full_ind.macd_data.histogram)
    if hist is not None and hist != 0:
        parts.append(
            ("MACD rising" if hist > 0 else "MACD falling",
             MACD_DRIFT if hist > 0 else -MACD_DRIFT)
        )

    # RSI: extremes revert, the 60s/30s persist a little.
    rsi = _last(full_ind.rsi14)
    if rsi is not None:
        if rsi >= 70:
            parts.append((f"RSI {rsi:.0f} overbought", -RSI_EXTREME_DRIFT))
        elif rsi <= 30:
            parts.append((f"RSI {rsi:.0f} oversold", RSI_EXTREME_DRIFT))
        elif rsi >= 60:
            parts.append((f"RSI {rsi:.0f} strong", RSI_MOMENTUM_DRIFT))
        elif rsi <= 40:
            parts.append((f"RSI {rsi:.0f} weak", -RSI_MOMENTUM_DRIFT))

    # Bollinger %B: outside the bands tends to snap back.
    pct_b = _last(full_ind.bb.pct_b)
    if pct_b is not None:
        if pct_b > 1.0:
            parts.append(("above upper Bollinger band", -BOLLINGER_DRIFT))
        elif pct_b < 0.0:
            parts.append(("below lower Bollinger band", BOLLINGER_DRIFT))

    drift = sum(d for _, d in parts)
    view.drift = max(-MAX_DRIFT, min(MAX_DRIFT, drift))
    view.parts = parts
    # Bias is the chart's DIRECTION (trend + MACD); RSI / Bollinger
    # snap-backs move the odds but don't make a downtrend "neutral".
    direction = sum(d for label, d in parts if "trend" in label or "SMA50" in label or "MACD" in label)
    view.bias = (
        "bullish" if direction > BIAS_THRESHOLD
        else "bearish" if direction < -BIAS_THRESHOLD else "neutral"
    )

    # Realised vol, 20 sessions.
    try:
        close = full_ind.close.astype(float).dropna()
        rets = (close / close.shift(1)).apply(math.log).dropna().tail(20)
        rv = float(rets.std() * math.sqrt(252))
        view.realized_vol = rv if math.isfinite(rv) else 0.0
    except Exception:
        view.realized_vol = 0.0

    view.atr = _last(full_ind.atr14) or 0.0
    try:
        view.squeeze = bool(full_ind.squeeze.dropna().iloc[-1])
    except Exception:
        view.squeeze = False
    view.levels = _indicator_levels(full_ind)
    return view


def _indicator_levels(full_ind) -> list[Level]:
    """Moving averages, VWAP, Bollinger bands and recent lows/highs as levels."""
    out: list[Level] = []

    def add(price, name, key, side):
        if price is not None and price > 0:
            out.append(Level(float(price), name, LEVEL_WEIGHTS[key], side))

    add(_last(full_ind.sma50), "SMA50", "SMA50", "both")
    add(_last(full_ind.sma20), "SMA20", "SMA20", "both")
    try:
        ema21 = full_ind.close.astype(float).ewm(span=21, adjust=False).mean()
        add(_last(ema21), "EMA21", "EMA21", "both")
    except Exception:
        pass
    add(_last(full_ind.vwap_series), "VWAP", "VWAP", "both")
    add(_last(full_ind.bb.lower), "lower Bollinger band", "Bollinger band", "support")
    add(_last(full_ind.bb.upper), "upper Bollinger band", "Bollinger band", "resistance")
    try:
        # Last 5 completed sessions ≈ last week; exclude today's bar.
        low5 = float(full_ind.low.iloc[-6:-1].min())
        high5 = float(full_ind.high.iloc[-6:-1].max())
        add(low5, "last week's low", "last week's", "support")
        add(high5, "last week's high", "last week's", "resistance")
        add(float(full_ind.low.tail(20).min()), "20-day low", "20-day", "support")
        add(float(full_ind.high.tail(20).max()), "20-day high", "20-day", "resistance")
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------------
# Probability of assignment
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def prob_itm(
    spot: float, strike: float, years: float, sigma: float, drift: float, option_type: str,
) -> float:
    """P(finishing in the money) under a lognormal with the given drift."""
    if spot <= 0 or strike <= 0 or years <= 0 or sigma <= 0:
        if option_type == "put":
            return 1.0 if spot < strike else 0.0
        return 1.0 if spot > strike else 0.0
    d2 = (math.log(spot / strike) + (drift - 0.5 * sigma * sigma) * years) / (
        sigma * math.sqrt(years)
    )
    return _norm_cdf(-d2) if option_type == "put" else _norm_cdf(d2)


@dataclass
class AssignmentEstimate:
    prob: float                      # final P(assigned at expiration)
    market_prob: float               # step 1: IV, no drift
    delta: Optional[float]           # |delta| for comparison
    drivers: list[tuple[str, float]] = field(default_factory=list)  # (label, Δprob)
    sigma: float = 0.0
    cushion: Optional[float] = None  # nearest level between spot and strike
    levels_between: list[str] = field(default_factory=list)  # e.g. "SMA50 $225.10"
    weekly_ranges: float = 0.0       # strike distance ÷ (ATR × √sessions)

    @property
    def touch_prob(self) -> float:
        """Chance the stock trades at the strike before expiry (≈ 2 × P(ITM))."""
        return min(1.0, 2.0 * self.prob)


def _levels_between(
    strike: float, spot: float, option_type: str, view: TechnicalView,
    supports: list[SupportResistanceLevel],
    resistances: list[SupportResistanceLevel],
) -> list[Level]:
    """Every floor (puts) / ceiling (calls) standing between price and strike."""
    want = "support" if option_type == "put" else "resistance"
    pool = [lv for lv in view.levels if lv.side in (want, "both")]
    pivots = supports if option_type == "put" else resistances
    pool += [
        Level(lv.price, f"swing {want}", 0.30 + 0.50 * max(0.0, min(1.0, lv.strength)), want)
        for lv in pivots
    ]
    if option_type == "put":
        return [lv for lv in pool if strike < lv.price < spot]
    return [lv for lv in pool if spot < lv.price < strike]


def _zones(levels: list[Level], atr: float) -> list[list[Level]]:
    """Merge levels within LEVEL_MERGE_ATR × ATR into confluence zones."""
    if not levels:
        return []
    gap = atr * LEVEL_MERGE_ATR if atr > 0 else 0.0
    ordered = sorted(levels, key=lambda lv: lv.price)
    zones: list[list[Level]] = [[ordered[0]]]
    for lv in ordered[1:]:
        if lv.price - zones[-1][-1].price <= gap:
            zones[-1].append(lv)
        else:
            zones.append([lv])
    return zones


def estimate_assignment(
    contract: OptionContract,
    spot: float,
    dte: int,
    view: TechnicalView,
    *,
    supports: Optional[list[SupportResistanceLevel]] = None,
    resistances: Optional[list[SupportResistanceLevel]] = None,
    earnings_in_window: bool = False,
) -> Optional[AssignmentEstimate]:
    """Step-by-step P(assignment) for one short option. None without IV."""
    iv = contract.implied_volatility or 0.0
    if iv <= 0 or spot <= 0:
        return None
    kind = contract.option_type
    years = max(dte, 1) / 365.0
    k = contract.strike

    # 1. Market view: IV, no drift.
    p_market = prob_itm(spot, k, years, iv, 0.0, kind)
    drivers: list[tuple[str, float]] = []

    # 2. Realised vs implied vol (not across earnings: that premium is real;
    #    not in a TTM squeeze: coiled volatility is about to expand).
    sigma = iv
    if not earnings_in_window and not view.squeeze and view.realized_vol > 0:
        sigma = max(
            (1 - RV_WEIGHT) * iv + RV_WEIGHT * view.realized_vol,
            iv * MIN_SIGMA_SHARE_OF_IV,
        )
    p_vol = prob_itm(spot, k, years, sigma, 0.0, kind)
    if abs(p_vol - p_market) >= 0.0005:
        label = (
            f"stock moves less than IV implies (realised {view.realized_vol:.0%} vs IV {iv:.0%})"
            if sigma < iv else
            f"stock moves more than IV implies (realised {view.realized_vol:.0%} vs IV {iv:.0%})"
        )
        drivers.append((label, p_vol - p_market))
    elif earnings_in_window:
        drivers.append(("earnings before expiry — full IV kept", 0.0))
    elif view.squeeze:
        drivers.append(("TTM squeeze on — breakout due, full IV kept", 0.0))

    # 3. Trend & momentum drift, split across its parts for the breakdown.
    p_drift = prob_itm(spot, k, years, sigma, view.drift, kind)
    total = p_drift - p_vol
    raw = sum(d for _, d in view.parts)
    if view.parts and raw != 0 and abs(total) >= 0.0005:
        scale = view.drift / raw            # cap applied proportionally
        for label, d in view.parts:
            drivers.append((label, total * (d * scale) / view.drift))

    # 4. Every level between price and strike, stacked by confluence zone.
    prob = p_drift
    between = _levels_between(k, spot, kind, view, supports or [], resistances or [])
    zones = _zones(between, view.atr)
    zone_cuts: list[tuple[str, float]] = []
    for zone in zones:
        weight = min(1.0, max(lv.weight for lv in zone) + CONFLUENCE_BONUS * (len(zone) - 1))
        mid = sum(lv.price for lv in zone) / len(zone)
        if view.atr > 0 and abs(spot - mid) < NEAR_SPOT_ATR * view.atr:
            weight *= 0.5                       # about to be tested anyway
        names = " + ".join(f"{lv.name} ${lv.price:,.2f}" for lv in sorted(zone, key=lambda l: -l.weight))
        zone_cuts.append((names, LEVEL_CUSHION * weight))
    keep = 1.0
    for _, f in zone_cuts:
        keep *= 1.0 - f
    total_cut = min(1.0 - keep, MAX_TOTAL_CUSHION)
    raw_cut = sum(f for _, f in zone_cuts)
    if total_cut > 0 and raw_cut > 0:
        removed = prob * total_cut
        label = "below" if kind == "put" else "above"
        for names, f in zone_cuts:
            share = removed * f / raw_cut
            if share >= 0.0005:
                drivers.append((f"{names} stands {label} price, before the strike", -share))
        prob -= removed

    # Distance in typical weekly ranges (ATR × √sessions to expiry).
    sessions = max(1.0, dte * 5.0 / 7.0)
    weekly_ranges = (
        abs(spot - k) / (view.atr * math.sqrt(sessions)) if view.atr > 0 else 0.0
    )
    nearest = (
        max(between, key=lambda lv: lv.price) if kind == "put"
        else min(between, key=lambda lv: lv.price)
    ) if between else None

    prob = max(PROB_FLOOR, min(PROB_CAP, prob))
    return AssignmentEstimate(
        prob=prob,
        market_prob=p_market,
        delta=abs(contract.delta) if contract.delta is not None else None,
        drivers=drivers,
        sigma=sigma,
        cushion=nearest.price if nearest is not None else None,
        levels_between=[
            f"{lv.name} ${lv.price:,.2f}"
            for lv in sorted(between, key=lambda l: -l.price if kind == "put" else l.price)
        ],
        weekly_ranges=weekly_ranges,
    )


# ---------------------------------------------------------------------------
# Picks: most premium under an assignment-probability limit
# ---------------------------------------------------------------------------

@dataclass
class SafePick:
    contract: OptionContract
    side: str                          # "CSP" | "CC"
    credit: float                      # mid, per share
    yield_pct: float                   # credit / collateral
    estimate: AssignmentEstimate

    @property
    def strike(self) -> float:
        return self.contract.strike

    @property
    def credit_per_contract(self) -> float:
        return self.credit * 100.0


@dataclass
class SidePicks:
    side: str
    picks: list[SafePick] = field(default_factory=list)   # best premium first
    excluded_reason: str = ""

    @property
    def best(self) -> Optional[SafePick]:
        return self.picks[0] if self.picks else None


def pick_low_assignment(
    contracts: list[OptionContract],
    side: str,
    spot: float,
    dte: int,
    view: TechnicalView,
    *,
    max_prob: float = 0.10,
    supports: Optional[list[SupportResistanceLevel]] = None,
    resistances: Optional[list[SupportResistanceLevel]] = None,
    earnings_in_window: bool = False,
    min_bid: float = 0.05,
    max_spread_pct: float = 0.15,
    min_open_interest: int = 0,
    desired_buy_price: Optional[float] = None,
    cost_basis: Optional[float] = None,
    top_n: int = 3,
) -> SidePicks:
    """
    The OTM strikes with the most premium whose estimated probability of
    assignment is at or under *max_prob*, best premium first.
    """
    out = SidePicks(side=side)
    kind = "put" if side == "CSP" else "call"
    eligible: list[SafePick] = []
    for c in contracts:
        if c.option_type != kind:
            continue
        if (kind == "put" and c.strike >= spot) or (kind == "call" and c.strike <= spot):
            continue
        if c.bid < min_bid or c.bid_ask_spread_pct > max_spread_pct:
            continue
        if (c.open_interest or 0) < min_open_interest:
            continue
        if kind == "put" and desired_buy_price and c.strike > desired_buy_price:
            continue
        if kind == "call" and cost_basis and c.strike < cost_basis:
            continue
        est = estimate_assignment(
            c, spot, dte, view, supports=supports, resistances=resistances,
            earnings_in_window=earnings_in_window,
        )
        if est is None or est.prob > max_prob:
            continue
        collateral = c.strike if kind == "put" else spot
        eligible.append(SafePick(
            contract=c, side=side, credit=c.mid,
            yield_pct=c.mid / collateral if collateral > 0 else 0.0,
            estimate=est,
        ))
    eligible.sort(key=lambda p: (p.credit, -p.estimate.prob), reverse=True)
    out.picks = eligible[:top_n]
    if not out.picks:
        out.excluded_reason = (
            f"No liquid OTM {kind} keeps assignment under {max_prob:.0%}"
            + (" at or below your desired buy price" if kind == "put" and desired_buy_price else "")
            + (" above your cost basis" if kind == "call" and cost_basis else "")
            + "."
        )
    return out


@dataclass
class SideVerdict:
    side: Optional[str]        # "CSP" | "CC" | None
    headline: str
    reasons: list[str]


def compare_sides(csp: SidePicks, cc: SidePicks, view: TechnicalView) -> SideVerdict:
    """Which side pays more this week at the same assignment limit — and why."""
    a, b = csp.best, cc.best
    reasons: list[str] = []
    if view.reasons:
        reasons.append(f"Chart is {view.bias}: " + ", ".join(view.reasons[:4]) + ".")
    if a is None and b is None:
        return SideVerdict(None, "Neither side clears your assignment limit this week.", reasons)
    if a is None or b is None:
        side = "CSP" if a is not None else "CC"
        return SideVerdict(
            side,
            f"Only {'cash-secured puts' if side == 'CSP' else 'covered calls'} "
            f"clear your assignment limit this week.",
            reasons,
        )
    reasons.append(
        f"Best CSP ${a.strike:g}P pays {a.yield_pct:.2%} of collateral at "
        f"{a.estimate.prob:.0%} assignment; best CC ${b.strike:g}C pays "
        f"{b.yield_pct:.2%} at {b.estimate.prob:.0%}."
    )
    # Yield is the main decider; the chart breaks near-ties (within 15%).
    hi, lo = (a, b) if a.yield_pct >= b.yield_pct else (b, a)
    if lo.yield_pct > 0 and hi.yield_pct / lo.yield_pct < 1.15:
        side = "CSP" if view.bias == "bullish" else "CC" if view.bias == "bearish" else hi.side
        why = (
            "the uptrend works for puts" if side == "CSP" and view.bias == "bullish"
            else "the downtrend works for calls" if side == "CC" and view.bias == "bearish"
            else "it pays slightly more"
        )
    else:
        side = hi.side
        why = "it pays meaningfully more at the same risk"
    name = "cash-secured puts" if side == "CSP" else "covered calls"
    return SideVerdict(side, f"This week favors {name} — {why}.", reasons)
