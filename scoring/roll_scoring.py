"""
Multi-factor scoring engine for CSP/CC roll candidates.

Picks the best Leg 2 strike using full chart context, not just delta.

Hard filters reject candidates outright:
  - ITM strikes (when spot is known)
  - |delta| >= DELTA_HARD_CAP (0.35)
  - poor liquidity (OI < min)
  - wide spreads (>15%)
  - DTE outside 7–21
  - earnings inside DTE window
  - net debit without sufficient strike improvement
  - CC strike below cost basis

Surviving candidates are scored on a weighted blend of:
  premium (25%) · theta (20%) · distance (15%) · liquidity (15%) ·
  structure alignment (15%) · expected-move positioning (10%)

Delta is then applied as a soft penalty (subtracted from the composite),
so it shapes the score without being a primary driver. The highest
premium does NOT automatically win Balanced.

Indicator inputs are optional. When omitted the corresponding signal
defaults to neutral and the structure sub-score defaults to 50.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from scoring.position_decision import (
    OpenPosition,
    RollCandidate,
    new_effective_cost,
    new_exit_profit,
)
from strategies.models import (
    SupportResistanceLevel,
    TechnicalIndicators,
)


# ---------------------------------------------------------------------------
# Weights (sum to 1.0)
# ---------------------------------------------------------------------------

WEIGHTS: dict[str, float] = {
    "premium":       0.25,
    "theta":         0.20,
    "distance":      0.15,
    "liquidity":     0.15,
    "structure":     0.15,
    "expected_move": 0.10,
}

# Backwards-compat aliases — same shape for both strategies; the
# strategy-specific behaviour now lives in the structure sub-score.
CSP_WEIGHTS = WEIGHTS
CC_WEIGHTS = WEIGHTS

DEFAULT_MIN_OPEN_INTEREST = 50
SAFE_DELTA_MAX = 0.20
DELTA_HARD_CAP = 0.35           # reject |delta| >= this
DELTA_PENALTY_MAX = 25.0        # max points subtracted from composite


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

DirectionBias = Literal["bullish", "neutral", "bearish"]
VolatilityContext = Literal["compressed", "expanding", "neutral"]
Structure = Literal["near_support", "near_resistance", "neutral"]


@dataclass
class Signals:
    """Three condensed indicator signals consumed by the scorer."""
    direction: DirectionBias = "neutral"
    volatility: VolatilityContext = "neutral"
    structure: Structure = "neutral"


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class RollScore:
    """Composite score plus weighted sub-scores and the delta penalty."""
    score: float
    sub_scores: dict[str, float] = field(default_factory=dict)
    rejected: bool = False
    rejection_reason: str = ""
    explanation: str = ""
    delta_penalty: float = 0.0
    signals: Optional[Signals] = None


@dataclass
class RollPicks:
    """Top picks. `aggressive` is kept as a read-only alias for `premium`."""
    balanced: Optional[RollCandidate] = None
    safe: Optional[RollCandidate] = None
    premium: Optional[RollCandidate] = None
    score_by_strike: dict[float, RollScore] = field(default_factory=dict)
    all_scored: list[tuple[RollCandidate, RollScore]] = field(default_factory=list)

    @property
    def aggressive(self) -> Optional[RollCandidate]:
        return self.premium


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


# ── Sub-score helpers ───────────────────────────────────────────────────

def _premium_score(credit: float) -> float:
    """Higher net credit = better. 0 at $0, 100 at +$2."""
    return _clamp(50.0 * credit)


def _theta_score(candidate: RollCandidate) -> float:
    """
    Daily decay proxy = mid / dte (in $/day). 0 at 0, 100 at $0.30/day.
    Approximates raw theta when an explicit theta value isn't on the
    candidate.
    """
    if candidate.dte <= 0:
        return 0.0
    daily = candidate.mid / candidate.dte
    return _clamp(daily / 0.003)


def _distance_score(candidate: RollCandidate, underlying: float) -> float:
    """
    OTM distance as % of spot. 0 at 0%, 100 at 8% OTM.
    Premium/theta sub-scores fight back if the strike is too far OTM.
    """
    if underlying <= 0:
        return 50.0
    pct = abs(candidate.strike - underlying) / underlying
    return _clamp(pct * 1250.0)


def _liquidity_score(open_interest: int, spread_pct: float) -> float:
    oi_score = _clamp(open_interest / 10.0)
    spread_score = _clamp(100.0 - spread_pct * 666.7)
    return (oi_score + spread_score) / 2


def _expected_move_score(
    candidate: RollCandidate,
    underlying: float,
    expected_move: float,
    strategy: str,
) -> float:
    if expected_move <= 0 or underlying <= 0:
        return 50.0
    if strategy == "CSP":
        lower = underlying - expected_move
        if candidate.strike <= lower:
            return _clamp(70.0 + (lower - candidate.strike) / expected_move * 30.0)
        inside = (candidate.strike - lower) / expected_move
        return _clamp(70.0 - inside * 50.0)
    upper = underlying + expected_move
    if candidate.strike >= upper:
        return _clamp(70.0 + (candidate.strike - upper) / expected_move * 30.0)
    inside = (upper - candidate.strike) / expected_move
    return _clamp(70.0 - inside * 50.0)


# ── Signal derivation ───────────────────────────────────────────────────

def _direction_bias(
    indicators: Optional[TechnicalIndicators],
) -> DirectionBias:
    if indicators is None:
        return "neutral"
    price = indicators.current_price
    sma20 = indicators.sma_20
    sma50 = indicators.sma_50
    rsi = indicators.rsi_14

    bull_flags = sum([price > sma20, sma20 > sma50, rsi > 55])
    bear_flags = sum([price < sma20, sma20 < sma50, rsi < 45])

    if bull_flags >= 2 and bull_flags > bear_flags:
        return "bullish"
    if bear_flags >= 2 and bear_flags > bull_flags:
        return "bearish"
    return "neutral"


def _volatility_context(
    indicators: Optional[TechnicalIndicators],
) -> VolatilityContext:
    if indicators is None:
        return "neutral"
    if indicators.squeeze_on:
        return "compressed"
    if indicators.adx_14 >= 25:
        return "expanding"
    return "neutral"


def _chart_structure(
    underlying: float,
    support_levels: Optional[list[SupportResistanceLevel]],
    resistance_levels: Optional[list[SupportResistanceLevel]],
) -> Structure:
    """Where is the underlying price sitting relative to key S/R?"""
    if underlying <= 0:
        return "neutral"
    for rl in resistance_levels or []:
        if rl.price > 0 and abs(underlying - rl.price) / rl.price <= 0.025:
            return "near_resistance"
    for sl in support_levels or []:
        if sl.price > 0 and abs(underlying - sl.price) / sl.price <= 0.025:
            return "near_support"
    return "neutral"


def derive_signals(
    indicators: Optional[TechnicalIndicators],
    underlying_price: float,
    support_levels: Optional[list[SupportResistanceLevel]],
    resistance_levels: Optional[list[SupportResistanceLevel]],
) -> Signals:
    """Public helper — exposed for testing and explanations."""
    return Signals(
        direction=_direction_bias(indicators),
        volatility=_volatility_context(indicators),
        structure=_chart_structure(
            underlying_price, support_levels, resistance_levels,
        ),
    )


def _strike_at_level(
    candidate: RollCandidate,
    levels: Optional[list[SupportResistanceLevel]],
) -> Optional[SupportResistanceLevel]:
    """Returns the matching level if candidate strike sits within 2%."""
    for lv in levels or []:
        if lv.price > 0 and abs(candidate.strike - lv.price) / lv.price <= 0.02:
            return lv
    return None


# ── Structure-alignment score ───────────────────────────────────────────

def _structure_score(
    candidate: RollCandidate,
    pos: OpenPosition,
    underlying: float,
    signals: Signals,
    support_levels: Optional[list[SupportResistanceLevel]],
    resistance_levels: Optional[list[SupportResistanceLevel]],
) -> float:
    """
    Combines the three signals into a strike-fit score.

    CSP (selling puts):
      bullish        → modest OTM (~2–4%) is the sweet spot
      bearish        → push farther OTM (>=5%)
      near_support   → reward strikes that align with a support level
      compressed vol → favour closer-in strikes
      expanding vol  → favour farther-OTM strikes

    CC (selling calls):
      bullish        → push farther OTM to preserve upside
      bearish        → closer OTM to maximize income
      near_resistance → closer OTM (call-away likely)
      compressed vol → closer-in
      expanding vol  → farther-OTM
    """
    score = 50.0
    if underlying <= 0:
        # No spot — fall back to a level-alignment nudge so the score
        # still reflects whatever S/R info is available.
        if pos.strategy == "CSP":
            level = _strike_at_level(candidate, support_levels)
            if level is not None:
                score += 20.0 * level.strength
        else:
            level = _strike_at_level(candidate, resistance_levels)
            if level is not None:
                score -= 10.0 * level.strength
        return _clamp(score)

    otm_pct = abs(candidate.strike - underlying) / underlying

    if pos.strategy == "CSP":
        level = _strike_at_level(candidate, support_levels)
        if level is not None:
            score += 20.0 * max(level.strength, 0.5)
        if signals.structure == "near_support":
            # Selling puts at chart support is high-quality income.
            score += 8.0
        if signals.direction == "bullish":
            # Sweet-spot ~2.5% OTM; gentler penalty far away.
            score += _clamp(20.0 - abs(otm_pct - 0.025) * 400.0, lo=-10.0, hi=20.0)
        elif signals.direction == "bearish":
            # Reward only when comfortably far OTM.
            score += _clamp((otm_pct - 0.05) * 400.0, lo=-15.0, hi=20.0)
        if signals.volatility == "expanding":
            score += _clamp((otm_pct - 0.03) * 200.0, lo=-5.0, hi=10.0)
        elif signals.volatility == "compressed":
            score += _clamp((0.03 - otm_pct) * 200.0, lo=-5.0, hi=10.0)
    else:
        level = _strike_at_level(candidate, resistance_levels)
        if level is not None:
            # Sitting AT resistance is a coin-flip for a CC: high income
            # but high call-away risk. Mild positive nudge so it isn't
            # double-penalized — the chart-context branch below adjusts
            # further when price itself is near resistance.
            score += 5.0 * level.strength
        if signals.direction == "bullish":
            score += _clamp((otm_pct - 0.04) * 400.0, lo=-15.0, hi=20.0)
        elif (
            signals.direction == "bearish"
            or signals.structure == "near_resistance"
        ):
            score += _clamp(20.0 - abs(otm_pct - 0.025) * 400.0, lo=-10.0, hi=20.0)
        if signals.volatility == "expanding":
            score += _clamp((otm_pct - 0.03) * 200.0, lo=-5.0, hi=10.0)
        elif signals.volatility == "compressed":
            score += _clamp((0.03 - otm_pct) * 200.0, lo=-5.0, hi=10.0)

    return _clamp(score)


# ── Delta penalty ───────────────────────────────────────────────────────

def _delta_penalty(delta_abs: float) -> float:
    """
    Quadratic penalty subtracted from composite.
      0.10 → 0     0.20 → 4      0.30 → 16     0.35 → 25 (cap)
    """
    if delta_abs <= 0.10:
        return 0.0
    excess = delta_abs - 0.10
    return _clamp(
        (excess / 0.25) ** 2 * DELTA_PENALTY_MAX,
        lo=0.0, hi=DELTA_PENALTY_MAX,
    )


# ---------------------------------------------------------------------------
# Scoring entry point
# ---------------------------------------------------------------------------

def score_roll_candidate(
    candidate: RollCandidate,
    pos: OpenPosition,
    *,
    indicators: Optional[TechnicalIndicators] = None,
    support_levels: Optional[list[SupportResistanceLevel]] = None,
    resistance_levels: Optional[list[SupportResistanceLevel]] = None,
    expected_move: float = 0.0,
    underlying_price: float = 0.0,
    has_earnings: bool = False,
    min_open_interest: int = DEFAULT_MIN_OPEN_INTEREST,
) -> RollScore:
    """Score a single roll candidate. Hard filters short-circuit with rejected=True."""
    delta_abs = abs(candidate.delta)

    # ── Hard filters ────────────────────────────────────────────────
    if candidate.spread_pct > 0.15:
        return RollScore(
            score=0.0, rejected=True,
            rejection_reason=f"Spread {candidate.spread_pct * 100:.1f}% > 15%",
        )
    if candidate.open_interest < min_open_interest:
        return RollScore(
            score=0.0, rejected=True,
            rejection_reason=(
                f"OI {candidate.open_interest} < min {min_open_interest}"
            ),
        )
    if not (7 <= candidate.dte <= 21):
        return RollScore(
            score=0.0, rejected=True,
            rejection_reason=f"DTE {candidate.dte} outside 7–21 window",
        )
    if has_earnings:
        return RollScore(
            score=0.0, rejected=True,
            rejection_reason="Earnings inside DTE window",
        )
    if delta_abs >= DELTA_HARD_CAP:
        return RollScore(
            score=0.0, rejected=True,
            rejection_reason=(
                f"Delta {delta_abs:.2f} ≥ {DELTA_HARD_CAP:.2f} hard cap"
            ),
        )
    if underlying_price > 0:
        if pos.strategy == "CC" and candidate.strike < underlying_price:
            return RollScore(
                score=0.0, rejected=True,
                rejection_reason=(
                    f"ITM call: strike ${candidate.strike:.2f} < "
                    f"spot ${underlying_price:.2f}"
                ),
            )
        if pos.strategy == "CSP" and candidate.strike > underlying_price:
            return RollScore(
                score=0.0, rejected=True,
                rejection_reason=(
                    f"ITM put: strike ${candidate.strike:.2f} > "
                    f"spot ${underlying_price:.2f}"
                ),
            )
    if pos.strategy == "CC" and pos.cost_basis > 0:
        if candidate.strike < pos.cost_basis:
            return RollScore(
                score=0.0, rejected=True,
                rejection_reason=(
                    f"Strike ${candidate.strike:.2f} below cost basis "
                    f"${pos.cost_basis:.2f}"
                ),
            )
    if candidate.roll_credit < 0:
        if pos.strategy == "CSP":
            big_improve = candidate.strike <= pos.strike * 0.98
        else:
            big_improve = candidate.strike >= pos.strike * 1.02
        if not (big_improve and abs(candidate.roll_credit) < 0.25):
            return RollScore(
                score=0.0, rejected=True,
                rejection_reason=(
                    "Net debit without sufficient strike improvement"
                ),
            )

    # ── Signals ─────────────────────────────────────────────────────
    signals = derive_signals(
        indicators, underlying_price, support_levels, resistance_levels,
    )

    # ── Sub-scores ──────────────────────────────────────────────────
    sub = {
        "premium":       _premium_score(candidate.roll_credit),
        "theta":         _theta_score(candidate),
        "distance":      _distance_score(candidate, underlying_price),
        "liquidity":     _liquidity_score(
            candidate.open_interest, candidate.spread_pct,
        ),
        "structure":     _structure_score(
            candidate, pos, underlying_price, signals,
            support_levels, resistance_levels,
        ),
        "expected_move": _expected_move_score(
            candidate, underlying_price, expected_move, pos.strategy,
        ),
    }

    composite = _clamp(sum(WEIGHTS[k] * sub[k] for k in WEIGHTS))

    # Soft delta penalty applied to the composite, not to a sub-score weight.
    delta_pen = _delta_penalty(delta_abs)
    composite = _clamp(composite - delta_pen)

    explanation = _build_explanation(candidate, pos, sub, signals, delta_pen)

    return RollScore(
        score=round(composite, 2),
        sub_scores={k: round(v, 1) for k, v in sub.items()},
        rejected=False,
        rejection_reason="",
        explanation=explanation,
        delta_penalty=round(delta_pen, 2),
        signals=signals,
    )


# ---------------------------------------------------------------------------
# Explanation
# ---------------------------------------------------------------------------

def _build_explanation(
    candidate: RollCandidate,
    pos: OpenPosition,
    sub: dict[str, float],
    signals: Signals,
    delta_pen: float,
) -> str:
    parts: list[str] = []

    if candidate.roll_credit > 0:
        parts.append(f"net credit ${candidate.roll_credit:.2f}")
    elif candidate.roll_credit < 0:
        parts.append(f"net debit ${abs(candidate.roll_credit):.2f}")
    else:
        parts.append("flat credit")

    if pos.strategy == "CSP":
        new_cost = new_effective_cost(pos, candidate)
        parts.append(f"effective cost ${new_cost:.2f}")
    else:
        new_profit = new_exit_profit(pos, candidate)
        parts.append(f"exit profit ${new_profit:.2f}")

    if sub["theta"] >= 70:
        parts.append("strong theta decay")
    elif sub["theta"] < 30:
        parts.append("thin theta decay")

    if sub["liquidity"] >= 75:
        parts.append("strong liquidity")
    elif sub["liquidity"] < 40:
        parts.append("thin liquidity")

    if signals.structure == "near_support" and pos.strategy == "CSP":
        parts.append("price at support")
    elif signals.structure == "near_resistance" and pos.strategy == "CC":
        parts.append("price at resistance")

    if signals.direction == "bullish":
        parts.append("bullish bias")
    elif signals.direction == "bearish":
        parts.append("bearish bias")

    if signals.volatility == "compressed":
        parts.append("vol compressed")
    elif signals.volatility == "expanding":
        parts.append("vol expanding")

    if sub["expected_move"] >= 75:
        parts.append("outside expected move")
    elif sub["expected_move"] < 40:
        parts.append("inside expected move")

    if delta_pen >= 12:
        parts.append(f"delta penalty -{delta_pen:.0f}")

    return ", ".join(parts) if parts else "balanced trade-off"


def format_roll_recommendation(
    candidate: RollCandidate,
    pos: OpenPosition,
    score: RollScore,
    bucket: str = "balanced",
) -> str:
    """User-facing one-liner."""
    side = "P" if pos.strategy == "CSP" else "C"
    sign = "+" if candidate.roll_credit >= 0 else "-"
    credit_label = "net credit" if candidate.roll_credit >= 0 else "net debit"
    return (
        f"Roll to ${candidate.strike:.2f}{side} exp {candidate.expiration} "
        f"for {sign}${abs(candidate.roll_credit):.2f} {credit_label}. "
        f"This is the best {bucket} roll because {score.explanation}."
    )


# ---------------------------------------------------------------------------
# Picker
# ---------------------------------------------------------------------------

def pick_top_rolls(
    candidates: list[RollCandidate],
    pos: OpenPosition,
    **kwargs,
) -> RollPicks:
    """
    Score every candidate and return Safe / Balanced / Premium picks.

      balanced — highest composite score overall (highest premium does
                 NOT auto-win — it has to clear the weighted blend with
                 acceptable delta penalty).
      safe     — lowest |delta|, preferring |delta| <= SAFE_DELTA_MAX.
                 Falls back to the absolute lowest-delta scored candidate
                 so Safe is never empty when any roll passes filters.
      premium  — highest net credit among non-ITM scored candidates.
                 Forced distinct from balanced when an alternative exists.
                 All hard filters (incl. |delta| < 0.35) still apply, so
                 this is "higher credit but still acceptable."
    """
    if not candidates:
        return RollPicks()

    scored: list[tuple[RollCandidate, RollScore]] = []
    for c in candidates:
        s = score_roll_candidate(c, pos, **kwargs)
        if not s.rejected:
            scored.append((c, s))

    if not scored:
        return RollPicks()

    by_score = sorted(scored, key=lambda pair: pair[1].score, reverse=True)
    score_by_strike = {c.strike: s for c, s in by_score}

    balanced = by_score[0][0]

    safe_bucket = [
        (c, s) for c, s in by_score if abs(c.delta) <= SAFE_DELTA_MAX
    ]
    if safe_bucket:
        safe = safe_bucket[0][0]
    else:
        safe = min(scored, key=lambda pair: abs(pair[0].delta))[0]

    underlying_price = float(kwargs.get("underlying_price", 0.0) or 0.0)
    if underlying_price > 0:
        if pos.strategy == "CC":
            otm = [(c, s) for c, s in scored if c.strike >= underlying_price]
        else:
            otm = [(c, s) for c, s in scored if c.strike <= underlying_price]
    else:
        otm = list(scored)

    pool = otm if otm else list(scored)
    by_credit = sorted(
        pool,
        key=lambda pair: (
            pair[0].roll_credit, pair[1].score, -abs(pair[0].delta),
        ),
        reverse=True,
    )
    premium = by_credit[0][0]

    if premium is balanced and len(by_credit) > 1:
        premium = by_credit[1][0]

    return RollPicks(
        balanced=balanced,
        safe=safe,
        premium=premium,
        score_by_strike=score_by_strike,
        all_scored=by_score,
    )
