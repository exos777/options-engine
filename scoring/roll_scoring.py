"""
Multi-factor scoring engine for CSP/CC roll candidates.

Picks the best Leg 2 strike when the verdict is ROLL — and surfaces
safe / balanced / aggressive alternatives by |delta| bucket.

Best != highest credit. Best = best weighted blend of:
  - net roll credit
  - new effective cost (CSP) / new exit profit (CC)
  - strike improvement / desired-buy alignment
  - delta (assignment / call-away risk)
  - liquidity (OI + spread)
  - technical alignment (S/R, RSI bias)
  - expected-move positioning

Hard filters reject candidates outright; a "rejected" RollScore has
score=0 and a rejection_reason. Soft penalties just lower the score.

All indicator inputs are optional. When omitted, the corresponding
sub-score defaults to neutral (50/100), so the engine stays useful even
without a full chart context wired in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

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
# Weights (must sum to 1.0)
# ---------------------------------------------------------------------------

CSP_WEIGHTS: dict[str, float] = {
    "credit":         0.25,
    "effective_cost": 0.20,
    "strike":         0.15,
    "delta":          0.15,
    "liquidity":      0.10,
    "technical":      0.10,
    "expected_move":  0.05,
}

CC_WEIGHTS: dict[str, float] = {
    "credit":         0.25,
    "exit_profit":    0.20,
    "strike":         0.15,
    "delta":          0.15,
    "liquidity":      0.10,
    "technical":      0.10,
    "expected_move":  0.05,
}

DEFAULT_MIN_OPEN_INTEREST = 50
SAFE_DELTA_MAX = 0.20
AGGRESSIVE_DELTA_MIN = 0.30


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class RollScore:
    """Composite + sub-scores for a single roll candidate."""
    score: float                                       # 0–100
    sub_scores: dict[str, float] = field(default_factory=dict)
    rejected: bool = False
    rejection_reason: str = ""
    explanation: str = ""


@dataclass
class RollPicks:
    """Top picks from the scoring pass."""
    balanced: Optional[RollCandidate] = None
    safe: Optional[RollCandidate] = None
    aggressive: Optional[RollCandidate] = None
    score_by_strike: dict[float, RollScore] = field(default_factory=dict)
    all_scored: list[tuple[RollCandidate, RollScore]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Sub-score helpers
# ---------------------------------------------------------------------------

def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def _credit_score(credit: float) -> float:
    """0 at -$1, 50 at $0, 100 at +$1 of net credit."""
    return _clamp(50.0 + 50.0 * credit)


def _effective_cost_score(improvement: float) -> float:
    """0 at -$5, 50 at $0, 100 at +$5 improvement vs assignment cost."""
    return _clamp(50.0 + 10.0 * improvement)


def _exit_profit_score(profit: float) -> float:
    """0 at -$50, 50 at $0, 100 at +$50 exit profit per share."""
    return _clamp(50.0 + profit)


def _csp_strike_score(candidate: RollCandidate, pos: OpenPosition) -> float:
    """Reward proximity to desired buy price; otherwise reward strike improvement."""
    if pos.desired_buy_price > 0:
        deviation = abs(candidate.strike - pos.desired_buy_price)
        return _clamp(100.0 - deviation * 10.0)
    delta_strike = pos.strike - candidate.strike  # +ve if rolled down
    return _clamp(50.0 + delta_strike * 5.0)


def _cc_strike_score(candidate: RollCandidate, pos: OpenPosition) -> float:
    """Reward higher (rolled-up) strikes for CC."""
    delta_strike = candidate.strike - pos.strike
    return _clamp(50.0 + delta_strike * 5.0)


def _delta_score(delta_abs: float) -> float:
    """Lower |delta| = safer. 0.10→100, 0.30→50, 0.50→0."""
    return _clamp(125.0 - 250.0 * delta_abs)


def _liquidity_score(open_interest: int, spread_pct: float) -> float:
    """Composite of OI depth and spread tightness."""
    oi_score = _clamp(open_interest / 10.0)              # 1000 OI → 100
    spread_score = _clamp(100.0 - spread_pct * 666.7)    # 5% → 67, 15% → 0
    return (oi_score + spread_score) / 2


def _technical_score_csp(
    candidate: RollCandidate,
    indicators: Optional[TechnicalIndicators],
    support_levels: Optional[list[SupportResistanceLevel]],
) -> float:
    score = 50.0
    if support_levels:
        for sl in support_levels:
            if sl.price > 0 and abs(candidate.strike - sl.price) / sl.price <= 0.02:
                score += 25.0 * sl.strength
                break
    if indicators is not None:
        rsi = indicators.rsi_14
        if rsi < 30:
            score += 15  # oversold → bullish for selling puts
        elif rsi < 50:
            score += 5
        elif rsi > 70:
            score -= 10  # overbought → risky to sell more puts
    return _clamp(score)


def _technical_score_cc(
    candidate: RollCandidate,
    indicators: Optional[TechnicalIndicators],
    resistance_levels: Optional[list[SupportResistanceLevel]],
) -> float:
    score = 50.0
    if resistance_levels:
        for rl in resistance_levels:
            if rl.price > 0 and abs(candidate.strike - rl.price) / rl.price <= 0.02:
                score += 25.0 * rl.strength
                break
    if indicators is not None:
        rsi = indicators.rsi_14
        if rsi > 70:
            score += 15  # overbought → bullish for selling calls
        elif rsi > 50:
            score += 5
        elif rsi < 30:
            score -= 10  # oversold → don't sell calls into a bottom
    return _clamp(score)


def _expected_move_score(
    candidate: RollCandidate,
    underlying_price: float,
    expected_move: float,
    strategy: str,
) -> float:
    if expected_move <= 0 or underlying_price <= 0:
        return 50.0
    if strategy == "CSP":
        lower = underlying_price - expected_move
        if candidate.strike <= lower:
            return _clamp(70.0 + (lower - candidate.strike) / expected_move * 30.0)
        inside = (candidate.strike - lower) / expected_move
        return _clamp(70.0 - inside * 50.0)
    upper = underlying_price + expected_move
    if candidate.strike >= upper:
        return _clamp(70.0 + (candidate.strike - upper) / expected_move * 30.0)
    inside = (upper - candidate.strike) / expected_move
    return _clamp(70.0 - inside * 50.0)


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
    """
    Score a single roll candidate. Hard filters short-circuit with
    rejected=True; otherwise returns a 0–100 composite plus sub-scores.
    """
    # ── Hard filters ──
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

    if pos.strategy == "CC" and pos.cost_basis > 0:
        if candidate.strike < pos.cost_basis:
            return RollScore(
                score=0.0, rejected=True,
                rejection_reason=(
                    f"Strike ${candidate.strike:.2f} below cost basis "
                    f"${pos.cost_basis:.2f}"
                ),
            )

    # Negative credit allowed only if strike improvement is meaningful
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

    # ── Sub-scores + soft penalties ──
    if pos.strategy == "CSP":
        weights = CSP_WEIGHTS
        assignment_cost = pos.strike - pos.original_premium
        new_cost = new_effective_cost(pos, candidate)
        improvement = assignment_cost - new_cost
        sub = {
            "credit":         _credit_score(candidate.roll_credit),
            "effective_cost": _effective_cost_score(improvement),
            "strike":         _csp_strike_score(candidate, pos),
            "delta":          _delta_score(candidate.delta),
            "liquidity":      _liquidity_score(
                candidate.open_interest, candidate.spread_pct,
            ),
            "technical":      _technical_score_csp(
                candidate, indicators, support_levels,
            ),
            "expected_move":  _expected_move_score(
                candidate, underlying_price, expected_move, "CSP",
            ),
        }
        # Heavy penalty if effective cost is worse than assignment cost
        if improvement < 0:
            sub["effective_cost"] = max(0.0, sub["effective_cost"] - 30.0)
    else:  # CC
        weights = CC_WEIGHTS
        new_profit = new_exit_profit(pos, candidate)
        sub = {
            "credit":         _credit_score(candidate.roll_credit),
            "exit_profit":    _exit_profit_score(new_profit),
            "strike":         _cc_strike_score(candidate, pos),
            "delta":          _delta_score(candidate.delta),
            "liquidity":      _liquidity_score(
                candidate.open_interest, candidate.spread_pct,
            ),
            "technical":      _technical_score_cc(
                candidate, indicators, resistance_levels,
            ),
            "expected_move":  _expected_move_score(
                candidate, underlying_price, expected_move, "CC",
            ),
        }

    composite = _clamp(sum(weights[k] * sub[k] for k in weights))

    explanation = _build_explanation(candidate, pos, sub)

    return RollScore(
        score=round(composite, 2),
        sub_scores={k: round(v, 1) for k, v in sub.items()},
        rejected=False,
        rejection_reason="",
        explanation=explanation,
    )


# ---------------------------------------------------------------------------
# Explanation
# ---------------------------------------------------------------------------

def _build_explanation(
    candidate: RollCandidate,
    pos: OpenPosition,
    sub: dict[str, float],
) -> str:
    parts: list[str] = []

    # Credit / debit
    if candidate.roll_credit > 0:
        parts.append(f"net credit ${candidate.roll_credit:.2f}")
    elif candidate.roll_credit < 0:
        parts.append(f"net debit ${abs(candidate.roll_credit):.2f}")
    else:
        parts.append("flat credit")

    # Strategy-specific outcome
    if pos.strategy == "CSP":
        new_cost = new_effective_cost(pos, candidate)
        parts.append(f"effective cost ${new_cost:.2f}")
    else:
        new_profit = new_exit_profit(pos, candidate)
        parts.append(f"exit profit ${new_profit:.2f}")

    # Liquidity
    if sub["liquidity"] >= 75:
        parts.append("strong liquidity")
    elif sub["liquidity"] < 40:
        parts.append("thin liquidity")

    # Delta
    if sub["delta"] >= 75:
        parts.append("low assignment risk")
    elif sub["delta"] < 40:
        parts.append("higher assignment risk")

    # Technical / S/R
    if sub["technical"] >= 70:
        parts.append(
            "strike near support" if pos.strategy == "CSP"
            else "strike near resistance"
        )

    # Expected move
    if sub["expected_move"] >= 75:
        parts.append("outside expected move")
    elif sub["expected_move"] < 40:
        parts.append("inside expected move")

    return ", ".join(parts) if parts else "balanced trade-off"


def format_roll_recommendation(
    candidate: RollCandidate,
    pos: OpenPosition,
    score: RollScore,
    bucket: str = "balanced",
) -> str:
    """
    User-facing one-liner like:

        "Roll to $370P exp 2026-05-15 for +$6.85 net credit. This is the
        best balanced roll because it lowers effective cost to $358.15,
        keeps the strike near support, has strong liquidity."
    """
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
    Score every candidate and return safe / balanced / aggressive picks.

      balanced    — highest composite score overall
      safe        — lowest |delta| (farthest from assignment). Prefers
                    candidates with |delta| <= SAFE_DELTA_MAX, but ALWAYS
                    falls back to the lowest-delta scored candidate so the
                    Safe slot is never empty when any roll passes filters.
      aggressive  — highest net credit (closest-to-assignment, max profit).
                    Forced distinct from balanced when an alternative exists.
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

    # Balanced — top composite score.
    balanced = by_score[0][0]

    # Safe — lowest delta. Prefer the strict ≤ SAFE_DELTA_MAX bucket
    # (highest-scored within it); otherwise fall back to the lowest-delta
    # scored candidate so Safe is always populated.
    safe_bucket = [(c, s) for c, s in by_score if c.delta <= SAFE_DELTA_MAX]
    if safe_bucket:
        safe = safe_bucket[0][0]
    else:
        safe = min(scored, key=lambda pair: pair[0].delta)[0]

    # Aggressive — highest net credit (closer to assignment = max profit
    # if it expires worthless). Tiebreak by higher delta, then by score.
    by_credit = sorted(
        scored,
        key=lambda pair: (pair[0].roll_credit, pair[0].delta, pair[1].score),
        reverse=True,
    )
    aggressive = by_credit[0][0]

    # Force Aggressive to differ from Balanced when there is a runner-up.
    if aggressive is balanced and len(by_credit) > 1:
        aggressive = by_credit[1][0]

    return RollPicks(
        balanced=balanced,
        safe=safe,
        aggressive=aggressive,
        score_by_strike=score_by_strike,
        all_scored=by_score,
    )
