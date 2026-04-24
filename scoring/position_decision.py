"""
Roll vs Assign decision engine.

Given an open position (CSP or CC), evaluates four possible actions:
  - Assign: accept assignment / let shares be called away
  - Roll: roll the option to a new strike/expiration
  - Close: buy back the option for a profit/loss
  - Wait: hold and let theta work

Each action is scored 0-100. The engine picks the best one and provides
sizing, explanation, and next-step guidance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from strategies.models import ChartRegime, RegimeResult, SupportResistanceLevel


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class OpenPosition:
    strategy: str  # "CSP" or "CC"
    strike: float
    original_premium: float
    current_bid: float
    current_ask: float
    close_price_mode: str = "realistic"
    times_rolled: int = 0
    total_premium_collected: float = 0.0
    desired_buy_price: float = 0.0
    cost_basis: float = 0.0

    @property
    def current_mid(self) -> float:
        return (self.current_bid + self.current_ask) / 2

    @property
    def close_cost(self) -> float:
        if self.close_price_mode == "conservative":
            return self.current_ask
        elif self.close_price_mode == "optimistic":
            return (self.current_bid + self.current_mid) / 2
        else:
            return self.current_mid

    @property
    def unrealized_pl(self) -> float:
        return self.original_premium - self.close_cost

    @property
    def net_assigned_cost(self) -> float:
        return self.strike - self.total_premium_collected


@dataclass
class RollCandidate:
    strike: float
    expiration: str
    dte: int
    bid: float
    ask: float
    mid: float
    delta: float
    open_interest: int
    roll_credit: float
    roll_type: str
    spread_pct: float = 0.0


@dataclass
class PositionDecision:
    recommendation: str
    confidence: float
    scores: dict
    explanation: str
    next_wheel_step: str
    roll_candidates: list[RollCandidate]
    warnings: list[str] = field(default_factory=list)
    position_size: str = "No New Exposure"
    position_size_reason: str = ""


# ---------------------------------------------------------------------------
# Expected move
# ---------------------------------------------------------------------------

def _calc_expected_move(
    current_price: float,
    implied_volatility: float,
    dte: int,
) -> float:
    if implied_volatility <= 0 or dte <= 0:
        return 0.0
    return current_price * implied_volatility * math.sqrt(dte / 365)


def _expected_move_score_csp(
    strike: float,
    current_price: float,
    expected_move: float,
    wants_assignment: bool,
) -> tuple[float, str]:
    if expected_move <= 0:
        return 0.0, "No IV data for expected move"

    lower_em = current_price - expected_move
    distance_beyond = lower_em - strike

    if distance_beyond > 0:
        pct_beyond = distance_beyond / expected_move
        score = min(20.0, pct_beyond * 40)
        return score, (
            f"strike ${strike:.2f} is outside "
            f"expected move (boundary ${lower_em:.2f})"
        )
    else:
        pct_inside = abs(distance_beyond) / expected_move
        if wants_assignment:
            return 5.0, "inside expected move — assignment more likely"
        else:
            penalty = min(20.0, pct_inside * 30)
            return -penalty, "strike inside expected move — elevated assignment risk"


def _expected_move_score_cc(
    strike: float,
    current_price: float,
    expected_move: float,
    wants_to_keep_shares: bool,
) -> tuple[float, str]:
    if expected_move <= 0:
        return 0.0, "No IV data for expected move"

    upper_em = current_price + expected_move
    distance_beyond = strike - upper_em

    if distance_beyond > 0:
        pct_beyond = distance_beyond / expected_move
        score = min(20.0, pct_beyond * 40)
        return score, (
            f"strike above expected move "
            f"(boundary ${upper_em:.2f})"
        )
    else:
        pct_inside = abs(distance_beyond) / expected_move
        if wants_to_keep_shares:
            penalty = min(20.0, pct_inside * 30)
            return -penalty, "inside expected move — call-away risk elevated"
        else:
            return 5.0, "inside expected move — call-away more likely"


# ---------------------------------------------------------------------------
# Wait scoring
# ---------------------------------------------------------------------------

def score_wait(
    pos: OpenPosition,
    current_price: float,
    dte_remaining: int,
) -> tuple[float, str]:
    score = 30.0
    reasons: list[str] = []

    if pos.strategy == "CSP":
        is_otm = current_price > pos.strike
    else:
        is_otm = current_price < pos.strike

    if is_otm:
        score += 25
        reasons.append("option is OTM — theta working for you")

    if dte_remaining > 5:
        score += 20
        reasons.append(f"{dte_remaining} DTE remaining — plenty of time for decay")
    elif dte_remaining > 2:
        score += 10
        reasons.append(f"{dte_remaining} DTE remaining — time decay still working")

    premium_captured_pct = (
        pos.unrealized_pl / pos.original_premium
        if pos.original_premium > 0 else 0
    )
    if premium_captured_pct < 0.50:
        score += 20
        reasons.append(
            f"only {premium_captured_pct * 100:.0f}% of "
            "premium captured — let theta work"
        )
    elif premium_captured_pct >= 0.50:
        penalty = 10 + (premium_captured_pct - 0.50) * 100
        score -= penalty
        reasons.append(
            f"{premium_captured_pct * 100:.0f}% captured — "
            "consider closing for profit"
        )

    if pos.times_rolled == 0:
        score += 5

    score = max(0.0, min(100.0, score))
    explanation = (
        "Waiting is optimal: " + ", ".join(reasons)
        if reasons else
        "No urgent action needed."
    )
    return score, explanation


# ---------------------------------------------------------------------------
# Assign scoring
# ---------------------------------------------------------------------------

def _score_assign_csp(
    pos: OpenPosition,
    current_price: float,
    support_levels: list[SupportResistanceLevel],
    wants_assignment: bool,
) -> tuple[float, str]:
    score = 40.0
    reasons: list[str] = []

    if not wants_assignment:
        score -= 20
        reasons.append("not wanting assignment")

    discount_pct = (
        (current_price - pos.net_assigned_cost) / current_price
        if current_price > 0 else 0
    )
    if discount_pct >= 0.05:
        score += 20
        reasons.append(
            f"net cost ${pos.net_assigned_cost:.2f} is "
            f"{discount_pct * 100:.1f}% below current price"
        )
    elif discount_pct >= 0.02:
        score += 10
        reasons.append(f"modest discount at ${pos.net_assigned_cost:.2f}")
    elif discount_pct < 0:
        score -= 15
        reasons.append("net cost above current price")

    if pos.desired_buy_price > 0:
        if pos.net_assigned_cost <= pos.desired_buy_price:
            score += 20
            reasons.append(
                f"net cost at/below desired buy ${pos.desired_buy_price:.2f}"
            )
        elif pos.net_assigned_cost <= pos.desired_buy_price * 1.02:
            score += 10
            reasons.append("net cost near desired buy price")

    for s in support_levels:
        if abs(pos.strike - s.price) / max(s.price, 0.01) <= 0.02:
            score += 10 * s.strength
            reasons.append(f"strike near support ${s.price:.2f}")
            break

    score = max(0.0, min(100.0, score))
    explanation = "; ".join(reasons) if reasons else "Standard assignment evaluation"
    return score, explanation


def _score_assign_cc(
    pos: OpenPosition,
    current_price: float,
    resistance_levels: list[SupportResistanceLevel],
) -> tuple[float, str]:
    score = 40.0
    reasons: list[str] = []

    if pos.cost_basis > 0:
        profit_margin = (pos.strike - pos.cost_basis) / pos.cost_basis
        if profit_margin >= 0.05:
            score += 25
            reasons.append(
                f"profitable exit: {profit_margin * 100:.1f}% above cost basis"
            )
        elif profit_margin >= 0:
            score += 15
            reasons.append("exit at/above cost basis")
        else:
            score -= 25
            reasons.append("assignment below cost basis")

    if pos.total_premium_collected > 0 and pos.cost_basis > 0:
        total_return = (
            (pos.strike - pos.cost_basis + pos.total_premium_collected)
            / pos.cost_basis
        )
        if total_return >= 0.10:
            score += 15
            reasons.append(f"total return {total_return * 100:.1f}%")

    score = max(0.0, min(100.0, score))
    explanation = "; ".join(reasons) if reasons else "Standard call-away evaluation"
    return score, explanation


# ---------------------------------------------------------------------------
# Close scoring
# ---------------------------------------------------------------------------

def _score_close(pos: OpenPosition) -> tuple[float, str]:
    score = 30.0
    reasons: list[str] = []

    if pos.original_premium > 0:
        captured_pct = pos.unrealized_pl / pos.original_premium
        if captured_pct >= 0.80:
            score += 45
            reasons.append(
                f"{captured_pct * 100:.0f}% premium captured — excellent exit"
            )
        elif captured_pct >= 0.50:
            score += 30
            reasons.append(f"{captured_pct * 100:.0f}% captured — good profit")
        elif captured_pct >= 0.25:
            score += 15
            reasons.append(f"{captured_pct * 100:.0f}% captured — modest profit")
        elif captured_pct < 0:
            score -= 15
            reasons.append("position at a loss")

    score = max(0.0, min(100.0, score))
    explanation = "; ".join(reasons) if reasons else "Close evaluation"
    return score, explanation


# ---------------------------------------------------------------------------
# Roll merit checks
# ---------------------------------------------------------------------------

def _csp_roll_has_merit(
    pos: OpenPosition,
    candidate: RollCandidate,
) -> tuple[bool, str]:
    reasons: list[str] = []

    if candidate.strike < pos.strike and candidate.roll_credit > 0:
        reasons.append(
            f"lowers strike from ${pos.strike:.2f} "
            f"to ${candidate.strike:.2f} with "
            f"${candidate.roll_credit:.2f} credit"
        )

    new_net = candidate.strike - (
        pos.total_premium_collected + candidate.roll_credit
    )
    if new_net < pos.net_assigned_cost:
        reasons.append(
            f"improves net assigned cost from "
            f"${pos.net_assigned_cost:.2f} to "
            f"${new_net:.2f}"
        )

    if pos.desired_buy_price > 0:
        current_distance = abs(pos.strike - pos.desired_buy_price)
        new_distance = abs(candidate.strike - pos.desired_buy_price)
        if new_distance < current_distance:
            reasons.append(
                f"moves strike closer to desired "
                f"buy price ${pos.desired_buy_price:.2f}"
            )

    if candidate.delta < 0.25 and candidate.dte <= pos.times_rolled * 7 + 14:
        reasons.append(
            f"reduces delta to {candidate.delta:.2f} "
            "within acceptable DTE"
        )

    has_merit = len(reasons) > 0
    return has_merit, (
        "; ".join(reasons) if reasons
        else "roll does not improve position"
    )


def _cc_roll_has_merit(
    pos: OpenPosition,
    candidate: RollCandidate,
    regime: RegimeResult,
) -> tuple[bool, str]:
    reasons: list[str] = []

    if candidate.strike < pos.cost_basis:
        return False, "new strike below cost basis — not allowed"

    if candidate.strike > pos.strike:
        reasons.append(
            f"raises exit price from "
            f"${pos.strike:.2f} to ${candidate.strike:.2f}"
        )

    if candidate.roll_credit >= 0.30:
        reasons.append(f"preserves ${candidate.roll_credit:.2f} credit")

    if regime.primary == ChartRegime.BULLISH:
        if candidate.strike > pos.strike:
            reasons.append("bullish regime — rolling up protects upside")

    if candidate.strike >= pos.cost_basis * 1.02:
        reasons.append(
            f"new strike ${candidate.strike:.2f} "
            f"above cost basis ${pos.cost_basis:.2f}"
        )

    has_merit = len(reasons) > 0
    return has_merit, (
        "; ".join(reasons) if reasons
        else "roll does not improve CC position"
    )


# ---------------------------------------------------------------------------
# Roll scoring
# ---------------------------------------------------------------------------

def _score_csp_roll(
    pos: OpenPosition,
    candidate: RollCandidate,
) -> tuple[float, str]:
    has_merit, merit_reason = _csp_roll_has_merit(pos, candidate)
    if not has_merit:
        return 0.0, f"Roll rejected: {merit_reason}"

    score = 40.0
    reasons = [merit_reason]

    if candidate.roll_credit > 0:
        score += min(20, candidate.roll_credit * 10)
    if candidate.strike < pos.strike:
        score += 10
    if 0.15 <= candidate.delta <= 0.30:
        score += 10
    if 7 <= candidate.dte <= 14:
        score += 5

    score = max(0.0, min(100.0, score))
    return score, "; ".join(reasons)


def _score_cc_roll(
    pos: OpenPosition,
    candidate: RollCandidate,
    regime: RegimeResult,
) -> tuple[float, str]:
    has_merit, merit_reason = _cc_roll_has_merit(pos, candidate, regime)
    if not has_merit:
        return 0.0, f"Roll rejected: {merit_reason}"

    score = 40.0
    reasons = [merit_reason]

    if candidate.roll_credit > 0:
        score += min(20, candidate.roll_credit * 10)
    if candidate.strike > pos.strike:
        score += 10
    if pos.cost_basis > 0 and candidate.strike >= pos.cost_basis * 1.02:
        score += 10

    score = max(0.0, min(100.0, score))
    return score, "; ".join(reasons)


# ---------------------------------------------------------------------------
# Assignment preference
# ---------------------------------------------------------------------------

def _prefer_assignment_over_roll(
    pos: OpenPosition,
    assign_score: float,
    roll_score: float,
    wants_assignment: bool,
) -> tuple[float, float, str]:
    note = ""

    if (
        wants_assignment
        and assign_score >= 55
        and roll_score > assign_score
    ):
        if roll_score - assign_score < 10:
            assign_score = min(100, assign_score + 10)
            roll_score = max(0, roll_score - 10)
            note = (
                "Assignment preferred over marginal roll — "
                "stock is wheel-worthy and cost is acceptable"
            )

    return assign_score, roll_score, note


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------

def _position_size_guidance(
    regime: RegimeResult,
    has_earnings: bool,
    dte_remaining: int,
    times_rolled: int,
    confidence: float,
) -> tuple[str, str]:
    score = confidence

    if regime.primary == ChartRegime.BULLISH:
        score += 10
    elif regime.primary == ChartRegime.NEAR_SUPPORT:
        score += 5
    elif regime.primary == ChartRegime.BEARISH:
        score -= 20
    elif regime.primary == ChartRegime.OVEREXTENDED:
        score -= 10

    if has_earnings:
        score -= 25
    if dte_remaining < 5:
        score -= 15
    elif dte_remaining > 21:
        score -= 10
    if times_rolled >= 2:
        score -= 20
    elif times_rolled == 1:
        score -= 5

    score = max(0.0, min(100.0, score))

    reasons: list[str] = []
    if has_earnings:
        reasons.append("earnings risk")
    if times_rolled >= 2:
        reasons.append(f"rolled {times_rolled}x")
    if regime.primary == ChartRegime.BEARISH:
        reasons.append("bearish regime")
    reason_str = ", ".join(reasons) if reasons else "conditions favorable"

    if score >= 75:
        return "✅ Full Size", f"Strong setup: {reason_str}"
    elif score >= 60:
        return "⚡ Half Size", f"Good setup, some risk: {reason_str}"
    elif score >= 45:
        return "⚠️ Quarter Size", f"Weak setup: {reason_str}"
    else:
        return "⛔ No New Exposure", f"Poor conditions: {reason_str}"


# ---------------------------------------------------------------------------
# Roll candidate filtering
# ---------------------------------------------------------------------------

def _build_roll_candidates_filtered(
    candidates: list[RollCandidate],
) -> list[RollCandidate]:
    filtered: list[RollCandidate] = []
    for c in candidates:
        spread_pct = (c.ask - c.bid) / max(c.mid, 0.01)
        c.spread_pct = spread_pct
        if spread_pct > 0.15:
            continue
        filtered.append(c)
    return filtered


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_position(
    pos: OpenPosition,
    current_price: float,
    regime: RegimeResult,
    support_levels: list[SupportResistanceLevel],
    resistance_levels: list[SupportResistanceLevel],
    roll_candidates: list[RollCandidate],
    next_cc_premium: float,
    has_earnings: bool,
    implied_volatility: float = 0.0,
    dte_remaining: int = 0,
    wants_assignment: bool = True,
    wants_to_keep_shares: bool = False,
) -> PositionDecision:
    warnings: list[str] = []

    filtered_rolls = _build_roll_candidates_filtered(roll_candidates)

    expected_move = _calc_expected_move(
        current_price, implied_volatility, dte_remaining
    )

    # --- Assign ---
    if pos.strategy == "CSP":
        assign_score, assign_reason = _score_assign_csp(
            pos, current_price, support_levels, wants_assignment
        )
    else:
        assign_score, assign_reason = _score_assign_cc(
            pos, current_price, resistance_levels
        )

    if pos.strategy == "CSP" and expected_move > 0:
        em_s, em_r = _expected_move_score_csp(
            pos.strike, current_price, expected_move, wants_assignment
        )
        assign_score = max(0.0, min(100.0, assign_score + em_s))
        if em_r:
            assign_reason += f"; {em_r}"
    elif pos.strategy == "CC" and expected_move > 0:
        em_s, em_r = _expected_move_score_cc(
            pos.strike, current_price, expected_move, wants_to_keep_shares
        )
        assign_score = max(0.0, min(100.0, assign_score + em_s))
        if em_r:
            assign_reason += f"; {em_r}"

    # --- Close ---
    close_score, close_reason = _score_close(pos)

    # --- Best roll ---
    best_roll_score = 0.0
    best_roll_reason = "No qualifying roll candidates"
    best_roll_candidate: Optional[RollCandidate] = None

    for c in filtered_rolls:
        if pos.strategy == "CSP":
            rs, rr = _score_csp_roll(pos, c)
        else:
            rs, rr = _score_cc_roll(pos, c, regime)
        if rs > best_roll_score:
            best_roll_score = rs
            best_roll_reason = rr
            best_roll_candidate = c

    # --- Wait ---
    wait_score, wait_reason = score_wait(pos, current_price, dte_remaining)

    # --- Prefer assignment over marginal roll (CSP) ---
    if pos.strategy == "CSP":
        assign_score, best_roll_score, pref_note = (
            _prefer_assignment_over_roll(
                pos, assign_score, best_roll_score, wants_assignment
            )
        )
        if pref_note:
            warnings.append(f"ℹ️ {pref_note}")

    # --- Scores ---
    scores = {
        "assign": assign_score,
        "roll": best_roll_score,
        "close": close_score,
        "wait": wait_score,
    }

    best_action = max(scores, key=lambda k: scores[k])
    confidence = scores[best_action]

    decision_labels = {
        "assign": (
            "\U0001f4cc Accept Assignment"
            if pos.strategy == "CSP"
            else "\U0001f4cc Let Shares Be Called Away"
        ),
        "roll": "\U0001f504 Roll Position",
        "close": "\U0001f4b0 Close for Profit",
        "wait": "⏸️ Wait — Let Theta Work",
    }

    next_steps = {
        "assign": (
            f"Take assignment at ${pos.strike:.2f}. "
            f"Net cost: ${pos.net_assigned_cost:.2f}. "
            f"Next: sell covered calls above cost basis."
            if pos.strategy == "CSP"
            else f"Shares called away at ${pos.strike:.2f}. "
            f"Wheel complete — start new CSP cycle."
        ),
        "roll": (
            f"Roll to ${best_roll_candidate.strike:.2f} "
            f"({best_roll_candidate.roll_type}) for "
            f"${best_roll_candidate.roll_credit:.2f} credit."
            if best_roll_candidate
            else "No qualifying roll candidates."
        ),
        "close": (
            f"Close position at ${pos.close_cost:.2f}. "
            f"Profit: ${pos.unrealized_pl:.2f}."
        ),
        "wait": (
            "Hold position. Monitor daily. "
            "Re-evaluate when DTE <= 2 or "
            "if stock moves through strike."
        ),
    }

    explanations = {
        "assign": assign_reason,
        "roll": best_roll_reason,
        "close": close_reason,
        "wait": wait_reason,
    }

    pos_size, pos_reason = _position_size_guidance(
        regime, has_earnings, dte_remaining, pos.times_rolled, confidence
    )

    return PositionDecision(
        recommendation=decision_labels[best_action],
        confidence=confidence,
        scores=scores,
        explanation=explanations[best_action],
        next_wheel_step=next_steps[best_action],
        roll_candidates=filtered_rolls,
        warnings=warnings,
        position_size=pos_size,
        position_size_reason=pos_reason,
    )
