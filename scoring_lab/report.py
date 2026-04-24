"""
Pandas-backed comparison report for the current vs new scoring engines.

Selects three picks per engine per ticker — safe / balanced / aggressive
by |delta| bucket — and prints per-ticker tables and a summary at the end.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from scoring_lab.current_scorer import current_score_all
from scoring_lab.data import Snapshot, StrikeData
from scoring_lab.new_scorer import new_score_all
from scoring_lab.outcomes import simulate
from scoring_lab.snapshots import all_snapshots


SAFE_MAX = 0.20
BALANCED_MAX = 0.30
AGGRESSIVE_MAX = 0.45

TIER_ORDER = ["safe", "balanced", "aggressive"]


def _tier_of(abs_delta: float) -> str | None:
    if abs_delta <= SAFE_MAX:
        return "safe"
    if abs_delta <= BALANCED_MAX:
        return "balanced"
    if abs_delta <= AGGRESSIVE_MAX:
        return "aggressive"
    return None


def _current_scored_map(snapshot: Snapshot) -> dict[float, float]:
    """strike -> current-engine score (0 if filtered out)."""
    out = {s.strike: 0.0 for s in snapshot.strikes}
    for s in current_score_all(snapshot):
        out[s.contract.strike] = s.score
    return out


def _pick_tier(
    snapshot: Snapshot,
    score_by_strike: dict[float, float],
) -> dict[str, StrikeData | None]:
    """For each delta tier, pick the highest-scoring strike."""
    buckets: dict[str, list[tuple[StrikeData, float]]] = {
        t: [] for t in TIER_ORDER
    }
    for s in snapshot.strikes:
        tier = _tier_of(abs(s.delta))
        if tier is None:
            continue
        buckets[tier].append((s, score_by_strike.get(s.strike, 0.0)))

    picks: dict[str, StrikeData | None] = {}
    for tier, rows in buckets.items():
        if not rows:
            picks[tier] = None
            continue
        rows.sort(key=lambda r: r[1], reverse=True)
        picks[tier] = rows[0][0]
    return picks


def _expected_move(spot: float, iv: float, dte: int) -> float:
    if iv <= 0 or dte <= 0:
        return 0.0
    return spot * iv * math.sqrt(dte / 365)


def _picks_dataframe(
    label: str,
    picks: dict[str, StrikeData | None],
    score_by_strike: dict[float, float],
    desired_buy_price: float,
) -> pd.DataFrame:
    rows = []
    for tier in TIER_ORDER:
        s = picks.get(tier)
        if s is None:
            rows.append({
                "Engine": label,
                "Tier": tier,
                "Strike": None,
                "Premium": None,
                "Break-Even": None,
                "BE vs Desired": None,
                "|Delta|": None,
                "Dist %": None,
                "Spread %": None,
                "Score": None,
            })
            continue
        rows.append({
            "Engine": label,
            "Tier": tier,
            "Strike": round(s.strike, 2),
            "Premium": round(s.premium, 2),
            "Break-Even": round(s.break_even, 2),
            "BE vs Desired": round(s.break_even - desired_buy_price, 2),
            "|Delta|": round(abs(s.delta), 3),
            "Dist %": round(s.dist_pct * 100, 2),
            "Spread %": round(s.spread_pct * 100, 2),
            "Score": round(score_by_strike.get(s.strike, 0.0), 2),
        })
    return pd.DataFrame(rows)


def _summary_row(
    engine: str,
    picks: dict[str, StrikeData | None],
    snapshot: Snapshot,
) -> dict:
    valid = [s for s in picks.values() if s is not None]
    if not valid:
        return {
            "Engine": engine,
            "Avg Premium": 0.0,
            "Avg BE vs Desired": 0.0,
            "% BE <= Desired": 0.0,
            "Avg |Delta|": 0.0,
            "Avg Spread %": 0.0,
            "% Inside EM": 0.0,
            "Risk Score": 0.0,
            "Avg Downside P/L": 0.0,
        }

    em = _expected_move(snapshot.underlying_price, valid[0].iv, snapshot.dte)
    em_lower = snapshot.underlying_price - em

    inside_em = sum(1 for s in valid if s.strike > em_lower) / len(valid) * 100
    be_ok = sum(
        1 for s in valid if s.break_even <= snapshot.desired_buy_price
    ) / len(valid) * 100
    risk = sum(abs(s.delta) * (1 - s.dist_pct) for s in valid) / len(valid)

    outcomes = [simulate(s, snapshot.desired_buy_price) for s in valid]
    avg_downside = sum(o.pnl_downside_shock for o in outcomes) / len(outcomes)

    return {
        "Engine": engine,
        "Avg Premium": round(
            sum(s.premium for s in valid) / len(valid), 2
        ),
        "Avg BE vs Desired": round(
            sum(s.break_even - snapshot.desired_buy_price for s in valid)
            / len(valid),
            2,
        ),
        "% BE <= Desired": round(be_ok, 1),
        "Avg |Delta|": round(
            sum(abs(s.delta) for s in valid) / len(valid), 3
        ),
        "Avg Spread %": round(
            sum(s.spread_pct for s in valid) / len(valid) * 100, 2
        ),
        "% Inside EM": round(inside_em, 1),
        "Risk Score": round(risk, 3),
        "Avg Downside P/L": round(avg_downside, 2),
    }


@dataclass
class TickerReport:
    ticker: str
    picks_table: pd.DataFrame
    summary_table: pd.DataFrame
    winner: str
    winner_reason: str


def _decide_winner(
    summary_current: dict, summary_new: dict,
) -> tuple[str, str]:
    """Heuristic: prefer the engine whose break-even is closer-to/below desired."""
    cur_be = summary_current["Avg BE vs Desired"]
    new_be = summary_new["Avg BE vs Desired"]
    cur_prem = summary_current["Avg Premium"]
    new_prem = summary_new["Avg Premium"]
    cur_risk = summary_current["Risk Score"]
    new_risk = summary_new["Risk Score"]

    cur_pts = 0
    new_pts = 0
    reasons: list[str] = []

    if new_be < cur_be:
        new_pts += 1
        reasons.append(f"new break-even {new_be:+.2f} vs current {cur_be:+.2f}")
    else:
        cur_pts += 1
        reasons.append(f"current break-even {cur_be:+.2f} vs new {new_be:+.2f}")

    if new_prem > cur_prem:
        new_pts += 1
        reasons.append(f"new premium ${new_prem:.2f} > current ${cur_prem:.2f}")
    else:
        cur_pts += 1
        reasons.append(f"current premium ${cur_prem:.2f} >= new ${new_prem:.2f}")

    if new_risk < cur_risk:
        new_pts += 1
        reasons.append(f"new risk {new_risk:.3f} < current {cur_risk:.3f}")
    else:
        cur_pts += 1
        reasons.append(f"current risk {cur_risk:.3f} <= new {new_risk:.3f}")

    winner = "NEW" if new_pts > cur_pts else (
        "CURRENT" if cur_pts > new_pts else "TIE"
    )
    return winner, "; ".join(reasons)


def build_ticker_report(snapshot: Snapshot) -> TickerReport:
    # Current engine
    current_map = _current_scored_map(snapshot)
    current_picks = _pick_tier(snapshot, current_map)

    # New engine
    new_ranked = new_score_all(snapshot.strikes, snapshot.desired_buy_price)
    new_map = {s.strike: sc for s, sc in new_ranked}
    new_picks = _pick_tier(snapshot, new_map)

    picks_df = pd.concat(
        [
            _picks_dataframe(
                "CURRENT", current_picks, current_map,
                snapshot.desired_buy_price,
            ),
            _picks_dataframe(
                "NEW", new_picks, new_map,
                snapshot.desired_buy_price,
            ),
        ],
        ignore_index=True,
    )

    summary_current = _summary_row("CURRENT", current_picks, snapshot)
    summary_new = _summary_row("NEW", new_picks, snapshot)
    summary_df = pd.DataFrame([summary_current, summary_new])

    winner, reason = _decide_winner(summary_current, summary_new)

    return TickerReport(
        ticker=snapshot.ticker,
        picks_table=picks_df,
        summary_table=summary_df,
        winner=winner,
        winner_reason=reason,
    )


def run_comparison(snapshots: list[Snapshot] | None = None) -> list[TickerReport]:
    """Build reports for every snapshot and print them."""
    snaps = snapshots if snapshots is not None else all_snapshots()
    reports: list[TickerReport] = []

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)

    print("=" * 78)
    print("Scoring Engine Comparison - CURRENT vs NEW")
    print("=" * 78)

    for snap in snaps:
        r = build_ticker_report(snap)
        reports.append(r)

        print()
        print("-" * 78)
        print(
            f"{r.ticker}  |  spot ${snap.underlying_price:.2f}  "
            f"|  desired buy ${snap.desired_buy_price:.2f}  "
            f"|  DTE {snap.dte}"
        )
        print("-" * 78)
        print("Picks by tier (safe / balanced / aggressive):")
        print(r.picks_table.to_string(index=False))
        print()
        print("Metric summary:")
        print(r.summary_table.to_string(index=False))
        print()
        print(f"-> Winner: {r.winner}")
        print(f"   {r.winner_reason}")

    # Final roll-up
    print()
    print("=" * 78)
    print("Overall")
    print("=" * 78)
    winners = [r.winner for r in reports]
    new_count = winners.count("NEW")
    cur_count = winners.count("CURRENT")
    tie_count = winners.count("TIE")
    print(
        f"NEW: {new_count}  |  CURRENT: {cur_count}  |  TIE: {tie_count}  "
        f"(of {len(reports)} tickers)"
    )
    print()
    if new_count > cur_count:
        print("-> NEW engine wins overall - safer entries / better premium mix")
    elif cur_count > new_count:
        print("-> CURRENT engine wins overall - better aligned with wheel strategy")
    else:
        print("-> No clear winner - engines are comparable on this data")

    return reports


if __name__ == "__main__":
    run_comparison()
