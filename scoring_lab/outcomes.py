"""
Simulated outcome P/L per strike under three scenarios:

  - expires worthless  → keep full premium
  - assigned (stock at or below strike) → P/L = premium - (strike - spot_final)
  - downside shock (stock drops 10%) → P/L under that price

Assignment quality is framed relative to the user's desired buy price:
lower break_even than desired = favourable; above desired = unfavourable.
"""

from __future__ import annotations

from dataclasses import dataclass

from scoring_lab.data import StrikeData


DOWNSIDE_SHOCK_PCT = 0.10


@dataclass
class Outcome:
    premium_earned: float              # always = premium (collected up front)
    pnl_expires: float                 # + premium
    pnl_assigned_at_strike: float      # premium (you just own shares at strike)
    pnl_downside_shock: float          # premium - max(0, strike - spot_shock)
    assignment_quality: float          # break_even - desired_buy_price (negative = good)
    downside_risk: float               # -pnl_downside_shock, clamped to >= 0


def simulate(sd: StrikeData, desired_buy_price: float) -> Outcome:
    spot_shock = sd.underlying_price * (1 - DOWNSIDE_SHOCK_PCT)
    assigned_loss = max(0.0, sd.strike - spot_shock)
    pnl_downside = sd.premium - assigned_loss

    return Outcome(
        premium_earned=sd.premium,
        pnl_expires=sd.premium,
        pnl_assigned_at_strike=sd.premium,
        pnl_downside_shock=pnl_downside,
        assignment_quality=sd.break_even - desired_buy_price,
        downside_risk=max(0.0, -pnl_downside),
    )
