"""
Thin wrapper around the existing CSP engine so it can be called with the
same StrikeData input as the new scorer.

Does not modify existing scoring logic — it just builds the minimal support
objects (neutral regime, no S/R, permissive filter params) that the engine
requires.
"""

from __future__ import annotations

from scoring.cash_secured_put import score_cash_secured_puts
from scoring_lab.data import Snapshot, StrikeData
from strategies.models import (
    ChartRegime,
    FilterParams,
    OptionContract,
    RegimeResult,
    RiskProfile,
    ScoredOption,
    Strategy,
    TechnicalIndicators,
)


def _to_contract(sd: StrikeData) -> OptionContract:
    half_spread = sd.premium * sd.spread_pct / 2
    bid = max(0.0, sd.premium - half_spread)
    ask = sd.premium + half_spread
    return OptionContract(
        strike=sd.strike,
        expiration="2099-01-01",  # engine doesn't parse this in scoring
        option_type="put",
        bid=bid,
        ask=ask,
        last=sd.premium,
        volume=max(sd.open_interest // 2, 1),
        open_interest=sd.open_interest,
        implied_volatility=sd.iv,
        delta=sd.delta,
        gamma=0.01,
        theta=sd.theta,
        vega=sd.vega,
    )


def _neutral_indicators(price: float) -> TechnicalIndicators:
    return TechnicalIndicators(
        sma_20=price,
        sma_50=price,
        rsi_14=50.0,
        atr_14=price * 0.02,
        avg_volume_20=1_000_000,
        current_price=price,
        weekly_atr_est=price * 0.04,
    )


def _neutral_regime() -> RegimeResult:
    return RegimeResult(
        primary=ChartRegime.NEUTRAL,
        secondary=None,
        description="Neutral (test)",
        trade_bias="caution",
    )


def _permissive_params(
    desired_buy_price: float,
    risk_profile: RiskProfile = RiskProfile.BALANCED,
) -> FilterParams:
    return FilterParams(
        strategy=Strategy.CASH_SECURED_PUT,
        risk_profile=risk_profile,
        max_delta=0.99,
        min_open_interest=0,
        max_spread_pct=0.99,
        min_premium=0.0,
        desired_buy_price=desired_buy_price,
    )


def current_score_all(snapshot: Snapshot) -> list[ScoredOption]:
    """Run the existing engine over every strike and return ScoredOption list."""
    contracts = [_to_contract(s) for s in snapshot.strikes]
    return score_cash_secured_puts(
        puts=contracts,
        current_price=snapshot.underlying_price,
        dte=snapshot.dte,
        indicators=_neutral_indicators(snapshot.underlying_price),
        regime=_neutral_regime(),
        support_levels=[],
        resistance_levels=[],
        params=_permissive_params(snapshot.desired_buy_price),
    )


def current_score(strike_data: StrikeData, snapshot: Snapshot) -> float:
    """Return the current-engine score for a specific strike in a snapshot."""
    scored = current_score_all(snapshot)
    for s in scored:
        if abs(s.contract.strike - strike_data.strike) < 0.005:
            return s.score
    return 0.0
