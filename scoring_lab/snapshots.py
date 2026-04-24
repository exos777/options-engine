"""
Simulated option chain snapshots for TSLA, NVDA, AAPL, SPY.

Strikes are priced with Black-Scholes and Greeks come from the repo's
greeks module, so the relative pricing across a chain is self-consistent.
Open interest and spread are stylised but ranked sensibly (deeper OTM =
lower OI, higher spread) to mirror what a real 7–14 DTE chain looks like.
"""

from __future__ import annotations

import math

from greeks.black_scholes import compute_greeks
from scoring_lab.data import Snapshot, StrikeData


_RISK_FREE_RATE = 0.045


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _bs_put_price(S: float, K: float, T: float, sigma: float, r: float = _RISK_FREE_RATE) -> float:
    """Black-Scholes put price (no dividends)."""
    if T <= 0 or sigma <= 0:
        return max(0.0, K - S)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def _build_strike(
    strike: float, spot: float, iv: float, dte: int,
) -> StrikeData:
    T = dte / 365
    premium = _bs_put_price(spot, strike, T, iv)
    premium = max(premium, 0.01)

    g = compute_greeks(S=spot, K=strike, T=T, sigma=iv, option_type="put")
    delta = g.delta if g else -0.3
    theta = g.theta if g else -0.05
    vega = g.vega if g else 0.1

    # Stylised liquidity: ATM strikes have deepest OI and tightest spreads.
    moneyness = abs(strike - spot) / spot
    oi_factor = math.exp(-moneyness * 10)
    open_interest = int(5_000 * oi_factor) + 20
    spread_pct = min(0.30, 0.02 + moneyness * 1.5)

    dist_pct = max(0.0, (spot - strike) / spot)

    return StrikeData(
        strike=round(strike, 2),
        premium=round(premium, 2),
        delta=round(delta, 3),
        theta=round(theta, 4),
        vega=round(vega, 4),
        iv=iv,
        open_interest=open_interest,
        spread_pct=round(spread_pct, 4),
        dist_pct=round(dist_pct, 4),
        break_even=round(strike - premium, 2),
        underlying_price=spot,
        dte=dte,
    )


def _build_snapshot(
    ticker: str,
    spot: float,
    iv: float,
    desired_buy_price: float,
    strike_step: float,
    strikes_below: int = 10,
    strikes_above: int = 4,
    dte: int = 10,
) -> Snapshot:
    strikes: list[StrikeData] = []
    for i in range(-strikes_below, strikes_above + 1):
        k = spot + i * strike_step
        if k <= 0:
            continue
        strikes.append(_build_strike(k, spot, iv, dte))
    strikes.sort(key=lambda s: s.strike)
    return Snapshot(
        ticker=ticker,
        underlying_price=spot,
        desired_buy_price=desired_buy_price,
        dte=dte,
        strikes=strikes,
    )


def all_snapshots() -> list[Snapshot]:
    """Four simulated 10-DTE chains covering a realistic volatility spread."""
    return [
        # High-vol, high-beta
        _build_snapshot(
            ticker="TSLA",
            spot=380.0,
            iv=0.55,
            desired_buy_price=360.0,   # ~5.3% below
            strike_step=5.0,
        ),
        # High-vol growth
        _build_snapshot(
            ticker="NVDA",
            spot=175.0,
            iv=0.50,
            desired_buy_price=165.0,   # ~5.7% below
            strike_step=2.5,
        ),
        # Mega-cap, lower vol
        _build_snapshot(
            ticker="AAPL",
            spot=240.0,
            iv=0.28,
            desired_buy_price=228.0,   # ~5.0% below
            strike_step=2.5,
        ),
        # Index, low vol
        _build_snapshot(
            ticker="SPY",
            spot=590.0,
            iv=0.18,
            desired_buy_price=570.0,   # ~3.4% below
            strike_step=2.0,
        ),
    ]
