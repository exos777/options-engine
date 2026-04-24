"""Data model shared by the new / current scorers and the report layer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StrikeData:
    """One row in a simulated option chain snapshot."""
    strike: float
    premium: float            # mid price
    delta: float              # negative for puts (signed)
    theta: float              # negative, per calendar day
    vega: float
    iv: float
    open_interest: int
    spread_pct: float
    dist_pct: float           # (underlying - strike) / underlying
    break_even: float         # strike - premium
    underlying_price: float
    dte: int


@dataclass
class Snapshot:
    """All strikes for one ticker at one expiration."""
    ticker: str
    underlying_price: float
    desired_buy_price: float
    dte: int
    strikes: list[StrikeData]
