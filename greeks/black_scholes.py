"""
Black-Scholes Greeks calculator.

Computes delta, gamma, theta, vega analytically from:
  - S: spot price
  - K: strike price
  - T: time to expiry in years (DTE / 365)
  - r: risk-free rate (annualised, e.g. 0.05)
  - sigma: implied volatility (annualised, e.g. 0.30)
  - option_type: "call" or "put"

All Greeks are expressed per-share (not per-contract).
Theta is returned as a negative value (daily decay for long positions).
Vega is returned per 1% move in IV (i.e. divided by 100).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Standard normal CDF & PDF (pure Python, no scipy dependency)
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    """Standard normal probability density function."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


# ---------------------------------------------------------------------------
# Core Black-Scholes
# ---------------------------------------------------------------------------

_DEFAULT_RISK_FREE_RATE = 0.045  # ~current US Treasury yield


@dataclass
class Greeks:
    """Computed Black-Scholes Greeks for a single option."""
    delta: float
    gamma: float
    theta: float   # per calendar day, negative for long positions
    vega: float    # per 1% IV change


def _d1_d2(
    S: float, K: float, T: float, r: float, sigma: float,
) -> tuple[float, float]:
    """Compute d1 and d2."""
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def compute_greeks(
    S: float,
    K: float,
    T: float,
    sigma: float,
    option_type: str,
    r: float = _DEFAULT_RISK_FREE_RATE,
) -> Optional[Greeks]:
    """
    Compute Black-Scholes Greeks for a European option.

    Returns None if inputs are invalid (T <= 0, sigma <= 0, S <= 0, K <= 0).
    At expiration (T ~ 0), returns intrinsic-value Greeks.
    """
    if S <= 0 or K <= 0 or sigma <= 0:
        return None

    if T <= 0:
        # At expiration: delta is 1/-1 if ITM, 0 if OTM; others are 0
        is_call = option_type.lower() == "call"
        if is_call:
            delta = 1.0 if S > K else 0.0
        else:
            delta = -1.0 if S < K else 0.0
        return Greeks(delta=delta, gamma=0.0, theta=0.0, vega=0.0)

    sqrt_T = math.sqrt(T)
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    nd1 = _norm_pdf(d1)
    Nd1 = _norm_cdf(d1)
    Nd2 = _norm_cdf(d2)

    # Gamma (same for calls and puts)
    gamma = nd1 / (S * sigma * sqrt_T)

    # Vega (same for calls and puts) — per 1% IV change
    vega = S * nd1 * sqrt_T / 100.0

    is_call = option_type.lower() == "call"

    if is_call:
        delta = Nd1
        theta_annual = (
            -(S * nd1 * sigma) / (2.0 * sqrt_T)
            - r * K * math.exp(-r * T) * Nd2
        )
    else:
        delta = Nd1 - 1.0
        theta_annual = (
            -(S * nd1 * sigma) / (2.0 * sqrt_T)
            + r * K * math.exp(-r * T) * _norm_cdf(-d2)
        )

    # Convert theta from per-year to per-calendar-day
    theta = theta_annual / 365.0

    return Greeks(delta=delta, gamma=gamma, theta=theta, vega=vega)


def backfill_greeks(
    strike: float,
    spot: float,
    dte: int,
    iv: Optional[float],
    option_type: str,
    r: float = _DEFAULT_RISK_FREE_RATE,
) -> Optional[Greeks]:
    """
    Convenience wrapper for the data provider.
    Accepts DTE as integer days and IV as decimal.
    Returns None if IV is unavailable.
    """
    if iv is None or iv <= 0:
        return None
    T = max(dte, 1) / 365.0
    return compute_greeks(spot, strike, T, iv, option_type, r)
