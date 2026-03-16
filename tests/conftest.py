"""
Shared pytest fixtures for the options engine test suite.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategies.models import (
    FilterParams,
    OptionContract,
    Quote,
    RegimeResult,
    RiskProfile,
    Strategy,
    TechnicalIndicators,
    ChartRegime,
    SupportResistanceLevel,
)


# ---------------------------------------------------------------------------
# OHLCV DataFrames
# ---------------------------------------------------------------------------

def _make_ohlcv(
    n: int = 120,
    start_price: float = 100.0,
    trend: float = 0.001,
    volatility: float = 0.015,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = [start_price]
    for _ in range(n - 1):
        move = closes[-1] * (trend + volatility * rng.standard_normal())
        closes.append(max(1.0, closes[-1] + move))

    closes = np.array(closes)
    highs = closes * (1 + abs(rng.uniform(0, 0.01, n)))
    lows = closes * (1 - abs(rng.uniform(0, 0.01, n)))
    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    volumes = rng.integers(500_000, 5_000_000, n).astype(float)

    idx = pd.date_range(end="2024-12-31", periods=n, freq="B")
    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volumes},
        index=idx,
    )


@pytest.fixture
def bullish_df() -> pd.DataFrame:
    """Trending up OHLCV data."""
    return _make_ohlcv(n=120, trend=0.003, volatility=0.010)


@pytest.fixture
def bearish_df() -> pd.DataFrame:
    """Trending down OHLCV data."""
    return _make_ohlcv(n=120, trend=-0.003, volatility=0.010)


@pytest.fixture
def flat_df() -> pd.DataFrame:
    """Range-bound OHLCV data."""
    return _make_ohlcv(n=120, trend=0.0, volatility=0.008)


# ---------------------------------------------------------------------------
# Technical indicators fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def bullish_indicators() -> TechnicalIndicators:
    return TechnicalIndicators(
        sma_20=105.0,
        sma_50=100.0,
        rsi_14=58.0,
        atr_14=2.0,
        avg_volume_20=1_000_000,
        current_price=108.0,
        weekly_atr_est=4.47,
    )


@pytest.fixture
def bearish_indicators() -> TechnicalIndicators:
    return TechnicalIndicators(
        sma_20=95.0,
        sma_50=100.0,
        rsi_14=38.0,
        atr_14=2.5,
        avg_volume_20=1_200_000,
        current_price=92.0,
        weekly_atr_est=5.59,
    )


@pytest.fixture
def neutral_indicators() -> TechnicalIndicators:
    return TechnicalIndicators(
        sma_20=102.0,
        sma_50=100.0,
        rsi_14=50.0,
        atr_14=2.0,
        avg_volume_20=900_000,
        current_price=101.0,
        weekly_atr_est=4.47,
    )


# ---------------------------------------------------------------------------
# Support / Resistance fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def support_levels() -> list[SupportResistanceLevel]:
    return [
        SupportResistanceLevel(price=95.0, level_type="support", strength=0.8, touches=4),
        SupportResistanceLevel(price=90.0, level_type="support", strength=0.5, touches=2),
    ]


@pytest.fixture
def resistance_levels() -> list[SupportResistanceLevel]:
    return [
        SupportResistanceLevel(price=112.0, level_type="resistance", strength=0.9, touches=5),
        SupportResistanceLevel(price=118.0, level_type="resistance", strength=0.4, touches=2),
    ]


# ---------------------------------------------------------------------------
# Regime fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def bullish_regime() -> RegimeResult:
    return RegimeResult(
        primary=ChartRegime.BULLISH,
        secondary=None,
        description="Bullish",
        trade_bias="go",
    )


@pytest.fixture
def bearish_regime() -> RegimeResult:
    return RegimeResult(
        primary=ChartRegime.BEARISH,
        secondary=None,
        description="Bearish",
        trade_bias="skip",
        skip_reason="Downtrend.",
    )


@pytest.fixture
def neutral_regime() -> RegimeResult:
    return RegimeResult(
        primary=ChartRegime.NEUTRAL,
        secondary=None,
        description="Neutral",
        trade_bias="caution",
    )


# ---------------------------------------------------------------------------
# Option contract fixtures
# ---------------------------------------------------------------------------

def _call(
    strike: float, bid: float, ask: float, oi: int, delta: float, iv: float,
    theta: float | None = None, vega: float | None = None,
) -> OptionContract:
    return OptionContract(
        strike=strike,
        expiration="2024-12-27",
        option_type="call",
        bid=bid,
        ask=ask,
        last=(bid + ask) / 2,
        volume=max(1, oi // 10),
        open_interest=oi,
        implied_volatility=iv,
        delta=delta,
        theta=theta,
        vega=vega,
    )


def _put(
    strike: float, bid: float, ask: float, oi: int, delta: float, iv: float,
    theta: float | None = None, vega: float | None = None,
) -> OptionContract:
    return OptionContract(
        strike=strike,
        expiration="2024-12-27",
        option_type="put",
        bid=bid,
        ask=ask,
        last=(bid + ask) / 2,
        volume=max(1, oi // 10),
        open_interest=oi,
        implied_volatility=iv,
        delta=delta,
        theta=theta,
        vega=vega,
    )


@pytest.fixture
def sample_calls() -> list[OptionContract]:
    """A realistic set of OTM call contracts around $108 spot."""
    return [
        _call(108.0, 2.10, 2.30, 500,  0.48, 0.28, theta=-0.15, vega=0.12),
        _call(110.0, 1.40, 1.60, 800,  0.38, 0.27, theta=-0.12, vega=0.11),
        _call(112.0, 0.80, 0.95, 600,  0.27, 0.26, theta=-0.08, vega=0.09),
        _call(115.0, 0.35, 0.45, 300,  0.16, 0.25, theta=-0.04, vega=0.06),
        _call(118.0, 0.12, 0.18, 150,  0.08, 0.24, theta=-0.02, vega=0.03),
    ]


@pytest.fixture
def sample_puts() -> list[OptionContract]:
    """A realistic set of OTM put contracts around $108 spot."""
    return [
        _put(107.0, 1.80, 2.00, 550, -0.46, 0.29, theta=-0.14, vega=0.12),
        _put(105.0, 1.20, 1.40, 750, -0.35, 0.28, theta=-0.10, vega=0.10),
        _put(103.0, 0.75, 0.90, 600, -0.26, 0.27, theta=-0.07, vega=0.08),
        _put(100.0, 0.40, 0.52, 400, -0.18, 0.26, theta=-0.04, vega=0.05),
        _put( 97.0, 0.18, 0.26, 200, -0.10, 0.25, theta=-0.02, vega=0.03),
    ]


# ---------------------------------------------------------------------------
# FilterParams fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def balanced_cc_params() -> FilterParams:
    return FilterParams(
        strategy=Strategy.COVERED_CALL,
        risk_profile=RiskProfile.BALANCED,
        max_delta=0.40,
        min_open_interest=50,
        max_spread_pct=0.30,
        min_premium=0.10,
        shares_owned=100,
        cost_basis=100.0,
    )


@pytest.fixture
def balanced_csp_params() -> FilterParams:
    return FilterParams(
        strategy=Strategy.CASH_SECURED_PUT,
        risk_profile=RiskProfile.BALANCED,
        max_delta=0.40,
        min_open_interest=50,
        max_spread_pct=0.30,
        min_premium=0.10,
        cash_available=10000.0,
        desired_buy_price=103.0,
    )


# ---------------------------------------------------------------------------
# Quote fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_quote() -> Quote:
    return Quote(
        ticker="TEST",
        price=108.0,
        prev_close=106.5,
        change=1.5,
        change_pct=0.014,
        volume=3_000_000,
        avg_volume=2_500_000,
    )
