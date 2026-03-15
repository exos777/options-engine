"""
Shared dataclasses and enumerations used across the options engine.
All monetary values are in USD. Percentages are expressed as decimals (0.05 = 5%).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import pandas as pd


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class Strategy(str, Enum):
    COVERED_CALL = "Covered Call"
    CASH_SECURED_PUT = "Cash Secured Put"


class RiskProfile(str, Enum):
    CONSERVATIVE = "Conservative"
    BALANCED = "Balanced"
    AGGRESSIVE = "Aggressive"


class ChartRegime(str, Enum):
    BULLISH = "Bullish"
    NEUTRAL = "Neutral"
    BEARISH = "Bearish"
    OVEREXTENDED = "Overextended"
    NEAR_SUPPORT = "Near Support"
    NEAR_RESISTANCE = "Near Resistance"


class RecommendationLabel(str, Enum):
    SAFEST_INCOME = "Safest Income"
    BEST_BALANCE = "Best Balance"
    MAX_PREMIUM = "Max Premium"


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------

@dataclass
class Quote:
    """Real-time stock quote."""
    ticker: str
    price: float
    prev_close: float
    change: float          # absolute dollar change
    change_pct: float      # e.g. -0.015 = -1.5%
    volume: int
    avg_volume: int
    market_cap: Optional[float] = None
    earnings_date: Optional[str] = None  # ISO date string


@dataclass
class OptionContract:
    """A single option contract row."""
    strike: float
    expiration: str       # ISO date string
    option_type: str      # "call" or "put"
    bid: float
    ask: float
    last: float
    volume: int
    open_interest: int
    implied_volatility: Optional[float] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None

    @property
    def mid(self) -> float:
        """Mid-market price."""
        return (self.bid + self.ask) / 2 if self.bid > 0 or self.ask > 0 else self.last

    @property
    def bid_ask_spread_pct(self) -> float:
        """Bid-ask spread as a fraction of the mid price. Returns 1.0 if mid is 0."""
        m = self.mid
        if m <= 0:
            return 1.0
        return (self.ask - self.bid) / m


# ---------------------------------------------------------------------------
# Technical analysis outputs
# ---------------------------------------------------------------------------

@dataclass
class TechnicalIndicators:
    """Output from the technical indicators module."""
    sma_20: float
    sma_50: float
    rsi_14: float
    atr_14: float
    avg_volume_20: float
    current_price: float
    weekly_atr_est: float   # ATR projected to weekly (ATR * sqrt(5))


@dataclass
class SupportResistanceLevel:
    """A single S/R zone."""
    price: float
    level_type: str         # "support" or "resistance"
    strength: float         # 0–1, how many times it was touched
    touches: int


@dataclass
class RegimeResult:
    """Output from the chart regime classifier."""
    primary: ChartRegime
    secondary: Optional[ChartRegime]
    description: str
    trade_bias: str         # "go", "caution", "skip"
    skip_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Filter / input parameters
# ---------------------------------------------------------------------------

@dataclass
class FilterParams:
    """User-supplied filter and preference parameters."""
    strategy: Strategy
    risk_profile: RiskProfile
    max_delta: float = 0.40
    min_open_interest: int = 50
    max_spread_pct: float = 0.30        # 30% bid-ask spread
    min_premium: float = 0.10
    # Covered call
    shares_owned: int = 100
    cost_basis: Optional[float] = None
    allow_below_basis: bool = False
    # Cash secured put
    cash_available: Optional[float] = None
    desired_buy_price: Optional[float] = None


# ---------------------------------------------------------------------------
# Scored option output
# ---------------------------------------------------------------------------

@dataclass
class ScoredOption:
    """An option contract decorated with all scoring metrics."""
    contract: OptionContract
    # Common metrics
    premium: float           # mid price used for calculations
    annualized_return: float  # annualized yield as a decimal
    distance_pct: float       # abs % distance from spot to strike
    break_even: float
    score: float             # 0–100
    # Subscores (0–100 each)
    premium_score: float
    delta_score: float
    liquidity_score: float
    chart_score: float
    basis_score: float       # 0 if n/a for CSPs
    iv_rank_score: float = 50.0  # 50 = neutral when 52w IV data unavailable
    # Explanatory fields
    near_support: bool = False
    near_resistance: bool = False
    above_cost_basis: Optional[bool] = None


@dataclass
class Recommendation:
    """A top recommendation with label and plain-English explanation."""
    label: RecommendationLabel
    option: ScoredOption
    explanation: str


@dataclass
class ScreenerResult:
    """Full output from the screening pipeline for one ticker / expiration."""
    quote: Quote
    expiration: str
    dte: int                 # calendar days to expiration
    indicators: TechnicalIndicators
    regime: RegimeResult
    support_levels: list[SupportResistanceLevel]
    resistance_levels: list[SupportResistanceLevel]
    expected_move: float     # ±$ expected move for the expiration
    all_options: pd.DataFrame          # full scored table
    recommendations: list[Recommendation]
    warnings: list[str] = field(default_factory=list)
